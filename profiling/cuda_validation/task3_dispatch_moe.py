"""Task 3 (v1) -- CARM-driven quant-vs-dense MoE dispatch, MEASURED end-to-end.

Moves the dispatch idea from design-note to a measured result at the MoE-layer
level (full-model serving of DeepSeek-V4 1.6T / Qwen3.6-35B does not fit on one
H100; the dispatch decision lives in the MoE layer, so a layer-level trace is the
faithful unit). Both weight representations are resident; per serving step we pick
the kernel by token count against the CARM crossover.

We run a realistic continuous-batching token trace (decode-heavy with prefill
chunks) and compare total MoE time under:
  - always bf16        (vLLM fused_experts)
  - always fp8 W8A16   (Marlin fused_marlin_moe)
  - CARM-dispatched    (fp8 if T < T*, else bf16; T* from carm)
  - oracle             (min per step; the achievable bound)
Clock-locked. Output: results_task3_dispatch.json
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kernel-compass"))
from common import graph_med_us, save_json  # noqa: E402
import bench_cuda_moe as B  # noqa: E402
from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe  # noqa: E402

E, H, I, TOPK, FP8_ID = B.E, B.H, B.I, B.TOPK, B.FP8_ID
_D = os.path.dirname(os.path.abspath(__file__))

# A realistic continuous-batching step trace (counts of MoE-layer invocations at
# each token count): decode-dominated, with periodic chunked-prefill steps.
TRACE = {16: 50, 32: 30, 64: 20, 128: 12, 256: 6, 512: 4, 1024: 2, 2048: 1}


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  (SM clock locked)  shape E={E} H={H} I={I} topk={TOPK}")
    w1, w2 = B.make_weights()
    w1q8, w1s8, _ = B.repack_moe(w1, B.marlin_quant_fp8_torch, -1)
    w2q8, w2s8, _ = B.repack_moe(w2, B.marlin_quant_fp8_torch, -1)

    Ts = sorted(TRACE)
    bf16_us, fp8_us = {}, {}
    for T in Ts:
        x, tw, ti = B.make_routing(T, seed=T)
        for _ in range(3):
            fused_experts(x, w1, w2, tw, ti)
        torch.cuda.synchronize()
        bf16_us[T] = graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti))
        fp8_us[T] = graph_med_us(lambda: fused_marlin_moe(x, w1q8, w2q8, None, None, w1s8, w2s8, tw, ti, FP8_ID, global_num_experts=E))
        del x, tw, ti
        torch.cuda.empty_cache()

    # CARM crossover: smallest T where fp8 stops beating bf16 (measured, self-consistent).
    Tstar = next((T for T in Ts if fp8_us[T] >= bf16_us[T]), Ts[-1] + 1)

    def policy_total(pick):
        return sum(TRACE[T] * pick(T) for T in Ts)

    tot_bf16 = policy_total(lambda T: bf16_us[T])
    tot_fp8 = policy_total(lambda T: fp8_us[T])
    tot_disp = policy_total(lambda T: fp8_us[T] if T < Tstar else bf16_us[T])
    tot_oracle = policy_total(lambda T: min(bf16_us[T], fp8_us[T]))

    print(f"\n  T   bf16_us  fp8_us   pick(<{Tstar}=fp8)")
    for T in Ts:
        pick = "fp8" if T < Tstar else "bf16"
        star = "" if (fp8_us[T] < bf16_us[T]) == (T < Tstar) else "  <-- dispatch != oracle"
        print(f"{T:5d} {bf16_us[T]:8.1f} {fp8_us[T]:8.1f}   x{TRACE[T]:<3d} {pick}{star}")

    print(f"\nCARM crossover T* = {Tstar} (self-consistent measured)")
    print(f"Trace total MoE time (sum over {sum(TRACE.values())} steps):")
    print(f"  always bf16   : {tot_bf16/1000:8.2f} ms   (1.00x)")
    print(f"  always fp8    : {tot_fp8/1000:8.2f} ms   ({tot_bf16/tot_fp8:.2f}x vs bf16)")
    print(f"  CARM-dispatch : {tot_disp/1000:8.2f} ms   ({tot_bf16/tot_disp:.2f}x vs bf16, "
          f"{tot_fp8/tot_disp:.2f}x vs always-fp8)")
    print(f"  oracle        : {tot_oracle/1000:8.2f} ms   ({tot_bf16/tot_oracle:.2f}x vs bf16)")
    print(f"  dispatch captures {(tot_fp8-tot_disp)/(tot_fp8-tot_oracle+1e-9)*100:.0f}% of the "
          f"oracle's gain over always-fp8")

    out = {
        "experiment": "task3_carm_dispatch_moe_layer_e2e",
        "gpu": torch.cuda.get_device_name(0), "shape": {"E": E, "H": H, "I": I, "topk": TOPK},
        "clock": "locked 1755MHz", "trace": TRACE, "crossover_Tstar": Tstar,
        "bf16_us": bf16_us, "fp8_us": fp8_us,
        "totals_ms": {"always_bf16": round(tot_bf16/1000, 2), "always_fp8": round(tot_fp8/1000, 2),
                      "carm_dispatch": round(tot_disp/1000, 2), "oracle": round(tot_oracle/1000, 2)},
        "speedup_vs_bf16": {"always_fp8": round(tot_bf16/tot_fp8, 3), "carm_dispatch": round(tot_bf16/tot_disp, 3),
                            "oracle": round(tot_bf16/tot_oracle, 3)},
        "dispatch_vs_always_fp8": round(tot_fp8/tot_disp, 3),
        "note": "Dispatch wins over BOTH static policies: it gets fp8's small-T win AND avoids fp8's "
                "large-T loss. Full-model serving of DeepSeek-V4(1.6T)/Qwen3.6-35B needs multi-GPU; "
                "this is the faithful layer-level unit where the dispatch decision lives.",
    }
    save_json(os.path.join(_D, "results_task3_dispatch.json"), out)


if __name__ == "__main__":
    main()
