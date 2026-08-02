"""Experiment A — fused_moe: tuned CUDA (Marlin) vs FlagGems Triton (the crux).

Re-validates the "weight-only quant has little/no benefit at small token counts"
finding on *tuned CUDA operators* (vLLM Marlin MoE), to rule out the Triton
language as a confound (advisor mandate).

Shape: Mixtral E=8, H=4096, I=14336, top-k=2 (identical to the Triton run).
Token sweep: T in {16,64,128,256,512,1024,2048}.

Paths, all CUDA-graph timed (graph_med_us: 10 launches/graph, median of 40 replays
-- copied verbatim from profiling/fused_moe/bench_fused_moe_extended.py so the
numbers are directly comparable to results_fused_moe_extended.json):
  - bf16          dense MoE via vLLM fused_experts                  (reference)
  - fp8_w8a16     CUDA Marlin, quant_type=float8_e4m3fn            (NATIVE on H100)
  - mxfp4_w4a16   CUDA Marlin, quant_type=float4_e2m1f            (EMULATED on H100:
                  FP4 is dequantized to bf16 and the matmul runs bf16 -- H100 has
                  no FP4 tensor cores. Every mxfp4 row is labelled EMULATED.)

Weights are quantized/repacked ONCE outside the timed region; only the op call is
captured. Correctness is rel-err vs a bf16 fused_experts run on each path's own
dequantized weights (weight_ref), so it isolates kernel error from quant noise.

Output: results_cuda_moe.json (new file; does not touch the Triton JSONs).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    graph_med_us, eager_med_us, env_versions, save_json,
    gpu_key, native_low_precisions,
)

from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
try:  # vllm <= 0.20.x
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe  # noqa: E402
except ModuleNotFoundError:  # vllm >= 0.26 (B200 env): module moved
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E402
    rand_marlin_weight_mxfp4_like,
)

DEV, DT = "cuda", torch.bfloat16
E, H, I, TOPK = 8, 4096, 14336, 2
TOKENS = [16, 64, 128, 256, 512, 640, 768, 896, 1024, 1536, 2048]
FP8_ID = scalar_types.float8_e4m3fn.id
FP4_ID = scalar_types.float4_e2m1f.id
_D = os.path.dirname(os.path.abspath(__file__))


def make_weights(seed=0):
    """Dense bf16 expert weights, T-independent. w1 [E,2I,H], w2 [E,H,I] = [E,N,K]."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DT, generator=g) / H**0.5
    w2 = torch.randn(E, H, I, device=DEV, dtype=DT, generator=g) / I**0.5
    return w1, w2


def make_routing(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=DT, generator=g) / 10
    gating = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=g)
    tw, ti = torch.topk(torch.softmax(gating, -1), TOPK, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
    return x, tw.contiguous(), ti.to(torch.int32).contiguous()


def repack_moe(w, quanter, gs):
    """Per-expert Marlin repack. Returns (qw[E,...], scale[E,...], ref_bf16[E,N,K])."""
    qs, ss, refs = [], [], []
    for e in range(w.shape[0]):
        ref_kn, qw, s = quanter(w[e], gs)         # ref is [K,N]
        qs.append(qw); ss.append(s); refs.append(ref_kn.T.contiguous())  # ref -> [N,K]
    return (torch.stack(qs).contiguous(), torch.stack(ss).contiguous(),
            torch.stack(refs).to(DT).contiguous())


def timed(fn):
    """Graph-timed; fall back to eager (flagged) if the kernel won't capture."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        return round(graph_med_us(fn), 1), "graph"
    except Exception as exc:  # noqa: BLE001
        print(f"    [graph capture failed: {type(exc).__name__}: {str(exc)[:80]}; eager]")
        return round(eager_med_us(fn), 1), "eager"


def main():
    ver = env_versions()
    print(f"GPU: {ver['gpu']}  torch={ver['torch']} vllm={ver['vllm']} triton={ver['triton']}")
    print(f"Shape E={E} H={H} I={I} topk={TOPK}\n")

    w1, w2 = make_weights()
    print("Repacking Marlin weights (once, outside timed region)...")
    w1q8, w1s8, w1r8 = repack_moe(w1, marlin_quant_fp8_torch, -1)
    w2q8, w2s8, w2r8 = repack_moe(w2, marlin_quant_fp8_torch, -1)
    w1q4, w1s4, w1r4 = repack_moe(w1, rand_marlin_weight_mxfp4_like, 32)
    w2q4, w2s4, w2r4 = repack_moe(w2, rand_marlin_weight_mxfp4_like, 32)
    print(f"  fp8:   w1q{tuple(w1q8.shape)} w2q{tuple(w2q8.shape)}")
    print(f"  mxfp4: w1q{tuple(w1q4.shape)} w2q{tuple(w2q4.shape)}")

    # ---- correctness (once, at a representative T), vs bf16 on each path's deq weights
    xc, twc, tic = make_routing(128, seed=128)
    fp8_out = fused_marlin_moe(xc, w1q8, w2q8, None, None, w1s8, w2s8, twc, tic, FP8_ID, global_num_experts=E)
    fp8_ref = fused_experts(xc, w1r8, w2r8, twc, tic)
    fp8_rel = ((fp8_out.float() - fp8_ref.float()).norm() / fp8_ref.float().norm()).item()
    mx_out = fused_marlin_moe(xc, w1q4, w2q4, None, None, w1s4, w2s4, twc, tic, FP4_ID, global_num_experts=E)
    mx_ref = fused_experts(xc, w1r4, w2r4, twc, tic)
    mx_rel = ((mx_out.float() - mx_ref.float()).norm() / mx_ref.float().norm()).item()
    print(f"\nCorrectness @T=128 (rel-err vs bf16 on deq weights): "
          f"fp8={fp8_rel:.4f}  mxfp4_EMU={mx_rel:.4f}")
    del w1r8, w2r8, w1r4, w2r4, fp8_ref, mx_ref
    torch.cuda.empty_cache()

    rows = []
    for T in TOKENS:
        x, tw, ti = make_routing(T, seed=T)
        bf16_us, m0 = timed(lambda: fused_experts(x, w1, w2, tw, ti))
        fp8_us, m1 = timed(lambda: fused_marlin_moe(x, w1q8, w2q8, None, None, w1s8, w2s8, tw, ti, FP8_ID, global_num_experts=E))
        mx_us, m2 = timed(lambda: fused_marlin_moe(x, w1q4, w2q4, None, None, w1s4, w2s4, tw, ti, FP4_ID, global_num_experts=E))
        row = {
            "T": T,
            "bf16": bf16_us, "fp8_w8a16": fp8_us, "mxfp4_w4a16_EMU": mx_us,
            "fp8_vs_bf16": round(bf16_us / fp8_us, 3),
            "mxfp4_EMU_vs_bf16": round(bf16_us / mx_us, 3),
            "timing": {"bf16": m0, "fp8": m1, "mxfp4": m2},
        }
        rows.append(row)
        print(f"T={T:4d}  bf16={bf16_us:7.0f}u  fp8={fp8_us:7.0f}u ({row['fp8_vs_bf16']:.2f}x)  "
              f"mxfp4_EMU={mx_us:7.0f}u ({row['mxfp4_EMU_vs_bf16']:.2f}x)")
        del x, tw, ti
        torch.cuda.empty_cache()

    key = gpu_key()
    fp4_native = "fp4" in native_low_precisions()
    out = {
        "experiment": "A_fused_moe_cuda_vs_triton",
        "gpu": ver["gpu"], "gpu_key": key, "versions": ver,
        "shape": {"E": E, "H": H, "I": I, "topk": TOPK},
        "method": "graph_med_us 10 launches/graph, median of 40 replays (matches Triton extended sweep)",
        "note_mxfp4": "mxfp4_w4a16 here is the Marlin path, which dequantizes FP4->bf16 "
                      "in-kernel and runs bf16 tensor cores on EVERY GPU (EMU is intrinsic "
                      "to Marlin, not a Hopper-only limitation). The NATIVE FP4-MMA leg "
                      "(matched-precision W4A4) uses a different kernel (cutlass/trtllm nvfp4) "
                      "and lives in bench_moe_nvfp4_native.py; it is only native on SM100+ "
                      f"(fp4_native_hw={fp4_native} on this {key}).",
        "fp4_native_hw": fp4_native,
        "correctness": {"fp8_w8a16_relerr": round(fp8_rel, 4), "mxfp4_w4a16_EMU_relerr": round(mx_rel, 4),
                        "ref": "bf16 fused_experts on each path's dequantized weights @T=128"},
        "rows": rows,
    }
    # GPU-keyed filename so H100/B200/B100 runs coexist; the committed
    # results_cuda_moe.json remains the historical H100 reference.
    save_json(os.path.join(_D, f"results_cuda_moe_{key}.json"), out)


if __name__ == "__main__":
    main()
