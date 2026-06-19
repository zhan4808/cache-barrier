"""Task 3 (v2) -- CARM dispatch at the REAL target-model MoE shapes.

DeepSeek-V4-Flash and Qwen3.6-35B-A3B are fine-grained MoE (E=256, small experts,
high top-k) -- very different from Mixtral (E=8, big experts, top-k=2). The point:
the quant-vs-dense crossover MOVES with shape, exactly as the shape-parameterized
CARM predicts, and the dispatcher generalizes (= oracle at every shape).

Configs from HF config.json (2026-06):
  Mixtral-8x7B (ref): E=8,   H=4096, I=14336, topk=2
  DeepSeek-V4-Flash : E=256, H=4096, I=2048,  topk=6   (deepseek-ai/DeepSeek-V4-Flash)
  Qwen3.6-35B-A3B   : E=256, H=2048, I=512,   topk=8   (Qwen/Qwen3.6-35B-A3B)

All graph-timed, clock-locked. Output: results_task3_target_shapes.json
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import graph_med_us, save_json  # noqa: E402
from bench_cuda_moe import repack_moe  # noqa: E402  (generic per-expert Marlin repack)
from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch  # noqa: E402

DEV, DT = "cuda", torch.bfloat16
FP8_ID = scalar_types.float8_e4m3fn.id
_D = os.path.dirname(os.path.abspath(__file__))

SHAPES = {
    "Mixtral-8x7B": dict(E=8, H=4096, I=14336, topk=2),
    "DeepSeek-V4-Flash": dict(E=256, H=4096, I=2048, topk=6),
    "Qwen3.6-35B-A3B": dict(E=256, H=2048, I=512, topk=8),
}
GRID = [16, 64, 128, 256, 512, 1024, 2048]
TRACE = {16: 50, 64: 25, 128: 12, 256: 8, 512: 6, 1024: 3, 2048: 1}


def make_weights(E, H, I, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DT, generator=g) / H**0.5
    w2 = torch.randn(E, H, I, device=DEV, dtype=DT, generator=g) / I**0.5
    return w1, w2


def make_routing(T, H, E, topk, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=DT, generator=g) / 10
    gate = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=g)
    tw, ti = torch.topk(torch.softmax(gate, -1), topk, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
    return x, tw.contiguous(), ti.to(torch.int32).contiguous()


def run_shape(name, cfg):
    E, H, I, topk = cfg["E"], cfg["H"], cfg["I"], cfg["topk"]
    print(f"\n=== {name}  E={E} H={H} I={I} topk={topk} ===")
    w1, w2 = make_weights(E, H, I)
    w1q, w1s, w1r = repack_moe(w1, marlin_quant_fp8_torch, -1)
    w2q, w2s, w2r = repack_moe(w2, marlin_quant_fp8_torch, -1)

    # correctness @T=128
    xc, twc, tic = make_routing(128, H, E, topk, seed=128)
    o_q = fused_marlin_moe(xc, w1q, w2q, None, None, w1s, w2s, twc, tic, FP8_ID, global_num_experts=E)
    o_r = fused_experts(xc, w1r, w2r, twc, tic)
    rel = ((o_q.float() - o_r.float()).norm() / o_r.float().norm()).item()
    del w1r, w2r, o_r
    torch.cuda.empty_cache()

    bf16_us, fp8_us = {}, {}
    for T in GRID:
        x, tw, ti = make_routing(T, H, E, topk, seed=T)
        for _ in range(3):
            fused_experts(x, w1, w2, tw, ti)
        torch.cuda.synchronize()
        bf16_us[T] = round(graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti)), 1)
        fp8_us[T] = round(graph_med_us(lambda: fused_marlin_moe(x, w1q, w2q, None, None, w1s, w2s, tw, ti, FP8_ID, global_num_experts=E)), 1)
        del x, tw, ti
        torch.cuda.empty_cache()

    Tstar = next((T for T in GRID if fp8_us[T] >= bf16_us[T]), None)
    print(f"  correctness rel-err={rel:.4f}")
    print(f"   T     bf16    fp8    fp8/bf16")
    for T in GRID:
        print(f"  {T:5d} {bf16_us[T]:8.1f} {fp8_us[T]:7.1f}   {bf16_us[T]/fp8_us[T]:.2f}x")
    print(f"  crossover T* = {Tstar if Tstar else '>2048 (fp8 wins whole range)'}")

    Ts = sorted(TRACE)
    tb = sum(TRACE[T] * bf16_us[T] for T in Ts)
    tf = sum(TRACE[T] * fp8_us[T] for T in Ts)
    Tc = Tstar if Tstar else 10**9
    td = sum(TRACE[T] * (fp8_us[T] if T < Tc else bf16_us[T]) for T in Ts)
    to = sum(TRACE[T] * min(bf16_us[T], fp8_us[T]) for T in Ts)
    print(f"  trace: bf16={tb/1000:.2f}ms fp8={tf/1000:.2f}ms dispatch={td/1000:.2f}ms "
          f"({tb/td:.2f}x vs bf16, {tf/td:.2f}x vs fp8, ==oracle:{abs(td-to)<1e-6})")

    del w1, w2, w1q, w2q
    torch.cuda.empty_cache()
    return {"cfg": cfg, "relerr": round(rel, 4), "bf16_us": bf16_us, "fp8_us": fp8_us,
            "crossover_Tstar": Tstar,
            "trace_totals_ms": {"bf16": round(tb/1000, 2), "fp8": round(tf/1000, 2),
                                "dispatch": round(td/1000, 2), "oracle": round(to/1000, 2)},
            "dispatch_speedup_vs_bf16": round(tb/td, 3), "dispatch_vs_fp8": round(tf/td, 3)}


def main():
    print(f"GPU {torch.cuda.get_device_name(0)} (SM clock locked)")
    out = {"experiment": "task3_target_shapes", "gpu": torch.cuda.get_device_name(0),
           "trace": TRACE, "shapes": {}}
    for name, cfg in SHAPES.items():
        out["shapes"][name] = run_shape(name, cfg)
    print("\n=== crossover moves with shape (the shape-parameterized CARM prediction) ===")
    for name in SHAPES:
        s = out["shapes"][name]
        print(f"  {name:20s} T*={str(s['crossover_Tstar']):>6s}   "
              f"dispatch {s['dispatch_speedup_vs_bf16']:.2f}x vs bf16, {s['dispatch_vs_fp8']:.2f}x vs fp8")
    save_json(os.path.join(_D, "results_task3_target_shapes.json"), out)


if __name__ == "__main__":
    main()
