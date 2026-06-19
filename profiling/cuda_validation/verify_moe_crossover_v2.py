"""Red-team v2 (clock-locked, drift-controlled). Isolates the GROUP_SIZE_M config
effect on the bf16 baseline and re-derives the crossover.

Controls added after v1 showed a possible order/thermal confound:
  - GPU SM clock locked to 1755 MHz (run `nvidia-smi -lgc 1755,1755` first).
  - For each T the three measurements (bf16 default cfg / bf16 GROUP_SIZE_M=16 /
    fp8 Marlin) are taken in a ROTATED order across 3 rounds; median per metric.
    A monotonic drift cancels; a real config effect survives.
  - Grid extended to small T (16-128) to check the small-T win is config-robust.
  - Sanity control: at small T (few M-blocks) GROUP_SIZE_M should NOT matter, so
    bf16_default ~= bf16_grp16 there; a real effect appears only at larger T.
"""
import contextlib
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import graph_med_us  # noqa: E402
import bench_cuda_moe as B  # noqa: E402
import vllm.model_executor.layers.fused_moe.fused_moe as VF  # noqa: E402
from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe  # noqa: E402

E, H, I, TOPK, FP8_ID = B.E, B.H, B.I, B.TOPK, B.FP8_ID
GRID = [16, 64, 128, 192, 256, 384, 512, 640, 768, 1024]


@contextlib.contextmanager
def force_group_m(g):
    orig = VF.try_get_optimal_moe_config

    def patched(*a, **k):
        cfg = dict(orig(*a, **k)); cfg["GROUP_SIZE_M"] = g; return cfg
    VF.try_get_optimal_moe_config = patched
    try:
        yield
    finally:
        VF.try_get_optimal_moe_config = orig


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  SM clock locked")
    print(f"clock={torch.cuda.clock_rate() if hasattr(torch.cuda,'clock_rate') else 'n/a'}")
    w1, w2 = B.make_weights()
    w1q8, w1s8, _ = B.repack_moe(w1, B.marlin_quant_fp8_torch, -1)
    w2q8, w2s8, _ = B.repack_moe(w2, B.marlin_quant_fp8_torch, -1)

    data = {T: {"def": [], "g16": [], "fp8": []} for T in GRID}
    inputs = {T: B.make_routing(T, seed=T) for T in GRID}

    def meas_def(T):
        x, tw, ti = inputs[T]
        return graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti))

    def meas_g16(T):
        x, tw, ti = inputs[T]
        with force_group_m(16):
            return graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti))

    def meas_fp8(T):
        x, tw, ti = inputs[T]
        return graph_med_us(lambda: fused_marlin_moe(x, w1q8, w2q8, None, None, w1s8, w2s8, tw, ti, FP8_ID, global_num_experts=E))

    fns = {"def": meas_def, "g16": meas_g16, "fp8": meas_fp8}
    orders = [["def", "g16", "fp8"], ["fp8", "def", "g16"], ["g16", "fp8", "def"]]
    # warmup all
    for T in GRID:
        for k in fns:
            fns[k](T)
    torch.cuda.synchronize()
    for r, order in enumerate(orders):
        for T in GRID:
            for k in order:
                data[T][k].append(fns[k](T))
        print(f"  round {r+1}/3 done")

    print("\n T   bf16_def  bf16_g16   fp8     def/fp8 g16/fp8  g16_speedup")
    rows = []
    for T in GRID:
        d = statistics.median(data[T]["def"]); g = statistics.median(data[T]["g16"]); f = statistics.median(data[T]["fp8"])
        rows.append((T, d, g, f))
        print(f"{T:5d} {d:8.1f}  {g:8.1f}  {f:8.1f}   {d/f:5.2f}  {g/f:5.2f}   {d/g:5.2f}x")

    def crossover(idx):
        prev = None
        for r in rows:
            ratio = r[idx] / r[3]
            if prev and prev[1] >= 1.0 > ratio:
                T0, r0 = prev; T1, r1 = r[0], ratio
                return int(round(T0 + (T1 - T0) * (r0 - 1.0) / (r0 - r1)))
            prev = (r[0], ratio)
        return None
    print(f"\nCROSSOVER (fp8 W8A16 stops beating bf16):")
    print(f"  vs stock-default bf16:        T* = {crossover(1)}")
    print(f"  vs properly-tuned bf16 (g16):  T* = {crossover(2)}")
    print(f"\nSmall-T control (GROUP_SIZE_M should be irrelevant at T<=128):")
    for T, d, g, f in rows:
        if T <= 128:
            print(f"  T={T:4d}: def/g16 = {d/g:.3f}  (≈1.0 => config-independent; fp8 win {d/f:.2f}x)")


if __name__ == "__main__":
    main()
