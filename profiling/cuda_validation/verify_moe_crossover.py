"""Red-team: is the bf16 T=512 spike real vLLM behavior, and does it inflate the
crossover? Fine-grained bf16/fp8 sweep with (a) vLLM's default config and (b)
GROUP_SIZE_M forced to 16 (the value the heuristic picks for T>=640). If the spike
is the GROUP_SIZE_M=1 default-heuristic cliff, forcing 16 removes it and the
crossover moves earlier -> the ~600 measured number is an under-tuned-baseline
artifact, and the honest crossover is closer to the roofline ~330-450.
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
GRID = [256, 320, 384, 448, 512, 576, 640, 704, 768]


@contextlib.contextmanager
def force_group_m(g):
    orig = VF.try_get_optimal_moe_config

    def patched(*a, **k):
        cfg = dict(orig(*a, **k))
        cfg["GROUP_SIZE_M"] = g
        return cfg
    VF.try_get_optimal_moe_config = patched
    try:
        yield
    finally:
        VF.try_get_optimal_moe_config = orig


def med3(fn):
    """median of 3 graph_med_us measurements + spread."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    xs = [graph_med_us(fn) for _ in range(3)]
    return round(statistics.median(xs), 1), round(min(xs), 1), round(max(xs), 1)


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  shape E={E} H={H} I={I} topk={TOPK}")
    w1, w2 = B.make_weights()
    w1q8, w1s8, _ = B.repack_moe(w1, B.marlin_quant_fp8_torch, -1)
    w2q8, w2s8, _ = B.repack_moe(w2, B.marlin_quant_fp8_torch, -1)

    print("\n T   M  defGRP  bf16_def   bf16_grp16   fp8       def/fp8  grp16/fp8")
    rows = []
    for T in GRID:
        x, tw, ti = B.make_routing(T, seed=T)
        defcfg = VF.try_get_optimal_moe_config((E, 2 * I, H), (E, H, I), TOPK, None, T * TOPK)
        bf_def = med3(lambda: fused_experts(x, w1, w2, tw, ti))[0]
        with force_group_m(16):
            bf_g16 = med3(lambda: fused_experts(x, w1, w2, tw, ti))[0]
        fp8 = med3(lambda: fused_marlin_moe(x, w1q8, w2q8, None, None, w1s8, w2s8, tw, ti, FP8_ID, global_num_experts=E))[0]
        rows.append((T, bf_def, bf_g16, fp8))
        print(f"{T:4d} {T*TOPK:5d}   {defcfg['GROUP_SIZE_M']:2d}   {bf_def:8.1f}   {bf_g16:8.1f}   {fp8:8.1f}   "
              f"{bf_def/fp8:5.2f}    {bf_g16/fp8:5.2f}")
        del x, tw, ti
        torch.cuda.empty_cache()

    def crossover(idx):  # idx 1=bf_def, 2=bf_g16
        prev = None
        for r in rows:
            ratio = r[idx] / r[3]
            if prev and prev[1] >= 1.0 > ratio:
                T0, r0 = prev; T1, r1 = r[0], ratio
                return int(round(T0 + (T1 - T0) * (r0 - 1.0) / (r0 - r1)))
            prev = (r[0], ratio)
        return None
    print(f"\ncrossover (fp8 stops beating bf16):")
    print(f"  vs bf16 DEFAULT config (GROUP_SIZE_M=1 below M=1280):  T* = {crossover(1)}")
    print(f"  vs bf16 GROUP_SIZE_M=16 (properly tuned):              T* = {crossover(2)}")


if __name__ == "__main__":
    main()
