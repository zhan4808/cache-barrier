"""Red-team the Task-3 target-shape claim: does fp8 still win across the token range
against a PROPERLY-TUNED bf16 baseline (not stock vLLM)? Same trap that inflated the
Mixtral crossover -- applied here to DeepSeek-V4-Flash and Qwen3.6-35B.

For each T: time fp8 Marlin, time bf16 with the stock default config AND with
GROUP_SIZE_M in {1,8,16,32,64} (best = tuned baseline). Report the crossover vs
stock and vs tuned, plus the default GROUP_SIZE_M (to see if stock is under-tuned).
Clock-locked. Output: results_target_tuned.json
"""
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import graph_med_us, save_json  # noqa: E402
from bench_cuda_moe import repack_moe  # noqa: E402
from task3_target_shapes import make_weights, make_routing  # noqa: E402
import vllm.model_executor.layers.fused_moe.fused_moe as VF  # noqa: E402
from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch  # noqa: E402

FP8_ID = scalar_types.float8_e4m3fn.id
_D = os.path.dirname(os.path.abspath(__file__))
SHAPES = {
    "DeepSeek-V4-Flash": dict(E=256, H=4096, I=2048, topk=6),
    "Qwen3.6-35B-A3B": dict(E=256, H=2048, I=512, topk=8),
}
GRID = [16, 64, 128, 256, 512, 1024, 2048]
GROUPS = [1, 8, 16, 32, 64]


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


def crossover(rows, key):
    prev = None
    for T in GRID:
        ratio = rows[T][key] / rows[T]["fp8"]
        if prev and prev[1] >= 1.0 > ratio:
            T0, r0 = prev
            return int(round(T0 + (T - T0) * (r0 - 1.0) / (r0 - ratio)))
        prev = (T, ratio)
    return None


def run(name, cfg):
    E, H, I, topk = cfg["E"], cfg["H"], cfg["I"], cfg["topk"]
    print(f"\n=== {name}  E={E} H={H} I={I} topk={topk} ===")
    w1, w2 = make_weights(E, H, I)
    w1q, w1s, _ = repack_moe(w1, marlin_quant_fp8_torch, -1)
    w2q, w2s, _ = repack_moe(w2, marlin_quant_fp8_torch, -1)

    rows = {}
    print("   T   defG  fp8_us  bf16_def  bf16_best(G)   def/fp8  best/fp8")
    for T in GRID:
        x, tw, ti = make_routing(T, H, E, topk, seed=T)
        defcfg = VF.try_get_optimal_moe_config((E, 2 * I, H), (E, H, I), topk, None, T * topk)
        for _ in range(3):
            fused_experts(x, w1, w2, tw, ti)
        torch.cuda.synchronize()
        fp8 = graph_med_us(lambda: fused_marlin_moe(x, w1q, w2q, None, None, w1s, w2s, tw, ti, FP8_ID, global_num_experts=E))
        bf_def = graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti))
        best, bestg = bf_def, "def"
        for g in GROUPS:
            with force_group_m(g):
                t = graph_med_us(lambda: fused_experts(x, w1, w2, tw, ti))
            if t < best:
                best, bestg = t, g
        rows[T] = {"fp8": fp8, "bf16_def": bf_def, "bf16_best": best, "bestG": bestg, "defG": defcfg["GROUP_SIZE_M"]}
        print(f"  {T:5d}  {defcfg['GROUP_SIZE_M']:3d}  {fp8:7.1f}  {bf_def:8.1f}  {best:8.1f}(G={str(bestg):>3s})   "
              f"{bf_def/fp8:5.2f}    {best/fp8:5.2f}")
        del x, tw, ti
        torch.cuda.empty_cache()

    xs, xt = crossover(rows, "bf16_def"), crossover(rows, "bf16_best")
    print(f"  crossover vs STOCK bf16: {xs or '>2048'}   vs TUNED bf16: {xt or '>2048'}")
    del w1, w2, w1q, w2q
    torch.cuda.empty_cache()
    return {"cfg": cfg, "rows": rows, "crossover_stock": xs, "crossover_tuned": xt}


def main():
    print(f"GPU {torch.cuda.get_device_name(0)} (clock locked)")
    out = {"experiment": "target_shape_tuned_baseline_recheck", "shapes": {}}
    for name, cfg in SHAPES.items():
        out["shapes"][name] = run(name, cfg)
    print("\n=== verdict: does fp8 still win across the range vs a TUNED bf16? ===")
    for name in SHAPES:
        s = out["shapes"][name]
        verdict = ("fp8 wins whole range" if s["crossover_tuned"] is None
                   else f"crossover at T~{s['crossover_tuned']} vs tuned bf16")
        print(f"  {name:20s} stock_xover={str(s['crossover_stock']):>6s}  tuned_xover={str(s['crossover_tuned']):>6s}  => {verdict}")
    save_json(os.path.join(_D, "results_target_tuned.json"), out)


if __name__ == "__main__":
    main()
