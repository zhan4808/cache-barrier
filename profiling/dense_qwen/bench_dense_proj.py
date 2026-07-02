"""Dense-model mixed-precision crossover — Qwen3.6-27B projections (Dr. Xiao todo #1).

Motivation. Dr. Xiao's kernel profile of Qwen3.6-27B (vLLM, decode-heavy) shows
aten::mm (cuBLAS nvjet GEMMs) = 86.2% of GPU time: the dense projections are the
whole game. Its per-layer weights also straddle H100's ~36 MB effective L2
(kv_proj 21 MB < C_eff < q/o_proj 63 MB << FFN 178-356 MB), so ONE model probes
all three CARM regimes at real shapes:

  * left edge  (operand fits L2)      -> quant should LOSE (kv_proj)
  * boundary   (bf16 spills, fp8 fits)-> quant should win SUPER-proportionally
                                         (q/o_proj: 63 MB bf16 -> 31 MB fp8
                                         crosses INTO L2 -- byte halving x BW-tier
                                         jump; a plain roofline cannot predict this)
  * middle     (HBM-streamed)         -> quant wins ~bytes ratio (gate_up/down)
  then compute-bound at large M on the right for every shape.

Paths (all CUDA-graph timed, methodology identical to cuda_validation):
  bf16    torch.mm on [M,K] @ [K,N] (weight stored [N,K], mm with .t() view --
          same cuBLAS nvjet TN path as the profile)
  w8a16   vLLM Marlin fp8 weight-only (apply_fp8_marlin_linear), dequant in-core
  w8a8    NATIVE matched fp8: dynamic per-tensor act quant (scaled_fp8_quant)
          + cutlass_scaled_mm. Act-quant cost is INCLUDED in the timed region
          (it is part of the deployed path); also reported mm-only.

Serving-realism mode (--rotate): each in-graph launch uses a different weight
copy (R copies sized so R x bytes > 2 x C_eff), defeating L2 residency the way
64 layers of real serving would. Comparing warm vs rotated isolates how much of
any win/loss is an L2-residency (microbenchmark) artifact vs robust.

Output: results_dense_proj_<gpu_key>.json
"""

import argparse
import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json, gpu_key  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    marlin_quant_fp8_torch, marlin_make_workspace_new, apply_fp8_marlin_linear,
)

DEV, DT = "cuda", torch.bfloat16
C_EFF_MB = 36.0  # measured effective L2 (cache-barrier 2026-06)

# Qwen/Qwen3.6-27B text_config (HF, 2026-07-01): hidden 5120, inter 17408,
# 24 q-heads x 256, 4 kv-heads x 256. K = input dim, N = output dim.
SHAPES = [
    # name          K       N       bf16 MB   fp8 MB   regime hypothesis
    ("kv_proj",     5120,   2048),   # 21.0    10.5    both fit L2 -> quant loses
    ("q_proj",      5120,   6144),   # 62.9    31.5    fp8 crosses INTO L2 -> super-win
    ("o_proj",      6144,   5120),   # 62.9    31.5    same boundary case
    ("down_proj",  17408,   5120),   # 178.3   89.2    HBM-streamed -> ~bytes ratio
    ("gate_up",     5120,  34816),   # 356.5   178.3   HBM-streamed -> ~bytes ratio
]
MS = [1, 4, 16, 64, 128, 256, 512, 1024, 2048]


def wt_mb(K, N, bytes_per=2):
    return K * N * bytes_per / 1e6


def make_copies(K, N, rotate, seed):
    """Weight copies [N,K]. rotate=True -> enough copies to defeat L2 residency."""
    n_cop = 1
    if rotate:
        n_cop = max(2, int(2 * C_EFF_MB * 1e6 / (K * N * 2)) + 1)
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5
            for _ in range(n_cop)]


def bench_shape(name, K, N, rotate):
    ws_marlin = marlin_make_workspace_new(DEV)
    weights = make_copies(K, N, rotate, seed=hash(name) % 2**31)
    n_cop = len(weights)

    # per-copy prep (outside timed region)
    wT = [w.t().contiguous().t() for w in weights]          # [N,K] col-major-ish view for mm
    marlin = [marlin_quant_fp8_torch(w, -1) for w in weights]  # (ref[K,N], qw, s)
    fp8 = []
    for w in weights:
        wq, wscale = ops.scaled_fp8_quant(w)                # [N,K] e4m3 + scale
        fp8.append((wq.t(), wscale))                        # cutlass wants [K,N] col-view

    x = torch.randn(MS[-1], K, device=DEV, dtype=DT) / 32

    # correctness once at M=64 vs each path's own reference
    xc = x[:64]
    ref, qw, s = marlin[0]
    y16 = apply_fp8_marlin_linear(xc, qw, s, ws_marlin, N, K, None)
    r16 = ((y16.float() - (xc @ ref.to(DT)).float()).norm() /
           (xc @ ref.to(DT)).float().norm()).item()
    xq, xs = ops.scaled_fp8_quant(xc)
    y8 = ops.cutlass_scaled_mm(xq, fp8[0][0], xs, fp8[0][1], DT)
    r8 = ((y8.float() - (xc @ weights[0].t()).float()).norm() /
          (xc @ weights[0].t()).float().norm()).item()

    rows = []
    for M in MS:
        xm = x[:M].contiguous()
        it = {"i": 0}

        def bf16_fn():
            torch.mm(xm, weights[it["i"] % n_cop].t()); it["i"] += 1

        def w8a16_fn():
            _, qw, s = marlin[it["i"] % n_cop]
            apply_fp8_marlin_linear(xm, qw, s, ws_marlin, N, K, None); it["i"] += 1

        def w8a8_fn():
            wq, wscale = fp8[it["i"] % n_cop]
            xq, xs = ops.scaled_fp8_quant(xm)
            ops.cutlass_scaled_mm(xq, wq, xs, wscale, DT); it["i"] += 1

        def w8a8_mm_fn():  # mm-only (act pre-quantized) for the act-quant overhead split
            wq, wscale = fp8[it["i"] % n_cop]
            ops.cutlass_scaled_mm(xq_s, wq, xs_s, wscale, DT); it["i"] += 1

        xq_s, xs_s = ops.scaled_fp8_quant(xm)
        r = {"M": M}
        for tag, fn in [("bf16", bf16_fn), ("w8a16", w8a16_fn),
                        ("w8a8", w8a8_fn), ("w8a8_mm", w8a8_mm_fn)]:
            it["i"] = 0
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            r[tag] = round(graph_med_us(fn), 2)
        r["w8a16_vs_bf16"] = round(r["bf16"] / r["w8a16"], 3)
        r["w8a8_vs_bf16"] = round(r["bf16"] / r["w8a8"], 3)
        rows.append(r)
        print(f"  M={M:5d}  bf16={r['bf16']:9.1f}u  w8a16={r['w8a16']:9.1f}u "
              f"({r['w8a16_vs_bf16']:5.2f}x)  w8a8={r['w8a8']:9.1f}u "
              f"({r['w8a8_vs_bf16']:5.2f}x)  [mm-only {r['w8a8_mm']:8.1f}u]")

    del weights, wT, marlin, fp8, x
    torch.cuda.empty_cache()
    return {"shape": name, "K": K, "N": N,
            "wt_mb_bf16": round(wt_mb(K, N), 1), "wt_mb_fp8": round(wt_mb(K, N, 1), 1),
            "n_weight_copies": n_cop, "rotated": rotate,
            "relerr": {"w8a16": round(r16, 4), "w8a8": round(r8, 4)},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", action="store_true",
                    help="cycle weight copies to defeat L2 residency (serving realism)")
    args = ap.parse_args()

    ver = env_versions()
    key = gpu_key()
    print(f"GPU {ver['gpu']} ({key})  torch={ver['torch']} vllm={ver['vllm']}  "
          f"rotate={args.rotate}\n")

    results = []
    for name, K, N in SHAPES:
        mb = wt_mb(K, N)
        print(f"== {name}  [{K}x{N}]  bf16 {mb:.0f} MB / fp8 {mb/2:.0f} MB  "
              f"(C_eff {C_EFF_MB:.0f} MB) ==")
        results.append(bench_shape(name, K, N, args.rotate))
        print()

    suffix = "_rotated" if args.rotate else ""
    out = {
        "experiment": "dense_qwen3.6-27B_projections_mixed_precision",
        "gpu": ver["gpu"], "gpu_key": key, "versions": ver,
        "model": "Qwen/Qwen3.6-27B (hidden 5120, inter 17408, 24qh/4kvh x 256)",
        "method": "graph_med_us 10/graph median-of-40; w8a8 includes dynamic "
                  "per-tensor act quant in timed region (w8a8_mm = mm only); "
                  "rotate mode cycles >2xC_eff of weight copies per graph",
        "c_eff_mb": C_EFF_MB,
        "results": results,
    }
    save_json(os.path.join(_D, f"results_dense_proj_{key}{suffix}.json"), out)


if __name__ == "__main__":
    main()
