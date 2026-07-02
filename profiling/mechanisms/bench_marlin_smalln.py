"""Mechanism A — Marlin's small-N fixed overhead (routing-relevant, roofline-invisible).

Evidence: L2 boundary sweep showed Marlin W8A16 taking 37 us on a 4 MB GEMM that
bf16 does in 6 us (N=256), while its large-N behavior is dequant-ceiling-like.
Hypothesis: a fixed cost independent of weight bytes (global workspace
reduction/sync across SMs + tile shape floor), i.e. t_marlin ~= t_fix(N,K) +
W/BW_marlin, with t_fix exploding as N shrinks (fewer N-tiles -> less parallelism
but same sync structure).

Sweep N x K at M in {1,16}; paths: bf16 mm / marlin W8A16 / cutlass W8A8 (mm-only).
Rotated weights (2 copies) to avoid L2-residency confounds for the small sizes —
we want the STREAMED behavior of each kernel, not the cache story.

Output: results_marlin_smalln_h100.json + fitted overhead table.
"""

import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    marlin_quant_fp8_torch, marlin_make_workspace_new, apply_fp8_marlin_linear,
)

DEV, DT = "cuda", torch.bfloat16
T0 = 2.78
NS = [256, 512, 1024, 2048, 4096, 8192, 16384]
KS = [2048, 5120, 14336]
MS = [1, 16]
R = 2  # weight copies (defeat residency for small sizes)


def one(K, N, M):
    g = torch.Generator(device=DEV).manual_seed(K + N)
    ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(R)]
    marl = [marlin_quant_fp8_torch(w, -1)[1:] for w in ws]
    fp8 = []
    for w in ws:
        wq, wsc = ops.scaled_fp8_quant(w)
        fp8.append((wq.t(), wsc))
    wspace = marlin_make_workspace_new(DEV)
    x = torch.randn(M, K, device=DEV, dtype=DT) / 32
    xq, xs = ops.scaled_fp8_quant(x)
    it = {"i": 0}

    def f_bf():
        torch.mm(x, ws[it["i"] % R].t()); it["i"] += 1

    def f_ml():
        qw, s = marl[it["i"] % R]
        apply_fp8_marlin_linear(x, qw, s, wspace, N, K, None); it["i"] += 1

    def f_88():
        wq, wsc = fp8[it["i"] % R]
        ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1

    out = {}
    for tag, fn in [("bf16", f_bf), ("marlin", f_ml), ("w8a8", f_88)]:
        it["i"] = 0
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        out[tag] = round(graph_med_us(fn), 2)
    del ws, marl, fp8
    torch.cuda.empty_cache()
    return out


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']} — Marlin small-N overhead (rotated x{R})\n")
    rows = []
    for K in KS:
        print(f"--- K={K} ---")
        print(f"{'N':>7} {'M':>3} {'MB(fp8)':>8} | {'bf16':>8} {'marlin':>8} {'w8a8':>8} | "
              f"{'ml_fix_us':>9} {'ml/bf16':>8}")
        for N in NS:
            for M in MS:
                r = one(K, N, M)
                mb8 = K * N / 1e6
                # marlin fixed overhead estimate: t - t0 - bytes/BW_streamed(2.5)
                fix = r["marlin"] - T0 - mb8 / 2.5
                rows.append({"K": K, "N": N, "M": M, "wt_mb_fp8": round(mb8, 2), **r,
                             "marlin_fixed_est_us": round(fix, 2),
                             "marlin_vs_bf16": round(r["bf16"] / r["marlin"], 2)})
                print(f"{N:>7} {M:>3} {mb8:>8.2f} | {r['bf16']:>8.2f} {r['marlin']:>8.2f} "
                      f"{r['w8a8']:>8.2f} | {fix:>9.2f} {r['bf16']/r['marlin']:>7.2f}x")
    save_json(os.path.join(_D, "results_marlin_smalln_h100.json"), {
        "experiment": "mechanism_A_marlin_smalln_overhead", "gpu": ver["gpu"],
        "method": f"graph_med_us; {R} rotated weight copies; fixed-cost est = "
                  "t - t0 - fp8_bytes/2.5TB/s", "rows": rows})


if __name__ == "__main__":
    main()
