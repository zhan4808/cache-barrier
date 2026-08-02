"""Fine-grained L2 boundary sweep — pins the capacity cliff for CARM v2.

Synthetic dense GEMMs (K=8192 fixed, N swept) so the bf16 weight crosses
C_eff ~= 36 MB in fine steps: 8..128 MB. M=16 (memory-bound, decode-like).
Paths: bf16 mm, w8a8 mm-only (act pre-quantized -- isolates the weight-serving
tier), w8a16 Marlin. Warm vs rotated per point.

Expected: bf16 effective BW steps down from L2-tier (~4.2) to HBM (~2.8) as
weight crosses ~36 MB; w8a8's step happens at 2x the bf16 size (its weight is
half); Marlin flat ~2.5 (no L2 exploitation, per the CARM v2 fit). Rotated:
everything flat at HBM tier.

Output: results_l2_boundary_h100.json
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
_ap = argparse.ArgumentParser()
_ap.add_argument("--c-eff-mb", type=float, default=36.0,
                 help="effective LLC capacity of THIS card (b200: 98.8)")
_ap.add_argument("--t0-us", type=float, default=2.78,
                 help="graph launch floor of THIS card (b200: 2.29)")
_ap.add_argument("--targets-mb", type=str, default="8,16,24,32,40,48,64,96,128",
                 help="bf16 weight sizes; on b200 extend above the gate, "
                      "e.g. 8,...,128,160,192,256,320")
ARGS = _ap.parse_args()
C_EFF_MB = ARGS.c_eff_mb
T0 = ARGS.t0_us
K = 8192
M = 16
# N chosen so bf16 weight MB = K*N*2/1e6 hits these targets
TARGETS_MB = [float(x) for x in ARGS.targets_mb.split(",")]


def one(N, rotate):
    n_cop = 1 if not rotate else max(2, int(2 * C_EFF_MB * 1e6 / (K * N * 2)) + 1)
    g = torch.Generator(device=DEV).manual_seed(N)
    ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(n_cop)]
    marl = [marlin_quant_fp8_torch(w, -1)[1:] for w in ws]     # (qw, s)
    fp8 = []
    for w in ws:
        wq, wsc = ops.scaled_fp8_quant(w)
        fp8.append((wq.t(), wsc))
    wspace = marlin_make_workspace_new(DEV)
    x = torch.randn(M, K, device=DEV, dtype=DT) / 32
    xq, xs = ops.scaled_fp8_quant(x)
    it = {"i": 0}

    def f_bf():
        torch.mm(x, ws[it["i"] % n_cop].t()); it["i"] += 1

    def f_88():
        wq, wsc = fp8[it["i"] % n_cop]
        ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1

    def f_16():
        qw, s = marl[it["i"] % n_cop]
        apply_fp8_marlin_linear(x, qw, s, wspace, N, K, None); it["i"] += 1

    r = {}
    for tag, fn in [("bf16", f_bf), ("w8a8_mm", f_88), ("w8a16", f_16)]:
        it["i"] = 0
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        r[tag] = round(graph_med_us(fn), 2)
    del ws, marl, fp8
    torch.cuda.empty_cache()
    return r, n_cop


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']}  K={K} M={M}  sweep bf16 weight 8->128 MB\n")
    print(f"{'MB':>5} {'N':>7} | {'bf16 us':>8} {'BW':>5} | {'w8a8 us':>8} {'BW':>5} | "
          f"{'w8a16 us':>8} {'BW':>5} | mode")
    rows = []
    for mode in ("warm", "rotated"):
        for mb in TARGETS_MB:
            N = int(mb * 1e6 / K / 2 / 256) * 256
            wmb = K * N * 2 / 1e6
            r, n_cop = one(N, mode == "rotated")
            bw = lambda m, u: round(m / max(u - T0, .05), 2)
            row = {"mode": mode, "N": N, "wt_mb_bf16": round(wmb, 1),
                   "wt_mb_fp8": round(wmb / 2, 1), "n_copies": n_cop, **r,
                   "bw_bf16": bw(wmb, r["bf16"]), "bw_w8a8": bw(wmb / 2, r["w8a8_mm"]),
                   "bw_w8a16": bw(wmb / 2, r["w8a16"])}
            rows.append(row)
            print(f"{wmb:>5.0f} {N:>7} | {r['bf16']:>8.2f} {row['bw_bf16']:>5.2f} | "
                  f"{r['w8a8_mm']:>8.2f} {row['bw_w8a8']:>5.2f} | "
                  f"{r['w8a16']:>8.2f} {row['bw_w8a16']:>5.2f} | {mode}")
        print()
    save_json(os.path.join(_D, f"results_l2_boundary_{gpu_key()}.json"), {
        "experiment": "l2_boundary_sweep_dense", "gpu": ver["gpu"], "K": K, "M": M,
        "method": "graph_med_us; BW = weight_MB/(t-t0); w8a8 is mm-only", "rows": rows})


if __name__ == "__main__":
    main()
