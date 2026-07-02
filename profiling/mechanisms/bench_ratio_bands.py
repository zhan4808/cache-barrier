"""Mechanism B — tile/wave-quantization bands in the quant-vs-bf16 ratio.

Evidence: the MoE sweep's T=512 spike / T=640 dip and the bf16 CARM MAPE of 22%
vs fp8's 12%. Hypothesis: bf16 (cuBLAS) and quant kernels use DIFFERENT tile
shapes and config heuristics, so their wave-quantization cliffs land at
different M — the speedup RATIO oscillates in bands that a smooth roofline
cannot express. A router keyed on a smooth model will misroute inside the bands.

Fine M sweep (step 8 to 256, step 32 to 1024) on two shapes:
  q_proj  (5120x6144)  — the boundary shape
  gate_up (5120x34816) — the streamed FFN shape
Paths: bf16 mm / cutlass W8A8 (mm-only) / Marlin W8A16. Rotated x2 (streamed
behavior; no residency confound).

Output: results_ratio_bands_h100.json + band statistics (how much does the
ratio swing between adjacent M?).
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
R = 2
MS = list(range(8, 257, 8)) + list(range(288, 1025, 32))
SHAPES = [("q_proj", 5120, 6144), ("gate_up", 5120, 34816)]


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']} — ratio bands, fine M sweep ({len(MS)} points/shape)\n")
    out_rows = []
    for name, K, N in SHAPES:
        g = torch.Generator(device=DEV).manual_seed(K + N)
        ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(R)]
        marl = [marlin_quant_fp8_torch(w, -1)[1:] for w in ws]
        fp8 = []
        for w in ws:
            wq, wsc = ops.scaled_fp8_quant(w)
            fp8.append((wq.t(), wsc))
        wspace = marlin_make_workspace_new(DEV)
        xfull = torch.randn(MS[-1], K, device=DEV, dtype=DT) / 32
        print(f"== {name} [{K}x{N}] ==")
        prev = None
        for M in MS:
            x = xfull[:M].contiguous()
            xq, xs = ops.scaled_fp8_quant(x)
            it = {"i": 0}

            def f_bf():
                torch.mm(x, ws[it["i"] % R].t()); it["i"] += 1

            def f_88():
                wq, wsc = fp8[it["i"] % R]
                ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1

            def f_ml():
                qw, s = marl[it["i"] % R]
                apply_fp8_marlin_linear(x, qw, s, wspace, N, K, None); it["i"] += 1

            r = {"shape": name, "M": M}
            for tag, fn in [("bf16", f_bf), ("w8a8", f_88), ("w8a16", f_ml)]:
                it["i"] = 0
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                r[tag] = round(graph_med_us(fn, reps=25), 2)
            r["w8a8_x"] = round(r["bf16"] / r["w8a8"], 3)
            r["w8a16_x"] = round(r["bf16"] / r["w8a16"], 3)
            out_rows.append(r)
            jump = "" if prev is None else f"  Δratio {r['w8a8_x']-prev:+.2f}"
            if M % 64 == 0 or (prev is not None and abs(r["w8a8_x"] - prev) > 0.3):
                print(f"  M={M:5d}  bf16={r['bf16']:8.1f}  w8a8={r['w8a8']:8.1f} "
                      f"({r['w8a8_x']:5.2f}x)  w8a16={r['w8a16']:8.1f} ({r['w8a16_x']:5.2f}x){jump}")
            prev = r["w8a8_x"]
        del ws, marl, fp8
        torch.cuda.empty_cache()
        print()
    save_json(os.path.join(_D, "results_ratio_bands_h100.json"), {
        "experiment": "mechanism_B_tile_quantization_bands", "gpu": ver["gpu"],
        "method": f"graph_med_us reps=25; rotated x{R}; fine M grid", "rows": out_rows})


if __name__ == "__main__":
    main()
