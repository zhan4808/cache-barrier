"""Figure: W8A8 INT8-MMA results for MLA reconstruction (graph-timed, H100)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "results_w8a8.json")) as f:
    R = json.load(f)

C_FP, C_W8, C_W4 = "#1565C0", "#2E7D32", "#C62828"
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("W8A8 (INT8 tensor-core MMA, scales on accumulator only) vs cuBLAS FP16 vs Triton W4A16\n"
             "MLA reconstruction BMM, H100, CUDA-graph timed; W8A8 includes dynamic activation quant",
             y=1.04)

# A: weight-size sweep at bs=1 (the decode case)
sz = R["size_sweep"]
mb = np.array([r["weight_mb"] for r in sz])
a.plot(mb, [r["fp16_us"] for r in sz], "o-", color=C_FP, lw=2, label="FP16 cuBLAS")
a.plot(mb, [r["w8a8_us"] for r in sz], "o-", color=C_W8, lw=2, label="W8A8 (this work)")
a.plot(mb, [r["w4a16_us"] for r in sz], "^--", color=C_W4, lw=1.5, alpha=0.8, label="W4A16 (dequant-bound)")
a.axvspan(32, 40, color="orange", alpha=0.15)
a.axvline(36, color="orange", ls="--", lw=1.2)
a.text(41, 51, "effective L2\ncapacity ≈36 MB", fontsize=8, color="#E65100")
a.text(16, 30, "L2-resident:\nFP16 wins (0.7x)\n— residency removes\nbyte-savings upside", fontsize=8, color=C_FP, ha="center")
a.text(85, 18, "HBM-served:\nW8A8 wins 1.4–1.5x\n— byte savings convert\nto latency with INT8 MMA", fontsize=8, color=C_W8, ha="center")
a.set_xlabel("FP16 weight size (MB)")
a.set_ylabel("latency per BMM (µs)")
a.set_title("A — bs=1 decode across the L2 cliff:\nfirst quantized kernel that beats cuBLAS FP16")
a.grid(alpha=0.25); a.legend(fontsize=8.5, loc="upper left")

# B: batch sweep at MLA shape
bs_rows = R["bs_sweep"]
bs = np.array([r["bs"] for r in bs_rows])
b.semilogx(bs, [r["w8a8_speedup"] for r in bs_rows], "o-", color=C_W8, lw=2,
           label="W8A8 vs FP16 (16 MB, L2-resident)")
b.semilogx(bs, [r["fp16_us"] / r["w4a16_us"] for r in bs_rows], "^--", color=C_W4,
           lw=1.5, alpha=0.8, label="W4A16 vs FP16")
sz_su = [r["w8a8_speedup"] for r in sz if r["weight_mb"] >= 48]
b.axhline(np.mean(sz_su), color=C_W8, ls=":", lw=1.5)
b.text(1.2, np.mean(sz_su) + 0.03, f"W8A8 above L2 cliff (bs=1): {np.mean(sz_su):.2f}x", fontsize=8, color=C_W8)
b.axhline(1.0, color="k", lw=1, alpha=0.5)
b.set_ylim(0, 1.8)
b.set_xlabel("batch size")
b.set_ylabel("speedup vs FP16 cuBLAS (>1 = quantized wins)")
b.set_title("B — at the 16 MB L2-resident MLA shape no quantized\nkernel wins at any bs (L2 barrier confirmed causally)")
b.grid(alpha=0.25); b.legend(fontsize=8.5)

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, f"w8a8_results.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
