"""Figure: FlagGems fused_moe mixed-precision analysis on H100 (Mixtral shape)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_D, "results_fused_moe_shipped.json")) as f:
    shipped = json.load(f)["sweep"]
with open(os.path.join(_D, "results_fused_moe_fixed.json")) as f:
    fixed = json.load(f)["sweep"]

T = np.array([r["tokens"] for r in shipped])
bf16 = np.array([r["bf16_us"] for r in shipped])
w8_ship = np.array([r["w8a16_us"] for r in shipped])
w8_fix = np.array([r["w8a16_us"] for r in fixed])
mxq = np.array([r["mxq_w8a16_us"] for r in shipped])

C_BF, C_SHIP, C_FIX, C_MXQ = "#1565C0", "#9E9E9E", "#2E7D32", "#C62828"
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle("FlagGems fused_moe mixed precision on H100 — Mixtral shape (E=8, H=4096, I=14336, topk=2), fp16\n"
             "PR #2336 kernel is numerically wrong (rel err ~15: computes 128 of 28672 N-columns, GEMM1 only); "
             "shipped W8A16 host-dequantizes all experts every call; fix = in-kernel dequant", y=1.06)

# A: latency vs tokens
a.loglog(T, bf16, "o-", color=C_BF, lw=2, label="bf16/fp16 (tiled, tl.dot)")
a.loglog(T, w8_ship, "s-", color=C_SHIP, lw=2,
         label="W8A16 as shipped (host dequant: ~10 GB/call)")
a.loglog(T, w8_fix, "o-", color=C_FIX, lw=2, label="W8A16 fixed (in-kernel dequant)")
a.loglog(T, mxq, "^--", color=C_MXQ, lw=1.5, alpha=0.8,
         label="PR mxq kernel (WRONG RESULTS, partial work)")
a.annotate("flat 8.5 ms =\nfull-tensor dequant\n(1.4 GB int8 → fp16 ×2)", xy=(16, 8640),
           xytext=(2.5, 3200), fontsize=8, color="#616161",
           arrowprops=dict(arrowstyle="->", color="#616161"))
a.set_xlabel("tokens"); a.set_ylabel("latency (µs)")
a.set_title("A — Latency vs token count")
a.grid(alpha=0.25, which="both"); a.legend(fontsize=7.5, loc="upper left")

# B: speedup vs bf16
b.semilogx(T, bf16 / w8_fix, "o-", color=C_FIX, lw=2, label="W8A16 fixed")
b.semilogx(T, bf16 / w8_ship, "s-", color=C_SHIP, lw=2, label="W8A16 as shipped")
b.axhline(1.0, color="k", lw=1, alpha=0.5)
b.fill_between([1, 230], 0, 2.0, color=C_FIX, alpha=0.07)
b.fill_between([230, 600], 0, 2.0, color=C_MXQ, alpha=0.07)
b.text(8, 1.85, "memory-bound:\nhalved weight bytes win", fontsize=8, color=C_FIX, ha="center")
b.text(400, 1.85, "dequant-\nstall-bound", fontsize=8, color=C_MXQ, ha="center")
b.annotate("crossover ≈ 230 tokens", xy=(230, 1.0), xytext=(40, 0.45),
           fontsize=9, arrowprops=dict(arrowstyle="->"))
b.set_ylim(0, 2.0)
b.set_xlabel("tokens"); b.set_ylabel("speedup vs bf16 (>1 = W8A16 wins)")
b.set_title("B — W8A16 speedup: wins ≤128 tokens, loses ≥256")
b.grid(alpha=0.25); b.legend(fontsize=8)

# C: effective TFLOPS + ceilings (CARM view)
flops = 2 * T * 2 * (2 * 4096 * 2 * 14336 + 4096 * 14336)
tf = lambda us: flops / (us * 1e-6) / 1e12
c.loglog(T, tf(bf16), "o-", color=C_BF, lw=2, label="bf16 measured")
c.loglog(T, tf(w8_fix), "o-", color=C_FIX, lw=2, label="W8A16 fixed measured")
c.axhline(989, color="k", lw=1.2, alpha=0.5)
c.text(1.2, 1050, "fp16 TC peak 989 TF", fontsize=8)
c.axhline(305, color=C_FIX, ls=":", lw=1.5)
c.text(1.2, 330, "W8A16 in-core ceiling ≈305 TF (NCU @T=512: 24% DRAM, 32% SM\n"
       "→ stalls on int8→fp16 convert feeding tl.dot)", fontsize=7.5, color=C_FIX)
c.set_xlabel("tokens"); c.set_ylabel("effective TFLOPS")
c.set_title("C — Same failure mode as MLA INT4:\na dequant in-core ceiling, just higher")
c.grid(alpha=0.25, which="both"); c.legend(fontsize=8, loc="lower right")

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, f"fused_moe_analysis.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
