"""Figure: extended fused_moe sweep (T=16-2048), fixed W8A16 vs bf16 vs W8A8."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "results_fused_moe_extended.json")) as f:
    R = json.load(f)["rows"]

T = np.array([r["T"] for r in R])
bf16 = np.array([r["bf16"] for r in R])
w8a16 = np.array([r["w8a16"] for r in R])
w8a8 = np.array([r["w8a8"] for r in R])

C_BF, C_16, C_8 = "#1565C0", "#2E7D32", "#C62828"
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))

a.loglog(T, bf16, "o-", color=C_BF, lw=2, label="bf16")
a.loglog(T, w8a16, "o-", color=C_16, lw=2, label="W8A16 (fixed, in-kernel dequant)")
a.loglog(T, w8a8, "^--", color=C_8, lw=1.5, alpha=0.85, label="W8A8 (use_int8_w8a8)")
a.set_xlabel("tokens")
a.set_ylabel("latency (µs, CUDA-graph)")
a.set_title("A — fused_moe latency, Mixtral shape (E=8, H=4096, I=14336, topk=2)")
a.grid(alpha=0.25, which="both")
a.legend(fontsize=8)

b.semilogx(T, bf16 / w8a16, "o-", color=C_16, lw=2, label="W8A16 fixed")
b.semilogx(T, bf16 / w8a8, "^--", color=C_8, lw=1.5, alpha=0.85, label="W8A8")
b.axhline(1.0, color="k", lw=1, alpha=0.5)
b.text(28, 2.55, "weight-byte-bound:\nhalved bytes win 1.7×", fontsize=8, color=C_16)
b.text(620, 1.35, "bf16 leaves L2-friendly\ntiling regime; W8A16\nsustains 2.7–3.0×", fontsize=8, color=C_16)
b.set_ylim(0.8, 3.3)
b.set_xlabel("tokens")
b.set_ylabel("speedup vs bf16")
b.set_title("B — Fixed W8A16 wins at every measured T (min 1.03× @ T=256)")
b.grid(alpha=0.25)
b.legend(fontsize=8)

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, f"fused_moe_extended.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
