"""Figure: fused_moe Triton-vs-CUDA speedup vs token count (the crux).

Left:  latency vs T for tuned-CUDA bf16 / fp8 W8A16 / mxfp4 W4A16 (EMU).
Right: speedup vs bf16 for CUDA fp8, CUDA mxfp4 (EMU), and the Triton W8A16/W8A8
       references. The tuned-CUDA quant paths CROSS BELOW 1.0 at ~600 tokens
       (weight-only dequant ceiling, compute-bound regime); the Triton paths stay
       above 1.0 only because the Triton bf16 baseline scaled super-linearly.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "results_cuda_moe.json")) as f:
    R = json.load(f)["rows"]
with open(os.path.join(_D, "..", "fused_moe", "results_fused_moe_extended.json")) as f:
    TR = json.load(f)["rows"]

T = np.array([r["T"] for r in R])
bf16 = np.array([r["bf16"] for r in R])
fp8 = np.array([r["fp8_w8a16"] for r in R])
mx = np.array([r["mxfp4_w4a16_EMU"] for r in R])

tT = np.array([r["T"] for r in TR])
t_w816 = np.array([r["bf16"] / r["w8a16"] for r in TR])
t_w88 = np.array([r["bf16"] / r["w8a8"] for r in TR])

C_BF, C_16, C_8, C_MX, C_TR = "#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#9E9E9E"
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9))

a.loglog(T, bf16, "o-", color=C_BF, lw=2, label="bf16 (vLLM fused_experts)")
a.loglog(T, fp8, "s-", color=C_16, lw=2, label="fp8 W8A16 (Marlin, native)")
a.loglog(T, mx, "^--", color=C_MX, lw=1.7, alpha=0.9, label="mxfp4 W4A16 (Marlin, EMULATED)")
a.set_xlabel("tokens")
a.set_ylabel("latency (µs, CUDA-graph)")
a.set_title("A — tuned-CUDA MoE latency  (Mixtral E=8,H=4096,I=14336,topk=2)")
a.grid(alpha=0.25, which="both")
a.legend(fontsize=8)

b.semilogx(T, bf16 / fp8, "s-", color=C_16, lw=2.2, label="CUDA fp8 W8A16")
b.semilogx(T, bf16 / mx, "^-", color=C_MX, lw=1.8, alpha=0.9, label="CUDA mxfp4 W4A16 (EMU)")
b.semilogx(tT, t_w816, "o:", color=C_TR, lw=1.6, alpha=0.95, label="Triton W8A16 (ref)")
b.semilogx(tT, t_w88, "x:", color="#616161", lw=1.4, alpha=0.85, label="Triton W8A8 (ref)")
b.axhline(1.0, color="k", lw=1, alpha=0.6)
b.axvspan(560, 700, color="orange", alpha=0.12)
b.text(170, 2.75, "memory-bound:\nquant wins (1.2–1.9×)", fontsize=8.5, color=C_16)
b.text(560, 0.52, "crossover\n~600 tok", fontsize=8.5, color="#C62828")
b.text(900, 1.95, "Triton 'high-T win'\n= bad bf16 baseline\n(artifact)", fontsize=8, color="#616161")
b.text(1050, 0.66, "tuned CUDA:\nquant LOSES\n(dequant ceiling)", fontsize=8.5, color=C_8)
b.set_ylim(0.45, 3.4)
b.set_xlabel("tokens")
b.set_ylabel("speedup vs bf16  (>1 = quant wins)")
b.set_title("B — CUDA quant crosses below 1.0 at ~600 tok;  Triton did not")
b.grid(alpha=0.25)
b.legend(fontsize=7.6, loc="upper center", ncol=2)

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, "figures", f"cuda_moe_triton_vs_cuda.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
