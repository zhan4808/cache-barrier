"""Figure: flashmla_sparse bf16 KV vs FP8 KV (DeepSeek V3.2 sparse MLA decode).

Left:  latency vs batch at C=4096 (genuinely sparse, topk=2048): bf16 KV
       (flash_mla_sparse_fwd) vs FP8 KV (flash_mla_with_kvcache).
Right: FP8/bf16-KV speedup vs batch, one line per context. FP8 KV pays off in the
       low-batch latency regime but WASHES OUT to parity (and below) at high batch
       -- the KV-cache analog of the MoE memory->compute transition. The large
       low-batch win is partly the two-kernel dispatch (vLLM uses the decode kernel
       for fp8, the prefill-style sparse kernel for bf16); the robust signal is the
       high-batch washout, where both kernels are efficient.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "results_flashmla_sparse.json")) as f:
    R = json.load(f)["rows"]

contexts = sorted({r["context"] for r in R})
batches = sorted({r["batch"] for r in R})
C_BF, C_F8 = "#1565C0", "#C62828"
ctx_colors = {512: "#90A4AE", 2048: "#1565C0", 4096: "#2E7D32", 8192: "#C62828"}

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9))

# Left: latency vs batch at C=4096
sel = sorted([r for r in R if r["context"] == 4096], key=lambda x: x["batch"])
Bs = [r["batch"] for r in sel]
a.plot(Bs, [r["bf16_kv_us"] for r in sel], "o-", color=C_BF, lw=2, label="bf16 KV (flash_mla_sparse_fwd)")
a.plot(Bs, [r["fp8_kv_us"] for r in sel], "s-", color=C_F8, lw=2, label="FP8 KV (flash_mla_with_kvcache)")
a.set_xscale("log", base=2); a.set_xticks(Bs); a.set_xticklabels(Bs)
a.set_xlabel("batch (decode requests)")
a.set_ylabel("latency (µs, CUDA-graph)")
a.set_title("A — sparse MLA decode latency  (C=4096, topk=2048, h_q=128)")
a.grid(alpha=0.25, which="both")
a.legend(fontsize=8)

# Right: speedup vs batch, per context
for C in contexts:
    sub = sorted([r for r in R if r["context"] == C], key=lambda x: x["batch"])
    a_b = [r["batch"] for r in sub]
    spd = [r["fp8_vs_bf16"] for r in sub]
    lab = f"C={C}" + (" (sparse)" if C > 2048 else " (dense)")
    b.plot(a_b, spd, "o-", color=ctx_colors[C], lw=1.9, label=lab)
b.set_xscale("log", base=2); b.set_xticks(batches); b.set_xticklabels(batches)
b.axhline(1.0, color="k", lw=1, alpha=0.6)
b.axvspan(20, 40, color="orange", alpha=0.12)
b.text(1.05, 2.6, "low batch: FP8 KV wins\n(latency regime;\n+ decode-kernel effect)", fontsize=8, color="#444")
b.text(13, 0.7, "high batch:\nwashout to parity", fontsize=8.5, color=C_F8)
b.set_xlabel("batch (decode requests)")
b.set_ylabel("FP8/bf16-KV speedup  (>1 = FP8 wins)")
b.set_title("B — FP8 KV washes out at high batch (throughput regime)")
b.grid(alpha=0.25)
b.legend(fontsize=8, loc="upper right")

plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, "figures", f"flashmla_bf16_vs_fp8.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
