"""Figure: multi-layer MLA stacking vs CARM prediction."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "..", "..", "kernel-compass"))
from profiling.carm import predict_fp16_recon_us  # noqa: E402

with open(os.path.join(_D, "results_mla_l2_stack.json")) as f:
    R = json.load(f)

rows = R["rows"]
mb = [r["weight_mb"] for r in rows]
H, K, N = 128, 128, 512

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mb, [r["fp16_us"] for r in rows], "o-", color="#1565C0", lw=2, label="FP16 measured")
ax.plot(mb, [r["w8a8_us"] for r in rows], "o-", color="#2E7D32", lw=2, label="W8A8 measured")
ax.plot(mb, [r["carm_fp16_pred_us"] for r in rows], "s--", color="#1565C0", alpha=0.7, label="CARM FP16 pred")
ax.axvline(36, color="orange", ls="--", lw=1.2, label="C_eff ≈ 36 MB")
ax.axvspan(0, 36, color="orange", alpha=0.08)
ax.set_xlabel("Stacked reconstruction weight size (MB)")
ax.set_ylabel("latency per decode step (µs, CUDA-graph)")
ax.set_title("Multi-layer MLA reconstruction stacking (bs=1, 16 MB/layer)")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, f"mla_l2_stack.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
