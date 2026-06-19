"""Figure: fp8/bf16 MoE speedup vs token count, per model shape -- the crossover
moves with shape. Coarse-grained Mixtral crosses below 1.0; the fine-grained
target models (DeepSeek-V4-Flash, Qwen3.6-35B) stay above 1.0 across the range."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "results_task3_target_shapes.json")) as f:
    R = json.load(f)["shapes"]

colors = {"Mixtral-8x7B": "#C62828", "DeepSeek-V4-Flash": "#1565C0", "Qwen3.6-35B-A3B": "#2E7D32"}
marks = {"Mixtral-8x7B": "o-", "DeepSeek-V4-Flash": "s-", "Qwen3.6-35B-A3B": "^-"}
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for name, s in R.items():
    bf = {int(k): v for k, v in s["bf16_us"].items()}
    fp = {int(k): v for k, v in s["fp8_us"].items()}
    Ts = sorted(bf)
    spd = [bf[T] / fp[T] for T in Ts]
    E, H, I, tk = s["cfg"]["E"], s["cfg"]["H"], s["cfg"]["I"], s["cfg"]["topk"]
    ax.semilogx(Ts, spd, marks[name], color=colors[name], lw=2,
                label=f"{name}  (E={E},I={I},k={tk})")
ax.axhline(1.0, color="k", lw=1, alpha=0.6)
ax.axhspan(0.0, 1.0, color="grey", alpha=0.07)
ax.text(20, 0.83, "quant loses", fontsize=9, color="#555")
ax.text(20, 2.05, "quant wins", fontsize=9, color="#555")
ax.annotate("Mixtral crosses\n~1024 tok (stock);\n~300 vs tuned bf16",
            xy=(1024, 0.78), xytext=(330, 0.55), fontsize=8, color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1))
ax.text(150, 1.75, "fine-grained targets stay\nmemory-bound → fp8 wins\nacross the range",
        fontsize=8.5, color="#1565C0")
ax.set_xlabel("tokens per MoE-layer call")
ax.set_ylabel("fp8 W8A16 speedup vs bf16  (>1 = quant wins)")
ax.set_title("Quant-vs-dense crossover moves with MoE shape (H100, fp8 W8A16)")
ax.set_ylim(0.4, 2.2)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8.5, loc="upper right")
plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(_D, "figures", f"task3_crossover_by_shape.{ext}")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    print("Saved", p)
