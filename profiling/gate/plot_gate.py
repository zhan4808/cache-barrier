"""P1 gate figure: speedup vs W/C_eff, one line per token count, CARM overlay.

Reads results_capacity_gate.json (bench_capacity_gate.py). Writes
fig_capacity_gate.png/.pdf and gate_mape.json (regime- and T-separated MAPE
for both CARM memory-term forms, guardrail 7).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from bench_capacity_gate import predict_us, C_EFF, P  # noqa: E402

with open(os.path.join(_D, "results_capacity_gate.json")) as f:
    D = json.load(f)
RS = D["results"]
C_MB = P["effective_l2_capacity_mb"]
TS = sorted(set(r["tokens"] for r in RS))
WS = sorted(set(r["w_mb"] for r in RS))

# ── MAPE tables (both model forms) ──────────────────────────────────────────
mape = {}
for split in (False, True):
    key = "split_mem" if split else "lumped"
    mape[key] = {}
    for prec, wfrac in (("bf16", 2.0), ("w4a16", 0.5), ("w8a8", 1.0)):
        below, above, per_t = [], [], {}
        for r in RS:
            pred = predict_us(prec, r["tokens"], r["N"], split_mem=split)
            err = abs(pred - r[f"{prec}_us"]) / r[f"{prec}_us"]
            operand = r["w_mb"] * 1048576 * wfrac / 2.0
            (below if operand < C_EFF else above).append(err)
            per_t.setdefault(r["tokens"], []).append(err)
        mape[key][prec] = {
            "below_gate_mape_pct": round(100 * sum(below) / len(below), 1) if below else None,
            "above_gate_mape_pct": round(100 * sum(above) / len(above), 1) if above else None,
            "per_token_mape_pct": {t: round(100 * sum(v) / len(v), 1) for t, v in per_t.items()},
        }
with open(os.path.join(_D, "gate_mape.json"), "w") as f:
    json.dump({"note": "operand-aware gate: regime split at each precision's own "
                       "operand bytes vs C_eff. w4a16 packed operand never exceeds "
                       "C_eff in this sweep, hence above=null.",
               "mape": mape}, f, indent=1)

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
cmap = plt.get_cmap("viridis")
colors = {t: cmap(i / max(len(TS) - 1, 1)) for i, t in enumerate(TS)}
OVERLAY_TS = [1, 32, 512]

for ax, prec, label in ((axes[0], "w8a8", "W8A8 (INT8 MMA, no in-core dequant)"),
                        (axes[1], "w4a16", "W4A16 (Triton in-core dequant)")):
    for t in TS:
        rows = [r for r in RS if r["tokens"] == t]
        x = [r["w_mb"] / C_MB for r in rows]
        y = [r[f"sp_{prec}"] for r in rows]
        ax.plot(x, y, "o-", ms=3.5, lw=1.4, color=colors[t], label=f"T={t}")
    for t in OVERLAY_TS:
        rows = [r for r in RS if r["tokens"] == t]
        x = [r["w_mb"] / C_MB for r in rows]
        yp = [predict_us("bf16", t, r["N"], split_mem=True) /
              predict_us(prec, t, r["N"], split_mem=True) for r in rows]
        ax.plot(x, yp, "--", lw=1.0, color=colors[t], alpha=0.65)
    ax.axvline(1.0, color="crimson", lw=1.2, ls=":",
               label="W/C$_{eff}$=1 (36 MB measured)" if prec == "w8a8" else None)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 3.5])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "3.5"])
    ax.set_xlabel("bf16 weight working set / C$_{eff}$")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("speedup over tuned cuBLAS bf16")
axes[0].legend(fontsize=7.5, ncol=2, loc="upper left")
axes[1].text(0.03, 0.95,
             "dashed = CARM prediction (T=1, 32, 512)\n"
             "W4A16 far field capped at parity by\n"
             "r_dequant = 0.496 TB/s (in-core ceiling)",
             transform=axes[1].transAxes, fontsize=7.5, va="top",
             bbox=dict(fc="white", alpha=0.8, ec="0.7"))
fig.suptitle("The capacity gate: quantization pays only above the measured "
             f"effective L2 capacity ({D['gpu']}, clocks locked)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(_D, f"fig_capacity_gate.{ext}"), dpi=180)
print("saved fig_capacity_gate.png/.pdf and gate_mape.json")
