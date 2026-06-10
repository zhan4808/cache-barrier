"""
Task 1: Memory-to-Compute Transition — COLD-CACHE NCU view (interpretation corrected 2026-06)
=============================================================================================
Produces two side-by-side figures from the NCU sweep data:

  Figure A (left):  Achieved DRAM Bandwidth (GB/s) vs FP16 weight size
  Figure B (right): Achieved SM Compute Throughput (TFLOPS proxy) vs FP16 weight size

IMPORTANT — what this data can and cannot show:
  The underlying NCU sweep was collected with kernel replay and the default
  `--cache-control all`, which flushes GPU caches before every measured launch.
  Every point below is therefore a COLD-CACHE measurement:

  - FP16 DRAM% rising from ~35% (8 MB) to ~83% (128 MB) is smooth amortization
    of fixed launch/ramp overheads as kernel duration grows (steepest rise is
    BELOW 32 MB, not at the L2 boundary). It is NOT an L2-residency knee and
    must not be cited as one.
  - INT4 SM% rising from ~33% to ~79% is the same duration-amortization effect;
    INT4 is dequant-compute-bound at EVERY size (valid observation).
  - There is no within-kernel "memory-bound -> compute-bound transition" here:
    FP16 is memory-side at all sizes, INT4 compute-side at all sizes. What
    changes with size (visible only in warm-state data, not in this file's
    inputs) is FP16's serving TIER: L2 below ~36 MB effective capacity, HBM above.

  For actual residency evidence use profiling/validation/ (warm-loop NCU with
  --cache-control none, CUDA-graph timing, weight-rotation intervention).

Inputs:  ncu_results/l2_sweep/ncu_sweep_summary.json   (cold-cache)
Outputs: ../paper/figures/memory_compute_transition.png
         ../paper/figures/memory_compute_transition.pdf
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── H100 SXM5 hardware peaks ─────────────────────────────────────────────────
H100_HBM_BW_GBs   = 3350.0   # GB/s  (3.35 TB/s)
H100_COMPUTE_TFLOPS = 989.4  # FP16 dense tensor core peak
L2_CAPACITY_MB    = 50.0

# ── Load NCU sweep summary ────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
ncu_path   = os.path.join(script_dir, "ncu_results", "l2_sweep", "ncu_sweep_summary.json")

with open(ncu_path) as f:
    ncu_raw = json.load(f)

fp16_entries = sorted([r for r in ncu_raw if r["kernel"] == "fp16"], key=lambda r: r["weight_mb"])
int4_entries = sorted([r for r in ncu_raw if r["kernel"] == "int4"], key=lambda r: r["weight_mb"])

def extract(entries, key):
    return [r[key] for r in entries]

weight_mbs_fp16 = extract(fp16_entries, "weight_mb")
weight_mbs_int4 = extract(int4_entries, "weight_mb")

# ── Derived metrics ───────────────────────────────────────────────────────────
# Figure A: achieved DRAM bandwidth  (dram_pct × HBM_peak)
fp16_dram_bw = [r["dram_pct"] / 100.0 * H100_HBM_BW_GBs for r in fp16_entries]
int4_dram_bw = [r["dram_pct"] / 100.0 * H100_HBM_BW_GBs for r in int4_entries]

# Figure B: SM compute throughput proxy  (sm_pct × compute_peak)
fp16_tflops = [r["sm_pct"] / 100.0 * H100_COMPUTE_TFLOPS for r in fp16_entries]
int4_tflops = [r["sm_pct"] / 100.0 * H100_COMPUTE_TFLOPS for r in int4_entries]

# ── Plot ──────────────────────────────────────────────────────────────────────
FP16_COLOR = "#2196F3"   # blue
INT4_COLOR = "#F44336"   # red
L2_COLOR   = "#757575"   # gray

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "FP16 cuBLAS vs INT4 Triton across Weight Sizes — COLD-CACHE NCU counters\n"
    "H100 SXM5 · BS=1 · H=128 · kernel replay with --cache-control all (caches flushed; no residency information)",
    fontsize=12, y=1.02,
)

# ── Shared axis helpers ───────────────────────────────────────────────────────
def add_l2_vline(ax, y_text, label="H100 L2\n(50 MB)"):
    ax.axvline(x=L2_CAPACITY_MB, color=L2_COLOR, linestyle="--", linewidth=1.4, alpha=0.7, zorder=1)
    ax.text(L2_CAPACITY_MB + 1.5, y_text, label, fontsize=8, color=L2_COLOR, va="top")

def style_ax(ax):
    ax.set_xlabel("FP16 Weight Size (MB)", fontsize=11)
    ax.set_xlim(0, max(weight_mbs_fp16) * 1.08)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

# ── Figure A: DRAM Bandwidth ──────────────────────────────────────────────────
ax_a.plot(weight_mbs_fp16, fp16_dram_bw, "o-", color=FP16_COLOR, linewidth=2,
          markersize=7, label="FP16 cuBLAS", zorder=5)
ax_a.plot(weight_mbs_int4, int4_dram_bw, "^-", color=INT4_COLOR, linewidth=2,
          markersize=7, label="INT4 Triton", zorder=5)

# Peak HBM reference
ax_a.axhline(y=H100_HBM_BW_GBs, color="black", linestyle=":", linewidth=1.0,
             alpha=0.35, label=f"HBM peak ({H100_HBM_BW_GBs:.0f} GB/s)")

# Shaded L2-resident / HBM-bound regions
x_max = max(weight_mbs_fp16) * 1.08
ax_a.axvspan(0,               L2_CAPACITY_MB, alpha=0.06, color=FP16_COLOR, zorder=0)
ax_a.axvspan(L2_CAPACITY_MB,  x_max,          alpha=0.06, color=INT4_COLOR, zorder=0)
ax_a.text(25,  80, "weights fit nominal L2\n(cold-cache here: no residency)", fontsize=7,
          color=FP16_COLOR, ha="center", style="italic")
ax_a.text(89, 80, "weights exceed L2", fontsize=7, color=INT4_COLOR, ha="center", style="italic")

add_l2_vline(ax_a, y_text=H100_HBM_BW_GBs * 0.96)

# Secondary y-axis: % of HBM peak
ax_a2 = ax_a.twinx()
ax_a2.set_ylim(0 / H100_HBM_BW_GBs * 100,
               ax_a.get_ylim()[1] / H100_HBM_BW_GBs * 100)
ax_a2.set_ylabel("% of HBM Peak", fontsize=10, color=L2_COLOR)
ax_a2.tick_params(axis="y", colors=L2_COLOR, labelsize=9)

ax_a.set_ylabel("Achieved DRAM Bandwidth (GB/s)", fontsize=11)
ax_a.set_title("Figure A — Memory Throughput vs Weight Size", fontsize=11, pad=8)
ax_a.set_ylim(bottom=0)
ax_a.legend(fontsize=9, loc="upper left")
style_ax(ax_a)

# Annotate key percentages
for r in fp16_entries:
    if r["weight_mb"] in (8.0, 128.0):
        bw = r["dram_pct"] / 100.0 * H100_HBM_BW_GBs
        ax_a.annotate(f'{r["dram_pct"]:.0f}%',
                      xy=(r["weight_mb"], bw),
                      xytext=(0, 8), textcoords="offset points",
                      fontsize=7.5, color=FP16_COLOR, ha="center", fontweight="bold")
for r in int4_entries:
    if r["weight_mb"] in (8.0, 128.0):
        bw = r["dram_pct"] / 100.0 * H100_HBM_BW_GBs
        ax_a.annotate(f'{r["dram_pct"]:.0f}%',
                      xy=(r["weight_mb"], bw),
                      xytext=(0, -14), textcoords="offset points",
                      fontsize=7.5, color=INT4_COLOR, ha="center", fontweight="bold")

# ── Figure B: Compute Throughput (SM proxy) ───────────────────────────────────
ax_b.plot(weight_mbs_fp16, fp16_tflops, "o-", color=FP16_COLOR, linewidth=2,
          markersize=7, label="FP16 cuBLAS", zorder=5)
ax_b.plot(weight_mbs_int4, int4_tflops, "^-", color=INT4_COLOR, linewidth=2,
          markersize=7, label="INT4 Triton", zorder=5)

# Compute ceiling reference
ax_b.axhline(y=H100_COMPUTE_TFLOPS, color="black", linestyle=":", linewidth=1.0,
             alpha=0.35, label=f"Compute peak ({H100_COMPUTE_TFLOPS:.0f} TFLOPS)")

add_l2_vline(ax_b, y_text=H100_COMPUTE_TFLOPS * 0.96)
ax_b.axvspan(0,               L2_CAPACITY_MB, alpha=0.06, color=FP16_COLOR, zorder=0)
ax_b.axvspan(L2_CAPACITY_MB,  x_max,          alpha=0.06, color=INT4_COLOR, zorder=0)

# Secondary y-axis: % of compute peak
ax_b2 = ax_b.twinx()
ax_b2.set_ylim(0,
               ax_b.get_ylim()[1] / H100_COMPUTE_TFLOPS * 100
               if ax_b.get_ylim()[1] > 0 else 100)
ax_b2.set_ylabel("% of Compute Peak", fontsize=10, color=L2_COLOR)
ax_b2.tick_params(axis="y", colors=L2_COLOR, labelsize=9)

ax_b.set_ylabel("SM Compute Throughput (sm_util × 989 TFLOPS peak)", fontsize=10)
ax_b.set_title("Figure B — Compute Throughput vs Weight Size", fontsize=11, pad=8)
ax_b.set_ylim(bottom=0)
ax_b.legend(fontsize=9, loc="upper left")
style_ax(ax_b)

# Annotate key SM util percentages at endpoints
for r in fp16_entries:
    if r["weight_mb"] in (8.0, 128.0):
        t = r["sm_pct"] / 100.0 * H100_COMPUTE_TFLOPS
        ax_b.annotate(f'{r["sm_pct"]:.0f}%',
                      xy=(r["weight_mb"], t),
                      xytext=(0, 8), textcoords="offset points",
                      fontsize=7.5, color=FP16_COLOR, ha="center", fontweight="bold")
for r in int4_entries:
    if r["weight_mb"] in (8.0, 128.0):
        t = r["sm_pct"] / 100.0 * H100_COMPUTE_TFLOPS
        ax_b.annotate(f'{r["sm_pct"]:.0f}%',
                      xy=(r["weight_mb"], t),
                      xytext=(0, -14), textcoords="offset points",
                      fontsize=7.5, color=INT4_COLOR, ha="center", fontweight="bold")

# Note: no regime crossover exists in this cold-cache data. FP16 is memory-side
# and INT4 dequant-compute-side at every size; both curves rise smoothly with
# kernel duration. The warm-state residency cliff (32-40 MB) is only visible in
# profiling/validation/ data.
ax_b.annotate(
    "cold-cache data:\nno regime crossover;\nboth curves = duration\namortization",
    xy=(48.0, fp16_tflops[list(weight_mbs_fp16).index(48.0)]),
    xytext=(68, fp16_tflops[0] + 20),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    fontsize=8, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(script_dir, "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"memory_compute_transition.{ext}")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()

# ── Console summary ───────────────────────────────────────────────────────────
print("\nNCU Sweep Summary")
print(f"{'Weight MB':>10} {'FP16 DRAM%':>11} {'FP16 SM%':>9} {'INT4 DRAM%':>11} {'INT4 SM%':>9}")
print("-" * 55)
for fp16, int4 in zip(fp16_entries, int4_entries):
    print(f"{fp16['weight_mb']:>10.0f} {fp16['dram_pct']:>11.1f} {fp16['sm_pct']:>9.1f}"
          f" {int4['dram_pct']:>11.1f} {int4['sm_pct']:>9.1f}")
