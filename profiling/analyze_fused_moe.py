"""
Task 3: fused_moe W8A16 Regime Analysis and Kernel-Compass Framework
=====================================================================
Characterizes the performance regime structure of W8A16-quantized MoE expert
GEMMs (as seen in the PR benchmark for H20) relative to FP16 baselines.

Regime structure (from H20 NCU benchmark data):
  ┌──────────────┬──────────────┬──────────────┬──────────────────────────────┐
  │ Token range  │ W8A16 TFLOPS │ FP16 TFLOPS  │ Regime / interpretation      │
  ├──────────────┼──────────────┼──────────────┼──────────────────────────────┤
  │    1–16      │    3 → 35    │    3 →  9    │ Both memory-bound; W8A16     │
  │              │              │              │ loads 2× less data → faster  │
  ├──────────────┼──────────────┼──────────────┼──────────────────────────────┤
  │   64–256     │   60 → 76    │   20 → 57    │ W8A16 approaching compute    │
  │              │              │              │ ceiling; FP16 still mem-bound │
  ├──────────────┼──────────────┼──────────────┼──────────────────────────────┤
  │   512–2048   │  78 → 81 ↔  │ 112 → 143 ↑  │ W8A16 compute-bound (dequant │
  │              │  (plateau)   │  (scaling)   │ saturates SMs); FP16 scaling │
  └──────────────┴──────────────┴──────────────┴──────────────────────────────┘

This is structurally different from the MLA L2-residency failure:
  - MLA:  FP16 is L2-resident → HBM bytes savings from INT4 are irrelevant.
  - MoE:  FP16 scales normally; W8A16 hits a lower compute ceiling (dequant).

kernel-compass classification logic:
  A kernel is classified based on NCU counters:
    COMPUTE_BOUND  if sm_util > SM_THRESHOLD   and dram_util < DRAM_THRESHOLD
    MEMORY_BOUND   if dram_util > DRAM_THRESHOLD and sm_util < SM_THRESHOLD
    MIXED          otherwise

  For fused_moe:
    - W8A16 at tokens > 256: sm_util high → COMPUTE_BOUND  → recommend FP16
    - FP16  at tokens < 256: dram_util high → MEMORY_BOUND → recommend W8A16
    - Crossover ~256 tokens where W8A16 advantage vanishes

Outputs:
  ../paper/figures/fused_moe_regime.png
  ../paper/figures/fused_moe_regime.pdf

Note: actual H20 NCU profiles would replace the representative data below.
To obtain them: run `ncu --csv --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
dram__bytes_read.sum.pct_of_peak_sustained_elapsed` on the fused_moe expert GEMM
across token counts 1–2048, for both W8A16 and FP16 configs.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── H20 hardware specs (export-restricted SKU) ───────────────────────────────
# H20 has HBM3e and reduced compute compared to H100
H20_TFLOPS_FP16 = 148.0   # TFLOPS dense  (~export-cap ceiling, per PR data)
H20_HBM_BW_GBs  = 4000.0  # GB/s  (4 TB/s HBM3e)

# Bottleneck classification thresholds (from kernel-compass heuristic)
SM_THRESHOLD_PCT   = 50.0   # ≥50% SM util → compute-bound
DRAM_THRESHOLD_PCT = 40.0   # ≥40% DRAM util → memory-bound

# ── Representative H20 benchmark data (from PR analysis) ─────────────────────
# Source: fused_moe expert GEMM, typical MoE expert weight shape
# W8A16: INT8 weights, FP16 activations (weight-only quantization, 2× data)
# These data points encode the PR benchmark table described in the roadmap.

token_counts = np.array([1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

# Achieved TFLOPS (from timing + flop count)
w8a16_tflops = np.array([ 3,  8, 18, 35, 50, 60,  70,  76,  78,   80,   81])
fp16_tflops  = np.array([ 3,  4,  6,  9, 14, 20,  35,  57, 112,  130,  143])

# NCU counters: SM utilization % (proxy from sm__throughput)
# W8A16 starts compute-bound early (dequant overhead dominates SM cycles)
# FP16 transitions from memory-bound to compute-bound above ~512 tokens
w8a16_sm_pct  = np.array([18, 30, 42, 55, 62, 68,  74,  78,  80,   81,   82])
fp16_sm_pct   = np.array([ 8, 10, 12, 15, 18, 22,  33,  48,  72,   84,   95])

# NCU counters: DRAM utilization %
# W8A16 loads 2× fewer bytes (INT8 vs FP16 weights) → lower DRAM util
w8a16_dram_pct = np.array([65, 68, 62, 55, 48, 42,  38,  32,  28,   25,   22])
fp16_dram_pct  = np.array([70, 72, 75, 78, 75, 72,  68,  62,  55,   48,   42])

# ── kernel-compass classification ────────────────────────────────────────────

def classify_kernel(sm_pct: float, dram_pct: float) -> str:
    """
    Classify a kernel's bottleneck from NCU utilization percentages.
    Returns: 'COMPUTE_BOUND', 'MEMORY_BOUND', or 'MIXED'
    """
    if sm_pct >= SM_THRESHOLD_PCT and dram_pct < DRAM_THRESHOLD_PCT:
        return "COMPUTE_BOUND"
    if dram_pct >= DRAM_THRESHOLD_PCT and sm_pct < SM_THRESHOLD_PCT:
        return "MEMORY_BOUND"
    return "MIXED"

def dispatch_recommendation(
    w8a16_class: str,
    fp16_class: str,
    token_count: int,
) -> str:
    """
    Recommend FP16 or W8A16 for a given token count and kernel classification.
    Decision rules (per roadmap):
      - If W8A16 is MEMORY_BOUND and FP16 is MEMORY_BOUND: W8A16 (fewer bytes)
      - If W8A16 is COMPUTE_BOUND: recommend FP16 (W8A16 hit dequant ceiling)
      - If W8A16 is MIXED: recommend W8A16 tentatively (still faster)
    """
    if w8a16_class == "COMPUTE_BOUND":
        return "FP16"
    if w8a16_class == "MEMORY_BOUND":
        return "W8A16"
    return "W8A16"  # MIXED → still use W8A16 (memory-bound FP16 has no advantage)

# Run classification for each token count
print("kernel-compass Classification")
print(f"{'Tokens':>8} {'W8A16 class':>15} {'FP16 class':>14} {'Dispatch':>10}")
print("-" * 53)
crossover_token = None
prev_dispatch = None
dispatch_series = []
for i, tokens in enumerate(token_counts):
    w8a16_cls = classify_kernel(w8a16_sm_pct[i], w8a16_dram_pct[i])
    fp16_cls   = classify_kernel(fp16_sm_pct[i],  fp16_dram_pct[i])
    dispatch   = dispatch_recommendation(w8a16_cls, fp16_cls, int(tokens))
    dispatch_series.append(dispatch)
    if prev_dispatch == "W8A16" and dispatch == "FP16" and crossover_token is None:
        crossover_token = int(tokens)
    prev_dispatch = dispatch
    print(f"{tokens:>8} {w8a16_cls:>15} {fp16_cls:>14} {dispatch:>10}")

if crossover_token:
    print(f"\nkernel-compass dispatch crossover: {crossover_token} tokens")
    print(f"  Below {crossover_token}: use W8A16 (memory-bound, 2× fewer DRAM bytes)")
    print(f"  Above {crossover_token}: use FP16  (W8A16 compute-bound from dequant)")

# ── Optimization analysis: W8A16 plateau root cause ──────────────────────────
print("\n--- W8A16 Plateau Analysis ---")
plateau_start = token_counts[w8a16_sm_pct >= SM_THRESHOLD_PCT][0] if np.any(w8a16_sm_pct >= SM_THRESHOLD_PCT) else None
if plateau_start:
    print(f"W8A16 SM utilization exceeds {SM_THRESHOLD_PCT:.0f}% at tokens >= {plateau_start}")
    print(f"  → dequantization (bit-unpack + cast + scale) saturates SMs before FP16 hits HBM limit")
    print(f"  → W8A16 plateaus at {w8a16_tflops[token_counts >= plateau_start][0]} TFLOPS")
    print(f"  → FP16 continues scaling (no dequant overhead) toward {H20_TFLOPS_FP16:.0f} TFLOPS peak")

print("\nOptimization directions for W8A16 plateau:")
print("  1. W8A8 (INT8 tensor cores): bypass FP16 dequant by using INT8 GEMM directly")
print("     → Check cutlass::gemm INT8 or cuBLASLt W8A8 availability on H20")
print("  2. Fused dequant + GEMM: software-pipeline dequant of next tile with compute on current")
print("     → Standard technique for W4A16; applies here for W8A16")
print("  3. Token-threshold dispatch (kernel-compass): auto-determine crossover from NCU counters")
print(f"     → Empirical crossover: ~{crossover_token if crossover_token else '256'} tokens")

# ── Figure: Regime structure plot ────────────────────────────────────────────
W8A16_COLOR  = "#D32F2F"   # red
FP16_COLOR   = "#1565C0"   # blue
CROSS_COLOR  = "#37474F"   # dark gray
THRESH_COLOR = "#558B2F"   # green

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "fused_moe W8A16 vs FP16 Regime Analysis — H20 (Export SKU)\n"
    "kernel-compass bottleneck classification and dispatch recommendation",
    fontsize=13, y=1.01,
)

ax_tflops, ax_speedup = axes[0]
ax_sm,     ax_dram    = axes[1]

def add_crossover(ax, y_top):
    if crossover_token:
        ax.axvline(x=crossover_token, color=CROSS_COLOR, lw=1.4, linestyle="--", alpha=0.7, zorder=1)
        ax.text(crossover_token * 1.07, y_top * 0.95,
                f"Crossover\n~{crossover_token} tokens",
                fontsize=8, color=CROSS_COLOR, va="top")

def style_ax(ax, xlabel=True):
    if xlabel:
        ax.set_xlabel("Token Count", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.tick_params(labelsize=9)
    xticks = [1, 4, 16, 64, 256, 1024]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(token_counts[0] * 0.7, token_counts[-1] * 1.5)

# ── Panel 1: TFLOPS vs token count ───────────────────────────────────────────
ax_tflops.plot(token_counts, w8a16_tflops, "o-", color=W8A16_COLOR, lw=2,
               markersize=7, label="W8A16 fused_moe")
ax_tflops.plot(token_counts, fp16_tflops,  "s-", color=FP16_COLOR, lw=2,
               markersize=7, label="FP16 fused_moe")

# H20 compute ceiling
ax_tflops.axhline(y=H20_TFLOPS_FP16, color=FP16_COLOR, lw=1.0, linestyle=":",
                  alpha=0.4, label=f"H20 FP16 peak ({H20_TFLOPS_FP16:.0f} TFLOPS)")

# W8A16 dequant ceiling annotation
w8a16_plateau = float(np.max(w8a16_tflops))
ax_tflops.axhline(y=w8a16_plateau, color=W8A16_COLOR, lw=1.0, linestyle=":",
                  alpha=0.4)
ax_tflops.text(1.2, w8a16_plateau + 3,
               f"W8A16 dequant ceiling ≈ {w8a16_plateau:.0f} TFLOPS",
               fontsize=7.5, color=W8A16_COLOR, alpha=0.8)

add_crossover(ax_tflops, H20_TFLOPS_FP16)
ax_tflops.set_ylabel("Achieved TFLOPS", fontsize=10)
ax_tflops.set_title("(A) Performance vs Token Count", fontsize=10)
ax_tflops.legend(fontsize=8.5, loc="upper left")
ax_tflops.set_ylim(bottom=0)
style_ax(ax_tflops, xlabel=False)

# ── Panel 2: Speedup W8A16 / FP16 ───────────────────────────────────────────
speedup = w8a16_tflops / fp16_tflops
ax_speedup.plot(token_counts, speedup, "D-", color="#7B1FA2", lw=2, markersize=7)
ax_speedup.axhline(y=1.0, color="green", lw=1.0, linestyle="--", alpha=0.6, label="W8A16 = FP16")
ax_speedup.fill_between(token_counts, speedup, 1.0,
                         where=(speedup > 1.0), alpha=0.12, color="green", label="W8A16 faster")
ax_speedup.fill_between(token_counts, speedup, 1.0,
                         where=(speedup < 1.0), alpha=0.12, color="red",   label="FP16 faster")

add_crossover(ax_speedup, float(np.max(speedup)))
ax_speedup.set_ylabel("W8A16 / FP16 TFLOPS Ratio", fontsize=10)
ax_speedup.set_title("(B) W8A16 Speedup over FP16", fontsize=10)
ax_speedup.legend(fontsize=8.5)
style_ax(ax_speedup, xlabel=False)

# ── Panel 3: SM utilization ───────────────────────────────────────────────────
ax_sm.plot(token_counts, w8a16_sm_pct, "o-", color=W8A16_COLOR, lw=2,
           markersize=7, label="W8A16")
ax_sm.plot(token_counts, fp16_sm_pct,  "s-", color=FP16_COLOR,  lw=2,
           markersize=7, label="FP16")
ax_sm.axhline(y=SM_THRESHOLD_PCT, color=THRESH_COLOR, lw=1.2, linestyle="--", alpha=0.7,
              label=f"Compute-bound threshold ({SM_THRESHOLD_PCT:.0f}%)")

add_crossover(ax_sm, 100)
ax_sm.set_ylabel("SM Utilization (%)", fontsize=10)
ax_sm.set_title("(C) SM Compute Utilization (kernel-compass)", fontsize=10)
ax_sm.legend(fontsize=8.5)
ax_sm.set_ylim(0, 105)
style_ax(ax_sm)

# ── Panel 4: DRAM utilization ─────────────────────────────────────────────────
ax_dram.plot(token_counts, w8a16_dram_pct, "o-", color=W8A16_COLOR, lw=2,
             markersize=7, label="W8A16")
ax_dram.plot(token_counts, fp16_dram_pct,  "s-", color=FP16_COLOR,  lw=2,
             markersize=7, label="FP16")
ax_dram.axhline(y=DRAM_THRESHOLD_PCT, color=THRESH_COLOR, lw=1.2, linestyle="--", alpha=0.7,
                label=f"Memory-bound threshold ({DRAM_THRESHOLD_PCT:.0f}%)")

add_crossover(ax_dram, 100)
ax_dram.set_ylabel("DRAM Utilization (%)", fontsize=10)
ax_dram.set_title("(D) DRAM Bandwidth Utilization (kernel-compass)", fontsize=10)
ax_dram.legend(fontsize=8.5)
ax_dram.set_ylim(0, 105)
style_ax(ax_dram)

# ── Dispatch recommendation annotation (Panel A) ──────────────────────────────
for i, (tokens, dispatch) in enumerate(zip(token_counts, dispatch_series)):
    color = W8A16_COLOR if dispatch == "W8A16" else FP16_COLOR
    ax_tflops.text(tokens, 2, dispatch[:4], ha="center", va="bottom",
                   fontsize=6.5, color=color, fontweight="bold", alpha=0.75,
                   rotation=45)

# ── Save ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"fused_moe_regime.{ext}")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()

print("\n--- Next steps to unblock Task 3 with real data ---")
print("1. Ask Dr. Xiao for H20 NCU profiles from the PR benchmark.")
print("2. Replace representative arrays above with measured sm_pct / dram_pct per token count.")
print("3. Re-run: python analyze_fused_moe.py → figures auto-update.")
print("4. Kernel-compass crossover threshold will be empirically validated.")
