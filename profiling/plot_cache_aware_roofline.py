"""
Task 2: Cache-Aware Roofline for MLA Reconstruction
====================================================
Extends the standard single-ceiling roofline to three memory hierarchy levels:

  Level          Bandwidth    Crossover AI (FLOP/byte)
  ─────────────  ──────────   ────────────────────────
  DRAM (HBM)     3.35 TB/s    989 / 3.35  ≈ 295
  L2 cache       ~12 TB/s     989 / 12    ≈  82
  L1/shared mem  ~80 TB/s     989 / 80    ≈  12

The MLA reconstruction BMMs (DeepSeek-V3) span batch sizes 1–1024.  Their
arithmetic intensity (AI) ranges from ~1 FLOP/byte (BS=1) to ~93 FLOP/byte
(BS=1024) — ALL below the DRAM crossover at 295, meaning every operating
point is bandwidth-limited by some memory level.

Key insight: for small batch sizes (BS≤64), the 16 MB weight matrix fits
inside H100's 50 MB L2 cache, so the effective bandwidth bottleneck is L2
(~12 TB/s), NOT HBM (3.35 TB/s).  Standard roofline predicts INT4 should
give ~3.9× speedup (moving AI right along the HBM slope).  The cache-aware
roofline shows FP16 is already near the L2 ceiling — there is no HBM headroom
to reclaim.

Inputs:  results_mla_reconstruction_v3.csv
         ncu_results/l2_sweep/ncu_sweep_summary.json  (for INT4 vs FP16 pts)
         results_l2_barrier.json
Outputs: ../paper/figures/cache_aware_roofline.png
         ../paper/figures/cache_aware_roofline.pdf
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── H100 SXM5 hardware specs ──────────────────────────────────────────────────
H100_DRAM_BW   = 3.35     # TB/s
H100_L2_BW     = 12.0     # TB/s  (effective aggregate)
H100_L1_BW     = 80.0     # TB/s  (L1 + shared memory)
H100_TFLOPS    = 989.4    # FP16 dense tensor core peak
L2_CAPACITY_MB = 50.0

# Crossover arithmetic intensities (FLOP/byte)
AI_L1_CROSS   = H100_TFLOPS / H100_L1_BW       # ~12.4
AI_L2_CROSS   = H100_TFLOPS / H100_L2_BW       # ~82.4
AI_DRAM_CROSS = H100_TFLOPS / H100_DRAM_BW     # ~295

# ── DeepSeek-V3 BMM geometry ──────────────────────────────────────────────────
# BMM1: (H=128, BS, K=128) × (H=128, K=128, N=512)  [w_kc, Q-absorption]
# BMM2: (H=128, BS, K=512) × (H=128, K=512, N=128)  [w_vc, V-reconstruction]
# Both BMMs have identical flop counts and byte counts (K+N=640 in both cases)

H, K_BMM1, N_BMM1 = 128, 128, 512
WEIGHT_MB = H * K_BMM1 * N_BMM1 * 2 / (1024 ** 2)  # 16 MB FP16

def compute_ai(batch_size: int) -> float:
    """Arithmetic intensity (FLOP/byte) for FP16 BMM1, including all tensors."""
    flops        = H * 2 * batch_size * K_BMM1 * N_BMM1
    weight_bytes = H * K_BMM1 * N_BMM1 * 2
    act_bytes    = H * batch_size * K_BMM1 * 2
    out_bytes    = H * batch_size * N_BMM1 * 2
    return flops / (weight_bytes + act_bytes + out_bytes)

def compute_ai_int4(batch_size: int) -> float:
    """Arithmetic intensity for INT4 (W4A16) BMM1."""
    flops            = H * 2 * batch_size * K_BMM1 * N_BMM1
    int4_weight_bytes = H * K_BMM1 * N_BMM1 // 2   # packed 4-bit
    scale_bytes       = H * N_BMM1 * 2               # per-column FP16 scales
    act_bytes         = H * batch_size * K_BMM1 * 2
    out_bytes         = H * batch_size * N_BMM1 * 2
    return flops / (int4_weight_bytes + scale_bytes + act_bytes + out_bytes)

# ── Load MLA reconstruction CSV ───────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "results_mla_reconstruction_v3.csv")

batch_sizes, fp16_measured_tflops, fp16_ai = [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        bs = int(row["batch_size"])
        tflops_val = row["bmm1_tflops"]
        if tflops_val and tflops_val != "N/A":
            batch_sizes.append(bs)
            fp16_measured_tflops.append(float(tflops_val))
            fp16_ai.append(compute_ai(bs))

# ── Load barrier JSON for INT4 timing at d_lora=512 ──────────────────────────
barrier_path = os.path.join(script_dir, "results_l2_barrier.json")
with open(barrier_path) as f:
    barrier = json.load(f)

# BS=1 operating points across weight sizes (to show DRAM-transition on roofline)
bs1_barrier = sorted([r for r in barrier["results"] if r["batch_size"] == 1],
                     key=lambda r: r["weight_mb"])

int4_ai_pts, int4_tflops_pts, int4_labels = [], [], []
fp16_hbm_ai_pts, fp16_hbm_tflops_pts      = [], []
FLOPS_FIXED = H * 2 * 1 * K_BMM1 * N_BMM1  # fixed at BS=1

for r in bs1_barrier:
    ai_fp16  = compute_ai(1)                          # same AI for all (BS=1)
    ai_int4  = compute_ai_int4(1)
    tflops_fp16 = FLOPS_FIXED / (r["fp16_ms"] * 1e-3) / 1e12
    tflops_int4 = FLOPS_FIXED / (r["int4_ms"] * 1e-3) / 1e12
    # Only show HBM-bound FP16 points (weight > L2)
    if not r["fits_l2"]:
        fp16_hbm_ai_pts.append(ai_fp16)
        fp16_hbm_tflops_pts.append(tflops_fp16)
    if r["weight_mb"] == 16.0:   # MLA config point for INT4
        int4_ai_pts.append(ai_int4)
        int4_tflops_pts.append(tflops_int4)
        int4_labels.append(f"INT4 (16 MB,\nBS=1)")

# ── Build roofline curves ─────────────────────────────────────────────────────
ai_range = np.logspace(-1.5, 3.5, 1000)

def roofline(ai, bw_TBs, peak_tflops):
    """Roofline ceiling: min(AI × BW, compute_peak)."""
    return np.minimum(ai * bw_TBs, peak_tflops)

perf_dram    = roofline(ai_range, H100_DRAM_BW, H100_TFLOPS)
perf_l2      = roofline(ai_range, H100_L2_BW,   H100_TFLOPS)
perf_l1      = roofline(ai_range, H100_L1_BW,   H100_TFLOPS)
perf_compute = np.full_like(ai_range, H100_TFLOPS)

# ── Plot ──────────────────────────────────────────────────────────────────────
FP16_L2_COLOR  = "#1565C0"   # dark blue  – L2-resident FP16
FP16_HBM_COLOR = "#42A5F5"   # light blue – HBM-bound FP16
INT4_COLOR     = "#C62828"   # dark red   – INT4 (compute-bound)
ROOF_ALPHA     = 0.85

fig, ax = plt.subplots(figsize=(10, 6.5))

# Roofline ceilings
ax.loglog(ai_range, perf_compute, "k-",  lw=1.5, alpha=0.4, zorder=1)
ax.loglog(ai_range, perf_l2,      "b--", lw=1.8, alpha=ROOF_ALPHA, zorder=2,
          label=f"L2 ceiling  (12 TB/s,  cross at AI≈{AI_L2_CROSS:.0f})")
ax.loglog(ai_range, perf_dram,    "r--", lw=1.8, alpha=ROOF_ALPHA, zorder=2,
          label=f"DRAM ceiling (3.35 TB/s, cross at AI≈{AI_DRAM_CROSS:.0f})")
ax.loglog(ai_range, perf_l1,      color="gray", lw=1.2, linestyle=":",
          alpha=0.6, zorder=2,
          label=f"L1/smem ceiling (80 TB/s, cross at AI≈{AI_L1_CROSS:.0f})")

# Compute ceiling label
ax.text(ai_range[-1] * 0.95, H100_TFLOPS * 1.08,
        f"Compute ceiling ({H100_TFLOPS:.0f} TFLOPS FP16)",
        ha="right", fontsize=8, color="black", alpha=0.55)

# Vertical crossover markers
for ai_x, label, color in [
    (AI_L1_CROSS,   f"L1 cross\nAI={AI_L1_CROSS:.0f}",   "gray"),
    (AI_L2_CROSS,   f"L2 cross\nAI={AI_L2_CROSS:.0f}",   "blue"),
    (AI_DRAM_CROSS, f"DRAM cross\nAI={AI_DRAM_CROSS:.0f}", "red"),
]:
    ax.axvline(x=ai_x, color=color, lw=0.8, linestyle=":", alpha=0.45, zorder=1)
    ax.text(ai_x * 1.05, 0.012, label, fontsize=7, color=color, alpha=0.7, va="bottom")

# ── FP16 operating points (MLA BS sweep, measured TFLOPS) ────────────────────
scatter_fp16 = ax.scatter(fp16_ai, fp16_measured_tflops,
                           s=90, color=FP16_L2_COLOR, marker="o",
                           zorder=6, edgecolors="white", linewidths=0.7,
                           label="FP16 cuBLAS (L2-resident, measured)")

# Label each batch size
for bs, ai, t in zip(batch_sizes, fp16_ai, fp16_measured_tflops):
    ax.annotate(f"BS={bs}", xy=(ai, t),
                xytext=(6, 3), textcoords="offset points",
                fontsize=7, color=FP16_L2_COLOR)

# ── INT4 operating point (MLA config, BS=1) ───────────────────────────────────
if int4_ai_pts:
    ax.scatter(int4_ai_pts, int4_tflops_pts,
               s=120, color=INT4_COLOR, marker="^",
               zorder=6, edgecolors="white", linewidths=0.7,
               label="INT4 Triton (compute-bound, measured)")
    for ai_v, t_v, lbl in zip(int4_ai_pts, int4_tflops_pts, int4_labels):
        ax.annotate(lbl, xy=(ai_v, t_v),
                    xytext=(-60, 8), textcoords="offset points",
                    fontsize=7.5, color=INT4_COLOR,
                    arrowprops=dict(arrowstyle="->", color=INT4_COLOR, lw=1.0))

# ── Shaded regime regions ─────────────────────────────────────────────────────
ax.axvspan(ai_range[0], AI_L2_CROSS,  alpha=0.04, color="blue",  zorder=0)
ax.axvspan(AI_L2_CROSS, AI_DRAM_CROSS, alpha=0.04, color="green", zorder=0)
ax.axvspan(AI_DRAM_CROSS, ai_range[-1], alpha=0.04, color="orange", zorder=0)

# Region labels
ax.text(0.12, 200,  "L2-bandwidth\nbound",   fontsize=8, color="blue",   alpha=0.7, style="italic")
ax.text(120,  200,  "HBM-bandwidth\nbound",  fontsize=8, color="green",  alpha=0.7, style="italic")
ax.text(400,  200,  "Compute\nbound",         fontsize=8, color="orange", alpha=0.7, style="italic")

# ── Standard vs cache-aware roofline annotation ───────────────────────────────
# Show what the standard (HBM-only) roofline would PREDICT for BS=1
ai_bs1 = fp16_ai[0]
t_bs1  = fp16_measured_tflops[0]
t_dram_pred  = min(ai_bs1 * H100_DRAM_BW, H100_TFLOPS)
t_l2_pred    = min(ai_bs1 * H100_L2_BW,   H100_TFLOPS)

ax.annotate(
    f"Standard roofline predicts\n{t_dram_pred:.2f} TFLOPS (HBM)\n"
    f"→ 3.9× INT4 speedup",
    xy=(ai_bs1, t_dram_pred),
    xytext=(0.25, 14),
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.6),
    fontsize=7.5, color="red", alpha=0.8,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
)
ax.annotate(
    f"L2 ceiling:\n{t_l2_pred:.1f} TFLOPS\n(FP16 already\nbounded by L2)",
    xy=(ai_bs1, t_l2_pred),
    xytext=(0.5, 35),
    arrowprops=dict(arrowstyle="->", color="blue", lw=1.2, alpha=0.8),
    fontsize=7.5, color="blue", alpha=0.9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7),
)

# ── Weight size reference lines (L2 residency context) ───────────────────────
ax.axvline(x=ai_bs1, color="navy", lw=0.7, linestyle="-.", alpha=0.5, zorder=1)
ax.text(ai_bs1 * 1.06, 0.05,
        f"MLA weight\n16 MB (BS=1)\nAI={ai_bs1:.2f}",
        fontsize=7, color="navy", alpha=0.75)

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
ax.set_ylabel("Achieved Performance (TFLOPS)", fontsize=12)
ax.set_title(
    "Cache-Aware Roofline: MLA Reconstruction on H100 SXM5\n"
    "DeepSeek-V3 · BMM1 (H=128, K=128, N=512) · FP16 vs INT4",
    fontsize=12,
)
ax.set_xlim(ai_range[0], ai_range[-1])
ax.set_ylim(0.008, H100_TFLOPS * 3)
ax.grid(True, alpha=0.2, which="both")
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(script_dir, "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout()
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"cache_aware_roofline.{ext}")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()

# ── Console summary ───────────────────────────────────────────────────────────
print("\nMLA Operating Points (FP16 BMM1)")
print(f"{'BS':>6} {'AI (F/B)':>10} {'Measured':>12} {'L2 pred':>10} {'DRAM pred':>11}")
print("-" * 55)
for bs, ai, t in zip(batch_sizes, fp16_ai, fp16_measured_tflops):
    l2_pred   = min(ai * H100_L2_BW,   H100_TFLOPS)
    dram_pred = min(ai * H100_DRAM_BW, H100_TFLOPS)
    region = ("L2-bound" if ai < AI_L2_CROSS
               else "compute-bound" if ai > AI_DRAM_CROSS else "HBM-bound")
    print(f"{bs:>6} {ai:>10.2f} {t:>10.2f} T {l2_pred:>8.1f} T {dram_pred:>9.2f} T  [{region}]")

print(f"\nCrossover AIs: L1={AI_L1_CROSS:.1f}, L2={AI_L2_CROSS:.1f}, DRAM={AI_DRAM_CROSS:.0f} FLOP/byte")
print(f"MLA 16 MB weight fits in L2 (< {L2_CAPACITY_MB:.0f} MB) → FP16 bounded by L2, not HBM")
print("→ INT4 cannot improve on FP16 by reducing DRAM bytes (FP16 isn't reading from DRAM at steady state)")
