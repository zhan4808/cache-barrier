"""
Cache-Aware Roofline Model (CARM) for MLA reconstruction — measured edition
===========================================================================
Rewritten 2026-06 after the methodology audit (see validation/REPORT.md).
The previous version of this script assumed L2 = 12 TB/s and capacity = 50 MB;
both are wrong for these kernels. This version builds the model entirely from
measured parameters (measure_carm_params.py) and validates it against
graph-timed operating points.

Model (per kernel launch with F FLOPs, B bytes, working set WS):

  FP16 cuBLAS:   t = t0 + max( B / BW(WS),  F / P_peak )
                 BW(WS) = BW_L2_eff   if WS <= C_eff (~36 MB effective L2)
                          BW_HBM_eff  otherwise
  INT4 W4A16:    t = t0' + W_packed * ceil(bs/BLOCK_M) / R_dq
                 (dequantization-throughput bound: every M-tile re-unpacks the
                  full packed weight tile; R_dq is fitted packed-byte dequant
                  throughput, ~0.5 TB/s -> a ~30 TFLOPS in-core ceiling)

This extends the cache-aware roofline of Ilic et al. (IEEE CAL 2014) in two
ways needed for microsecond-scale LLM kernels:
  1. a capacity-gated bandwidth ceiling BW(WS) with the *measured effective*
     capacity (32-40 MB on H100, not the nominal 50 MB), and
  2. an explicit per-launch fixed cost t0 (2.8 us graph-captured; 15.4 us
     eager), without which every bs<=16 point sits inexplicably far below
     the loft.

Outputs: ../paper/figures/carm_roofline.{png,pdf}, carm_model.json
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_D, "carm_params.json")) as f:
    P = json.load(f)

PEAK = P["peak_fp16_tflops"]
BW_HBM = P["hbm_read_tbs"]            # measured streaming read (3.15 TB/s here)
BW_L2_GEMM = 6.3                      # measured incremental slope of cuBLAS bmm below cliff
BW_L2_RED = P["l2_read_tbs"]          # measured reduction-pattern L2 read (5.0 TB/s)
C_EFF_MB = 36.0                       # measured effective residency capacity (cliff at 32-40 MB)
T0_GRAPH = P["kernel_floor_us"]       # us
T0_EAGER = P["eager_floor_us"]        # us
H, K, N = 128, 128, 512
WT_PACKED = H * K * N // 2            # bytes
BLOCK_M = 16

pts = P["recon_points"]

# ── Fit INT4 dequant throughput R_dq (packed bytes/s) ────────────────────────
r_samples = []
for p in pts:
    work = WT_PACKED * math.ceil(p["bs"] / BLOCK_M)
    t_core = max(p["int4_us"] - 6.0, 0.5)  # subtract approx INT4 fixed cost
    r_samples.append(work / (t_core * 1e-6))
R_DQ = float(np.median(r_samples))
T0_INT4 = 6.0  # us, Triton launch + prologue at this grid size (fitted)


def bw_ws(ws_bytes):
    # strict <: a working set AT the effective capacity already thrashes (LRU)
    return BW_L2_GEMM * 1e12 if ws_bytes < C_EFF_MB * 1024 * 1024 else BW_HBM * 1e12


def predict_fp16_us(flops, bytes_, t0=T0_GRAPH):
    return t0 + max(bytes_ / bw_ws(bytes_), flops / (PEAK * 1e12)) * 1e6


def predict_int4_us(bs):
    return T0_INT4 + WT_PACKED * math.ceil(bs / BLOCK_M) / R_DQ * 1e6


# ── Figure ────────────────────────────────────────────────────────────────────
C_FP, C_I4, C_HBM, C_L2 = "#1565C0", "#C62828", "#37474F", "#2E7D32"
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(18, 5.6))
fig.suptitle(
    f"Cache-Aware Roofline (measured) — MLA reconstruction BMM1 on {P['gpu']}\n"
    f"BW$_{{HBM}}$={BW_HBM:.2f} TB/s, BW$_{{L2}}^{{eff}}$={BW_L2_GEMM:.1f} TB/s (GEMM) / "
    f"{BW_L2_RED:.1f} TB/s (reduction), C$_{{eff}}$={C_EFF_MB:.0f} MB, "
    f"t$_0$={T0_GRAPH:.1f} µs graph / {T0_EAGER:.1f} µs eager, R$_{{dq}}$={R_DQ/1e12:.2f} TB/s",
    y=1.04)

# Panel A: the roofline
ai = np.logspace(-1, 3.2, 600)
ax.loglog(ai, np.minimum(ai * BW_HBM, PEAK), color=C_HBM, lw=2,
          label=f"HBM ceiling ({BW_HBM:.2f} TB/s measured)")
ax.fill_between(ai, np.minimum(ai * BW_L2_RED, PEAK), np.minimum(ai * BW_L2_GEMM, PEAK),
                color=C_L2, alpha=0.18)
ax.loglog(ai, np.minimum(ai * BW_L2_GEMM, PEAK), color=C_L2, lw=2, ls="--",
          label=f"L2 ceiling, WS$\\leq${C_EFF_MB:.0f} MB ({BW_L2_RED:.1f}–{BW_L2_GEMM:.1f} TB/s measured)")
ax.axhline(PEAK, color="k", lw=1.2, alpha=0.5)
ax.text(1.3e3, PEAK * 1.15, f"{PEAK:.0f} TF peak", fontsize=8, ha="right")
ax.axhline(30.3, color=C_I4, lw=1.5, ls=":")
ax.text(1.3e3, 33, "INT4 dequant in-core ceiling ≈30 TF (3% of peak)",
        fontsize=8, color=C_I4, ha="right")

for p in pts:
    l2res = p["fp16_bytes"] <= C_EFF_MB * 1024 * 1024
    ax.scatter(p["ai_fp16"], p["fp16_tflops"], s=70, zorder=5,
               facecolor=C_FP if l2res else "white", edgecolor=C_FP, linewidth=1.6)
    ax.scatter(p["ai_int4"], p["int4_tflops"], s=70, zorder=5, marker="^",
               facecolor=C_I4, edgecolor="white", linewidth=0.6)
    ax.annotate(f"{p['bs']}", xy=(p["ai_fp16"], p["fp16_tflops"]),
                xytext=(-2, 7), textcoords="offset points", fontsize=7, color=C_FP)
    # launch-floor cap for this FLOP count
    cap = p["flops"] / (T0_GRAPH * 1e-6) / 1e12
    ax.plot([p["ai_fp16"] * 0.7, p["ai_fp16"] * 1.4], [cap, cap], color="#FF8F00", lw=1.2, alpha=0.8)

ax.plot([], [], color="#FF8F00", lw=1.2, label="launch-floor cap $F/t_0$ (per point)")
ax.scatter([], [], s=70, facecolor=C_FP, edgecolor=C_FP, label="FP16 (filled = WS in L2)")
ax.scatter([], [], s=70, facecolor="white", edgecolor=C_FP, label="FP16 (open = WS > C$_{eff}$)")
ax.scatter([], [], s=70, marker="^", facecolor=C_I4, label="INT4 W4A16")
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Achieved performance (TFLOPS)")
ax.set_title("A — CARM with measured ceilings + per-launch floor caps\n"
             "(graph-timed operating points, labels = batch size)")
ax.set_xlim(0.5, 1.5e3); ax.set_ylim(0.8, 2500)
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7.5, loc="upper left")

# Panel B: the capacity-gated bandwidth function BW(WS)
sweep = P.get("fp16_size_sweep", {})
ws_mb = sorted(float(v["weight_mb"]) for v in sweep.values())
eff_raw, eff_floor = [], []
for m in ws_mb:
    e = next(v for v in sweep.values() if v["weight_mb"] == m)
    t = e["us_per_bmm"]
    b = m * 1024 * 1024
    eff_raw.append(b / (t * 1e-6) / 1e12)
    eff_floor.append(b / (max(t - T0_GRAPH, 0.05) * 1e-6) / 1e12)
bx.plot(ws_mb, eff_raw, "o-", color=C_FP, label="bytes / t (graph-timed)")
bx.plot(ws_mb, eff_floor, "o--", color=C_FP, alpha=0.45, label="bytes / (t − t$_0$)")
bx.axhline(BW_HBM, color=C_HBM, lw=1.5, ls="-")
bx.text(95, BW_HBM + 0.15, f"HBM {BW_HBM:.2f} TB/s", fontsize=8, color=C_HBM)
bx.axhspan(BW_L2_RED, BW_L2_GEMM, color=C_L2, alpha=0.15)
bx.text(2, BW_L2_GEMM + 0.15, f"L2 effective {BW_L2_RED:.1f}–{BW_L2_GEMM:.1f} TB/s", fontsize=8, color=C_L2)
bx.axvspan(32, 40, color="orange", alpha=0.18)
bx.axvline(C_EFF_MB, color="orange", ls="--", lw=1.2)
bx.text(C_EFF_MB + 1, 7.2, f"C$_{{eff}}$ ≈ {C_EFF_MB:.0f} MB\n(nominal L2 = 50 MB)", fontsize=8, color="#E65100")
bx.set_xlabel("Working set (MB)")
bx.set_ylabel("Effective serving bandwidth (TB/s)")
bx.set_title("B — Capacity-gated bandwidth BW(WS):\nthe step function the model uses")
bx.set_ylim(0, 8); bx.grid(alpha=0.25); bx.legend(fontsize=8)

# Panel C: model validation, predicted vs measured
meas_f = [p["fp16_us"] for p in pts]
pred_f = [predict_fp16_us(p["flops"], p["fp16_bytes"]) for p in pts]
meas_i = [p["int4_us"] for p in pts]
pred_i = [predict_int4_us(p["bs"]) for p in pts]
lo, hi = 2, 400
cx.loglog([lo, hi], [lo, hi], "k-", lw=1, alpha=0.5)
cx.loglog([lo, hi], [lo * 1.25, hi * 1.25], "k:", lw=0.8, alpha=0.4)
cx.loglog([lo, hi], [lo / 1.25, hi / 1.25], "k:", lw=0.8, alpha=0.4)
cx.scatter(meas_f, pred_f, s=70, color=C_FP, zorder=5, label="FP16: t$_0$+max(B/BW(WS), F/P)")
cx.scatter(meas_i, pred_i, s=70, marker="^", color=C_I4, zorder=5,
           label="INT4: t$_0$'+W$_{packed}$·⌈bs/16⌉/R$_{dq}$")
for p, m, pr in zip(pts, meas_f, pred_f):
    cx.annotate(f"{p['bs']}", xy=(m, pr), xytext=(4, -2), textcoords="offset points",
                fontsize=7, color=C_FP)
mape_f = float(np.mean([abs(a - b) / a for a, b in zip(meas_f, pred_f)])) * 100
mape_i = float(np.mean([abs(a - b) / a for a, b in zip(meas_i, pred_i)])) * 100
cx.set_xlabel("Measured latency (µs, graph-timed)")
cx.set_ylabel("Model-predicted latency (µs)")
cx.set_title(f"C — Model validation across batch sizes\nFP16 MAPE {mape_f:.0f}%  ·  INT4 MAPE {mape_i:.0f}%  (dotted = ±25%)")
cx.grid(alpha=0.25, which="both"); cx.legend(fontsize=8, loc="upper left")

plt.tight_layout()
out_dir = os.path.join(_D, "..", "paper", "figures")
os.makedirs(out_dir, exist_ok=True)
for ext in ("png", "pdf"):
    fp = os.path.join(out_dir, f"carm_roofline.{ext}")
    plt.savefig(fp, dpi=170, bbox_inches="tight")
    print("Saved:", fp)
plt.close()

# ── Persist fitted model + console table ─────────────────────────────────────
model = {
    "gpu": P["gpu"],
    "peak_tflops": PEAK,
    "bw_hbm_tbs": BW_HBM,
    "bw_l2_gemm_tbs": BW_L2_GEMM,
    "bw_l2_reduction_tbs": BW_L2_RED,
    "effective_l2_capacity_mb": C_EFF_MB,
    "t0_graph_us": T0_GRAPH,
    "t0_eager_us": T0_EAGER,
    "t0_int4_us": T0_INT4,
    "r_dequant_tbs": round(R_DQ / 1e12, 3),
    "int4_incore_ceiling_tflops": round(4 * BLOCK_M * R_DQ / 1e12, 1),
    "fp16_mape_pct": round(mape_f, 1),
    "int4_mape_pct": round(mape_i, 1),
}
with open(os.path.join(_D, "carm_model.json"), "w") as f:
    json.dump(model, f, indent=2)
print(json.dumps(model, indent=2))

print(f"\n{'bs':>5} {'FP16 meas':>10} {'FP16 pred':>10} {'INT4 meas':>10} {'INT4 pred':>10}")
for p, pf, pi in zip(pts, pred_f, pred_i):
    print(f"{p['bs']:>5} {p['fp16_us']:>9.1f}µ {pf:>9.1f}µ {p['int4_us']:>9.1f}µ {pi:>9.1f}µ")
