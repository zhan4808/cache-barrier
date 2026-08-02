"""
B200 transfer validation (2026-08-02 B200 session, KICKOFF_B200 goal 1).

Third architecture point for the transfer claim. Replicates the final A100
recipe (fit_footprint_transition.py validation (b), transfer_validation.py)
verbatim on the B200 self-consistent target:

  fp16 ZERO-SHOT ladder, reported per variant (guardrail 7, regime-separated):
    1. measured constants only        (binary operand gate, split-mem)
    2. + baseline kernel term         (two-point below-gate weight-BW
                                       calibration on bs=1 @ 8 & 24 MB --
                                       "baseline kernels are kernel terms too";
                                       a no-op if it matches harness bw_l2,
                                       as on H100: 6.51 vs 6.3)
    3. + transferred footprint band   (H100-fitted C_hi/C_eff=1.56,
                                       floor/bw_hbm=0.668, applied normalized,
                                       ZERO-SHOT -- nothing refit on B200)

  W4A16: two-point calibrated (bs=1 @ 8 & 32 MB) + naive-H100-scaling control.

Held-out scoring: fp16 calibration cells (bs=1, 8 & 24 MB) excluded from fp16
MAPE; int4 calibration cells (bs=1, 8 & 32 MB) excluded from int4 MAPE.

Output: results_transfer_b200.json, fig_transfer_b200.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))
H, K = 128, 128

_P = json.load(open(os.path.join(_D, "params_nvidia-b200.json")))
C = _P["effective_l2_capacity_mb"] * 1048576
BL2 = _P["bw_l2_tbs"] * 1e12
BH = _P["bw_hbm_tbs"] * 1e12
PK = _P["peak_fp16_tflops"] * 1e12
T0 = _P["t0_graph_us"]

_B = json.load(open(os.path.join(_D, "..", "gate",
                                 "results_footprint_transition.json")))
C_HI_RATIO = _B["normalized_band"]["c_hi_over_c_eff"]
FLOOR_RATIO = _B["normalized_band"]["floor_over_bw_hbm"]

H100_R_DQ = 0.579e12
H100_PEAK = 736.8e12


def make_bw_w(bw_l2, c_eff, c_hi, bw_floor):
    def bw_w(fp):
        if fp <= c_eff:
            return bw_l2
        if fp >= c_hi:
            return bw_floor
        x = (fp - c_eff) / (c_hi - c_eff)
        return 1.0 / ((1 - x) / bw_l2 + x / bw_floor)
    return bw_w


def terms(r):
    n, bs = r["d_lora"], r["batch_size"]
    wb = 2.0 * H * K * n
    act = H * bs * K * 2.0
    out = H * bs * n * 2.0
    flops = 2.0 * H * bs * K * n
    return wb, act, out, flops


def mape(pairs):
    errs = [abs(p - m) / m for m, p in pairs]
    return round(100 * sum(errs) / len(errs), 1) if errs else None


def main():
    da = json.load(open(os.path.join(_D, "results_l2_barrier_b200_graphtimed.json")))
    rows = da["results"]
    bs1 = {r["weight_mb"]: r for r in rows if r["batch_size"] == 1}

    # baseline kernel term: two-point below-gate weight-BW (bs=1, 8 & 24 MB)
    b0, b1 = bs1[8.0], bs1[24.0]
    d_wb = 2.0 * H * K * (b1["d_lora"] - b0["d_lora"])
    dt = (b1["fp16_ms"] - b0["fp16_ms"]) * 1e-3
    r_base = d_wb / dt
    base_matches_harness = abs(r_base - BL2) / BL2 < 0.25

    # W4A16 kernel: two-point calibration (bs=1, 8 & 32 MB)
    p0, p1 = bs1[8.0], bs1[32.0]
    dp = 0.5 * H * K * (p1["d_lora"] - p0["d_lora"])
    dt4 = (p1["int4_ms"] - p0["int4_ms"]) * 1e-3
    r_dq_cal = dp / dt4
    fixed_cal = p0["int4_ms"] * 1000 - 0.5 * H * K * p0["d_lora"] / r_dq_cal * 1e6
    r_dq_naive = H100_R_DQ * (PK / H100_PEAK)

    bl2_eff = min(BL2, r_base)
    bw_w = make_bw_w(bl2_eff, C, C_HI_RATIO * C, FLOOR_RATIO * BH)

    fp16_variants = {
        "measured constants only": lambda wb, act, out, fl: (
            T0 + max(wb / (BL2 if wb < C else BH) + (act + out) / BH,
                     fl / PK) * 1e6),
        "+ baseline kernel term": lambda wb, act, out, fl: (
            T0 + max(wb / (bl2_eff if wb < C else BH) + (act + out) / BH,
                     fl / PK) * 1e6),
        "+ transferred band": lambda wb, act, out, fl: (
            T0 + max(wb / bw_w(wb + act + out) + (act + out) / BH,
                     fl / PK) * 1e6),
        # B200 finding: the H100-fitted persistent floor does NOT transfer
        # (above-gate cells stream at full bw_hbm), and the 2-pt baseline term
        # is corrupted by sm_100 kernel-selection jaggedness. The band's
        # EXTENT (C_hi/C_eff) does transfer: zero-parameter variant ramps
        # 1/bw from bw_l2 at C_eff to bw_hbm at C_hi, bw_hbm beyond, harness
        # bw_l2 below (no baseline term, no fitted floor -- fully zero-shot).
        "zero-param band (ramp to bw_hbm)": lambda wb, act, out, fl: (
            T0 + max(wb / make_bw_w(BL2, C, C_HI_RATIO * C, BH)(wb + act + out)
                     + (act + out) / BH, fl / PK) * 1e6),
    }

    def pred_int4(r, r_dq, fixed_us):
        wb, act, out, fl = terms(r)
        packed = 0.5 * H * K * r["d_lora"]
        mem = (packed + 2.0 * H * r["d_lora"] + act + out) / BL2
        comp = max(fl / PK, packed / r_dq)
        return fixed_us + max(mem, comp) * 1e6

    fp16_cal_cells = {(1, 8.0), (1, 24.0)}
    int4_cal_cells = {(1, 8.0), (1, 32.0)}

    variant_mape, table = {}, []
    for tag, fn in fp16_variants.items():
        below, above = [], []
        for r in rows:
            if (r["batch_size"], r["weight_mb"]) in fp16_cal_cells:
                continue
            wb, act, out, fl = terms(r)
            m = r["fp16_ms"] * 1000
            (below if wb < C else above).append((m, fn(wb, act, out, fl)))
        variant_mape[tag] = {
            "below_gate_pct": mape(below), "above_gate_pct": mape(above),
            "n_below": len(below), "n_above": len(above)}

    int4_cal_pairs, int4_naive_pairs = [], []
    full_fn = fp16_variants["zero-param band (ramp to bw_hbm)"]
    for r in rows:
        wb, act, out, fl = terms(r)
        m_f, m_i = r["fp16_ms"] * 1000, r["int4_ms"] * 1000
        p_f = full_fn(wb, act, out, fl)
        p_ic = pred_int4(r, r_dq_cal, fixed_cal)
        if (r["batch_size"], r["weight_mb"]) not in int4_cal_cells:
            int4_cal_pairs.append((m_i, p_ic))
        int4_naive_pairs.append((m_i, pred_int4(r, r_dq_naive, T0)))
        table.append(dict(bs=r["batch_size"], w_mb=r["weight_mb"],
                          fp16_us=round(m_f, 2), fp16_pred=round(p_f, 1),
                          int4_us=round(m_i, 2), int4_pred_cal=round(p_ic, 1)))

    out = {
        "gpu": da["gpu"],
        "clock_lock": da.get("clock_lock"),
        "params_source": "params_nvidia-b200.json (measured this session)",
        "band_source": "results_footprint_transition.json (H100-fitted, "
                       "transferred normalized, zero-shot)",
        "band_applied": {"c_hi_mb": round(C_HI_RATIO * C / 1048576, 1),
                         "floor_tbs": round(FLOOR_RATIO * BH / 1e12, 2)},
        "baseline_kernel_check": {
            "two_point_below_gate_bw_tbs": round(r_base / 1e12, 2),
            "harness_bw_l2_tbs": round(BL2 / 1e12, 2),
            "matches_harness_within_25pct": base_matches_harness,
            "calibration_points_mb": [8, 24],
            "note": "H100 analogue was 6.51 vs 6.3 (no hidden term); "
                    "A100 was 1.28 vs 3.85 (sm_80 kernel floor)"},
        "calibrated_kernel_terms": {
            "r_dequant_tbs": round(r_dq_cal / 1e12, 4),
            "fixed_us": round(fixed_cal, 2),
            "calibration_points_mb": [8, 32],
            "harness_measured_r_dequant_tbs": _P["r_dequant_tbs"],
            "naive_peak_scaled_r_dequant_tbs": round(r_dq_naive / 1e12, 4)},
        "mape_fp16_zero_shot_held_out": variant_mape,
        "mape_int4": {
            "two_point_calibrated_pct": mape(int4_cal_pairs),
            "naive_scaled_pct": mape(int4_naive_pairs)},
        "rows": table,
    }
    path = os.path.join(_D, "results_transfer_b200.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

    # figure: bs=1 panel, same layout as fig_transfer_a100.png
    c_eff_mb = _P["effective_l2_capacity_mb"]
    ws = sorted(bs1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    axes[0].plot(ws, [bs1[w]["fp16_ms"] * 1000 for w in ws], "o-",
                 c="#2171B5", label="measured")
    axes[0].plot(ws, [full_fn(*terms(bs1[w])) for w in ws], "--",
                 c="#2171B5", label="predicted (zero-shot, zero-param band)")
    axes[0].set_title("B200 fp16 cuBLAS -- no B200 fitting", fontsize=10)
    axes[1].plot(ws, [bs1[w]["int4_ms"] * 1000 for w in ws], "o-",
                 c="#E6550D", label="measured")
    axes[1].plot(ws, [pred_int4(bs1[w], r_dq_cal, fixed_cal) for w in ws], "--",
                 c="#E6550D", label="predicted (2-pt calibrated)")
    axes[1].plot(ws, [pred_int4(bs1[w], r_dq_naive, T0) for w in ws], ":",
                 c="gray", label="naive H100 scaling")
    axes[1].set_title("B200 W4A16 Triton -- 2-point kernel calibration", fontsize=10)
    for ax in axes:
        ax.axvline(c_eff_mb, color="crimson", ls=":", lw=1,
                   label=f"C$_{{eff}}$ measured {c_eff_mb:.1f} MB")
        ax.set_xlabel("weight working set (MB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("latency (us, graph-timed)")
    fig.suptitle("Transfer, third architecture: CARM constants + 2-pt kernel terms "
                 "+ normalized band predict the B200", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(_D, "fig_transfer_b200.png"), dpi=180)
    print("saved results_transfer_b200.json, fig_transfer_b200.png")


if __name__ == "__main__":
    main()
