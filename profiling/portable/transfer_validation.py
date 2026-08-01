"""
P5 — cross-architecture transfer validation (DIRECTION.md §6 P5).

Question: fit on architecture A, predict architecture B — what MAPE?

2026-08 A100-session update: the A100 hardware constants are now MEASURED
(measure_params.py on an A100-SXM4-40GB, clock-locked, graph-timed), replacing
the carm.py estimates. Two targets are scored:

  PRIMARY — self-consistent: the graph-timed 48-cell sweep re-measured on the
    SAME SXM4-40GB the parameters were measured on
    (results_l2_barrier_a100_40gb_graphtimed.json). No hardware mismatch, no
    eager floor. Uses t0_graph.
  SECONDARY — the in-repo 2026-06 A100-SXM4-80GB sweep
    (results_l2_barrier_a100_extended.json; eager-timed). Two knowing
    mismatches, stated per guardrail 8: bw_hbm (40GB HBM2 1.51 vs 80GB HBM2e
    ~1.94 TB/s) and t0_eager (32.8 us on this host vs ~15.5 us floor on the
    2026-06 host). Kept because it is the original transfer claim's dataset.

  - fp16 predictions are ZERO-SHOT: model form + hardware constants, nothing
    fitted to any A100 kernel measurement.
  - int4 (W4A16) predictions are TWO-POINT calibrated per target: the kernel's
    (r_dequant, fixed cost) from two operating points (8 and 32 MB at bs=1),
    then predict the rest.
  - A NAIVE-SCALING variant (H100 r_dequant scaled by peak-TFLOPS ratio) is
    scored to test whether kernel terms can be extrapolated instead of
    measured. (They cannot.)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))

# A100 hardware constants — MEASURED (params_nvidia-a100-sxm4-40gb.json,
# clock-locked 1410 MHz, graph-timed differential slopes).
with open(os.path.join(_D, "params_nvidia-a100-sxm4-40gb.json")) as f:
    _P = json.load(f)
A100 = dict(bw_hbm=_P["bw_hbm_tbs"] * 1e12,
            bw_l2=_P["bw_l2_tbs"] * 1e12,
            c_eff=_P["effective_l2_capacity_mb"] * 1048576,
            peak=_P["peak_fp16_tflops"] * 1e12,
            t0_graph=_P["t0_graph_us"],
            t0_eager=_P["t0_eager_us"])
# 2026-08-01 estimates, superseded by the measured values above:
# A100 = dict(bw_hbm=1.94e12, bw_l2=4.0e12, c_eff=29 * 1048576,
#             peak=312e12, t0_eager=18.0)
H100_R_DQ = 0.579e12   # measured by measure_params.py on H100
H100_PEAK = 736.8e12

H, K = 128, 128

TARGETS = [
    # (key, filename, t0_us, notes)
    ("a100_40gb_graphtimed", "results_l2_barrier_a100_40gb_graphtimed.json",
     A100["t0_graph"],
     "PRIMARY: same GPU as measured params, graph-timed — self-consistent"),
    ("a100_80gb_eager_2026_06", os.path.join("..", "results_l2_barrier_a100_extended.json"),
     A100["t0_eager"],
     "SECONDARY: 80GB HBM2e target vs 40GB-measured bw_hbm (1.51 vs ~1.94 TB/s) "
     "and host-mismatched eager floor (32.8 us measured here vs ~15.5 us there) "
     "— both mismatches known, stated, and the reason this target is secondary"),
]


def shapes(r):
    n = r["d_lora"]
    E = H * K * n
    bs = r["batch_size"]
    act = H * bs * K * 2.0
    out = H * bs * n * 2.0
    flops = 2.0 * H * bs * K * n
    return E, act, out, flops


def predict_fp16_us(r, t0):
    E, act, out, flops = shapes(r)
    wb = 2.0 * E
    bw = A100["bw_l2"] if wb < A100["c_eff"] else A100["bw_hbm"]
    mem = wb / bw + (act + out) / A100["bw_hbm"]
    return t0 + max(mem, flops / A100["peak"]) * 1e6


def predict_int4_us(r, r_dq, fixed_us):
    E, act, out, flops = shapes(r)
    packed = 0.5 * E
    mem = (packed + 2.0 * H * r["d_lora"] + act + out) / A100["bw_l2"]
    comp = max(flops / A100["peak"], packed / r_dq)
    return fixed_us + max(mem, comp) * 1e6


def mape(pairs):
    errs = [abs(p - m) / m for m, p in pairs]
    return round(100 * sum(errs) / len(errs), 1) if errs else None


def score_target(fname, t0, notes):
    with open(os.path.join(_D, fname)) as f:
        data = json.load(f)
    rows = data["results"]
    bs1 = {r["weight_mb"]: r for r in rows if r["batch_size"] == 1}

    # Two-point calibration of the W4A16 kernel on the target (8 and 32 MB)
    p0, p1 = bs1[8.0], bs1[32.0]
    dp_bytes = 0.5 * H * K * (p1["d_lora"] - p0["d_lora"])
    dt_us = p1["int4_ms"] * 1000 - p0["int4_ms"] * 1000
    r_dq_cal = dp_bytes / (dt_us * 1e-6)
    fixed_cal = p0["int4_ms"] * 1000 - 0.5 * H * K * p0["d_lora"] / r_dq_cal * 1e6

    # Naive scaling variant (no target measurement at all)
    r_dq_naive = H100_R_DQ * (A100["peak"] / H100_PEAK)

    fp16_below, fp16_above, int4_cal_pairs, int4_naive_pairs = [], [], [], []
    table = []
    for r in rows:
        m_f, m_i = r["fp16_ms"] * 1000, r["int4_ms"] * 1000
        p_f = predict_fp16_us(r, t0)
        p_ic = predict_int4_us(r, r_dq_cal, fixed_cal)
        p_in = predict_int4_us(r, r_dq_naive, t0)
        wb = 2.0 * H * K * r["d_lora"]
        (fp16_below if wb < A100["c_eff"] else fp16_above).append((m_f, p_f))
        is_cal = r["batch_size"] == 1 and r["weight_mb"] in (8.0, 32.0)
        if not is_cal:
            int4_cal_pairs.append((m_i, p_ic))
        int4_naive_pairs.append((m_i, p_in))
        table.append(dict(bs=r["batch_size"], w_mb=r["weight_mb"],
                          fp16_us=round(m_f, 2), fp16_pred=round(p_f, 1),
                          int4_us=round(m_i, 2), int4_pred_cal=round(p_ic, 1),
                          int4_pred_naive=round(p_in, 1), calibration_point=is_cal))

    return {
        "target_gpu": data["gpu"],
        "target_file": fname.replace("\\", "/"),
        "timing": data.get("timing", "eager CUDA events (2026-06)"),
        "t0_us_used": round(t0, 3),
        "notes": notes,
        "calibrated_kernel_terms": {
            "r_dequant_tbs": round(r_dq_cal / 1e12, 4),
            "fixed_us": round(fixed_cal, 1),
            "calibration_points_mb": [8, 32],
            "h100_reference_r_dequant_tbs": round(H100_R_DQ / 1e12, 4),
            "naive_peak_scaled_r_dequant_tbs": round(r_dq_naive / 1e12, 4),
            "harness_measured_r_dequant_tbs": _P["r_dequant_tbs"],
        },
        "mape": {
            "fp16_zero_shot_below_gate_pct": mape(fp16_below),
            "fp16_zero_shot_above_gate_pct": mape(fp16_above),
            "int4_two_point_calibrated_pct": mape(int4_cal_pairs),
            "int4_naive_scaled_pct": mape(int4_naive_pairs),
        },
        "rows": table,
    }, bs1, r_dq_cal, fixed_cal, r_dq_naive


def main():
    out = {
        "a100_hw_params_are_estimates": False,
        "a100_hw_params_source": "params_nvidia-a100-sxm4-40gb.json (measured, "
                                 "clock-locked 1410 MHz)",
        "measured_params": {k: _P[k] for k in (
            "gpu", "effective_l2_capacity_mb", "bw_l2_tbs", "bw_hbm_tbs",
            "peak_fp16_tflops", "r_dequant_tbs", "t0_graph_us", "t0_eager_us")},
        "targets": {},
    }
    fig_data = None
    for key, fname, t0, notes in TARGETS:
        scored, bs1, r_dq_cal, fixed_cal, r_dq_naive = score_target(fname, t0, notes)
        out["targets"][key] = scored
        print(f"\n=== {key} ({scored['target_gpu']}) ===")
        print(json.dumps(scored["calibrated_kernel_terms"], indent=2))
        print(json.dumps(scored["mape"], indent=2))
        if key == "a100_40gb_graphtimed":
            fig_data = (bs1, t0, r_dq_cal, fixed_cal, r_dq_naive)

    path = os.path.join(_D, "results_transfer_a100.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    # Figure: primary (self-consistent 40GB) target, bs=1
    bs1, t0, r_dq_cal, fixed_cal, r_dq_naive = fig_data
    c_eff_mb = _P["effective_l2_capacity_mb"]
    ws = sorted(bs1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    axes[0].plot(ws, [bs1[w]["fp16_ms"] * 1000 for w in ws], "o-", c="#2171B5", label="measured")
    axes[0].plot(ws, [predict_fp16_us(bs1[w], t0) for w in ws], "--", c="#2171B5",
                 label="predicted (zero-shot)")
    axes[0].set_title("A100 fp16 cuBLAS — no A100 fitting", fontsize=10)
    axes[1].plot(ws, [bs1[w]["int4_ms"] * 1000 for w in ws], "o-", c="#E6550D", label="measured")
    axes[1].plot(ws, [predict_int4_us(bs1[w], r_dq_cal, fixed_cal) for w in ws], "--",
                 c="#E6550D", label="predicted (2-pt calibrated)")
    axes[1].plot(ws, [predict_int4_us(bs1[w], r_dq_naive, t0) for w in ws], ":",
                 c="gray", label="naive H100 scaling (fails)")
    axes[1].set_title("A100 W4A16 Triton — 2-point kernel calibration", fontsize=10)
    for ax in axes:
        ax.axvline(c_eff_mb, color="crimson", ls=":", lw=1,
                   label=f"C$_{{eff}}$ measured {c_eff_mb:.1f} MB")
        ax.set_xlabel("weight working set (MB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("latency (µs, graph-timed)")
    fig.suptitle("Transfer: CARM form + measured constants + cheap per-target "
                 "microbenchmarks predict an unseen GPU", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(_D, "fig_transfer_a100.png"), dpi=180)
    print("\nsaved results_transfer_a100.json, fig_transfer_a100.png")


if __name__ == "__main__":
    main()
