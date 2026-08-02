"""
Soft footprint-collapse transition term (2026-08-02).

The binary footprint gate recovers only ~2 points of below-gate MAPE because
the residency collapse is soft: 2-3 TB/s across a 40-60 MB footprint band
(results_footprint_gate.json). This fits a two-parameter transition:

  1/bw_w(fp) = 1/BW_L2                          fp <= C_eff
             = lerp(1/BW_L2 -> 1/bw_floor)      C_eff < fp < C_hi   (inverse-BW linear in fp)
             = 1/bw_floor                       fp >= C_hi

with C_eff FIXED at the measured value (36 MB, not fitted); fitted params are
C_hi and bw_floor, trained ONLY on the 55-cell footprint dataset by grid
search on latency MAPE under t = t0 + max(wb/bw_w + (act+out)/BW_HBM, F/P).

Validation is transfer-style: the fitted band, normalized (c_hi_ratio =
C_hi/C_eff, floor_ratio = bw_floor/BW_HBM), is applied UNCHANGED to
  (a) the H100 gate-sweep bf16 below-gate cells (35 cells, separate dataset),
  (b) the A100 graph-timed transfer target (with the sm_80 baseline kernel
      floor from the 2-pt calibration, which is a separate, additive term).

Output: results_footprint_transition.json
"""

import json
import os

_D = os.path.dirname(os.path.abspath(__file__))
H = K = 128


def make_bw_w(bw_l2, c_eff, c_hi, bw_floor):
    def bw_w(fp):
        if fp <= c_eff:
            return bw_l2
        if fp >= c_hi:
            return bw_floor
        x = (fp - c_eff) / (c_hi - c_eff)
        inv = (1 - x) / bw_l2 + x / bw_floor
        return 1.0 / inv
    return bw_w


def predict(wb, act, out, flops, t0, peak, bw_hbm, bw_w):
    fp = wb + act + out
    mem = wb / bw_w(fp) + (act + out) / bw_hbm
    return t0 + max(mem, flops / peak) * 1e6


def mape(errs):
    return round(100 * sum(errs) / len(errs), 1)


def main():
    # ---- H100 params + footprint training data ----
    P = json.load(open(os.path.join(_D, "..", "carm_model.json")))
    C = P["effective_l2_capacity_mb"] * 1048576
    BL2 = P["bw_l2_gemm_tbs"] * 1e12
    BH = P["bw_hbm_tbs"] * 1e12
    T0 = P["t0_graph_us"]
    PK = P["peak_tflops"] * 1e12

    fpd = json.load(open(os.path.join(_D, "results_footprint_gate.json")))["results"]

    def cell_terms(r):
        N, T = r["N"], r["tokens"]
        wb = 2.0 * H * K * N / 2  # N.B. footprint bench stores N = 32*w_mb with fp16 wb = H*K*N*2
        wb = H * K * N * 2.0
        act = H * T * K * 2.0
        out = H * T * N * 2.0
        flops = 2.0 * H * T * K * N
        return wb, act, out, flops

    # grid-search C_hi (MB) and bw_floor (TB/s) on the footprint dataset
    best = None
    for c_hi_mb in [x * 2 for x in range(20, 61)]:          # 40..120 MB
        if c_hi_mb * 1048576 <= C:
            continue
        for floor in [x * 0.05e12 for x in range(20, 64)]:  # 1.0..3.15 TB/s
            bw_w = make_bw_w(BL2, C, c_hi_mb * 1048576, floor)
            errs = []
            for r in fpd:
                wb, act, out, fl = cell_terms(r)
                p = predict(wb, act, out, fl, T0, PK, BH, bw_w)
                errs.append(abs(p - r["t_us"]) / r["t_us"])
            m = sum(errs) / len(errs)
            if best is None or m < best[0]:
                best = (m, c_hi_mb, floor)
    m_fit, c_hi_mb, floor = best
    print(f"fitted on footprint data: C_hi={c_hi_mb} MB, bw_floor={floor/1e12:.2f} TB/s, "
          f"train MAPE {100*m_fit:.1f}%")

    # baseline (binary operand gate, split-mem) on same training data for reference
    bw_bin = lambda fp: BL2  # noqa: E731  (operand-gated cells here all have wb<C)
    errs_bin = []
    for r in fpd:
        wb, act, out, fl = cell_terms(r)
        bw = BL2 if wb < C else BH
        p = T0 + max(wb / bw + (act + out) / BH, fl / PK) * 1e6
        errs_bin.append(abs(p - r["t_us"]) / r["t_us"])
    print(f"  reference binary-operand model on same data: {mape(errs_bin)}%")

    # ---- validation (a): H100 gate sweep, bf16 below-gate ----
    gd = json.load(open(os.path.join(_D, "results_capacity_gate.json")))["results"]
    bw_w = make_bw_w(BL2, C, c_hi_mb * 1048576, floor)
    for tag, use_ramp in (("binary operand gate", False), ("footprint ramp", True)):
        below, above = [], []
        for r in gd:
            N, T = r["N"], r["tokens"]
            wb = 2.0 * H * K * N
            act = H * T * K * 2.0
            out = H * T * N * 2.0
            fl = 2.0 * H * T * K * N
            if use_ramp:
                p = predict(wb, act, out, fl, T0, PK, BH, bw_w)
            else:
                bw = BL2 if wb < C else BH
                p = T0 + max(wb / bw + (act + out) / BH, fl / PK) * 1e6
            err = abs(p - r["bf16_us"]) / r["bf16_us"]
            (below if wb < C else above).append(err)
        print(f"gate sweep bf16 [{tag}]: below {mape(below)}% (n={len(below)})  "
              f"above {mape(above)}% (n={len(above)})")

    # ---- validation (b): A100 transfer target, normalized band + kernel floor ----
    pa = json.load(open(os.path.join(_D, "..", "portable",
                                     "params_nvidia-a100-sxm4-40gb.json")))
    C_A = pa["effective_l2_capacity_mb"] * 1048576
    BL2A = pa["bw_l2_tbs"] * 1e12
    BHA = pa["bw_hbm_tbs"] * 1e12
    T0A = pa["t0_graph_us"]
    PKA = pa["peak_fp16_tflops"] * 1e12
    R_BASE = 1.28e12  # sm_80 baseline kernel floor, 2-pt calibrated (session 6)
    c_hi_ratio = c_hi_mb * 1048576 / C
    floor_ratio = floor / BH
    bw_w_a = make_bw_w(min(BL2A, R_BASE), C_A, c_hi_ratio * C_A, floor_ratio * BHA)

    da = json.load(open(os.path.join(_D, "..", "portable",
                                     "results_l2_barrier_a100_40gb_graphtimed.json")))
    variants = {
        "measured constants only": lambda wb, act, out, fl: (
            T0A + max(wb / (BL2A if wb < C_A else BHA) + (act + out) / BHA,
                      fl / PKA) * 1e6),
        "+ sm80 kernel floor": lambda wb, act, out, fl: (
            T0A + max(wb / (min(BL2A, R_BASE) if wb < C_A else BHA)
                      + (act + out) / BHA, fl / PKA) * 1e6),
        "+ kernel floor + transferred ramp": lambda wb, act, out, fl: predict(
            wb, act, out, fl, T0A, PKA, BHA, bw_w_a),
    }
    out_v = {}
    for tag, fn in variants.items():
        below, above = [], []
        for r in da["results"]:
            n, bs = r["d_lora"], r["batch_size"]
            wb = 2.0 * H * K * n
            act = H * bs * K * 2.0
            outb = H * bs * n * 2.0
            fl = 2.0 * H * bs * K * n
            m = r["fp16_ms"] * 1000
            err = abs(fn(wb, act, outb, fl) - m) / m
            (below if wb < C_A else above).append(err)
        out_v[tag] = {"below": mape(below), "above": mape(above)}
        print(f"A100 transfer fp16 [{tag}]: below {mape(below)}%  above {mape(above)}%")

    res = {
        "fitted_on": "results_footprint_gate.json (55 cells, H100)",
        "c_eff_mb_fixed": P["effective_l2_capacity_mb"],
        "fitted_c_hi_mb": c_hi_mb,
        "fitted_bw_floor_tbs": round(floor / 1e12, 2),
        "train_mape_pct": round(100 * m_fit, 1),
        "normalized_band": {"c_hi_over_c_eff": round(c_hi_ratio, 2),
                            "floor_over_bw_hbm": round(floor_ratio, 3)},
        "a100_validation": out_v,
        "note": "ramp = inverse-BW linear interpolation in footprint between "
                "C_eff and C_hi; A100 applies the H100-fitted band normalized "
                "by each card's C_eff and bw_hbm (zero-shot band transfer)",
    }
    with open(os.path.join(_D, "results_footprint_transition.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_footprint_transition.json")


if __name__ == "__main__":
    main()
