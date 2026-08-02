"""Floor-free band refit on H100 (session 9; answers session 7's open question).

The B200 leg refuted the fitted persistent floor as a transferable element:
B200's far field streams at full bw_hbm, and the floor-free zero-parameter
band (keep C_hi/C_eff = 1.56, ramp to bw_hbm, no fitted floor, no baseline
term) scored 16.6/3.9 zero-shot. Session 7 left open what H100's own held-out
gate sweep pays if the floor goes away (fitted-floor band scored 14.3 below /
13.3 above, vs 20.9 / 11.6 for the binary gate).

Variants scored on the held-out H100 gate sweep (bf16 cells,
results_capacity_gate.json), all with C_eff fixed at the measured 36 MB:

  A binary operand gate                  (reference; session-6 form)
  B fitted-floor ramp   C_hi=56, floor=2.10   (session-7 form, reproduced)
  C floor-free band     C_hi=1.56*C_eff, floor=bw_hbm  (the B200 form)
  D floor-free, C_hi refit on the 55-cell footprint dataset (one parameter)

Output: results_floorfree_band_h100.json
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
    P = json.load(open(os.path.join(_D, "..", "carm_model.json")))
    C = P["effective_l2_capacity_mb"] * 1048576
    BL2 = P["bw_l2_gemm_tbs"] * 1e12
    BH = P["bw_hbm_tbs"] * 1e12
    T0 = P["t0_graph_us"]
    PK = P["peak_tflops"] * 1e12

    band = json.load(open(os.path.join(_D, "results_footprint_transition.json")))
    c_hi_fit = band["fitted_c_hi_mb"] * 1048576
    floor_fit = band["fitted_bw_floor_tbs"] * 1e12
    c_hi_ratio = band["normalized_band"]["c_hi_over_c_eff"]

    # D: refit C_hi alone on the footprint training set, floor pinned at bw_hbm
    fpd = json.load(open(os.path.join(_D, "results_footprint_gate.json")))["results"]
    best = None
    for c_hi_mb in [x * 2 for x in range(20, 61)]:
        if c_hi_mb * 1048576 <= C:
            continue
        bw_w = make_bw_w(BL2, C, c_hi_mb * 1048576, BH)
        errs = []
        for r in fpd:
            wb = H * K * r["N"] * 2.0
            act = H * r["tokens"] * K * 2.0
            out = H * r["tokens"] * r["N"] * 2.0
            fl = 2.0 * H * r["tokens"] * K * r["N"]
            errs.append(abs(predict(wb, act, out, fl, T0, PK, BH, bw_w) - r["t_us"])
                        / r["t_us"])
        m = sum(errs) / len(errs)
        if best is None or m < best[0]:
            best = (m, c_hi_mb)
    m_refit, c_hi_refit_mb = best
    print(f"floor-free C_hi refit on footprint data: C_hi={c_hi_refit_mb} MB "
          f"({c_hi_refit_mb * 1048576 / C:.2f}x C_eff), train MAPE {100 * m_refit:.1f}%")

    gd = json.load(open(os.path.join(_D, "results_capacity_gate.json")))["results"]
    variants = {
        "A binary operand gate": None,
        "B fitted-floor ramp (session 7)": make_bw_w(BL2, C, c_hi_fit, floor_fit),
        "C floor-free band (B200 form, C_hi=1.56x)": make_bw_w(
            BL2, C, c_hi_ratio * C, BH),
        "D floor-free, C_hi refit": make_bw_w(BL2, C, c_hi_refit_mb * 1048576, BH),
    }
    scores = {}
    for tag, bw_w in variants.items():
        below, above = [], []
        for r in gd:
            N, T = r["N"], r["tokens"]
            wb = 2.0 * H * K * N
            act = H * T * K * 2.0
            out = H * T * N * 2.0
            fl = 2.0 * H * T * K * N
            if bw_w is None:
                bw = BL2 if wb < C else BH
                p = T0 + max(wb / bw + (act + out) / BH, fl / PK) * 1e6
            else:
                p = predict(wb, act, out, fl, T0, PK, BH, bw_w)
            err = abs(p - r["bf16_us"]) / r["bf16_us"]
            (below if wb < C else above).append(err)
        scores[tag] = {"below_gate_pct": mape(below), "above_gate_pct": mape(above),
                       "n_below": len(below), "n_above": len(above)}
        print(f"{tag}: below {mape(below)}%  above {mape(above)}%")

    res = {
        "question": "session 7 open item: what does H100's above-gate branch pay "
                    "with the floor-free band, now that B200 refuted the floor",
        "held_out_on": "results_capacity_gate.json bf16 cells",
        "c_eff_mb_fixed": P["effective_l2_capacity_mb"],
        "c_hi_refit_floor_free_mb": c_hi_refit_mb,
        "c_hi_refit_over_c_eff": round(c_hi_refit_mb * 1048576 / C, 2),
        "train_mape_floor_free_pct": round(100 * m_refit, 1),
        "scores": scores,
    }
    with open(os.path.join(_D, "results_floorfree_band_h100.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_floorfree_band_h100.json")


if __name__ == "__main__":
    main()
