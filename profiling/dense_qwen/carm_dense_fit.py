"""CARM v2 — fit the cache-aware model on the dense Qwen3.6-27B sweep.

Extends CARM with the terms the dense experiment measured:
  t(path, shape, M, mode) = t0 + max(W_bytes / BW_tier, F / P_path) + Q(path, M)

  * BW_tier: capacity-gated PER OPERAND AND MODE — the operand that must be
    re-read (the quantized weight for w8a16/w8a8, the bf16 weight for bf16)
    gets the L2 tier iff it fits C_eff AND mode is warm (residency survives);
    rotated mode (serving eviction) always gets the HBM tier. This is the
    operand-aware capacity gate.
  * P_path: per-path compute ceiling (bf16 GEMM peak; Marlin in-core dequant
    ceiling — dense variant; W8A8 fp8-MMA peak), fit from M=2048 cells.
  * Q(path, M): deployed-path overhead — for w8a8 the dynamic act-quant kernel
    (fit t0q + act_bytes/BW_q from the measured w8a8 - w8a8_mm gap). Zero for
    bf16/w8a16.

Fits every parameter from designated cells, then reports MAPE over ALL cells
(5 shapes x 9 Ms x 3 deployed paths x 2 modes = 270 points) and the predicted
vs measured W8A16 crossovers per shape.
"""

import json
import os
import statistics

_D = os.path.dirname(os.path.abspath(__file__))
T0 = 2.78          # graph launch floor (us), measured
C_EFF_MB = 36.0

SHAPE_DIMS = {"kv_proj": (5120, 2048), "q_proj": (5120, 6144), "o_proj": (6144, 5120),
              "down_proj": (17408, 5120), "gate_up": (5120, 34816)}


def load(mode):
    f = "results_dense_proj_h100.json" if mode == "warm" else "results_dense_proj_h100_rotated.json"
    return json.load(open(os.path.join(_D, f)))["results"]


def cells(mode):
    for s in load(mode):
        K, N = SHAPE_DIMS[s["shape"]]
        for r in s["rows"]:
            yield s["shape"], K, N, r["M"], r, s


def eff_bw(w_mb, us):
    return w_mb / max(us - T0, 0.05)          # MB/us == TB/s


def fit():
    warm = list(cells("warm"))
    rot = list(cells("rotated"))

    # --- tier bandwidths (per path family), from M<=16 memory-bound cells ---
    def wbw(pts, path, resident):
        out = []
        for name, K, N, M, r, s in pts:
            if M > 16:
                continue
            mb = s["wt_mb_bf16"] if path == "bf16" else s["wt_mb_fp8"]
            fits = mb < C_EFF_MB
            if fits != resident:
                continue
            us = r["bf16"] if path == "bf16" else (r["w8a8_mm"] if path == "w8a8" else r["w8a16"])
            out.append(eff_bw(mb, us))
        return statistics.median(out) if out else None

    BW = {
        ("bf16", "hbm"): wbw(rot, "bf16", False) or 2.8,
        ("bf16", "l2"):  wbw(warm, "bf16", True),
        ("w8a8", "hbm"): wbw(rot, "w8a8", False),
        ("w8a8", "l2"):  wbw(warm, "w8a8", True),
        ("w8a16", "hbm"): wbw(rot, "w8a16", False),
        ("w8a16", "l2"):  wbw(warm, "w8a16", True),
    }

    # --- compute ceilings from M=2048 cells (flops/(t-t0)), median over shapes ---
    def ceil_tflops(pts, key):
        vals = []
        for name, K, N, M, r, s in pts:
            if M != 2048:
                continue
            f = 2 * M * K * N
            vals.append(f / max(r[key] - T0, 1) / 1e6)   # us -> TFLOPS
        return statistics.median(vals)

    P = {"bf16": ceil_tflops(warm + rot, "bf16"),
         "w8a16": ceil_tflops(warm + rot, "w8a16"),
         "w8a8": ceil_tflops(warm + rot, "w8a8_mm")}

    # --- act-quant overhead: w8a8 - w8a8_mm = t0q + M*K*3 / BW_q ---
    import numpy as np
    xs, ys = [], []
    for name, K, N, M, r, s in warm + rot:
        gap = r["w8a8"] - r["w8a8_mm"]
        if gap > 0:
            xs.append(M * K * 3 / 1e6)   # MB moved by quant kernel (read bf16 2B + write 1B)
            ys.append(gap)
    A = np.vstack([np.ones(len(xs)), xs]).T
    (t0q, inv_bwq), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    BW_q = 1 / inv_bwq if inv_bwq > 0 else float("inf")

    return BW, P, (t0q, BW_q)


def predict(BW, P, Q, name, K, N, M, path, mode, s):
    mb = s["wt_mb_bf16"] if path == "bf16" else s["wt_mb_fp8"]
    fam = path
    resident = (mb < C_EFF_MB) and mode == "warm"
    bw = BW[(fam, "l2" if resident else "hbm")] or BW[(fam, "hbm")]
    t_mem = mb / bw
    t_comp = 2 * M * K * N / (P[path] * 1e6)
    t = T0 + max(t_mem, t_comp)
    if path == "w8a8":
        t0q, bwq = Q
        t += t0q + (M * K * 3 / 1e6) / bwq
    return t


def main():
    BW, P, Q = fit()
    print("== fitted parameters ==")
    for k, v in BW.items():
        print(f"  BW {k}: {v and round(v,2)} TB/s")
    for k, v in P.items():
        print(f"  P  {k}: {round(v,1)} TFLOPS")
    print(f"  act-quant: t0q={Q[0]:.2f} us + bytes/{Q[1]:.2f} TB/s")

    meas_key = {"bf16": "bf16", "w8a16": "w8a16", "w8a8": "w8a8"}
    errs = {p: [] for p in meas_key}
    xover = {}
    for mode in ("warm", "rotated"):
        for name, K, N, M, r, s in cells(mode):
            for p, mk in meas_key.items():
                pred = predict(BW, P, Q, name, K, N, M, p, mode, s)
                errs[p].append(abs(pred - r[mk]) / r[mk])

    print("\n== MAPE over all cells (5 shapes x 9 M x 2 modes) ==")
    for p, e in errs.items():
        print(f"  {p:7s}: {100*sum(e)/len(e):5.1f}%   (n={len(e)})")

    print("\n== W8A16 crossover (quant stops winning), measured vs predicted, warm ==")
    for s in load("warm"):
        K, N = SHAPE_DIMS[s["shape"]]
        meas = next((r["M"] for r in s["rows"] if r["w8a16_vs_bf16"] < 1.0), None)
        pred = None
        for M in range(1, 4097):
            pw = predict(BW, P, Q, s["shape"], K, N, M, "w8a16", "warm", s)
            pb = predict(BW, P, Q, s["shape"], K, N, M, "bf16", "warm", s)
            if pw >= pb:
                pred = M
                break
        print(f"  {s['shape']:10s} measured M*~{meas}   predicted M*~{pred}")

    out = {
        "fit": {"t0_us": T0, "c_eff_mb": C_EFF_MB,
                "bw_tbs": {f"{k[0]}_{k[1]}": v for k, v in BW.items()},
                "ceilings_tflops": P,
                "act_quant": {"t0q_us": round(float(Q[0]), 2), "bw_tbs": round(float(Q[1]), 2)}},
        "mape_pct": {p: round(100*sum(e)/len(e), 1) for p, e in errs.items()},
        "model": "t = t0 + max(W/BW_tier(operand,mode), F/P_path) + Q_actquant; "
                 "tier gate: quantized-weight size vs C_eff AND warm-mode only",
    }
    with open(os.path.join(_D, "carm_dense_fit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved carm_dense_fit.json")


if __name__ == "__main__":
    main()
