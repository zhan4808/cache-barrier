"""Fine-grid residency-cliff sweep (session 9 follow-up to the P5 harness).

measure_c_eff's stock grid is 14 points at 0.4-1.5x nominal L2, so its
resolution is half a step = 4.2% of nominal, and the candidate C_eff/nominal
ratios are the SAME rationals on every GPU (the midpoint of cells i..i+1 is
0.4 + 1.1(i+0.5)/13 regardless of card). All three architectures broke in
the same cell, which makes the reported 0.780/0.780/0.781 agree to three
decimals by construction. This sweep re-measures the cliff at 0.5 MB steps
to establish the ratio at real precision on whatever card it runs on.

Same primitive (warm re-read, graph-timed), same detection rule (first size
>5% below the running max; C_eff = midpoint of argmax and that size), three
independent repeats to bound run-to-run jitter.
"""

import json
import os
import re
import sys

import torch

from measure_params import read_bw_tbs

_D = os.path.dirname(os.path.abspath(__file__))
STEP_MB = 0.5
REPEATS = 3


def detect(pts):
    run_max, m_max = 0.0, None
    for m, b in pts:
        if b > run_max:
            run_max, m_max = b, m
        elif b < 0.95 * run_max:
            return (m_max + m) / 2, m_max, m
    return None, m_max, None


def main():
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    gpu = props.name
    l2_nom_mb = props.L2_cache_size / 1048576
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    # coarse grid bounds the break to one cell; sweep that cell +/- one cell
    lo, hi = 0.55 * l2_nom_mb, 1.0 * l2_nom_mb
    print(f"GPU: {gpu}  nominal L2 {l2_nom_mb:.1f} MB  sweep {lo:.1f}-{hi:.1f} MB "
          f"@ {STEP_MB} MB, {REPEATS} repeats")

    runs = []
    for rep in range(REPEATS):
        pts = []
        mb = lo
        while mb <= hi + 1e-9:
            bw, _ = read_bw_tbs(mb)
            pts.append((round(mb, 2), round(bw, 3)))
            mb += STEP_MB
        c_eff, m_argmax, m_break = detect(pts)
        runs.append({
            "c_eff_mb": round(c_eff, 2) if c_eff else None,
            "argmax_mb": m_argmax, "break_mb": m_break,
            "ratio": round(c_eff / l2_nom_mb, 4) if c_eff else None,
            "points": pts,
        })
        print(f"rep {rep}: C_eff {runs[-1]['c_eff_mb']} MB  "
              f"ratio {runs[-1]['ratio']}  (argmax {m_argmax}, break {m_break})")

    out = {
        "gpu": gpu,
        "nominal_l2_mb": round(l2_nom_mb, 1),
        "step_mb": STEP_MB,
        "detection": "first point >5% below running max; C_eff = midpoint of argmax and break",
        "coarse_grid_note": "stock measure_c_eff resolution is 0.042x nominal and its "
                            "candidate ratios are grid-identical across GPUs; this sweep "
                            "resolves the ratio to +/-0.25 MB",
        "runs": runs,
        "c_eff_mb_median": sorted(r["c_eff_mb"] for r in runs)[len(runs) // 2],
    }
    out["ratio_median"] = round(out["c_eff_mb_median"] / l2_nom_mb, 4)
    path = os.path.join(_D, f"results_cliff_finegrain_{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"median C_eff {out['c_eff_mb_median']} MB  ratio {out['ratio_median']}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
