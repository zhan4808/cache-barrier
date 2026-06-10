"""Summarize warm NCU CSVs for fused_moe_kernel rows only."""

import csv
import glob
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))


def parse(path: str) -> dict | None:
    by_id: dict[str, dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            kn = r.get("Kernel Name", "")
            if "fused_moe_kernel" not in kn:
                continue
            kid = r["ID"]
            by_id.setdefault(kid, {})
            by_id[kid][r["Metric Name"]] = float(r["Metric Value"])
    if not by_id:
        return None
    m = max(by_id.values(), key=lambda x: x.get("gpu__time_duration.sum", 0))
    hit = m.get("lts__t_sectors_op_read_lookup_hit.sum", 0)
    miss = m.get("lts__t_sectors_op_read_lookup_miss.sum", 0)
    return {
        "us": m.get("gpu__time_duration.sum", 0) / 1e3,
        "dram_pct": m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0),
        "sm_pct": m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0),
        "l2_hit_pct": 100 * hit / (hit + miss) if hit + miss else 0,
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_D, "ncu_warm")
    for path in sorted(glob.glob(os.path.join(d, "T*"))):
        s = parse(path)
        base = os.path.basename(path)
        if s is None:
            print(f"{base}: no fused_moe_kernel rows")
        else:
            print(f"{base}: us={s['us']:.1f} dram={s['dram_pct']:.1f}% sm={s['sm_pct']:.1f}% l2={s['l2_hit_pct']:.0f}%")


if __name__ == "__main__":
    main()
