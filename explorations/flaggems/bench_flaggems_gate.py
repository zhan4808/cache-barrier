"""FlagGems leg — the gate under a second kernel ecosystem (same silicon).

The transfer claim says hardware constants transfer while kernel terms are
implementation properties. Strongest same-silicon test: swap the kernel
ecosystem (cuBLAS nvjet -> FlagGems Triton mm) on THIS H100 and re-measure
the below/above-gate structure at T=1. Predictions: the CLIFF LOCATION
(a hardware property) stays at the GEMM-context capacity (~34 MB, NCU) /
re-read onset 39.8; the achieved BW tiers and any floors (kernel terms)
may differ arbitrarily.
"""
import json
import os
import statistics

import torch
import flag_gems

_D = os.path.dirname(os.path.abspath(__file__))
K = 8192
T = 1
T0 = 2.33


def graph_time_us(fn, ni=10, nr=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ni):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(nr):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / ni * 1000


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    rows = []
    print(f"{'W MB':>6} {'cublas us':>10} {'BW':>6} {'gems us':>9} {'BW':>6}")
    for mb in [8, 16, 24, 28, 32, 34, 36, 38, 40, 44, 48, 56, 64, 80, 96]:
        n = max(256, int(mb * 1048576 / K / 2 / 128) * 128)
        wmb = K * n * 2 / 1048576
        x = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(K, n, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(T, n, dtype=torch.bfloat16, device="cuda")
        cu = graph_time_us(lambda: torch.matmul(x, w, out=o))
        with flag_gems.use_gems():
            gm = graph_time_us(lambda: torch.mm(x, w))
        bw = lambda u: wmb * 1048576 / ((u - T0) * 1e-6) / 1e12
        r = {"w_mb": round(wmb, 1), "cublas_us": round(cu, 3),
             "gems_us": round(gm, 3), "cublas_bw": round(bw(cu), 2),
             "gems_bw": round(bw(gm), 2)}
        rows.append(r)
        print(f"{wmb:>6.1f} {cu:>10.2f} {r['cublas_bw']:>6.2f} "
              f"{gm:>9.2f} {r['gems_bw']:>6.2f}")
        del x, w, o
        torch.cuda.empty_cache()
    json.dump({"gpu": gpu, "K": K, "T": T, "torch": torch.__version__,
               "rows": rows},
              open(os.path.join(_D, "results_flaggems_gate_h100.json"), "w"),
              indent=1)
    print("saved results_flaggems_gate_h100.json")


if __name__ == "__main__":
    main()
