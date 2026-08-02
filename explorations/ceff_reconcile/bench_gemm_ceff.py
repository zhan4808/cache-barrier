"""Autoloop A — reconcile C_eff(GEMM context, 36 MB) vs C_eff(re-read, 39.8 MB).

The project carries two effective-capacity numbers for the same H100:
carm_model.json (June, fitted on GEMM sweeps) says 36 MB; the portable
harness's warm re-read cliff (fine-grid, session 9) says onset 39.8 +/- 0.5.
Session 6 established footprint gating: residency is set by W + act + out,
not the weight operand alone. Pre-registered hypothesis:

    W_cliff(T) = C_eff_total - (act + out)(T),  C_eff_total ~= 39.8 MB

i.e. the GEMM weight-operand cliff shifts LEFT with token count by exactly
the activation+output bytes, and the June 36 was a footprint-inclusive
reading. Alternative outcome: W_cliff(T->1) stays well below 39.8 - eps,
which would mean a genuine kernel-context capacity term (GEMM tiling holds
residency worse than pure re-read) — also a finding, and a new local term
for the model.

Method: warm bf16 GEMM x[T,K] @ W[K,N], K=8192, graph-timed; sweep W
30->48 MB at 0.5 MB steps for T in {1, 128, 256, 384}; per T, effective
weight BW = W_bytes / (t - t0); cliff = argmax -> first >5% drop midpoint
(same rule as cliff_finegrain). Then regress W_cliff against (act+out)
bytes: slope -1 and intercept ~39.8 confirms the hypothesis.
"""

import json
import os
import statistics

import torch

_D = os.path.dirname(os.path.abspath(__file__))
DEV, DT = "cuda", torch.bfloat16
K = 8192
T0_US = 2.33
TS = [1, 128, 256, 384]
W_MB = [18 + 0.5 * i for i in range(45)]  # 18..40


def graph_time_us(fn, n_inner=10, n_rep=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000


def cell(t, n):
    x = torch.randn(t, K, dtype=DT, device=DEV)
    w = torch.randn(K, n, dtype=DT, device=DEV)
    o = torch.empty(t, n, dtype=DT, device=DEV)
    us = graph_time_us(lambda: torch.matmul(x, w, out=o))
    del x, w, o
    torch.cuda.empty_cache()
    return us


def detect(pts):
    run_max, m_max = 0.0, None
    for m, b in pts:
        if b > run_max:
            run_max, m_max = b, m
        elif b < 0.95 * run_max:
            return (m_max + m) / 2
    return None


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu}  K={K}  W sweep {W_MB[0]}-{W_MB[-1]} MB @0.5  T {TS}")
    out_rows, cliffs = [], {}
    for t in TS:
        pts = []
        for mb in W_MB:
            n = max(256, int(mb * 1048576 / K / 2 / 128) * 128)
            wmb = K * n * 2 / 1048576
            us = cell(t, n)
            bw = wmb * 1048576 / ((us - T0_US) * 1e-6) / 1e12
            pts.append((wmb, bw))
            out_rows.append({"T": t, "w_mb": round(wmb, 2), "us": round(us, 3),
                             "bw_tbs": round(bw, 3)})
        c = detect(pts)
        act_out_mb = (t * K * 2 + t * int(36 * 1048576 / K / 2) * 2) / 1048576
        cliffs[t] = {"w_cliff_mb": round(c, 2) if c else None,
                     "act_out_mb_at_cliff_scale": round(act_out_mb, 2)}
        print(f"T={t:>4}: W_cliff {cliffs[t]['w_cliff_mb']} MB  "
              f"(act+out ~{act_out_mb:.2f} MB)")

    # regression W_cliff = a + b*(act+out)
    xs = [cliffs[t]["act_out_mb_at_cliff_scale"] for t in TS
          if cliffs[t]["w_cliff_mb"]]
    ys = [cliffs[t]["w_cliff_mb"] for t in TS if cliffs[t]["w_cliff_mb"]]
    if len(xs) >= 2:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
            max(sum((x - mx) ** 2 for x in xs), 1e-9)
        a = my - b * mx
        print(f"fit: W_cliff = {a:.1f} {b:+.2f} * (act+out) MB "
              f"[hypothesis: 39.8 - 1.00x]")
        fit = {"intercept_mb": round(a, 2), "slope": round(b, 3)}
    else:
        fit = None

    res = {
        "experiment": "gemm_context_ceff_reconciliation",
        "gpu": gpu, "torch": torch.__version__, "K": K, "t0_us": T0_US,
        "hypothesis": "W_cliff(T) = 39.8 - (act+out); slope -1 reconciles "
                      "carm 36 vs harness 39.8; flat slope or low intercept "
                      "= genuine GEMM-context capacity term",
        "cliffs": {str(k): v for k, v in cliffs.items()},
        "fit": fit,
        "rows": out_rows,
    }
    with open(os.path.join(_D, "results_gemm_ceff_h100.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_gemm_ceff_h100.json")


if __name__ == "__main__":
    main()
