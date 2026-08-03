"""Residency-aware GDN decode kernel — the 2.7x-headroom claim, tested.

`bench_fla_gdn.py` measured fla's fused_recurrent_gated_delta_rule decode
at a flat 2.3-2.4 TB/s (below HBM streaming rate) from 8 to 160 MB of
state: the production kernel is L2-blind, and the capacity gate predicts a
~2.7x window (bw_l2 6.3 vs 2.35 achieved) at below-gate footprints.

This file is the existence proof attempt: a Triton gated-delta-rule decode
step in ONE fused pass over the state,

    S <- exp(g) * S
    u  = k^T S                       (rank-1 delta readout)
    S <- S + beta * k (x) (v - u)    (delta-rule update)
    o  = q^T S

so state traffic is exactly read-S + write-S (2x footprint), with q/k/v/g
per (batch, head) scalars/vectors held in registers. Grid = (B*H, DV/BV).

Success criterion (pre-registered): warm below-gate BW well above the fla
floor and above HBM rate — approaching the L2 tier — with the advantage
collapsing toward the fla/HBM level above C_eff. Failure (staying at
~2.4 TB/s) would say the floor is not kernel-structural, which is also a
finding.

Run inside ~/fla-env (triton 3.3): correctness vs a pure-torch reference
of the same formula, then the warm/rotated sweep mirroring bench_fla_gdn.
"""

import json
import os
import statistics

import torch
import triton
import triton.language as tl

_D = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
DK = DV = 128
H = 16
STATE_B = DK * DV * 4


@triton.jit
def _gdn_step(S, Q, K, V, G, BETA, O,
              DKc: tl.constexpr, DVc: tl.constexpr, BV: tl.constexpr):
    pid = tl.program_id(0)          # flat (b*h) index
    pv = tl.program_id(1)           # v-block index
    offk = tl.arange(0, DKc)
    offv = pv * BV + tl.arange(0, BV)
    sp = S + pid * DKc * DVc + offk[:, None] * DVc + offv[None, :]
    s = tl.load(sp)                                        # [DK, BV]
    k = tl.load(K + pid * DKc + offk)                      # [DK]
    q = tl.load(Q + pid * DKc + offk)
    v = tl.load(V + pid * DVc + offv)                      # [BV]
    g = tl.load(G + pid)
    beta = tl.load(BETA + pid)
    s = s * tl.exp(g)
    u = tl.sum(s * k[:, None], 0)                          # k^T S  [BV]
    s = s + beta * k[:, None] * (v - u)[None, :]
    tl.store(sp, s)
    o = tl.sum(s * q[:, None], 0)
    tl.store(O + pid * DVc + offv, o)


def gdn_step(S, q, k, v, g, beta, o, BV=64, num_warps=4):
    n = S.shape[0]
    _gdn_step[(n, DV // BV)](S, q, k, v, g, beta, o,
                             DKc=DK, DVc=DV, BV=BV, num_warps=num_warps)


def torch_ref(S, q, k, v, g, beta):
    S = S * torch.exp(g)[:, None, None]
    u = torch.einsum("nk,nkv->nv", k, S)
    S = S + beta[:, None, None] * torch.einsum("nk,nv->nkv", k, v - u)
    o = torch.einsum("nk,nkv->nv", q, S)
    return S, o


def graph_time_us(fn, ni=10, nr=30):
    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(ni):
            fn(i)
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


def correctness():
    n = 64
    S0 = torch.randn(n, DK, DV, dtype=torch.float32, device=DEV)
    q = torch.randn(n, DK, dtype=torch.float32, device=DEV)
    k = torch.nn.functional.normalize(
        torch.randn(n, DK, dtype=torch.float32, device=DEV), dim=-1)
    v = torch.randn(n, DV, dtype=torch.float32, device=DEV)
    g = torch.full((n,), -0.1, dtype=torch.float32, device=DEV)
    beta = torch.rand(n, dtype=torch.float32, device=DEV)
    S_ref, o_ref = torch_ref(S0.clone(), q, k, v, g, beta)
    S_t = S0.clone()
    o_t = torch.empty(n, DV, dtype=torch.float32, device=DEV)
    gdn_step(S_t, q, k, v, g, beta, o_t)
    es = (S_t - S_ref).norm() / S_ref.norm()
    eo = (o_t - o_ref).norm() / o_ref.norm()
    print(f"correctness: state rel-err {es:.2e}  out rel-err {eo:.2e}")
    assert es < 1e-5 and eo < 1e-5
    return float(es), float(eo)


def one(b, n_cop, bv, nw):
    n = b * H
    S = [torch.randn(n, DK, DV, dtype=torch.float32, device=DEV)
         for _ in range(n_cop)]
    q = torch.randn(n, DK, dtype=torch.float32, device=DEV)
    k = torch.nn.functional.normalize(
        torch.randn(n, DK, dtype=torch.float32, device=DEV), dim=-1)
    v = torch.randn(n, DV, dtype=torch.float32, device=DEV)
    g = torch.full((n,), -0.1, dtype=torch.float32, device=DEV)
    beta = torch.rand(n, dtype=torch.float32, device=DEV)
    o = torch.empty(n, DV, dtype=torch.float32, device=DEV)

    def step(i):
        gdn_step(S[i % n_cop], q, k, v, g, beta, o, BV=bv, num_warps=nw)

    us = graph_time_us(step)
    del S, q, k, v, g, beta, o
    torch.cuda.empty_cache()
    return us


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    es, eo = correctness()

    # config search at one below-gate and one above-gate size
    print("config search (warm us):")
    best = None
    for bv in (32, 64, 128):
        for nw in (2, 4, 8):
            try:
                t_lo = one(16, 1, bv, nw)   # 16 MB
                t_hi = one(96, 1, bv, nw)   # 96 MB
            except Exception as ex:
                print(f"  BV={bv} nw={nw}: {type(ex).__name__}")
                continue
            print(f"  BV={bv} nw={nw}: 16MB {t_lo:.2f}  96MB {t_hi:.2f}")
            if best is None or t_lo < best[0]:
                best = (t_lo, bv, nw)
    _, BV, NW = best
    print(f"chosen: BV={BV} num_warps={NW}")

    print(f"{'B':>5} {'MB':>7} {'warm us':>9} {'warm TB/s':>9} "
          f"{'rot us':>9} {'rot TB/s':>9} {'rot/warm':>8}")
    rows = []
    for b in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 160]:
        fp = b * H * STATE_B
        mb = fp / 1048576
        warm = one(b, 1, BV, NW)
        n_cop = max(2, int(96 * 1048576 / fp) + 1)
        rot = one(b, n_cop, BV, NW)
        bw = lambda u: 2 * fp / (u * 1e-6) / 1e12
        r = {"B": b, "state_mb": round(mb, 1),
             "warm_us": round(warm, 3), "rot_us": round(rot, 3),
             "warm_bw_tbs": round(bw(warm), 3), "rot_bw_tbs": round(bw(rot), 3),
             "rot_over_warm": round(rot / warm, 3)}
        rows.append(r)
        print(f"{b:>5} {mb:>7.1f} {warm:>9.2f} {r['warm_bw_tbs']:>9.2f} "
              f"{rot:>9.2f} {r['rot_bw_tbs']:>9.2f} {r['rot_over_warm']:>8.2f}")

    res = {
        "experiment": "residency_aware_gdn_kernel",
        "gpu": gpu, "torch": torch.__version__,
        "triton": triton.__version__,
        "config": {"BV": BV, "num_warps": NW, "H": H, "dk": DK, "dv": DV,
                   "state_dtype": "fp32"},
        "correctness_rel_err": {"state": es, "out": eo},
        "traffic_model": "2x state per step (single fused read+update+write)",
        "fla_reference": "results_fla_gdn_h100.json (2.3-2.4 TB/s flat)",
        "rows": rows,
    }
    with open(os.path.join(_D, "results_gdn_l2_kernel_h100.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_gdn_l2_kernel_h100.json")


if __name__ == "__main__":
    main()
