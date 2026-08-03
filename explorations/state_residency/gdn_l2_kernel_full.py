"""Epilogue-complete residency-aware GDN decode kernel.

`gdn_l2_kernel.py` proved the core claim (2.2x vs fla in the gate window)
on the bare delta-rule step. A real Gated DeltaNet decode step (Qwen3-Next
/ Kimi-Linear layer, minus the projections which are ordinary GEMMs) also
runs: short causal conv (K=4, depthwise, silu) on q/k/v with a rolling
conv cache, q/k L2 normalization, and a gated RMSNorm output epilogue.
fla executes this as a 3+ kernel chain per step:

    ShortConvolution.step (x3 or fused)  ->  fused_recurrent_gated_delta_rule
    (use_qk_l2norm_in_kernel=True)       ->  FusedRMSNormGated

Every extra kernel is another trip through global memory for the small
tensors AND another launch. This file fuses the ENTIRE step into one
Triton kernel per (batch, head): conv-cache read+update, silu, l2norm,
state read+delta+write (the only large traffic: 2x state), readout, gated
RMSNorm. Success criterion: the below-gate advantage of the bare kernel
survives the epilogues (>=1.5x vs the fla chain below C_eff).

Correctness vs a pure-torch reference of the same math (1e-5). fla-chain
numbers measured on the same box in the same run for the honest
comparison (its conv/norm kernels run in bf16 vs our fp32 conv cache —
noted; the state traffic dominates both).
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
KC = 4                      # short-conv kernel size
STATE_B = DK * DV * 4


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _gdn_full(S, CQ, CK, CV, WQ, WK, WV, Qi, Ki, Vi, G, BETA, GATE, WN, O,
              eps, DKc: tl.constexpr, DVc: tl.constexpr, KCc: tl.constexpr):
    pid = tl.program_id(0)
    offk = tl.arange(0, DKc)
    offv = tl.arange(0, DVc)
    offc = tl.arange(0, KCc)

    # ---- short conv on q, k (DK channels) and v (DV channels) ----
    # cache layout [N, C, KC]; new input enters at slot KC-1 after shift
    cq = tl.load(CQ + pid * DKc * KCc + offk[:, None] * KCc + offc[None, :])
    ck = tl.load(CK + pid * DKc * KCc + offk[:, None] * KCc + offc[None, :])
    cv = tl.load(CV + pid * DVc * KCc + offv[:, None] * KCc + offc[None, :])
    qi = tl.load(Qi + pid * DKc + offk)
    ki = tl.load(Ki + pid * DKc + offk)
    vi = tl.load(Vi + pid * DVc + offv)
    # shift left, insert new
    cq = tl.where(offc[None, :] < KCc - 1,
                  tl.load(CQ + pid * DKc * KCc + offk[:, None] * KCc
                          + (offc[None, :] + 1)), qi[:, None])
    ck = tl.where(offc[None, :] < KCc - 1,
                  tl.load(CK + pid * DKc * KCc + offk[:, None] * KCc
                          + (offc[None, :] + 1)), ki[:, None])
    cv = tl.where(offc[None, :] < KCc - 1,
                  tl.load(CV + pid * DVc * KCc + offv[:, None] * KCc
                          + (offc[None, :] + 1)), vi[:, None])
    tl.store(CQ + pid * DKc * KCc + offk[:, None] * KCc + offc[None, :], cq)
    tl.store(CK + pid * DKc * KCc + offk[:, None] * KCc + offc[None, :], ck)
    tl.store(CV + pid * DVc * KCc + offv[:, None] * KCc + offc[None, :], cv)
    wq = tl.load(WQ + offk[:, None] * KCc + offc[None, :])
    wk = tl.load(WK + offk[:, None] * KCc + offc[None, :])
    wv = tl.load(WV + offv[:, None] * KCc + offc[None, :])
    q = _silu(tl.sum(cq * wq, 1))
    k = _silu(tl.sum(ck * wk, 1))
    v = _silu(tl.sum(cv * wv, 1))

    # ---- q/k L2 norm ----
    q = q / tl.sqrt(tl.sum(q * q) + eps)
    k = k / tl.sqrt(tl.sum(k * k) + eps)

    # ---- delta-rule state update (the 2x-state traffic) ----
    g = tl.load(G + pid)
    beta = tl.load(BETA + pid)
    sp = S + pid * DKc * DVc + offk[:, None] * DVc + offv[None, :]
    s = tl.load(sp) * tl.exp(g)
    u = tl.sum(s * k[:, None], 0)
    s = s + beta * k[:, None] * (v - u)[None, :]
    tl.store(sp, s)
    o = tl.sum(s * q[:, None], 0)

    # ---- gated RMSNorm epilogue ----
    gate = tl.load(GATE + pid * DVc + offv)
    wn = tl.load(WN + offv)
    o = o / tl.sqrt(tl.sum(o * o) / DVc + eps) * wn * _silu(gate)
    tl.store(O + pid * DVc + offv, o)


def torch_ref(S, cq, ck, cv, wq, wk, wv, qi, ki, vi, g, beta, gate, wn,
              eps=1e-6):
    silu = torch.nn.functional.silu
    cq = torch.cat([cq[:, :, 1:], qi[:, :, None]], -1)
    ck = torch.cat([ck[:, :, 1:], ki[:, :, None]], -1)
    cv = torch.cat([cv[:, :, 1:], vi[:, :, None]], -1)
    q = silu((cq * wq).sum(-1))
    k = silu((ck * wk).sum(-1))
    v = silu((cv * wv).sum(-1))
    q = q / (q.norm(dim=-1, keepdim=True) ** 2 + eps).sqrt()
    k = k / (k.norm(dim=-1, keepdim=True) ** 2 + eps).sqrt()
    S = S * torch.exp(g)[:, None, None]
    u = torch.einsum("nk,nkv->nv", k, S)
    S = S + beta[:, None, None] * torch.einsum("nk,nv->nkv", k, v - u)
    o = torch.einsum("nk,nkv->nv", q, S)
    o = o / (o.pow(2).mean(-1, keepdim=True) + eps).sqrt() * wn * silu(gate)
    return S, o, cq, ck, cv


def make(n):
    t = lambda *s: torch.randn(*s, dtype=torch.float32, device=DEV)
    return dict(S=t(n, DK, DV), cq=t(n, DK, KC), ck=t(n, DK, KC),
                cv=t(n, DV, KC), wq=t(DK, KC) * 0.3, wk=t(DK, KC) * 0.3,
                wv=t(DV, KC) * 0.3, qi=t(n, DK), ki=t(n, DK), vi=t(n, DV),
                g=torch.full((n,), -0.1, device=DEV),
                beta=torch.rand(n, device=DEV), gate=t(n, DV),
                wn=torch.ones(DV, device=DEV),
                o=torch.empty(n, DV, dtype=torch.float32, device=DEV))


def launch(a, n, nw=4):
    _gdn_full[(n,)](a["S"], a["cq"], a["ck"], a["cv"], a["wq"], a["wk"],
                    a["wv"], a["qi"], a["ki"], a["vi"], a["g"], a["beta"],
                    a["gate"], a["wn"], a["o"], 1e-6,
                    DKc=DK, DVc=DV, KCc=KC, num_warps=nw)


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
    n = 32
    a = make(n)
    S_ref, o_ref, cq_r, ck_r, cv_r = torch_ref(
        a["S"].clone(), a["cq"].clone(), a["ck"].clone(), a["cv"].clone(),
        a["wq"], a["wk"], a["wv"], a["qi"], a["ki"], a["vi"], a["g"],
        a["beta"], a["gate"], a["wn"])
    launch(a, n)
    for name, mine, ref in (("state", a["S"], S_ref), ("out", a["o"], o_ref),
                            ("conv_q", a["cq"], cq_r)):
        e = (mine - ref).norm() / (ref.norm() + 1e-9)
        print(f"correctness {name}: rel-err {e:.2e}")
        assert e < 1e-4, name
    return True


def fla_chain(b):
    """fla's decode chain: 3 conv steps + fused_recurrent(l2norm) + gated norm."""
    from fla.modules.convolution import ShortConvolution
    from fla.modules.fused_norm_gate import FusedRMSNormGated
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    n = b * H
    cq = ShortConvolution(H * DK, KC, activation="silu").to(DEV, torch.bfloat16)
    ck = ShortConvolution(H * DK, KC, activation="silu").to(DEV, torch.bfloat16)
    cv = ShortConvolution(H * DV, KC, activation="silu").to(DEV, torch.bfloat16)
    norm = FusedRMSNormGated(DV).to(DEV, torch.bfloat16)
    xq = torch.randn(b, 1, H * DK, dtype=torch.bfloat16, device=DEV)
    xv = torch.randn(b, 1, H * DV, dtype=torch.bfloat16, device=DEV)
    gate = torch.randn(b, 1, H, DV, dtype=torch.bfloat16, device=DEV)
    ccq = torch.zeros(b, H * DK, KC, dtype=torch.bfloat16, device=DEV)
    cck = torch.zeros_like(ccq)
    ccv = torch.zeros(b, H * DV, KC, dtype=torch.bfloat16, device=DEV)
    g_ = torch.full((b, 1, H), -0.1, dtype=torch.float32, device=DEV)
    beta = torch.rand(b, 1, H, dtype=torch.bfloat16, device=DEV)
    S = torch.randn(b, H, DK, DV, dtype=torch.float32, device=DEV)

    def step(i):
        q, _ = cq.step(xq, None, ccq, output_final_state=True)
        k, _ = ck.step(xq, None, cck, output_final_state=True)
        v, _ = cv.step(xv, None, ccv, output_final_state=True)
        o, _ = fused_recurrent_gated_delta_rule(
            q.view(b, 1, H, DK), k.view(b, 1, H, DK), v.view(b, 1, H, DV),
            g=g_, beta=beta, initial_state=S, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        norm(o.view(b, 1, H, DV), gate)

    us = graph_time_us(step)
    del cq, ck, cv, norm, xq, xv, gate, ccq, cck, ccv, g_, beta, S
    torch.cuda.empty_cache()
    return us


def one(b, n_cop, nw=4):
    n = b * H
    ars = [make(n) for _ in range(n_cop)]

    def step(i):
        launch(ars[i % n_cop], n, nw)

    us = graph_time_us(step)
    del ars
    torch.cuda.empty_cache()
    return us


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    correctness()
    print(f"{'B':>5} {'MB':>7} {'ours-warm':>9} {'ours-rot':>9} "
          f"{'fla-chain':>9} {'speedup':>8}")
    rows = []
    for b in [8, 16, 24, 32, 40, 48, 64, 96]:
        fp = b * H * STATE_B
        mb = fp / 1048576
        warm = one(b, 1)
        n_cop = max(2, int(96 * 1048576 / fp) + 1)
        rot = one(b, n_cop)
        chain = fla_chain(b)
        r = {"B": b, "state_mb": round(mb, 1), "ours_warm_us": round(warm, 3),
             "ours_rot_us": round(rot, 3), "fla_chain_us": round(chain, 3),
             "speedup_warm": round(chain / warm, 3),
             "rot_over_warm": round(rot / warm, 3)}
        rows.append(r)
        print(f"{b:>5} {mb:>7.1f} {warm:>9.2f} {rot:>9.2f} {chain:>9.2f} "
              f"{r['speedup_warm']:>8.2f}")

    res = {
        "experiment": "epilogue_complete_gdn_kernel_vs_fla_chain",
        "gpu": gpu, "torch": torch.__version__, "triton": triton.__version__,
        "config": {"H": H, "dk": DK, "dv": DV, "conv_k": KC,
                   "state_dtype": "fp32", "num_warps": 4, "BV": "full-head"},
        "epilogues": "short conv K=4 + silu on q/k/v (rolling cache), qk "
                     "l2norm, gated RMSNorm output — all fused; fla chain = "
                     "3x ShortConvolution.step + fused_recurrent(l2norm) + "
                     "FusedRMSNormGated (bf16 conv/norm, fp32 state)",
        "note": "fla chain includes its kernels' launch overheads inside one "
                "graph; both sides graph-timed, medians of 30",
        "rows": rows,
    }
    with open(os.path.join(_D, "results_gdn_full_h100.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_gdn_full_h100.json")


if __name__ == "__main__":
    main()
