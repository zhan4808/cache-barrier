"""Autoloop B — REAL fused GDN kernel (fla) residency vs the gate.

Upgrades D1's emulated lower bound: fla's fused_recurrent_gated_delta_rule
is the production-family Triton kernel for Gated DeltaNet decode (the
Qwen3-Next / Kimi-Linear operator class). State is fp32 [B, H, dk, dv] =
64 KB/head at dk=dv=128. Decode step: kernel reads the full state, applies
the gated rank-1 update, writes it back — per-step traffic ~2x state (+
output read of S folded into the same pass; the fused kernel's traffic
model is 2x, unlike the 3x two-op emulation).

Sweep total state footprint across the H100 gate (fine-grid onset 39.8 MB):
warm (same state tensor each step — real decode) vs rotated copies (forced
far field). Run inside ~/fla-env (torch 2.7 + fla 0.5.2).
"""

import json
import os
import statistics

import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

_D = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
DK = DV = 128
H = 16                      # heads per request (Qwen3-Next-like)
GRAPH_INNER = 10
GRAPH_REPS = 30
STATE_B = DK * DV * 4       # fp32


def graph_time_us(fn):
    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(GRAPH_INNER):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(GRAPH_REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / GRAPH_INNER * 1000


def one(b, n_cop):
    q = torch.randn(b, 1, H, DK, dtype=torch.bfloat16, device=DEV)
    k = torch.nn.functional.normalize(
        torch.randn(b, 1, H, DK, dtype=torch.bfloat16, device=DEV), dim=-1)
    v = torch.randn(b, 1, H, DV, dtype=torch.bfloat16, device=DEV)
    g_ = torch.full((b, 1, H), -0.1, dtype=torch.float32, device=DEV)
    beta = torch.rand(b, 1, H, dtype=torch.bfloat16, device=DEV)
    S = [torch.randn(b, H, DK, DV, dtype=torch.float32, device=DEV)
         for _ in range(n_cop)]

    def step(i):
        fused_recurrent_gated_delta_rule(
            q, k, v, g=g_, beta=beta, initial_state=S[i % n_cop],
            output_final_state=True)

    us = graph_time_us(step)
    del S, q, k, v, g_, beta
    torch.cuda.empty_cache()
    return us


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu}  fla fused_recurrent_gated_delta_rule  "
          f"H={H} dk=dv={DK} fp32 state ({H * STATE_B // 1024} KB/request)")
    print(f"{'B':>5} {'MB':>7} {'warm us':>9} {'warm TB/s':>9} "
          f"{'rot us':>9} {'rot/warm':>8}")
    rows = []
    for b in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 160]:
        fp = b * H * STATE_B
        mb = fp / 1048576
        warm = one(b, 1)
        n_cop = max(2, int(96 * 1048576 / fp) + 1)
        rot = one(b, n_cop)
        bw = 2 * fp / (warm * 1e-6) / 1e12
        r = {"B": b, "state_mb": round(mb, 1), "warm_us": round(warm, 3),
             "rot_us": round(rot, 3), "warm_bw_2x_tbs": round(bw, 3),
             "rot_copies": n_cop, "rot_over_warm": round(rot / warm, 3)}
        rows.append(r)
        print(f"{b:>5} {mb:>7.1f} {warm:>9.2f} {bw:>9.2f} {rot:>9.2f} "
              f"{r['rot_over_warm']:>8.2f}")

    res = {
        "experiment": "fla_gdn_fused_recurrent_state_residency",
        "gpu": gpu, "torch": torch.__version__,
        "kernel": "fla 0.5.2 fused_recurrent_gated_delta_rule",
        "H": H, "dk": DK, "dv": DV, "state_dtype": "fp32",
        "traffic_model": "2x state per step (fused read+update+write)",
        "timing": f"CUDA graphs, {GRAPH_INNER}/graph, median of {GRAPH_REPS}",
        "prediction": "warm advantage below ~40 MB, gone above ~48; B* knee "
                      "at C_eff/(H*64KB) ~= 40 requests at H=16",
        "rows": rows,
    }
    with open(os.path.join(_D, "results_fla_gdn_h100.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved results_fla_gdn_h100.json")


if __name__ == "__main__":
    main()
