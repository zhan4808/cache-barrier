"""
Footprint-vs-operand gating (2026-08 A100 follow-up).

Question: does weight L2 residency break when the WEIGHT OPERAND crosses
C_eff (the current gate predicate), or when the TOTAL per-launch footprint
(weights + activations + outputs) does?

Motivation, from existing data:
  - H100 gate sweep (results_capacity_gate.json): bf16 below-gate cells break
    down (measured/predicted 1.2-2.2x) exactly where W + act + out crosses
    ~C_eff, while cells with the same W and small T are fine.
  - A100 reduced sweep: w8a8's advantage step-down lands at operand ~20 MB
    < C_eff 31.2, where footprint ~ 25 MB.

Method: bf16 cuBLAS bmm (the family all CARM params were fit on), graph-timed,
clock-locked. Fix W below the operand gate, sweep T so the footprint crosses
C_eff at a W-dependent token count. For each cell report the effective weight
bandwidth under split-memory accounting:

  bw_w_eff = w_bytes / (t_meas - t0 - (act + out) / bw_hbm)

If gating is operand-based, bw_w_eff stays at L2 rate for all T (W fixed
below C_eff). If footprint-based, bw_w_eff collapses toward HBM rate when
W + act + out crosses C_eff, and the collapse point aligns across different
W when plotted against footprint.

Output: results_footprint_gate.json next to this file.
"""

import json
import os
import statistics

import torch

_D = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_D, "..", "carm_model.json")) as f:
    P = json.load(f)

C_EFF_MB = P["effective_l2_capacity_mb"]
BW_L2 = P["bw_l2_gemm_tbs"] * 1e12
BW_HBM = P["bw_hbm_tbs"] * 1e12
T0_US = P["t0_graph_us"]
PEAK = P["peak_tflops"] * 1e12

H, K = 128, 128
W_MB_SWEEP = [8, 16, 24, 28, 32]
T_SWEEP = [1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

GRAPH_INNER = 10
GRAPH_REPS = 30


def graph_time_us(fn, reps=GRAPH_REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_INNER):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / GRAPH_INNER)
    return statistics.median(times)


def sm_clock_loaded(samples=5, iters=60):
    """SM clock sampled while a GEMM loop saturates the card. An idle-time
    nvidia-smi query reads the DVFS floor, not the clock the timed kernels
    actually run at (hundreds of MHz apart on power-limited instances)."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    vals = []
    for _ in range(samples):
        for _ in range(iters):
            a @ b
        v = os.popen("nvidia-smi --query-gpu=clocks.sm "
                     "--format=csv,noheader,nounits").read().strip().splitlines()[0]
        torch.cuda.synchronize()
        vals.append(int(v))
    del a, b
    torch.cuda.empty_cache()
    vals.sort()
    return (f"{vals[0]}-{vals[-1]} MHz sampled under load "
            f"(median {vals[len(vals) // 2]})")


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    clocks = sm_clock_loaded()
    print(f"GPU: {gpu}  SM clock: {clocks}  C_eff {C_EFF_MB} MB")

    results = []
    for w_mb in W_MB_SWEEP:
        N = 32 * w_mb
        print(f"\n=== W={w_mb} MB (N={N}) ===")
        print(f"{'T':>4} {'fp(MB)':>8} {'t(us)':>9} {'bw_w_eff':>9} {'note':>6}")
        for T in T_SWEEP:
            x = torch.randn(H, T, K, dtype=torch.float16, device="cuda") / 4
            w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
            c = torch.empty(H, T, N, dtype=torch.float16, device="cuda")
            t = graph_time_us(lambda: torch.bmm(x, w, out=c))
            del x, w, c
            torch.cuda.empty_cache()

            wb = H * K * N * 2.0
            act = H * T * K * 2.0
            out = H * T * N * 2.0
            fp_mb = (wb + act + out) / 1048576
            flops = 2.0 * H * T * K * N
            t_comp_us = flops / PEAK * 1e6
            t_w_us = t - T0_US - (act + out) / BW_HBM * 1e6
            bw_w_eff = wb / (t_w_us * 1e-6) / 1e12 if t_w_us > 0 else None
            note = "comp?" if t_comp_us > 0.5 * t else ""
            print(f"{T:>4} {fp_mb:>8.1f} {t:>9.2f} "
                  f"{bw_w_eff if bw_w_eff else float('nan'):>9.2f} {note:>6}",
                  flush=True)
            results.append({
                "w_mb": w_mb, "tokens": T, "N": N,
                "t_us": round(t, 3),
                "footprint_mb": round(fp_mb, 1),
                "bw_w_eff_tbs": round(bw_w_eff, 3) if bw_w_eff else None,
                "t_compute_us": round(t_comp_us, 3),
            })

    out = {
        "gpu": gpu, "sm_clock": clocks,
        "H": H, "K": K,
        "c_eff_mb": C_EFF_MB,
        "bw_l2_tbs": BW_L2 / 1e12, "bw_hbm_tbs": BW_HBM / 1e12,
        "t0_graph_us": T0_US,
        "timing": f"CUDA graphs, {GRAPH_INNER} launches/graph, median of {GRAPH_REPS} replays",
        "accounting": "bw_w_eff = w_bytes / (t - t0 - (act+out)/bw_hbm); "
                      "split-memory attribution of all non-weight traffic to HBM",
        "results": results,
    }
    path = os.path.join(_D, "results_footprint_gate.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
