"""
B200 transfer-target sweep (2026-08-02 B200 session, KICKOFF_B200 goal 1).

Same 48-cell grid + timing discipline as remeasure_sweep_graphtimed.py (the
A100 self-consistent target), with one B200-specific extension: four extra
d_lora points {5120, 6144, 8192, 10240} -> W in {160, 192, 256, 320} MB.
Reason: C_eff measured 98.8 MB on this card, so the ORIGINAL grid's largest
cell (128 MB) sits at only 1.30x the gate — inside the soft transition band
(C_hi ~ 1.56 x C_eff = 154 MB transferred from H100). Without the extension
the above-gate regime would have zero clean cells and the regime-separated
MAPE (guardrail 7) could not be reported. The original 12 sizes are kept
unchanged for cross-architecture comparability.

Clock note: clock locking is UNAVAILABLE on this instance (nvidia-smi -lgc
denied even with sudo; -ac deprecated). Sustained-load SM clock band measured
1237-1320 MHz (power-limited DVFS, stable +/-3%). Graph-timed medians of 30.

Output: results_l2_barrier_b200_graphtimed.json next to this file, row format
compatible with transfer_validation.py (fp16_ms / int4_ms in ms).
"""

import json
import os
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, ".."))  # profiling/

from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4  # noqa: E402

H, K = 128, 128
D_LORA_SWEEP = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096,
                5120, 6144, 8192, 10240]  # last four: B200 above-gate extension
BATCH_SIZES = [1, 4, 16, 64]

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


def run_cell(bs, d_lora):
    n = d_lora
    x = torch.randn(H, bs, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, n, dtype=torch.float16, device="cuda") / 8
    c = torch.empty(H, bs, n, dtype=torch.float16, device="cuda")

    t_fp16 = graph_time_us(lambda: torch.bmm(x, w, out=c))

    wp, sc = quantize_weights_int4(w)
    t_int4 = graph_time_us(
        lambda: batched_int4_gemm(x, wp, sc, K, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128))

    del x, w, c, wp, sc
    torch.cuda.empty_cache()
    return t_fp16, t_int4


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
    print(f"GPU: {gpu}  SM clock now: {clocks} "
          f"(lock UNAVAILABLE on this instance; sustained band 1237-1320 MHz)")

    results = []
    for bs in BATCH_SIZES:
        print(f"\n=== bs={bs} ===")
        print(f"{'d_lora':>8} {'Wt MB':>8} {'fp16(us)':>10} {'int4(us)':>10} {'ratio':>7}")
        for d_lora in D_LORA_SWEEP:
            wt_mb = H * K * d_lora * 2 / 1048576
            t_f, t_i = run_cell(bs, d_lora)
            print(f"{d_lora:>8} {wt_mb:>8.1f} {t_f:>10.2f} {t_i:>10.2f} "
                  f"{t_i / t_f:>6.2f}x", flush=True)
            results.append({
                "batch_size": bs,
                "d_lora": d_lora,
                "weight_mb": round(wt_mb, 1),
                "fp16_ms": round(t_f / 1000, 5),
                "int4_ms": round(t_i / 1000, 5),
                "int4_fp16_ratio": round(t_i / t_f, 3),
            })

    out = {
        "gpu": gpu,
        "sm_clock": clocks,
        "clock_lock": "UNAVAILABLE (lgc denied w/ sudo, ac deprecated); "
                      "sustained-load band 1237-1320 MHz, +/-3%",
        "H": H, "D_NOPE": K,
        "d_lora_sweep": D_LORA_SWEEP,
        "timing": f"CUDA graphs, {GRAPH_INNER} launches/graph, median of {GRAPH_REPS} replays",
        "note": "B200 self-consistent transfer target: same GPU as "
                "params_nvidia-b200.json; original A100/H100 grid + 4 above-gate "
                "extension sizes (C_eff 98.8 MB needs W > 154 MB for clean "
                "above-band cells)",
        "results": results,
    }
    path = os.path.join(_D, "results_l2_barrier_b200_graphtimed.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
