"""
P5 follow-up — graph-timed re-measure of the transfer-target sweep on the
local A100 (2026-08-01 A100 session).

Why this exists: the in-repo 2026-06 A100 transfer target
(results_l2_barrier_a100_extended.json) is (a) eager-event timed with a
~15.5 us launch floor (guardrail 2 caveat) and (b) from an SXM4-80GB, while
this session's measured hardware constants come from an SXM4-40GB
(HBM2 ~1.56 TB/s vs HBM2e ~1.94 — same GA100 die otherwise). Re-measuring the
same 48-cell grid graph-timed on THIS card makes params and target
self-consistent: same GPU, same timing discipline.

Grid: identical to the 2026-06 sweep — bs in {1,4,16,64} x d_lora in
[256..4096], H=K=128. Kernels: torch.bmm fp16 + batched_int4_gemm
(bench_l2_barrier.py, BLOCK_N=64 as in the original). Timing: the CUDA-graph
timer from profiling/gate/bench_capacity_gate.py (10 launches/graph, median
of 30 replays). Lock clocks before running.

Output: results_l2_barrier_a100_40gb_graphtimed.json next to this file, row
format compatible with transfer_validation.py (fp16_ms / int4_ms in ms).
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
D_LORA_SWEEP = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
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


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    clocks = os.popen(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}  SM clock: {clocks} (lock externally: nvidia-smi -lgc 1410,1410)")

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
        "H": H, "D_NOPE": K,
        "d_lora_sweep": D_LORA_SWEEP,
        "timing": f"CUDA graphs, {GRAPH_INNER} launches/graph, median of {GRAPH_REPS} replays",
        "note": "graph-timed re-measure of the 2026-06 eager-timed 80GB sweep, "
                "on the same GPU the measured CARM params come from (SXM4-40GB)",
        "results": results,
    }
    path = os.path.join(_D, "results_l2_barrier_a100_40gb_graphtimed.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
