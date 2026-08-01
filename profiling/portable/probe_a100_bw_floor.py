"""
A100 small-kernel BW floor — root-cause probes (2026-08-01, session 6 follow-up).

Observed: bf16 batched-bmm (H=128, T=1, K=128) achieves only ~1.2-1.4 TB/s
effective on A100-SXM4-40GB at bs<=4, on BOTH sides of the capacity gate
(H100: ~8 TB/s on the same cells). Linear in W, intercept ~ t0 -> not a
fixed-cost artifact. Three timing probes to discriminate:

  P1 kernel identity: which cuBLAS kernel + grid does torch.bmm dispatch?
  P2 stream concurrency: run the same bmm on 1/2/4 streams with independent
     weight copies (HBM-regime size so residency does not confound).
     Aggregate BW scaling with streams => memory system NOT saturated by one
     kernel => single-kernel parallelism/latency floor.
  P3 parallelism sweep: fixed weight bytes, H in {32,128,512,1024} (N adjusted)
     => does effective BW rise with available CTA count?

Graph-timed where applicable; lock clocks (1410) before running.
Output: probe_a100_bw_floor.json next to this file.
"""

import json
import os
import statistics

import torch

_D = os.path.dirname(os.path.abspath(__file__))
K = 128


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


def p1_kernel_identity():
    """Profiler: kernel names + grids for T=1 bmm at two weight sizes."""
    out = {}
    for w_mb in (8, 64):
        N = 32 * w_mb
        x = torch.randn(128, 1, K, dtype=torch.float16, device="cuda")
        w = torch.randn(128, K, N, dtype=torch.float16, device="cuda")
        c = torch.empty(128, 1, N, dtype=torch.float16, device="cuda")
        for _ in range(5):
            torch.bmm(x, w, out=c)
        torch.cuda.synchronize()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                torch.bmm(x, w, out=c)
            torch.cuda.synchronize()
        ks = [e for e in prof.key_averages() if e.device_type ==
              torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
        out[f"{w_mb}MB"] = [(e.key[:80], e.count,
                             round(e.self_device_time_total / max(e.count, 1), 1))
                            for e in ks]
        del x, w, c
        torch.cuda.empty_cache()
    return out


def p2_stream_concurrency():
    """1/2/4 concurrent bmm streams, independent 64 MB weights (HBM regime)."""
    w_mb = 64
    N = 32 * w_mb
    res = {}
    for n_streams in (1, 2, 4):
        streams = [torch.cuda.Stream() for _ in range(n_streams)]
        bufs = []
        for _ in range(n_streams):
            x = torch.randn(128, 1, K, dtype=torch.float16, device="cuda")
            w = torch.randn(128, K, N, dtype=torch.float16, device="cuda")
            c = torch.empty(128, 1, N, dtype=torch.float16, device="cuda")
            bufs.append((x, w, c))
        # warm
        for st, (x, w, c) in zip(streams, bufs):
            with torch.cuda.stream(st):
                for _ in range(5):
                    torch.bmm(x, w, out=c)
        torch.cuda.synchronize()
        n_rep, n_inner = 20, 10
        ts = []
        for _ in range(n_rep):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_inner):
                for st, (x, w, c) in zip(streams, bufs):
                    with torch.cuda.stream(st):
                        torch.bmm(x, w, out=c)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        t_us = statistics.median(ts) / n_inner * 1000
        wbytes_total = n_streams * 128 * K * N * 2.0
        res[n_streams] = {
            "t_us_per_round": round(t_us, 2),
            "aggregate_weight_bw_tbs": round(wbytes_total / (t_us * 1e-6) / 1e12, 3),
        }
        del bufs
        torch.cuda.empty_cache()
    return res


def p3_parallelism_sweep():
    """Fixed 32 MB total weights (HBM-adjacent) and fixed 8 MB, vary H."""
    res = {}
    for w_mb in (8, 64):
        row = {}
        for H in (32, 128, 512, 1024):
            N = int(w_mb * 1048576 / (H * K * 2))
            if N < 16:
                continue
            x = torch.randn(H, 1, K, dtype=torch.float16, device="cuda")
            w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
            c = torch.empty(H, 1, N, dtype=torch.float16, device="cuda")
            t = graph_time_us(lambda: torch.bmm(x, w, out=c))
            wbytes = H * K * N * 2.0
            row[H] = {"N": N, "t_us": round(t, 2),
                      "eff_bw_tbs": round(wbytes / (t * 1e-6) / 1e12, 3)}
            del x, w, c
            torch.cuda.empty_cache()
        res[f"{w_mb}MB"] = row
    return res


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    clocks = os.popen(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}  SM clock: {clocks}")
    out = {"gpu": gpu, "sm_clock": clocks,
           "p1_kernel_identity": p1_kernel_identity(),
           "p2_stream_concurrency": p2_stream_concurrency(),
           "p3_parallelism_sweep": p3_parallelism_sweep()}
    path = os.path.join(_D, "probe_a100_bw_floor.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
