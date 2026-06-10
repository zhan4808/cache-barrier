"""
Causal diagnostics for the 'L2 cache barrier' claim in cache-barrier.

Claim under test: in the timed benchmark loop, FP16 cuBLAS bmm weights (16 MB)
are served from L2 at ~12 TB/s, which is why (a) latency is flat from 8-32 MB
and (b) INT4's HBM savings don't help.

Diagnostics (all timing-only, no NCU artifacts):

  D1. Launch-overhead floor: time trivially small bmm to measure the fixed
      per-iteration cost of the eventing methodology.

  D2. Weight rotation at FIXED shape (d_lora=512, 16 MB): cycle through K
      independent weight copies per iteration. K=1 reproduces the original
      'warm' loop; K=8 gives a 128 MB working set that cannot be L2-resident,
      with the exact same kernel, shape, and launch pattern.
        - If L2 residency explains FP16's speed, K>=4 should add roughly
          16MB/3.35TB/s - 16MB/12TB/s = ~3.4 us per iteration.
        - INT4 rotation is the control (4 MB packed; 8 copies = 32 MB, still
          fits L2 -> also run K up to 32).

  D3. CUDA-graph timing of the size sweep: capture N back-to-back bmm calls
      in a graph, replay -> removes per-launch CPU overhead so the true
      data-path time is visible. Slope of latency vs bytes gives the actual
      serving bandwidth in the flat region.

  D4. Eviction intervention, absolute-delta version, with a no-op control:
      warm vs evict-256MB vs evict-8MB (too small to displace 16MB of weights
      fully) for FP16 and INT4. Reports ABSOLUTE deltas; if FP16 and INT4
      deltas are similar in microseconds, the paper's relative-percentage
      asymmetry argument is an artifact of different baselines.
"""

import json
import statistics
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4

H, K_DIM = 128, 128
DEV = "cuda"


def time_loop(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def make_fp16(bs, d_lora):
    x = torch.randn(H, bs, K_DIM, dtype=torch.float16, device=DEV)
    w = torch.randn(H, K_DIM, d_lora, dtype=torch.float16, device=DEV)
    return x, w


# ── D1: launch overhead floor ────────────────────────────────────────────────
def d1_floor():
    out = {}
    for d_lora in [16, 64, 128]:
        x, w = make_fp16(1, d_lora)
        out[f"fp16_d{d_lora}"] = time_loop(lambda: torch.bmm(x, w))
    # tiny int4 launch as well
    x, w = make_fp16(1, 64)
    wp, sc = quantize_weights_int4(w)
    out["int4_d64"] = time_loop(lambda: batched_int4_gemm(x, wp, sc, K_DIM))
    return out


# ── D2: weight rotation ──────────────────────────────────────────────────────
def d2_rotation():
    d_lora = 512  # 16 MB FP16
    bs = 1
    res = {"fp16": {}, "int4": {}}
    for k in [1, 2, 3, 4, 6, 8]:
        x = torch.randn(H, bs, K_DIM, dtype=torch.float16, device=DEV)
        ws = [torch.randn(H, K_DIM, d_lora, dtype=torch.float16, device=DEV) for _ in range(k)]
        idx = [0]

        def fn():
            torch.bmm(x, ws[idx[0] % k])
            idx[0] += 1

        res["fp16"][k] = round(time_loop(fn), 5)
        del ws
    for k in [1, 4, 8, 16, 32]:
        x = torch.randn(H, bs, K_DIM, dtype=torch.float16, device=DEV)
        packs = []
        for _ in range(k):
            w = torch.randn(H, K_DIM, d_lora, dtype=torch.float16, device=DEV)
            packs.append(quantize_weights_int4(w))
            del w
        idx = [0]

        def fn():
            wp, sc = packs[idx[0] % k]
            batched_int4_gemm(x, wp, sc, K_DIM)
            idx[0] += 1

        res["int4"][k] = round(time_loop(fn), 5)
        del packs
    return res


# ── D3: CUDA graph sweep ─────────────────────────────────────────────────────
def d3_graph_sweep():
    bs = 1
    n_inner = 20  # bmms per graph; amortizes graph-launch cost
    out = {}
    for d_lora in [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]:
        x, w = make_fp16(bs, d_lora)
        for _ in range(10):
            torch.bmm(x, w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_inner):
                torch.bmm(x, w)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        per_kernel_us = statistics.median(times) / n_inner * 1000
        wt_mb = H * K_DIM * d_lora * 2 / 1024 / 1024
        out[d_lora] = {"weight_mb": wt_mb, "us_per_bmm": round(per_kernel_us, 3),
                       "eff_wt_bw_tbs": round(wt_mb / 1024 / (per_kernel_us / 1e6) / 1e12 * 1024**3 / 1e12, 3)}
        del x, w, g
        torch.cuda.empty_cache()
    return out


# ── D4: eviction with absolute deltas + small-evict control ─────────────────
def d4_eviction():
    d_lora, bs = 512, 1
    x, w = make_fp16(bs, d_lora)
    wp, sc = quantize_weights_int4(w)
    bufs = {
        "evict8mb": torch.randn(8 * 1024 * 1024 // 2, dtype=torch.float16, device=DEV),
        "evict64mb": torch.randn(64 * 1024 * 1024 // 2, dtype=torch.float16, device=DEV),
        "evict256mb": torch.randn(256 * 1024 * 1024 // 2, dtype=torch.float16, device=DEV),
    }

    def bench(kernel, cond, warmup=20, iters=200):
        def run():
            if kernel == "fp16":
                torch.bmm(x, w)
            else:
                batched_int4_gemm(x, wp, sc, K_DIM)

        for _ in range(warmup):
            if cond != "warm":
                bufs[cond].add_(0.001)
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            if cond != "warm":
                bufs[cond].add_(0.001)
                torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    res = {}
    for kern in ["fp16", "int4"]:
        res[kern] = {}
        for cond in ["warm", "evict8mb", "evict64mb", "evict256mb"]:
            res[kern][cond] = round(bench(kern, cond), 5)
    return res


if __name__ == "__main__":
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = {}
    print("\n[D1] launch-overhead floor (ms):")
    results["d1_floor"] = d1_floor()
    print(json.dumps(results["d1_floor"], indent=2))

    print("\n[D2] weight rotation at fixed 16MB shape (ms):")
    results["d2_rotation"] = d2_rotation()
    print(json.dumps(results["d2_rotation"], indent=2))

    print("\n[D3] CUDA-graph sweep, per-bmm time (us) and effective weight bandwidth:")
    results["d3_graph"] = d3_graph_sweep()
    print(json.dumps(results["d3_graph"], indent=2))

    print("\n[D4] eviction intervention with controls (ms):")
    results["d4_evict"] = d4_eviction()
    print(json.dumps(results["d4_evict"], indent=2))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
