"""
P5 — portable CARM parameter harness (DIRECTION.md §6 P5).

Emits the five CARM hardware parameters for ANY CUDA-capable backend, with no
architecture-specific constants: every sweep size is derived from the device's
nominal L2 (torch device properties), every number is measured, none is a
datasheet value (guardrail 6).

  C_eff       effective LLC residency capacity (warm re-read bandwidth cliff)
  bw_l2       L2-resident streaming read bandwidth (differential slope)
  bw_hbm      DRAM streaming read bandwidth (differential slope)
  peak        achieved fp16 tensor-core TFLOPS on a large square GEMM
  r_dequant   in-core W4A16 dequant throughput, packed bytes/s (differential
              slope at decode shape; requires Triton — skipped if unavailable)
  t0          per-launch fixed cost under CUDA graphs

Timing is CUDA-graph only (guardrail 2). Lock clocks before running.
Output: params_<gpu-slug>.json next to this file.

Portability caveat (honest): "any backend" today means any backend with
torch.cuda semantics + CUDA graphs (NVIDIA, ROCm via hipify, several domestic
accelerators' torch plugins). Backends without graph capture need the eager
floor subtracted instead — measured here as t0_eager for that purpose.
"""

import json
import os
import re
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"


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


def read_bw_tbs(mb, n_inner=12):
    """Achieved warm re-read bandwidth for an mb-sized fp32 buffer (sum-reduce)."""
    x = torch.randn(int(mb * 1048576 // 4), dtype=torch.float32, device=DEV)
    us = graph_time_us(lambda: x.sum(), n_inner=n_inner)
    del x
    torch.cuda.empty_cache()
    return mb * 1048576 / (us * 1e-6) / 1e12, us


def diff_bw_tbs(mb_small, mb_large):
    """Differential slope between two sizes cancels the fixed cost."""
    _, t_s = read_bw_tbs(mb_small)
    _, t_l = read_bw_tbs(mb_large)
    return (mb_large - mb_small) * 1048576 / ((t_l - t_s) * 1e-6) / 1e12


def measure_t0():
    x = torch.randn(8, 1, 64, dtype=torch.float16, device=DEV)
    w = torch.randn(8, 64, 16, dtype=torch.float16, device=DEV)
    return graph_time_us(lambda: torch.bmm(x, w), n_inner=20)


def measure_t0_eager():
    x = torch.randn(8, 1, 64, dtype=torch.float16, device=DEV)
    w = torch.randn(8, 64, 16, dtype=torch.float16, device=DEV)
    for _ in range(50):
        torch.bmm(x, w)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.bmm(x, w)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1000


def measure_peak_tflops():
    best = 0.0
    for m, k, n in ((8192, 8192, 8192), (4096, 16384, 8192), (16384, 8192, 8192)):
        a = torch.randn(m, k, dtype=torch.float16, device=DEV)
        b = torch.randn(k, n, dtype=torch.float16, device=DEV)
        c = torch.empty(m, n, dtype=torch.float16, device=DEV)
        us = graph_time_us(lambda: torch.matmul(a, b, out=c), n_inner=5, n_rep=20)
        best = max(best, 2.0 * m * k * n / (us * 1e-6) / 1e12)
        del a, b, c
        torch.cuda.empty_cache()
    return best


def measure_c_eff(l2_nom_mb, bw_l2, bw_hbm):
    """Residency cliff: warm re-read BW vs buffer size around nominal L2.

    Achieved single-buffer BW is contaminated by the per-launch fixed cost, so
    absolute plateau levels are unreliable — but the residency COLLAPSE is not:
    warm re-read BW rises with size while resident (fixed cost amortizes), then
    breaks downward at the effective capacity. C_eff = midpoint between the
    argmax and the first size whose BW drops >5% below the running max. Pure
    timing, no vendor counters — so it ports. Resolution: half a sweep step.
    """
    lo, hi = 0.4 * l2_nom_mb, 1.5 * l2_nom_mb
    sizes = [lo + i * (hi - lo) / 13 for i in range(14)]
    pts = []
    for mb in sizes:
        bw, _ = read_bw_tbs(mb)
        pts.append((mb, bw))
    c_eff = None
    run_max, m_max = 0.0, None
    for m, b in pts:
        if b > run_max:
            run_max, m_max = b, m
        elif b < 0.95 * run_max:
            c_eff = (m_max + m) / 2
            break
    return c_eff, [(round(m, 1), round(b, 3)) for m, b in pts]


def measure_r_dequant():
    """W4A16 in-core dequant throughput at decode shape (T=1, BLOCK_M=16).

    Differential slope over packed bytes between two L2-resident sizes cancels
    t0 and the (small) activation/output terms. Requires Triton.
    """
    sys.path.insert(0, os.path.join(_D, ".."))
    try:
        from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4
    except Exception as e:  # noqa: BLE001
        return None, f"triton kernel unavailable: {e}"
    H, K = 128, 128
    times = {}
    for n in (256, 512):  # 8 and 16 MB fp16 -> 2 and 4 MB packed
        x = torch.randn(H, 1, K, dtype=torch.float16, device=DEV)
        w = torch.randn(H, K, n, dtype=torch.float16, device=DEV)
        wp, sc = quantize_weights_int4(w)
        fn = lambda: batched_int4_gemm(x, wp, sc, K, BLOCK_M=16)  # noqa: E731
        times[n] = graph_time_us(fn)
        del x, w, wp, sc
        torch.cuda.empty_cache()
    d_packed = 128 * 128 * (512 - 256) // 2
    r = d_packed / ((times[512] - times[256]) * 1e-6) / 1e12
    return r, {str(k): round(v, 3) for k, v in times.items()}


def main():
    torch.manual_seed(0)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    gpu = props.name
    l2_nom_mb = props.L2_cache_size / 1048576
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    print(f"GPU: {gpu}  nominal L2: {l2_nom_mb:.0f} MB  SMs: {props.multi_processor_count}")

    t0 = measure_t0()
    t0_eager = measure_t0_eager()
    bw_l2 = diff_bw_tbs(0.15 * l2_nom_mb, 0.55 * l2_nom_mb)
    big = min(24 * l2_nom_mb, 2048)
    bw_hbm = diff_bw_tbs(big / 4, big)
    peak = measure_peak_tflops()
    c_eff, cliff = measure_c_eff(l2_nom_mb, bw_l2, bw_hbm)
    r_dq, r_dq_detail = measure_r_dequant()

    params = {
        "gpu": gpu,
        "nominal_l2_mb": round(l2_nom_mb, 1),
        "sm_count": props.multi_processor_count,
        "effective_l2_capacity_mb": round(c_eff, 1) if c_eff else None,
        "bw_l2_tbs": round(bw_l2, 3),
        "bw_hbm_tbs": round(bw_hbm, 3),
        "peak_fp16_tflops": round(peak, 1),
        "r_dequant_tbs": round(r_dq, 4) if r_dq else None,
        "t0_graph_us": round(t0, 3),
        "t0_eager_us": round(t0_eager, 3),
        "residency_cliff_points_mb_tbs": cliff,
        "r_dequant_detail_us": r_dq_detail,
        "method": "graph-timed, differential slopes, timing-only residency cliff; "
                  "no vendor counters, no datasheet numbers",
    }
    out = os.path.join(_D, f"params_{slug}.json")
    with open(out, "w") as f:
        json.dump(params, f, indent=1)
    print(json.dumps({k: v for k, v in params.items()
                      if k != "residency_cliff_points_mb_tbs"}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
