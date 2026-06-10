"""
Graph-timed W8A8 vs cuBLAS FP16 vs Triton W4A16 for MLA reconstruction BMMs.

Two sweeps, both CUDA-graph timed (20 launches/graph, median of 50 replays),
matching the validation methodology:

  1. batch-size sweep at the MLA shape (H=128, K=128, N=512; 16 MB fp16 weights)
  2. weight-size sweep at bs=1 (d_lora 256..4096; 8..128 MB) across the
     L2 residency cliff

W8A8 is timed end-to-end (dynamic activation quant kernel + INT8 BMM), since
that is what a deployment pays per launch.

Outputs: results_w8a8.json
"""

import json
import os
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
sys.path.insert(0, os.path.join(_D, ".."))

from w8a8_bmm import quantize_acts_w8, quantize_weights_w8, w8a8_bmm  # noqa: E402
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4  # noqa: E402

H, K = 128, 128
N_GRAPH = 20


def graph_med_us(fn, n_inner=N_GRAPH, reps=50):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000


def bench_point(bs, d_lora):
    a = torch.randn(H, bs, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, d_lora, dtype=torch.float16, device="cuda") / 8

    # FP16 cuBLAS
    for _ in range(10):
        torch.bmm(a, w)
    torch.cuda.synchronize()
    fp16_us = graph_med_us(lambda: torch.bmm(a, w))

    # W8A8 (act-quant + INT8 BMM, static buffers for graph capture)
    wq, ws = quantize_weights_w8(w)
    qbuf = torch.empty_like(a, dtype=torch.int8)
    sbuf = torch.empty(H * bs, dtype=torch.float32, device="cuda")
    obuf = torch.empty(H, bs, d_lora, dtype=torch.float16, device="cuda")

    def w8a8_call():
        quantize_acts_w8(a, qbuf, sbuf)
        w8a8_bmm(qbuf, wq, sbuf, ws, obuf)

    for _ in range(10):
        w8a8_call()
    torch.cuda.synchronize()
    w8a8_us = graph_med_us(w8a8_call)

    # W8A8 BMM only (upper bound when act-quant is fused upstream)
    w8a8_mm_us = graph_med_us(lambda: w8a8_bmm(qbuf, wq, sbuf, ws, obuf))

    # W4A16 (existing kernel)
    wp, sc = quantize_weights_int4(w)
    for _ in range(10):
        batched_int4_gemm(a, wp, sc, K)
    torch.cuda.synchronize()
    w4a16_us = graph_med_us(lambda: batched_int4_gemm(a, wp, sc, K))

    # correctness
    ref = torch.bmm(a, w).float()
    quantize_acts_w8(a, qbuf, sbuf)
    rel = ((w8a8_bmm(qbuf, wq, sbuf, ws, obuf).float() - ref).norm() / ref.norm()).item()

    flops = 2 * H * bs * K * d_lora
    row = {
        "bs": bs, "d_lora": d_lora,
        "weight_mb": H * K * d_lora * 2 / 2**20,
        "fp16_us": round(fp16_us, 3),
        "w8a8_us": round(w8a8_us, 3),
        "w8a8_mm_us": round(w8a8_mm_us, 3),
        "w4a16_us": round(w4a16_us, 3),
        "w8a8_speedup": round(fp16_us / w8a8_us, 3),
        "rel_err": round(rel, 5),
        "fp16_tflops": round(flops / fp16_us / 1e6, 2),
        "w8a8_tflops": round(flops / w8a8_us / 1e6, 2),
    }
    del a, w, wq, qbuf, sbuf, obuf, wp, sc
    torch.cuda.empty_cache()
    return row


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    print("\n== batch-size sweep @ MLA shape (d_lora=512, 16 MB) ==")
    bs_rows = []
    for bs in [1, 4, 16, 64, 128, 256, 512]:
        r = bench_point(bs, 512)
        bs_rows.append(r)
        print(f"bs={bs:4d}  fp16={r['fp16_us']:8.2f}  w8a8={r['w8a8_us']:8.2f} "
              f"(mm-only {r['w8a8_mm_us']:7.2f})  w4a16={r['w4a16_us']:8.2f}  "
              f"speedup={r['w8a8_speedup']:5.2f}x  err={r['rel_err']:.4f}")
    out["bs_sweep"] = bs_rows

    print("\n== weight-size sweep @ bs=1 (across L2 cliff) ==")
    sz_rows = []
    for d in [256, 512, 1024, 1536, 2048, 3072, 4096]:
        r = bench_point(1, d)
        sz_rows.append(r)
        print(f"d_lora={d:5d} ({r['weight_mb']:6.1f} MB)  fp16={r['fp16_us']:7.2f}  "
              f"w8a8={r['w8a8_us']:7.2f}  w4a16={r['w4a16_us']:7.2f}  "
              f"speedup={r['w8a8_speedup']:5.2f}x")
    out["size_sweep"] = sz_rows

    path = os.path.join(_D, "results_w8a8.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
