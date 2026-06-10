"""CUDA-graph-timed sweep for the INT4 Triton kernel (and FP16 control), bs=1."""
import json
import statistics
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4

H, K_DIM, BS = 128, 128, 1
SWEEP = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
N_INNER = 20

out = {}
for d_lora in SWEEP:
    x = torch.randn(H, BS, K_DIM, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K_DIM, d_lora, dtype=torch.float16, device="cuda")
    wp, sc = quantize_weights_int4(w)

    def run():
        batched_int4_gemm(x, wp, sc, K_DIM)

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N_INNER):
            run()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    us = statistics.median(ts) / N_INNER * 1000
    wt_mb = H * K_DIM * d_lora * 2 / 1024 / 1024
    out[d_lora] = {"weight_mb": wt_mb, "us_per_kernel": round(us, 3)}
    print(d_lora, out[d_lora])
    del x, w, wp, sc, g
    torch.cuda.empty_cache()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_sweep_int4.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved")
