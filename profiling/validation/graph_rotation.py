"""Graph-timed weight rotation: true kernel time for FP16/INT4 at fixed 16MB
shape with L2 residency intact (1 copy) vs destroyed (6 copies = 96MB)."""
import json
import statistics
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4

H, K_DIM, BS, D_LORA = 128, 128, 1, 512
N_INNER = 24

def timed_graph(build):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build()
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
    return statistics.median(ts) / N_INNER * 1000  # us/kernel

out = {}
x = torch.randn(H, BS, K_DIM, dtype=torch.float16, device="cuda")
for ncopies in [1, 2, 3, 6]:
    ws = [torch.randn(H, K_DIM, D_LORA, dtype=torch.float16, device="cuda") for _ in range(ncopies)]
    for w in ws:
        torch.bmm(x, w)
    torch.cuda.synchronize()

    def build():
        for i in range(N_INNER):
            torch.bmm(x, ws[i % ncopies])

    out[f"fp16_copies{ncopies}"] = round(timed_graph(build), 3)
    del ws
    torch.cuda.empty_cache()

for ncopies in [1, 6]:
    packs = []
    for _ in range(ncopies):
        w = torch.randn(H, K_DIM, D_LORA, dtype=torch.float16, device="cuda")
        packs.append(quantize_weights_int4(w))
        del w
    for wp, sc in packs:
        batched_int4_gemm(x, wp, sc, K_DIM)
    torch.cuda.synchronize()

    def build():
        for i in range(N_INNER):
            wp, sc = packs[i % ncopies]
            batched_int4_gemm(x, wp, sc, K_DIM)

    out[f"int4_copies{ncopies}"] = round(timed_graph(build), 3)
    del packs
    torch.cuda.empty_cache()

print(json.dumps(out, indent=2))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_rotation.json"), "w") as f:
    json.dump(out, f, indent=2)
