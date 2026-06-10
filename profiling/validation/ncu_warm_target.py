"""Steady-state warm loop target for NCU. Usage: ncu_warm_target.py <fp16|int4> <d_lora>"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4

H, K_DIM, BS = 128, 128, 1
kernel, d_lora = sys.argv[1], int(sys.argv[2])

x = torch.randn(H, BS, K_DIM, dtype=torch.float16, device="cuda")
w = torch.randn(H, K_DIM, d_lora, dtype=torch.float16, device="cuda")
wp, sc = quantize_weights_int4(w)

run = (lambda: torch.bmm(x, w)) if kernel == "fp16" else (lambda: batched_int4_gemm(x, wp, sc, K_DIM))

for _ in range(30):  # steady state: weights L2-resident if they fit
    run()
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
for _ in range(5):
    run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
