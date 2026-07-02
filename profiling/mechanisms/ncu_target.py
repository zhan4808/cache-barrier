"""NCU profiling target — Mechanism D. Runs ONE kernel family a few times.

Usage: ncu --profile-from-start off ... python ncu_target.py <path> <M>
Paths: bf16 | w8a8_pt | marlin | triton_blk
Uses torch.cuda.profiler start/stop to scope collection to the warm region.
"""

import sys

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch, marlin_make_workspace_new, apply_fp8_marlin_linear,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8, w8a8_triton_block_scaled_mm,
)

DEV, DT = "cuda", torch.bfloat16
K, N, BLK = 5120, 6144, 128
path, M = sys.argv[1], int(sys.argv[2])

g = torch.Generator(device=DEV).manual_seed(0)
w = torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5
x = torch.randn(M, K, device=DEV, dtype=DT, generator=g) / 32

if path == "bf16":
    fn = lambda: torch.mm(x, w.t())
elif path == "w8a8_pt":
    wq, ws = ops.scaled_fp8_quant(w)
    xq, xs = ops.scaled_fp8_quant(x)
    fn = lambda: ops.cutlass_scaled_mm(xq, wq.t(), xs, ws, DT)
elif path == "marlin":
    _, qw, s = marlin_quant_fp8_torch(w, -1)
    wsp = marlin_make_workspace_new(DEV)
    fn = lambda: apply_fp8_marlin_linear(x, qw, s, wsp, N, K, None)
elif path == "triton_blk":
    sys.path.insert(0, "/home/ubuntu/robert-nfs/cache-barrier-project/repos/"
                       "cache-barrier/profiling/mechanisms")
    from bench_block_fp8 import block_quant_weight
    bq, bs = block_quant_weight(w)
    xq, xs = per_token_group_quant_fp8(x, BLK)
    fn = lambda: w8a8_triton_block_scaled_mm(xq, bq, xs, bs, [BLK, BLK], DT)
else:
    raise SystemExit(f"unknown path {path}")

for _ in range(10):   # warm (incl. autotune)
    fn()
torch.cuda.synchronize()
torch.cuda.profiler.start()
for _ in range(3):
    fn()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"done {path} M={M}")
