"""Mechanism E — BLOCK_M padding waste / step structure at decode-scale M.

Every GEMM kernel pads M up to its BLOCK_M tile: at M=1, (BLOCK_M-1)/BLOCK_M of
the tile's MMA work is padding. For memory-bound bf16 this waste is hidden
behind the weight stream; for DEQUANT-bound kernels (Marlin) the in-core work is
the bottleneck, so the waste is on the critical path — one reason quant kernels
miss their memory entitlement at decode.

Empirical signature: latency vs M in steps of 1 is a STAIRCASE — flat within a
tile, stepping at BLOCK_M boundaries. Step locations reveal each kernel's
BLOCK_M; step heights show what the marginal tile costs; flatness within the
tile is the padding-waste region.

M = 1..64 step 1 (then 65..129 step 8), q_proj shape, rotated x2.
Output: results_padding_steps_h100.json
"""

import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    marlin_quant_fp8_torch, marlin_make_workspace_new, apply_fp8_marlin_linear,
)

DEV, DT = "cuda", torch.bfloat16
K, N = 5120, 6144
R = 2
MS = list(range(1, 65)) + list(range(65, 130, 8))


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']} — BLOCK_M staircase, q_proj [{K}x{N}], M=1..129\n")
    g = torch.Generator(device=DEV).manual_seed(0)
    ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(R)]
    marl = [marlin_quant_fp8_torch(w, -1)[1:] for w in ws]
    fp8 = []
    for w in ws:
        wq, wsc = ops.scaled_fp8_quant(w)
        fp8.append((wq.t(), wsc))
    wspace = marlin_make_workspace_new(DEV)
    xfull = torch.randn(MS[-1], K, device=DEV, dtype=DT) / 32

    rows = []
    prev = {}
    for M in MS:
        x = xfull[:M].contiguous()
        xq, xs = ops.scaled_fp8_quant(x)
        it = {"i": 0}

        def f_bf():
            torch.mm(x, ws[it["i"] % R].t()); it["i"] += 1

        def f_88():
            wq, wsc = fp8[it["i"] % R]
            ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1

        def f_ml():
            qw, s = marl[it["i"] % R]
            apply_fp8_marlin_linear(x, qw, s, wspace, N, K, None); it["i"] += 1

        r = {"M": M}
        for tag, fn in [("bf16", f_bf), ("w8a8", f_88), ("w8a16", f_ml)]:
            it["i"] = 0
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            r[tag] = round(graph_med_us(fn, reps=20), 2)
        rows.append(r)
        steps = [t for t in ("bf16", "w8a8", "w8a16")
                 if t in prev and r[t] > prev[t] * 1.12]
        if steps or M in (1, 16, 32, 48, 64):
            print(f"  M={M:4d}  bf16={r['bf16']:7.2f}  w8a8={r['w8a8']:7.2f}  "
                  f"w8a16={r['w8a16']:7.2f}" + ("   STEP: " + ",".join(steps) if steps else ""))
        prev = r
    save_json(os.path.join(_D, "results_padding_steps_h100.json"), {
        "experiment": "mechanism_E_blockm_staircase", "gpu": ver["gpu"],
        "shape": {"K": K, "N": N}, "method": f"graph reps=20, rotated x{R}, M step 1",
        "rows": rows})


if __name__ == "__main__":
    main()
