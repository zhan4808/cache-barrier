"""Mechanism C — blockwise-scale fp8 (the PRODUCTION format) vs per-tensor.

Production fp8 deployments (DeepSeek-style) use 128x128 weight blocks +
per-token-group-128 activation scales, not the per-tensor scales we measured.
Block scales add a second read stream and per-block epilogue math. If blockwise
costs X% over per-tensor, every W8A8 routing number shifts by X.

Paths at q_proj / gate_up shapes, M in {1,16,64,256,1024,2048}, rotated x2:
  bf16          torch.mm
  w8a8_pt       cutlass per-tensor (mm-only, act pre-quantized)
  w8a8_blk      w8a8_triton_block_scaled_mm, 128x128 weight blocks (mm-only,
                act pre-group-quantized via per_token_group_quant_fp8)
  + deployed variants including their respective act-quant kernels.

Output: results_block_fp8_h100.json
"""

import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8, w8a8_triton_block_scaled_mm,
)

DEV, DT = "cuda", torch.bfloat16
FP8 = torch.float8_e4m3fn
R = 2
BLK = 128
MS = [1, 16, 64, 256, 1024, 2048]
SHAPES = [("q_proj", 5120, 6144), ("gate_up", 5120, 34816)]


def block_quant_weight(w):
    """[N,K] bf16 -> fp8 with [N/BLK, K/BLK] scales (pad-free shapes here)."""
    N, K = w.shape
    wb = w.view(N // BLK, BLK, K // BLK, BLK).permute(0, 2, 1, 3).float()
    amax = wb.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    s = amax / torch.finfo(FP8).max
    q = (wb / s).clamp(torch.finfo(FP8).min, torch.finfo(FP8).max).to(FP8)
    q = q.permute(0, 2, 1, 3).reshape(N, K).contiguous()
    return q, s.view(N // BLK, K // BLK).contiguous().float()


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']} — blockwise (128x128) vs per-tensor fp8\n")
    rows = []
    for name, K, N in SHAPES:
        g = torch.Generator(device=DEV).manual_seed(K + N)
        ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(R)]
        pt = []
        blk = []
        for w in ws:
            wq, wsc = ops.scaled_fp8_quant(w)
            pt.append((wq.t(), wsc))
            bq, bs = block_quant_weight(w)
            blk.append((bq, bs))
        print(f"== {name} [{K}x{N}] ==")
        print(f"{'M':>5} | {'bf16':>8} | {'pt mm':>8} {'blk mm':>8} {'blk/pt':>7} | "
              f"{'pt dep':>8} {'blk dep':>8}")
        for M in MS:
            x = torch.randn(M, K, device=DEV, dtype=DT) / 32
            xq_pt, xs_pt = ops.scaled_fp8_quant(x)
            xq_g, xs_g = per_token_group_quant_fp8(x, BLK)
            it = {"i": 0}

            def f_bf():
                torch.mm(x, ws[it["i"] % R].t()); it["i"] += 1

            def f_pt():
                wq, wsc = pt[it["i"] % R]
                ops.cutlass_scaled_mm(xq_pt, wq, xs_pt, wsc, DT); it["i"] += 1

            def f_blk():
                bq, bs = blk[it["i"] % R]
                w8a8_triton_block_scaled_mm(xq_g, bq, xs_g, bs, [BLK, BLK], DT); it["i"] += 1

            def f_pt_dep():
                a, s = ops.scaled_fp8_quant(x)
                wq, wsc = pt[it["i"] % R]
                ops.cutlass_scaled_mm(a, wq, s, wsc, DT); it["i"] += 1

            def f_blk_dep():
                a, s = per_token_group_quant_fp8(x, BLK)
                bq, bs = blk[it["i"] % R]
                w8a8_triton_block_scaled_mm(a, bq, s, bs, [BLK, BLK], DT); it["i"] += 1

            r = {"shape": name, "M": M}
            for tag, fn in [("bf16", f_bf), ("pt_mm", f_pt), ("blk_mm", f_blk),
                            ("pt_dep", f_pt_dep), ("blk_dep", f_blk_dep)]:
                it["i"] = 0
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                r[tag] = round(graph_med_us(fn), 2)
            r["blk_over_pt_mm"] = round(r["blk_mm"] / r["pt_mm"], 2)
            rows.append(r)
            print(f"{M:>5} | {r['bf16']:>8.1f} | {r['pt_mm']:>8.1f} {r['blk_mm']:>8.1f} "
                  f"{r['blk_over_pt_mm']:>6.2f}x | {r['pt_dep']:>8.1f} {r['blk_dep']:>8.1f}")
        # correctness once
        x = torch.randn(64, K, device=DEV, dtype=DT) / 32
        ref = x @ ws[0].t()
        xq_g, xs_g = per_token_group_quant_fp8(x, BLK)
        got = w8a8_triton_block_scaled_mm(xq_g, blk[0][0], xs_g, blk[0][1], [BLK, BLK], DT)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"  blockwise relerr vs bf16: {rel:.4f}\n")
        rows.append({"shape": name, "blk_relerr": round(rel, 4)})
        del ws, pt, blk
        torch.cuda.empty_cache()
    save_json(os.path.join(_D, "results_block_fp8_h100.json"), {
        "experiment": "mechanism_C_blockwise_scale_overhead", "gpu": ver["gpu"],
        "method": f"graph_med_us; rotated x{R}; blk=128x128 wt + per-token-group-128 act "
                  "(triton w8a8_block kernel, tuned configs if shipped)", "rows": rows})


if __name__ == "__main__":
    main()
