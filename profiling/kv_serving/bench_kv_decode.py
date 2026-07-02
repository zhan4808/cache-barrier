"""KV-cache quantization under serving conditions (Dr. Xiao todo #2).

Question: are KV-cache reads in a real inference service limited by L2 capacity,
and does fp8-KV quantization change that?

Setup mirrors vLLM's production path exactly (v1 flash_attn backend, the
`_vllm_fa3_C::fwd` bucket in Dr. Xiao's Qwen3.6-27B profile): FA3
`flash_attn_varlen_func` decode with a PAGED KV cache (block_size 16), GQA
24 q-heads / 4 kv-heads x head_dim 256. fp8 path stores KV as float8_e4m3fn and
passes k/v_descale of shape (batch, kv_heads) -- identical to
vllm/v1/attention/backends/flash_attn.py (q stays bf16; KV-only quant).

Per-decode-step KV working set = B x ctx x 2 x h_kv x d x bytes:
              bf16        fp8       (C_eff ~= 36 MB)
  1k x B1     4.2 MB      2.1 MB    both L2-resident
  1k x B8    33.6 MB     16.8 MB    bf16 at boundary, fp8 inside
  8k x B1    33.6 MB     16.8 MB    same boundary case
  8k x B8   268 MB      134 MB      streamed
  32k, B32+  GB-scale                streamed
So the L2 question has real testable cells at low batch x short context, and
"fp8 pulls the working set into L2" boundary cells -- same structure as the
dense-weights experiment.

Modes: warm (same cache re-read per launch -- microbenchmark residency) and
--rotate (R cache regions cycled per in-graph launch, R x ws > 2 x C_eff --
the multi-layer/serving eviction control; 16 full-attn layers in the real model).

Amdahl context: full-attention is 2.67% of runtime in Dr. Xiao's profile, so
even infinite KV speedup moves end-to-end < 3% on THIS model. Reported so the
answer is honest at both the operator and model level.

Output: results_kv_decode_<gpu_key>[_rotated].json
"""

import argparse
import math
import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json, gpu_key  # noqa: E402

from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

DEV, DT = "cuda", torch.bfloat16
FP8 = torch.float8_e4m3fn
HQ, HKV, D = 24, 4, 256            # Qwen3.6-27B full-attention GQA
BLK = 16                           # vLLM block size
C_EFF_MB = 36.0
SM_SCALE = D ** -0.5

CONFIGS = [(1024, 1), (1024, 8), (1024, 32), (1024, 128),
           (8192, 1), (8192, 8), (8192, 32),
           (32768, 1), (32768, 8), (32768, 32)]


def ws_mb(ctx, B, bytes_per):
    return B * ctx * 2 * HKV * D * bytes_per / 1e6


def build_cache(ctx, B, n_reg, fp8):
    """One paged cache holding n_reg independent regions; per-region block tables.
    Seeded so the bf16 and fp8 paths quantize the SAME underlying values."""
    bps = ctx // BLK                       # blocks per seq
    nblk = n_reg * B * bps
    g = torch.Generator(device=DEV).manual_seed(ctx * 1000 + B)
    kv = torch.randn(2, nblk, BLK, HKV, D, device=DEV, dtype=DT, generator=g) / 8
    if fp8:
        kvq = kv.to(FP8)
        k_cache, v_cache = kvq.unbind(0)
    else:
        k_cache, v_cache = kv.unbind(0)
    tables = []
    for r in range(n_reg):
        base = r * B * bps
        t = (torch.arange(B, device=DEV).unsqueeze(1) * bps +
             torch.arange(bps, device=DEV).unsqueeze(0) + base).int()
        tables.append(t)
    return k_cache, v_cache, tables, kv


def run_cfg(ctx, B, rotate):
    bytes_bf16, bytes_fp8 = 2, 1
    n_reg = 1
    if rotate:
        n_reg = max(2, math.ceil(2 * C_EFF_MB / ws_mb(ctx, B, 1)) + 1)
        n_reg = min(n_reg, 40)  # cap memory
    q = torch.randn(B, HQ, D, device=DEV, dtype=DT) / 8
    q_fp8 = q.to(FP8)  # vLLM quantizes the query too when kv_cache_dtype=fp8
    cu_q = torch.arange(B + 1, device=DEV, dtype=torch.int32)
    seqused = torch.full((B,), ctx, device=DEV, dtype=torch.int32)
    descale = torch.full((B, HKV), 1.0, device=DEV, dtype=torch.float32)
    out = {}

    for tag, fp8 in [("bf16", False), ("fp8", True)]:
        k_cache, v_cache, tables, hold = build_cache(ctx, B, n_reg, fp8)
        qq = q_fp8 if fp8 else q
        it = {"i": 0}

        def fn():
            t = tables[it["i"] % n_reg]
            flash_attn_varlen_func(
                q=qq, k=k_cache, v=v_cache,
                max_seqlen_q=1, cu_seqlens_q=cu_q,
                max_seqlen_k=ctx, seqused_k=seqused,
                causal=True, softmax_scale=SM_SCALE,
                block_table=t, fa_version=3,
                q_descale=descale if fp8 else None,
                k_descale=descale if fp8 else None,
                v_descale=descale if fp8 else None)
            it["i"] += 1

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        out[tag] = round(graph_med_us(fn), 2)
        if tag == "bf16":
            ref = flash_attn_varlen_func(
                q=q, k=k_cache, v=v_cache, max_seqlen_q=1, cu_seqlens_q=cu_q,
                max_seqlen_k=ctx, seqused_k=seqused, causal=True,
                softmax_scale=SM_SCALE, block_table=tables[0], fa_version=3)
        else:
            got = flash_attn_varlen_func(
                q=q_fp8, k=k_cache, v=v_cache, max_seqlen_q=1, cu_seqlens_q=cu_q,
                max_seqlen_k=ctx, seqused_k=seqused, causal=True,
                softmax_scale=SM_SCALE, block_table=tables[0], fa_version=3,
                q_descale=descale, k_descale=descale, v_descale=descale)
            out["relerr"] = round(((got.float() - ref.float()).norm() /
                                   ref.float().norm()).item(), 4)
        del k_cache, v_cache, tables, hold
        torch.cuda.empty_cache()

    wsb, wsf = ws_mb(ctx, B, bytes_bf16), ws_mb(ctx, B, bytes_fp8)
    t0 = 2.78
    bw = lambda mb, us: mb / max(us - t0, 0.1)  # MB/us == TB/s
    out.update({
        "ctx": ctx, "B": B, "n_regions": n_reg,
        "ws_mb_bf16": round(wsb, 1), "ws_mb_fp8": round(wsf, 1),
        "fits_l2_bf16": wsb < C_EFF_MB, "fits_l2_fp8": wsf < C_EFF_MB,
        "fp8_vs_bf16": round(out["bf16"] / out["fp8"], 3),
        "bw_bf16_tbs": round(bw(wsb, out["bf16"]), 2),
        "bw_fp8_tbs": round(bw(wsf, out["fp8"]), 2),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", action="store_true")
    args = ap.parse_args()
    ver = env_versions()
    key = gpu_key()
    print(f"GPU {ver['gpu']} ({key})  FA3 paged decode GQA {HQ}q/{HKV}kv x {D}  "
          f"rotate={args.rotate}\n")
    print(f"{'ctx':>6} {'B':>4} {'ws bf16':>8} {'ws fp8':>7} {'bf16 us':>9} "
          f"{'fp8 us':>8} {'fp8/bf16':>8} {'bwB':>6} {'bwF':>6} {'relerr':>7} {'nreg':>5}")
    rows = []
    for ctx, B in CONFIGS:
        r = run_cfg(ctx, B, args.rotate)
        rows.append(r)
        print(f"{ctx:>6} {B:>4} {r['ws_mb_bf16']:>7.1f}M {r['ws_mb_fp8']:>6.1f}M "
              f"{r['bf16']:>9.1f} {r['fp8']:>8.1f} {r['fp8_vs_bf16']:>8.3f} "
              f"{r['bw_bf16_tbs']:>6.2f} {r['bw_fp8_tbs']:>6.2f} "
              f"{r.get('relerr',0):>7.4f} {r['n_regions']:>5}")

    suffix = "_rotated" if args.rotate else ""
    save_json(os.path.join(_D, f"results_kv_decode_{key}{suffix}.json"), {
        "experiment": "kv_cache_quant_serving_qwen3.6-27B_full_attn",
        "gpu": ver["gpu"], "gpu_key": key, "versions": ver,
        "shape": {"h_q": HQ, "h_kv": HKV, "d": D, "block": BLK},
        "method": "FA3 flash_attn_varlen_func paged decode (vLLM v1 flash_attn "
                  "backend path); fp8 = e4m3 KV + (B,h_kv) descales, q bf16; "
                  "graph_med_us 10/graph median-of-40; rotate cycles cache "
                  "regions (>2xC_eff) per in-graph launch",
        "amdahl_note": "full-attention = 2.67% of Qwen3.6-27B runtime in Dr. "
                       "Xiao's profile -> model-level ceiling of KV-read gains <3%",
        "c_eff_mb": C_EFF_MB, "rotated": args.rotate,
        "rows": rows,
    })


if __name__ == "__main__":
    main()
