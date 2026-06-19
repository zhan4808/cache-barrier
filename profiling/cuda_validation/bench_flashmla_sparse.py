"""Experiment B — flashmla_sparse: bf16 KV vs FP8 KV (DeepSeek V3.2 sparse MLA).

Mirrors vLLM's flashmla_sparse backend dispatch exactly:
  - bf16 KV  -> flash_mla_sparse_fwd            (the sparse prefill/decode kernel)
  - FP8  KV  -> concat_and_cache_mla("fp8_ds_mla") + flash_mla_with_kvcache(
                 is_fp8_kvcache=True, indices=...)
Compute stays bf16 in both; the only thing that changes is the KV-cache precision
(and, faithfully to vLLM, the kernel selected for it). This is the KV-cache
quant/dequant regime -- same "does it wash out at small sizes?" question as MoE.

MLA dims: d_qk=576 (512 NoPE + 64 RoPE), d_v=512, h_q=128 (DeepSeek-V3 heads).
Sparse top-k = 2048 (V3.2 "DSA" setting), capped at context length C.

Sweep: context C in {512,2048,4096,8192} x batch B in {1,8,32}.
All graph-timed (graph_med_us; eager fallback flagged). Output:
results_flashmla_sparse.json (new file).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import graph_med_us, eager_med_us, env_versions, save_json  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.attention.ops.flashmla import (  # noqa: E402
    flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata,
    is_flashmla_sparse_supported,
)

DEV = "cuda"
KVL, ROPE, DQK, DV = 512, 64, 576, 512
HQ = 128
TOPK_SPARSE = 2048           # DeepSeek V3.2 sparse selection
BLK = 64                     # page block size
CONTEXTS = [512, 2048, 4096, 8192]
BATCHES = [1, 8, 32]
_D = os.path.dirname(os.path.abspath(__file__))


def build(B, C, topk, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    s_kv = B * C
    kv_c = torch.randn(s_kv, KVL, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    k_pe = torch.randn(s_kv, ROPE, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    q = torch.randn(B, HQ, DQK, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    idx = torch.empty(B, 1, topk, device=DEV, dtype=torch.int32)
    for b in range(B):
        sel = torch.randperm(C, generator=g, device=DEV)[:topk].sort().values
        idx[b, 0, :] = (b * C + sel).to(torch.int32)
    return kv_c, k_pe, q, idx, s_kv


def timed(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    try:
        return round(graph_med_us(fn), 2), "graph"
    except Exception as exc:  # noqa: BLE001
        return round(eager_med_us(fn), 2), f"eager({type(exc).__name__})"


def main():
    ver = env_versions()
    print(f"GPU: {ver['gpu']}  vllm={ver['vllm']}  sparse_supported={is_flashmla_sparse_supported()}")
    print(f"MLA d_qk={DQK} d_v={DV} h_q={HQ}  topk(V3.2)={TOPK_SPARSE}\n")
    rows = []
    for C in CONTEXTS:
        for B in BATCHES:
            topk = min(TOPK_SPARSE, C)
            kv_c, k_pe, q, idx, s_kv = build(B, C, topk, seed=B * 100 + C)

            # bf16 leg: flash_mla_sparse_fwd on a [s_kv,1,576] bf16 KV
            kv_bf16 = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1).contiguous()
            scale = DQK ** -0.5
            bf16_fn = lambda: flash_mla_sparse_fwd(q, kv_bf16, idx, scale, d_v=DV)[0]
            bf16_us, bm = timed(bf16_fn)

            # fp8 leg: concat_and_cache_mla -> flash_mla_with_kvcache(is_fp8_kvcache=True)
            nblk = (s_kv + BLK - 1) // BLK
            kv_cache = torch.zeros(nblk, BLK, 656, device=DEV, dtype=torch.uint8)
            slot = torch.arange(s_kv, device=DEV, dtype=torch.int64)
            ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot, "fp8_ds_mla",
                                     torch.tensor(1.0, device=DEV, dtype=torch.float32))
            kc = kv_cache.unsqueeze(-2)
            qd = q.unsqueeze(1).contiguous()
            meta, _ = get_mla_metadata()
            fp8_fn = lambda: flash_mla_with_kvcache(
                q=qd, k_cache=kc, block_table=None, cache_seqlens=None,
                head_dim_v=DV, tile_scheduler_metadata=meta,
                is_fp8_kvcache=True, indices=idx, softmax_scale=scale)[0]
            fp8_us, fm = timed(fp8_fn)

            # correctness: fp8 KV vs bf16 KV (same logical attention)
            ob = bf16_fn().float()
            of = fp8_fn().squeeze(1).float()
            rel = ((of - ob).norm() / ob.norm()).item()

            row = {
                "context": C, "batch": B, "topk": topk, "s_kv": s_kv,
                "sparse": C > TOPK_SPARSE,
                "bf16_kv_us": bf16_us, "fp8_kv_us": fp8_us,
                "fp8_vs_bf16": round(bf16_us / fp8_us, 3),
                "relerr_fp8_vs_bf16": round(rel, 4),
                "timing": {"bf16": bm, "fp8": fm},
            }
            rows.append(row)
            print(f"C={C:5d} B={B:2d} topk={topk:5d} ({'SPARSE' if row['sparse'] else 'dense '})  "
                  f"bf16_kv={bf16_us:8.2f}u  fp8_kv={fp8_us:8.2f}u  "
                  f"({row['fp8_vs_bf16']:.2f}x)  relerr={rel:.4f}")
            del kv_c, k_pe, q, idx, kv_bf16, kv_cache, kc, qd
            torch.cuda.empty_cache()

    out = {
        "experiment": "B_flashmla_sparse_bf16_vs_fp8kv",
        "gpu": ver["gpu"], "versions": ver,
        "mla": {"d_qk": DQK, "d_v": DV, "h_q": HQ, "topk_sparse": TOPK_SPARSE, "block": BLK},
        "method": "graph_med_us 10 launches/graph, median of 40 replays",
        "dispatch": "bf16 KV -> flash_mla_sparse_fwd; fp8 KV -> flash_mla_with_kvcache(is_fp8_kvcache=True) "
                    "(matches vllm flashmla_sparse backend). compute bf16 both; KV precision differs.",
        "rows": rows,
    }
    save_json(os.path.join(_D, "results_flashmla_sparse.json"), out)


if __name__ == "__main__":
    main()
