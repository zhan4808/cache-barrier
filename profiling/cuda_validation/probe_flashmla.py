"""De-risking probe for Experiment B (flashmla_sparse): build DeepSeek-MLA sparse
shapes, run the bf16 leg (flash_mla_sparse_fwd) and the FP8-KV leg
(concat_and_cache_mla -> flash_mla_with_kvcache, is_fp8_kvcache=True), and
cross-check the two outputs (rel-err small => layout correct).

Mirrors vLLM's flashmla_sparse backend dispatch: bf16 KV -> sparse_fwd kernel,
fp8 KV -> with_kvcache kernel. d_qk=576 (512 NoPE + 64 RoPE), d_v=512, h_q=128.
"""
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata,
    is_flashmla_dense_supported, is_flashmla_sparse_supported,
)

DEV = "cuda"
KVL, ROPE, DQK, DV = 512, 64, 576, 512   # MLA latent dims
HQ = 128                                  # query heads


def build(B, C, topk, blk=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    s_kv = B * C
    kv_c = (torch.randn(s_kv, KVL, device=DEV, dtype=torch.bfloat16, generator=g) / 8)
    k_pe = (torch.randn(s_kv, ROPE, device=DEV, dtype=torch.bfloat16, generator=g) / 8)
    q = (torch.randn(B, HQ, DQK, device=DEV, dtype=torch.bfloat16, generator=g) / 8)
    # per-request sparse indices: request b selects topk of its own [b*C,(b+1)*C)
    idx = torch.empty(B, 1, topk, device=DEV, dtype=torch.int32)
    for b in range(B):
        sel = torch.randperm(C, generator=g, device=DEV)[:topk].sort().values
        idx[b, 0, :] = (b * C + sel).to(torch.int32)
    return kv_c, k_pe, q, idx, s_kv


def run_bf16(kv_c, k_pe, q, idx):
    kv = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1).contiguous()  # [s_kv,1,576]
    scale = DQK ** -0.5
    out = flash_mla_sparse_fwd(q, kv, idx, scale, d_v=DV)[0]        # [B,HQ,DV]
    return out


def run_fp8(kv_c, k_pe, q, idx, s_kv, blk=64):
    nblk = (s_kv + blk - 1) // blk
    kv_cache = torch.zeros(nblk, blk, 656, device=DEV, dtype=torch.uint8)
    slot = torch.arange(s_kv, device=DEV, dtype=torch.int64)
    scale = torch.tensor(1.0, device=DEV, dtype=torch.float32)
    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot, "fp8_ds_mla", scale)
    qd = q.unsqueeze(1).contiguous()                               # [B,1,HQ,576]
    meta, _ = get_mla_metadata()
    out, lse = flash_mla_with_kvcache(
        q=qd, k_cache=kv_cache.unsqueeze(-2), block_table=None,
        cache_seqlens=None, head_dim_v=DV, tile_scheduler_metadata=meta,
        is_fp8_kvcache=True, indices=idx, softmax_scale=DQK ** -0.5,
    )
    return out.squeeze(1)                                          # [B,HQ,DV]


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    print("dense_supported:", is_flashmla_dense_supported(), "sparse_supported:", is_flashmla_sparse_supported())
    for B, C in [(1, 4096), (8, 4096)]:
        topk = min(2048, C)
        kv_c, k_pe, q, idx, s_kv = build(B, C, topk)
        ob = run_bf16(kv_c, k_pe, q, idx)
        of = run_fp8(kv_c, k_pe, q, idx, s_kv)
        rel = ((of.float() - ob.float()).norm() / ob.float().norm()).item()
        print(f"B={B} C={C} topk={topk}  bf16_out{tuple(ob.shape)} fp8_out{tuple(of.shape)}  "
              f"finite(bf16={torch.isfinite(ob).all().item()},fp8={torch.isfinite(of).all().item()})  "
              f"rel_err(fp8_vs_bf16)={rel:.4f}")
