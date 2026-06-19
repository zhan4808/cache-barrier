"""Probe the SAME-KERNEL dense MLA decode: flash_mla_with_kvcache toggling only
is_fp8_kvcache (bf16 KV vs fp8 KV). Isolates KV precision from kernel choice.

RESULT (recorded): a clean same-kernel / bf16-compute / fp8-KV control is NOT
available on Hopper FlashMLA.
  - Dense bf16 KV via flash_mla_with_kvcache(is_fp8_kvcache=False): works.
  - Dense fp8 KV via the SAME entry point: RuntimeError "query and key must have
    the same dtype" -- the dense fp8 decode requires q ALSO in fp8 (dedicated
    flash_mla_with_kvcache_fp8 with descale_q/descale_k), i.e. a full-FP8 compute
    regime, NOT the bf16-compute / fp8-KV regime the study targets.
  - The sparse decode kernel asserts is_fp8_kvcache=True (sparse is fp8-KV-only).
=> The bf16-vs-fp8-KV comparison in the sparse regime is necessarily a two-kernel
   comparison (bf16->sparse_fwd, fp8->with_kvcache), which is exactly what vLLM's
   flashmla_sparse backend dispatches. Done in bench_flashmla_sparse.py.
"""
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

DEV = "cuda"
KVL, ROPE, DQK, DV, HQ, BLK = 512, 64, 576, 512, 128, 64


def build_paged(B, C, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    blocks_per = (C + BLK - 1) // BLK
    nblk = B * blocks_per
    kv_c = torch.randn(B * C, KVL, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    k_pe = torch.randn(B * C, ROPE, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    q = torch.randn(B, 1, HQ, DQK, device=DEV, dtype=torch.bfloat16, generator=g) / 8
    block_table = torch.arange(nblk, device=DEV, dtype=torch.int32).view(B, blocks_per)
    cache_seqlens = torch.full((B,), C, device=DEV, dtype=torch.int32)
    # slot_mapping: request b, token t -> block_table[b, t//BLK]*BLK + t%BLK
    slot = torch.empty(B * C, device=DEV, dtype=torch.int64)
    for b in range(B):
        base = b * C
        for t in range(C):
            slot[base + t] = block_table[b, t // BLK].item() * BLK + (t % BLK)
    return kv_c, k_pe, q, block_table, cache_seqlens, slot, nblk


def run(B, C):
    kv_c, k_pe, q, bt, csl, slot, nblk = build_paged(B, C)
    scale = DQK ** -0.5
    sc = torch.tensor(1.0, device=DEV, dtype=torch.float32)

    # bf16 KV paged cache (kv_cache_dtype "auto" -> bf16, 576 per token)
    cache_bf16 = torch.zeros(nblk, BLK, DQK, device=DEV, dtype=torch.bfloat16)
    ops.concat_and_cache_mla(kv_c, k_pe, cache_bf16, slot, "auto", sc)
    meta_b, _ = get_mla_metadata()
    ob, _ = flash_mla_with_kvcache(q=q, k_cache=cache_bf16.unsqueeze(-2), block_table=bt,
                                   cache_seqlens=csl, head_dim_v=DV,
                                   tile_scheduler_metadata=meta_b, causal=False,
                                   is_fp8_kvcache=False, softmax_scale=scale)

    # fp8 KV paged cache (fp8_ds_mla, 656 bytes/token)
    cache_fp8 = torch.zeros(nblk, BLK, 656, device=DEV, dtype=torch.uint8)
    ops.concat_and_cache_mla(kv_c, k_pe, cache_fp8, slot, "fp8_ds_mla", sc)
    meta_f, _ = get_mla_metadata()
    of, _ = flash_mla_with_kvcache(q=q, k_cache=cache_fp8.unsqueeze(-2), block_table=bt,
                                   cache_seqlens=csl, head_dim_v=DV,
                                   tile_scheduler_metadata=meta_f, causal=False,
                                   is_fp8_kvcache=True, softmax_scale=scale)
    rel = ((of.float() - ob.float()).norm() / ob.float().norm()).item()
    print(f"B={B} C={C}  bf16{tuple(ob.shape)} fp8{tuple(of.shape)}  "
          f"finite({torch.isfinite(ob).all().item()},{torch.isfinite(of).all().item()})  rel={rel:.4f}")


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    for B, C in [(1, 4096), (8, 4096)]:
        run(B, C)
