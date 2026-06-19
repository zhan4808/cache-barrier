"""Red-team: is the FlashMLA FP8-KV high-batch washout robust (not a transient)?
Clock-locked, C=4096 (genuinely sparse, topk=2048), batch extended to 128, with a
roofline check: report achieved KV bandwidth so we can see WHICH regime each point
is in (byte-bound vs occupancy/gather-bound)."""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import graph_med_us  # noqa: E402
import bench_flashmla_sparse as M  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.v1.attention.ops.flashmla import (  # noqa: E402
    flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata)

DEV = "cuda"
C = 4096
TOPK = 2048
BATCHES = [1, 8, 16, 32, 64, 128]
DQK, DV, HQ, BLK = 576, 512, 128, 64


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}  SM clock locked  C={C} topk={TOPK} h_q={HQ}")
    print("\n  B   bf16_us  fp8_us  fp8/bf16  bf16_KV_BW  fp8_KV_BW  regime")
    rows = []
    for B in BATCHES:
        kv_c, k_pe, q, idx, s_kv = M.build(B, C, TOPK, seed=B)
        kv_bf16 = torch.cat([kv_c, k_pe], -1).unsqueeze(1).contiguous()
        scale = DQK ** -0.5
        bf16_fn = lambda: flash_mla_sparse_fwd(q, kv_bf16, idx, scale, d_v=DV)[0]

        nblk = (s_kv + BLK - 1) // BLK
        kv_cache = torch.zeros(nblk, BLK, 656, device=DEV, dtype=torch.uint8)
        slot = torch.arange(s_kv, device=DEV, dtype=torch.int64)
        ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot, "fp8_ds_mla",
                                 torch.tensor(1.0, device=DEV, dtype=torch.float32))
        kc = kv_cache.unsqueeze(-2); qd = q.unsqueeze(1).contiguous()
        meta, _ = get_mla_metadata()
        fp8_fn = lambda: flash_mla_with_kvcache(
            q=qd, k_cache=kc, block_table=None, cache_seqlens=None, head_dim_v=DV,
            tile_scheduler_metadata=meta, is_fp8_kvcache=True, indices=idx, softmax_scale=scale)[0]

        for _ in range(5):
            bf16_fn(); fp8_fn()
        torch.cuda.synchronize()
        bf = statistics.median([graph_med_us(bf16_fn) for _ in range(3)])
        f8 = statistics.median([graph_med_us(fp8_fn) for _ in range(3)])
        # KV bytes actually touched per call = B * topk * (bytes/token)
        bf_bw = B * TOPK * (DQK * 2) / (bf * 1e-6) / 1e12   # bf16 KV = 576*2 B/token
        f8_bw = B * TOPK * 656 / (f8 * 1e-6) / 1e12         # fp8 KV = 656 B/token
        regime = "byte-bound" if max(bf_bw, f8_bw) > 1.5 else "gather/occ-bound"
        rows.append((B, bf, f8))
        print(f"{B:4d}  {bf:7.1f}  {f8:7.1f}   {bf/f8:5.2f}   {bf_bw:6.2f}TB/s  {f8_bw:6.2f}TB/s  {regime}")
        del kv_c, k_pe, q, idx, kv_bf16, kv_cache, kc, qd
        torch.cuda.empty_cache()

    print("\nVerdict: FP8/bf16-KV ratio vs batch:",
          "  ".join(f"B{B}:{bf/f8:.2f}" for B, bf, f8 in rows))


if __name__ == "__main__":
    main()
