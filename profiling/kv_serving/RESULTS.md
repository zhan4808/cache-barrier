# KV-cache quantization under serving conditions — results (2026-07-01, H100)

Dr. Xiao todo #2: *"Verify whether components in real inference services, such as
KV cache quantization, are limited by L2 cache capacity."*
Setup: exact vLLM v1 production path (FA3 `flash_attn_varlen_func`, paged KV,
block 16), Qwen3.6-27B full-attention GQA (24q/4kv × 256), bf16-KV vs fp8-KV
(e4m3 KV **and** e4m3 query with descales — vLLM quantizes q too; measured
rel-err 0.031–0.034). ctx {1k,8k,32k} × B {1..128}, warm + rotated modes.
`bench_kv_decode.py`, `results_kv_decode_h100{,_rotated}.json`.

## Answer: No — KV reads are not limited by L2 capacity.

| cell (ctx×B) | ws bf16/fp8 | fp8/bf16 warm | fp8/bf16 rotated | bf16 BW | fp8 BW |
|---|---|---:|---:|---:|---:|
| 1k×1 (both fit L2) | 4.2/2.1 MB | **0.71×** | 0.72× | 0.41 | 0.14 |
| 1k×8 / 8k×1 (boundary) | 33.6/16.8 MB | **0.69–0.72×** | 0.85–0.88× | 2.2–2.35 | 0.7–0.8 |
| streamed (≥134 MB) | up to 4.3 GB | 0.96–1.07× | 0.95–1.06× | 2.4–3.1 | 1.1–1.65 |

Three observations, each measured:

1. **No L2 tier ever appears for KV.** Even the 4.2 MB cell serves at 0.41 TB/s
   (occupancy-bound: decode attention at B=1 can't fill the machine), and no
   cell exceeds the ~3.1 TB/s HBM tier. Rotation (the multi-layer eviction
   control) changes ratios by ≲0.03 almost everywhere — nothing resident to
   evict. Contrast the dense-weight experiment, where warm→rotated moved
   effective BW by 1.5–1.7×. **Decode KV reads are occupancy-bound at small
   working sets and HBM-streamed at large ones; the L2 capacity term is inert
   here — and CARM correctly classifies it as inert.**

2. **The fp8 decode kernel's own bandwidth ceiling dominates the outcome.**
   fp8 path saturates at ~1.5–1.65 TB/s vs bf16's ~2.8–3.1 — so fp8 needs its
   2× byte advantage just to reach parity, wins at most 1.06–1.07× at the
   largest streamed configs, and **loses 0.69–0.72× at small batch/context**.
   This reproduces the FlashMLA finding (fp8 0.8 vs bf16 1.9 TB/s) on a second,
   independent attention family (GQA/FA3) — the "kernel-BW ceiling" mechanism
   generalizes.

3. **Model-level (Amdahl):** full attention is 2.67% of Qwen3.6-27B runtime in
   Dr. Xiao's own profile → the best measured fp8-KV operator gain (+7%) moves
   end-to-end throughput by **≤0.2%**, while a decode-heavy small-batch service
   would see the operator *lose* ~30%. fp8-KV on this model is a memory-capacity
   feature (half the KV footprint → longer contexts / more sequences), not a
   speed feature.

## Caveats
- Descales fixed at 1.0 (uniform); real per-head scales don't change bandwidth.
- KV-write (`reshape_and_cache`) not included: 0.06% of runtime in the profile.
- Single-op microbench; the rotated mode is our serving-eviction proxy.
