# D2+D5 — sparse-attention KV hot sets and quantized KV vs the gate (H100)

**Question**: DSA-class sparse attention shrinks the hot KV working set to
~2K selected tokens per request (single-digit MB) — does the capacity gate
govern this operand class, and is the gate byte-located (so KV quantization
buys resident batch/context arithmetically)?

**Setup**: `bench_kv_hotset.py` — graph-timed decode GQA sdpa
(q [B,32,1,128], KV [B,Hkv,2048,128] fp16), warm (same KV each step — the
decode access pattern) vs rotated-copies control; B sweeps the batch hot
set across 8–96 MB at two per-token byte costs (Hkv=8: 8 MB/req,
Hkv=4: 4 MB/req).

## Findings

1. **The gate governs sparse-KV hot sets** (D2). Warm re-read advantage
   1.11–1.31x below the gate, collapsing to 1.00 between 40 and 48 MB —
   consistent with the fine-grid C_eff onset (39.8 MB) plus the soft band.
   The dense-KV null (Exp B: "KV is not L2-limited") was a statement about
   operand SIZE, not about attention: once sparse selection shrinks the hot
   set into gate range, KV behaves like any other gated operand.

2. **The cliff is byte-located, not token- or request-located** (D5 core).
   At half the per-token bytes (Hkv=4), the identical warm/rot pattern
   appears at exactly 2x the batch and the SAME MB position (peak 1.31 at
   32 MB, collapse at 48 MB, both sweeps). Corollary, now arithmetic
   rather than conjecture: **fp8 KV doubles and 4-bit KV quadruples the
   resident batch (or selected context) before the gate closes.** For a
   DSA-like 8 MB/request hot set on H100: B* ~= 5 at fp16, ~10 at fp8,
   ~20 at int4; on B300 (C_eff ~150 MB predicted) ~4x those.

3. **Failed leg, stated**: the fp8-storage spot check (inline
   `.to(fp16)` dequant then sdpa) is dominated by whole-tensor dequant
   materialization (155 vs 32 us at B=8) — plain-torch sdpa cannot express
   dequant-on-read. The residency claim for stored-fp8 KV needs a fused
   fp8 attention kernel (FlashAttention-3 fp8 / FlashMLA fp8 / vLLM); the
   byte-governance result in (2) already carries the D5 conclusion, but
   the kernel-level confirmation is open.

4. **Magnitudes are kernel-floored again**: decode-shape sdpa moves the
   8 MB hot set at ~0.6 TB/s effective (launch-bound at 12.4 us) — the
   residency LOCATION is clean, the exploitable magnitude awaits better
   kernels. Same division as D1 and the paper's model form: capacity
   structure transfers; magnitude is a per-kernel term.

## Serving-side reading

Sparse attention does not remove KV from the gate's jurisdiction — it
moves KV INTO it. Selection (DSA top-k), quantization (fp8/int4 KV), and
batch size jointly set whether the hot set is resident; the gate prices
the trade. A serving-policy corollary worth a follow-up: cap
B x hot-KV bytes at C_eff per attention layer group when latency-critical.

Data: `results_kv_hotset_nvidia-h100-80gb-hbm3.json`.
