# Mechanism taxonomy — why quantized kernels lose when rooflines say they win
**2026-07-02, H100 · `profiling/mechanisms/` · all mechanisms measured, committed per-experiment**

Charter (Robert): rigorously hunt the mechanisms that precision-routing rooflines
overlook — the things that make quantized kernels *slower* than baseline. Five new
mechanisms measured tonight, joining the three previously established (L2 capacity
gate, in-core dequant ceiling, act-quant tax). The taxonomy below is the paper-facing
artifact: every row is a measurement, not a hypothesis.

## The taxonomy

| # | Mechanism | Symptom | Measured magnitude (H100) | Detection predicate | Routing rule |
|---|---|---|---|---|---|
| 1 | **L2 capacity gate** (operand-aware) | quant loses when its operand is L2-resident; super-wins when quantization pulls it in | 0.70× resident → 2.4–3.0× boundary (warm); cliffs at each operand's own size | operand_bytes vs C_eff≈36 MB | don't quantize L2-resident operands (if residency survives — see #2) |
| 2 | **Contention step-function** | all cache effects vanish under co-tenancy | L2 tier → HBM tier the moment Σ(hot WS) > C_eff; no partial credit | Σ co-tenant WS vs C_eff | full-model serving ⇒ assume HBM tier; L2 rules apply only to intra-step reuse |
| 3 | **In-core dequant ceiling** (weight-only) | W8A16/W4A16 caps out and loses once compute-bound | 334 TFLOPS dense / 423 MoE (vs bf16 713–727); crossover M≈64–128 dense, T≈300 MoE | F/P_dequant > W/BW | matched precision (W8A8) for high-M; W8A16 only below crossover |
| 4 | **Act-quant tax** | deployed W8A8 loses at decode despite winning kernel | 8.4 µs fixed + ~1.6 TB/s; kills small-M wins (1.96×→0.98×) | M small & quant unfused | fuse into preceding norm/silu (+2.2 µs fused — measured) |
| 5 | **SM-parallelism floor / tile starvation** *(new, A)* | Marlin slower than bf16 at small N despite ½ the bytes | 16–29 µs pure overhead at N=256 (worse with K); gone by N≥4096; **cutlass W8A8 immune** | N/tile_N ≪ #SMs (Marlin = persistent 132-CTA grid) | never route small-N layers to Marlin; kernel-specific, check per-kernel |
| 6 | **Misaligned wave-quantization bands** *(new, B)* | quant-vs-bf16 ratio non-monotonic in M; same shape swings win↔lose | ratio 0.75–2.71× on one shape; band edges at 128→136, 256→288, 416→448…; bf16 latency itself non-monotonic (460→280 µs for *more* work) | \|Δratio\|>0.25 across a tile/wave boundary | never interpolate a routing decision across band edges; per-band tables or wave-aware model |
| 7 | **Scale-format / implementation split** *(new, C)* | production blockwise fp8 slower than per-tensor — or than bf16 | cutlass blockwise: +15–21% (M≥256), ~2× (M≤64), **unsupported M<4**; triton fallback 1.75–4× slower, loses to bf16 at decode | format ⇒ which kernel dispatches (M<4 ⇒ triton) | block-fp8 on H100 at decode can be slower than not quantizing; verify the dispatch, not the format |
| 8 | **Occupancy floor at decode** *(new, D — refutes register hypothesis)* | all GEMMs idle at M=1 | 8–14% warp occupancy, 5–21% SM throughput, grids 96–132 CTAs; cutlass fp8 uses only **34 regs/thread** | M ≲ BLOCK_M and grid ≲ #SMs | decode perf = streaming efficiency of few CTAs; registers are not the lever |
| 9 | **BLOCK_M staircase / padding** *(new, E)* | tokens within a tile are free; marginal tile costs differ per kernel | cutlass W8A8: flat M=1→64 (BLOCK_M=64), steps at 65/129; Marlin steps at 49/65 with 3–4× higher marginal-tile cost (re-dequants per tile) | M mod BLOCK_M | batch decode to tile boundaries; Marlin's per-tile dequant re-run penalizes multi-tile M |

Supporting counter-level evidence (NCU): warm >C_eff operands show **67% L2 hit**
directly in DRAM counters — the "partial residency" the capacity model assumes.

## What this changes

1. **A roofline (even cache-aware CARM v2) is necessary but not sufficient for
   precision routing.** Mechanisms 5–7 are *kernel-implementation* properties —
   invisible to any model keyed only on bytes, flops, and cache. The routing layer
   needs per-kernel predicates (tile counts, BLOCK_M, dispatch min-M, band edges).
2. **"W8A8 wins everywhere" is now precisely scoped:** true for per-tensor cutlass
   (1.80–1.94× vs tuned bf16, no cliff); false at decode for production blockwise
   (triton fallback loses to bf16); conditional on fused act-quant at small M.
3. **The MAPE story resolves:** CARM v2's bf16 22%/dense 7.4% residuals are mostly
   mechanism 6 (bands) — irreducible for a smooth model, addressable by band-aware
   routing tables.

## Caveats / open
- Cutlass-blockwise direct call has a scale-layout mismatch (rel-err 0.10 vs 0.037
  expected) — timings volume-valid; re-verify correctness via vLLM `Fp8LinearOp`.
- Band edges mapped on two shapes; per-shape band tables would be generated, not
  hand-derived. Marlin step at 49 (not 65) unexplained — likely 16-row m-tiles × 3.
- All on H100 / vLLM 0.20.2 kernels; mechanisms 5–7 are implementation-versioned.

Files: `bench_marlin_smalln.py`, `bench_ratio_bands.py`, `bench_block_fp8.py`
(+ cutlass block probe), `ncu_target.py`, `bench_padding_steps.py`, results JSONs.
