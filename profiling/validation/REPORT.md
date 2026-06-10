# Methodology Audit: the "L2 Cache Barrier" claim (cache-barrier / kernel-compass)

Hardware: same as paper — H100 80GB SXM5 (HBM3), 50 MB L2. Stack here: torch 2.7.0+cu126, triton 3.3.0, ncu 2025.1.1 (paper: torch 2.9.1, triton 3.5.1).
All new code/data in this directory (`profiling/validation/`).

## Verdict on the paper's causal claim

> "INT4 fails to outperform FP16 because the 16 MB weights are L2-resident, served at ~12 TB/s."

**Partially true mechanism, wrong evidence, overstated causality.**

| Paper claim | Status | Measured reality |
|---|---|---|
| FP16 weights L2-served in the timed loop | TRUE (mechanism) | 75% of weight bytes from L2 at 16 MB, 93% at 32 MB (NCU `--cache-control none`, `dram__bytes_read` 4.0 MB / 2.2 MB per launch) |
| ...at ~12 TB/s | FALSE | effective serving BW 3.5–4.6 TB/s; incremental slope 6.3 TB/s below the cliff |
| Boundary at 50 MB (L2 capacity) | FALSE | residency collapses between 32 and 40 MB (effective LRU capacity ≈ 36 MB; at 40 MB, 95% of weights already re-read from DRAM) |
| NCU sweep "confirms the mechanism" | INVALID | all repo NCU runs used kernel replay with default `--cache-control all`, which **flushes L2 before every measured launch**. Their counters are cold-cache at every size; their own "warm vs cold" singlepass file shows identical DRAM% (48.4 vs 48.8) because both were cold. |
| FP16 DRAM% knee "just after the L2 boundary" | FALSE | their own data rises smoothly 35→51→65→71→76→74→81→83%; steepest below 32 MB. The rise is duration-amortization of fixed overhead on a cold cache, not an L2 effect. |
| Flat FP16 latency 8–32 MB = L2 serving | ARTIFACT | per-launch CUDA-event methodology has a ~15.5 µs floor (a 0.5 MB bmm also times 15.6 µs). True kernel times (CUDA-graph): 3.5→7.3 µs, i.e. NOT flat — scales with size at 6.3 TB/s. |
| Warm/evict asymmetry (FP16 +34% vs INT4 +17%) proves L2 dependence | MISLEADING | absolute deltas nearly equal: FP16 +4.8 µs (= full 16 MB HBM reload), INT4 +3.9 µs (expected reload only 1.3 µs → ~2.6 µs is common-mode eviction overhead). The relative asymmetry is a baseline artifact (15.8 vs 41.3 µs denominators). |
| **L2 residency is the root cause of INT4's failure** | **REFUTED as sufficient cause** | rotation intervention at fixed 16 MB shape: cycling 6 weight copies (96 MB working set) forces FP16 to HBM → FP16 8.3 µs vs INT4 9.1–9.3 µs. **INT4 still loses with zero L2 advantage.** |

## The decomposition that the paper should have reported (16 MB MLA point, bs=1, graph-timed)

- FP16 L2-warm: **4.8 µs**
- FP16 forced-HBM (L2 destroyed): **8.3 µs**
- INT4 (insensitive to cache state): **9.1–9.3 µs**
- INT4 memory-entitlement (4.2 MB @ 2.7 TB/s + fixed): **~3.4 µs**

So of INT4's 1.91× kernel-level deficit:
1. L2 residency explains the 1.12× → 1.91× portion (FP16's tier advantage). Real, but not sufficient.
2. The INT4 kernel runs 2.7× above its own memory entitlement because it is **dequant-compute-bound** (NCU: SM 46–79%, DRAM 10–23% at all sizes): scalar mask/shift/select/convert per packed byte, `BLOCK_M=16` tiles for M=1 (15/16 of tensor-core work padded away), and 2× `tl.dot` per K-block from the even/odd interleave.
3. Under the repo's event-per-launch timing, a ~15.5 µs launch/eventing floor dominates everything below 32 MB and inflates FP16 "latency" ~3×; Triton's higher launch overhead also pollutes the INT4/FP16 ratio (event-timed 2.75×, true kernel 1.91×).

## On "memory-bound → compute-bound transition caused by L2"

There is no such *transition* in either kernel:
- INT4 is compute(dequant)-bound at every size — including ≫L2 — so L2 cannot be its cause.
- FP16 stays memory-side at every size; what changes at ~36 MB is the serving **tier** (L2 → HBM), i.e. a bandwidth-ceiling change, not a memory→compute transition.
- The requested validation figures confirm: useful compute peaks at 4.6 TFLOPS = 0.5% of peak (nobody is compute-throughput-bound), and FP16 effective bandwidth exceeds the 3.35 TB/s HBM peak only below 32 MB — direct, assumption-free proof of (bounded) L2 serving.

## Robustness warnings

- Published magnitudes do not transfer across stacks: event-timed ratio at 16 MB is 1.86× in the repo's data (triton 3.5.1) vs 2.75× here (triton 3.3.0); at 128 MB 1.08× vs 1.58×. Direction holds, numbers don't.
- cuBLAS dispatches different `nvjet` tile variants across the sweep (128x8/256x8/512x8), so the FP16 curve mixes kernel implementations.
- The 8 MB point shows 100% DRAM re-read even warm (likely an evict-policy choice of that nvjet variant) — the "L2-resident regime" is not even monotonic in size.
- bs=1 numbers from per-launch event timing are launch-overhead measurements, not kernel measurements; this contaminates RESULTS.md §3 (BMM bandwidth/TFLOPS tables) as well.

## What survives, corrected

1. MLA reconstruction BMMs at small bs are tiny, latency-floor-dominated ops; weight-only INT4 with a dequant-heavy Triton kernel cannot win there. (Holds.)
2. A cache-tier effect exists with effective capacity ~36 MB and ~4–6 TB/s effective L2-era bandwidth; any roofline for these shapes must be multi-tier. (Holds with corrected numbers.)
3. The paper's headline causal sentence should be weakened to: "L2 residency removes most of the theoretical INT4 upside; the remainder is destroyed by dequantization compute and launch overhead — INT4 never wins at any size, reaching only parity above the cache cliff."

## Implications for next steps (items 2–3 of the plan)

- **Cache-aware roofline (Ilic et al., CARM):** viable and now properly grounded — use measured tier bandwidths (L2_eff ≈ 6.3 TB/s incremental, HBM_eff ≈ 2.7 TB/s) and effective capacity (~36 MB), plus an explicit latency-floor extension for µs-scale kernels (classic CARM has no such term; at bs=1 every MLA kernel sits below the loft because of the ~2 µs kernel fixed cost / ~15 µs launch cost).
- **FlagGems mixed-precision fused_moe (PR #2336):** the reported degradation with token count is consistent with the same failure mode (dequant-compute-bound inner kernel + tile padding). The rotation/CUDA-graph/cache-control-none toolkit built here transfers directly and should be applied before any optimization claims.
