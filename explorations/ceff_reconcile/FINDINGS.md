# Autoloop A — GEMM-context C_eff vs re-read C_eff (partial, honest)

**Question**: carm_model.json says C_eff=36 MB (June, GEMM-fitted); the
harness re-read cliff says onset 39.8 +/- 0.5. Pre-registered hypothesis:
the gap is the activation+output footprint (W_cliff = 39.8 - act - out).

**Status: hypothesis NOT confirmed; experiment hit two instrument limits**
(`bench_gemm_ceff.py`, `results_gemm_ceff_h100.json`):

1. The T-sweep leg is invalid above T~58: bf16 GEMM goes compute-bound
   (roofline max() flips), so "weight BW" cliffs at T in {128,256,384}
   are not interpretable. Slope estimates from this run are meaningless.
2. At T=1 (valid, memory-bound), the GEMM weight cliff sits at ~31-34 MB
   — clearly BELOW the re-read onset 39.8, and act+out (~20 KB at T=1)
   cannot explain the gap. Direction: a genuine GEMM-context capacity
   term (cuBLAS tiling holds residency worse than pure re-read), worth
   ~6-9 MB on H100. Detection is unstable cell-to-cell (nvjet per-shape
   kernel selection), so the number is a range, not a point.

**Follow-up design** (for a session with NCU): pin one kernel across the
sweep (fixed N, scale W via batched copies), sweep at T<=32 only, and
corroborate with warm-state DRAM-read counters. If the ~6-9 MB gap holds,
the model should carry C_eff(re-read) and C_eff(GEMM) as two measured
constants — the June 36 was not wrong, it was a different operand context.
