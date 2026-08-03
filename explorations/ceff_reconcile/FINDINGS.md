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

---

# Addendum (2026-08-03) — counter corroboration: the gap is real

nsight-compute installed on this box; warm-state DRAM reads
(`--cache-control none`, launch-skip 12, avg of 3) per launch:

| MB | GEMM (T=1) | re-read (sum) |
|----|-----------|----------------|
| 28 | 2.0       | 0.7  |
| 32 | 0.6       | 3.2  |
| 36 | 12.9      | 7.4  |
| 40 | 23.3      | 22.7 |
| 44 | 27.5      | 41.8 |

Reading: the GEMM's residency transition centers ~34 +/- 2 MB (2% miss at
32 -> 36% at 36); the pure re-read's centers ~40 +/- 2 (20% at 36 -> full
streaming at 44). The ~6 MB GEMM-context capacity gap from the timing
sweep is corroborated in counters. Texture, honestly noted: at 44 MB the
GEMM still hits 37% (tiling reuse gives it a partial-residency tail the
flat re-read lacks), and the re-read leaks a little earlier below its
break (soft ~3.5 MB rolloff, session-9 fine-grid). Conclusion stands:
C_eff is operand-context-dependent — carry C_eff(re-read) 39.8 and
C_eff(GEMM) ~34 as two measured constants; June's 36 sat between them.

Files: `ncu_target.py`; raw numbers above (per-launch averages).
