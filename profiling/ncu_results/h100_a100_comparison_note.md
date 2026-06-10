## A100 vs H100 NCU Comparison (L2 Sweep)

> **CRITICAL CAVEAT (2026-06 audit):** these NCU runs used kernel replay with the
> default `--cache-control all`, which flushes GPU caches before every measured
> launch. All counters below are therefore **cold-cache** numbers: they contain no
> information about L2 residency, and the FP16 DRAM% rise with size is smooth
> duration-amortization of fixed overheads (steepest *below* 32 MB), not a knee at
> the L2 boundary. Do not cite this file as residency evidence; use the
> `--cache-control none` warm-loop counters in `profiling/validation/` instead.
> The INT4 SM%-dominance observation (dequant-bound at all sizes) is unaffected.

- Figure: `profiling/ncu_results/figure_ncu_h100_a100_side_by_side.png`
- Merged data: `profiling/ncu_results/h100_a100_l2_sweep_merged.csv`

### Key observations

- **FP16 DRAM trend (knee-consistent):**
  - H100 FP16 DRAM: below-50MB avg `55.5%` -> above-50MB avg `78.6%`
  - A100 FP16 DRAM: below-40MB avg `43.0%` -> above-40MB avg `66.2%`
- **INT4 remains SM-heavy on both GPUs:**
  - H100 INT4 SM: below-50MB avg `51.7%` -> above-50MB avg `73.8%`
  - A100 INT4 SM: below-40MB avg `51.2%` -> above-40MB avg `74.5%`
- **Mechanism consistency:** FP16 becomes increasingly DRAM-driven beyond each GPU's L2 boundary (50MB H100, 40MB A100), while INT4 remains dominated by SM-side dequant work.

### Suggested paper usage

Use the side-by-side figure as the primary cross-hardware counter evidence, and keep tables as appendix/detail.
