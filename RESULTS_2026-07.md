# RESULTS — 2026-07 pivot cycle (written 2026-08-01)

What moved, what held, and what broke across the 2026-07 pivot (`DIRECTION.md`)
and its execution: the 2026-07-01→02 session (dense/KV/W8A8/mechanisms) and the
2026-08-01 session (P0 reframe + P1 gate figure).

---

## Status vs the DIRECTION.md phase plan

| Phase | Status | Where |
|---|---|---|
| P0 reframe (README, paper title/abstract, related-work positioning) | **Done 2026-08-01.** Abstract's first sentence is capacity-gated precision dispatch; no MLA/INT4. Paper *body* still needs the same surgery | `README.md`, `paper/main.tex` |
| P1 gate figure ★ | **Done 2026-08-01** | `profiling/gate/` |
| P2 dense in scope? (Xiao todo #1) | **Done 2026-07-01→02**, beyond spec: not a shape census but direct measurement of all three regimes on real Qwen3.6-27B projections | `profiling/dense_qwen/` |
| P3 serving-load C_eff (Xiao todo #2) | **Substantively done 2026-07**, with a stronger result than the hypothesis (see "what broke") | `profiling/dense_qwen/results_contention_h100.json`, `profiling/kv_serving/` |
| P4 compute quantization (Xiao todo #3) | **Done 2026-07**: W8A8 1.80–1.94× vs tuned bf16, no cliff; W4A4 impossible natively on Hopper | `cuda_validation/verify_moe_w8a8_tuned.py` |
| P5 portability harness | **Not done** (partial A100 data exists) | — |
| P6 dispatch cost model | **Not done** | — |
| In-flight: served A/B + accuracy | **Staged, never run** (instance died); needs vLLM env + model re-download | `profiling/served/`, `docs/HANDOFF_2026-07-02.md` §2 |

## P1 — the capacity-gate figure (new, 2026-08-01)

Sweep: W ∈ {8…128} MB × T ∈ {1,16,32,64,128,256,512}, bf16 cuBLAS (tuned
baseline) vs W4A16 (Triton in-core dequant) vs W8A8 (INT8 IMMA). Clock-locked
1755 MHz, CUDA-graph timing (10 launches/graph, median of 30 replays), per-cell
mini-autotune, kernels numerically verified (rel err 3e-4 / 9e-3).

**The gate is exactly where CARM says it is.** At T ≤ 32 the win/loss sign
flips between W=32 and W=40 MB — bracketing the measured C_eff = 36 MB:

| | below gate (8–32 MB) | above gate (40–128 MB) |
|---|---|---|
| W8A8, T ≤ 32 | 0.49–0.70× | **1.19–1.46×** |
| W4A16, T ≤ 16 | 0.46–0.71× | **0.85–1.06×** (parity — dequant-ceiling-capped) |
| both, T ≥ 64 | < 1 everywhere | < 1 everywhere (compute-bound; the MARLIN batch-64 collapse) |

MARLIN reconciliation as reframed: the near field (L2-resident, no benefit)
and far field (HBM-streamed, byte-ratio benefit) are two regimes of one model.
Our Triton W4A16 reaches only parity in the far field because its measured
dequant ceiling (0.496 TB/s packed) binds — the 3–4× MARLIN far field requires
a ceiling-free kernel, which is what W8A8 (no in-core dequant) and tuned CUDA
Marlin (ceiling 423 TFLOPS, `cuda_validation/`) demonstrate.

**Model accuracy, regime-separated** (`profiling/gate/gate_mape.json`; split-mem
CARM form — weights at gated BW, act/out at HBM BW — adopted after beating the
lumped form 20.9% vs 25.3% on below-gate bf16):

- bf16 cuBLAS: 12–21% MAPE everywhere.
- Quantized Triton kernels: 8–39% at the dispatch-relevant decode T (1–32).
- **Honest failure:** at T ≥ 64 the Triton quant kernels fall progressively off
  their own rooflines (w4a16 65–81%, w8a8 43–53% MAPE) while cuBLAS stays ~17%.
  That is a kernel limitation (guardrail 5), consistent with the mechanisms
  taxonomy (BLOCK_M staircase, occupancy), confined to the regime where
  dispatch would never pick them anyway. The w4a16 packed operand never exceeds
  C_eff in this sweep (128 MB bf16 → 32 MB packed), so its operand-aware
  above-gate bucket is empty — the honest reading is that *its own* operand is
  always L2-servable even when the bf16 baseline's is not; the speedup above
  the gate comes from the baseline losing its L2 tier, which is the gate
  mechanism stated from the other side.

## What held

- **C_eff = 36 MB** (not nominal 50): the P1 flip brackets it independently of
  the 2026-06 NCU residency measurement and the W8A8 REPORT.
- **The W8A8 constructive result** generalizes from the June MLA shape to the
  full W×T plane and to tuned-CUDA MoE (1.80–1.94×, no cliff).
- **The dequant ceiling** as a predictive (not just diagnostic) term: it
  correctly predicts W4A16's parity cap in the far field.
- **The tuned-baseline guardrail**: crossover 263 (tuned) vs 601 (stock) stands
  as the paper's methodological warning.

## What broke / changed

1. **DIRECTION.md's P3 hypothesis was understated.** It predicted "C_eff under
   load is materially below 36 MB, the gate moves left." Measured (2026-07):
   contention is a **step function** — the L2-resident regime *dies* once the
   sum of hot working sets exceeds C_eff; it does not shift smoothly. The
   below-gate branch is a microbenchmark/small-model regime under serving.
   The paper now states this as a boundary condition, not a footnote.
2. **KV-cache traffic is not part of the capacity story.** DIRECTION.md's
   FP8-KV toggle experiment presumed KV reads contend for GEMM-visible L2;
   `kv_serving/` shows KV decode is not L2-limited at any working-set size and
   fp8-KV is bounded by its own kernel bandwidth ceiling (≤1.07× streamed,
   0.69–0.72× small-batch), with end-to-end ceiling ≤0.2%. Todo #2 is answered,
   but the mechanism is kernel BW, not capacity restoration.
3. **DIRECTION.md was drafted against the June repo state** and re-requested
   work that July had already done (P2/P3/P4). Reconciliation note added at the
   top of the committed `DIRECTION.md`; this file is the ledger.
4. **Roofline ≠ dispatch.** The mechanisms taxonomy (A–E, `mechanisms/`) means
   CARM alone cannot route; per-kernel predicates are part of the deliverable.
   This upgrades P6 from "cost model" to "routing layer spec."

## Next (priority order)

1. **Served A/B + accuracy** (`profiling/served/`, harnesses committed, never
   run): needs vLLM env restore + Qwen3.6-27B re-download (`HANDOFF_2026-07-02.md` §0).
2. **P5 portability**: generalize `measure_carm_params.py`; A100 fit-here,
   predict-there MAPE is the one number that separates a benchmark from a paper.
3. **Paper body surgery**: abstract/title/related-work are reframed; sections
   still tell the MLA story.
4. **P6 dispatch cost model** (analysis only, no GPU).

Session artifacts, 2026-08-01: commits `8f62ad1` (P1 + direction docs) and the
P0 reframe commit; figure `profiling/gate/fig_capacity_gate.png` with data in
`results_capacity_gate.json` and `gate_mape.json`; clocks were locked at
1755 MHz for the whole sweep and reset afterward.
