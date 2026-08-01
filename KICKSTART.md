# KICKSTART — agent prompt for the next work session

Companion to `DIRECTION.md` (read that first; it has the reasoning). This file carries
the operational plan and the methodology guardrails.

## The thesis in one sentence

Quantization's speedup is gated by a cache-capacity condition that standard roofline
cannot express; a three-parameter, microbenchmarkable model predicts the precision
crossover on unseen architectures cheaply enough to evaluate at dispatch time.

## The gate

```
W     = weight working set bytes
C_eff = effective last-level-cache capacity (measured, NOT nominal)

W < C_eff   → weights already LLC-served; quantization saves traffic you weren't paying
              → speedup ≈ 1.0, or < 1.0 if the kernel dequantizes in-core
W >> C_eff  → weights stream from HBM; quantization saves real traffic
              → speedup → min(bytes_ratio, dequant_ceiling / T_compute)
```

MARLIN's reported ~4× at batch ≤32 is the second branch. This repo's null result is the
first branch. They are two regimes of one model, not a contradiction.

## Measured hardware parameters (H100 80GB SXM5, `profiling/carm_model.json`)

```
effective_l2_capacity_mb    36.0     ← NOT the nominal 50 MB
bw_l2_gemm_tbs               6.3
bw_hbm_tbs                   3.146
r_dequant_tbs                0.496
int4_incore_ceiling_tflops  31.7
peak_tflops                985–989
model MAPE                  10.2% (FP16) / 18.2% (INT4); 12.2% on CUDA Marlin fused_moe FP8
```

## Task order

P1 gate figure (`profiling/gate/`) ★ → P2 dense shape census → P3 serving-load C_eff →
P4 compute quantization → P5 portability harness → P6 dispatch cost model.
Full specs in `DIRECTION.md` §6. Status as of 2026-08-01: P2/P3/P4 substantively done
in the 2026-07-01→02 session (`dense_qwen/`, `kv_serving/`, `mechanisms/`,
`cuda_validation/bench_moe_w8a8.py`); P1 done 2026-08-01 (`profiling/gate/`).

## Guardrails — learned the hard way in this repo. Violating them invalidates results.

1. **NCU cache control.** `ncu` defaults to `--cache-control all`, which flushes GPU
   caches before every replayed launch. All residency conclusions must use
   `--cache-control none` with warm-loop counters. Cold-cache sweeps exist in the repo
   but must never be read as residency evidence.
2. **Never use per-launch CUDA-event timing for kernel-level claims.** It has a
   ~15.5 µs floor; the "flat" FP16 region below 32 MB in the original sweep was that
   floor, not L2 serving. Use CUDA graphs. Graph floor is ~2.8 µs.
3. **Always compare against a properly tuned baseline.** This repo's most expensive
   mistake: the crossover measured 601 tokens against vLLM's stock under-tuned bf16
   (`GROUP_SIZE_M=1`) but 263 against a properly tuned baseline (`GROUP_SIZE_M=16`),
   matching the roofline prediction of 334. A flattering baseline manufactures a
   quantization win that isn't there. Tune both sides, and report which you used.
4. **Lock clocks.** `nvidia-smi -lgc 1755` before measurement, `nvidia-smi -rgc`
   after. Note drift control in any results file.
5. **Triton is a confound.** A finding on Triton kernels may be a Triton limitation,
   not a hardware effect. Any headline claim must be reproduced on tuned CUDA
   (vLLM Marlin / cuBLAS / CUTLASS). `profiling/cuda_validation/` did this once —
   follow that pattern.
6. **Nominal ≠ effective.** H100 L2 is nominally 50 MB; effective residency capacity
   is 36 MB. Never use datasheet capacity in the model.
7. **Report regime-separated error.** Blended MAPE across the gate hides where the
   model is weak. Report below-gate and above-gate separately.
8. **State negative results plainly.** "CARM predicts no benefit here, and here's the
   gate that says so" is a legitimate result. Do not bury or soften it.

## Working style

- Commit incrementally with descriptive messages, matching the existing log style
- Write results to JSON alongside every figure; never leave a plot without its data
- If a result contradicts `DIRECTION.md`, say so loudly — the direction is a
  hypothesis, not a conclusion to defend
- Before claiming any speedup, verify the baseline is tuned (Guardrail 3)
