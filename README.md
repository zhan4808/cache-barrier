# When Quantization Stops Paying: The Capacity Gate

Code and profiling data for the paper **"When Quantization Stops Paying: A Capacity-Gated Roofline Model for Precision Dispatch on GPUs"** (Robert Zhang). Direction: `DIRECTION.md`; methodology guardrails: `KICKSTART.md`.

## The claim

Whether a low-precision GEMM kernel beats its high-precision counterpart is governed by a **cache-capacity condition standard roofline analysis cannot express** — roofline has no capacity term:

```
W     = weight working set bytes (at the operand's own precision)
C_eff = effective last-level-cache capacity (measured: 36 MB on H100, NOT the nominal 50)

W < C_eff   → weights already L2-served: quantization saves traffic you weren't paying
              → speedup ≈ 1.0, and < 1.0 if the kernel dequantizes in-core
W >> C_eff  → weights stream from HBM: quantization saves real traffic
              → speedup → min(bytes_ratio, dequant_ceiling / T_compute)
```

MARLIN's ~4× at batch ≤32 is the second branch; this repo's L2-resident null results are the first. **Two regimes of one model, not a contradiction.**

## Figure 1 — the gate, measured (`profiling/gate/`)

2D sweep (weight working set 8–128 MB × tokens 1–512, clock-locked, CUDA-graph timed) against tuned cuBLAS bf16:

- **The sign flip sits at the measured capacity.** At decode token counts (T ≤ 32), W8A8 (INT8 IMMA, no in-core dequant) goes from **0.67–0.70× below** the gate to **1.2–1.46× above** it, flipping between 32 and 40 MB — exactly the measured `C_eff = 36 MB`.
- **The dequant ceiling caps the far field.** W4A16 (in-core dequant, `r_dequant = 0.496 TB/s`) flips at the same boundary but only reaches parity — CARM predicts both the flip location and the cap. Ceiling-free kernels (W8A8 here; CUDA Marlin with its 423 TFLOPS ceiling in `profiling/cuda_validation/`) are what reach the byte-ratio far field.
- **Above T ≈ 64 the gate washes out** into the compute-bound regime where weight-only quantization loses everywhere — the MARLIN batch-64 collapse, predicted by the same model.
- CARM (three microbenchmarked parameters: `C_eff`, capacity-gated bandwidth, dequant throughput) predicts the tuned cuBLAS baseline to 12–21% MAPE everywhere and the quantized Triton kernels to 8–39% in the dispatch-relevant decode regime. At T ≥ 64 the Triton quant kernels fall off their own rooflines (kernel limitation, reported per-regime in `profiling/gate/gate_mape.json` — guardrail 7).

## Boundary conditions (measured 2026-07: `profiling/dense_qwen/`, `profiling/kv_serving/`, `profiling/mechanisms/`)

1. **Serving contention kills the L2-resident regime as a step function.** Once the sum of hot working sets exceeds `C_eff`, the L2 tier dies — isolated-microbenchmark crossovers (including ours) are optimistic in production. The gate's "quantization does not pay" branch is a microbenchmark/small-model regime on H100 today, and an *expanding* one as LLC grows (A100 40 MB → H100 50 → B200 ~126).
2. **A roofline is necessary but not sufficient for dispatch.** Five kernel-implementation mechanisms (tile-starvation floors, wave-quantization bands, dispatch splits, occupancy floors, per-tile re-dequant staircases) each independently flip win↔lose. Dispatch = CARM + per-kernel predicates.
3. **On real dense shapes (Qwen3.6-27B)** the operand-aware gate reproduces per-projection win/loss flips; matched-precision W8A8 sustains **1.80–1.94× over tuned bf16 with no crossover**; KV-cache reads are outside the capacity story entirely.

## Reproducing key experiments

Requirements: H100 80 GB SXM5 (exclusive, no MIG), PyTorch ≥ 2.1, Triton ≥ 3.0. Lock clocks first: `sudo nvidia-smi -lgc 1755` (reset with `-rgc`).

```bash
cd profiling/gate && python bench_capacity_gate.py && python plot_gate.py   # Figure 1
cd profiling && python measure_carm_params.py                               # the 3 parameters
cd profiling/w8a8 && python bench_w8a8.py                                   # constructive W8A8 result
cd profiling/dense_qwen && python bench_l2_boundary.py                      # real-shape operand gate
cd profiling/validation && python diag_l2_residency.py                      # methodology audit
```

See `profiling/RUNBOOK.md` for full instructions and per-directory `RESULTS.md`/`REPORT.md` for findings.

## Repo structure

```
DIRECTION.md              Research direction (2026-07 pivot), claim triage, phase plan
KICKSTART.md              Operational plan + methodology guardrails
profiling/gate/           ★ Figure 1: the capacity-gate 2D sweep, CARM overlay, MAPE
profiling/carm_model.json Measured H100 CARM parameters (the three-parameter model)
profiling/dense_qwen/     Dense Qwen3.6-27B shapes: all three regimes on real projections
profiling/kv_serving/     KV decode: not L2-limited; fp8-KV kernel ceilings
profiling/mechanisms/     The 9-mechanism routing taxonomy (per-kernel predicates)
profiling/cuda_validation/  Tuned-CUDA reproduction (vLLM Marlin/cuBLAS) — Triton is a confound
profiling/w8a8/           W8A8 INT8-MMA kernel: the constructive gate result
profiling/served/         Served A/B + accuracy harnesses (staged; not yet run)
profiling/validation/     2026-06 methodology audit (causal experiments, corrected figures)
profiling/fused_moe/      FlagGems W8A16 host-dequant fix (upstream patch)  [historical]
kernels/                  Triton/PyTorch kernels and benchmarks              [historical]
paper/                    LaTeX source; title/abstract reframed 2026-08, body mid-surgery
```

## Paper

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Historical note:** this repo began as an MLA-reconstruction/INT4 study ("The Hidden Bottleneck in MLA Serving"). Dr. Xiao's 2026-07-01 verification closed the sparse/MLA direction (those operators are not cache-capacity-limited); MLA reconstruction survives as one case study of the gate. See `DIRECTION.md` §3 for the claim triage.
