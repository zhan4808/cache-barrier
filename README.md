# The L2 Cache Barrier in MLA Serving

Code and profiling data for the paper **"The Hidden Bottleneck in MLA Serving: Reconstruction GEMMs, INT4 Quantization, and the L2 Cache Barrier"** (Robert Zhang).

## Key Claim (revised 2026-06 after methodology audit)

INT4 quantization of MLA reconstruction weights fails to outperform FP16 cuBLAS for two stacked reasons:

1. **Partial L2 residency** — in steady state, ~75% of the 16 MB weight matrix is served from L2 (measured with `ncu --cache-control none`: only 4 MB read from DRAM per launch). Effective serving bandwidth is **3.5–4.6 TB/s** (not the ~12 TB/s aggregate L2 figure), and the effective residency capacity is **~32–40 MB, not the nominal 50 MB** — residency collapses between 32 and 40 MB.
2. **The INT4 kernel is dequantization-compute-bound at every size** — a weight-rotation intervention that forces FP16 weights out of L2 (fixed 16 MB shape, working set > L2) shows INT4 *still* loses (1.12x slower). Above the residency cliff INT4 reaches parity at best; it never wins.

L2 residency therefore removes most of the theoretical INT4 upside, and dequantization overhead destroys the rest. An earlier version of this claim attributed the failure solely to L2 residency at 50 MB / 12 TB/s; see `profiling/validation/` for the audit, corrected measurements, and causal experiments (CUDA-graph timing, weight rotation, warm-state NCU).

**Constructive confirmation (W8A8, 2026-06):** removing dequantization from the inner loop entirely — INT8 `tl.dot` → int32 accumulator on Hopper IMMA tensor cores, scales applied once on the accumulator — produces the **first quantized kernel that beats cuBLAS FP16**: 1.4–1.5× at bs=1 when weights exceed the ~36 MB effective L2 capacity, and 0.70× (still losing) when they are L2-served. The win/loss boundary sits exactly at the measured residency capacity. See `profiling/w8a8/REPORT.md`.

**Generalization (FlagGems fused_moe):** the shipped per-channel INT8 W8A16 path host-dequantized all expert weights on every call (flat ~8.5 ms); with the in-kernel fix it is 1.07–2.63× over bf16 on Mixtral shapes. Patch + PR body in `profiling/fused_moe/`. The same CARM regime structure (weight-byte-bound vs conversion-bound) governs the token-count crossover.

> **Profiling caveat:** NCU's default `--cache-control all` flushes GPU caches before every replayed launch. All cache-residency conclusions in this repo are now based on `--cache-control none` warm-loop counters and timing-only interventions; cold-cache NCU sweeps are retained but must not be read as residency evidence.

## Repo Structure

```
kernels/                Triton/PyTorch transformer kernels and benchmarks
profiling/              Microbenchmarks, NCU profiling scripts, and results
profiling/validation/   2026-06 methodology audit: causal experiments, corrected figures, REPORT.md
profiling/w8a8/         W8A8 INT8-MMA BMM kernel: the constructive result (REPORT.md, figure)
profiling/fused_moe/    FlagGems mixed-precision fused_moe analysis, W8A16 fix patch, PR body
paper/                  LaTeX source and figures (dist/arxiv_submission.tar.gz = arXiv package)
```

## Requirements

- NVIDIA H100 80 GB SXM5 (L2 cache = 50 MB); some experiments also run on A100
- PyTorch ≥ 2.1 with CUDA
- Triton ≥ 3.0

```bash
pip install torch triton
```

## Reproducing Key Experiments

**L2 barrier sweep** (scales weight matrix from 8 MB → 128 MB across the 50 MB L2 boundary):
```bash
cd profiling
python bench_l2_barrier.py
```

**INT4 batched GEMM benchmark:**
```bash
python bench_int4_bmm.py
```

**MLA reconstruction profiling:**
```bash
python profile_mla_reconstruction.py
```

**NCU kernel profiling** (requires `ncu` on PATH):
```bash
bash ncu_profile.sh
python analyze_ncu.py
```

**End-to-end profiling:**
```bash
bash profile_e2e.sh
```

**Methodology-audit causal experiments** (CUDA-graph timing, weight rotation, warm-state NCU):
```bash
cd profiling/validation
python diag_l2_residency.py        # launch floor, rotation, graph sweep, eviction
python graph_sweep_int4.py         # graph-timed INT4 size sweep
python graph_rotation.py           # fixed-shape residency intervention
python make_figures.py             # audit figures (figures/)
```

**Measured cache-aware roofline (CARM):**
```bash
cd profiling
python measure_carm_params.py      # measure tier bandwidths, fixed costs, operating points
python plot_cache_aware_roofline.py  # build + validate model, render paper figure
```

**W8A8 INT8-MMA kernel (constructive result):**
```bash
cd profiling/w8a8
python w8a8_bmm.py                 # correctness check
python bench_w8a8.py               # graph-timed bs + weight-size sweeps vs cuBLAS/W4A16
python plot_w8a8.py                # figure
```

See `profiling/RUNBOOK.md` for full instructions and `profiling/RESULTS.md` for a summary of findings.

## Paper

The LaTeX source is in `paper/`. Build with:
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
