# The L2 Cache Barrier in MLA Serving

Code and profiling data for the paper **"The Hidden Bottleneck in MLA Serving: Reconstruction GEMMs, INT4 Quantization, and the L2 Cache Barrier"** (Robert Zhang).

## Key Claim (revised 2026-06 after methodology audit)

INT4 quantization of MLA reconstruction weights fails to outperform FP16 cuBLAS for two stacked reasons:

1. **Partial L2 residency** — in steady state, ~75% of the 16 MB weight matrix is served from L2 (measured with `ncu --cache-control none`: only 4 MB read from DRAM per launch). Effective serving bandwidth is **3.5–4.6 TB/s** (not the ~12 TB/s aggregate L2 figure), and the effective residency capacity is **~32–40 MB, not the nominal 50 MB** — residency collapses between 32 and 40 MB.
2. **The INT4 kernel is dequantization-compute-bound at every size** — a weight-rotation intervention that forces FP16 weights out of L2 (fixed 16 MB shape, working set > L2) shows INT4 *still* loses (1.12x slower). Above the residency cliff INT4 reaches parity at best; it never wins.

L2 residency therefore removes most of the theoretical INT4 upside, and dequantization overhead destroys the rest. An earlier version of this claim attributed the failure solely to L2 residency at 50 MB / 12 TB/s; see `profiling/validation/` for the audit, corrected measurements, and causal experiments (CUDA-graph timing, weight rotation, warm-state NCU).

> **Profiling caveat:** NCU's default `--cache-control all` flushes GPU caches before every replayed launch. All cache-residency conclusions in this repo are now based on `--cache-control none` warm-loop counters and timing-only interventions; cold-cache NCU sweeps are retained but must not be read as residency evidence.

## Repo Structure

```
kernels/                Triton/PyTorch transformer kernels and benchmarks
profiling/              Microbenchmarks, NCU profiling scripts, and results
profiling/validation/   2026-06 methodology audit: causal experiments, corrected figures, REPORT.md
paper/                  LaTeX source and figures
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

See `profiling/RUNBOOK.md` for full instructions and `profiling/RESULTS.md` for a summary of findings.

## Paper

The LaTeX source is in `paper/`. Build with:
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
