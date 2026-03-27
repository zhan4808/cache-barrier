# The L2 Cache Barrier in MLA Serving

Code and profiling data for the paper **"The Hidden Bottleneck in MLA Serving: Reconstruction GEMMs, INT4 Quantization, and the L2 Cache Barrier"** (Robert Zhang).

## Key Claim

INT4 quantization of MLA reconstruction weights fails to outperform FP16 cuBLAS because the 16 MB weight matrix fits inside the H100's 50 MB L2 cache. Weights are served at L2 bandwidth (~12 TB/s) rather than HBM bandwidth (3.35 TB/s), making quantization's HBM savings irrelevant. The speedup only materializes once weights exceed L2 capacity.

## Repo Structure

```
kernels/          Triton/PyTorch transformer kernels and benchmarks
profiling/        Microbenchmarks, NCU profiling scripts, and results
paper/            LaTeX source and figures
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

See `profiling/RUNBOOK.md` for full instructions and `profiling/RESULTS.md` for a summary of findings.

## Paper

The LaTeX source is in `paper/`. Build with:
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
