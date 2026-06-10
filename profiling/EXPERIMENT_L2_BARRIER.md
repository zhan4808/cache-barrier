# L2 Cache Barrier Experiment — Instructions for GPU Instance

## Context

This experiment is the final piece of evidence for a systems paper about MLA (Multi-head Latent Attention) serving on H100 GPUs.

**The claim we need to prove:** INT4 quantization of MLA reconstruction weights fails to outperform FP16 cuBLAS because the 16 MB weight matrix fits inside the H100's 50 MB L2 cache. Weights are served from L2 (~12 TB/s), not HBM (3.35 TB/s), making HBM bandwidth savings from quantization irrelevant.

**The experiment:** Scale the weight matrix size from 8 MB to 128 MB, crossing the 50 MB L2 boundary. If the L2 hypothesis is correct, the INT4/FP16 performance ratio should *improve* (move toward or above 1.0x) once weights exceed L2 capacity and FP16 must fall back to HBM.

## Requirements

- **GPU:** NVIDIA H100 80GB SXM5 (must be H100 — the L2 cache size is 50 MB)
- **Software:** PyTorch >= 2.1 with CUDA, Triton >= 3.0
- **Memory:** ~10 GB GPU memory (largest config: 128 MB weights × 2 for FP16 + INT4)

## Setup

```bash
# Clone the repo
git clone https://github.com/robertzhang/sglang.git
cd sglang

# Install dependencies (if not already available)
pip install torch triton

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA H100 80GB HBM3
```

## Running the experiment

```bash
cd profiling
python bench_l2_barrier.py
```

**Expected runtime:** ~5-10 minutes (12 weight sizes × 2 batch sizes × ~15s each).

The script sweeps `d_lora` (the latent dimension) across these values:
`[256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]`

Corresponding FP16 weight sizes (H=128, d_nope=128):
- d_lora=256 → 8 MB (well inside L2)
- d_lora=512 → 16 MB ← **current MLA config**
- d_lora=1536 → 48 MB (just under L2)
- d_lora=1792 → 56 MB (just over L2) ← **critical boundary**
- d_lora=4096 → 128 MB (well outside L2)

## Output files

The script produces:
1. `results_l2_barrier.csv` — one row per (batch_size, d_lora) with columns: `batch_size, d_lora, weight_mb, fits_l2, fp16_ms, int4_ms, int4_fp16_ratio`
2. `results_l2_barrier.json` — same data in JSON with GPU metadata
3. Stdout summary with analysis

## What to look for

**If the L2 hypothesis is correct**, you should see:

1. For weight sizes **below 50 MB**: INT4/FP16 ratio stays around 1.5–2.5x (INT4 is *slower* — ratio > 1.0 means INT4 takes longer)
2. For weight sizes **above 50 MB**: the ratio drops toward or below 1.0x (INT4 starts catching up or winning)
3. The transition should happen near d_lora=1536–1792 (48–56 MB)

**Note:** INT4 may not *fully* beat FP16 even above L2, because dequantization overhead and cuBLAS dispatch advantages still apply. The key signal is a **clear improvement in INT4/FP16 ratio** once weights exceed L2 capacity. Even a shift from 2.0x to 1.3x would confirm the hypothesis.

**If the hypothesis is wrong**, the ratio would stay flat across all weight sizes.

## Interpretation corrections (2026-06 audit — read before citing this experiment)

The sweep above reproduces, but three things in the original interpretation are wrong; see `validation/` for the experiments:

1. **The knee sits at 32–40 MB, not 48–56 MB.** Warm-state NCU (`--cache-control none`) shows DRAM re-reads collapse at 16–32 MB weights and snap to ~100% of weight size from 40 MB onward. Effective LRU residency capacity is ~36 MB. A knee "at the 50 MB boundary" should not be expected or claimed.
2. **The ratio improvement is necessary but not sufficient evidence.** The event-timed ratio decline conflates (a) FP16 leaving a ~15.5 µs launch-overhead floor, (b) FP16's serving tier changing L2→HBM, and (c) both kernels amortizing fixed costs with size. The clean causal test is the *weight-rotation intervention at fixed shape* (`validation/diag_l2_residency.py`, `graph_rotation.py`): same kernel, same size, residency toggled by cycling weight copies.
3. **This experiment alone cannot establish L2 residency as THE root cause of INT4's failure** — rotation shows INT4 loses even with FP16 forced to HBM. Cite the residency effect as removing most of the theoretical INT4 upside, with dequant compute consuming the remainder.

Also note: cuBLAS dispatches different `nvjet` tile variants across this sweep, so the FP16 curve mixes kernel implementations; and ratios are stack-sensitive (torch 2.9/triton 3.5: 1.86→1.08x; torch 2.7/triton 3.3: 2.75→1.58x).

## After running

Please provide:
1. The full stdout output
2. The `results_l2_barrier.json` file contents
3. A brief note on whether the ratio improved past the 50 MB boundary

## Troubleshooting

- **OOM:** Reduce `D_LORA_SWEEP` — remove the 4096 entry if GPU memory is tight
- **Triton compile errors at large N:** The kernel's `BLOCK_N=64` should handle all sizes, but if errors occur at very large d_lora, try reducing the sweep
- **Not H100:** The experiment is specific to H100's 50 MB L2. On A100 (40 MB L2) the boundary shifts to ~d_lora=1280. On L40 or other GPUs, adjust expectations based on their L2 size. The script will print the GPU name for reference.
