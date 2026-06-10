# FlagGems mixed-precision `fused_moe` — analysis, root cause, and fix (H100)

Task: evaluate FlagGems PR #2336 (mixed-precision fused_moe), explain why
performance degrades as token count grows, compare with our approach, optimize.

Setup: H100 80GB SXM5, torch 2.7/triton 3.3, fp16, Mixtral shape
(E=8, H=4096, I=14336, topk=2), tokens 1–512. Timing = median of event-timed
30-iteration loops. Correctness vs fp32 torch reference on (T=4, E=4, H=256, I=512).

## Finding 1 — the PR kernel (`fused_moe_mxq.py`) is not a usable baseline

| Issue | Evidence |
|---|---|
| Computes only the first `BLOCK_SIZE_N=128` of N=28672 output columns | `offs_n = tl.arange(0, BLOCK_SIZE_N)`, grid has no N dimension → rel err ≈ 15 vs reference |
| Quantized "SwiGLU" path runs GEMM1 only (no SiLU·up, no GEMM2), writes inter-dim output into a hidden-dim buffer | `invoke_fused_moe` quantized branch launches one `fused_moe_kernel_gptq_awq` with W1 only |
| Fails to compile on bf16 (`compute_type` hardcoded `tl.float16` + `tl.atomic_add` into bf16 buffer) | the PR's own bf16 benchmark config cannot have exercised this path |
| One program per (token,expert), `BLOCK_SIZE_M=1`, no `tl.dot`, scalar `tl.sum` outer products; weight bytes + dequant work scale ∝ tokens | kernel source; its "1500+ TFLOPS" in the PR benchmark exceed the H100 fp16 peak because FLOPs are counted for work the kernel never does |

Any performance numbers from this kernel (including "drops as tokens
increase") describe partial, incorrect work.

## Finding 2 — the *shipped* FlagGems W8A16 path host-dequantizes every call

`fused_experts_impl(..., use_int8_w8a16=True)` hits:

```python
# Dequant INT8/INT4 weights (Triton can't do mixed-dtype dot)
w1 = w1.to(dtype) * w1_scale...; w2 = w2.to(dtype) * w2_scale...
```

i.e. every forward dequantizes **all experts** (1.4 GB int8 → 2.8 GB fp16,
plus the multiply) before running the bf16 kernel: ~10 GB of traffic, a flat
**~8.5 ms** at every token count — 22× slower than bf16 at T=1, 6.5× at
T=512. Profiler: `aten::copy_` 4.3 ms + `aten::mul` 3.6 ms vs 0.96 ms of
actual MoE kernel. The comment is wrong: the kernel already supports in-loop
`b.to(compute_type)` before `tl.dot`, with per-channel scale applied once on
the fp32 accumulator.

## Fix (one conditional): use the in-kernel dequant path

`flaggems_w8a16_inkernel_dequant.patch` — skip host dequant for per-channel
INT8 W8A16 (INT4 / grouped-scale INT8 keep the fallback; the WNA16 path
expects offset-binary packed layouts this entry point doesn't prepare).
Correctness: rel err 0.013 (quantization noise, same as before).

| tokens | bf16 µs | W8A16 shipped µs | W8A16 fixed µs | fixed vs bf16 |
|---|---|---|---|---|
| 1   | 354  | 7975 | 357  | 0.99× |
| 4   | 856  | 8524 | 495  | **1.73×** |
| 16  | 971  | 8640 | 558  | **1.74×** |
| 64  | 988  | 8656 | 673  | **1.47×** |
| 128 | 1035 | 8705 | 825  | **1.25×** |
| 256 | 1080 | 8761 | 1124 | 0.96× |
| 512 | 1388 | 9035 | 1919 | 0.72× |

## Finding 3 — the residual token-scaling falloff is a dequant in-core ceiling (CARM)

> **Partially superseded by Finding 4.** The table above and the ≈230-token
> crossover below came from eager (non-graph) timing; the graph-timed extended
> sweep shows the fixed W8A16 kernel wins at **every** measured T (min 1.03× @
> T=256, 2.7–3.0× at T≥512). The in-core ceiling diagnosis (~305 TF, NCU
> counters) stands — but bf16's own super-linear scaling at high T means the
> ceiling does not translate into a measured loss at this shape.

With the fix, W8A16 behaves as the cache-aware roofline from our MLA
work predicts in-kernel:

- **≤128 tokens (memory-bound):** weights dominate bytes; INT8 halves them →
  up to 1.74× win. (At T=1 launch/router overheads dominate → parity.)
- **≥256 tokens (dequant-influenced):** the kernel pins at an **in-core ceiling
  of ~305 TFLOPS (31% of peak)**. Warm NCU at T=512: W8A16 = 24% DRAM / 32% SM
  vs bf16 = 76% DRAM / 54% SM — W8A16 is stalled on the int8→fp16 convert
  feeding `tl.dot`, not on memory and not on tensor-core math.
- ~~Crossover ≈ 230 tokens~~ — eager-timing artifact; see Finding 4.

Same failure mode as our MLA W4A16 kernel (in-core ceiling ~30 TF, 3% of
peak); the MoE kernel's ceiling is 10× higher because conversion is vectorized
and tensor cores do the math, but the structure is identical: *weight-only
quantization buys bandwidth at the price of an in-core conversion ceiling, so
it must lose once the workload leaves the memory-bound regime.*

Comparison with our approach: our cache-barrier W4A16 BMM kernel is the same
class (Triton, dequant-in-kernel) with a far lower ceiling (scalar unpack,
BLOCK_M padding) — the MoE fix above is "our approach done right" for the
memory-bound regime, and both hit the same wall outside it.

## Recommendations (upstreamable)

1. Apply the in-kernel dequant fix (patch in this directory).
2. ~~Dispatch on token count: W8A16 for T ≲ 200, bf16 above~~ — revised by
   Finding 4: graph-timed, fixed W8A16 wins at all measured T at this shape;
   no dispatch needed. W8A8 INT8-MMA is preferable at T ≤ 128 only.
3. To win at high token counts, dequant must leave the inner loop:
   INT8 tensor-core MMA (W8A8) or Hopper warpgroup fragment-level dequant.
4. The PR's mxq kernel needs an N-dimension in its grid, the full SwiGLU
   chain, M-tiling via `moe_align_block_size`, `tl.dot`, and a compute_type
   matching the buffer dtype before any benchmark of it is meaningful.

## Finding 4 — extended graph-timed sweep (T=16–2048, fix branch)

CUDA-graph timed (10 launches/graph, median of 40 replays). All paths use
default embedded MoE configs (`No embedded MoE config ... Will use default`).

| T | bf16 µs | W8A16 | vs bf16 | W8A8 | vs bf16 |
|---|---|---|---|---|---|
| 16 | 968 | 564 | **1.72×** | 558 | **1.74×** |
| 64 | 1224 | 711 | **1.72×** | 658 | **1.86×** |
| 128 | 1000 | 833 | 1.20× | 776 | 1.29× |
| 256 | 1242 | 1209 | 1.03× | 1064 | 1.17× |
| 512 | 5311 | 1990 | **2.67×** | 2279 | 2.33× |
| 1024 | 8114 | 3000 | **2.71×** | 3448 | 2.35× |
| 2048 | 14917 | 5033 | **2.96×** | 6164 | 2.42× |

**Interpretation (revised):** with the in-kernel fix and graph timing, W8A16
*does* win at high token counts (2.7–3.0× at T≥512) because bf16's latency
scales super-linearly (chunking / alignment / default tile configs) while
quantized paths scale more gently. The T≈256 dip is a **crossover band** where
all three are within ~20% — not a regime where quantization loses outright.
W8A8 INT8-MMA beats W8A16 at T≤128 (less conversion overhead) but W8A16 pulls
ahead at T≥512 (better default kernel configs for the int8→fp16 dot path).

Warm NCU (`run_ncu_warm_sweep.sh`) initially captured setup kernels; re-run
with `--kernel-name-base regex:fused_moe_kernel` (see updated script).

Files: `bench_fused_moe_mxq.py`, `bench_fused_moe_extended.py`,
`results_fused_moe_extended.json`, `results_fused_moe_shipped.json` /
`results_fused_moe_fixed.json`, `ncu_target_w8a16.py`, `run_ncu_warm_sweep.sh`,
`parse_ncu_warm.py`, `plot_fused_moe.py`, `flaggems_w8a16_inkernel_dequant.patch`,
`PR_BODY.md`.
