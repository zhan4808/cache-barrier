# PR: Fix per-channel INT8 W8A16 fused_moe — in-kernel dequant instead of host-side

Branch: `fix-w8a16-inkernel-dequant` (based on `origin/master`, single commit)
Target: `flagos-ai/FlagGems` `master`

## Problem

`fused_experts_impl(use_int8_w8a16=True)` host-dequantizes **all** expert
weights on every call:

```python
# Dequant INT8/INT4 weights (Triton can't do mixed-dtype dot)
if use_int8_w8a16 or use_int4_w4a16:
    w1 = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1)...
```

On Mixtral-8x7B shapes (E=8, H=4096, I=14336) this moves ~10 GB per call —
a flat ~8.5 ms on H100 — making W8A16 **6–22× slower than bf16 at every
token count**. The comment is also outdated: mixed-dtype `tl.dot` is not a
Triton limitation; the non-fused `fused_moe_kernel` already converts the
int8 tile in-loop (`b.to(compute_type)`) and applies the per-N scale once
on the fp32 accumulator.

The actual blocker was that the `FUSE_SILU` fused gate/up path had no
W8A16 handling (fp16 × int8 `tl.dot` fails to compile), so the host
fallback covered for it.

## Fix

1. Add W8A16 support to the `FUSE_SILU` two-pass path: convert the int8
   gate/up tiles in both K-loops, apply per-channel gate/up scales once on
   the accumulators (mirrors the existing non-fused logic).
2. Disable `PAIR_GATE_UP_DOT` for W8A16 (the paired dot has no
   conversion/scale handling); the two-pass fused path is used instead.
3. Route per-channel INT8 W8A16 through the kernel. The host-dequant
   fallback remains only for INT4 and grouped-scale INT8, where the WNA16
   kernel expects offset-binary packed layouts this entry point does not
   prepare.

## Measurements (H100 80GB, fp16, Mixtral-8x7B shapes, top-k=2)

| T | bf16 | W8A16 before | W8A16 after | after vs bf16 |
|---|---|---|---|---|
| 1 | 382 µs | ~8.5 ms | 357 µs | 1.07× |
| 16 | 992 µs | ~8.6 ms | 584 µs | **1.70×** |
| 64 | 1245 µs | ~8.7 ms | 732 µs | **1.70×** |
| 128 | 1017 µs | ~8.9 ms | 845 µs | 1.20× |
| 256 | 1252 µs | ~9.3 ms | 1220 µs | 1.03× |
| 512 | 5333 µs | ~10.2 ms | 2024 µs | **2.63×** |

Correctness: output relative error vs fp16 reference ~0.012 across naive /
sorted / fused-silu dispatch paths (T = 2, 8, 64), consistent with
per-channel INT8 weight quantization error itself.

Repro scripts: `cache-barrier/profiling/fused_moe/bench_fused_moe_mxq.py`
(github.com/zhan4808/cache-barrier).

## Status

- Fork: `git@github.com:zhan4808/FlagGems.git`
- Branch pushed: `fix-w8a16-inkernel-dequant` (commit `bfddbfddc`)
- **Open PR:** https://github.com/flagos-ai/FlagGems/compare/master...zhan4808:fix-w8a16-inkernel-dequant?expand=1

Sign in on GitHub, click **Create pull request**, paste this file as the body (minus this Status section).
