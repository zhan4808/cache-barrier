# W8A8 INT8-MMA kernel for MLA reconstruction — the constructive result (H100)

The audit established that W4A16/W8A16 Triton kernels lose to cuBLAS FP16
because per-element dequantization in the inner loop imposes an in-core
ceiling (~30 TF for the MLA W4A16 kernel). This kernel removes that ceiling:

- weights: static symmetric per-(head, channel) INT8
- activations: dynamic symmetric per-(head, token) INT8 (one small kernel)
- inner loop: `tl.dot(int8, int8) -> int32` (Hopper IMMA tensor cores)
- epilogue: `acc * a_scale[m] * w_scale[n]` — one multiply per output element

Accuracy: rel err ~0.009 vs FP16 (both GEMM operands quantized).
Timing: CUDA-graph (20 launches/graph, median of 50 replays); W8A8 numbers
include the activation-quant kernel.

## Result 1 — first quantized kernel to beat cuBLAS FP16 (bs=1 decode, HBM-served)

| weights | FP16 µs | W8A8 µs | W4A16 µs | W8A8 speedup |
|---|---|---|---|---|
| 8 MB   | 3.6  | 5.1  | 5.5  | 0.70× |
| 16 MB (MLA) | 4.9 | 6.9 | 9.0 | 0.70× |
| 32 MB  | 7.6  | 10.4 | 15.6 | 0.73× |
| 48 MB  | 21.7 | 14.3 | 22.5 | **1.52×** |
| 64 MB  | 27.5 | 17.9 | 29.0 | **1.53×** |
| 96 MB  | 38.6 | 28.2 | 41.8 | **1.37×** |
| 128 MB | 49.1 | 34.4 | 55.7 | **1.43×** |

The flip sits exactly at the measured effective L2 capacity (~36 MB):
below it FP16's L2 residency removes the byte-savings upside (W8A8 0.7×);
above it the 2× weight-byte reduction converts to latency because INT8 MMA
adds no in-core dequant cost. This confirms the paper's causal claim from the
constructive side: *fix the dequant path and the L2 boundary becomes exactly
the regime boundary the cache-aware roofline predicts.*

## Result 2 — at the L2-resident 16 MB MLA shape, no quantized kernel wins at any bs

W8A8 is 0.60–0.71× at bs 1–16 and worse at large bs (cuBLAS nvjet with TMA is
extremely strong at large M; K=128 gives INT8 tiles little MMA work per byte).
W8A8 nonetheless beats W4A16 at every point (1.4–2.9×).

## Result 3 — FlagGems MoE W8A8 (`use_int8_w8a8`) does not rescue high token counts

On the Mixtral shape, FlagGems' W8A8 INT8-MMA path: 1.24–1.44× over bf16 at
T=4–128 (slightly below fixed W8A16), but 0.50–0.70× at T≥256 — the
`fused_moe_kernel` itself runs 2.03 ms at T=512 vs bf16's 1.37 ms. At high T
the MoE GEMMs are compute-bound, where INT8's theoretical 2× MMA throughput
would need a tuned pipeline (vLLM-style per-shape configs, missing here:
"No embedded MoE config ... int8_w8a8" falls back to defaults) to materialize.
Conclusion for FlagGems today: fixed W8A16 for T≲230, bf16 above; W8A8 needs
config tuning upstream before it changes that picture.

## Deployment rule (CARM-derived, all measured)

| regime | best choice |
|---|---|
| working set < ~36 MB (L2-served), any bs | FP16/bf16 — do not quantize for speed |
| working set > ~36 MB, small bs (weight-byte-bound) | W8A8 INT8-MMA (1.4–1.5×) |
| compute-bound (large bs / high T) | FP16/bf16 (or W8A8 only with tuned pipelines) |

Files: `w8a8_bmm.py` (kernel + quant helpers), `bench_w8a8.py`,
`results_w8a8.json`, `plot_w8a8.py` → `w8a8_results.png`.
