# Multi-layer MLA L2 stacking (P5)

Graph-timed sequential reconstruction BMMs, bs=1, 16 MB/layer (H=128, K=128, N=512).

| Layers | Weight MB | FP16 µs | W8A8 µs | W8A8 vs FP16 | CARM pred |
|--------|-----------|---------|---------|---------------|-----------|
| 1 | 16 | 4.9 | 6.9 | 0.72× | 6.0 |
| 2 | 32 | 9.6 | 12.1 | 0.79× | 9.2 |
| 3 | 48 | 24.6 | 17.1 | **1.44×** | 19.0 |
| 4 | 64 | 31.4 | 23.6 | 1.33× | 24.3 |
| 5 | 80 | 40.8 | 34.5 | 1.18× | 29.7 |
| 6 | 96 | 48.8 | 41.7 | 1.17× | 35.1 |

**Finding:** W8A8 crosses over at 3 stacked layers (48 MB total WS), between the measured
effective L2 capacity (~36 MB) and the nominal 50 MB. Single-layer MLA (16 MB) stays
FP16-favorable; depth pushes the stack into HBM-served territory where byte savings convert.

Run: `python3 bench_mla_l2_stack.py` → `plot_mla_l2_stack.py`
