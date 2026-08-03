# FlagGems leg — the gate under a second kernel ecosystem (H100, session 12)

Same silicon, swapped ecosystem (cuBLAS nvjet vs FlagGems 5.0.2 Triton
mm, fork @ fix-w8a16-inkernel-dequant), T=1 bf16 GEMM, W 8-96 MB,
graph-timed (`bench_flaggems_gate.py`):

- **cuBLAS reproduces the hardware structure**: BW peaks 4.07 TB/s at
  32 MB, breaks over 34-40, far field 2.6-2.8 — the GEMM-context cliff
  at ~34 MB (matches the NCU two-capacities number).
- **FlagGems' mm never sees the cache at all**: ~23.5 us fixed cost +
  streaming (24.0 us at 8 MB where cuBLAS takes 6.5); effective BW rises
  monotonically 0.39 -> 2.48 TB/s and saturates at the HBM tier with NO
  residency break anywhere.

Reading: the strongest same-silicon confirmation of the model's division
of labor — the capacity structure is a hardware property (visible to any
kernel that can exploit it), while whether a kernel sees the L2 tier AT
ALL is an implementation property. FlagGems mm at decode shapes joins
fla's GDN decode kernel in the "L2-blind" class; the residency-aware-
kernel opportunity is ecosystem-wide, not a one-off.

Caveats: FlagGems mm targets large-M training shapes; M=1 skinny GEMM is
its worst case (config/fixed-cost dominated); one op, one library
version. Data: results_flaggems_gate_h100.json.
