# Overnight profiling log (H100)

## 2026-06-10 09:30 UTC — tick 0 (manual start)

**P0 FlagGems (partial complete)**
- Ran `bench_fused_moe_extended.py`: graph-timed T=16..2048, bf16 vs W8A16 vs W8A8.
  Key result: W8A16 1.7× at T≤64, parity band T≈256, **2.7–3.0× at T≥512**.
  W8A8 wins at low T; W8A16 wins at T≥512.
- NCU warm sweep ran but captured setup kernels (no `fused_moe_kernel` rows).
  Fixed `run_ncu_warm_sweep.sh` with `--kernel-name-base regex:fused_moe_kernel`.
  Added `parse_ncu_warm.py`. **Next tick:** re-run NCU only for T=16/128/512.
- Updated `profiling/fused_moe/REPORT.md` Finding 4.

**Loop:** armed 45m sentinel (PID tracked in session).
