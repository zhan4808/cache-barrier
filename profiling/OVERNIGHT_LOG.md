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

## 2026-06-10 ~10:00 UTC — tick 1 (45m loop)

**P0 FlagGems NCU (complete)**
- Re-ran warm NCU with `-k regex:fused_moe`; fixed CSV parser (skip NCU header lines).
- Warm `fused_moe_kernel` counters (longest invocation):
  - T=16: bf16 668µs 84% DRAM / 9% SM; w8a16 383µs 74% DRAM / **82% SM**
  - T=128: bf16 676µs 84% DRAM; w8a16 607µs 48% DRAM / **52% SM**
  - T=512: bf16 5371µs 64% DRAM; w8a16 1580µs 24% DRAM / 30% SM
- Confirms T≤128 memory-bound (bf16 DRAM-heavy), W8A16 shifts to conversion/compute;
  at T=512 bf16 still DRAM-bound but much slower per kernel than W8A16.
- Saved `ncu_warm_summary.json`. P0 done → next tick P1 W8A8 MLA autotune.

## 2026-06-10 ~10:45 UTC — ticks 2–9 (45m loop)

**P1 blocked** — shell/workspace disconnect on ticks 2–9. Loop killed on user request.

## 2026-06-10 — P1 complete (manual resume)

**Autotune** (`autotune_w8a8.py`, bs=64–512): configs match existing `_pick_config`
(<5% delta vs `results_w8a8.json`); no kernel change.

**Warm NCU** (bs=1, `cache-control none`):
| weights | mode | kernel | µs | DRAM | SM | L2 hit |
|---|---|---|---|---|---|---|
| 16 MB | fp16 | nvjet | 10.3 | 49% | 8% | **61%** |
| 16 MB | w8a8 | _w8a8_bmm | 8.4 | 14% | 39% | **93%** |
| 48 MB | fp16 | nvjet | 24.4 | 69% | 6% | **1%** |
| 48 MB | w8a8 | _w8a8_bmm | 18.4 | 41% | 54% | 11% |

L2 cliff confirmed: 16 MB fp16 61% L2-served; 48 MB fp16 1% L2 (HBM). W8A8 BMM
faster per-kernel at both sizes but loses end-to-end at 16 MB due to act-quant overhead.
Next: **P2 CARM revalidate**.

## 2026-06-10 — P2 complete

- Reran `measure_carm_params.py` (hbm 3.146, l2 5.331, t0 2.802/18.048 µs).
- `validate_carm_mape.py`: **FP16 MAPE 15.5%**, **INT4 MAPE 10.9%** on recon_points.
- `kernel-compass/profiling/carm.py`: refreshed params, `validate_recon_mape`,
  `predict_fp16/int4_recon_us`, MoE helpers (`predict_moe_*`, `moe_crossover_tokens`).
  With W8A16 fix, measured crossover is **None** (wins at all T; min ratio 1.03 @ T=256).
- Next: **P3** kernel-compass w8a8 baselines + `--compare`.

## 2026-06-10 — P3 complete

- `kernel-compass/kernels/baselines.py`: `make_w8a8_bmm_fn`, `batched_w8a8_gemm`, `bench_w8a8_bmm`.
- `optimizer/loop.py` uses baselines for w8a8 arm.
- `python3 -m optimizer.loop --compare`: 128 MB grid accept@1, FFN grid accept@2, LLM accept@1.
  (16 MB L2-resident: 0 candidates — expected.)

## 2026-06-10 — P4 complete

- Added `figures/w8a8_results.pdf` + Figure in §5.7 (`fig:w8a8`).
- Rebuilt `main.pdf` and `dist/arxiv_submission.tar.gz` (includes `carm_roofline.pdf`, `w8a8_results.pdf`).
- Updated §5.8 MoE crossover text (2.7–3.0× at T≥512, no bf16 crossover to T=2048).

## 2026-06-10 — P5 complete

- `profiling/mla_l2_stack/`: graph-timed 1–6 layer stack; W8A8 wins from L=3 (48 MB).
- Figure `mla_l2_stack.pdf`; stacking sentence added to §5.7.
