# Session handoff — H100 baseline lock + B200 harness prep (2026-07-01)

Fresh H100 80GB instance (driver 580.105.08 — **same as the 2026-06-19 capture**,
so the byte-exact vLLM 0.20.2 venv restores and imports directly). Goal this
session: get the H100 profiling locked, then make the harness ready to "quickly
move to B100/200". Continues [SESSION_STATE_2026-06-19](SESSION_STATE_2026-06-19.md).

## Env restore (done)
`bash env/vllm-0.20.2/restore.sh` → `/home/ubuntu/vllm-env` (byte-exact copy OK,
`torch 2.11.0+cu130 vllm 0.20.2 cuda_avail True`). Run work with
`~/vllm-env/bin/python` + `VLLM_LOGGING_LEVEL=WARNING`. Passwordless sudo works
(clock lock available). **`ncu`/`nsys` are NOT installed** on this box (the CUDA
validation uses CUDA-graph timing + clock lock, so not needed).

## 1. Baseline reproduced & locked ✅
Re-ran the full 2026-06-19 validation on the fresh instance. **Every load-bearing
claim reproduces within noise.** Details + table in
`repos/cache-barrier/profiling/cuda_validation/REPRODUCTION_2026-07-01.md`.
- Exp A (MoE): FP8 path within ±0.7% at every T; correctness identical.
- Exp B (MLA): high-batch washout within ±1.3%; fp8 KV BW ceiling ~0.8 vs bf16 1.9 TB/s.
- Exp C (CARM): MAPE identical (bf16 22.3%, fp8 12.2%).
- Red-team (clock-locked 1755): crossover 600 stock / 259 tuned (ref 601/263);
  DeepSeek none, Qwen 1920/1915 (ref 1908/1904).
- Repro JSONs saved as `*_repro_2026-07-01.json` (committed refs untouched).

## 2. Measurement/param gaps filled ✅
- Added **b100** CARM param set + registered **b200/b100 in GPU_SPECS**
  (`kernel-compass/profiling/{carm.py,metrics.py}`) and in
  `carm_cuda_fit.py GPU_PARAMS`; fit loop now emits h100/b200/b100.
  Emitted crossovers: b200 w4a4=none(wins)/w8a16 T*=294; b100 w4a4=none(wins)/w8a16 T*=244.
- Documented the key correction: **Marlin `float4_e2m1f` is EMU on EVERY GPU**
  (dequant→bf16 is intrinsic to Marlin, not a Hopper limitation). The native-FP4
  win needs a *different* kernel (cutlass/trtllm nvfp4).

## 3. B200-portable harness ✅
- `common.py`: `gpu_key()` (cap→h100/a100/b200/b100) + `native_low_precisions()`
  (fp4 native only on SM100+). Tested on H100.
- `bench_cuda_moe.py`: GPU-portable — self-selects, writes GPU-keyed
  `results_cuda_moe_<key>.json`, auto-labels EMU vs native. (committed
  `results_cuda_moe.json` left as historical H100 ref.)
- `bench_moe_nvfp4_native.py`: NEW native-W4A4 leg (the thesis-completer).
  Skips cleanly on H100; on SM100 raises `NotImplementedError` at the weight-prep
  boundary (won't emit wrong numbers). Entry points wired: `run_cutlass_moe_fp4`,
  `scaled_fp4_quant`.
- `B200_RUNBOOK.md`: full plug-and-run plan + the **env blocker**.

## Open items / next steps
1. **B200 env blocker (important):** the H100 byte-exact venv has **NO native FP4
   ops** — `cutlass_moe_fp4` op absent, `deep_gemm`/`triton_kernels` missing
   (flashinfer present). B200 needs a fresh Blackwell vLLM build (CUDA13/SM100),
   not the copied venv. See `B200_RUNBOOK.md §1`.
2. **Finalize `bench_moe_nvfp4_native.py` on B200** (§3 of runbook) — weight prep +
   `run_cutlass_moe_fp4` call. Expected: native W4A4 wins across the whole token
   range (vs Marlin W4A16 crossing over) — the H100→Blackwell money plot.
3. **Optional, doable on H100 now (task #4):** matched **W8A8** (native FP8-MMA)
   MoE via `run_cutlass_moe_fp8` (`cutlass_moe_mm` op IS built here). Would show
   matched-precision beats the dequant ceiling where weight-only W8A16 loses —
   the fp8 analog of the B200 fp4 experiment, on real silicon. Not run yet
   (new experiment; confirm scope).
4. Replace PROJECTED b200/b100 CARM params with measured once on Blackwell (§4).

## Uncommitted
All changes are **uncommitted** (this box has no push key by design). cache-barrier:
3 modified + 6 new files; kernel-compass: 2 modified. Commit locally if desired.
