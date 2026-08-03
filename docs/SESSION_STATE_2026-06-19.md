# Session handoff — CUDA validation of the cache-aware quant finding (2026-06-19)

H100 80GB instance. All deliverables are committed + pushed to
github.com/zhan4808/{cache-barrier, kernel-compass} and live under
`repos/cache-barrier/profiling/cuda_validation/`. This doc is the cross-instance
handoff; the env to reproduce is in `../env/vllm-0.20.2/` (run `restore.sh`).

## What was done
Re-validated the "weight-only quant has little benefit at small token counts"
finding on **tuned CUDA** operators (vLLM 0.20.2 Marlin MoE + bundled FlashMLA),
to clear the advisor's worry that it was a Triton-language artifact. Then re-fit
the cache-aware roofline (CARM) on the CUDA numbers and GPU-parameterized it; ran
a red-team pass; and took the dispatch idea to a measured layer-level result on
the real target models.

## Headline results (all in REPORT.md)
- **MoE (Exp A):** tuned-CUDA fp8/MXFP4 quant wins ~1.8x at small T, **loses above
  ~300 tokens** (vs a properly-tuned bf16). The Triton "high-T win" was a bad-bf16-
  baseline artifact. fp8 rel-err 0.006; reproduced ±2%.
- **MLA (Exp B):** FP8 KV wins ~3x at low batch, **net loss 0.71x at high batch**
  (fp8 decode kernel saturates KV BW at 0.8 TB/s vs bf16 sparse_fwd 1.9). Necessarily
  a two-kernel comparison (FlashMLA sparse is fp8-KV-only by design — documented).
- **CARM (Exp C):** re-fit on CUDA, quant-path MAPE **12.2%** (was 18.2% Triton).
  GPU-parameterized (effective L2 + native-MMA set → per-GPU crossover). h100/a100/b200
  param sets in kernel-compass `profiling/carm.py`.
- **Red-team (REPORT §7):** clock-locked, drift-controlled. The MoE crossover
  reconciles to **~300 tok** vs a fair baseline (= roofline 334); the ~600 figure was
  vLLM's under-tuned default config (GROUP_SIZE_M=1, no tuned bf16 config for
  E=8,N=14336 on H100). Small-T win and large-T loss are config-independent.
- **Task 3 (REPORT §6):** crossover MOVES with MoE shape. The real targets are
  fine-grained: **DeepSeek-V4-Flash** (E256/H4096/I2048/k6) fp8 wins the whole range;
  **Qwen3.6-35B-A3B** (E256/H2048/I512/k8) wins to ~1900 tok. Re-checked vs a TUNED
  bf16 (GROUP_SIZE_M sweep) — inert for fine-grained MoE because it's HBM-weight-byte-
  bound, not L2-reuse-bound. => these targets should **always-quantize** the experts;
  token-count dispatch only matters for coarse MoE like Mixtral. Dispatcher == oracle.

Figures: `figures/{cuda_moe_triton_vs_cuda, flashmla_bf16_vs_fp8, task3_crossover_by_shape}.png`.

## Open items / next steps
1. **WeChat update to Dr. Xiao** — drafted (2-chunk version in the chat transcript /
   REPORT framing). Robert sends it on WeChat (advisor comms are WeChat, NOT email —
   the mailbox has no address by construction). Ask in it: **B200/GB200 access**.
2. **Blackwell MXFP4 leg (the one open hole)** — BLOCKED on H100: Hopper has no FP4
   tensor cores, so all MXFP4 numbers here are EMULATED (dequant-to-bf16, labelled).
   The native-FP4 leg (MXFP4 winning in the compute-bound regime; H100→Blackwell as
   the model's cross-hardware prediction) needs B200. CARM b200 param set is ready as
   the hook (`carm.py` CARM_PARAMS["b200"], PROJECTED).
3. **vLLM dispatch hook** — PARKED by decision. Task 3 showed the named targets don't
   need it (always-quantize wins). Only worth building for coarse-MoE deployments.
4. **Paper** — fold REPORT.md + figures into the writeup once Dr. Xiao replies.

## Gotchas captured this session
- vLLM 0.20.2 bundles FlashMLA (`vllm.third_party.flashmla`); no source build.
- Marlin MoE repack: `marlin_quant_fp8_torch` / `rand_marlin_weight_mxfp4_like` +
  `fused_marlin_moe(quant_type_id=scalar_types.float8_e4m3fn.id / float4_e2m1f.id)`.
- FP8 sparse MLA KV built via `concat_and_cache_mla(..., "fp8_ds_mla", scale)`.
- Lock SM clock (`nvidia-smi -lgc 1755,1755`) for drift-free timing ratios; `-rgc` to reset.
- bf16 fused_experts is itself tuned-Triton; the "no Triton confound" claim is about
  the QUANT kernels (now CUDA Marlin), with a competent (not nec. CUDA) bf16 baseline.
