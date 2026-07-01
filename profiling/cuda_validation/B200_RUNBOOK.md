# B200 / B100 runbook — the Blackwell native-FP4 leg

Goal: measure the **one open hole** of the cache-barrier thesis — that a
*matched-precision* native-FP4 (W4A4) MoE kernel breaks the in-core dequant
ceiling and keeps winning in the compute-bound regime, where the weight-only
(Marlin, dequant→bf16) path crosses over and loses. This is the H100→Blackwell
cross-hardware prediction of the GPU-parameterized CARM.

On H100 every FP4 number is emulated (Hopper has no FP4 tensor cores). Everything
below is the plug-and-run plan for the first Blackwell box.

---

## 0. TL;DR order of operations on the B200 box

```bash
# 1. env (see §1 — the H100 byte-exact venv will NOT have native FP4 ops)
# 2. sanity: which FP4 backends does this build expose?
python probe_blackwell_fp4.py              # (write per §2; confirms native path)
# 3. runs immediately, no code changes (Marlin EMU curves + correct labels):
python bench_cuda_moe.py                   # -> results_cuda_moe_b200.json
python bench_flashmla_sparse.py            # -> Exp B (fp8 KV) on Blackwell
python carm_cuda_fit.py                    # refit; emits b200/b100 crossovers
# 4. the thesis-completer (finalize §3 first):
python bench_moe_nvfp4_native.py           # -> results_moe_nvfp4_native_b200.json
# 5. re-measure the projected CARM params (§4) and replace the PROJECTED sets
```

---

## 1. Environment — the H100 venv is NOT sufficient

The restored `~/vllm-env` (vLLM 0.20.2, torch 2.11+cu130) was captured on H100.
Confirmed on the H100 box:

| dep | H100 venv | needed for native FP4 |
|---|---|---|
| `cutlass_moe_fp4` custom op | **absent** (`hasattr(ops,'cutlass_moe_fp4')`→False) | **yes** (SM100 cutlass) |
| `deep_gemm` | **missing** | yes (DEEPGEMM_MXFP4 backend) |
| `triton_kernels` | **missing** | optional (Triton FP4 backend) |
| `flashinfer` | installed (0.6.8.post1) | yes (TRTLLM/CUTLASS FP4 backends) |

The vLLM `_C` extension in the byte-exact venv was compiled for the H100 arch and
does **not** contain SM100 cutlass FP4 kernels. On Blackwell, do ONE of:

- **(preferred)** fresh install of a Blackwell-supporting vLLM built with CUDA 13
  + SM100 (`pip install vllm==<blackwell-ok>` on the B200 box, matching driver),
  then `pip install flashinfer deep_gemm` as needed. Snapshot it to NFS like the
  H100 env (`env/restore.sh` pattern).
- rebuild vLLM 0.20.2 from source with `TORCH_CUDA_ARCH_LIST=10.0` so the cutlass
  FP4 ops are compiled in (heavier; only if 0.20.2 parity matters).

Do **not** try to run native FP4 off the copied H100 venv — the ops are not in it.

## 2. Sanity probe (write `probe_blackwell_fp4.py`)

Confirm the platform reports native FP4 and the backend oracle selects a native
kernel (not Marlin):

```python
from vllm.platforms import current_platform as p
print(p.get_device_capability())                 # expect major>=10
print(p.is_device_capability_family(100))        # expect True
import vllm._custom_ops as ops
print("cutlass_moe_fp4:", hasattr(ops, "cutlass_moe_fp4"))   # expect True
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend
# weight-only W4A16: on H100 -> MARLIN; on B200 -> DEEPGEMM/TRTLLM/CUTLASS
from common import gpu_key, native_low_precisions
print(gpu_key(), native_low_precisions())        # expect ('int8','fp8','fp4')
```

`common.gpu_key()` already resolves SM100 → `b200` (or `b100` by name); the CARM
`b200`/`b100` param sets are wired in `kernel-compass/profiling/carm.py` and
`carm_cuda_fit.py`.

## 3. The native NVFP4 W4A4 leg — `bench_moe_nvfp4_native.py`

Skeleton is in place and skips cleanly on H100. On SM100 it currently raises
`NotImplementedError` at the weight-prep / kernel-call boundary so it can't emit
wrong numbers. Finalize these against the confirmed vLLM 0.20.2 API:

- **Kernel:** `fused_moe.experts.cutlass_moe.run_cutlass_moe_fp4(...)`
  (full signature captured in the script; needs `output, a, a1_gscale, w1_fp4,
  w1_blockscale, w1_alphas, a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
  topk_weights, topk_ids, activation, workspace13, workspace2, m, n, k, e,
  device`). NVFP4 block size = 16; `w1_fp4` is `[E, 2N, K//2]` uint8 (E2M1),
  blockscales `float8_e4m3`.
- **Activation quant:** `vllm._custom_ops.scaled_fp4_quant` (functional + `.out`).
- **Weight packing:** modelopt / compressed-tensors NVFP4 packer, or
  `quantization.utils.marlin_utils_fp4.rand_marlin_weight_nvfp4_like` as a
  quick random-weight stand-in for latency (NOT for correctness — Marlin layout
  differs from cutlass; use the cutlass/modelopt packer for the real path).
- **Higher-level alternative:** route through the oracle
  `oracle.nvfp4.select_nvfp4_moe_backend(...)` → `CutlassExpertsFp4` /
  `FlashInferExperts`, which handle weight prep + workspaces. Simpler but pulls
  in `FusedMoEConfig` plumbing.
- **Timing/correctness:** identical to `bench_cuda_moe.py`
  (`graph_med_us`, 10 launches/graph, median of 40; rel-err vs bf16
  `fused_experts` on dequantized weights). Emit `nvfp4_w4a4_us` and
  `nvfp4_vs_bf16` per T, comparable to `results_cuda_moe_b200.json`.

**Expected result (the prediction to confirm):** unlike the Marlin W4A16 curve
(crosses bf16 at T*≈294 projected on B200 and loses at high T), the native W4A4
curve should **stay >1× across the whole token range** — `carm_cuda_fit.py`
already prints `b200 w4a4_mxfp4: none(wins)`. Confirming this on-device closes the
thesis.

## 4. Replace PROJECTED CARM params with measured (§Exp C)

`carm.py` `CARM_PARAMS["b200"|"b100"]` and `carm_cuda_fit.py GPU_PARAMS` are
**PROJECTED** (peak, HBM/L2 BW, c_eff, dequant ceiling). On the B200 box:

- re-measure base params: `kernel-compass/profiling/measure_carm_params.py`
  (HBM/L2 effective BW, graph floor t0) — same script used on H100.
- fit `peak_bf16_tflops` and `dequant_ceiling_tflops` from the B200 bf16 and
  fp8-W8A16 sweeps (`carm_cuda_fit.py` does this automatically once
  `results_cuda_moe_b200.json` exists — extend its H100-only fit block to the
  live gpu_key).
- fit a **native-FP4 peak** from the new W4A4 sweep and set the `fp4`
  `native_peak_mult` from measurement (currently projected 4.0×).
- flip `"measured": True` for the b200 row and drop the PROJECTED tags.

## 5. What to hand back / commit

- `results_cuda_moe_b200.json`, `results_flashmla_sparse*_b200*`,
  `results_moe_nvfp4_native_b200.json`, refit `carm_cuda_params.json`
  (now with a measured b200 block).
- A short `REPORT` addendum: the native-W4A4-wins-everywhere figure vs the
  Marlin-W4A16-crosses-over figure on the SAME B200 — the cross-hardware money
  plot (H100 EMU loses at high T ↔ B200 native wins at high T).
- Snapshot the B200 env to NFS (mirror `env/vllm-0.20.2/`).
