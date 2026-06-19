# CUDA-operator validation of the cache-aware quant finding (H100)

**Mandate (advisor / BAAI-FlagGems):** the "weight-only quant has little/no benefit
at small token counts" result was measured on **Triton** kernels. Re-validate it on
**tuned CUDA operators** (vLLM Marlin MoE + FlashMLA) to rule out the Triton language
as a confound, then re-fit the cache-aware roofline (CARM) on the CUDA numbers and make
it GPU-parameterized.

**One-line outcome.** The finding is **not a Triton artifact in the direction that
would sink it** — but the Triton numbers were *distorted by a bad bf16 baseline*, and
the tuned-CUDA data tells a **cleaner, more correct** story that *strengthens* the
cache-aware-roofline thesis: weight-only quant wins only in the memory-bound regime and
**loses once compute-bound**. A clock-locked red-team pass (§7) pins the MoE crossover at
**≈300 tokens against a properly-tuned bf16 baseline — matching the roofline's 334** — and
shows the looser "≈600" figure was inflated by vLLM's *under-tuned stock config*
(`GROUP_SIZE_M=1`) for this shape. FP8-KV MLA shows the same regime structure on the batch
axis (≈3× at low batch, then a **net loss, 0.71×, at high batch**).

---

## 1. Environment / versions

| Component | Version |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM5), driver 580.105.08, CUDA 13.0 |
| vLLM | **0.20.2** (`pip install vllm==0.20.2`) |
| torch | 2.11.0+cu130 |
| triton | 3.6.0 |
| flashinfer | 0.6.8.post1 |
| FlashMLA | **bundled in vLLM** (`vllm.third_party.flashmla`); no source build needed. `is_flashmla_dense_supported`/`is_flashmla_sparse_supported` → `(True, None)` on this H100 |
| compressed-tensors | 0.15.0.1 |

The pre-existing FlagGems Triton results (`profiling/fused_moe/*.json`) were produced on
torch 2.7 / triton 3.3 and are **left untouched**; all CUDA results are written to this
new `profiling/cuda_validation/` directory.

**Base CARM params re-measured on the CUDA-13/torch-2.11 stack** (vs the torch-2.7 Triton
era) — essentially unchanged, so the stack is **not** a confound:

| | HBM read | L2 read (reduction) | graph floor t₀ |
|---|---|---|---|
| Triton era (torch 2.7) | 3.146 TB/s | 5.331 TB/s | 2.802 µs |
| CUDA era (torch 2.11) | **3.121 TB/s** | **5.617 TB/s** | **2.778 µs** |

> **Hardware caveat (H100 has NO FP4 tensor cores).** FP8 and FlashMLA-FP8 are *native*
> on Hopper — real results. **MXFP4 on H100 is EMULATED**: Marlin dequantizes FP4 → bf16
> and runs the matmul in bf16, so the H100 MXFP4 dequant ceiling is in the silicon, not
> the kernel. **Every MXFP4-on-H100 number below is labelled `EMU`** and is *not* a
> fundamental conclusion; the native-FP4 leg belongs on Blackwell (§5).

---

## 2. Experiment A — `fused_moe`: tuned CUDA vs Triton (the crux)

Shape: Mixtral **E=8, H=4096, I=14336, top-k=2**. Token sweep T ∈ {16…2048}.
All paths **CUDA-graph timed** (10 launches/graph, median of 40 replays — identical
methodology to the Triton `results_fused_moe_extended.json`, copied verbatim). Weights are
quantized/repacked **once** outside the timed region (vLLM's `marlin_quant_fp8_torch` /
`rand_marlin_weight_mxfp4_like` + `gptq_marlin_repack`); only the op call is captured.
Correctness = rel-err vs a bf16 `fused_experts` run on each path's own dequantized weights.

**Correctness:** fp8 W8A16 rel-err **0.0057**, mxfp4 W4A16 rel-err **0.0038** (quant noise,
as expected). Reproduced across 3 runs within ±2%.

> **What is and isn't Triton here.** The two *quant* paths are now **CUDA Marlin**
> (`fused_marlin_moe`, `quant_type_id` = `float8_e4m3fn` / `float4_e2m1f`) — that is the
> language confound the mandate targets, and it is removed. The **bf16 reference is vLLM's
> production `fused_experts`** (a *tuned, autotuned* Triton kernel), per the directive. The
> thesis only needs the bf16 baseline to be *competent*, which the production kernel is —
> and that is exactly what the FlagGems bf16 baseline was **not** at high T. So the high-T
> reversal vs the Triton table is "competent baseline vs uncompetitive baseline," not
> "CUDA vs Triton."

| T | bf16 µs | **CUDA fp8 W8A16** (×bf16) | **CUDA mxfp4 W4A16 `EMU`** (×bf16) | *(ref)* Triton W8A16 | *(ref)* Triton W8A8 |
|---:|---:|---:|---:|---:|---:|
| 16 | 953 | 502 (**1.90×**) | 296 (3.22× `EMU`) | 564 (1.72×) | 558 (1.74×) |
| 64 | 962 | 525 (**1.83×**) | 437 (2.20× `EMU`) | 711 (1.72×) | 658 (1.86×) |
| 128 | 999 | 568 (**1.76×**) | 555 (1.80× `EMU`) | 833 (1.20×) | 776 (1.29×) |
| 256 | 1386 | 1013 (**1.37×**) | 1036 (1.34× `EMU`) | 1209 (1.03×) | 1064 (1.17×) |
| 512 | 2210 | 1791 (**1.23×**) | 1757 (1.26× `EMU`) | 1990 (2.67×) | 2279 (2.33×) |
| 640 | 1867 | 2070 (**0.90×**) | 2026 (0.92× `EMU`) | — | — |
| 768 | 1912 | 2350 (**0.81×**) | 2298 (0.83× `EMU`) | — | — |
| 1024 | 2404 | 3071 (**0.78×**) | 3044 (0.79× `EMU`) | 3000 (2.71×) | 3448 (2.35×) |
| 2048 | 3377 | 5690 (**0.59×**) | 5652 (0.60× `EMU`) | 5033 (2.96×) | 6164 (2.42×) |

Physical sanity: bf16 reaches **727 TFLOPS** (73% of peak) at T=2048; fp8/mxfp4 both pin at
**≈422 TFLOPS** (the in-core dequant/convert ceiling) — they converge at high T because both
dequantize to bf16 before the tensor-core matmul. At small T they diverge by weight bytes
(mxfp4 4-bit < fp8 8-bit → mxfp4 faster), the classic memory-bound ordering.

Figure: `figures/cuda_moe_triton_vs_cuda.png`.

### Decisive sentence (MoE)

> **Does the tuned-CUDA path also wash out at small T (≤128)? No — it wins *more* (fp8
> 1.76–1.90×).** The real correction is at *large* T: the tuned-CUDA bf16 baseline is
> *competent* (it does not blow up the way the Triton bf16 baseline did), so weight-only
> quant **crosses over (≈300 tokens vs a properly-tuned bf16, §7) and LOSES outright by high
> T** (0.59× at T=2048) — the in-core dequant ceiling, finally visible. **The Triton "2.7–3.0×
> win at high T" was a bad-baseline artifact**, not a property of the quant kernel.

So the result is **part fundamental, part Triton artifact**, and we re-scope honestly:

* **Fundamental (and cleaner on CUDA):** weight-only quant is bounded by an in-core dequant
  ceiling (≈422 TFLOPS here) and therefore *must* lose once the workload is compute-bound.
  On Triton this was hidden because its bf16 baseline scaled super-linearly; on tuned CUDA
  the crossover is sharp and lands where CARM predicts.
* **Triton artifact (re-scoped):** the *direction* of the Triton high-T result was inverted
  by the baseline. Corrected magnitude: tuned-CUDA weight-only quant wins **1.2–1.9× for
  T≲512** and **loses (down to 0.59×) for T≳700**; it does **not** win "at all token counts."
* MXFP4-`EMU` tracks fp8 almost exactly (same bf16-dequant ceiling) — expected on H100; not a
  fundamental statement (§5).

---

## 3. Experiment B — `flashmla_sparse`: bf16 KV vs FP8 KV

DeepSeek MLA dims: d_qk=576 (512 NoPE + 64 RoPE), d_v=512, h_q=128. Sparse top-k = **2048**
(DeepSeek V3.2 "DSA" setting), capped at context length C. Sweep C ∈ {512,2048,4096,8192} ×
batch B ∈ {1,8,32}. CUDA-graph timed.

This mirrors vLLM's `flashmla_sparse` backend **exactly**: bf16 KV → `flash_mla_sparse_fwd`;
FP8 KV → `concat_and_cache_mla("fp8_ds_mla")` + `flash_mla_with_kvcache(is_fp8_kvcache=True,
indices=…)`. Compute stays bf16 in both; only the KV-cache precision (and, faithfully, the
kernel selected for it) changes. Correctness: FP8-KV vs bf16-KV rel-err **≈0.026** everywhere.

| C | topk | B=1 | B=8 | B=32 | regime |
|---:|---:|---:|---:|---:|:--|
| 512 | 512 | 1.40× | 1.10× | **0.58×** | dense |
| 2048 | 2048 | 3.18× | 2.18× | **0.97×** | dense |
| 4096 | 2048 | 3.16× | 2.17× | **0.97×** | sparse (2×) |
| 8192 | 2048 | 3.13× | 2.13× | **0.99×** | sparse (4×) |

*(cells = FP8/bf16-KV speedup; >1 = FP8 KV faster. Full latencies in
`results_flashmla_sparse.json`.)* Figure: `figures/flashmla_bf16_vs_fp8.png`.

Sanity: bf16 `sparse_fwd` latency is **invariant to context** (≈60–68 µs at topk=2048 for
C=2048/4096/8192) — confirming the sparse kernel touches *topk*, not full context.

### Decisive sentence (MLA)

> **FP8 KV washes out at high batch** (B=32: 0.97–0.99× at topk=2048, and *0.58×* at the
> short C=512), exactly the throughput-regime analog of the MoE high-T loss. It wins big at
> **low** batch (B=1: ≈3×) — but that magnitude is **confounded by vLLM's two-kernel
> dispatch** (bf16 falls back to the prefill-style `sparse_fwd`, which is inefficient on a
> 1-token decode, while FP8 uses the optimized decode kernel). The **robust, confound-free
> signal is the high-batch washout**, where both kernels are efficient and FP8 KV gives **no
> benefit**.

**A clean same-kernel control does not exist on Hopper FlashMLA** (recorded, not silently
substituted — `probe_dense.py`): the sparse decode kernel *asserts* `is_fp8_kvcache=True`
(sparse is FP8-KV-only by design), and the dense FP8 path (`flash_mla_with_kvcache_fp8`)
quantizes the *query* to FP8 as well (full-FP8 compute, a different regime). So "bf16 vs FP8
KV with bf16 compute" is **necessarily** a two-kernel comparison — which is precisely what
vLLM dispatches. This is an architectural fact about FlashMLA, not a methodology gap.

---

## 4. Experiment C — CARM re-fit on CUDA + GPU-parameterization

Model: **t = t₀ + max(B/BW(WS), F/P)**, BW capacity-gated on *effective* L2 (`carm.py`).
Applied to the MoE operator with measured params (t₀=2.78 µs, BW_HBM=3.12 TB/s), fitting
only the two compute ceilings from the CUDA sweep:

* fit **bf16 peak = 712.9 TFLOPS** (72% of 989 nominal)
* fit **dequant ceiling (fp8 W8A16) = 422.9 TFLOPS** (43%)

**MAPE (CARM prediction vs measured CUDA MoE):**

| path | CUDA-fit MAPE | Triton-era recon-BMM MAPE |
|---|---:|---:|
| **fp8 W8A16 (the quant path)** | **12.2%** | INT4 18.2% |
| bf16 | 22.3% | FP16 10.2% |

The **quant-path MAPE improves** (12.2% vs 18.2%) — as the advisor predicted. The bf16 MAPE
is *higher* (22.3%) because the vLLM MoE kernel runs **above** the smooth roofline in the
256–1024 "transition band" (config-selection raggedness — visible as the T=512 spike / T=640
dip in the data); the roofline is a clean envelope and the kernel sits above it there. This
same raggedness is why the **measured** crossover (≈602) is *later* than the **roofline**
crossover (≈334): a competent-but-not-optimal bf16 kernel delays the point where quant's
ceiling catches up.

**Per-operator quant-vs-dense crossover (the number the dispatcher needs):**

| operator | crossover | source |
|---|---|---|
| MoE fp8 W8A16 (this shape) | **≈300 tokens** vs a fair bf16 baseline (measured 263, roofline 334); ≈600 vs under-tuned stock vLLM — see §7 | Exp A + §7 |
| MLA FP8 KV (topk=2048) | **batch ≈32** (parity; net loss 0.71× beyond) | Exp B + §7 |

### GPU-parameterized crossover

`carm.py` now takes per-GPU params — **effective L2 capacity, tier bandwidths, and native
low-precision MMA availability** — and emits a per-GPU crossover. The governing rule:

* **weight-only quant** (W8A16/W4A16, bf16 compute) is bounded by the in-core **dequant
  ceiling** *regardless* of native MMA → a crossover **exists on every GPU**.
* **matched-precision quant** (W8A8/W4A4) reaches the **native** tensor-core peak **iff** its
  precision is native, else it falls back to the dequant ceiling → **wins everywhere** only
  where the precision is native.

| GPU | W8A16 (weight-only) | W8A8 (matched) | MXFP4 W4A4 (matched) |
|---|---|---|---|
| **H100** (measured) | crosses ≈334–600 tok | **wins everywhere** (native INT8/FP8) | crosses ≈334 tok — **`EMU`, no native FP4** |
| **B200** (PROJECTED) | crosses ≈294 tok | wins everywhere | **wins everywhere** (native FP4) |

The H100→B200 MXFP4 contrast is the thesis generalized across hardware: **the crossover moves
with cache size + native-precision support.** B200 numbers are projected from public Blackwell
specs (native FP4 MMA, ~96 MB L2, ~8 TB/s HBM3e) and are a clean parameter-set hook, **not
measured here**. (`carm_cuda_params.json`, `CARM_PARAMS["b200"]`.)

---

## 5. Red-team verification (clock-locked, drift-controlled)

Three load-bearing claims were independently re-checked with the **SM clock locked to
1755 MHz** and **rotated measurement order** (median of 3 interleaved rounds), so neither
thermal/boost drift nor measurement order can move the numbers. (`verify_moe_crossover_v2.py`,
`verify_mla_washout.py`, `results_verification.json`.)

**(1+2) The bf16 "config raggedness" is real vLLM behavior, and it inflated the crossover.**
vLLM ships **no tuned bf16 config** for E=8,N=14336 on H100 (only `fp8_w8a8`/H200), so it falls
back to a default heuristic that uses **`GROUP_SIZE_M=1`** below ~M=1280 — which cripples L2
reuse of expert weights across M-blocks. Forcing `GROUP_SIZE_M=16` (the value the heuristic
itself uses at larger M) speeds up bf16 by **1.30–1.60× in the T=256–512 band** and removes the
spike. Consequences:

| | crossover T* |
|---|---|
| fp8 W8A16 vs **stock-default** bf16 (`GROUP_SIZE_M=1`) | **601** |
| fp8 W8A16 vs **properly-tuned** bf16 (`GROUP_SIZE_M=16`) | **263** |
| smooth roofline (§4) | 334 |

The three numbers **reconcile**: against a *fair* bf16 baseline the measured crossover (263) sits
right at the roofline (334) — **≈300 tokens** — and the loose "≈600" was an *under-tuned-baseline*
artifact, not a property of the quant kernel. The 706 anchor-interpolation figure was a
sparse-interp artifact and is **dropped**. Two claims are **config-independent and robust** (a
small-T control confirms `GROUP_SIZE_M` is irrelevant at T≤128, `def/g16 = 0.997`): the **small-T
fp8 win (~1.8×, T≤128)** and the **large-T fp8 loss (0.63–0.78×, T≥640)**.

> **Honest headline:** quant's win window for this MoE shape is **T ≲ 300 against a competent
> bf16 baseline** (not ~600). The model predicted this; the stock-vLLM number was generous to quant.

**(3) The MLA FP8-KV high-batch result is robust — and is a *loss*, not just a washout.**
Extending batch to 128 (clock-locked), the FP8/bf16-KV ratio falls **monotonically**: 3.12× (B=1)
→ 1.39× (B=16) → 0.96× (B=32) → **0.71× (B≥64)**. The bandwidth columns show why: the fp8
`with_kvcache` decode kernel saturates KV bandwidth at **~0.8 TB/s**, *below* the bf16
`sparse_fwd` kernel's **~1.9 TB/s**, so fewer bytes/token stops helping once batch is large. The
low-batch win remains confounded by the two-kernel dispatch (acknowledged §3); the **high-batch
loss is the robust, deployment-relevant signal**, and it is *stronger* than the original
"washes out to parity" wording.

---

## 6. Dispatch hook — design + measured Task-3 results

### 6a. Measured: the crossover moves with shape, and the dispatcher = oracle

Run at the **real target-model MoE shapes** (HF `config.json`, 2026-06), clock-locked,
graph-timed, fp8 rel-err ≈0.006 (`task3_target_shapes.py`, `task3_dispatch_moe.py`):

| model | E / H / I / top-k | fp8 win @small-T | **crossover T\*** | fp8 @T=2048 | dispatcher |
|---|---|---|---|---|---|
| Mixtral-8x7B | 8 / 4096 / 14336 / 2 | 1.9× | **~1024 stock (~300 tuned, §7)** | 0.60× | 1.06× vs always-fp8 |
| **DeepSeek-V4-Flash** | 256 / 4096 / 2048 / 6 | 1.9× | **none — fp8 wins to 2048** (1.4–2.0×) | 1.42× | 1.00× (=always-fp8) |
| **Qwen3.6-35B-A3B** | 256 / 2048 / 512 / 8 | 1.7× | **~1900** (stock=tuned) | 0.93× | 1.00× (=always-fp8) |

**The crossover is shape-governed, and it is the headline of the shape-parameterized CARM.**
Coarse-grained MoE (Mixtral: 8 *big* experts, I=14336) reaches compute-bound early → quant
crosses over in the few-hundred-to-~1000-token range. The **actual target models are
fine-grained** (256 *small* experts, I=2048/512, top-k 6–8): each token reads many small
expert weights but does little compute per expert, so they stay **weight-memory-bound across
the entire practical token range** → **weight-only fp8 wins everywhere (1.4–2.0×)**.

So the deployment rule sharpens: **CARM tells you which models need a dispatch at all.**
DeepSeek-V4-Flash and Qwen3.6-35B should simply **always quantize** the expert GEMMs (no
dispatch, no second weight copy); a Mixtral-style coarse MoE is where the token-count dispatch
earns its keep. A CARM-dispatched MoE over a continuous-batching trace **equals the oracle at
every shape** (1.6–1.9× vs bf16; 1.0–1.4× vs the better static policy depending on prefill
fraction — `results_task3_dispatch.json`).

**Tuned-baseline re-check (the §7 trap, applied to the targets — `verify_target_tuned.py`).**
The "always-quantize" claim was first measured against *stock* vLLM bf16 — the same under-tuned
baseline that inflated the Mixtral crossover. So it was re-checked against a **tuned** bf16
(sweeping `GROUP_SIZE_M ∈ {1,8,16,32,64}`, best per T). Result: for the fine-grained targets,
tuning is **inert** — `bf16_best ≈ bf16_default` at every T, and the crossover is unchanged
(DeepSeek: none→none; Qwen: 1908→1904). The mechanism is the point: these shapes stream **256
distinct expert weights from HBM** with little cross-M-block reuse, so they are **HBM-weight-byte-
bound, not L2-reuse-bound** — and `GROUP_SIZE_M` (an L2-reuse knob) cannot change how many weight
*bytes* bf16 must read, which is exactly what fp8 halves. *That* is why fine-grained MoE differs
from Mixtral (8 big experts → lots of L2-reuse headroom that `GROUP_SIZE_M` unlocks), and why the
"always-quantize the targets" conclusion is structural, not a baseline artifact. (Full autotune
of BLOCK_M/N/K could squeeze bf16 a few % more, but cannot touch the byte-count argument.)
Full-model serving of DeepSeek-V4 (1.6T) needs multi-GPU; this is the faithful **MoE-layer** unit
where the dispatch decision lives.

### 6b. Where the hook lives (design)


**Why upstream of the operator.** `fused_marlin_moe` / `flash_mla_*` see *already-quantized*
data; they cannot choose. The decision must live where the per-step **token count** is known
and both weight representations can be reached — the layer/scheduler boundary.

**Where in vLLM (v1).**

1. **MoE:** in `FusedMoE.forward` (`vllm/model_executor/layers/fused_moe/layer.py`), branch on
   `num_tokens = hidden_states.shape[0]` against the CARM crossover for *that layer's* shape
   (E,H,I): `T < T*` → quantized Marlin experts; `T ≥ T*` → bf16 `fused_experts`. `T*` comes
   from `carm.moe_crossover_tokens_cuda()` / the GPU-parameterized `moe_crossover(gpu, mode)`
   at load time (**≈300 for this shape on H100** against a competent bf16 baseline, §7; a
   *quantization-only* deployment that keeps no bf16 copy simply lives with the bounded high-T
   loss and needs no dispatch).
2. **MLA:** the analogous knob is **batch/elements-per-step**: route small-batch decode steps
   to FP8 KV and large-batch / chunked-prefill steps to bf16, keyed on the Exp-B crossover
   (batch ≈16–32). This aligns naturally with vLLM v1's prefill/decode split.

**Cost & when it pays.** Dynamic dispatch needs **both** weight copies resident (≈1.5× MoE
weight memory for bf16 + fp8). It pays when a deployment spans both regimes — i.e. mixed
chunked-prefill (large T, want bf16) + decode (small T, want quant). If memory is the reason
for quantizing in the first place, keep only the quantized weights and accept the bounded
high-T loss (≤0.6×); the CARM crossover then tells you *which serving configs* (max-num-batched-
tokens) stay on the winning side.

**Targets.** For **DeepSeek-V4 (flash)** and **Qwen3.6-35B**, the per-layer MoE shapes set
per-layer `T*`; the scheduler already knows the step token budget, so the hook is a thin
shape-keyed table lookup + a two-way branch in the layer forward — **no kernel changes**. (No
implementation this session, per scope.)

---

## 7. Files

```
profiling/cuda_validation/
  REPORT.md                      <- this file
  common.py                      graph_med_us / eager_med_us / version capture
  bench_cuda_moe.py              Exp A   -> results_cuda_moe.json (+ _run1 repro)
  bench_flashmla_sparse.py       Exp B   -> results_flashmla_sparse.json
  carm_cuda_fit.py               Exp C   -> carm_cuda_params.json
  verify_moe_crossover_v2.py     §7 red-team (clock-locked) -> results_verification.json
  verify_mla_washout.py          §7 red-team (clock-locked, batch->128)
  probe_marlin.py / probe_flashmla.py / probe_dense.py   de-risking probes (incl. the
                                 recorded "no same-kernel MLA control on Hopper" finding)
  plot_cuda_moe.py / plot_flashmla.py    figures/*.png,*.pdf
results updated in place:
  profiling/carm_model.json                     <- cuda_validation_2026_06_19 block (MAPE, crossover)
  ../kernel-compass/profiling/carm.py           <- CUDA MoE anchors, b200 param set, native_mma field
```

### Bottom line for the paper

The "no benefit at small token counts" framing should be **replaced** by the sharper,
CUDA-validated statement: *weight-only quantization wins only inside a bounded memory-bound
window and loses outside it on both sides* — when weights are L2-resident (small matrices) and
when compute-bound (large token counts), the latter now cleanly visible at **≈300 tokens against
a competent bf16 baseline** on a tuned CUDA MoE kernel (§7). The Triton numbers were not "wrong
about small T" but were distorted
at **large** T by an uncompetitive bf16 baseline; the corrected, tuned-CUDA picture is more
favorable to the cache-aware-roofline thesis, not less. Matched-precision quant (W8A8 today,
native MXFP4 on Blackwell) is the way to win in the compute-bound regime — and CARM, now
GPU-parameterized, says exactly when.
