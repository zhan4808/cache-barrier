# DIRECTION — repo pivot, 2026-07

**Status:** active direction as of 2026-07-31. Supersedes the framing in `README.md` and `paper/main.tex`.

> **Reconciliation note (added 2026-08-01):** this document was drafted against the
> 2026-06 repo state. The 2026-07-01→02 session (see `docs/HANDOFF_2026-07-02.md` on
> NFS) already executed most of P2/P3/P4 below (`profiling/dense_qwen/`,
> `profiling/kv_serving/`, `profiling/mechanisms/`, `cuda_validation/bench_moe_w8a8.py`),
> and two of its findings **qualify** this document's claims:
> (1) contention is a **step function** — the L2-resident regime *dies* under full-model
> serving once Σ hot working sets > C_eff, it does not merely "move left" (§P3 hypothesis
> was directionally right but understated); (2) KV reads are **not** L2-limited at any
> working-set size, so the KV-toggle experiment in P3 measures act-quant kernel ceilings,
> not capacity restoration. P1 (the unified gate figure), P0 (reframe), P5, and P6
> remained undone as of 2026-08-01. See `RESULTS_2026-07.md`.

---

## 1. Why this document exists

Two things changed after the paper draft was written:

**(a) Dr. Xiao closed the sparse/MLA direction on 2026-07-01.**

> "Verification on sparse operators such as `fused_marlin_moe` and `flashmla_sparse` shows that CARM has no effect, because these operators are not limited by cache size."

The repo is currently titled *"The Hidden Bottleneck in MLA Serving: Reconstruction GEMMs, INT4 Quantization, and the L2 Cache Barrier."* The MLA framing, the INT4-failure headline, and the sparse-operator generalization are all now dead ends by his own verification. New scope: **dense models, real serving components, compute quantization.**

**(b) The industry moved underneath the original premise.**

- DeepSeek-V4-Flash (Apr 2026) is **MXFP4 quantization-aware-trained** — routed experts (96% of the model) ship *natively* in 4-bit, dense layers in FP8/BF16. Post-hoc weight-only quantization of frontier models is becoming obsolete because the checkpoint already arrives quantized.
- DeepSeek chose MXFP4 over NVIDIA FP8 **deliberately for cross-vendor portability**. BAAI's FlagOS then adapted V4 across **8+ domestic accelerator families** (Ascend, Hygon, Muxi, Moore Threads, Cambricon, Kunlun Xin, Suiyuan, Tianshu).
- **Most of those chips have no FP4 tensor cores.** They must dequantize native MXFP4 weights before the matmul — which is exactly the in-core dequant ceiling this repo measured (`int4_incore_ceiling_tflops: 31.7`). It is no longer a benchmark artifact to control for. It is the production path for the most important open model of 2026 on most non-Blackwell hardware.

FlagGems — Dr. Xiao's own group — is the operator layer that has to make the dispatch decision, per shape, on every one of those architectures. That cannot be hand-tuned or exhaustively autotuned per vendor per release.

**That is the problem this repo should be solving.**

---

## 2. The thesis, restated

> Quantization's speedup is gated by a **cache-capacity condition** that standard roofline analysis cannot express, because standard roofline has no capacity term. Once the weight working set becomes last-level-cache resident, the memory-traffic savings that motivate quantization evaporate, and any in-core dequantization cost turns the quantized path into a net loss. We give a cache-aware model with **three microbenchmarkable hardware parameters** — effective LLC capacity, effective bandwidth at that capacity, and dequantization throughput — that predicts the precision crossover on unseen architectures and is cheap enough to evaluate at dispatch time.

### The gate, formally

For a GEMM with weight working set `W` bytes on hardware with effective capacity `C_eff`:

```
if W < C_eff:          weights are already LLC-served at bw_l2
                       → quantization saves traffic you were not paying
                       → speedup ≈ 1.0, and NEGATIVE if the kernel dequantizes in-core
if W >> C_eff:         weights stream from HBM at bw_hbm
                       → quantization saves real traffic
                       → speedup → min(bytes_ratio, ceiling_dequant / T_compute)
```

The second branch is what MARLIN measures and reports as ~4× at batch ≤32. The first branch is what this repo measured and found nothing in. **They are not in conflict — they are two regimes of one model.**

### Why this is the right framing

The original claim "weight-only quantization gives no benefit at small token counts" contradicts MARLIN's headline result (near-ideal 4× at batch ≤32) and will lose that argument in any review. The capacity-gated claim **explains both results with one mechanism** and makes MARLIN the far-field limit of our model rather than a contradiction.

This is a strictly stronger scientific position than the one we started with, and it costs us nothing we actually had.

---

## 3. Claim triage — what survives, what dies

| Claim | Status | Action |
|---|---|---|
| Quant loses at **large** token counts | **Not novel.** MARLIN documents collapse after batch ~64; APEX4 reports MARLIN 20.3% *slower* than FP16 at batch 64; advantage "disappears by batch 256" | Demote to a calibration table. Never lead with it |
| Quant gives no benefit at **small** token counts | **At risk as phrased.** Directly contradicts MARLIN | **Reframe as the capacity gate.** This is P1, the single highest-value work item |
| In-core dequant ceiling | **Partially scooped.** QServe (MLSys'25) built W4A8 around main-loop dequant overhead | Keep, but cite QServe in the intro. Our delta: we make it a *composable roofline ceiling used predictively*, not a kernel-design obstacle to engineer around |
| Effective L2 capacity ≠ nominal (36 MB vs 50 MB) | **Solid, measured, ours** | Promote. This is the parameter that makes the gate predictive |
| W8A8 win/loss boundary sits **exactly at** measured capacity | **Solid, constructive, ours, and currently buried** | **Promote to headline result.** See §4 |
| MLA reconstruction is the dominant bottleneck | **Dead** — Xiao verified CARM has no effect on `flashmla_sparse` | Cut from the paper's critical path; retain as an appendix case study at most |

---

## 4. The result we already have and are under-selling

From `README.md` / `profiling/w8a8/REPORT.md`:

> W8A8 (INT8 `tl.dot` → int32 accumulator on Hopper IMMA, scales applied once on the accumulator) is **1.4–1.5× over cuBLAS FP16 at bs=1 when weights exceed the ~36 MB effective L2 capacity, and 0.70× when they are L2-served. The win/loss boundary sits exactly at the measured residency capacity.**

Read that again. **That is the capacity gate, demonstrated constructively, with a sign flip across the predicted boundary, in a kernel that has no dequant confound.** It is currently written up as a supporting note for an INT4 negative result.

It should be Figure 1 and the abstract's first sentence.

---

## 5. Existing assets (do not rebuild these)

| Asset | Path | Carries forward as |
|---|---|---|
| CARM params: `C_eff=36 MB`, `bw_l2=6.3 TB/s`, `bw_hbm=3.146 TB/s`, `r_dequant=0.496 TB/s`, `ceiling=31.7 TFLOPS` | `profiling/carm_model.json` | **The three hardware parameters.** Core of the contribution |
| Model validation, MAPE 10.2% FP16 / 18.2% INT4 | `profiling/validate_carm_mape.py` | Accuracy evidence |
| CUDA re-validation on Marlin `fused_moe`, FP8 MAPE 12.2% | `profiling/cuda_validation/` | Answers "is it a Triton artifact" — **already done, keep it** |
| GPU-parameterized crossover table | `carm_model.json → gpu_parameterized_crossover` | **The portability story, already scaffolded.** Needs more hardware rows |
| Weight-size sweep 8→128 MB, FP16 + INT4 | `profiling/bench_l2_barrier.py`, `carm_params.json` | Backbone of P1; needs a token-count axis |
| L2 interference under synthetic pressure | `profiling/bench_l2_interference.py` | Backbone of P3; needs *real* KV traffic instead of a synthetic stream |
| W8A8 IMMA kernel that beats cuBLAS | `profiling/w8a8/` | **New Figure 1** |
| Multi-layer L2 stacking | `profiling/mla_l2_stack/` | Feeds P3 (multi-layer residency) |
| Methodology audit | `profiling/validation/` | Guardrails — see `KICKSTART.md` §Guardrails |

Roughly 70% of the infrastructure for the new direction already exists. The work is reframing plus four targeted gaps.

---

## 6. Work phases

Ordered by (value to thesis) ÷ (GPU hours). **P1 is the one that matters most; do it first.**

### P0 — Reframe (no GPU, ~1 day)
- Rewrite `README.md` and the paper title/abstract around the capacity gate, not MLA/INT4
- Formalize the gate predicate (§2) as a stated model with an explicit capacity term
- Add a Related Work paragraph that positions against **tritonBLAS**, **QServe**, **MARLIN**, and **APEX4** (see §7)
- Demote MLA reconstruction to a case study

**Done when:** the abstract's first sentence is about capacity-gated precision dispatch and mentions neither MLA nor INT4.

### P1 — The gate figure (the MARLIN reconciliation) ★ highest value
The experiment that converts "we contradict MARLIN" into "MARLIN is our far-field limit."
- 2D sweep: **weight working set** `W ∈ {8,12,16,24,32,40,48,56,64,96,128}` MB × **token count** `T ∈ {1,16,32,64,128,256,512}`
- Precisions: `bf16` (tuned baseline), `W4A16`, `W8A8`
- Plot speedup vs **`W / C_eff`** (normalized x-axis), one line per T
- **Success criterion:** curves collapse onto a single family with the sign flip at `W/C_eff ≈ 1`; W4A16 speedup ≈ 1.0 below the gate and rises toward MARLIN's reported 3–4× well above it

New file: `profiling/gate/bench_capacity_gate.py`, `profiling/gate/plot_gate.py`
Reuse: the sweep skeleton in `bench_l2_barrier.py`, the W8A8 kernel in `profiling/w8a8/w8a8_bmm.py`

**This single figure is the paper.** Everything else is support.

### P2 — Dense model in scope? (Xiao todo #1, ~1 day, mostly analysis)
- Export every GEMM shape from Qwen3.6-27B via FlagOSTune (`--scenario shape --gems-mode mm --gems-once true`)
- Classify each shape against `C_eff`; weight by its share of the 25.8 s / 86.2% of runtime that `aten::mm` consumes in Dr. Xiao's profile
- **Deliverable:** "X% of dense GEMM time is in the L2-resident regime, therefore CARM does / does not apply to dense models"

Either answer is publishable and both answer his question. New file: `profiling/dense/shape_census.py`.

> **Amdahl framing to lead with:** in his `perf_analysis_qwen3.6-27b` profile, `aten::mm` is **86.21%** of GPU time (25,806.9 ms of 29,935 ms, 62,756 launches). 2× on `mm` = **1.76×** end-to-end; *everything else at infinite speed* = **1.16×**. Only GEMM is worth touching.

### P3 — Serving-realistic effective capacity (Xiao todo #2) ★ differentiator
`bench_l2_interference.py` currently applies **synthetic** pressure from a concurrent read stream. Upgrade to real conditions:
- Measure `C_eff` under (a) isolated microbenchmark — current 36 MB — and (b) **live vLLM serving, `p32768d1024`, concurrency 64, real KV cache traffic**
- Instrument per-kernel L2 with Nsight Compute: `lts__t_sectors`, `lts__t_sector_hit_rate`
- Then toggle **FP8 KV cache on/off** — that is his todo #2 stated literally, and it becomes a clean test: does shrinking KV traffic restore GEMM-visible L2 capacity and move the gate back?

**Hypothesis:** `C_eff` under load is materially below 36 MB, so the gate moves left and every isolated-microbenchmark crossover in the literature — **including our own** — is optimistic in production.

Supporting prior art: L2 *capacity* contention persists even under strict SM isolation (green contexts); decode kernels are L2-pollution-sensitive.

New files: `profiling/serving/bench_serving_l2.py`, `profiling/serving/ncu_serving.sh`.

### P4 — Compute quantization (Xiao todo #3)
- W8A8 on dense shapes (native H100 IMMA, dequant in epilogue → **ceiling term should vanish**)
- W4A4 / MXFP4 emulated on H100 (**no FP4 tensor core → ceiling term should dominate**)
- **Model claim to test:** one parameterization predicts weight-only *and* compute-quant by setting `ceiling_dequant → ∞` when the precision has native MMA support

If that holds, this stops being a curve fit and becomes a model. Extend `profiling/cuda_validation/carm_cuda_fit.py`.

### P5 — Portability (the FlagOS pitch)
- Package the parameter measurement as a standalone microbenchmark that emits `{C_eff, bw_l2, bw_hbm, r_dequant, ceiling}` for **any** backend
- Populate more rows in `gpu_parameterized_crossover` — A100 data already exists in-repo; **one non-NVIDIA backend would be transformative**
- **Fit on architecture A, predict on architecture B, report MAPE.** That single number is the difference between a benchmark study and a paper

New file: `profiling/portable/measure_params.py` (generalize `measure_carm_params.py`, strip H100 assumptions).

### P6 — Dispatch cost model (analysis only)
No paper found that quantifies this. Runtime precision dispatch must either hold **both** weight formats resident (memory blowup → smaller KV cache budget → lower max concurrency → lower throughput) or **repack on the fly** (latency on the critical path). Write the cost model; note it couples back to P3 (more weight memory → less KV cache → gate moves again).

New file: `profiling/dispatch/cost_model.py`. This is what separates "an interesting measurement" from "a dispatch policy someone can ship."

---

## 7. Related work we must position against

| Work | Threat | Our differentiation |
|---|---|---|
| **tritonBLAS** (arXiv 2512.04226) — analytical model using **cache hierarchy** + shape to pick GEMM configs, **in Triton**, ≥95% of autotuning at zero tuning cost, portable by retuning hardware params | **Highest.** Nearly our sentence minus precision | They select **tile/blocking config at fixed precision**; we select **precision**. Cite in the intro, state the delta in one line, consider *building on* them |
| Microbenchmark-Driven Analytical Perf Modeling Across GPU Architectures (2605.04178) | High — owns "portable via microbenchmarked params" | We add the capacity gate + the precision decision |
| **QServe** (2405.04532) | Medium — owns dequant-overhead-in-main-loop | We make it a composable, predictive roofline ceiling |
| **MARLIN** (2408.11743) | Medium — owns the batch crossover | We explain *why* via capacity and predict *where* |
| APEX4 (2606.08761), SharQ, COMET | Low — W4A4 kernel design | Orthogonal; they build kernels, we decide when to dispatch them |
| Cache-Resident LLM Inference (2606.25353) | Low — **CPU** GB-scale 3D-stacked LLC | Same intuition, different hardware. Cite as validation, not competition |
| LLM Inference Unveiled (2402.16363), RooflineBench (2602.11506) | Low | Confirms the gap: LLM roofline work analyzes precision **without a capacity term** |

**Bring tritonBLAS to Dr. Xiao unprompted, with the differentiation already worked out.** Surfacing the nearest competing work yourself is exactly the judgment an advisor is checking for.

---

## 8. Explicitly out of scope

- Anything further on MoE / sparse operators — he closed it 2026-07-01
- Chasing B200 access — not on the critical path; asking for hardware before showing a dense result is the wrong order. **Ask for a non-NVIDIA backend instead** (P5), which is worth far more to the thesis
- Re-running weight-only sweeps for more precision on the large-T crossover — settled and unexciting
- Reviving MLA reconstruction as a headline

---

## 9. Hardware strategy — why H100 is the right platform, not a limitation

This objection will come up in the meeting and in review: *"Your results are on H100; the industry is moving to B200/GB200/Rubin. Is this already dated?"* The answer is no, and it should be a **paragraph in the paper**, not a defensive footnote.

**(a) H100 is not being retired — it is being reassigned to inference, which is our workload.**
Industry view as of GTC 2026: frontier silicon takes training, prior-generation GPUs cascade to inference — "the fleet is a waterfall, not a scrapheap." A100s (2020) remain fully booked for inference in 2026; H100s off expired contracts rebooked at ~95% of original pricing. The Azure K80/P100 precedent implies 7–9 year service lives, putting H100 viable into 2030–2032. Rubin entered production Jan 2026 with H2 volume, ~30k units in 2026 and ~100k in 2027 — negligible against the installed Hopper/Ampere base, and allocated to frontier training first.

We study **inference kernel dispatch**. The platform is converging on the subject matter.

**(b) The LLC growth trend makes the gate MORE important, not less.** ★ this is the argument to lead with

| Architecture | LLC | Native low precision |
|---|---|---|
| A100 | 40 MB L2 | — |
| H100 | 50 MB L2 (**36 MB effective**) | FP8 |
| B200 | ~126 MB L2 | FP4 |
| B300 | ~192 MB L2 | FP4 |
| CPU (3D-stacked) | GB-scale LLC | — |

The gate predicate says quantization stops paying once `W < C_eff`. **As `C_eff` grows, the fraction of GEMM shapes falling below the gate grows.** The "quantization does not pay" regime is *expanding* with every hardware generation.

Two consequences worth stating explicitly in the paper:
1. Had this study started on B200, the effect would have been **stronger**. The H100 measurement is the **conservative** one — the bias runs in the favorable direction.
2. It gives us a trend line rather than a snapshot: three cache sizes, the gate moving predictably, and an extrapolation to Rubin-class LLCs. That is a discussion-section contribution.

**(c) Cross-architecture transfer is the immunization — not a newer chip.**
Redoing everything on B200 produces a B200 paper that dies when Rubin ships: the treadmill, one lap ahead. The escape is a model parameterized by `(C_eff, bw_l2, bw_hbm, r_dequant, ceiling)`, **fit on one architecture and validated on another**. Then "what about Rubin?" answers itself: measure three numbers, run the model. The gap becomes the feature. This is P5, and it is why P5 matters more than any hardware upgrade.

**(d) A100 is the high-value next data point, not B200.**
A100 → H100 → B200 spans **both axes the model is parameterized on**: cache size (40 / 50 / 126 MB) and native-precision support (none / FP8 / FP4). Prove transfer across the first hop cheaply (partial A100 data already in-repo), and reserve B200 for one specific later prediction: *the ceiling term vanishes when the precision has native tensor-core MMA*.

**(e) Dr. Xiao's group is on H100.** His `perf_analysis_qwen3.6-27b` profile is dominated by `nvjet_sm90_*` kernels — SM90 is Hopper, captured July 2026. Most FlagOS domestic silicon is Hopper-class or below. Matching his platform makes results directly actionable for his team rather than aspirational.

**Honest residual risk:** the model absorbs a *magnitude* change in cache capacity by construction, but not a *structural* change in the memory hierarchy. If Rubin reorganizes the hierarchy qualitatively rather than scaling it, the parameterization needs revisiting. Monitor; do not preempt.

**Hardware shopping list**
- **Primary:** H100 80GB **SXM5** (PCIe has different L2 behavior; all repo params are SXM5). Hard requirements: root or `NVreg_RestrictProfilingToAdminUsers=0` (else `ERR_NVGPUCTRPERM` blocks all `lts__*` metrics and kills P1 validation + all of P3); **exclusive whole GPU, no MIG** (MIG partitions L2, making contention experiments meaningless); `nvidia-smi -lgc` privileges for clock locking.
- **For P2/P3 only:** Qwen3.6-27B bf16 ≈ 54 GB of weights; at `p32768d1024` concurrency 64 an 80 GB card leaves ~14 GB for KV, likely insufficient. Use **H200 141GB** or **TP=2 × H100**.
- **Cheap and high-value:** a few hours on **A100 80GB** for P5 transfer validation.
- **Later, briefly, at spot:** B200 for the native-FP4 ceiling prediction only.

---

## 10. One-paragraph pitch (for the meeting, and for the abstract)

> Modern checkpoints ship natively quantized (MXFP4 experts, FP8 dense) and must run across a dozen accelerator architectures with different cache hierarchies and dequantization throughputs, most without native FP4 tensor cores. Whether the low-precision path actually wins for a given GEMM is therefore a per-shape, per-architecture decision, and the field currently makes it with hand-tuned heuristics or exhaustive autotuning, neither of which ports. We show the decision is governed by a capacity condition invisible to standard roofline analysis — quantization's benefit collapses once the weight working set becomes last-level-cache resident — and give a cache-aware model, parameterized by three microbenchmarkable hardware constants, that predicts the crossover on unseen architectures and can be evaluated at dispatch time.
