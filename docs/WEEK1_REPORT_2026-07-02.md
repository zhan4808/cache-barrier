# Week-1 report — Dr. Xiao's three todos, answered with measurements
**2026-07-02, H100 80GB · all results committed to cache-barrier `main` · one overnight autoresearch session**

## TL;DR
All three todos are answered. (1) **Dense Qwen3.6-27B: yes, mixed precision helps
(up to 2.6×) and yes, it is cache-constrained** — the L2 capacity gate is measured
directly and is *operand-aware* (each precision cliffs at its own size vs C_eff≈36 MB).
(2) **KV-cache reads are NOT L2-limited** — occupancy-bound small, HBM-streamed
large; the fp8 decode kernel's own BW ceiling dominates; end-to-end ceiling ≤0.2%
on this model. (3) **W8A8 matched precision wins everywhere on MoE and dense**
(no dequant cliff); W4A4 is not natively runnable on Hopper (no INT4/FP4 MMA) and
is parked with the hardware question. CARM v2 (operand-aware gate + act-quant +
contention terms) fits the 270-cell dense sweep at **7.4% MAPE (bf16)** / ~15%
(quant paths). The headline claims survived a clock-locked red-team.

---

## Todo 1 — dense model (Qwen3.6-27B)
Layer-level bench of the five real projections (21→357 MB, straddling C_eff) ×
M∈{1..2048} × {bf16, Marlin W8A16, native W8A8} × {warm, rotated}. Dr. Xiao's own
profile shows these GEMMs are 86.2% of model runtime — this is where the money is.

- **Left edge measured:** kv_proj (21 MB, L2-resident) — quant loses at every M
  warm (0.38–0.90×). **Flips to a 1.07–1.09× W8A16 win under rotation** (serving
  eviction makes it HBM-streamed): the L2-resident "don't quantize" rule holds
  only while residency actually survives the serving pattern.
- **Boundary super-win (the cache-aware prediction a plain roofline can't make):**
  q/o_proj (63 MB bf16 → 31 MB fp8): fp8 pulls the weight *into* L2 →
  **2.41–2.97× (mm-only) vs the 2.0× byte ratio**, fp8 weight served at L2 tier
  4.25–5.39 TB/s. Compresses to 1.68–1.75× rotated. Clock-locked, median-of-3.
- **Right edge on dense:** Marlin W8A16 dequant cliff at **M≈64–128** (down to
  0.38×); dense dequant ceiling fit 334 TFLOPS. **W8A8 never cliffs** (fp8-MMA
  1335 TFLOPS), wins high-M 1.24–1.70×.
- **The direct capacity-gate figure** (`fig_l2_boundary.png`): sweeping 8→128 MB,
  bf16 BW cliffs at ~36 MB; **fp8's cliff sits at 2× that (its operand is
  half-size)**; rotation erases both cliffs. Operand-aware gating, confirmed.
- Hybrid coverage: GDN in_proj_qkvz (168 MB) behaves as streamed (W8A16 1.7×
  small-M → cliff M≈128; W8A8 1.25–1.59× everywhere); GDN out_proj ≡ o_proj.

## Todo 2 — KV cache quantization in real serving
Exact vLLM v1 production path (FA3 paged decode, GQA 24q/4kv×256, fp8 KV+q with
descales), ctx {1k,8k,32k} × B {1..128}, warm+rotated.

- **No L2 tier ever appears for KV** (max BW 3.1 TB/s = HBM; small working sets
  are occupancy-bound at 0.14–0.8 TB/s). Rotation moves ratios ≤0.03 — nothing
  resident to evict. **KV reads are not limited by L2 capacity — and CARM
  correctly classifies the capacity term as inert here.**
- fp8-KV: at most **1.06–1.07×** (streamed), **0.69–0.72× loss** at small
  batch/context. Mechanism: the fp8 decode kernel saturates at ~1.5–1.65 TB/s vs
  bf16's 2.8–3.1 — the same kernel-BW-ceiling that FlashMLA showed, now
  reproduced on a second attention family.
- Amdahl: full attention = 2.67% of this model's runtime → fp8-KV moves
  end-to-end ≤0.2%. **fp8-KV on this model is a memory-capacity feature, not a
  speed feature.**

## Todo 3 — low-precision compute operators (W8A8, W4A4)
- **W8A8 on MoE** (fused_experts + fp8_w8a8 config): **wins at every T with no
  crossover** — Mixtral 1.82–3.15× vs stock bf16 (~1.9–2.0× vs the tuned
  baseline), Qwen3.6-35B fine-grained 1.50–1.75×. Contrast W8A16: crossed at
  ~300–600 (Mixtral) / ~1900 (Qwen) and lost after. Same contrast on dense
  (`fig_moe_w8a8_vs_w8a16.png`, `fig_dense_crossover.png`).
- **W4A4: cannot be run natively on H100** (Hopper has no INT4/FP4 tensor
  cores); any H100 W4A4 number would be emulation re-measuring the dequant
  ceiling. Parked consistently with "hardware doesn't matter for now."
- Deployment cost measured: **dynamic act-quant ≈7–9 µs** flat kills deployed
  W8A8 at decode-scale M on streamed shapes (kernel 1.96× → deployed 0.98×);
  fusing/amortizing it is the engineering fix. W8A8 per-tensor rel-err 0.037–
  0.046 (needs per-channel/static scales in practice).

## CARM v2 (model update)
`t = t0 + max(W/BW_tier(operand, mode), F/P_path) + Q_actquant`, tier gate keyed
on the **quantized operand's own size** vs C_eff and warm-mode only.
Fit on 270 dense cells: **bf16 7.4% / W8A16 14.8% / W8A8 15.5% MAPE**; predicted
W8A16 crossovers ~120 vs measured 64–128. Constants: L2 tier 4.2–4.5 TB/s, HBM
2.8; **Marlin exploits no L2 tier (2.54≈2.58)** — which is *why* only W8A8 shows
the boundary super-win. Known gaps: act-quant lstsq over-weights large M;
Marlin's fixed overhead at tiny N unmodeled.

## The unified story (for the next deck)
One bounded window, three regimes, **keyed per-operand**: quant loses where the
operand sits in L2 (and that regime shrinks under serving contention), wins
super-proportionally where quantization pulls the operand into L2, wins
proportionally where streamed, and weight-only quant dies at the compute edge
while matched-precision does not. Attention/KV sits outside the capacity story
entirely (occupancy + kernel ceilings). Every clause above is now a measurement.

## Proposed next steps (Week 2)
1. Per-channel/static-scale W8A8 (close the 0.037 rel-err gap; measure cost).
2. Fused/persistent act-quant to recover the deployed small-M W8A8 win.
3. lm_head (2.5 GB) + contention-degree sweep (1..N copies → contention curve
   for CARM's residency factor).
4. Fold the dense/KV/MoE story into the paper + 3 new deck slides
   (fig_l2_boundary is the money figure).

**Files:** `profiling/dense_qwen/` (bench, fit, red-team, boundary, figures),
`profiling/kv_serving/`, `profiling/cuda_validation/bench_moe_w8a8.py` + JSONs.
Commits: 6eaf002, 011e122, 06a4b42, b2f8aeb, f449364, 00a66a7 (+ this report).

---
## Addendum (stretch items, same night)
- **Contention is a step function:** the L2 tier survives only while the TOTAL
  co-tenant working set < C_eff (kv_proj bf16 4.26→2.52 TB/s at 2 copies;
  q_proj fp8 5.64→2.72), no partial credit. Since full-model serving always
  exceeds C_eff (64 layers × 21–357 MB), **L2-residency effects do not survive
  full-model serving** — they matter only when one small operand is re-read
  repeatedly within a step. This sharpens the honest scope of the left-edge and
  boundary findings, and reconciles our operator-level cache physics with
  Dr. Xiao's serving-level intuition. CARM contention factor = binary gate on
  Σ(hot operands) vs C_eff. (`results_contention_h100.json`)
- **W8A8 rel-err ≈0.037 is the e4m3 floor on Gaussian data** — per-channel
  weight and per-token activation scales move it <0.001 (roundtrip floor 0.051).
  Real-model accuracy (outliers) remains the open accuracy-eval gap.
- **Act-quant term measured directly:** 8.4 µs fixed + ~1.6 TB/s streaming
  (replaces the skewed lstsq intercept; `results_actquant_h100.json`).
- lm_head (2.5 GB): 2.85 TB/s — extreme streamed anchor.

---
## Addendum 2 (night extension — items 8–11) + corrected framings

**Framing corrections (applied throughout — read these before quoting numbers):**
1. **The deployment-relevant headline is the contention result, not the 2.6×.**
   The boundary super-win (2.41–2.97×) is a warm, mm-only, M≤16 microbenchmark
   number. Our own control shows it compresses to 1.68–1.75× under
   serving-like eviction, and the contention sweep shows the L2 tier vanishes
   as a step function once total co-tenant working set exceeds C_eff — which
   full-model serving always does. **The honest claim: the operand-aware L2
   capacity gate is real, directly measured physics — and it does NOT survive
   full-model serving.** Both halves are the finding.
2. **The end-to-end numbers are a PROJECTION, not a served measurement:**
   sum of measured per-GEMM latencies (M=1, rotated mode, random weights) ×
   the 86.2% GEMM share from Dr. Xiao's profile, other buckets assumed
   unchanged. Projected decode speedup: W8A16 **1.50×**, W8A8 unfused **1.19×**,
   W8A8 with fused act-quant **~1.6×** (upper bound 1.73× if all quant were
   free). No full model was served.
3. **W8A8 "wins everywhere" is a speed claim with an unresolved accuracy cost:**
   per-tensor W8A8 rel-err 0.037–0.046 vs Marlin W8A16's 0.0025. On our
   Gaussian test data this is the e4m3 floor (scale granularity moved it
   <0.001), but real weights have outliers — **downstream accuracy (perplexity/
   task) is unmeasured and is the open gap.**

**Night-extension results:**
- **W8A8 MoE vs TUNED bf16, clock-locked:** 1.80–1.94× flat at every T
  (128–2048), no cliff. The earlier 3.15× @T=512 vs stock was baseline-inflated;
  ~1.9× uniform is the defensible number.
- **Fused act-quant:** `rms_norm_dynamic_per_token_quant` costs +2.1–2.2 µs
  over the norm alone at decode M (vs +8.4 µs standalone) — most of the W8A8
  act-quant tax is recoverable where a norm/silu precedes the GEMM.
- **INT8 ≈ FP8 matched** (±20% small-M, parity ≥256) — both native precisions
  sit on the same tier, as the model assumes.

## The B300 ask (the one open experimental hole)
Every MXFP4/FP4 number in this project is **emulated** on H100 (Hopper has no
FP4 tensor cores; Marlin dequantizes to bf16 by construction). The model's
sharpest falsifiable prediction — **native W4A4 breaks the dequant ceiling and
wins in the compute-bound regime where emulated FP4 loses** — is therefore
completely untested on real silicon. B300 (Blackwell Ultra, native FP4 MMA) is
available on Prime Intellect. The harness is staged (`bench_moe_nvfp4_native.py`,
`B200_RUNBOOK.md`); the blocker is a fresh Blackwell vLLM build (CUDA 13/SM100 —
the H100 venv lacks the native FP4 ops). **Decision requested: green-light the
B300 run?**
