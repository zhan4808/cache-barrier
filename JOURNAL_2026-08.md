# JOURNAL — 2026-08 work log

Running log of the 2026-08 sessions. Newest entries at the bottom. Companion to
`RESULTS_2026-07.md` (the pivot-cycle ledger) and `DIRECTION.md` (the plan).

---

## 2026-08-01 (session 1, morning)

**Environment**: fresh H100 80GB SXM5 instance, NFS intact. System python has
torch 2.7.0 + triton 3.3.0 + CUDA — no venv restore needed for microbenchmarks.
Clock lock verified working (`sudo nvidia-smi -lgc 1755`, passwordless).

**Reconciliation** — DIRECTION.md (drafted 2026-07-31) was written against the
June repo state; the 2026-07-01→02 session had already executed P2/P3/P4
(`dense_qwen/`, `kv_serving/`, `mechanisms/`, W8A8-vs-tuned). Two July findings
supersede DIRECTION's framing: contention kills the L2 regime as a *step
function* (not "gate moves left"), and KV reads are not L2-limited. Added
reconciliation note atop the committed DIRECTION.md; ledger in RESULTS_2026-07.md.

**P1 — capacity-gate figure** (`profiling/gate/`, commit `8f62ad1`):
- Sweep W ∈ 8–128 MB × T ∈ 1–512 × {bf16 cuBLAS, W4A16 Triton, W8A8 IMMA},
  batched-BMM family (H=128, K=128) that all CARM params were fit on.
  CUDA-graph timing, clock-locked, per-cell mini-autotune (BLOCK_N 64/128),
  kernels verified vs explicit-dequant references (rel err 3e-4 / 9e-3).
- **Result: the sign flip brackets C_eff.** T ≤ 32: W8A8 0.67–0.70× below →
  1.2–1.46× above, flipping between W=32 and 40 MB (measured C_eff = 36).
  W4A16 flips at the same place but far-fields at *parity* — its 0.496 TB/s
  in-core dequant ceiling binds, exactly as CARM predicts. T ≥ 64: everything
  < 1 (compute-bound; the MARLIN batch-64 collapse).
- Model form improvement: split-mem CARM (weights at gated BW, act/out at HBM
  BW) beats the lumped kernel-compass form: bf16 below-gate MAPE 25.3→20.9%,
  w8a8 43.4→33.8%. Adopted for the overlay; both reported in gate_mape.json.
- **Honest failure logged**: Triton quant kernels fall off their own rooflines
  at T ≥ 64 (w4a16 65–81% MAPE, w8a8 43–53%) while cuBLAS stays ~17%.
  Kernel limitation (guardrail 5), confined to the never-dispatched regime.
- Note for the paper: the kickstart's "W4A16 rises toward MARLIN's 3–4×"
  success criterion is *unreachable for this kernel by construction* — the
  ceiling-free far field is carried by W8A8 and by the CUDA Marlin validation
  (ceiling 423 TFLOPS). The figure says this in its own caption.

**P0 — reframe** (commit `f24a5a3`): new title "When Quantization Stops
Paying: A Capacity-Gated Roofline Model for Precision Dispatch on GPUs";
abstract rewritten (gate first sentence, MARLIN as far-field limit, contention
step + mechanisms as boundary conditions, MLA → case study); related-work
positioning vs tritonBLAS/QServe/MARLIN/APEX4 added; README rewritten; paper
compiles with 0 errors. Body still needs the same surgery.

Also: RESULTS_2026-07.md written; WeChat draft at
`docs/wechat_update_2026-08-01.md`; clocks reset after measurement.

---

## 2026-08-01 (session 2) — P5 portability + P6 dispatch cost model

Goal: run the phase plan to the end of P6. Entries below written as the work
happens.

### P5 — portable parameter harness + A100 transfer (`profiling/portable/`)

**Harness** (`measure_params.py`): emits {C_eff, bw_l2, bw_hbm, peak,
r_dequant, t0} on any torch.cuda backend; every sweep size derived from the
device's nominal L2, no datasheet numbers, graph-timed differential slopes.
C_eff detection lesson: absolute BW plateaus of a single warm buffer are
launch-cost-contaminated, but the residency *collapse* is unmistakable — warm
re-read BW rises with size (fixed cost amortizes) then breaks downward at the
effective capacity. Detector = argmax-then-first->5%-drop, midpoint, ±half-step.

**H100 self-consistency** (clock-locked): C_eff 39.0 MB (NCU-based June band
32–40, headline 36) ✓; bw_l2 5.26 vs 5.33 ✓; bw_hbm 3.26 vs 3.15 ✓;
t0 2.33 vs 2.80 ✓; r_dequant 0.58 vs 0.50 (different operating points, 17%) ~;
**peak fp16 737 TFLOPS achieved vs 989 in carm_model.json** — the 989 was
near-datasheet; 737 is what cuBLAS actually achieves on large square GEMMs and
is the honest compute ceiling. Doesn't move the gate (memory-side), but the
compute-bound far-field predictions should use it.

**A100 transfer** (`transfer_validation.py`, against the in-repo 2026-06 A100
sweep, eager-timed):
- fp16 **zero-shot** (model form + estimated A100 hw constants, nothing fitted
  to A100 kernels): 28.8% MAPE below gate / 22.8% above. Upper bound — the
  A100 constants are carm.py *estimates*; a few hours of A100 time with the
  new harness replaces them.
- W4A16 **two-point calibrated** (r_dequant + fixed cost from 2 operating
  points, predict the other 22): **19.9% MAPE**.
- **Naive scaling fails**: scaling H100's r_dequant by peak ratio gives 40.6%
  MAPE and completely misses the A100 kernel's measured **58.7 µs fixed cost**
  (H100: ~6 µs at the same shape). Kernel terms are implementation properties,
  not hardware-scalable — they must be *measured* per architecture, which is a
  two-point microbenchmark. This sharpens the P5 pitch: the portable harness
  is not optional tooling, it is the transfer mechanism.

### P6 — dispatch cost model (`profiling/dispatch/`)

One input measured live (H100, graph-timed): unfused int8→bf16 repack runs at
0.133 T elems/s (27B model: 203 ms; fused lower bound bw_hbm/3 B = 25 ms).
Three storage policies for runtime precision dispatch, Qwen3.6-27B-class dense
model at p32768d1024 (KV ≈ 2.21 GB/seq — only the 16 full-attn layers hold KV):

- **A. Dual-resident**: the second format is paid in KV budget. H100 80 GB:
  **infeasible** (base concurrency 9 → 0). H200 141 GB: 36 → 24 seqs =
  **−33% concurrency ≈ −33% decode throughput**.
- **B. Repack on switch**: 1% overhead needs a switch period ≥ 20 s (measured
  unfused) / ≥ 2.5 s (fused bound). Engine-mode granularity only; per-phase
  (~100 ms) or per-layer switching is out.
- **C. JIT dequant via HBM scratch**: traffic per use = 4.5× per int4 elem-byte
  vs 2× for resident bf16 → **2.25× worse than not quantizing. Never pays.**
  JIT dequant *in-kernel* is not a new option — it is exactly the r_dequant
  ceiling CARM already prices.

**The conclusion the numbers force** (and the inversion worth putting in the
paper): with bf16-primary storage, dispatch is memory-infeasible or
switch-limited. The only zero-marginal-cost policy is **quantized-primary
storage** with per-shape choice between quantized-compute kernels (W8A8/W4A4)
and dequant-in-kernel bf16 compute priced by r_dequant. So the dispatcher's
question is not "when do we quantize?" but "when do we pay the dequant ceiling
vs take the quantized-compute path?" — which is exactly the world natively-
quantized checkpoints (MXFP4 experts, FP8 dense) already put us in
(DIRECTION.md §1). Couples to P3: dual-residency would also shrink KV traffic
and raise GEMM-visible L2 capacity, but that is second-order vs the
concurrency loss.

**Phase plan status: P0–P6 all executed** (P2–P4 in the 2026-07 session;
P1/P0 this morning; P5/P6 this session). Remaining open threads are the
served A/B harnesses (need vLLM env + model download), a real A100 run of the
portable harness to replace the estimated constants, and paper body surgery.

### Presentation

Built the MLSys-style deck: `docs/presentation_2026-08-01_gate.html` (NFS docs,
not in git — matches convention for decks). 12 slides, assertion-evidence
style, one claim + one visual per slide; data figures re-rendered in the deck
palette from the committed JSONs; concept diagrams (hierarchy, gate predicate,
contention step, LLC trend) as inline SVG. Also published as a private
artifact for browser viewing. Flow for a cold reader: promise → contradiction
(MARLIN vs us) → missing capacity term → the gate → Figure 1 → the
three-parameter model → measurement hazards → boundary conditions → A100
transfer → dispatch storage policies → LLC growth trend + takeaways.

## 2026-08-01 (session 3) — deck completion + the KV question

**KV-cache quantization, placed** (extends `dispatch/cost_model.py`): fp8 KV
is a *memory* lever, not a speed lever. Speed: e2e ceiling ≤0.2% (KV reads not
L2-limited, full attn 2.67% of runtime, fp8 decode kernel BW-ceiling-bound).
Memory: halving the 2.21 GB/seq budget doubles max concurrency (9→18 on
H100-80GB, 36→73 on H200) — and decode throughput scales with concurrency.
Verdict: outside the gate (KV is not a GEMM operand), inside the budget —
quantize KV for concurrency, not kernel speed. Deep KV-compression research
(KVQuant etc.) stays out of scope: crowded field, orthogonal mechanism.

**Deck finished** (12 → 17 slides): added Amdahl motivation (mm = 86.2%),
dense-Qwen operand-aware boundary (new deck render of
results_l2_boundary_h100.json), contention step upgraded from sketch SVG to
the real kv_proj data (results_contention_h100.json), mechanisms expanded to
a six-panel taxonomy slide, the KV verdict slide, and a status/roadmap slide.
Every data slide now has committed JSON behind it.

## 2026-08-01 (session 4) — served pipeline, paper body, deck v2

**Deck v2** (`docs/presentation_2026-08-01_gate.html`): horizontal
arrow-key/swipe navigation (presentation-style, one slide at a time),
per-slide data-source line (repo path of every number shown), viewport-fit
constraints (figures ≤52vh), content column centered at ≤1180px. Verified
with Playwright at 1600×900 (all 17) and 1280×720 (figure slides): caught and
fixed (1) missing `<meta charset>` → mojibake under Chromium's windows-1252
guess, (2) `.src` descender clipping, (3) an SVG caption overrunning its
viewBox, (4) undersized visuals from the old narrow column.

**Paper body surgery** (commit `6244190`): new §3–§6 (Capacity Gate /
Boundary Conditions / Transfer / Dispatch) with the three new figures;
MLA+INT4 demoted to Case Study sections; conclusion + limitations rewritten.
**Correction of an earlier claim**: session-1's "compiles clean" checks were
reading a stale June `main.log` — pdflatex was not installed on this
instance. Installed TeX Live; the reframed paper genuinely compiles: 18
pages, 0 errors, no undefined refs.

**Served pipeline debugging** (fresh instance realities, in order):
1. vLLM default `max_num_seqs=1024` > 343 Mamba cache blocks of the hybrid
   GDN model → engine init failure. Fix: `max_num_seqs=64` in both harnesses.
2. vLLM 0.20.2's deep_gemm fp8-eligibility scan raises when `deep_gemm` is
   absent — even on the bf16 leg. Fix: `VLLM_USE_DEEP_GEMM=0
   VLLM_MOE_USE_DEEP_GEMM=0` (cutlass fp8 is the deployed path we model
   anyway).
3. GDN linear-attn JIT needs `ninja` (lost with the old instance's apt/pip
   state). Fix: pip+apt install.
Also: `pkill` of the runner leaves EngineCore children holding 33 GB — kill
via `nvidia-smi --query-compute-apps` pids before relaunching.

**layer_relerr on real weights** (`served/results_layer_relerr_h100.json`):
w8a16 rel-err ≈0.027, all w8a8 granularities ≈0.0375 ≈ the gaussian floor,
including the kurtosis-120 o_proj — accuracy is not the discriminator between
these precisions on this checkpoint; speed (the gate) is.

**Served A/B complete** (`served/results_served_ab_h100.json`, 64 seqs,
max_model_len 4096, deep_gemm off → cutlass fp8 path):
- decode (256 gen tok/seq): bf16 2025.5 → fp8 **2930.9 gen tok/s = 1.447×**.
  The e2e decode projection (dense_qwen, 2026-07: W8A8-unfused 1.19× /
  fused ~1.6×) brackets the measured 1.45× — the projection is now a
  measurement.
- prefill (~27k prompt tok, 1 gen): 12563 → **15096 total tok/s = 1.20×**
  (compute-bound regime, smaller win, as the model says).
- band demo, stated plainly (guardrail 8): capping max_num_batched_tokens
  (512-aligned vs 460-misaligned) only *hurts* decode throughput (2931 →
  2718/2818) — chunked-prefill overhead swamps wave-band effects at engine
  level for this workload. Kernel-level bands (mechanism B) do not surface
  in a decode-dominated engine A/B; they matter for prefill batch shaping,
  which this workload doesn't exercise. Demo closed as a negative result.

## 2026-08-01 (session 5, A100) — measured constants replace estimates

**Environment**: A100-SXM4-**40GB** (not the 80GB the 2026-06 target data came
from — same GA100 die, L2 and SMs, but HBM2 at ~1.5 TB/s vs the 80GB's HBM2e
~1.94; flagged before measuring, decision below). MIG disabled, passwordless
clock lock at 1410 MHz, system torch 2.7.0 + triton 3.3.0. NFS copy of the
project rsynced from the east box (`robert-nfs-west-2`).

**The 40-vs-80GB decision**: measured constants from this card cannot honestly
predict the 80GB eager-timed target — bw_hbm differs by ~28% and it is the
dominant above-gate parameter. Chosen resolution: promote the graph-timed
re-measure (kickoff stretch #2) into the primary path and score transfer
against a **self-consistent target** — the same 48-cell sweep re-measured on
the same GPU the parameters come from. The 80GB eager dataset stays as a
secondary target with both mismatches (bw_hbm, host eager floor) stated.

**Harness run** (`params_nvidia-a100-sxm4-40gb.json`, committed `198b589`):
C_eff **31.2 MB** = 0.78× nominal 40 — the same effective/nominal ratio as
H100 (39.0/50). bw_l2 3.86, bw_hbm 1.51 (HBM2, as expected for the 40GB
part), peak fp16 achieved 257.7 TF (vs 312 datasheet; same achieved-vs-quoted
gap as H100's 737/989), t0 2.30 µs graphed / 32.8 µs eager (this host's eager
floor is ~2× the 2026-06 host's 15.5 µs — floors are host properties).
r_dequant **0.394 TB/s vs 0.406 fitted from the 2026-06 data** (3% apart,
different host and timing discipline) — kickoff stretch #3 answered for free:
the two-point microbenchmark and the full-sweep fit agree on this kernel.

**Graph-timed re-measure of the 48-cell sweep**
(`results_l2_barrier_a100_40gb_graphtimed.json`): removing the eager floor
rewrites the qualitative picture of the 2026-06 dataset. The int4 kernel's
notorious 58.7 µs "fixed cost" collapses to **3.5 µs** under graphs — it was
eager launch overhead, not a kernel property. Kernel-level, int4 *wins* at
bs ≤ 4 (0.72–0.94×, advantage growing past C_eff), sits at ~parity at bs=16
(losing only below the gate, 1.25–1.32×), and loses everywhere at bs=64
(2.4–3.8×, dequant-compute-bound) — the gate pattern, visible in raw
latencies, on a second architecture. The measured fp16 curve shows a slope
break at ~31 MB, i.e. C_eff is visible by eye in the latency data.

**Transfer validation re-scored** (`transfer_validation.py`, both targets,
guardrail 7/8):
- PRIMARY (self-consistent 40GB, graph-timed): fp16 zero-shot **44.0% below
  gate / 19.0% above**; W4A16 two-point calibrated 30.9%, naive scaling 61.4%.
  The below-gate number is the honest headline of this leg: with every
  constant measured on the very GPU being predicted, the split-mem CARM form
  *underpredicts small L2-resident kernels* — same direction as H100's
  below-gate residual (20.9% after the split-mem fix), larger here. Model-form
  finding, not a constants problem; the constants excuse is now spent.
- SECONDARY (2026-06 80GB eager): fp16 zero-shot **28.8→18.3% below /
  22.8→13.0% above** — measured constants improved the original claim's
  numbers despite the two stated mismatches. Naive kernel scaling 40.6→28.1%,
  still ~1.4× worse than the 19.9% two-point calibration; the "kernel terms
  must be measured" conclusion survives the constants upgrade.
- Calibrated r_dequant on the graph-timed target: 0.418 TB/s vs harness 0.394
  (6%) — consistent operating-point picture across three measurement routes.

**Gate flip on A100** (kickoff stretch #1; `bench_capacity_gate.py`
parameterized to `--params/--tsweep/--out`, reduced sweep T ∈ {1,16,32}):
33 cells, sanity rel_err 3e-4 / 9e-3, results in
`profiling/gate/results_capacity_gate_a100.json`. **The H100 sign flip does
not replicate; the capacity structure does.** On this card w8a8 wins below
the gate at T=1 (1.05–1.11×, no flip at all) and flips at 8–16 MB at
T=16/32 — nowhere near C_eff. Mechanism: H100's below-gate w8a8 *loss*
(0.67–0.70×) was act-quant overhead against a fast bf16 baseline; A100's
bf16 is relatively slower (bw_l2 3.86 vs 5.3, 1410 vs 1755 MHz), so the
overhead never wins. What does transfer is the gate's *derivative*
structure: the w8a8 advantage peaks exactly in the predicted
asymmetric-residency band 31–62 MB bf16-equivalent (T=1: 1.83× at 32 →
**2.07× at 40** → declining once w8a8's own operand approaches capacity),
the bf16 latency curve breaks at ~31 MB, and w4a16 at T=32 loses everywhere
(0.54–0.77×, dequant ceiling binds). One model-refinement lead: at T=32 the
w8a8 step-down lands at operand ~20 MB < C_eff, consistent with
total-footprint (weights+act+output ≈ 25 MB) crossing capacity rather than
the weight operand alone — the operand-aware gate may need a footprint term
at larger T. MAPE repeats the transfer finding: bf16 18.6% above gate,
53.6% below (n=12/21) — the below-gate model-form weakness is now confirmed
on two independent A100 datasets. So the transfer claim upgrades carefully:
**the gate's capacity structure transfers; the sign of the below-gate branch
is architecture-dependent** (it hinges on baseline kernel quality, which is
exactly what the per-kernel terms of the model are for).

**Paper/deck**: §Transfer, contributions bullet, setup hardware paragraph,
limitation bullet, and conclusion updated to the measured numbers; portability
slide and status slide updated (⌁ source lines + MAPE). WeChat addendum
drafted. Clocks reset (`-rgc`) at session end.
