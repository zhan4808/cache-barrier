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

## 2026-08-01 (session 6, H100) — footprint gate, below-gate decomposition, paper compression, prefill bands

**Environment**: fresh H100 80GB SXM5 instance (new IP), NFS intact with the
A100 session's commits synced back from `robert-nfs-west-2` (the A100 leg ran
on a west-region copy; `repos/` + `docs/` rsynced back, both regions now
carry `67c1138`+). System torch 2.7.0/triton 3.3.0 for microbenchmarks;
served env restored from the NFS snapshot; Qwen3.6-27B re-downloaded (52 GB).
Two restore lessons journaled: the 2026-08 extras addendum in `restore.sh`
sat *after* the byte-exact path's `exit 0` and never ran (fixed — extras now
install on every path), and flashinfer's GDN JIT needs *apt* ninja
(`/usr/bin/ninja`) because the venv's pip ninja isn't on PATH when the venv
python is invoked directly — the first band-sweep launch failed exactly there.

**Below-gate residual, decomposed** (the A100 session's open finding —
44% below-gate fp16 MAPE with measured constants). Residual analysis of the
H100 gate sweep showed the below-gate error is *structured*: cells with small
footprint sit at 0.87–0.96 measured/predicted, and the blowup rows switch on
exactly where W + act + out crosses ~36–40 MB, plateauing at ~2.2×.

**Footprint-vs-operand experiment** (`profiling/gate/bench_footprint_gate.py`,
commit `aa9fdca`): bf16 bmm, W ∈ {8,16,24,28,32} × T ∈ {1..256}, graph-timed,
clock-locked 1755. **Residency is gated by the launch's total footprint, not
the weight operand**: effective weight BW (split-mem accounting) stays
L2-class until footprint ≈ C_eff then collapses toward HBM rate — W=16 holds
through T=128 (fp 36 MB) and collapses at T=192 (fp 46); W=32 breaks already
at T=24–32 (fp 39–41). Operand gating cannot explain both. Two honest
caveats: the collapse is **soft** (2–3 TB/s across a 40–60 MB band), so a
binary footprint gate recovers only ~2 MAPE points (H100 20.9→20.5, A100
44.0→42.3); and the A100's below-gate residual is dominated by something
else entirely — **a small-kernel bandwidth floor** (~1.2–1.4 TB/s effective
at bs≤4 on BOTH sides of the gate, linear in W, intercept ≈ t0; below even
bw_hbm 1.51). That is kernel saturation, not capacity physics; H100 shows
~8 TB/s on the same cells.

**Correction (guardrail 8)**: session 5's claim that "the bf16 latency curve
breaks at ~31 MB, C_eff visible by eye" on A100 does not survive close
numerical reading — the marginal slope *bumps* at 32–40 MB and settles; the
gate on that card is visible in the relative w8a8 structure, not raw bf16
latency. Paper caption and this journal now say so.

**Paper** (commit `bc4457a`): §Transfer gains the below-gate decomposition;
§Boundary Conditions notes footprint gating as the within-kernel form of the
contention step; caption corrected; limitation bullet updated. Case-study
compression executed (the HANDOFF open item): validation section 260→13
lines (three findings kept), four duplicative tikz bar charts cut with their
tables/prose kept, roofline/e2e/repro folded to prose, Discussion subsections
merged, Limitations kept verbatim. **18 → 15 pages, 0 errors, no undefined
refs** (compiled on this box).

**Mechanism B, prefill leg** (`profiling/served/run_band_prefill.sh`): the
2026-08-01 band demo ran the *decode* workload and closed negative; the
prefill-heavy version (27k-token prompts, chunked prefill,
max_num_batched_tokens ∈ {uncapped, 512, 460, 1024, 896, 2048, 1792}, fp8)
is the unexercised prediction. Honest prior stated before results: prefill
chunks have large CTA counts, so engine-level band effects may wash out here
too.
Result (fp8, total tok/s, repeats where noted): 460 → 11.9k; 512 → 13.1k;
**896 → 13.0k (×2, ±0.4%)**; **1024 → 16.1k (×2, ±0.4%)**; 1792 → 13.6k/16.7k;
uncapped → 13.7k/16.6k; **2048 → engine HANG** (100% GPU spin, frozen at
10/64 prompts, killed after 30 min — vLLM 0.20.2 + hybrid GDN + chunked
prefill at this cap; stack bug, not a band effect). The repeat pass changed
the story and is why it exists: the first pass read as "1024 beats uncapped
by 17%", but uncapped/1792 turned out to have ±10% run-to-run variance and
their better runs sit at 1024's plateau (~16.1–16.7k). What survives
repetition is the **+24% step between adjacent caps 896 → 1024** (both ends
±0.4% across repeats — non-smooth in cap size, the wave-band signature,
mechanism B surfacing at engine level in prefill) and the monotone penalty
for small caps (512/460 at −20/−28% vs plateau). Operational rule, decode
and prefill now consistent: **engine-level batch shaping never gains over
uncapped; its use is avoiding caps that lose** — and the loss holes are
real, reproducible, and sit where kernel-level band data says they should.

**Open after this session**: capture the soft footprint-collapse band in the
model form (a two-parameter transition would likely recover most of the
below-gate MAPE on both architectures); root-cause the A100 small-kernel BW
floor (needs A100 time + NCU); non-NVIDIA backend; B200; GitHub push (still
no key on these boxes).

**Addendum — A100 BW floor root-caused** (probe run on the still-live A100
before shutdown; `profiling/portable/probe_a100_bw_floor.{py,json}`): at M=1,
sm_80 cuBLAS dispatches `cutlass_80_wmma_...16x16` (8 MB) and
`internal::gemvx` (64 MB) — no TMA path. Above the gate a single stream hits
1.53 TB/s ≈ 98% of the 40GB card's HBM spec (no deficit; the "floor" claim
for the HBM regime was wrong and is corrected here); below the gate the
kernels cap at ~0.9–1.3 TB/s invariant to CTA count (H=32→1024) — they
cannot exploit L2-resident weights, where H100's nvjet draws ~8 TB/s.
Pricing the baseline with the same two-point calibration used for W4A16
(below-gate BW 1.28 TB/s from the 8 & 24 MB points) drops A100 below-gate
zero-shot MAPE **44.0% → 12.8%** on the 14 held-out cells. Closes the
below-gate story: not model form, not constants — an unmeasured baseline
kernel term. *Baseline kernels are kernel terms too.* Paper §Transfer +
limitations updated; A100 clocks reset (-rgc) after the probe.

## 2026-08-02 (session 7, H100) — the residuals close

Four items, all landed (commits `dbfd795`+):

**Symmetry check**: two-point baseline calibration on H100 = 6.51 TB/s ≈
harness 6.3 — nvjet carries no hidden kernel term; H100's below-gate
residual lives at T≥64 (23.3% vs 17.6% at T≤32), i.e. the band. Perfect
division with A100 (whose residual was the baseline kernel).

**Soft transition term** (`profiling/gate/fit_footprint_transition.py`):
inverse-BW interpolated in footprint from bw_l2 at C_eff to a fitted floor
at C_hi; fitted ONLY on the 55-cell footprint dataset (C_hi=56 MB, floor
2.10 TB/s; C_eff held at measured 36). Held-out H100 gate sweep: below-gate
**20.9→14.3%** (above pays 11.6→13.3 — floor binds large-footprint
above-gate cells, stated). Transferred to A100 *normalized* (C_hi/C_eff,
floor/bw_hbm) on top of the sm_80 baseline floor: held-out zero-shot fp16
**12.4% below / 13.2% above** — from 28.8/22.8 (estimates) via 44/19
(constants only). The transfer story is now: constants (harness) +
per-kernel two-point terms + one normalized band shape.

**Kernel↔engine bridge for the prefill band step**
(`bench_prefill_band_bridge.py`): the model's own prefill GEMM shapes
(qkv/o/gate_up/down from the 27B config), graph-timed at M∈{768..1152},
show a per-token sawtooth: 1024 is a local minimum (639 ns/tok fp8),
896 costs **+17%**, 832 +25%, 1088 +15% — same shape in bf16. The engine's
+24% step at 896→1024 is mechanism B in the served GEMMs plus chunk-count
overhead. Precise wave/tile attribution left open (nvjet tile configs vary
per M; grid is clustered (2,66)).

**NCU counter corroboration** (`ncu_footprint_target.py`,
`results_ncu_footprint.json`; nsight-compute installed): warm-state
`--cache-control none` DRAM reads at fixed W=24 MB: 0.2 MB → 18.1 → 29.3 →
31.5 per launch as T pushes footprint 30.5 → 38.6 → 53 → 67 MB; W=16 shows
the same ramp. Footprint gating and the soft collapse, in counters.

Paper updated (§Transfer closes both residuals + counter sentence;
§Boundary mechanism B gains the engine bridge): 16 pp, 0 errors.
Remaining open: non-NVIDIA backend, B200, GitHub push, wave/tile
attribution of the sawtooth.

## 2026-08-02 (session 8, B200) — the third architecture point

**Environment**: B200 (sm_100, 183 GB, driver 595.71.05), fresh box, project
rsynced from the H100-region NFS over ssh (regime-router excluded). System
torch 2.8.0+cu128 + triton 3.4.0 for goal 1. **Clock locking is UNAVAILABLE
on this instance** — `nvidia-smi -lgc` is denied even with sudo (virtualized
tier), `-ac` deprecated. Measured mitigation: sustained-load SM clock sits in
a stable 1237–1320 MHz power-limited band (±3%); noted in every results file.
MIG disabled, graph-timed medians of 30 throughout.

**Harness** (`params_nvidia-b200.json`): C_eff **98.8 MB = 0.781× nominal
126.5** — the effective/nominal ratio is now **0.780 / 0.780 / 0.781 on
A100 / H100 / B200**. Across a 3.2× span of nominal LLC and three
generations, effective residency capacity is a constant fraction of nominal;
slide 12's LLC-growth story gets its cleanest data point. bw_l2 13.34, bw_hbm
6.80, peak fp16 achieved 1547 TF (same achieved-vs-quoted pattern as
H100/A100 under power-limited clocks), r_dequant 1.09 TB/s, t0 2.29 µs —
t0 is now 2.30±0.02 µs on three architectures; the graph floor is a
host-software property, not a GPU one.

**Reduced gate sweep** (`results_capacity_gate_b200.json`, T ∈ {1,16,32},
sanity rel_err 3e-4/9e-3, + extension run W ∈ {160..320} MB in
`results_capacity_gate_b200_ext.json`): the below-gate regime expands exactly
as the thesis predicts — the ENTIRE original W grid (8–128 MB) sits at ≤1.3×
the gate on this card; weight sets that are deep HBM territory on H100 are
L2-resident here. Below the gate w8a8 never wins (sp8 0.5–0.9) — the H100
pattern (act-quant overhead vs a fast bf16 baseline at bw_l2 13.3), not the
A100 one, confirming the sign of the below-gate branch is baseline-quality-
dependent as claimed. **Negative result, stated plainly (guardrail 8): the
above-gate w8a8 advantage does not materialize on B200 with these Triton
kernels** — sp8 plateaus at 0.77–0.82 out to 3.2× C_eff where H100 shows
1.19–1.46. The bf16 baseline streams at full HBM rate (above-gate model MAPE
1.5–2.5%); the Triton w8a8 kernel achieves only ~2.6 TB/s effective weight
BW on sm_100 (2.6× off streaming rate). Guardrail-5 confound until the FP4
leg's tuned-CUDA kernels rule: triton 3.4's int8 path on Blackwell, not
architecture physics — do not read this as "quantization stopped working on
B200" before `bench_cuda_moe.py` runs.

**Transfer, third architecture** (`remeasure_sweep_b200.py` — original grid
+ four above-gate sizes {160,192,256,320} MB since C_eff 98.8 leaves the
stock grid gateless; `transfer_validation_b200.py`,
`results_transfer_b200.json`): scored as a variant ladder, held-out,
regime-separated. Constants only: **17.6% below / 1.8% above** — the best
constants-only architecture yet (A100 was 44/19). Then the A100-recipe terms
BOTH FAIL here, and the failures are informative:
- the two-point baseline calibration (8 & 24 MB) reads 4.28 TB/s against
  harness bw_l2 13.34 and blows below-gate MAPE to 66.6% — B200's below-gate
  bf16 latency is jagged from per-shape cuBLAS kernel selection (the 56 MB
  cell costs 2× its neighbors at every T; marginal BW swings 1.1↔60 TB/s
  cell-to-cell). No two-point slope is meaningful on a curve like that; the
  per-kernel-term recipe needs a robustness caveat: it presumes the target's
  baseline latency is locally smooth in W.
- the transferred H100 band applied verbatim wrecks the above-gate branch
  (1.8→41.1%): B200's far field streams at FULL bw_hbm (marginal BW 6.2–7.6
  TB/s from 160→320 MB) — **the fitted persistent floor is an H100-specific
  artifact, answering session 7's open question: a regime-dependent floor is
  not justified as a transferable model element.** The band's EXTENT is real
  and does transfer: marginal BW dips to ~3 TB/s exactly in the transferred
  window (96→128 MB, C_eff..1.56×C_eff).
- keeping the normalized C_hi ratio but ramping to bw_hbm with NO fitted
  floor and NO baseline term — a **zero-parameter band** — gives the final
  B200 numbers: **16.6% below / 3.9% above, fully zero-shot** (n=42/20
  held-out). The transfer story simplifies on this card: constants + one
  normalized band-extent ratio, no per-kernel terms needed.

W4A16 two-point calibration: r_dequant 0.989 TB/s calibrated vs 1.091
harness (9% apart — three-route consistency again), int4 MAPE 30.9%
calibrated vs 42.9% naive-scaled; measured-kernel-terms conclusion survives
unchanged.

**Open after this leg**: goal 2 (native FP4, B200_RUNBOOK) — fresh Blackwell
vLLM env building at session time; the w8a8 CUDA reproduction folds into it.
Model-form lead: the A100/H100/B200 evidence now reads as "one zero-param
band + architecture-specific kernel terms where the target's kernels are
non-smooth" — consider refitting H100 with the floor-free band to see what
its above-gate branch pays.

## 2026-08-02 (session 8 continued, B200) — goal 2: the FP4 prediction confirms

**Environment**: fresh vLLM 0.26.0 venv (`~/vllm-b200-env`, torch
2.11.0+cu130) alongside the system-torch goal-1 stack; the runbook's
prediction that the H100 venv lacks SM100 FP4 ops held. API drift from
0.20.2 handled with import shims (`fused_marlin_moe` moved to
`experts.marlin_moe`; `cutlass_moe_fp4` op renamed `cutlass_fp4_moe_mm`);
`bench_moe_nvfp4_native.py` finalized against `run_cutlass_moe_fp4` +
`scaled_fp4_quant` (recipe mirrored from vllm's online nvfp4 method:
per-expert global scales, block-16 fp8 scales swizzled, neutral activation
gscales, alphas = weight scale_2).

**The headline: the one open falsifiable prediction CONFIRMS.** On the same
box, same Mixtral shape, same graph timing: Marlin W4A16 (dequant→bf16)
crosses under bf16 at **T*≈159 measured** and dies to 0.41× at T=2048;
native-MMA W4A4 NVFP4 **never crosses — 2.11–3.10× vs bf16 across
T=16…2048**. The two FP4 representations differ by **6.3× at T=2048**
(785.6 vs 4944.8 µs): the in-core dequant ceiling, measured, then removed
by hardware. `carm_cuda_fit.py`'s pre-registered `b200 w4a4: none(wins)`
verdict is now an on-device result. r_dequant → ∞ in CARM terms: fitted
dequant ceiling 488.6 TF vs native-FP4 peak 3068.6 TF.

**Refit** (`carm_cuda_params_b200.json`): b200 block flipped to MEASURED
(C_eff/bw_l2 from the goal-1 harness; bw_hbm 6.68, t0 1.80 from this
stack; peak_bf16 1178.7 TF fitted; native_peak_mult["fp4"] = **2.6
measured** vs 4.0 projected — power-limited clocks + real kernel
efficiency, the projection was optimistic). MoE MAPE bf16 21.3% / fp8
16.2%. Crossover model-vs-measured: 194 vs 159.

**Correctness, decomposed honestly**: native output rel-err 0.2212 vs a
no-act-quant bf16 reference — but a software W4A4 emulation (manual E2M1
block-16 qdq on both GEMM inputs) scores 0.2221 vs the same reference. The
22% is the intrinsic activation-quant cost of matched-precision fp4 on
random data, not kernel error. It belongs in any serving recommendation
built on this result.

**Guardrail-3 caveat, stated**: the bf16 fused_experts baseline uses vLLM's
DEFAULT triton config (no tuned E=8,N=14336 B200 config file exists in
0.26.0). The vs-bf16 magnitudes could deflate under a tuned baseline; the
native-vs-Marlin 6.3× comparison shares no bf16 baseline and is the clean
ceiling-break number. This also resolves goal 1's open triton-w8a8
confound in direction: with tuned CUDA kernels, quantization DOES win on
B200 in the streaming regime (1.85× at T=16) — the missing above-gate
advantage in the goal-1 gate sweep was triton 3.4 kernel immaturity on
sm_100, as suspected, not architecture physics.

**Exp B** (`results_flashmla_sparse_b200.json`): fp8-KV loses everywhere
(0.33–0.81×) on Blackwell too; FlashMLA dense is Hopper-only in this build,
sparse ran natively. KV-not-L2-limited conclusion unchanged on a third
architecture.

**Artifacts**: results_cuda_moe_b200.json, results_moe_nvfp4_native_b200.json,
results_flashmla_sparse_b200.json, carm_cuda_params_b200.json,
probe_blackwell_fp4.py, figures/nvfp4_ceiling_break_h100_vs_b200.{png,pdf},
REPORT.md addendum. Env snapshot + sync + paper/deck: next.

## 2026-08-02 (session 9, H100 canonical box) — ratio precision, floor-free refit, clock hygiene

**Environment**: back on the canonical H100 (68.209.75.33), clock locked
`-lgc 1755`. GitHub auth moved to a dedicated SSH key on the NFS root
(`github-robert-ed25519`), wired into all three repos via local
`core.sshCommand` — push/pull now works from any box that mounts the NFS.
kernel-compass's stranded `0c582fa` (b200/b100 GPU_SPECS) pushed.

**Ratio-precision correction (honest statement, guardrail 8)**: the
0.780/0.780/0.781 "three-decimal constancy" was a grid-quantization
artifact. `measure_c_eff`'s sweep grid is nominal-relative (0.4–1.5×, 14
points), so the candidate C_eff/nominal ratios are the SAME rationals on
every card; all three GPUs broke between grid indices 4 and 5, whose
midpoint is 0.78077× nominal on any card — agreement to three decimals was
guaranteed by construction, real resolution ±4.2% of nominal (half step).
Fine-grid re-sweep on H100 (`cliff_finegrain.py`, 0.5 MB steps, 3 repeats,
`results_cliff_finegrain_nvidia-h100-80gb-hbm3.json`): collapse onset
**39.8±0.5 MB = 0.795±0.010× nominal**, rolloff completing over ~3.5 MB
(soft, mirrors the footprint-band picture). The honest cross-architecture
claim: a common ≈0.8× fraction (same grid cell across a 3.2× span), not a
three-decimal constant. Paper (4 sites), deck slide 12, WeChat item 14
corrected; WeChat item 15 states the correction explicitly.

**Floor-free band refit on H100** (`refit_floorfree_band.py`,
`results_floorfree_band_h100.json`) — the other half of session 7's open
question, session-7 numbers reproduced exactly: binary gate 20.9/11.6,
fitted-floor band 14.3/13.3 (below/above). Floor-free B200 form (C_hi=1.56×,
ramp to bw_hbm): **17.3/13.4**; floor-free with C_hi refit on the footprint
set: C_hi=1.22× C_eff, **16.8/12.1**. Conclusion: the floor is worth ~3
points below-gate on H100 and nothing above-gate — H100's far field
genuinely streams below HBM rate, so the floor is a REAL local kernel term,
but (per B200) not a transferable element. Model form settles as: zero-param
band transfers; per-architecture floors are local add-ons. Paper
§limitations updated with the numbers.

**sm_clock metadata fix**: five bench scripts (gate ×3, portable sweeps ×2)
recorded a startup idle nvidia-smi read — the B200 files' 705/750 MHz was
the DVFS floor, not the run clock. All five now use `sm_clock_loaded()`
(samples during a saturating GEMM loop); the three affected B200 JSONs got
an `sm_clock_note` annotation (measured data untouched). Found in passing:
even with `-lgc 1755`, saturating compute pulls this H100 to ~1380–1395 MHz
(power-limited) — the lock is a cap, not a floor; "clock-locked" claims
bound compute-heavy cells from above only. Memory-bound cells (the cliff
sweeps) draw less power and sit at the lock.

**B200 w8a8 prep** (handoff item 4, box gone):
`dense_qwen/bench_l2_boundary.py` parameterized (--c-eff-mb, --t0-us,
--targets-mb; defaults preserve H100 behavior; compile-checked, needs vLLM
env to run) — it is the tuned-CUDA w8a8 instrument (cutlass_scaled_mm).
B200_RUNBOOK §6 added with the exact invocation and readout criteria.

**Open**: FlagOS/non-NVIDIA leg; wave/tile sawtooth attribution; fine-grid
cliff re-sweeps on the NEXT A100/B200 (tighten their ±4% ratios the same
way); venue formatting pass.

## 2026-08-03 (session 10, H100) — the gate as a kernel-design tool

**The kernel-opportunity claim confirmed by construction**
(`explorations/state_residency/gdn_l2_kernel.py`): a ~100-line Triton
gated-delta-rule decode kernel (one fused pass, traffic exactly 2x state,
correctness 1e-8) runs **2.2x faster than fla's fused_recurrent kernel at
24 MB state** (11.43 vs 25.12 us), 1.5-1.9x across the below-gate range,
collapsing to ~1.1x above — the speedup window closes at B=40-48, exactly
the pre-registered B* = C_eff/(H x 64 KB) ~= 40. Warm below-gate 3.4-4.4
TB/s (70% of L2 tier) vs fla's flat L2-blind 2.3; far field 2.52 (80% of
HBM). The capacity gate predicted where the headroom lived; the kernel
captured ~80% of it on first config search.

**Two-capacities question: counter-corroborated**
(`explorations/ceff_reconcile/ncu_target.py`, nsight-compute now on this
box): warm-state DRAM reads put the GEMM residency transition at ~34 +/- 2
MB vs re-read ~40 +/- 2 — the ~6 MB GEMM-context gap is real. Model
consequence: C_eff is operand-context-dependent; carry both constants.
Texture noted: GEMM keeps a 37%-hit tail at 44 MB (tiling reuse).

Both threads pre-registered in the autoloop report before measurement.
Next: epilogue-complete kernel (short-conv + gating), chunked-prefill
variant, upstream conversation with fla; NCU on the GDN kernels; fold
into paper as the "gate as design tool" section.

## 2026-08-03 (session 10 continued) — epilogue-complete kernel, upstream draft, paper section

**Epilogue-complete kernel** (`gdn_l2_kernel_full.py`): the FULL decode
step (short conv K=4 + silu with rolling cache, qk l2norm, delta rule,
gated RMSNorm) fused into one program per (batch,head), vs fla's real
3+-kernel chain: **2.00-2.34x below the gate, 1.21-1.30x above** — the
decomposition is fusion (~1.25x everywhere) x residency (~1.9x in the
window), crossover still at 40-48 MB = C_eff. Correctness 1e-8/exact.
Chunked-prefill variant analyzed and deliberately NOT built (matmul-form
chunking owns prefill via tensor cores; the residency claim is decode).

**Upstream**: ready-to-post fla issue draft at
`explorations/state_residency/UPSTREAM_fla_ISSUE_DRAFT.md` (hardware-
factual framing, repro pointers, PR offer). Posting needs Robert's GitHub
review/auth — no gh on this box, and it goes out under his name.

**Paper**: new §"The Gate as a Design Tool" (before §Attention Kernel
Validation): L2-blind production kernel, the 2.7x window, the fused
kernel capturing it, pre-registration noted, two-capacities finding
(C_eff^re-read vs C_eff^GEMM, counter-corroborated). Limitations scoped
(one GPU, fixed geometry, decode-only, harness measures re-read capacity
only). 17 pp, 0 errors.

## 2026-08-03 (session 11, B300 SXM6 AC via ssh) — the fourth architecture point

Robert provided a B300 instance. Headlines, all data in repo:

- **Nominal L2 = 126.5 MB, IDENTICAL to B200 (148 SMs too).** The
  192 MB secondary-source figure is contradicted on-device; the paper's
  softened wording (session 9) aged well.
- **P1 FALSIFIED, and instructively**: fine-grid C_eff **91.6±1.0 MB =
  0.724±0.008x** (5 reps), outside the pre-registered 0.77-0.82 band —
  while the coarse harness reads 0.781 (same grid cell as all prior
  cards), the grid-quantization critique made flesh. The cross-arch claim
  is now "large but architecture-varying 0.72-0.80 fraction" (paper,
  deck, prereg OUTCOMES all updated).
- **P2 mechanism confirmed**: t0 1.52 us on torch 2.13 (2.30 -> 1.80 ->
  1.52 across stacks, four silicon generations): host-software property.
- **P5 early direction: the ratio GROWS** — bw_l2 16.5 TB/s (+24% vs
  B200) at flat bw_hbm 6.7 -> L2:HBM 2.46 (B200: 1.96). r_dequant 1.13.
- **Design-tool cross-validation**: our GDN kernel's speedup window moved
  40 MB -> ~92 MB tracking C_eff exactly; 2.05x vs fla at 56 MB; fla
  still residency-blind (<=1.11) despite running 2x faster than on H100.
- Gate-sweep w8a8 leg blocked: triton 3.7 int8 tl.dot API break on the
  3.3-era kernel (compile error logged) — port needed, honest framing:
  compatibility break, not measured immaturity.
- Box state: /root/bench-env (torch 2.13+cu130 + triton 3.7 + fla),
  /root/bench/{profiling,explorations}; clock lock denied; results
  copied back. Remaining if box persists: kv_hotset byte-governance,
  vLLM FP4 leg (P6), w8a8 kernel port.

Paper: LLC-trend passage + abstract + design-tool section updated with
measured B300 numbers; 17 pp, 0 errors. Deck slide 12: fourth bar
126.5/⌁91.6 with corrected caption.

## 2026-08-03 (session 11 continued, both GPUs + agent fleet)

- **triton 3.7 int8 break RESOLVED**: `out_dtype=tl.int32` on the int8
  tl.dot (w8a8_bmm.py:72, now in repo, works on 3.3+). B300 gate sweep
  ran clean: bf16 MAPE 28.9/3.5 (below/above) — sm_103 keeps the jagged
  below-gate kernel selection; **triton w8a8 STILL never wins on sm_103**
  (sp8 <= 0.95 out to 1.4x C_eff, current-version toolchain) — third
  Blackwell-family point for the kernel-maturity story (far field beyond
  128 MB unmeasured on this grid; stated).
- **NCU on the GDN kernels (H100)**: ours 86% L2-hit below gate, exact 2x
  streaming above (129.5 vs 128 predicted); fla misses 56% of state
  traffic to DRAM even when resident — cache-unfriendly access, not just
  latency-bound (FINDINGS addendum 5).
- Docs-sync agent updated deck status slide, WeChat items 16/17, and the
  upstream fla draft (now two-architecture). Its audit caught the journal
  path bug (entry recovered) and the unsynced w8a8 patch (now landed).
- B300 FP4 leg (P6) running in vllm-env 0.26; venue/related-work agent
  pending.

**Venue/prior-art agent returned** (explorations/venue_relatedwork.md):
prior art CLEAR on the capacity-gated residency-window claim, with two
must-engage neighbors now cited in paper + fla draft: ReplaySSM (Dao,
6/2026 — halves state traffic algorithmically, cache-blind; orthogonal
and COMPOSABLE, footprint unchanged so the window stands) and the FPGA
persistent-state GDN accelerator (2603.05931) whose "L2 cannot persist
state" premise our 86%-warm-hit counters refute. Target venue: MLSys
2027 (deadline Oct 30, 2026), backup ISPASS; ATC discontinued. 12
related-work items + bib delta in the memo.

**Session 11 close-out (box released)**: FP4 leg completed and copied —
ceiling-break reproduces on sm_103 (native 2.66-2.95x, Marlin 0.40x at
T=2048, rel-err 0.2212 == B200); P6 likely falsified at ~2.7x vs band
3.4-5.0 (this SXM6 AC = B200 silicon + 24% L2 BW; proper carm fit
pending). All artifacts local: params, fine-grid cliff, gate sweep, GDN
kernel pair, MoE trio. B300 box state documented; safe to terminate.
