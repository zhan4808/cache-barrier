# PREREG — B300 (Blackwell Ultra) and Rubin/R100 predictions

**Status: PRE-REGISTERED, 2026-08-02.** Written before any B300/GB300 or
Rubin device access. Nothing below is a measurement of those chips. Every
number is tagged **[measured]** (this project's on-device data, committed
JSONs), **[datasheet]** (vendor-published, never treated as achieved), or
**[press]** (secondary reporting, lowest confidence). This document follows
the FP4 precedent: the `b200 w4a4: none(wins)` verdict was pre-registered in
`carm_cuda_fit.py` and confirmed on-device 2026-08-02 (session 8). These are
the next falsifiable claims in that series.

**Convention**: MB = 10^6 bytes throughout (matches the harness JSONs). The
MiB/MB gap (4.9%) does not flip any verdict below except cells explicitly
flagged "marginal".

**Confirmation protocol** (what counts, in order, on first device access):
1. `measure_params.py` (portable harness) — nominal L2 from the device query,
   C_eff, bw_l2, bw_hbm, peak, r_dequant, t0. Record the loaded SM clock
   (`sm_clock_loaded()`, session-9 fix) and whether clock lock is available.
2. `cliff_finegrain.py` at 0.5 MB steps around the predicted onset (the
   coarse grid's ±0.042×-nominal resolution cannot arbitrate P1's fine band —
   session 9 lesson).
3. Reduced gate sweep + `remeasure_sweep` + `transfer_validation` variant
   ladder, zero-shot first, exactly as on B200 (session 8).
4. `bench_moe_nvfp4_native.py` for P6.

---

## 0. Input specs as publicly known, August 2026

### B300 / GB300 (Blackwell Ultra) — shipping since late 2025

| Quantity | Value | Tag | Notes |
|---|---|---|---|
| SM count | 160 (2 dies × 80) | [datasheet] | NVIDIA architecture blog; B200 ships 148 |
| HBM3e capacity | 288 GB | [datasheet] | |
| HBM3e bandwidth | 8 TB/s | [datasheet] | vs B200 datasheet 8 TB/s-class; our B200 **measured** bw_hbm is 6.80 TB/s — expect a similar achieved/quoted gap |
| Dense NVFP4 | 15 PF (NVIDIA blog); 14 PF for HGX B300 | [datasheet] | vs 9–10 PF dense on B200; the FP4 pipes gained ~1.5× per-SM |
| Dense FP8 | ~4.5–5 PF | [datasheet] | scales ~with SM count only; no per-SM FP8 gain |
| Dense BF16 | ~2.25 PF | [datasheet] | unchanged per-SM vs B200 |
| TDP | up to 1,400 W | [datasheet] | vs 1,000 W B200 |
| TMEM | 256 KB/SM = 40 MB aggregate | [datasheet] | not L2; some press confuses these |
| **Nominal L2** | **CONTESTED: 192 MB vs ~128 MB** | [press] | see below |

**The L2 dispute (this project currently assumes ~192 MB — that assumption
is NOT confirmed by primary sources).** Several secondary/database sites
(Spheron, slyd, server-parts.eu) state 192 MB L2 for B300 vs 126–128 MB on
B200. But: (a) NVIDIA's own "Inside Blackwell Ultra" architecture blog gives
SMs, TMEM, HBM, and FLOPS and is *silent on L2*; (b) the transistor count
reported for Blackwell Ultra (208 B) equals B200's, which is hard to square
with +50% L2 SRAM; (c) the most careful technical page we found
(glennklockwood.com/garden/processors/b300) lists no L2 either, while noting
B300 *is* a new tape-out (so a changed L2 is not impossible). We could not
find a primary NVIDIA document stating 192 MB. **Resolution is step 1 of the
protocol**: the harness reads nominal L2 from the device query (B200
reported 126.5 MB this way). P1 is therefore stated as a ratio claim with
two nominal branches, and P4 carries both branches.

### Rubin / R100 — GTC 2026 numbers, cloud availability expected H2 2026

| Quantity | Value | Tag | Notes |
|---|---|---|---|
| HBM4 capacity | 288 GB | [press] | |
| HBM4 bandwidth | "up to 22 TB/s" (GTC 2026); earlier reporting 13 TB/s | [press] | treat as a 13–22 TB/s claim band; final SKU unknown |
| FP4 inference | "50 PF" | [press] | almost certainly a marketing number (sparse and/or per-package); do not use as a dense per-GPU peak |
| L2 size / bandwidth | unknown | — | no public claim found; P5 is written precisely because of this gap |
| Process | TSMC 3nm-class, dual die, ~336 B transistors | [press] | |

Sources: NVIDIA developer blog "Inside NVIDIA Blackwell Ultra"; NVIDIA DGX
B300 page; Tom's Hardware GTC/Hot Chips 2025 coverage; glennklockwood.com
B300 page; Thunder Compute / Spheron / Introl Rubin roundups (Aug 2026).
Vendor and press numbers above are inputs to predictions, never evidence.

---

## Measured baseline the predictions extrapolate from

All **[measured]**, from committed JSONs
(`profiling/portable/params_*.json`, fine-grid cliff, gate/transfer files):

| Quantity | A100-40GB | H100-80GB | B200 |
|---|---|---|---|
| nominal L2 (device) | 40.0 MB | 50.0 MB | 126.5 MB |
| C_eff | 31.2 | 39.0 (fine-grid onset 39.8±0.5) | 98.8 |
| C_eff / nominal | 0.78 (±0.042 grid) | **0.795±0.010 fine-grid** | 0.781 (±0.042 grid) |
| bw_l2 (TB/s) | 3.855 | 5.262 | 13.34 |
| bw_hbm (TB/s) | 1.514 | 3.264 | 6.799 |
| bw_l2 / bw_hbm | 2.55 | 1.61 | 1.96 |
| t0_graph (µs) | 2.304 | 2.326 | 2.285 |
| peak fp16 achieved (TF) | 257.7 | 736.8 | 1547.1 |

Session-9 honesty note carried forward: the coarse-grid ratios are
grid-quantized (all three cards break in the same nominal-relative cell,
midpoint 0.781, half-step ±0.042); only the H100 fine-grid 0.795±0.010 is a
resolved ratio. The honest cross-architecture claim is "a common ≈0.8×
fraction", not a three-decimal constant.

Transfer precedents **[measured]**: B200 zero-shot with constants + the
zero-parameter band (C_hi = 1.56×C_eff, inverse-BW ramp to bw_hbm, no
floor): **16.6% MAPE below gate / 3.9% above** (n=42/20 held-out). A100
zero-shot needed one local kernel term (the sm_80 below-gate baseline floor):
44.0/19.0 constants-only → **12.4/13.2** with it. H100 in-architecture with
the floor-free band: 17.3/13.4 (its fitted floor is real but local, worth ~3
points below-gate; B200 refuted it as a transferable element).

---

## P1 — B300 effective LLC capacity

**Statement.** The timing-only residency cliff on B300 will sit at the same
≈0.8× fraction of nominal L2 that A100/H100/B200 show. Coarse-grid: the
break lands in the same nominal-relative grid cell, i.e. C_eff/nominal ∈
**[0.74, 0.82]**. Fine-grid point estimate: **0.795× nominal** (H100
fine-grid value; the only resolved measurement of the fraction).

**Point estimate + interval.**
- Branch A (nominal = 192 MB, if the press claim is right): C_eff ≈
  **153 MB**, range 142–157 MB.
- Branch B (nominal ≈ 126.5–128 MB, B200-class silicon): C_eff ≈
  **101 MB**, range 95–105 MB.
The ratio claim is the prediction; the nominal branch is resolved by the
device query on day one, before any sweep runs.

**Extrapolates from.** `params_*.json` ×3 (0.78/0.78/0.78 coarse),
`results_cliff_finegrain_nvidia-h100-80gb-hbm3.json` (0.795±0.010), spanning
a 3.2× range of nominal LLC and three generations.

**Falsified if.** Fine-grid onset ratio outside [0.77, 0.82] (equivalently,
coarse break outside the shared grid cell). Secondary falsifier: the rolloff
completing in <1 MB or >8 MB would break the "soft ~3.5 MB collapse"
picture even if the midpoint ratio survives.

**Either way we learn.** Confirmed: the ≈0.8 fraction is a
policy/architecture-family invariant across four chips and 5–6× nominal
span — strong enough to state as a design rule (effective residency capacity
≈ 0.8× nominal, plan capacity gates accordingly). Falsified: the fraction is
not an invariant but a coincidence of GA100/GH100/GB100 cache policy, and
the model must carry C_eff as a per-chip measured constant with no shortcut —
which the harness already supports, so the cost of being wrong is one sweep.

---

## P2 — t0 graph-launch floor

**Statement.** The graph-mode launch/timing floor t0 is a host-software
property, not a GPU property. On B300 (and on Rubin), the same harness on a
torch 2.7/2.8-era stack will measure t0_graph ≈ **2.30 µs**, interval
**±0.15 µs**.

**Extrapolates from.** t0_graph = 2.304 / 2.326 / 2.285 µs on A100 / H100 /
B200 **[measured]** — 2.30±0.02 µs across three architectures, two vendors'
worth of generational change in everything else. Supporting evidence that it
tracks software, not silicon: (a) t0_eager varies 14.9–32.8 µs across
*hosts* (session 5: "floors are host properties"); (b) the B200 goal-2 stack
(torch 2.11/cu130) measured t0 = 1.80 µs *on the same GPU* that measured
2.285 µs under torch 2.8 — the stack moved it, the GPU didn't.

**Falsification criterion.** On a matched stack, t0_graph outside 2.30±0.3
µs; or, more diagnostically, t0 found to correlate with GPU generation
(e.g., monotone in SM count or die count) rather than with driver/torch
version when both are varied.

**Either way we learn.** Confirmed: the dispatch cost model's fixed term is
portable and needs no per-GPU calibration — only per-stack. Falsified:
Blackwell-Ultra/Rubin changed the launch path in hardware (e.g., grid-launch
offload), which would itself be a finding worth a section, and t0 joins the
per-architecture measured constants.

---

## P3 — zero-shot transfer quality (constants + zero-parameter band)

**Statement.** On B300, the split-mem CARM form with (i) harness-measured
constants and (ii) the zero-parameter band (C_hi/C_eff = 1.56, inverse-BW
ramp from bw_l2 at C_eff to bw_hbm at C_hi, **no floor, nothing fitted on
B300**) will score, on a held-out gate sweep, regime-separated MAPE:

- below gate: **point 17%, interval 10–25%**
- above gate: **point 5%, interval 2–14%**

**Extrapolates from.** The precedent ladder **[measured]**: B200 fully
zero-shot 16.6/3.9; H100 floor-free 17.3/13.4; A100 12.4/13.2 (with its one
local term). B300 is the same architecture family as B200 with the same
cuBLAS generation, so the B200 numbers are the closest prior; the above-gate
interval's upper end covers the possibility that B300's far field streams
below full HBM rate the way H100's does.

**Conditions under which per-kernel local terms will be needed** (stated
now so their use later is not a silent retreat):
1. *Non-smooth kernel selection.* If B300's below-gate bf16 latency is
   jagged in W the way B200's is (the 56 MB cell costing 2× its neighbors;
   marginal BW swinging 1.1↔60 TB/s cell-to-cell), then two-point baseline
   calibration is invalid *and unnecessary* — we predict it stays invalid on
   B300 (same cuBLAS/nvjet family) and that constants+band beat any
   two-point-calibrated variant, as on B200.
2. *Local floors.* If the far field streams measurably below bw_hbm (H100
   pattern), a fitted local floor is admissible as a per-architecture add-on
   only, per session 9. We predict B300 does NOT need one (B200's far field
   streamed at full rate, 1.5–2.5% above-gate MAPE).

**Falsified if.** Zero-shot below-gate MAPE > 35% or above-gate > 20% —
i.e., worse than the worst constants-only precedent after the band is
applied — or if a fitted floor is *required* to get above-gate under 15%.

**Either way we learn.** Confirmed: "constants + one normalized band, local
terms only where kernels are non-smooth" survives a fourth architecture and
becomes the paper's transfer recipe with n=4. Falsified: the band's
1.56 ratio is Hopper/Blackwell-specific, and the recipe honestly demotes to
"constants transfer; the transition shape must be refit per family" — a
weaker but still zero-GPU-hours-per-kernel claim.

---

## P4 — gate expansion: which LLM weight operands drop below the gate on B300

**Setup.** Operand-level gate at C_eff. Branch A: C_eff ≈ 150 MB (nominal
192). Branch B: C_eff ≈ 101 MB (nominal 128; then B300 ≈ B200 and only the
B200 column below moves by rounding). Sizes are decimal MB; per-GEMM weight
operands; bf16/fp8/fp4 = 2/1/0.5 B per element. Footprint caveat carried
from session 6: residency is gated by *total launch footprint*, so at large
T these thresholds shift left by the activation+output footprint; verdicts
below are the T→small (decode-shaped) reading the gate figure uses.

**The arithmetic** (largest per-layer operands; 27B row uses the repo's
Qwen3.6-27B shapes from `bench_prefill_band_bridge.py`):

| Operand (K×N) | params | bf16 MB | fp8 MB | fp4 MB |
|---|---|---|---|---|
| 7B qkv (4096×12288) | 50.3 M | 100.7 | 50.3 | 25.2 |
| 7B gate_up (4096×22016) | 90.2 M | 180.4 | 90.2 | 45.1 |
| 7B down (11008×4096) | 45.1 M | 90.2 | 45.1 | 22.5 |
| 13B gate_up (5120×27648) | 141.6 M | 283.1 | 141.6 | 70.8 |
| 27B qkv (5120×8192) | 41.9 M | 83.9 | 41.9 | 21.0 |
| 27B gate_up (5120×34816) | 178.3 M | 356.5 | 178.3 | 89.1 |
| 27B down (17408×5120) | 89.1 M | 178.3 | 89.1 | 44.6 |
| 70B qkv (8192×10240) | 83.9 M | 167.8 | 83.9 | 41.9 |
| 70B gate_up (8192×57344) | 469.8 M | 939.5 | 469.8 | 234.9 |
| 70B down (28672×8192) | 234.9 M | 469.8 | 234.9 | 117.4 |
| Mixtral expert w1/w3 (4096×14336) | 58.7 M | 117.4 | 58.7 | 29.4 |
| Mixtral expert fused w1w3 (4096×28672) | 117.4 M | 234.9 | 117.4 | 58.7 |
| Mixtral per-expert total (3 mats) | 176.2 M | 352.3 | 176.2 | 88.1 |
| DeepSeek-class expert mat (7168×2048) | 14.7 M | 29.4 | 14.7 | 7.3 |
| DeepSeek-class per-expert total | 44.0 M | 88.1 | 44.0 | 22.0 |

**Predicted gate verdicts, Branch A (C_eff ≈ 150 MB), vs B200 (98.8):**

Newly below the gate on B300 (the headline crossings):
1. **Mixtral-class per-expert operands at bf16** (117.4 MB < 150; was
   *above* on B200's 98.8). Unfused w1/w3/w2 were already below on B200
   (58.7); the fused gate_up operand crosses on B300. Consequence, stated
   as the falsifiable dispatch claim: *quantizing Mixtral-class expert
   weights loses its residency rationale on B300* — per the gate, fp8/fp4
   expert weights buy bandwidth only where the bf16 operand already
   misses residency, and on B300 it doesn't. Expected observable: the
   w8a8/fp8 speedup on per-expert GEMMs at decode T collapses toward the
   below-gate pattern (≤1× against a good bf16 baseline, B200 session-8
   sign).
2. **13B gate_up at fp8** (141.6 < 150, *marginal* — inside the MiB/MB and
   C_eff-interval blur; scored as a soft prediction).
3. **7B everything at every precision** (largest operand 180.4 bf16 is the
   only 7B holdout above B200's gate; it stays above 150 — so 7B is
   "all-resident except bf16 gate_up" on both branches).

Still above the gate at every precision including fp4:
4. **70B gate_up** (fp4 234.9 > 150): above the gate at every precision.
   For 70B-class dense, quantization remains a pure bandwidth/far-field
   play on B300; the gate predicts its speedup structure stays B200-like
   there. (70B down at fp4, 117.4 MB, is a Branch-A crossing — marginal
   under footprint pressure.)
5. **27B gate_up: fp4-only crossing** (bf16 356.5 and fp8 178.3 above;
   fp4 89.1 below). Crisp, checkable dispatch prediction: on B300 the
   largest 27B-class operand achieves residency *only* at 4-bit — the
   capacity gate and the native-FP4 story intersect on exactly this shape.

Fine-grained MoE (DeepSeek-class, expert ffn ≈ 2048): per-expert operands
(14.7 MB) and even whole experts (44 MB total at bf16… 88.1 bf16) sit below
the gate at all precisions on B300 *and already on H100/B200*. The gate
predicts quantization of fine-grained expert weights buys capacity/HBM
footprint, not per-GEMM latency — same verdict as KV (session 3): inside
the budget, outside the gate. (Grouped-GEMM execution that streams many
experts per launch re-enters via the footprint term: 9 activated experts
× 44 MB bf16 ≈ 396 MB total is far above C_eff, so the *batched* MoE launch
is above-gate even when each expert alone is below — the operand-vs-
footprint distinction, which the B300 sweep can test directly.)

**Branch B (C_eff ≈ 101 MB):** B300 ≈ B200 within grid noise; crossings 1–2
do not occur (117.4 and 141.6 both > 101); the 27B fp4-only crossing (89.1 <
101) survives — it is the branch-independent prediction.

**Falsified if.** The measured B300 gate sweep places the sign
flip/advantage-peak structure away from the predicted C_eff band (that's
P1's failure), or — the P4-specific claim — operands in rows 1/5 don't
change regime relative to B200 as tabulated (e.g., fused 117.4 MB Mixtral
operands still behave above-gate on B300 despite C_eff ≈ 150).

**Either way we learn.** Confirmed: the gate arithmetic becomes a
forward-planning tool — you can read a model config and a datasheet and
predict which precisions pay *before the chip exists*, which is the paper's
strongest applied claim. Falsified: the operand/footprint accounting is
missing a term at large capacity (e.g., partitioned-L2 locality across two
dies), and the model gains a die-topology term — B200 never isolated this
because its C_eff still sat below most interesting operands.

---

## P5 — Rubin L2:HBM bandwidth ratio (the cache barrier's survival)

**Statement.** The gate's *latency* value rests on the measured bandwidth
gap bw_l2/bw_hbm — 1.61 on H100, 1.96 on B200, call it the ~1.6–2.0 band
(A100's 2.55 reflects its cut-down HBM2 part). For the barrier to retain
its H100/B200-era value on Rubin:

- if HBM4 ships at ~13 TB/s achieved: bw_l2 must measure **21–26 TB/s**;
- if HBM4 ships at ~22 TB/s achieved: bw_l2 must measure **35–44 TB/s**.

Measured bw_l2 history is 3.86 → 5.26 → 13.34 TB/s (×1.36, ×2.54 per
generation). Reaching 35+ TB/s requires another ≥2.6× L2-bandwidth jump in
one generation *on top of* whatever fraction of the HBM4 claim is achieved.
**Point prediction: the ratio compresses.** If achieved HBM4 lands near the
low claim (13-ish), predict ratio **1.4–1.8** (barrier retains most value);
if achieved HBM4 lands near 22, predict ratio **1.0–1.5** (barrier
substantially compressed). No public Rubin L2 spec exists to anchor this —
that absence is why the prediction is stated as a conditional.

**The falsifiable claim.** Measure bw_l2 and bw_hbm with the harness on
first Rubin access. If bw_l2/bw_hbm < **~1.3**, the capacity gate's latency
value collapses even as C_eff grows: below-gate residency then buys ≤30%
bandwidth over streaming, the below-gate/above-gate latency contrast that
defines the gate shrinks toward measurement noise, and the "quantization
doesn't pay below the gate" branch weakens correspondingly (the *capacity*
gate survives as a footprint/HBM-budget statement — P4's fine-grained-MoE
logic — but its per-GEMM latency arithmetic does not). If the ratio holds
≥1.6, the whole H100/B200-era model transfers with constants only, per P3.

**Extrapolates from.** params_*.json bandwidth pairs ×3 [measured]; HBM4
13–22 TB/s claim band [press]; the achieved/quoted precedent (B200: 6.80
measured vs 8 quoted = 0.85; H100: 3.26 vs ~3.35 = 0.97).

**Either way we learn.** Ratio holds: NVIDIA scaled the L2/crossbar with
HBM4, and the cache barrier is a durable design axis, not a two-generation
artifact. Ratio compresses: the honest headline becomes "HBM4 closed the
cache barrier" — which *inverts* the dispatch advice (quantize for
capacity/energy, stop special-casing residency), and the model's value
migrates to its footprint/budget terms. Both outcomes are publishable;
pre-registering the threshold (1.3) is what makes the second one a finding
rather than a post-hoc story.

---

## P6 — native-FP4 continuation: B300 native_peak_mult

**Statement.** On B300, `carm_cuda_fit.py`'s native_peak_mult["fp4"]
(measured native-FP4 MoE peak ÷ measured/fitted bf16 peak, same box, same
timing) will land at **point 4.2, interval 3.4–5.0** — up from B200's
measured 2.6, but well short of the datasheet ratio.

**Reasoning and data.**
- B200 [measured]: projected mult 4.0 (datasheet dense FP4:BF16 = 9:2.25);
  measured 2.6 (native peak 3068.6 TF vs fitted peak_bf16 1178.7 TF).
  Realization factor of the datasheet *ratio*: 2.6/4.0 = **0.65**, under
  power-limited clocks (sustained 1237–1320 MHz vs 1965 boost) with real
  kernel efficiency.
- B300 [datasheet]: dense FP4 15 PF vs BF16 ~2.25 PF → datasheet ratio
  **6.67** (the Ultra tensor cores gained ~1.5× FP4 per SM; FP8/BF16 gained
  only the 148→160 SM count).
- Power reasoning: the FP4 pipes grew 1.67× against a 1.4× TDP increase,
  part of which feeds 288 GB of HBM3e; dense-FP4 load is exactly the
  highest-power-density regime, so the DVFS haircut on the fp4 *numerator*
  should be at least B200's. Applying the B200 realization factor to the
  ratio: 6.67 × 0.65 ≈ **4.3**; the interval's low end (3.4) allows a
  worse haircut (realization 0.5), the high end (5.0) a mildly better
  kernel (cutlass fp4 MoE has matured since the 0.26.0 measurements).
- Both branches of P1 leave this prediction unchanged (mult is
  capacity-independent).

**Falsified if.** Measured mult < **3.0** (no meaningful gain over B200's
2.6 despite 1.67× datasheet FP4 — would mean the Ultra FP4 pipes are
power-starved into irrelevance or the kernels can't feed them) or > **5.5**
(would mean the B200 measurement, not the datasheet, was the outlier —
prompting a re-run of the B200 leg under a tuned baseline, per the session-8
guardrail-3 caveat). Companion replication claim, carried from session 8:
native W4A4 *never crosses under* bf16 at any T on B300 (Marlin-style
dequant still dies at large T; the 6.3×-class native-vs-dequant gap at
T=2048 persists or widens).

**Either way we learn.** Confirmed: "datasheet ratio × ~0.65 power-limited
realization" becomes a two-point rule for projecting native-precision
ceilings, and the CARM far field for future chips can be written from a
datasheet with stated error bars. Falsified low: power, not arithmetic
width, is now the binding resource for low-precision throughput — a
different paper's thesis, and CARM's peak term needs an explicit power
model. Falsified high: our B200 fp4 peak was kernel-limited and the
measured-vs-projected framing needs a kernel-maturity axis.

---

## Scoring

Each prediction scores on first device access, using the protocol at top,
in a session journaled like sessions 5/8 (A100/B200 legs): zero-shot
numbers first, variant ladder second, failures stated plainly (guardrail
8). No prediction may be revised after the first sweep starts; a wrong
branch of P1/P4 is scored wrong on the branch actually realized, and the
other branch is void (not "confirmed").

**Spec-belief correction logged now, before measurement**: the project's
working assumption of a ~192 MB B300 nominal LLC is press-sourced only; we
could not confirm it in any primary NVIDIA material, and the equal
transistor count argues against it. If the device query reports ~128 MB,
that is not a falsification of anything in this document — P1/P4 branch on
it — but it *is* a correction to DIRECTION-level framing that assumed the
LLC-growth trend (40→50→126→192) continues monotonically at the Ultra
mid-generation step.

Sources (accessed 2026-08-02): NVIDIA developer blog "Inside NVIDIA
Blackwell Ultra"; nvidia.com DGX B300; tomshardware.com Blackwell Ultra
announce + Hot Chips 2025 detail; glennklockwood.com/garden/processors/b300;
videocardz.com GB300 spec leak; spheron.network, server-parts.eu, slyd.com
(192 MB L2 claims, secondary); thundercompute.com and introl.com Rubin
roundups (HBM4 22 TB/s GTC-2026 claim; earlier 13 TB/s reporting);
tech-insider.org GTC 2026 Rubin analysis.
