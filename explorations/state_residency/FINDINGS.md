# D1 — recurrent-state residency and the capacity gate (2026-08-02/03, H100)

**Question**: hybrid linear-attention models (Qwen 3.5 / Nemotron 3 / Mamba-3:
~75% linear layers + MoE) replace the KV cache with a constant per-request
state re-read+rewritten every decode step. Does the capacity gate govern this
new operand class?

**Setup**: `bench_state_residency.py` — batched Gated-DeltaNet-style step
(`S <- 0.99 S + k v^T` via one fused baddbmm, readout `q^T S`), dk=dv=128
fp16 (32 KB/head), graph-timed, sweep total state footprint 8–96 MB, warm
(state re-read each step) vs rotated-copies control (forced far field), plus
a contended mode (24 MB warm state + X MB interleaved streaming read).
Clock: locked cap 1755, ~1380–1485 under this load, sampled and recorded.

## Findings

1. **The gate location transfers to recurrent state.** Warm/rotated
   advantage is 1.15–1.28x below the gate and collapses to 1.00 at
   **38–40 MB — exactly the fine-grid C_eff onset (39.8 ± 0.5 MB)** from
   `results_cliff_finegrain_nvidia-h100-80gb-hbm3.json`. The capacity
   structure shows up on an operand class that did not exist in the
   original study.

2. **Magnitude is kernel-limited (a per-kernel term, as the model form
   predicts).** Absolute warm BW is ~1.8–2.1 TB/s — far below bw_l2
   (5.3–6.3): cuBLAS batched rank-1 update + readout on 128x128 blocks is
   saturation-limited, the same class of kernel floor as the A100 sm_80
   baseline story. Real fused GDN/Mamba chunked-scan kernels may realize a
   larger residency win; measuring them (fla / vLLM GDN Triton kernels) is
   the follow-up.

3. **Contention washes the advantage out through the soft band.** With
   24 MB warm state, interleaving X MB of streaming traffic degrades state
   BW progressively from X=8 (total 32 MB) and floors at the far-field
   rate by total ~56 MB — consistent with the fitted transition band
   (C_eff..1.56xC_eff = 40..62 MB). In full-model decode, a layer's state
   must survive all other layers' traffic between steps, so cross-step
   state residency in production requires the per-layer hot set
   (weights + batch state), not the state alone, to sit below the gate.

4. **Implication — a predicted batch-size knee for hybrid decode.** Batch
   state is a first-class footprint contributor: B x H x dk x dv x 2 per
   linear layer. At Qwen3.5-like geometry (16 linear heads, dk=dv=128:
   0.5 MB/request/layer), the state ALONE crosses H100's C_eff at
   **B* ~= 80 requests** — inside production decode batch ranges — and the
   weights+state total crosses earlier. Predicted consequences,
   falsifiable on real kernels/serving:
   - hybrid-model decode has a throughput knee in B that dense-KV models
     do not have at the same batch;
   - the LLC-growth story (B300 C_eff ~150 MB) buys hybrid models ~3.8x
     the resident batch, i.e., the gate math for the next generation must
     price weights + recurrent state jointly;
   - per-layer state is the natural unit for cache-pinning / persistent-
     kernel scheduling (layer-fused execution keeps state resident,
     defeating the washout in (3)).

## Honest caveats

- Emulated kernel, not fla/vLLM GDN — magnitudes are lower bounds on
  what fused kernels could exploit; the *location* (C_eff onset) is the
  transferable claim.
- Contended-mode subtraction (stream cost measured alone, subtracted) is
  first-order; interleaving also changes the stream's own residency.
- Single layer emulated; multi-layer round-robin (the real decode pattern)
  is the next fidelity step.

Data: `results_state_residency_nvidia-h100-80gb-hbm3.json`.

---

# Addendum (autoloop, 2026-08-03) — the REAL fused GDN kernel changes the story

`bench_fla_gdn.py` (fla 0.5.2 `fused_recurrent_gated_delta_rule`, the
production kernel family for Qwen3-Next/Kimi-Linear decode, fp32 state,
graph-timed, warm vs rotated):

1. **The kernel is saturation-limited at ~2.3-2.4 TB/s — BELOW the HBM
   streaming rate (3.15)** — flat from 8 to 160 MB of state. Warm
   advantage is <=1.13x and only below ~16 MB; by 24 MB it is 1.00.
   Today's fused GDN decode kernel cannot see the L2 tier at all.
2. **Reframing of D1's implication**: the predicted batch knee
   B* = C_eff/state exists in the emulation but is HIDDEN in production
   kernels behind a kernel floor — the same phenomenon as the A100 sm_80
   baseline and the B200 triton w8a8. The concrete opportunity: a
   residency-aware GDN decode kernel has ~2.7x headroom (bw_l2 6.3 vs
   2.35 achieved) at below-gate footprints (B <= ~40 at H=16), and the
   gate predicts exactly where that speedup lives and where it dies.
3. **Washout, measured with the real kernel**: 4-layer round-robin
   (4 MB state + 12 MB weight stream per layer, 64 MB total) costs
   1.29x the sum of its isolated parts — mutual eviction beyond C_eff,
   confirming that per-layer hot sets, not per-operand sizes, are the
   unit the gate prices in real decode.

Data: `results_fla_gdn_h100.json`; washout one-off in scratchpad,
numbers quoted here (state 14.56 us, 12 MB stream 9.17 us, RR 122.82 vs
94.92 us sum).

---

# Addendum 2 (2026-08-03) — the residency-aware kernel EXISTS: 2.2x in the gate window

`gdn_l2_kernel.py`: one fused Triton pass per state tile (decay, k^T S,
delta-rule rank-1 update, store, q^T S readout), traffic exactly 2x state,
correctness 1e-8 vs reference. Config BV=32/nw=4 chosen by search.
Measured against fla 0.5.2 on identical shapes (H=16, dk=dv=128, fp32):

| state MB | fla warm us | ours warm us | speedup | ours rot/warm |
|----------|------------|--------------|---------|---------------|
| 8   | 7.51  | 4.93  | 1.52x | 1.70 |
| 16  | 14.56 | 8.45  | 1.72x | 1.75 |
| 24  | 25.12 | 11.43 | **2.20x** | 1.87 |
| 32  | 32.00 | 16.70 | 1.92x | 1.66 |
| 40  | 38.81 | 31.01 | 1.25x | 1.09 |
| 48+ | ~     | ~     | ~1.1x | 1.00 |

- Warm below-gate BW 3.4-4.4 TB/s (70% of the L2 tier) vs fla's flat 2.3;
  far field 2.52 TB/s (80% of HBM rate) — still above fla. The
  pre-registered success criterion is met: **a gate-shaped speedup, large
  below C_eff, collapsing at the fine-grid onset**, window closing at
  B = 40-48 exactly as B* = C_eff/(H x 64 KB) ~= 40 predicted.
- This is the capacity gate acting as a KERNEL-DESIGN tool: the model
  said where 2.7x headroom lived; a ~100-line Triton kernel captured
  ~80% of it (2.2x) on the first config search. Industry relevance:
  this operator family (Gated DeltaNet) is the decode inner loop of the
  Qwen3-Next / Kimi-Linear / Nemotron-class hybrids that converged in
  March 2026; B300's C_eff (~150 MB predicted) widens the window ~3.8x.
- Honest scope: single decode step, no short-conv/gating epilogues, fp32
  state only, one GPU; fla's kernel handles varlen/beta-vectors/etc. —
  the claim is the residency window and its magnitude, not a drop-in
  replacement.

Data: `results_gdn_l2_kernel_h100.json`.

---

# Addendum 3 (2026-08-03) — epilogue-complete: the win SURVIVES the real layer

`gdn_l2_kernel_full.py`: the full decode step (short conv K=4 + silu with
rolling cache on q/k/v, qk l2norm, delta rule, gated RMSNorm) in ONE
program per (batch, head), vs fla's real chain (3x ShortConvolution.step
+ fused_recurrent(use_qk_l2norm_in_kernel) + FusedRMSNormGated), both
graph-timed, correctness 1e-8/exact:

  below gate: 2.00-2.34x vs the chain (peak at 16 MB state)
  above gate: 1.21-1.30x (pure kernel-fusion dividend, no residency)
  residency signature: rot/warm 1.34-1.58 below, 1.00 at >=40 MB

Decomposition: fusing the 5-kernel chain is worth ~1.25x everywhere;
the gate window multiplies it to ~2.2x. Both effects were predicted:
launch/kernel count from the t0/dispatch work, the window from C_eff.

**Chunked-prefill variant: analyzed, deliberately not built.** fla's
chunk_gated_delta_rule uses matmul-form chunking precisely to hit tensor
cores; a sequential register-resident scan spends 2*dk*dv SIMT FMAs per
token and loses on arithmetic throughput regardless of residency.
Residency-aware design pays where tensor-core economies are absent —
decode — which is where serving spends its memory-bound time anyway.
Scope of the claim stays: decode.

Caveats: fla chain conv/norm run in bf16 (their defaults) vs our fp32
throughout; our kernel lacks varlen/beta-vector/headdim!=128 paths; the
2x-state traffic dominates both sides, so the dtype asymmetry is
second-order (state fp32 in both).

Data: `results_gdn_full_h100.json`.

---

# Addendum 4 (2026-08-03) — B300: the window moves exactly as C_eff says

Fourth architecture (B300 SXM6 AC, nominal L2 126.5 MB — measured
identical to B200; fine-grid C_eff 91.6±1.0 = 0.724x). Same benches,
torch 2.13/triton 3.7 (`results_fla_gdn_b300.json`,
`results_gdn_l2_kernel_b300.json`):

- fla fused_recurrent: 4.6-5.2 TB/s (much better than on H100 — newer
  triton + sm_103), but STILL residency-blind: warm advantage <=1.11,
  gone by 48 MB.
- Our kernel: warm 7.0-9.4 TB/s below the gate (peak 9.38 at 56 MB),
  rot/warm 1.25-1.68 below, collapse at 96 MB; far field 6.1-6.2 TB/s
  = 92% of measured bw_hbm.
- Head-to-head: 1.5x at 16-24 MB, **2.05x at 56 MB**, 1.28x at 96, 1.18x
  far field. **The speedup window moved from ~40 MB (H100) to ~92 MB
  (B300), tracking measured C_eff exactly; B* ~= 92 requests at H=16.**

The design-tool claim is now cross-architecture: the gate predicted the
window's new location on unseen silicon before the kernel ran there.
Caveats: clock lock denied on this tier (air-cooled SXM6, power-limited;
peak fp16 1456 TF); w8a8 triton leg blocked by a triton-3.7 int8 tl.dot
API break (see ceff_reconcile/b300_triton_w8a8_compile_error.log) — a
kernel-port, not physics.
