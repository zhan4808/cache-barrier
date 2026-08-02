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
