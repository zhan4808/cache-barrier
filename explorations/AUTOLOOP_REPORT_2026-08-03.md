# Autonomous research loop report — 2026-08-02 23:44 → ~00:45 UTC (H100)

Agenda chosen at my discretion after the six-direction exploration sprint:
the three highest-value follow-ups runnable on this box, executed
GPU-serialized under the clock lock, plus honest write-ups. Finished ahead
of the 2-hour budget because the last planned experiment (B* knee
end-to-end) was resolved analytically by an earlier result.

## 1. Real-kernel leg for recurrent state (the night's biggest finding)

Installed fla 0.5.2 (flash-linear-attention, the production Triton kernel
family for Qwen3-Next / Kimi-Linear) in an isolated venv (`~/fla-env`,
box-local) and measured `fused_recurrent_gated_delta_rule` decode:

- **The fused GDN decode kernel is saturation-limited at ~2.3–2.4 TB/s,
  below HBM streaming rate (3.15), flat from 8 to 160 MB of state.** Warm
  residency buys ≤13% and only below 16 MB. Today's hybrid-model decode
  kernel cannot see the L2 tier.
- **Kernel-opportunity claim (new, concrete, falsifiable): a
  residency-aware GDN decode kernel has ~2.7× headroom** (bw_l2 6.3 vs
  2.35 achieved) at below-gate footprints (B ≤ ~40 requests at
  Qwen3-Next-like geometry), and the capacity gate predicts exactly where
  the speedup lives (B×H×state < C_eff) and where it dies.
- The emulation-vs-real-kernel contrast reproduces the project's central
  model form a third time: capacity structure transfers (both show the
  gate), magnitude is a per-kernel term (emulation 1.15–1.28×, real
  kernel ≤1.13×, both floored well under bw_l2).
- **Washout measured with the real kernel**: 4-layer round-robin decode
  (state + weight stream per layer, 64 MB total) costs 1.29× the sum of
  its isolated parts — per-layer hot sets are the unit the gate prices.

Files: `state_residency/bench_fla_gdn.py`, `results_fla_gdn_h100.json`,
FINDINGS.md addendum.

## 2. C_eff(GEMM) vs C_eff(re-read) reconciliation — hypothesis refuted, gap real

Pre-registered: the June GEMM-fitted 36 MB vs harness re-read 39.8 MB gap
equals activation+output bytes. Result: **not confirmed** — and the
instrument taught two lessons: (a) the T-sweep leg is invalid above T≈58
(bf16 GEMM goes compute-bound; those cliffs are uninterpretable — stated,
not hidden); (b) at T=1 (valid), the GEMM weight cliff sits at **~31–34
MB, below re-read 39.8, unexplained by act bytes** → direction: a genuine
GEMM-context capacity term worth ~6–9 MB on H100 (cuBLAS tiling holds
residency worse than pure re-read). Needs NCU + pinned-kernel sweep to
become a number; follow-up design written. Consequence if it holds: the
model should carry two measured capacities — C_eff(re-read) and
C_eff(GEMM) — and the June 36 was a different operand context, not an
error.

Files: `ceff_reconcile/bench_gemm_ceff.py`, `results_gemm_ceff_h100.json`,
FINDINGS.md.

## 3. B* batch-knee experiment — resolved without running

The planned end-to-end knee measurement was cancelled on evidence: with
the production kernel floored at 2.3 TB/s everywhere, no knee is currently
observable at the serving level (the floor hides it). The knee claim is
now correctly stated as **conditional**: it appears exactly when kernels
become residency-aware — which is the kernel-opportunity claim in §1. A
predictable null was not worth GPU-hours; recorded here instead.

## Strategic synthesis for the debrief

- The strongest new thread this loop opened: **"the L2-invisible GDN
  kernel"** — a measured 2.7× headroom with a capacity-gate-shaped
  speedup region, on the exact operator class the industry converged on
  in March. That is a kernel-engineering paper (or a strong paper section
  + open-source kernel) with the CARM model as its design tool, and it is
  differentiated from generic "optimize Mamba kernels" work by predicting
  the residency window quantitatively across architectures (B300's
  C_eff ≈150 MB widens the window ~3.8×).
- Second thread: the **two-capacities question** (§2) touches the
  foundations of the paper's own constant and has a designed follow-up.
- Everything committed and pushed; venv documented (box-local, restore =
  `python3 -m venv ~/fla-env && pip install torch==2.7.0 flash-linear-attention`).

## Session hygiene

Clock lock held (cap semantics per session 9); GPU serialized throughout;
no agent touched the GPU; all negative/invalid legs stated in FINDINGS
rather than discarded.
