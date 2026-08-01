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
