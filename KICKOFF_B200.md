# KICKOFF — B200/Blackwell session (the two remaining predictions)

Paste the prompt below into a fresh agent session on a Blackwell box with the
NFS mount (or a synced copy of `cache-barrier-project`). Everything it needs
is committed in this repo. Companion runbook (read second):
`profiling/cuda_validation/B200_RUNBOOK.md` — the FP4 leg's full plan.

---

## Context (2 minutes)

You are working on `cache-barrier`. Read `DIRECTION.md` §2 and the 2026-08-01/02
entries of `JOURNAL_2026-08.md` first — do not re-derive them.

State of the thesis after the A100 + H100 sessions (through commit `29e9bdd`):
the capacity gate is measured on two architectures; transfer is closed at
**12.4%/13.2% MAPE zero-shot (below/above gate)** via three cheap measured
terms — harness constants + per-kernel two-point calibrations (baseline
kernels included) + one normalized footprint-transition band. Residency is
**footprint-gated** (W+act+out), confirmed in NCU counters; the collapse is
soft (fitted band C_hi≈1.56×C_eff, floor≈0.67×bw_hbm on H100).

**This session has TWO goals, in order:**

1. **Third architecture point** (~1 h GPU): run
   `profiling/portable/measure_params.py` on the B200. Blackwell's ~126 MB LLC
   is the largest jump in the LLC-growth story (slide 12); the thesis predicts
   the below-gate regime expands. Sanity: C_eff should land at 0.7–0.8×
   nominal if the H100/A100 effective/nominal ratio (0.78) holds — that ratio
   holding OR breaking is a headline result either way. Then the reduced gate
   sweep (`profiling/gate/bench_capacity_gate.py --params <b200 json>
   --tsweep 1,16,32`) — success = the w8a8 advantage structure tracks the
   *measured* B200 C_eff. Also transfer-validate: two-point-calibrate the
   baseline + kernels on 2 points, predict the rest, report regime-separated
   MAPE with the H100-fitted band transferred normalized (zero-shot). That
   makes the transfer claim three-architecture.

2. **The native-FP4 prediction** (the one open falsifiable hole; full plan in
   `B200_RUNBOOK.md`): a matched-precision W4A4 kernel with native FP4 MMA
   should break the in-core dequant ceiling (r_dequant → ∞ in CARM terms) and
   keep winning in the compute-bound regime where dequant-path W4A16 crosses
   back under. Follow the runbook's order: probe FP4 backends, run the
   no-code-change benches, then `bench_moe_nvfp4_native.py`.

## Hardware requirements (verify before any measurement)

- Whole GPU, MIG disabled (`nvidia-smi --query-gpu=mig.mode.current`).
- Clock lock: query max with `nvidia-smi --query-gpu=clocks.max.sm
  --format=csv` and `sudo nvidia-smi -lgc <max>,<max>`; reset with `-rgc`
  before finishing. Note lock status in every results file.
- The H100 venv will NOT have native FP4 ops (B200_RUNBOOK §1) — goal 1 needs
  only system torch+triton with CUDA graphs; goal 2 needs the runbook's env.

## Guardrails (full list in KICKSTART.md)

- CUDA-graph timing only; never nominal capacity in the model; regime-separated
  MAPE; negative/worse results reported plainly (the FP4 prediction FAILING
  would be a major honest result — do not rescue it).
- The per-run repeat lesson (session 6): engine-level and first-run numbers
  need a repeat pass before any claim; kernel-level graph-timed medians are
  stable.

## Deliverables

1. `profiling/portable/params_<b200-slug>.json` — committed
2. Gate sweep + transfer results JSONs + updated figures
3. FP4 leg results per B200_RUNBOOK (or a plainly-stated blocker report)
4. `JOURNAL_2026-08.md` entry (same style; newest at bottom)
5. Paper: §Setup hardware, §Transfer (three-architecture), the B200 bullet in
   the LLC-trend paragraph, limitations; deck: slide 12 trend + status slide;
   one-line WeChat addendum
6. `sudo nvidia-smi -rgc`; sync any non-canonical copy back per
   `docs/HANDOFF_2026-08-02.md` §sync
