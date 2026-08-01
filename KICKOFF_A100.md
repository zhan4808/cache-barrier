# KICKOFF — A100 session (transfer validation leg)

Paste the prompt below into a fresh agent session on the A100 box with the NFS
mount available. Everything it needs is committed in this repo.

---

## Context (2 minutes)

You are working on `cache-barrier` at
`/lambda/nfs/robert-nfs/cache-barrier-project/repos/cache-barrier` (adjust the
mount path if different on this box). Read `DIRECTION.md` §2 and the
2026-08-01 entries of `JOURNAL_2026-08.md` first — do not re-derive them.

One-sentence thesis: quantization's speedup is gated by a cache-capacity
condition; a three-parameter measured model (CARM) predicts the win/loss flip
per shape, per architecture, at dispatch time.

**This session has ONE primary goal**: replace the *estimated* A100 constants
in the cross-architecture transfer validation with *measured* ones, and
re-report the transfer MAPE. That single number (currently a 23–29% zero-shot
upper bound built on estimates) is the paper's portability claim. Everything
else is stretch.

## Hardware requirements (verify before any measurement)

- A100 80GB **SXM4** (PCIe has a different memory system; the in-repo 2026-06
  A100 dataset is SXM4). `nvidia-smi -L` to confirm.
- **Exclusive whole GPU, no MIG** (MIG partitions L2 → capacity measurements
  meaningless). `nvidia-smi --query-gpu=mig.mode.current --format=csv` must
  say Disabled.
- Clock-lock privileges: `sudo nvidia-smi -lgc 1410,1410` (A100 max SM clock;
  reset with `-rgc` when done). If this fails, note it in every results file.

## Setup (~5 min)

```bash
cd <repo>   # the cache-barrier checkout above
python3 -c "import torch,triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
# if missing: pip3 install --user torch triton matplotlib
sudo nvidia-smi -lgc 1410,1410
```

No vLLM, no model download, no venv restore needed for the primary goal.

## Primary task (~30 min GPU)

```bash
cd profiling/portable
python3 measure_params.py          # emits params_nvidia-a100-sxm4-80gb.json
```

Sanity-check the output against expectations: C_eff should land somewhere in
25–36 MB (nominal 40; H100's effective/nominal ratio was 0.72–0.78),
bw_hbm ≈ 1.8–2.0 TB/s, bw_l2 ≈ 3.5–5 TB/s, t0 ≈ 2–4 µs. If C_eff comes back
null, look at `residency_cliff_points_mb_tbs` in the JSON — the collapse
should be visible by eye; adjust the detector threshold only with the cliff
data in front of you, and say so in the journal.

Then update `transfer_validation.py`:
1. Replace the `A100 = dict(...)` estimates with the measured values
   (keep the old dict in a comment labeled "2026-08-01 estimates, superseded").
   Use `t0_eager` from the harness for the eager-timed 2026-06 target data.
2. Re-run: `python3 transfer_validation.py`. Report the new fp16 zero-shot
   MAPE (below/above gate separately, guardrail 7) and the naive-vs-calibrated
   W4A16 numbers. **Either direction of change is a legitimate result** — if
   measured constants make MAPE worse, that is a finding about the model form,
   not a failure to hide (guardrail 8).

## Stretch tasks, in value order (only after the primary is committed)

1. **Gate flip on A100** — the cross-architecture version of Figure 1.
   `profiling/gate/bench_capacity_gate.py` currently loads H100 params from
   `../carm_model.json`; parameterize it to read the A100 params file
   (C_eff/bw/peak) and run a reduced sweep: T ∈ {1, 16, 32}, full W list.
   Success = the sign flip brackets the *measured* A100 C_eff, not H100's 36 MB.
   Graph-timed, clock-locked, same guardrails. This upgrades the paper's
   transfer section from "latency MAPE" to "the gate itself transfers".
2. **Graph-timed re-measure of the 2026-06 A100 sweep points** (the current
   transfer target is eager-timed with a 15.5 µs floor — guardrail 2 caveat).
   Reuse `bench_l2_barrier.py` shapes with the graph timer from
   `profiling/gate/bench_capacity_gate.py`; re-report transfer MAPE against
   clean data.
3. **r_dequant two-point microbenchmark** is already inside measure_params.py;
   compare its A100 value against the 0.406 TB/s fitted from the 2026-06 data.

## Deliverables

1. `profiling/portable/params_nvidia-a100-*.json` — committed
2. Updated `transfer_validation.py` + regenerated `results_transfer_a100.json`
   and `fig_transfer_a100.png`
3. A `JOURNAL_2026-08.md` entry (same style as the 2026-08-01 sessions)
4. If numbers changed: update the transfer paragraph in `paper/main.tex`
   §Cross-Architecture Transfer, the A100 limitation bullet, and slide 13 of
   `docs/presentation_2026-08-01_gate.html` (the `⌁ source` line and MAPE
   numbers; republish artifact if you have the URL context)
5. One-line WeChat addendum: "A100 实测参数替换估计值:迁移 MAPE X%→Y%"
6. `sudo nvidia-smi -rgc` before you leave

## Guardrails (full list in KICKSTART.md — these are the ones this session can violate)

- CUDA-graph timing only for kernel claims (eager floor ~15.5 µs on A100 too)
- Lock clocks; note it in every results file
- Nominal ≠ effective: never put 40 MB in the model
- Regime-separated MAPE, below/above gate
- Negative/worse results reported plainly — the direction is a hypothesis
