---
name: cache-barrier-project-state
description: State of the cache-barrier research project as of 2026-08-02 B200 session (session 8)
metadata: 
  node_type: memory
  type: project
  originSessionId: a864ca3e-f250-43f1-a4a7-dc1bd9551d1e
  modified: 2026-08-03T01:40:44.150Z
---

Cache-barrier project (CARM roofline + quantization gate paper) lives at
`/lambda/nfs/robert-nfs/cache-barrier-project/repos/cache-barrier`. As of
2026-08-02 evening: local main == origin/main at `9fb3f06` (B200 session
pushed from the B200 box with a PAT; kernel-compass `0c582fa` pushed via the
NFS SSH key). HANDOFF_2026-08-02_B200.md's "push pending" item is stale.
Commit `e9b4cc7` has ad-hoc author "Robert (B200 session)" — published,
cosmetic, fix needs force-push. User is Robert Zhang <zhan4808@purdue.edu>
(github zhan4808).
GitHub auth: dedicated key `/lambda/nfs/robert-nfs/github-robert-ed25519`
(created 2026-08-02), wired into all three repos via local `core.sshCommand`
— works on any machine mounting the NFS. (`lambda-robert-ed25519` at the NFS
root is the Lambda instance key, not a GitHub key.)

Session 8 (B200, 2026-08-02) established: third architecture point
(C_eff 98.8 MB), zero-parameter transfer band (H100 floor refuted), and the
native-FP4 confirmation (W4A4 NVFP4 never crosses bf16; 6.3× over Marlin
W4A16 at T=2048 = dequant ceiling removed by SM100 MMA).

**Session 9 (2026-08-02, run by me on the canonical H100 box, which is
68.209.75.33 — the box Claude runs on)**: the grid-quantization issue I
found is FIXED — fine-grid cliff re-sweep on H100 (cliff_finegrain.py) gives
collapse onset 39.8±0.5 MB = 0.795±0.010× nominal; paper/deck/WeChat
restated as "common ≈0.8× fraction" (WeChat item 15 = the correction).
Floor-free H100 refit (refit_floorfree_band.py): floor worth ~3 pts
below-gate locally (14.3→17.3), above-gate neutral — real local term, not
transferable; zero-param band is the transferable form. sm_clock idle-
snapshot bug fixed in 5 bench scripts (now sample under load); B200 JSONs
annotated. Found: -lgc 1755 on H100 still throttles to ~1385 MHz under
compute saturation (lock caps, doesn't pin). bench_l2_boundary.py
parameterized for the future B200 tuned-CUDA w8a8 run (B200_RUNBOOK §6).
This box had no TeX (fresh instance) — texlive installed session 9.

**Exploration sprint (2026-08-02/03, session 9+)**: six new directions in
`explorations/` (commits 7ad031d, 0158f7b). Measured on H100: recurrent
state (D1) and sparse-selected KV (D2) are both gate-governed — collapse at
the fine-grid C_eff onset; cliff is BYTE-located (D5) so fp8/int4 KV buys
2x/4x resident batch; predicted hybrid-decode batch knee B* = C_eff/state.
Agent memos: amd_portability.md (MI300X L2_cache_size reports one XCD =
4 MB, harness would break; two-cliff IC hypothesis; ~$6 rental run),
carm_router_design.md (regime-router June study was a completed NEGATIVE
result on request-level precision routing; pursue modeling claim not
systems race), PREREG_B300_RUBIN.md (P1-P6 falsifiable predictions; B300
192 MB LLC is secondary-source only — paper softened accordingly).

**Autoloop (2026-08-03, commit e6c2482)**: real fla GDN decode kernel is
L2-BLIND — saturates 2.3-2.4 TB/s (below HBM rate) flat to 160 MB state;
kernel-opportunity claim: residency-aware GDN decode has ~2.7x headroom in
the gate window B*H*state < C_eff. Round-robin washout 1.29x measured.
GEMM-context C_eff gap real (~31-34 vs 39.8 re-read; act+out hypothesis
refuted; two-capacities question open, NCU follow-up designed). fla venv
at ~/fla-env (box-local). Full report: explorations/AUTOLOOP_REPORT.

**Session 10 (2026-08-03, commit fb9ca54)**: BOTH follow-ups landed.
gdn_l2_kernel.py (Triton, ~100 lines, correctness 1e-8) is 2.2x faster
than fla at 24 MB state, 1.5-1.9x below gate, window closes at B=40-48 =
pre-registered B* — the gate as a kernel-design tool, on-device. NCU
(installed on this box) corroborates two capacities: GEMM transition
~34 MB vs re-read ~40 MB. Both were pre-registered before measurement.

**Session 10b (commit 862a7e5)**: epilogue-complete kernel
(gdn_l2_kernel_full.py: conv+silu+l2norm+delta+gated-RMSNorm fused) is
2.00-2.34x vs fla's real 3-kernel chain below gate, 1.21-1.30x above
(fusion x residency decomposition); chunked prefill analyzed-not-built
(tensor cores own it; claim is decode-only). Paper has new section "The
Gate as a Design Tool" (17 pp, 0 errors). UPSTREAM_fla_ISSUE_DRAFT.md
ready — POSTING NEEDS ROBERT (no gh auth; goes out under his name).

**Session 11 (B300, commit 664c156)**: user's B300 box (ssh
root@195.26.233.156 -p 18376, /root/bench-env torch 2.13+cu130). Nominal
L2 = 126.5 MB = B200 (192 rumor dead). Fine-grid C_eff 91.6 = 0.724 —
**P1 falsified** (band was 0.77-0.82); coarse grid reads 0.781 (the
quantization critique demonstrated). Cross-arch claim now "0.72-0.80,
architecture-varying". L2:HBM GROWS to 2.46. t0 1.52us (torch 2.13;
stack-tracking). GDN kernel window moved 40->92 MB tracking C_eff; 2.05x
vs fla at 56 MB. triton 3.7 int8 tl.dot API break blocks w8a8 (logged).

**Session 12 (full board, commits through 'Serving baseline')**: B200
fine-grid C_eff 90.3 = 0.714 -> **family-clustered ratios** (Hopper 0.80,
dual-die Blackwell 0.71-0.72) — the big remaining claim revision, in
paper/deck/prereg. w8a8 confound CLOSED (cutlass 2.0-2.7x above gate).
GDN window third arch. Sawtooth attributed (persistent nvjet, tile-ceil
x variant selection). P6 falsified at 2.66. FlagGems: whole ecosystems
can be L2-blind; hardware cliff invariant. Related work integrated
(18pp 0 err, MLSys 2027 target Oct 30). Serving baseline: no knee,
kernel-masked as predicted (Qwen3-Next-80B/B200 curve committed).
gdn_window money figure in paper. ReplaySSM cited as composable.

Open — user: post fla issue (2-arch draft ready), MI300X budget, B200
box disposition. Research: A100 fine-grid (needs A100), residency-aware
kernel serving integration (the beat-this-curve experiment), MLSys
compression 18->10pp, inline-arXiv bib fixes (flagged pre-submission).
Prereg culture: falsifications (P1, P6) presented as evidence the
confirmed predictions are meaningful — keep that framing.
