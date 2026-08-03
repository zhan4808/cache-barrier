# Pre-shutdown checklist (2026-06-10)

## Must-do before power-off (~2–3 hr)

- [x] Push all repos to GitHub (cache-barrier @ 972ea73, kernel-compass @ 872bc7c)
- [x] NFS snapshot via `scripts/backup_to_nfs.sh`
- [ ] Paper consistency pass (30 min):
  - Fix abstract/contributions: remove ~230-token MoE crossover → "2.7–3.0× at T≥512"
  - Update CARM MAPE: 15.5% FP16 / 10.9% INT4 (not 10%/18%)
  - Fix fig:l2barrier caption (12 TB/s → 4–6 TB/s)
  - Add `mla_l2_stack.pdf` + MoE figure to paper §5.7/§5.8
  - Rebuild arxiv tarball; verify compile from tarball alone
- [ ] Commit `parse_ncu_warm.py` + reconcile `fused_moe/REPORT.md` Finding 3
- [ ] Append W8A8/CARM/MoE sections to `profiling/RESULTS.md`

## Should-do if time (~4–6 hr)

- [ ] kernel-compass: fix `test_classifier` pytest fixture; add W8A8 validation case
- [ ] kernel-compass: add W8A8 to LLM search space (`optimizer/llm.py`)
- [ ] `git submodule update --init` in kernel-compass
- [ ] FlagGems upstream PR status note in fused_moe/REPORT.md

## Next instance (~1–2 days)

- [ ] arXiv submit `paper/dist/arxiv_submission.tar.gz`
- [ ] kernel-compass closed loop: implement `select_candidate/propose/validate`
- [ ] pytest CI for CARM + graph-timed accept/reject matrix
- [ ] Native FP8 W8A8 path (cuBLASLt) as optional tier
