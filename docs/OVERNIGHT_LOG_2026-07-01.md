# Overnight autoresearch log — 2026-07-01 → 2026-07-02

## ⭐ TL;DR (read this first)
**All 6 queue items + synthesis DONE. Dr. Xiao's three todos are answered with
measurements, committed to cache-barrier main (7 commits).**
1. **Dense (todo 1):** all three cache regimes measured on Qwen3.6-27B; boundary
   super-win 2.4–3.0× (clock-locked confirmed); the direct capacity-gate figure
   (`fig_l2_boundary.png`) shows each precision cliffing at its own operand size —
   and rotation (serving eviction) erasing the cliffs.
2. **KV serving (todo 2):** KV reads are NOT L2-limited (no L2 tier anywhere;
   rotation-invariant). fp8-KV ≤1.07× streamed, 0.69–0.72× loss small-batch;
   end-to-end ceiling ≤0.2% on this model (Amdahl vs his own profile).
3. **W8A8 (todo 3):** wins at EVERY T on MoE (1.5–3.15×) and dense — no dequant
   cliff; W4A4 not native on Hopper, parked. Act-quant ~7–9µs measured as the
   deployed-path tax.
4. **CARM v2:** operand-aware gate + act-quant + contention; 7.4% MAPE bf16 /
   ~15% quant over 270 cells.
Full story + next-steps: `docs/WEEK1_REPORT_2026-07-02.md`. Figures in
`profiling/dense_qwen/fig_*.png`. GPU clocks reset; repo clean.

---

## Iteration 1 — loop start
- Plan: `OVERNIGHT_PLAN_2026-07-01.md`. Queue item 1: KV-cache quant under
  serving conditions (Dr. Xiao todo #2), Qwen3.6-27B full-attn GQA shape
  (24 q / 4 kv heads × 256).
- Action: probed fp8-KV decode paths. Findings: vLLM 0.20.2 FA3
  `flash_attn_varlen_func(fa_version=3)` is the production path; **fp8 KV
  requires fp8 query too** (vLLM quantizes q in the attention layer —
  `q_descale`); default fa_version=2 rejects mixed dtypes.
- Built `profiling/kv_serving/bench_kv_decode.py` (paged FA3 decode, GQA
  24q/4kv×256, seeded caches, warm+rotated). Smoke: relerr 0.033;
  early signal fp8-KV 0.71× at (1k,B1) [L2-resident, loses] and 1.01× at
  (8k,B8) [268MB streamed, washout] — same structure as FlashMLA.
- Full sweep (10 configs × warm+rotated) launched in background.
- Next: on completion → analyze, RESULTS.md, commit; then queue item 2 (CARM v2 fit).

## Iteration 2 — KV answer + CARM v2 (items 1–2 DONE)
- **Item 1 ✅ (commit 011e122):** KV-cache reads are **NOT L2-limited** — no L2
  BW tier at any working set (occupancy-bound small, HBM-streamed large;
  rotation moves ratios ≤0.03). fp8-KV: ≤1.07× streamed, **0.69–0.72× loss**
  at small batch/ctx; fp8 kernel BW ceiling ~1.5–1.65 vs bf16 2.8–3.1 TB/s —
  FlashMLA mechanism reproduced on GQA/FA3. Amdahl: ≤0.2% end-to-end.
  → `kv_serving/RESULTS.md`.
- **Item 2 ✅ (commit 06a4b42):** CARM v2 fit (operand-aware capacity gate +
  act-quant term): bf16 MAPE **7.4%**, w8a16 14.8%, w8a8 15.5% (270 cells).
  New findings from constants: Marlin W8A16 gets **no L2 tier** (2.54≈2.58 TB/s
  — why only W8A8 super-wins the boundary); dense dequant ceiling 334 TFLOPS.
  Caveats: act-quant lstsq over-weights large-M; kv_proj M*=1 vs pred 80
  (Marlin fixed overhead not modeled).
- **Item 3 launched:** clock-locked red-team of claim A (boundary super-win)
  + claim B (kv_proj rotation flip), 3 interleaved rounds. Clock will be reset
  by the same job.
- Next wake: analyze red-team → commit → item 4 (W8A8 MoE).

## Iteration 3 — red-team confirmed (item 3 DONE), W8A8 MoE launched
- **Item 3 ✅ (commit b2f8aeb):** clock-locked, both claims CONFIRMED and
  stronger: super-win 2.41–2.97× warm (L2 tier 4.25–5.39 TB/s) → 1.68–1.75×
  rotated; kv_proj flip 0.77–0.84× warm → 1.07–1.09× rotated at M≤16 (no flip
  M=64 — dequant overhead dominates). Clocks reset (verified 345 MHz idle).
- **Item 4 launched:** W8A8 MoE via fused_experts + fp8_w8a8_moe_quant_config.
  Smoke (Mixtral): 1.86× @T=16, **3.12× @T=512 — NO cliff** (W8A16 was 1.23×
  and crossing). ⚠ analysis must reconcile vs TUNED bf16 (part of mid-T
  magnitude is the §7 stock-baseline artifact; tuned bf16 T=512 ≈1377µs
  clock-locked → w8a8 ≈1.9× vs fair baseline). relerr 0.046 (per-tensor scales
  on random weights — note).
- Next wake: analyze W8A8 MoE → RESULTS + commit → item 5 (GDN hybrid shapes).

## Iteration 4 — W8A8 MoE DONE (item 4), items 5+6 launched
- **Item 4 ✅ (commit f449364):** W8A8 MoE wins at EVERY T, no cliff.
  Mixtral 1.82–3.15× vs stock bf16 (~1.9–2.0× vs tuned); Qwen fine 1.50–1.75×.
  CARM "matched precision wins everywhere native" now measured on MoE + dense.
  Task list #4 closed.
- **Items 5+6 launched (one serialized GPU job):** GDN in_proj_qkvz
  (5120→16384, 168 MB — hybrid-layer coverage; out_proj ≡ o_proj already
  measured) + fine L2 boundary sweep (8→128 MB, K=8192, M=16, 3 paths,
  warm/rotated) to pin the capacity cliff for CARM v2.
- Remaining queue: item 7 synthesis (WEEK1 report + figures + memory + TL;DR).

## Iteration 6 — stretch items DONE; run concluding
- Contention sweep ✅ (commit 4677a8a): **residency collapse is a STEP function**
  — L2 tier only while Σ co-tenant WS < C_eff; full-model serving always exceeds
  it → L2 effects don't survive full-model serving (sharpens honest scope; CARM
  contention factor = binary gate on total hot WS).
- Per-channel/per-token scales: rel-err stuck at ~0.037 = e4m3 floor on Gaussian
  data (+2.3% latency); real-model accuracy eval remains the open gap.
- Act-quant measured directly: 8.4 µs fixed + ~1.6 TB/s (fixes CARM term).
- lm_head anchor: 2.85 TB/s (2.5 GB streamed).
- **Full plan + all stretch items complete.** Remaining optional: 3 new deck
  slides for the next meeting (deferred to Robert's direction). Loop idling on
  long heartbeat; GPU clocks default; repo clean and committed (9 commits).

## Iteration 7 — night extension (user confirmed: keep going all night)
- Extended plan with items 8–13 (see plan file).
- **Item 8 ✅ (57635f2):** W8A8 MoE vs TUNED bf16, clock-locked: **1.80–1.94×
  flat at every T, no cliff** — the 3.15× was baseline-inflated; honest headline
  ~1.9× uniform. Clocks reset.
- **Item 11 ✅ (57635f2):** e2e Qwen3.6-27B decode model: W8A16 **1.50×** |
  W8A8 unfused **1.19×** | W8A8 fused-quant bound **1.73×** (realistic ~1.6×).
  Key deployment insight: at decode M=1 the act-quant tax beats the dequant
  ceiling → W8A16 > unfused W8A8; fusion is the unlock.
- **Item 9 ✅ (this commit):** fused rms_norm+quant tax = +2.1–2.2 µs (vs 8.4
  standalone) at decode M → tax mostly recoverable; silu+quant fusions exist.
- **Item 10 ✅ (this commit):** INT8 ≈ FP8 matched (±20% small, parity large).
- Remaining: item 12 (deck additions draft), item 13 (CARM v2 → kernel-compass).

## Iteration 8 — packaging & loop shutdown (per Robert's direction)
- Framing corrections applied (report Addendum 2 + update deck): (1) headline is
  the contention/step-function finding, 2.6× presented with its washout in the
  same breath; (2) e2e numbers explicitly labeled PROJECTION (sum-of-GEMMs ×
  86.2% profile share — never a served model); (3) W8A8 speed claim paired with
  its accuracy cost (0.037 vs 0.0025; real-model accuracy = open gap).
- Item 12 ✅: `docs/update_2026-07-02_overnight.html` (6 slides, EN/ZH, figures
  embedded, ends with the B300 native-FP4 ask). WeChat draft (ZH+EN):
  `docs/wechat_draft_2026-07-02.md`.
- Item 13 SKIPPED by decision (housekeeping; carm.py port deferred).
- No third overnight wave. **Loop stopped.** GPU clocks default; repos clean.
