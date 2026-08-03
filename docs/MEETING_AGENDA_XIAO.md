# Meeting agenda — Dr. Xiao

**Duration:** 45–60 min  
**Goal:** Align on arxiv submission, kernel-compass scope, and next experiments.

---

## 1. Progress since last meeting (10 min)

**cache-barrier paper**
- Full methodology audit incorporated; paper builds clean (13 pp).
- Main result: L2 cache barrier is *measurable* (C_eff ≈ 36 MB, not nominal 50 MB).
- Constructive confirmation: W8A8 INT8-MMA wins 1.4–1.5× above cliff; first quantized win.
- MoE generalization (FlagGems): fixed host-dequant bug; 2.7–3.0× at high token counts.
- Multi-layer stacking: W8A8 crosses at 3×16 MB layers (48 MB total WS).

**kernel-compass**
- End-to-end demo: 16 MB → reject quantization; 128 MB → accept W8A8 (-29% graph-timed).
- CARM wired into classifier + graph-timed validation.
- Grid vs LLM compare: grid finds W8A8 on HBM-bound shapes; LLM scaffolded.

## 2. Key numbers to discuss (5 min)

| Claim | Evidence |
|-------|----------|
| Effective L2 capacity | Weight-size sweep + warm NCU (61% L2 hit @ 16 MB → 1% @ 48 MB) |
| W8A8 crossover | results_w8a8.json + mla_l2_stack (L=3) |
| CARM predictive power | FP16 MAPE 15.5%, INT4 10.9% |
| MoE regime | Extended T=16–2048 sweep; warm NCU @ T=512 |

## 3. Open questions for advisor (15 min)

1. **arxiv framing:** Lead with audit + CARM, or lead with W8A8 constructive result?
2. **Title scope:** "Cache Barrier" vs "Cache-Aware Quantization for MLA" — too narrow?
3. **kernel-compass:** Separate tool paper, or appendix to cache-barrier?
4. **Stage 4 LLM:** Worth the API cost / complexity for the narrative, or grid-only is enough?
5. **Next hardware:** Any A100 access planned? (paper claims H100-only for now.)

## 4. Proposed next steps (10 min)

**Immediate (this week)**
- Paper consistency pass (abstract MoE crossover, MAPE, add missing figures).
- arxiv submit.

**Short-term (2–4 weeks)**
- kernel-compass: pytest suite for accept/reject matrix + W8A8 validation case.
- FlagGems upstream PR merge + W8A8 MoE path investigation (loses at high T).
- Prefill batch-size sweep with CARM overlay (alternative P5 follow-up).

**Medium-term (1–2 months)**
- Closed-loop optimizer (stateful accept/revert, tile autotune).
- DS-V2-Lite e2e with selective W8A8 on HBM-bound layers only.
- Outreach: Dr. Lin (per DIRECTION.md) with demo + proposal.

## 5. What I need from this meeting (5 min)

- [ ] Green light on arxiv submission timeline
- [ ] Feedback on whether CARM MAPE 15% is acceptable or needs more tuning points
- [ ] Priority: paper polish vs kernel-compass robustness vs new experiments
- [ ] Any collaborators / compute resources to mention in acknowledgments

## 6. Backup slides (if time)

- W8A8 kernel design (one slide: int8 dot, scales on acc)
- kernel-compass `--demo` terminal output
- Deployment rule flowchart (L2 / HBM / compute regimes)
