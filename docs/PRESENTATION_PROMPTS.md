# Claude presentation prompts

Paste repo path or upload `cache-barrier` + `kernel-compass` as context. Each prompt is self-contained.

---

## Prompt 1: 20-min lab meeting (technical)

```
You have access to the cache-barrier and kernel-compass repos. Build a 20-minute
lab-meeting slide deck (12–15 slides, outline + speaker notes per slide).

Core narrative:
1. Problem: MLA reconstruction GEMMs are small, partially L2-resident, and
   standard INT4/W4A16 kernels lose to cuBLAS FP16 for two stacked reasons
   (L2 residency removes byte savings + dequant saturates SMs).
2. Method: methodology audit (CUDA-graph timing, warm NCU with cache-control none,
   weight rotation intervention) — cold-cache NCU is invalid for residency claims.
3. Model: measured cache-aware roofline (CARM) with C_eff≈36MB, capacity-gated BW,
   explicit t₀, fitted dequant ceiling.
4. Constructive result: W8A8 INT8-MMA (tl.dot int8→int32, scales on acc) wins
   1.4–1.5× above cliff, 0.70× below — first quantized win, flip at measured C_eff.
5. Generalization: FlagGems fused_moe — host dequant bug fix, same regime structure.
6. Tool: kernel-compass optimizer enumerates W8A8 only when HBM-bound; rejects at 16MB.

Include: one roofline figure, one W8A8 cliff figure, deployment rule table
(L2-served→FP16, HBM weight-bound→W8A8, compute-bound→FP16).
Audience: systems + ML hardware. Purdue CS. Cite measured numbers from
profiling/w8a8/results_w8a8.json and profiling/carm_params.json.
```

---

## Prompt 2: 10-min advisor update (Dr. Xiao)

```
Create a 8-slide executive update for my advisor meeting. Tone: results-first,
honest about limitations. Repos: cache-barrier (paper) + kernel-compass (tool).

Slides:
1. One-sentence thesis: cache residency decides whether quantization helps on
   µs-scale GEMMs — measurable, not heuristic.
2. What we proved: INT4 fails predictably; W8A8 succeeds exactly where model says.
3. Three numbers that matter: 36MB cliff, 1.4× W8A8 win, 15% CARM MAPE.
4. MoE generalization + FlagGems upstream fix (PR status).
5. kernel-compass demo: 16MB→reject quant, 128MB→accept W8A8 automatically.
6. What's NOT done: LLM stage 4, native FP8 tensor cores, multi-GPU MLA e2e.
7. Paper status: draft complete, arxiv package needs figure pass + submit.
8. Ask: feedback on arxiv framing; scope for kernel-compass as standalone tool paper.

Include "questions for advisor" slide with 3 specific asks.
```

---

## Prompt 3: Conference-style talk (ISCA/MICRO flavor)

```
Design a 25-minute conference talk from cache-barrier. Structure:

- Motivation (2 min): DeepSeek MLA, reconstruction as hidden decode bottleneck
- Failed optimization (3 min): roofline says 4× INT4; measured 0.49× — why?
- Audit (4 min): three methodological bugs (cold NCU, event-timing floor, nominal L2)
- CARM model (5 min): equations, measured parameters, MAPE validation
- W8A8 constructive proof (5 min): kernel design, cliff experiment, multi-layer stacking
- MoE + optimizer (4 min): independent confirmation + automated regime selection
- Takeaway (2 min): three-way deployment rule, falsifiable predictions

For each section: key figure, one memorable sentence, anticipated question + answer.
Target audience: computer architecture. No marketing language.
```

---

## Prompt 4: Poster (SC/SysML)

```
Generate a single-page research poster layout (sections + bullet content, not graphics).

Title: "The L2 Cache Barrier for MLA Reconstruction: When Quantization Helps"
Authors: Robert Zhang

Columns: Problem | Methods | Results | Takeaway
Max 600 words total. Emphasize: measured C_eff, W8A8 as constructive confirmation,
kernel-compass as open-source artifact. Include QR-code placeholders for GitHub repos.
```

---

## Prompt 5: kernel-compass tool demo (5 min video script)

```
Write a 5-minute demo script for kernel-compass. Show terminal commands:

  python3 -m optimizer.loop --demo
  python3 -m optimizer.loop --compare

Explain what each output line means (CARM regime, ACCEPTED/REJECTED, graph-timed µs).
Contrast with naive "always quantize weights" approach.
End with roadmap: LLM-guided Stage 4, pytest CI, FP8 native path.
Tone: demo day, live-coding friendly.
```
