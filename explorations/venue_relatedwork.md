# Venue + related-work memo — cache-barrier paper (2026-08-03)

Scope: prior-art scan for the residency-aware GDN decode-kernel claim, related-work
delta since June 2026, venue scan with real August-2026 dates, and a
split-vs-integrate recommendation for the kernel result. All links verified by
fetch on 2026-08-03 unless marked otherwise.

---

## 1. Prior-art verdict on the kernel claim: **CLEAR, but the neighborhood is hot**

The specific claim — *a GPU decode kernel for Gated DeltaNet whose state is
deliberately L2-resident, with the speedup window located and bounded by a
measured effective-capacity model (C_eff), demonstrated on two architectures with
the window moving as C_eff moves* — is, as far as hard searching can establish,
**not anticipated anywhere**. No paper, blog, or fla/vLLM/SGLang issue proposes
L2-capacity-gated residency for linear-attention/SSM decode state. But four
lines of work are close enough that the paper section and the fla issue MUST
engage them explicitly, and one of them changes the framing:

### 1a. The FPGA persistent-state GDN accelerator — closest, and it helps us

**Gupta, Wang, Kannan, Prasanna, "A Persistent-State Dataflow Accelerator for
Memory-Bound Linear Attention Decode on FPGA"** (arXiv:2603.05931, March 2026).
https://arxiv.org/abs/2603.05931

- Identifies *exactly our problem*: GDN decode is memory-bound; they write that
  "the H100's 50 MB L2 is a hardware-managed cache with **no guarantee of
  persistence across kernel invocations**, so the 2 MB GDN state must be
  round-tripped through HBM every token."
- Their solution is to leave the GPU: Alveo U55C FPGA, state held in BRAM
  scratchpad, 4.5x vs an H100 PCIe GPU baseline, 60x energy.
- **How close**: same operator, same bottleneck, explicit L2-residency framing —
  but they assert GPU L2 residency is unattainable and build hardware instead.
  Our measurement refutes their premise below C_eff: warm state *does* persist
  across decode-step launches on H100/B300, and a ~100-line Triton kernel
  captures it. This is the strongest possible related-work setup: "prior work
  declared this impossible on GPUs and taped out an FPGA; the capacity gate says
  it is possible below C_eff, and predicts precisely where it stops being
  possible." Cite it prominently; it converts a threat into motivation.

### 1b. ReplaySSM (Tri Dao, June 2026) — the one that must be answered

**"ReplaySSM: Cache SSM Inputs, Not State"** — blog post 2026-06-15
(https://tridao.me/blog/2026/replayssm/) + vLLM RFC #47572
(https://github.com/vllm-project/vllm/issues/47572). No formal paper found.

- Mechanism: algorithmic traffic reduction — cache recent *inputs* in a small
  buffer, checkpoint the state only at flush steps, replay inputs to
  reconstruct. "Roughly halves the dominant state traffic."
- Numbers: **1.43–1.84x kernel speedup, 1.20–1.48x end-to-end** on Nemotron-3
  and Qwen3.5; 1.87–1.96x with speculative decoding; **on H100 and B300** —
  the same models and the same two GPUs as our result.
- **No mention of L2 residency or capacity anywhere** — it is orthogonal in
  mechanism (reduce bytes vs make bytes hit L2) and the two should compose
  (replayed-input traffic is even smaller and even more residency-friendly).
- **How close**: not prior art for the residency claim, but it is a June-2026
  baseline from the field's most prominent kernel author, on the same kernels,
  with overlapping speedup magnitude. Any reviewer who knows the area will ask
  "why not compare to ReplaySSM?" — and the fla issue draft, written without
  knowledge of it, currently reads as if fla's fused_recurrent is the frontier.
  **The paper section and UPSTREAM_fla_ISSUE_DRAFT.md must both cite it**,
  state the orthogonality, and ideally measure ours against (or on top of) the
  ReplaySSM traffic pattern. This is the single most important fix this memo
  recommends.

### 1c. SpecLA (July 2026) — SRAM-resident state, different scope

**"SpecLA: Efficient Speculative Decoding for Linear-Attention Models"**
(arXiv:2607.16673, July 2026). https://arxiv.org/abs/2607.16673

- Keeps GDN state tiles **SMEM-resident across speculative candidate tokens
  within one launch** (layer-major reordering, V-dim tiling into 228 KiB/SM),
  delayed state updates to avoid round trips. 1.70x on GDN-1.3B/H100.
- **How close**: "keep the state on-chip" is the shared instinct, but it is
  shared-memory residency *within* a kernel for a speculation window, not L2
  residency *across* decode-step launches, and there is no capacity model, no
  batch window, no cross-architecture prediction. Cite as the SMEM-tier cousin.

### 1d. Megakernels / persistent kernels — the fusion half of our decomposition

Mirage Persistent Kernel (https://github.com/mirage-project/mirage, 2025),
AutoMegaKernel (arXiv:2606.09682, June 2026), Ada-MK (arXiv:2605.11581,
May 2026), Fleet (arXiv:2604.15379), HazyResearch low-latency megakernels,
Kog's MI300X single-kernel engine. All fuse decode into persistent kernels for
launch-overhead and locality wins; none place recurrent state under an L2
capacity model or predict a batch-size window. Our measured decomposition
(fusion ~1.25x everywhere, residency ~1.9x inside the window) is exactly the
right instrument to position against this line: the fusion dividend is theirs;
the residency window is ours. PERKS (arXiv:2204.02064) is the older HPC
precedent for cache-resident persistent-kernel iteration — worth one sentence.

### 1e. Things checked that came up empty

- **fla-org/flash-linear-attention issues/PRs**: GitHub search for
  decode+bandwidth+L2 → **0 results**; the open decode-related issues
  (#1096 grid-size bug, #1028/#1030 split-decode bugs, #502 chunk-prefill RFC)
  are correctness/plumbing, not bandwidth or residency. Nothing anticipates the
  issue we plan to file. (Search: https://github.com/search?q=repo%3Afla-org%2Fflash-linear-attention+decode+bandwidth+L2&type=issues)
- **FlashQLA** (QwenLM, Apr 2026, https://github.com/QwenLM/FlashQLA): Qwen's
  purpose-built GDN kernels, 2–3x vs fla — **chunked prefill/training forward+
  backward only, no decode kernel, no L2/residency discussion** (verified in
  README). Consistent with our own "matmul-form owns prefill" analysis; cite it
  as evidence the prefill side is served while decode is not.
- **FlashInfer-Bench** has `gdn_decode_*` kernel tasks
  (https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last) — an
  engineering leaderboard, no residency-aware submissions found on the page;
  worth monitoring, and potentially a distribution channel for our kernel.
- **cudaAccessPolicyWindow / L2 persistence** literature: CUDA-doc-level
  material and one KV-prefetch paper (below); nobody applies set-aside L2
  persistence to SSM/linear-attention state.
- "Cache-Resident LLM Inference in GB-Scale Last-Level Caches"
  (arXiv:2606.25353, June 2026) is **CPU** LLC weight residency — related-work
  for the gate framing, not kernel prior art.

**Verdict for the director**: kernel claim is **clear** — proceed with the fla
issue and the paper section, but both must be updated to cite ReplaySSM (1b)
and the FPGA paper (1a) before anything goes public. The FPGA paper makes our
framing stronger; ReplaySSM unanswered makes it weaker.

---

## 2. Related-work delta since ~June 2026 (plus two slightly older must-adds)

1. **KernelSight-LM** (arXiv:2606.28565, June 26 2026) — kernel-level LLM
   inference simulator; roofline + *learned* efficiency term, 12.1% cross-gen
   error vs 22% plain roofline. Relevance: same "roofline is not enough" thesis,
   but patches it with learning where we add a measured capacity term —
   the natural head-to-head citation for CARM's positioning.
2. **Microbenchmarking NVIDIA's Blackwell Architecture** (arXiv:2512.02189,
   Dec 2025, rev. 2026) — B200 deep-dive: 148 SMs, four L2 partitions, TMEM;
   reports L2 hit-rate behavior vs precision. Relevance: independent
   corroboration of our B200 constants and the partitioned-L2 topology caveat
   in P4's "either way we learn."
3. **Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks**
   (arXiv:2507.10789, July 2025) — L2 latency/bandwidth H100-vs-B200.
   Relevance: the microbenchmark lineage (Volta/Turing papers' successor) our
   harness extends with *effective residency capacity*, which none of them
   measure.
4. **Jarmusch & Chandrasekaran, Microbenchmark-Driven Analytical Performance
   Modeling Across Modern GPU Architectures** (arXiv:2605.04178, May 2026) —
   B200/MI300A analytical models incl. cache params, 1.3% MAE on 21 kernels.
   Relevance: closest current methodological neighbor to the portable harness;
   differentiate on capacity-gating + zero-shot cross-arch transfer + the gate
   as a *dispatch* predicate.
5. **Cache-Resident LLM Inference in GB-Scale Last-Level Caches**
   (arXiv:2606.25353, June 24 2026) — CPU clusters, weights held LLC-resident,
   2.04–11.51x TPOT. Relevance: the gate's "residency changes the dispatch
   calculus" logic independently arriving on CPUs; strengthens the
   LLC-growth-trend section.
6. **ReplaySSM** (blog + vLLM RFC #47572, June 15 2026) — see 1b. Relevance:
   competing/composable traffic-reduction baseline for the GDN section.
7. **Gupta et al., Persistent-State Dataflow Accelerator for Memory-Bound
   Linear Attention Decode on FPGA** (arXiv:2603.05931, Mar 2026) — see 1a.
   Relevance: motivates the kernel section; its GPU-impossibility premise is
   what our measurement overturns.
8. **SpecLA** (arXiv:2607.16673, July 2026) — see 1c. Relevance: SMEM-tier
   state residency for speculation; cite to delimit our L2-tier claim.
9. **Gated DeltaNet-2** (arXiv:2605.22791, NVIDIA, May 2026) — erase/write
   decoupled delta rule, fused Triton chunked kernels. Relevance: the operator
   family is moving fast at the *algorithm* level; our kernel/window analysis
   applies to its decode step too (same state shape) — one sentence of forward
   relevance.
10. **AutoMegaKernel** (arXiv:2606.09682, June 2026) — agent-synthesized
    whole-model persistent kernels. Relevance: the fusion axis of our
    fusion-x-residency decomposition, now automated; underlines that the
    residency axis is the un-mined one.
11. **Asynchronous KV Cache Prefetching** (arXiv:2504.06319, Apr 2025) — uses
    `cp.async.bulk.prefetch.L2` to stage KV into L2 ahead of attention.
    Relevance: the only L2-residency-for-inference kernel work we found on the
    KV side; contrast with our finding that KV reads are not L2-limited, and
    with state (which is).
12. **Arithmetic-Intensity-Aware Quantization** (arXiv:2512.14090, Dec 2025) —
    chooses quantization by roofline arithmetic intensity. Relevance: the
    capacity-blind version of precision dispatch; exactly what the gate
    subsumes (AI-aware but capacity-blind → wrong below C_eff).

### BibTeX

```bibtex
@misc{yao2026kernelsight,
  title={KernelSight-LM: A Kernel-Level LLM Inference Simulator},
  author={Yao, Xiteng and Kim, Taeho and Pei, Hengzhi and others},
  year={2026}, eprint={2606.28565}, archivePrefix={arXiv}, primaryClass={cs.PF}
}
@misc{blackwell2025microbench,
  title={Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis},
  year={2025}, eprint={2512.02189}, archivePrefix={arXiv}, primaryClass={cs.AR},
  note={B200: 148 SMs, four L2 partitions, TMEM; L2 hit-rate vs precision}
}
@misc{dissecting2025blackwell,
  title={Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks},
  year={2025}, eprint={2507.10789}, archivePrefix={arXiv}, primaryClass={cs.AR}
}
@misc{jarmusch2026analytical,
  title={Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures},
  author={Jarmusch, Aaron and Chandrasekaran, Sunita},
  year={2026}, eprint={2605.04178}, archivePrefix={arXiv}, primaryClass={cs.DC}
}
@misc{zhang2026cacheresident,
  title={Cache-Resident LLM Inference in GB-Scale Last-Level Caches},
  author={Zhang, Wanning and Gu, Tongzhou and Canini, Marco and Xu, Ceyu and Weng, Jian},
  year={2026}, eprint={2606.25353}, archivePrefix={arXiv}, primaryClass={cs.AR}
}
@misc{dao2026replayssm,
  title={ReplaySSM: Cache SSM Inputs, Not State},
  author={Dao, Tri and others},
  year={2026}, month={jun},
  howpublished={\url{https://tridao.me/blog/2026/replayssm/}},
  note={vLLM RFC \#47572; 1.43--1.84x GDN/Mamba2 decode kernel speedup on H100/B300}
}
@misc{gupta2026persistentstate,
  title={A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA},
  author={Gupta, Neelesh and Wang, Peter and Kannan, Rajgopal and Prasanna, Viktor K.},
  year={2026}, eprint={2603.05931}, archivePrefix={arXiv}, primaryClass={cs.AR}
}
@misc{wang2026specla,
  title={SpecLA: Efficient Speculative Decoding for Linear-Attention Models},
  author={Wang and Han and Yang and Liu and Li and Gu and Zhong and Tian},
  year={2026}, eprint={2607.16673}, archivePrefix={arXiv}, primaryClass={cs.LG},
  note={Author list to be completed from the arXiv page before camera-ready}
}
@misc{nvidia2026gdn2,
  title={Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention},
  year={2026}, eprint={2605.22791}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
@misc{automegakernel2026,
  title={AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis},
  year={2026}, eprint={2606.09682}, archivePrefix={arXiv}, primaryClass={cs.DC}
}
@misc{kvprefetch2025,
  title={Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching},
  year={2025}, eprint={2504.06319}, archivePrefix={arXiv}, primaryClass={cs.DC},
  note={cp.async.bulk.prefetch.L2 staging of KV into L2}
}
@misc{aiaq2025,
  title={Arithmetic-Intensity-Aware Quantization},
  year={2025}, eprint={2512.14090}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
@misc{flashqla2026,
  title={FlashQLA: Fused GPU Kernels for Gated DeltaNet},
  author={{Qwen Team}},
  year={2026}, month={apr},
  howpublished={\url{https://github.com/QwenLM/FlashQLA}},
  note={Chunked prefill/training kernels, SM90/SM100/SM103/SM120; no decode kernel}
}
```

Caveat: author lists for entries marked incomplete were not fully resolvable
from fetched pages; whoever integrates should pull exact lists from the arXiv
abs pages (30 seconds each) before the bibliography ships.

---

## 3. Venue scan (dates verified 2026-08-03)

| Venue | Deadline (abstract / full) | Pages | AE | Fit for a measurement-heavy, no-e2e-system paper |
|---|---|---|---|---|
| **HPCA 2027** | July 24 / **July 31 2026 — PASSED** | — | — | Was decent; missed by 3 days. Salt Lake City, Mar 20–24 2027. |
| **ASPLOS 2027 (Sept cycle)** | **Sept 9 2026** (no separate abstract) | 11 pp + refs | Traditional, post-accept | Good on paper — CFP *explicitly* welcomes characterization / "understanding existing systems" papers. Risk: architecture-systems PC may ask "where's the system?"; 5 weeks to compress 17pp→11pp double-column is brutal. |
| **EuroSys 2027 (fall)** | Sept 17 / Sept 24 2026 | ~12 pp typical | Yes | Weakest fit: EuroSys wants built systems; a measurement+model paper without an e2e serving artifact reads out-of-scope there. Rabat, Apr 19–24 2027. |
| **MLSys 2027** | **Oct 30 2026** | 10 pp + refs, unlimited appendix | Voluntary, doesn't affect decision | **Best fit.** Audience is exactly kernels + quantization + serving; CFP explicitly includes benchmarks/measurement/tooling; MARLIN, QServe, FlashInfer-line papers live here; the gate-as-design-tool section is a native MLSys story. Bellevue, May 17–22 2027; rebuttal Jan 12–16, notify Jan 26 2027. |
| **ISCA 2027** | Nov 10 / Nov 17 2026 (2nd deadline Dec 12) | ~11 pp | Yes | Possible but risky: ISCA rewards architectural novelty/proposals; a measured model of existing silicon usually gets "nice characterization, no architecture" reviews. The effective-vs-nominal-capacity finding alone won't carry it. Atlanta, Jun 5–9 2027. |
| **ISPASS 2027** | Not yet announced; pattern says ~Dec 2026 (2026 cycle: abs Dec 8 / full Dec 15 2025) | ~11 pp | Light | **Natural home by charter** (performance analysis of systems and software) and would love this paper — but it's a Rank-B venue; use as safety, not target. |
| USENIX ATC | — | — | — | **The series ended in 2025.** Remove from consideration. |

### Recommendation

- **Primary: MLSys 2027, deadline Oct 30 2026.** Three months of runway, the
  right audience (the people who wrote MARLIN/QServe/fla will review it), a
  10-page format that *forces* the compression the paper needs anyway, and the
  kernel-design-tool result is a first-class MLSys contribution rather than an
  oddity. Voluntary AE with the repo's committed-JSON discipline is a badge
  nearly for free.
- **Backup: ISPASS 2027 (~Dec 2026)** — near-certain fit if MLSys bounces, and
  the timeline chains perfectly (MLSys notify Jan 26 is after the likely ISPASS
  deadline, so the real chain is MLSys → ISCA-2nd-deadline Dec 12 or ISPASS if
  we're willing to decide before notification; otherwise MLSys → EuroSys'28
  spring / HPCA'28). If the director wants a *fast* shot and can stomach a
  5-week compression, ASPLOS Sept 9 is the only earlier door, and I'd advise
  against it: the paper would go in under-polished against a PC with weaker
  affinity.
- **What the paper still lacks for MLSys**: (1) an answer to ReplaySSM in the
  GDN section (cite + orthogonality argument at minimum; a composed or
  head-to-head measurement is the strong version); (2) at least one
  engine-level number for the GDN kernel (even a single vLLM/SGLang
  decode-throughput A/B at B inside vs outside the window would immunize the
  "kernel microbenchmark only" review); (3) compression 17pp→10pp — the
  case-study material and part of the mechanisms taxonomy go to the appendix;
  (4) artifact packaging (one runbook that reproduces Fig-1, the transfer
  table, and the GDN window from the committed JSONs).

---

## 4. Split the kernel result out, or keep it in?

**Keep it in the paper as the design-tool section — do not split now.** The
kernel's scientific value is almost entirely *derivative of the model*: the gate
predicted the headroom (2.7x), its location (below C_eff), its close (B*≈40),
and its migration on unseen silicon (B300, ~92 MB) — pre-registered. Inside this
paper that is the payoff that upgrades CARM from "explains measurements" to
"tells you what to build"; standing alone it becomes "a 2x Triton kernel for GDN
decode," which then competes head-to-head with ReplaySSM (1.4–1.8x, Tri Dao, on
the same GPUs and models, with vLLM integration) and with production kernels
that carry varlen/beta-vector/head-dim generality ours lacks — a fight our
current artifact isn't equipped to win and doesn't need to fight. The right
split is temporal, not structural: paper section + fla issue/PR now; if the
upstream conversation goes well, a short companion (MLSys workshop or
FlashInfer-Bench submission, workshop deadlines will land ~Feb–Mar 2027) written
*with* the fla/ReplaySSM authors' baselines is the follow-on with much higher
expected value than a standalone kernel paper today.

---

## 5. Report-back summary

- **Prior-art verdict: CLEAR** for the specific claim (L2-capacity-gated,
  model-predicted residency window for linear-attention decode state; no
  anticipation found in papers, fla/vLLM trackers, FlashQLA, or FlashInfer).
  Contested neighborhood: FPGA accelerator (arXiv:2603.05931) claims GPU L2
  can't do it (we refute — cite prominently); **ReplaySSM (tridao.me, June
  2026, vLLM RFC #47572) is the must-answer baseline** — same models, same
  GPUs, 1.4–1.8x by orthogonal means; SpecLA (arXiv:2607.16673) is the SMEM
  cousin. fla tracker itself: nothing.
- **Top venue: MLSys 2027, deadline Oct 30 2026** (10 pp + refs, voluntary AE);
  backup ISPASS 2027 (~Dec). HPCA 2027 passed July 31; USENIX ATC no longer
  exists; ASPLOS Sept 9 possible but not advised.
- **One thing to fix before submission**: the GDN section and
  `UPSTREAM_fla_ISSUE_DRAFT.md` were written unaware of ReplaySSM — add the
  citation and orthogonality/composition argument (and ideally one measurement
  against or atop it) before the issue is posted or the paper goes out;
  as written, the "production kernels are residency-blind and this is the
  frontier" framing is attackable by anyone who has read Tri Dao's June post.
