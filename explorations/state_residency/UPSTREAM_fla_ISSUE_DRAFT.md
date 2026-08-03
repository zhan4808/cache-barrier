# Draft issue for github.com/fla-org/flash-linear-attention

Ready to post after your review. Suggested title:

> `fused_recurrent_gated_delta_rule` decode is L2-blind on H100 and B300
> (~2.3 / ~4.6–5.2 TB/s flat vs residency); a fused single-pass step reaches
> 2.0–2.3× (H100) and up to 2.05× (B300) at cache-resident batch sizes —
> measurements + prototype attached

---

## Body

**Setup.** Two machines, same fla 0.5.2, same decode shapes (`H=16,
dk=dv=128`, fp32 state = 64 KB/head, 1 MB/request), batch swept so total
recurrent-state footprint crosses each GPU's effective L2 residency
capacity (measured with a 0.5 MB-step warm re-read sweep: **~40 MB on
H100, ~92 MB on B300** — both well under nominal L2). All timings are
CUDA-graph medians (10 launches/graph, 30 replays), warm vs
rotated-copies (forced-eviction) controls.

- H100 80GB SXM: clock cap `-lgc 1755` (sustained ~1380–1485 MHz under
  load, sampled and recorded per run), torch 2.7.0+cu126, triton 3.3.0.
- B300 SXM6 (air-cooled): clock lock unavailable on this tier
  (power-limited; sustained clocks sampled under load and recorded),
  torch 2.13.0+cu130, triton 3.7.1.

**Observation 1 — the decode kernel does not exploit L2 residency, on
either architecture.** `fused_recurrent_gated_delta_rule` at T=1 runs at
a flat state-traffic rate (2× footprint per step) regardless of whether
the state is cache-resident:

- H100: ~2.3–2.4 TB/s from 8 to 160 MB of total state — below the HBM
  streaming rate (~3.15 TB/s measured). Warm-vs-evicted deltas ≤13%, and
  only below 16 MB.
- B300: ~4.6–5.2 TB/s from 16 to 160 MB — much better in absolute terms
  (newer triton + sm_103), but still residency-blind: warm-vs-evicted
  deltas ≤11%, gone by 48 MB, on a card whose measured effective L2
  capacity is ~92 MB and whose L2 bandwidth is ~2.5× its HBM rate.

Since decode re-reads and rewrites the full state every step, batches
whose state fits in effective L2 (B ≤ ~40 on H100, B ≤ ~92 on B300 at
this geometry) could in principle be served at L2 bandwidth, not
HBM-or-below.

**Observation 2 — a single-pass step captures most of that headroom.** A
~100-line Triton prototype doing the whole decode step in one program per
(batch, head) — short-conv cache update + silu, qk l2norm, gated
delta-rule state update, gated-RMSNorm output — reaches 3.4–4.4 TB/s warm
on H100 and 7.0–9.4 TB/s warm on B300 below each card's residency limit.

H100 — against the practical decode chain (`ShortConvolution.step` ×3 →
`fused_recurrent_gated_delta_rule(use_qk_l2norm_in_kernel=True)` →
`FusedRMSNormGated`), both sides graph-timed:

| total state | chain | fused single-pass | speedup |
|---|---|---|---|
| 8 MB (B=8) | 14.0 µs | 7.0 µs | 2.00× |
| 16 MB (B=16) | 25.0 µs | 10.7 µs | 2.34× |
| 32 MB (B=32) | 43.4 µs | 23.3 µs | 1.86× |
| 48 MB (B=48) | 57.6 µs | 44.3 µs | 1.30× |
| 96 MB (B=96) | 99.4 µs | 82.0 µs | 1.21× |

B300 — prototype recurrence step vs `fused_recurrent_gated_delta_rule`
alone (single-op comparison; the epilogue-complete chain comparison has
not been run on this box yet), both graph-timed warm:

| total state | fused_recurrent | prototype | speedup |
|---|---|---|---|
| 16 MB (B=16) | 7.3 µs | 4.8 µs | 1.51× |
| 32 MB (B=32) | 13.1 µs | 8.5 µs | 1.55× |
| 56 MB (B=56) | 25.6 µs | 12.5 µs | 2.05× |
| 96 MB (B=96) | 40.7 µs | 31.7 µs | 1.28× |
| 160 MB (B=160) | 64.3 µs | 54.3 µs | 1.18× |

On H100 the ~1.2–1.3× above 40 MB is the kernel-fusion dividend (launch
count + small-tensor round trips); the ~2× below is fusion × L2
residency. On B300 the residual ~1.2× far-field is bandwidth (the
prototype streams at 6.1–6.2 TB/s ≈ 92% of measured HBM rate). The key
point is that the crossover tracks each GPU's *measured* effective L2
capacity — ~40 MB on H100, ~92 MB on B300 — so the win region is
predictable per GPU: `B* ≈ C_eff / (H · dk · dv · 4B)` (≈40 requests on
H100, ≈92 on B300 at this geometry), and it widens as L2:HBM bandwidth
ratios grow across generations.

**Scope/caveats, honestly:** prototype supports fixed headdim 128, scalar
beta, no varlen/cu_seqlens; correctness 1e-8 vs a reference implementation
of the same formulas; conv/norm run fp32 in the prototype vs bf16 in the
chain (state traffic dominates both, so this is second-order). The B300
table compares the recurrence op only, not the full chain; the B300 box
could not be clock-locked (sustained clocks recorded per run). This is a
decode-only claim — chunked prefill is a different regime (matmul-form
chunking → tensor cores) where this approach does not apply.

**Relation to ReplaySSM** (Dao, June 2026; vLLM RFC #47572): ReplaySSM
attacks the same decode step by algorithmically halving state traffic
(cache inputs, recompute state) with no cache-hierarchy awareness — it is
orthogonal and composable with this observation. The residency window is
set by state *footprint*, which replay does not change, so a
replay-style kernel that is also L2-aware would stack both effects inside
the window (fewer bytes per step × cheaper bytes). Our NCU counters on
H100 show the current fused_recurrent misses 56% of its state traffic to
DRAM even when the state fits in L2 — that loss applies to replayed
traffic too, so the two fixes are complements, not competitors.

**Question/offer.** Is there interest in (a) an L2-aware decode path
(e.g., selecting a fused single-pass step when `B·H·dk·dv·4 < C_eff`), or
(b) the benchmark harness as a repro? Happy to PR the prototype +
benchmarks — all self-contained scripts:
`results_fla_gdn_h100.json`, `results_gdn_full_h100.json`,
`results_fla_gdn_b300.json`, `results_gdn_l2_kernel_b300.json`,
`bench_fla_gdn.py`, `gdn_l2_kernel.py`, `gdn_l2_kernel_full.py`
(measurement methodology: CUDA-graph medians, warm vs rotated-copies
controls, clock sampled under load).

---

## Posting notes (for Robert, not part of the issue)

- Repo: `fla-org/flash-linear-attention` → New issue.
- Attach or link the four JSONs + three scripts (public repo:
  `zhan4808/cache-barrier`, `explorations/state_residency/`).
- The C_eff/B* framing is our paper's language; the issue keeps it
  hardware-factual so it stands alone. If maintainers engage, the PR is
  the adoption vector and the paper citation follows naturally.
- The two-architecture version strengthens the ask: the same blindness on
  triton 3.3/sm_90 and 3.7/sm_103 means it is a kernel-structure issue,
  not a triton-version artifact — and the window is wider on newer
  silicon, so the payoff grows going forward.
