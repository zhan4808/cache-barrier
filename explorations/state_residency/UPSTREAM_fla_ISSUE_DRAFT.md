# Draft issue for github.com/fla-org/flash-linear-attention

Ready to post after your review. Suggested title:

> `fused_recurrent_gated_delta_rule` decode is L2-blind on H100 (~2.3 TB/s
> flat); fused single-pass step reaches 2.0–2.3× at cache-resident batch
> sizes — measurements + prototype attached

---

## Body

**Setup.** H100 80GB (SXM, clock cap `-lgc 1755`, sustained ~1380–1485 MHz
under load, sampled and recorded per run), fla 0.5.2, torch 2.7.0+cu126,
triton 3.3.0. All timings CUDA-graph medians (10 launches/graph, 30
replays). Decode shapes: `H=16, dk=dv=128`, fp32 state (64 KB/head, 1
MB/request), batch swept so total recurrent-state footprint crosses the
H100 L2 (measured effective residency capacity: ~40 MB, established with a
0.5 MB-step warm re-read sweep).

**Observation 1 — the decode kernel does not exploit L2 residency.**
`fused_recurrent_gated_delta_rule` at T=1 runs at a flat ~2.3–2.4 TB/s of
state traffic (2× footprint per step) from 8 MB to 160 MB of total state —
below H100's HBM streaming rate (~3.15 TB/s measured). Warm-vs-evicted
deltas are ≤13% and only below 16 MB. Since decode re-reads and rewrites
the full state every step, batches whose state fits in L2 (B ≤ ~40 at this
geometry) could in principle be served at L2 bandwidth, not HBM-or-below.

**Observation 2 — a single-pass step captures most of that headroom.** A
~100-line Triton prototype doing the whole decode step in one program per
(batch, head) — short-conv cache update + silu, qk l2norm, gated
delta-rule state update, gated-RMSNorm output — reaches 3.4–4.4 TB/s warm
below the residency limit. Against the practical decode chain
(`ShortConvolution.step` ×3 → `fused_recurrent_gated_delta_rule
(use_qk_l2norm_in_kernel=True)` → `FusedRMSNormGated`), both sides
graph-timed:

| total state | chain | fused single-pass | speedup |
|---|---|---|---|
| 8 MB (B=8) | 14.0 µs | 7.0 µs | 2.00× |
| 16 MB (B=16) | 25.0 µs | 10.7 µs | 2.34× |
| 32 MB (B=32) | 43.4 µs | 23.3 µs | 1.86× |
| 48 MB (B=48) | 57.6 µs | 44.3 µs | 1.30× |
| 96 MB (B=96) | 99.4 µs | 82.0 µs | 1.21× |

The ~1.2–1.3× above 40 MB is the kernel-fusion dividend (launch count +
small-tensor round trips); the ~2× below is fusion × L2 residency. The
crossover at ~40 MB total state matches the measured H100 effective L2
capacity, so the win region is predictable per GPU: `B* ≈ C_eff / (H · dk
· dv · 4B)`. On B200 we measure C_eff ≈ 99 MB, so the window is ~2.5×
wider there.

**Scope/caveats, honestly:** prototype supports fixed headdim 128, scalar
beta, no varlen/cu_seqlens; correctness 1e-8 vs a reference implementation
of the same formulas; conv/norm run fp32 in the prototype vs bf16 in the
chain (state traffic dominates both, so this is second-order). This is a
decode-only claim — chunked prefill is a different regime (matmul-form
chunking → tensor cores) where this approach does not apply.

**Question/offer.** Is there interest in (a) an L2-aware decode path
(e.g., selecting a fused single-pass step when `B·H·dk·dv·4 < C_eff`), or
(b) the benchmark harness as a repro? Happy to PR the prototype +
benchmarks — both are self-contained scripts:
`results_fla_gdn_h100.json`, `results_gdn_full_h100.json`,
`bench_fla_gdn.py`, `gdn_l2_kernel.py`, `gdn_l2_kernel_full.py`
(measurement methodology: CUDA-graph medians, warm vs rotated-copies
controls, clock sampled under load).

---

## Posting notes (for Robert, not part of the issue)

- Repo: `fla-org/flash-linear-attention` → New issue.
- Attach or link the two JSONs + three scripts (public repo:
  `zhan4808/cache-barrier`, `explorations/state_residency/`).
- The C_eff/B* framing is our paper's language; the issue keeps it
  hardware-factual so it stands alone. If maintainers engage, the PR is
  the adoption vector and the paper citation follows naturally.
