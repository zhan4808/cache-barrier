# Dense Qwen3.6-27B mixed-precision crossover — results (2026-07-01, H100)

Dr. Xiao todo #1: *"Verify whether mixed-precision operators can improve performance
on dense models, such as Qwen3.6-27B, and determine whether they are constrained by
cache size."* Layer-level, real config shapes, CUDA-graph timed (`bench_dense_proj.py`);
paths: bf16 `torch.mm` (same cuBLAS nvjet path as Dr. Xiao's 86.2%-of-runtime profile
bucket), Marlin fp8 **W8A16** (weight-only), native cutlass **W8A8** (dynamic act quant
included; mm-only also recorded). Two modes: **warm** (microbenchmark residency) and
**rotated** (>2×C_eff of weight copies cycled per graph — serving-like eviction).

## Headline: all three CARM regimes appear on this one dense model

**Effective weight-serving bandwidth at M=1 (MB/(t−t₀)) — the capacity-gate, directly:**

| shape | bf16 MB / fp8 MB | bf16 warm | fp8 warm | bf16 rot | fp8 rot |
|---|---|---:|---:|---:|---:|
| kv_proj | 21 / 10 (both < C_eff) | **3.89** | 2.48 | 2.41 | 2.29 |
| q_proj | 63 / 32 (fp8 crosses in) | 2.87 | **4.69** | 2.80 | 2.70 |
| o_proj | 63 / 32 (fp8 crosses in) | 2.77 | **4.17** | 2.83 | 2.65 |
| down_proj | 178 / 89 (both out) | 2.75 | 2.80 | 2.75 | 2.80 |
| gate_up | 357 / 178 (both out) | 2.84 | 3.04 | 2.78 | 3.04 |

L2 tier ≈ 4–6 TB/s, HBM tier ≈ 2.8 TB/s. Warm: whichever operand fits under
C_eff≈36 MB gets the L2 tier (bf16 for kv_proj; fp8 for q/o_proj). Rotated:
**every entry collapses to the HBM tier** — the capacity gate is erased by
serving-like eviction. `down/gate_up` are tier-invariant in both modes
(consistency control).

## Per-regime findings

**Left edge (kv_proj, 21 MB, L2-resident bf16).** Warm: quant loses at every M
(W8A16 0.38–0.90×, W8A8 0.48–0.82×) — the original L2-residency claim, confirmed
on a real dense shape. **Rotated: the verdict flips at decode scale** (W8A16
1.11–1.17× at M≤16): once eviction makes the weight HBM-streamed, byte-halving
pays again. → The "quant doesn't pay when weights fit L2" rule holds only as far
as the serving pattern actually keeps weights resident.

**Boundary (q/o_proj, 63 MB bf16 → 31 MB fp8).** The cache-aware prediction a
plain roofline cannot make: fp8 pulls the operand *into* L2 →
**super-proportional win, 2.5–2.6× (mm-only, M≤16) vs the 2.0× byte ratio.**
Under rotation it compresses to 1.7–1.75× (sub-proportional; residency bonus
gone). W8A16 Marlin gets exactly the proportional 2.0× (its dequant path doesn't
exploit the L2 tier).

**HBM-streamed middle (down/gate_up, 178–357 MB).** Proportional small-M wins
(W8A16 1.7–1.85×, W8A8-mm ~1.9×), rotation-invariant, as predicted.

**Right edge (compute-bound).** W8A16 crosses over and loses at **M≈64–128 on
every dense shape** (0.38–0.55× by M≥256) — the dequant ceiling exists on dense
GEMMs too, *earlier* than the MoE crossover (~300). **W8A8 (native MMA) has no
such cliff**: it wins the high-M regime everywhere (1.24–1.70× at M≥1024) —
matched precision beats the ceiling (todo #3's W8A8 answer, on dense).

## Deployment costs measured (previously model gaps)

- **Dynamic act-quant overhead ~7–9 µs flat** is first-order at decode scale:
  deployed W8A8 loses to W8A16 at small M on streamed shapes despite the better
  GEMM (e.g. down_proj M=1: mm-only 1.96× → deployed 0.98×). Fusing/amortizing
  the quant is the fix; the model now has a measured term for it.
- Accuracy: W8A16 rel-err 0.0025; W8A8 per-tensor dynamic **0.0375** (needs
  per-channel/static scales in practice — flag for the accuracy-loss gap).

## One-line summary for Dr. Xiao

Dense Qwen3.6-27B shows the full cache-aware structure: quant loses where
weights are L2-resident, wins super-proportionally where fp8 pulls the weight
into L2 (2.6× > byte-ratio 2×), wins proportionally where streamed, and
weight-only quant dies at the compute edge (M≳128) while native W8A8 does not —
**and the L2-capacity effects are real but compress toward plain byte-ratio
behavior under serving-like multi-layer eviction** (rotated mode), which is the
honest answer to "are real services constrained by L2 capacity" for weights.

Files: `bench_dense_proj.py`, `results_dense_proj_h100.json`,
`results_dense_proj_h100_rotated.json`.
