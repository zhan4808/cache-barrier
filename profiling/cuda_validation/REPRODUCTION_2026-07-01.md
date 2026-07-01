# H100 baseline reproduction — fresh instance (2026-07-01)

Re-ran the full 2026-06-19 CUDA validation on a **fresh H100 80GB instance**
(driver 580.105.08, byte-exact vLLM 0.20.2 venv restored from NFS) to lock the
baseline before the B200/B100 leg. **Every load-bearing claim reproduces within
noise.** Repro JSONs saved alongside the committed references as
`*_repro_2026-07-01.json` (committed references left untouched).

| Instance | GPU | driver | vLLM | torch | env |
|---|---|---|---|---|---|
| capture (2026-06-19) | H100 80GB HBM3 | 580.105.08 | 0.20.2 | 2.11.0+cu130 | — |
| repro (2026-07-01) | H100 80GB HBM3 | 580.105.08 | 0.20.2 | 2.11.0+cu130 | byte-exact copy imports directly (driver match) |

## Exp A — MoE fp8/mxfp4 vs bf16 (default boost clocks)
Load-bearing FP8 path reproduces within **±0.7% at every T**. Max deviation
across all ratios 2.9%, both outliers on the *emulated* mxfp4 path at
noise-sensitive points (T=16 ≈296µs; T=640 in the documented ragged transition
band). Correctness rel-errs identical (fp8 0.0057, mxfp4 0.0038).

| T | fp8×bf16 ref → repro | mxfp4-EMU×bf16 ref → repro |
|---:|---|---|
| 16 | 1.898 → 1.890 | 3.218 → 3.140 |
| 256 | 1.368 → 1.371 | 1.338 → 1.353 |
| 1024 | 0.783 → 0.784 | 0.790 → 0.790 |
| 2048 | 0.593 → 0.593 | 0.597 → 0.593 |

## Exp B — FlashMLA bf16-KV vs fp8-KV (default boost clocks)
Both load-bearing signals hold: low-batch ≈3.1× and the **high-batch washout**
(B=32: 0.97–0.99×, within ±1.3%). Single 3.1% point (C=4096,B=1) is a low-batch
cell already flagged as two-kernel-dispatch-confounded.

## Exp C — CARM refit (CPU)
MAPE **identical**: bf16 22.3%, fp8 W8A16 **12.2%**. Base params reproduce
(HBM 3.121 TB/s, floor 2.765 µs; L2 fit 5.87 vs 5.62 TB/s — not load-bearing).
Crossover outputs reproduce: H100 measured T*≈602, roofline 334; B200-projected
W4A4 "wins everywhere".

## Red-team (SM clock locked to 1755 MHz, rotated median of 3)
| claim | reference | repro 2026-07-01 |
|---|---|---|
| MoE crossover vs stock bf16 | T* = 601 | **T* = 600** |
| MoE crossover vs tuned bf16 (G=16) | T* = 263 | **T* = 259** |
| small-T control def/g16 (T≤128) | 0.995–0.997 | 0.995–0.998 |
| small-T fp8 win | ~1.8× | 1.77–1.88× |
| MLA fp8-KV vs batch | 3.12→1.39→0.96→0.71 | 3.07→1.39→0.96→0.73→0.75 |
| MLA fp8 KV BW ceiling | ~0.8 vs bf16 1.9 TB/s | 0.80–0.82 vs 1.92 TB/s |
| DeepSeek-V4-Flash crossover | none (fp8 wins to 2048) | **none** (1.42–1.96×) |
| Qwen3.6-35B crossover (stock/tuned) | 1908 / 1904 | **1920 / 1915** |

**Verdict: baseline locked.** The fresh instance reproduces the H100 headline
numbers within measurement noise; these repro JSONs are the reference the
B200/B100 numbers will be diffed against.
