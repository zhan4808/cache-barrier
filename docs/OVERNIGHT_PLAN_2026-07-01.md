# Overnight autoresearch plan — 2026-07-01 (~16 h unattended)

Operator: Claude (orchestrator) while Robert sleeps. Mission: execute the
two-week plan's Week-1/2 experiments on this H100, in priority order, with
red-team verification, committing after each milestone. Meeting-driver:
Dr. Xiao's three todos (dense ✅ done today; KV-serving; W8A8/W4A4) + folding
measured terms into CARM.

## Standing rules
- **GPU work strictly serialized** — one benchmark process at a time (CUDA-graph
  timing is contention-sensitive). Subagents may do code-reading/analysis/writing
  in parallel, never GPU runs.
- Env: `~/vllm-env/bin/python`, `VLLM_LOGGING_LEVEL=WARNING`. Repo on NFS at
  `~/robert-nfs/cache-barrier-project/repos/cache-barrier` (commit locally after
  each milestone; NO push — no key). kernel-compass sibling for carm.py.
- Clock-lock (`sudo nvidia-smi -lgc 1755,1755`) ONLY for red-team passes; always
  `-rgc` reset after. Default boost for main sweeps (matches existing method).
- Log every iteration to `docs/OVERNIGHT_LOG_2026-07-01.md`: timestamp, what ran,
  headline numbers, anomalies, next intent. Update the checklist below in place.
- If something breaks: record, park, move to next item. Never leave the GPU
  clock locked or the repo in a half-committed state.
- Honesty rules: label EMU vs native; note act-quant inclusion; per-tensor vs
  per-channel scales; warm vs rotated mode on every number.

## Work queue (execute in order; check off in place)

### 1. KV-cache quantization under serving conditions (Dr. Xiao todo #2) — [ ]
Qwen3.6-27B full-attention shape (24 q-heads / 4 kv-heads × 256, FA3 path).
- Bench decode attention bf16-KV vs fp8-KV across batch {1,8,32,128} × context
  {1k,8k,32k}; include `reshape_and_cache_flash` (quant cost) in a variant.
- Compute per-step KV working set vs C_eff=36 MB; identify any L2-resident cells
  (prediction: only tiny batch×context; mostly HBM-streamed).
- Multi-layer contention variant (rotate KV buffers like dense rotation).
- Amdahl framing vs Dr. Xiao's profile: full-attn = 2.67% of runtime → report
  best-case end-to-end gain from KV quant on this model.
- Deliverable: `profiling/kv_serving/bench_kv_decode.py` + results JSON + RESULTS.md section. Commit.

### 2. CARM v2: fold measured terms into the model — [ ]
In kernel-compass `profiling/carm.py` + a fit script in cache-barrier:
- Operand-aware capacity gate: regime keyed off (operand, size, residency mode).
- New terms: act-quant fixed cost (~7–9 µs, fit from dense data), dequant
  ceiling for dense Marlin (fit), contention factor (warm→rotated compression).
- Fit on `dense_qwen` data (both modes); report MAPE per path; predicted vs
  measured W8A16 crossovers (expect M≈64–128) and the q_proj super-win ratio.
- Deliverable: carm.py update + `profiling/dense_qwen/carm_dense_fit.py` +
  MAPE table in RESULTS.md. Commit both repos.

### 3. Red-team the two headline dense claims (clock-locked) — [ ]
- (a) q/o_proj super-proportional win (warm 2.5–2.6× mm-only vs 2.0× byte
  ratio): lock clocks, rotated measurement order, median of 3; confirm the
  effective-BW tier jump (≈4.2–4.7 vs ≈2.8 TB/s).
- (b) kv_proj rotation flip (loses warm → W8A16 wins 1.1–1.2× rotated at M≤16).
- Also sanity: kv_proj w8a16 M=64 warm anomaly (0.53× — config switch?).
- Deliverable: `verify_dense_claims.py` + results. Reset clocks. Commit.

### 4. W8A8 on the MoE CUDA path (task #4 / todo #3 completion) — [ ]
Matched-precision fp8 MoE (cutlass path or vLLM w8a8 triton MoE) at the Mixtral
shape + one fine-grained target shape; compare vs bf16 and vs Marlin W8A16.
Prediction: no dequant cliff; wins or ties at high T where W8A16 loses.
- Deliverable: `profiling/cuda_validation/bench_moe_w8a8.py` + results. Commit.

### 5. Hybrid coverage: GDN linear-attention projections — [ ]
Qwen3.6-27B linear-attn layers (3.7% bucket): projection weight sizes from
config (16 k-heads/48 v-heads × 128); quick M-sweep bf16 vs W8A16 vs W8A8,
warm+rotated. Do the L2 regimes appear in the hybrid block?
- Deliverable: extend dense_qwen bench with GDN shapes. Commit.

### 6. Fine-grained L2 boundary sweep (strengthens the capacity-gate fit) — [ ]
Synthetic square-ish GEMMs sweeping weight size 8→128 MB in ~8 steps across
C_eff (like the old MLA sweep but with all three paths + rotation). Pins the
cliff location and the tier BWs for CARM v2.
- Deliverable: `profiling/dense_qwen/bench_l2_boundary.py` + results. Commit.

### 7. Synthesis for Robert + Dr. Xiao — [ ]
- Update `dense_qwen/RESULTS.md` + write `docs/WEEK1_REPORT_2026-07-02.md`:
  findings vs the three todos, CARM v2 MAPE, honest caveats, proposed next deck
  changes (2–3 new slides: dense three-regime figure, KV-serving answer,
  W8A8-vs-W8A16 ceiling contrast).
- Generate matplotlib figures for the money results (dense BW-tier table as a
  plot; crossover curves; KV results).
- Final commit; update claude-memory; leave a wake-up summary at the TOP of
  OVERNIGHT_LOG (TL;DR first).

### Stretch (only if queue done and time remains)
- lm_head (5120×248k ≈ 2.5 GB) decode GEMM — the extreme streamed case.
- Contention-degree sweep: 1..N weight copies → map the compression curve for
  the CARM contention factor.
- Per-channel W8A8 scales (relerr 0.0375 → ?) accuracy/latency tradeoff.

## Progress log pointer
See `docs/OVERNIGHT_LOG_2026-07-01.md` (created on first iteration).

## Night-extension queue (added iteration 7 — original queue complete)
### 8. Rigor: W8A8 MoE vs TUNED bf16, clock-locked — [ ]
The 3.15× @T=512 is partly the §7 stock-baseline artifact. Measure w8a8 vs
bf16(GROUP_SIZE_M=16) directly, clock-locked, T∈{128..2048}. Headline-grade.
### 9. Fused norm+quant: recover the deployed small-M W8A8 win — [ ]
vLLM ships fused rms_norm+dynamic-quant kernels; measure act-quant cost fused
vs standalone (8.4µs). If fused ≈ free, deployed W8A8 wins small-M too.
### 10. INT8 vs FP8 matched W8A8 (dense) — [ ]
cutlass_scaled_mm supports both; completes the native-precision matrix on H100.
### 11. End-to-end decode-step model (CPU) — [ ]
Combine measured per-op numbers with Dr. Xiao's profile call counts → predicted
full-model Qwen3.6-27B speedup from W8A8-ing the GEMM bucket (86.2%).
### 12. Deck additions draft (CPU) — [ ]
3 slides: capacity-gate figure, KV answer, W8A8-vs-W8A16. Separate file for review.
### 13. CARM v2 → kernel-compass integration (CPU) — [ ]
Port operand-aware gate + act-quant + step-contention into carm.py; commit.
