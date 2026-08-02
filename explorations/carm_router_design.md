# CARM-informed serving router — design exploration

Date: 2026-08-02. Status: **exploration, no GPU work done for this document.**
Sources: `profiling/dispatch/`, `profiling/served/`, `JOURNAL_2026-08.md`
(sessions 2–8), `/lambda/nfs/robert-nfs/regime-router-project` (June 2026,
dormant), and August-2026 web state of vLLM / NVIDIA Dynamo / SGLang.

---

## (a) What exists

### a.1 The CARM + dispatch cost model (this repo)

**CARM proper.** A measured, capacity-gated roofline: parameters
`{C_eff, bw_l2, bw_hbm, peak_achieved, r_dequant, t0}` emitted by the portable
harness (`profiling/portable/measure_params.py`) on any torch.cuda backend.
Validated on three architectures — A100 (C_eff 31.2 MB), H100 (39.0), B200
(98.8); effective/nominal LLC ratio is 0.780±0.001 on all three. Held-out
zero-shot latency MAPE after the session-7/8 residual closure: H100 ~14/13%
below/above gate, A100 12.8/13.0%, B200 16.6/3.9% (zero-parameter band, no
fitted floor). The gate is a **footprint** gate (weights+act+out vs C_eff),
with a soft collapse band C_eff→C_hi confirmed in NCU DRAM counters. Per-kernel
predicates for five mechanisms (A–E: act-quant overhead, wave/band
quantization, contention step-function, dequant ceiling r_dequant, L2
residency/footprint) flip the win/loss sign of a precision choice per shape.

**P6 dispatch cost model** (`profiling/dispatch/cost_model.py`,
`results_cost_model.json`, journal session 2). Prices the *storage* problem of
runtime precision dispatch for a Qwen3.6-27B-class dense model, with one live
H100 measurement (repack rate 0.133 T elems/s unfused; fused bound
bw_hbm/3 = 1.09 T elems/s):

- **Policy A, dual-resident:** the second weight format is paid out of the KV
  budget → concurrency. H100 80 GB: 9 → 0 sequences = **infeasible**. H200
  141 GB: 36 → 24 = −33% decode throughput.
- **Policy B, repack-on-switch:** 27B repack = 203 ms measured (25 ms fused
  bound); ≤1% overhead needs a switch period ≥20 s (≥2.5 s fused). Feasible
  only at **engine-mode granularity** — never per-phase (~100 ms) or per-layer.
- **Policy C, JIT dequant via HBM scratch:** 4.5× traffic per elem-byte vs 2×
  resident bf16 → 2.25× worse, **never pays**. In-kernel JIT dequant is not a
  new option; it *is* the r_dequant ceiling CARM already prices.
- **Conclusion the numbers force:** **quantized-primary storage is the only
  zero-marginal-cost dispatch policy.** The dispatcher's question is not "when
  to quantize" but "when to pay the dequant ceiling vs take the
  quantized-compute path," per shape, via the CARM predicates.
- **KV-precision corollary** (session 3): fp8 KV is a **concurrency/memory
  lever, not a speed lever** (e2e decode-speed ceiling ≤0.2% on this profile;
  full attention 2.67% of runtime). It doubles max concurrency (H100 9→18,
  H200 36→73) and re-opens dual-residency headroom on large-memory parts.

So the current model can already price, without further measurement:

| Decision | Priced by | Status |
|---|---|---|
| Precision per layer/shape (bf16 / W8A8 / W4A16 / fp8) | capacity gate + mechanisms A–E + r_dequant | measured, 3 architectures |
| Storage policy (dual-resident / repack / JIT / quantized-primary) | P6 policies A–C | measured (H100), analytic elsewhere |
| Mode-switch granularity (per-layer / per-phase / engine-mode) | P6 policy B amortization | measured |
| KV dtype as a concurrency lever | P6 kv_scale + kv_serving/ | measured |
| Batch shaping (chunked-prefill token caps) | mechanism B bands + served band sweeps | measured, engine level |
| Prefill-vs-decode precision payoff | served A/B | measured, engine level |

### a.2 The served evidence (`profiling/served/`)

vLLM 0.20.2, Qwen3.6-27B hybrid-GDN, H100 80 GB, CUTLASS fp8 path:

- **Decode** (64 seqs × 256 gen tok): bf16 2025.5 → fp8 **2930.9 gen tok/s =
  1.447×** — bracketed in advance by the dense_qwen projection (1.19× unfused
  … ~1.6× fused). The model's number became a measurement.
- **Prefill** (~27k prompt tok): 12,563 → **15,096 total tok/s = 1.20×** —
  smaller win in the compute-bound regime, as CARM predicts.
- **Quality:** WikiText-2 PPL 7.5281 (bf16) → 7.5601 (fp8), **+0.032 PPL**;
  per-layer rel-err ≈ gaussian floor (accuracy is not the discriminator between
  these precisions on this checkpoint; speed is).
- **Batch shaping:** decode caps only hurt (2931 → 2718/2818). Prefill shows a
  reproducible **+24% step between adjacent caps 896→1024** (±0.4% across
  repeats), matching the kernel-level per-token sawtooth
  (`bench_prefill_band_bridge.py`: 1024 is a local minimum, 896 costs +17%).
  Operational rule: engine-level batch shaping never *gains* over uncapped;
  its value is **avoiding the loss holes**, which sit where the kernel band
  data says they should. (Also found: vLLM 0.20.2 + GDN hangs hard at cap
  2048 — a stack bug the sweep tripped.)

### a.3 The regime-router project (dormant, June 2026)

`/lambda/nfs/robert-nfs/regime-router-project` — 55 GB total, but almost all of
it is two full engine forks (`forks/vllm` on branch `regime-aware-fp8-skip`,
`forks/sglang` incl. the Rust `sgl-model-gateway`), two venvs, and an HF cache.
The project itself is `regime-router/`: ~1,800 lines of Python plus complete
writeups (README, RESULTS §1–10, FINDINGS, PROGRESS, BLOG, dashboard). State:
**finished and deliberately closed** (tag `final-h100-study`, handoff doc
`docs/SESSION_STATE_2026-06-21.md` says "COMPLETE — do not redo").

What it built (H100, Qwen3-8B):
- An OpenAI-compatible serving stack + load harness: `bench/loadgen.py`,
  `workloads.py` (chat/rag/code mix), `sweep.sh`, `soak.py` (1000/1000, 0 mem
  drift), quality gates (`quality_gate.py`, needle test), version-stamped
  baseline matrix vs current vLLM 0.23 + SGLang 0.5.13.
- A live routing proxy (`regime_router/router_server.py` + two-backend launch
  scripts) with a boot-time-calibrated cost model
  (`regime_router/cost_model.py`).
- **Findings, all measured:** (1) FP8 weights dominate the whole batch range
  on H100 (1.22–1.43× over bf16); W4A16 never beats FP8 → **no request-level
  weight-precision crossover**, the router routes 100%→FP8 and merely matches
  static FP8 (±3%). (2) KV-precision heterogeneity is real (long context under
  cache saturation prefers FP8-KV by +5.5%) but **unexploitable**: KV dtype is
  launch-global, so routing forces a two-backend memory split that
  self-inflicts the very pressure FP8-KV would relieve (router 0.84–0.99× of
  best static). (3) **Calibration is load-bearing:** an offline-seeded cost
  model routed 19% *slower* than bf16; a boot-time calibration sweep on the
  live GPU made the same router match the per-layer oracle (28-point swing),
  and a learned predictor did not beat the measured table. (4) A real SGLang
  correctness bug (w8a8_int8 silent garbage, PR sgl-project/sglang#28806) and
  an honest-negative vLLM regime-skip branch.

**Overlap/complementarity with a CARM-informed router:** it is the
*request-level precision* router, and its two clean negatives are exactly the
axes a new router must *not* re-attempt on Hopper. What it contributes is (i)
the reusable serving harness (loadgen, proxy skeleton, two-backend launch,
quality gates), (ii) the boot-calibration pattern — the serving-layer twin of
P5's "kernel terms must be measured," and (iii) the missing-primitive framing
(per-request precision/KV-dtype inside one full-memory engine). A
CARM-informed *phase/placement/shaping* router is a different decision axis —
where a request runs and how batches are shaped, not which precision a request
gets — so the June negatives do not block it, but they set the bar: any split
architecture must beat the best *static full-memory* configuration, because
the split itself costs memory headroom.

---

## (b) Gap analysis: from cost model to working router

What the project has is a **pricing function**; a router needs a **decision
loop** around it. The gaps, in dependency order:

1. **Online inputs.** CARM prices a (shape, precision, footprint) tuple; the
   serving layer must supply per-step M (decode batch), chunk size, prefix-hit
   fraction, and KV-pool occupancy. vLLM exposes these in the scheduler and in
   Prometheus metrics; nothing in the repo consumes them yet.
2. **Phase-level model.** The served A/B gives two points (decode 1.45×,
   prefill 1.20×). A router needs TTFT/ITL as a function of (ISL, OSL,
   concurrency, precision, cap) — i.e. the CARM GEMM model composed over the
   layer stack plus attention + overhead terms. The dense_qwen composition
   already did this offline for decode (its 1.19–1.6× bracket held); it needs
   the prefill leg and a validation against served TTFT/ITL, not just
   throughput. This is the main *modeling* gap.
3. **Aggregate footprint, not per-request footprint.** The capacity gate binds
   on the *launch's* total footprint (session 6). In a served engine the
   effective per-GEMM footprint is set by batch composition — so the gate is a
   property the **scheduler** controls (via token budget / batch makeup), not
   a property of a request. This is why the natural home of the gate is batch
   shaping and pool sizing, not request classification.
4. **A workload with real heterogeneity.** The June study died partly because
   the mixed chat/rag/code workload had no crossover. Prefix-cache-heavy
   agentic traces (shared system prompts, tool loops, high ISL variance) give
   the P/D ratio and prefix-hit structure that make routing decisions
   non-trivial. Need a trace generator or a public trace (Mooncake-style).
5. **An integration point** (next section) and a baseline discipline:
   static-best single instance at full memory is the opponent, per the June
   lesson — never static-*worst*.
6. **Engineering hygiene:** current vLLM (0.2x line has moved), the 0.20.2 GDN
   hang at cap 2048, and the fact that none of this is wired to consume
   `carm_params.json` at boot.

## (c) The narrowest real integration points (August 2026)

Surveyed surface:

- **vLLM** disaggregated prefill/decode is driven entirely by
  `--kv-transfer-config` with pluggable connectors (`NixlConnector`,
  `LMCacheConnector`, `MooncakeConnector`); the *policy* of which instance gets
  a request lives **outside the engine** in whatever proxy/router fronts the
  instances. Separately, the v1 engine has a **pluggable scheduler**
  (`--scheduler-cls`, `vllm.v1.core.sched.interface.SchedulerInterface`) — a
  supported hook used in production by hardware plugins (e.g. vllm-spyre).
- **NVIDIA Dynamo 1.0** (GA March 2026): KV-aware router scores workers by
  predicted prefill cost (non-overlapped blocks) + decode cost (active
  blocks); the **SLA planner** scales P/D pools using *performance
  interpolators fitted to brute-force pre-deployment profiling grids*
  (`profile_sla`), with TTFT/ITL targets. That interpolator is precisely the
  slot a measured analytic model competes with: CARM's pitch is *predict the
  grid instead of sweeping it* (and predict the non-smooth parts — gate,
  bands — that interpolation misses).
- **SGLang**: engines take `--disaggregation-mode prefill|decode`; the Rust
  `sgl-model-gateway` exposes `--policy` / `--prefill-policy` /
  `--decode-policy` behind a trait + factory
  (`sgl-model-gateway/src/policies/{cache_aware,power_of_two,...}.rs`) —
  adding a policy is implementing one Rust trait, but the useful signals
  (queue depth, cache overlap) are already consumed by `cache_aware`.

**Narrowest viable point for a research prototype: a CARM-informed policy in
an external OpenAI-compatible router in front of two same-GPU vLLM instances —
the regime-router proxy skeleton, re-targeted.** Zero engine forks, engine
version pinned but swappable, and the decision loop is ~100 lines around the
existing cost model. **Second, in-engine point: a vLLM v1 `--scheduler-cls`
subclass** that sets the chunked-prefill token budget from the mechanism-B
band table (avoid the measured −17…−28% loss-hole caps, sit on the sawtooth
minima) — a supported plugin hook, single process, and it monetizes an
already-measured +24% effect. The **Dynamo SLA-planner interpolator** is the
highest-leverage upstream target ("CARM replaces the profiling sweep") but is
multi-node infrastructure — an end-of-project demo, not the prototype.

### Single-H100 prototype plan

Emulated disaggregation on one GPU — explicitly an emulation for *policy*
research, stated as such:

1. **Two processes, split memory** (reuse
   `regime-router/scripts/launch_two_backends.sh` pattern): a prefill-role
   vLLM instance (fp8, chunked prefill, CARM-chosen token cap from the band
   table) and a decode-role instance (fp8 weights + fp8-KV — P6 says KV
   precision is the decode pool's concurrency lever), `gpu_mem_util`
   ~0.42/0.42. KV handoff v0: prefix-cache replay (decode instance re-prefills
   with APC on) so no connector work blocks the loop; v1: intra-node
   NixlConnector/LMCache if the replay tax obscures the signal. A
   two-CUDA-stream variant is *not* worth it — no isolation, same-process
   scheduler fights.
2. **Router = CARM decision loop** in the proxy: per request, estimate
   non-cached prefill tokens (prefix-hash against a running radix of served
   prompts); price prefill time via the phase-level CARM (compute-bound leg,
   band-aware cap) and decode occupancy via KV bytes/token vs the decode
   pool's budget; admit to {prefill pool, decode-direct (high prefix hit),
   queue} and set per-batch token budgets. All constants from
   `carm_params.json` + a boot-time calibration sweep (the June 28-point
   lesson, mandatory).
3. **Workload:** prefix-cache-heavy agentic traces — multi-turn tool-call
   sessions with a shared 2–8k system prompt, ISL log-normal to ~27k, short
   decode bursts; plus the adversarial mix (band-hole ISLs) where the model
   predicts static caps lose.
4. **Baselines:** (i) single static fp8 instance, full memory, uncapped,
   APC on — *the honest opponent*; (ii) same, round-robin over two split
   instances (isolates the split tax from the policy); (iii) cache-aware
   round-robin (Dynamo/sgl-gateway default behavior proxy).
5. **Metrics + falsifiable predictions, pre-registered:** TTFT/ITL p50/p99,
   goodput under SLO, and — the differentiator — **predicted vs measured**
   phase latencies per decision. The prototype wins if CARM's predicted regime
   boundaries (cap loss-holes, KV-saturation onset, prefill/decode crossover)
   match measured boundaries within stated error, *even where the end-to-end
   throughput win is small*. Success criterion for the throughput leg: beat
   baseline (iii) on the agentic trace and never lose >3% to baseline (i) —
   the June study shows losing to (i) is the default outcome of splits, so
   equal-throughput-with-predictive-control is a reportable result; a win
   likely requires the trace's KV-pressure regime, which the fp8-KV decode
   pool is positioned to absorb.

## (d) Honest assessment

**The publishable claim (4–6 months), if it works:** *"A portable, measured,
capacity-gated cost model — parameterized in ~an hour of microbenchmarks per
GPU, transferring across A100/H100/B200 — can drive disaggregated-serving
control decisions (P/D pool shaping, batch token budgets, precision/KV-dtype
per pool) as well as brute-force pre-deployment profiling, and predicts the
regime boundaries (LLC capacity gate, wave-band loss holes, KV-saturation
onset) that black-box interpolation cannot."* Dynamo's SLA planner and the
autoscaling literature interpolate profiled grids per (model, GPU, parallel
config) — they do not model *why* the surface bends, cannot extrapolate to an
unprofiled config, and are blind to non-smooth structure (the 896→1024 +24%
step is invisible to a coarse profiling grid). Measured capacity-gate
awareness plus the P6 storage-policy analysis (quantized-primary is the only
free dispatch; fp8-KV is a concurrency lever) is genuinely not in the
Dynamo/vLLM papers. The weaker but safer fallback paper is "prediction
replaces profiling": score CARM against Dynamo's own `profile_sla` grids on
2–3 GPUs — no router needed at all.

**The risks, plainly:**
- **The June precedent.** The last router built on this NFS discovered that
  the routable heterogeneity, though real, was not capturable, and shipped a
  characterization instead. Phase-level routing has more room than
  request-level precision routing (P/D disaggregation demonstrably pays at
  scale — that's Dynamo's existence proof), but on *one* emulated GPU the
  split tax (measured 1–16% in June) may again eat the policy win. The
  prototype must be judged on predictive control quality, with throughput
  parity as the floor — and that framing must be pre-registered, not adopted
  after a negative.
- **Engineering-heavy, fast-moving baselines.** vLLM shipped three minor
  versions during this project's lifetime; Dynamo went 0.4→1.0 in under a
  year and could ship model-based interpolators itself. Every week spent on
  KV-connector plumbing is a week competing with NVIDIA's paid staff on their
  home turf. Mitigation: stay in the external-proxy + `--scheduler-cls` tier,
  never fork an engine (the June forks are 50 GB of dead weight on this NFS).
- **Emulation credibility.** Reviewers at MLSys/serving venues will ask why
  single-GPU emulated disaggregation predicts multi-node behavior. The honest
  answer — the *model* is the contribution and it transfers because its
  parameters are measured per GPU — must carry the paper; budget for at least
  one small multi-GPU validation run late.
- **Scope creep is the real killer.** The project's strength is
  measurement-first physics with pre-registered predictions; a router is a
  systems artifact judged on end-to-end wins against sprinting baselines.

**One-paragraph verdict:** worth pursuing **only in the narrow form**: CARM as
the *measured brain* inside the thinnest possible control shim (external proxy
+ scheduler plugin), evaluated on prediction quality first and throughput
second, with the "replace Dynamo's profiling sweep" comparison as the anchored,
router-free fallback. Not worth pursuing as a "beat vLLM/Dynamo end-to-end"
systems project — that fight is engineering-bound, the baselines move faster
than one researcher, and the June study already demonstrated on this exact
hardware how routing projects collapse into characterizations. The capacity
gate, the band holes, the storage-policy exclusion, and the three-architecture
transfer are assets no serving paper currently has; the router should exist
only insofar as it *demonstrates* them end-to-end, and the plan above is sized
so that a negative routing result still yields the fallback paper.

---

### References (integration surface, checked 2026-08-02)

- vLLM disaggregated prefill + KV connectors: https://docs.vllm.ai/en/stable/features/disagg_prefill/
- vLLM v1 pluggable scheduler (`--scheduler-cls`): https://github.com/vllm-project/vllm/pull/14466 ; https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- NVIDIA Dynamo 1.0 — KV-aware router: https://docs.nvidia.com/dynamo/user-guides/kv-cache-aware-routing ; SLA planner + pre-deployment profiling: https://docs.nvidia.com/dynamo/v-0-8-1/components/planner/sla-based-planner , https://github.com/ai-dynamo/dynamo/blob/main/docs/planner/sla_planner.md
- SGLang PD disaggregation + model gateway policies: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md ; https://docs.sglang.io/advanced_features/router.html
- NIXL / LMCache / Mooncake connectors: https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/ ; https://docs.lmcache.ai/mp/disaggregated_prefill.html ; https://kvcache-ai.github.io/Mooncake/
