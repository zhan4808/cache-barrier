# AMD/ROCm portability — audit, hierarchy, and a 3-hour MI300X plan

*2026-08-02. Desk study only: code audit of `profiling/portable/measure_params.py`
and `cliff_finegrain.py` plus web research. Nothing here has been run on AMD
hardware — that is the point of the runbook in §4. The paper currently claims
ROCm portability "via hipify" (P5, `DIRECTION.md` §6); this memo is the honest
accounting of what that claim rests on.*

**TL;DR.** The harness will *run* on MI300X — every torch API it touches exists
on ROCm and CUDA-graph capture is production-exercised there by vLLM. But it
will run *wrong*: ROCm reports `L2_cache_size` = 4 MB (one XCD of eight), so
every sweep range derived from nominal L2 is scaled down ~8×, the C_eff sweep
(1.6–6 MB) never reaches the 32 MB aggregate L2 let alone the 256 MB Infinity
Cache, and the "bw_hbm" differential (24 vs 96 MB) lands entirely *inside* the
Infinity Cache — it would report ~10 TB/s of cache bandwidth as DRAM bandwidth.
One added override (nominal-LLC from CLI/env) plus one log-spaced full-range
sweep fixes all of it. Scientifically, MI300X is the best possible stress test
of the C_eff concept: a three-level hierarchy where the warm re-read curve
should show **two cliffs**, each independently testing the ≈0.8× ratio.

---

## 1. Portability audit — `measure_params.py` / `cliff_finegrain.py` on ROCm

Torch-on-ROCm masquerades as `torch.cuda` (HIP behind the same names), so the
question is never "does the symbol exist" but "does it mean the same thing".

| # | Assumption in code | ROCm status (torch ≥2.7 / ROCm ≥6.3, MI300X gfx942) | Risk | Mitigation |
|---|---|---|---|---|
| 1 | `torch.cuda.CUDAGraph()` + `torch.cuda.graph(g)` capture, `g.replay()` (`graph_time_us`, all timing) | Maps to hipGraph. Production-exercised: vLLM uses graph capture/replay for decode on MI300X, and ROCm 7.2.x release notes actively optimize `hipGraphLaunch` dispatch. But edge cases are real: capture segfault on large graphs ([pytorch#155720](https://github.com/pytorch/pytorch/issues/155720), torch 2.7+rocm6.3), missing error-checking during capture — illegal ops that raise on CUDA silently produce wrong replay output ([pytorch#155684](https://github.com/pytorch/pytorch/issues/155684)) | **Medium** — our graphs are tiny (10–20 nodes), so segfault unlikely; *silent wrong output* is the sharper worry | Smoke test at session start: capture `x.sum()`, compare replay result vs eager to 0 ULP. On failure, fall back to the eager path the harness docstring already anticipates (`t0_eager` subtraction) |
| 2 | `torch.cuda.Event(enable_timing=True)` around `g.replay()`, `elapsed_time` | Supported (hipEvent). Crucially the harness records events **outside** the graph, around replay — events *inside* captured graphs are the known-broken pattern on ROCm ([rocm-systems#2380](https://github.com/ROCm/rocm-systems/issues/2380)). We don't do that | **Low** | None needed; keep events outside capture. Verify timer floor by timing an empty replay |
| 3 | `props.L2_cache_size` = the LLC whose cliff we sweep (both files, line `l2_nom_mb = props.L2_cache_size / 1048576`) | **Wrong on MI300X.** hipDeviceProp `l2CacheSize` reports 4 MB — one XCD's L2, not the 32 MB device aggregate, and not the 256 MB Infinity Cache. Confirmed upstream: [ROCm#4203](https://github.com/ROCm/ROCm/issues/4203) ("reports the size of a single compute die... should be 32MB") | **High — the load-bearing bug.** Downstream: C_eff sweep = 1.6–6 MB (no cliff there → returns `None`); `bw_l2` differential at 0.6 vs 2.2 MB (launch-overhead-dominated, ~75 KB/XCD); `big = min(24·4, 2048) = 96` MB → `bw_hbm` differential 24 vs 96 MB sits **inside the 256 MB Infinity Cache** and measures IC bandwidth, mislabeled | Add `--nominal-llc-mb` override (env or CLI) — keeps guardrail 6 (no *hard-coded* datasheet constants) since the value is an input, recorded in the JSON. Better: add a self-scaling log sweep 1 MB→1 GB that *finds* the cliffs with no prior (runbook step 3) |
| 4 | `props.multi_processor_count` | Returns CU count (304 on MI300X SPX). Different unit than SM but only reported, never computed with | **Low** | Record as-is; label "CU" in the JSON for AMD |
| 5 | `torch.cuda.empty_cache()`, caching allocator, allocations under graph capture (graph-private pool) | HIP caching allocator mirrors CUDA's, including graph memory pools; `x.sum()` workspace allocation under capture is the same path torch.compile+graphs uses on ROCm | **Low–medium** | Covered by the same smoke test as #1 |
| 6 | Warm re-read hits *one shared* LLC (the physical model behind `measure_c_eff`) | **Different on MI300X.** The 32 MB "L2" is 8 disjoint per-XCD caches. A buffer partitioned across workgroups is cached piecewise, and workgroup→XCD assignment across *separate replays* is not architecturally guaranteed stable — a chunk warm in XCD0's L2 may be re-read from XCD3, missing L2, hitting IC. The IC, by contrast, is genuinely shared | **Medium (scientific, not crash)** — the aggregate-L2 cliff may be smeared or absent; the IC cliff should be clean. This is hypothesis material (§3), not a bug | Run both: default SPX mode, and if time allows CPX partition mode (1 XCD = 1 device, clean single-L2 measurement at 4 MB nominal) |
| 7 | Triton `batched_int4_gemm` for r_dequant (`tl.dot`, uint8 unpack, BLOCK 16×64×128) | Triton's AMD backend is official and mature for gfx942 (it is what torchinductor emits on ROCm). `tl.dot` with BLOCK_M=16 maps to MFMA 16×16; wave size is 64 vs 32 so `num_warps` semantics differ (default 4 warps = 256 threads, fine). Graph-capturing a Triton kernel is the vLLM-on-ROCm hot path | **Low for correctness, medium for interpretation** — r_dequant magnitude reflects AMD's int→fp16 convert throughput, not comparable device-to-device except through the model (which is the design) | Verify kernel output vs fp16 reference (rel err check already exists in `bench_l2_barrier`). The 8/16 MB "L2-resident" sizes exceed 4 MB/XCD but sit inside IC — differential still cancels the fixed term; note the residency level in the JSON |
| 8 | `torch.matmul` fp16 peak (8192³ etc.) | hipBLASLt, mature. MI300X dense fp16 peak 1307 TFLOPS; achieved fractions on hipBLASLt historically lower than cuBLAS's ~75–80% | **Low** | Report achieved as always; it is a measured parameter, not a datasheet check |
| 9 | "Lock clocks before running" (docstring) | `nvidia-smi -lgc` equivalent is `rocm-smi --setperfdeterminism <MHz>` / `amd-smi`; MI300X boost variance is large (up to 2100 MHz, thermally limited at 750 W) | **Medium** for repeatability | Runbook step 0 locks clocks; record `rocm-smi` clocks in the JSON |
| 10 | Docstring's own claim: "ROCm via hipify" | No hipify needed at all — torch-rocm aliases `torch.cuda`. The wording overstates the *tested* portability and understates the *actual* mechanism | **Reputational** | After the AMD session, replace with what was measured. Until then, soften to "expected, untested" in the paper |

Bottom line: one real code change (nominal-LLC override + full-range sweep),
one smoke test, one clock-lock command. Everything else is interpretation.

---

## 2. MI300X / MI355X memory hierarchy — the numbers

Sources: [Chips and Cheese MI300X microbenchmarks](https://chipsandcheese.com/p/testing-amds-giant-mi300x),
[AMD CDNA3 whitepaper / Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf),
[ROCm MI300 microarchitecture docs](https://rocm.docs.amd.com/en/docs-7.0.0/conceptual/gpu-arch/mi300.html),
[CDNA4 whitepaper coverage](https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/),
[AMD MI355X product page](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html).

| Level | MI300X (CDNA3, gfx942) | MI355X (CDNA4) | H100 for scale |
|---|---|---|---|
| L2, per XCD | 4 MB, coherence point per die | 4 MB (coherent across XCDs per CDNA4 coverage) | — |
| L2, device aggregate | 32 MB (8 XCDs) — **disjoint, not one cache** | 32 MB (8 XCDs, 32 CU each) | 50 MB, *one* shared L2 |
| Infinity Cache (MALL) | 256 MB = 128 slices × 2 MB on 4 IO dies; 17.2 TB/s theoretical, **11.9 TB/s measured**, ~218 ns latency (Chips and Cheese) | 256 MB = 128 channels × 2 MB, now on 2 IO dies (~14 % lower cross-IOD latency) | none (L2 is the LLC) |
| HBM | 192 GB HBM3, 5.3 TB/s (measured close to peak) | 288 GB HBM3e, 8 TB/s | 80 GB, 3.35 TB/s |
| Dense fp16 matrix peak | 1.31 PFLOPS | 2.5 PFLOPS | 0.99 PFLOPS |

The structural point for us: **the Infinity Cache is a memory-side (MALL)
cache** — it sits with the memory controllers on the IO dies, is transparent to
the programming model, and is filled by address-hash across 128 slices. But
"memory-side" does *not* mean bandwidth-flat: Chips and Cheese measure it as a
distinct level in both latency (~218 ns vs longer for HBM) and bandwidth
(11.9 vs 5.3 TB/s). So a warm re-read that spills the IC should lose roughly
2× bandwidth — an eminently detectable cliff, larger in relative terms than the
L2→HBM cliffs we measured on NVIDIA. One caution from the same source: they
observe a latency jump at 64 MB attributable to TLB reach, not cache capacity.
A 5 %-drop detector on a fine sweep could misfire there; the fix is to check
whether the break at ~64 MB moves with page size (2 MB huge pages via
`HSA_XNACK`/THP) — a capacity cliff won't move, a TLB cliff will.

---

## 3. The two-cliff / no-cliff question

The warm re-read sweep, run over the full 1 MB–1 GB range on MI300X, has three
plausible outcomes. Each is informative; only one is bad for the paper, and
even that one is a finding.

**H1 — two cliffs (aggregate L2 at ~26 MB, IC at ~205 MB, if ≈0.8× holds).**
Requires workgroup→XCD placement to be stable enough across graph replays that
per-XCD L2 warmth survives. If both cliffs appear *and both* land near 0.8×
nominal, the ≈0.8× regularity graduates from "an NVIDIA L2 property observed
three times" to a cross-vendor, cross-cache-level regularity — measured twice
on one card, on two caches with completely different microarchitectures (SM-side
set-associative L2 vs memory-side address-hashed MALL). That would be the
strongest single datapoint in the paper. If the cliffs appear at *different*
fractions (say L2 at 0.8×, IC at ~1.0×), that is nearly as good: it localizes
the 0.8× to processor-side caches and gives the model a per-level effective
capacity, which it can absorb (C_eff is a fitted parameter per level, not a law).
A memory-side cache plausibly *should* sit nearer 1.0×: it is filled by address
hash with no competing SM-side traffic (instructions, spills) claiming ways.

**H2 — one cliff (IC only).** If replay-to-replay XCD placement is unstable,
the "warm" re-read misses the local L2 ~7/8 of the time regardless of buffer
size, the L2 plateau never forms, and the first visible cliff is the IC's. Then
on AMD the operative C_eff for the capacity gate is ~200+ MB, and the
quantization-gate predictions shift by ~8× in the W-axis: a 256 MB IC holds the
entire quantized weight set of layers that on H100 are deep in the HBM regime.
The gate model still applies, but the crossover geometry is different in an
interesting, publishable way ("on MI300X the gate sits at the Infinity Cache,
and W4A16 pays off almost nowhere at decode because bw_IC is already 11.9 TB/s").
CPX partition mode is the control: one XCD as its own device has a private
4 MB L2 and *must* show a clean small cliff if the method is sound.

**H3 — no cliff at all (harness as-is).** This is not physics, it is bug #3:
with nominal = 4 MB the stock sweep tops out at 6 MB and `measure_c_eff`
returns `None`. Worth stating because it is what happens if someone "just runs
it" to check the paper's claim — the harness must not be shipped in this state
with a portability paragraph pointing at it.

For the cross-vendor claim as currently worded (0.8× of *nominal L2*): note
that on MI300X "nominal L2" is ambiguous three ways (4 / 32 / 256 MB depending
on what you call the LLC), and the reported hipDeviceProp value is the least
useful of the three. Whatever the outcome, the paper's phrasing should shift
from "0.8× nominal L2" to "0.8× of the capacity of the cache level whose cliff
gates the workload", with AMD as the case that forces the precision.

---

## 4. 3-hour MI300X session runbook

Target: one MI300X, ROCm ≥ 6.4 (7.x preferred), torch-rocm ≥ 2.7 with Triton.
Provider images: `rocm/pytorch:latest` (torch + Triton preinstalled) on RunPod /
Hot Aisle / DigitalOcean; TensorWave provides equivalent. Budget ~$6–9 at
$2–3/GPU-hr. Prepare *before* the clock starts: push the repo, pre-write the
override patch and the log-sweep script below.

**T+0:00 — environment + properties (15 min).**
```bash
python -c "import torch; print(torch.__version__, torch.version.hip)"
python - <<'EOF'
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, getattr(p, "gcnArchName", "?"), p.multi_processor_count,
      p.L2_cache_size / 1048576, "MB reported L2")
EOF
rocm-smi --showproductname --showmeminfo vram
rocm-smi --setperfdeterminism 1900   # lock clocks; record actual with --showclocks
```
Record the reported `L2_cache_size` verbatim — it is itself a result (expected:
4.0 MB, per ROCm#4203; if a newer ROCm fixed it, that changes the audit table).

**T+0:15 — graph-capture smoke test (15 min). Decision point 1.**
Capture `x.sum()` in a `torch.cuda.graph`, replay, compare to eager bitwise;
then time an empty graph replay to establish the timer floor. If capture fails
or replay output mismatches → flip `graph_time_us` to eager events + subtract
`t0_eager` (the fallback the harness docstring specifies) and note it in the
JSON `method` field. Do not burn more than 15 min debugging hipGraph.

**T+0:30 — naive run, for the record (15 min).**
`python measure_params.py` unmodified. Expected: `nominal_l2_mb: 4.0`,
`c_eff: None` (or a spurious sub-6 MB value), `bw_hbm` ≈ 8–12 TB/s (that is the
IC, mislabeled — keep this JSON as the cautionary artifact for the paper's
portability section).

**T+0:45 — full-range log sweep: the money measurement (45 min).**
```python
# sweep_full_mi300x.py — warm re-read over 1 MB..1 GB, log grid
from measure_params import read_bw_tbs
import json, numpy as np
pts = [(float(mb), *read_bw_tbs(float(mb)))
       for mb in np.unique(np.round(np.logspace(0, 3, 48)))]
json.dump(pts, open("sweep_full_mi300x.json", "w"))
```
Expected shapes: BW rising then breaking near ~26 MB (H1) and/or near ~205 MB
(H1/H2), with a possible TLB artifact near 64 MB. **Decision point 2:** if a
~64 MB break appears, re-run 48–80 MB with THP/huge pages toggled; a break that
moves is TLB, not capacity — exclude it from cliff detection.

**T+1:30 — fine sweeps around each observed cliff (40 min).**
`cliff_finegrain.py` with nominal overridden to 32 (and then 256) so its
0.55–1.0× window brackets each cliff; 0.5 MB steps for L2, 4 MB steps for IC,
3 repeats each. Output: C_eff and ratio per level, the AMD rows of the
cross-vendor table.

**T+2:10 — corrected bandwidth + compute params (30 min).**
`bw_l2` differential inside 8–24 MB (aggregate-L2 window); `bw_ic` differential
inside 96–200 MB (new parameter — NVIDIA has no such level); `bw_hbm`
differential 512 vs 2048 MB (beyond IC); `measure_peak_tflops` and
`measure_r_dequant` as-is (fixed sizes, unaffected by bug #3). Sanity anchors:
bw_hbm ≲ 5.3, bw_ic ≲ 11.9 TB/s.

**T+2:40 — buffer / stretch (20 min).**
If ahead of schedule and the L2 cliff was absent (H2): switch to CPX partition
mode (`amd-smi set --compute-partition CPX`, or ask provider), rerun the fine
sweep on one XCD-device for the clean 4 MB control. Pull all JSONs off the box
before release.

---

## 5. Rental options (checked 2026-08; prices move, treat as ±20 %)

Sources: [getdeploying MI300X index](https://getdeploying.com/gpus/amd-mi300x),
[Spheron pricing survey, 8 Jul 2026](https://www.spheron.network/blog/amd-mi300x-mi355x-pricing-2026/),
[Thunder Compute MI300X pricing](https://www.thundercompute.com/blog/amd-mi300x-pricing),
[gpucost.org](https://gpucost.org/gpu/mi300x).

| Provider | MI300X $/GPU-hr | Min size | Notes for us |
|---|---|---|---|
| TensorWave | ~1.71 | 1 | Cheapest tracked on-demand; AMD-first shop, ROCm images standard |
| Hot Aisle | ~1.99 | 1 | AMD-only cloud, per-minute billing, no contract — good fit for a 3 h session; developer-friendly bare access (clock locking should work) |
| RunPod | ~1.99–2.49 (spot ~1.49) | 1 | `rocm/pytorch` template; spot fine for us (all state is JSONs, re-runnable) |
| DigitalOcean (ex-Paperspace) | ~1.99 | 1 | Single instances |
| Vultr | ~1.85 | 8-GPU node | $14.80/hr — only if we want a full node |
| Vast.ai | spotty MI300X listings | 1 | NVIDIA-dominated marketplace; MI355X spot seen at ~4.80 |
| Azure / Oracle / CoreWeave | 6.00–7.86 | 8 / 8 / 1 | Not worth it for this |

MI355X (only after MI300X succeeds — same 256 MB IC, so the scientific
marginal value is the CDNA4 datapoint + 8 TB/s HBM3e, not a new hierarchy):
Vultr ~2.59 (8-GPU pod), CoreWeave 7.20 on-demand, OVH ~7.10. Defer.

**Recommendation:** Hot Aisle or RunPod, 1× MI300X, 3 h, ~$6. Hot Aisle's
per-minute billing and AMD focus (likelier `rocm-smi` privileges for clock
locking and CPX mode) makes it the first choice; RunPod the fallback for
instant availability. Total cost is noise; the preparation in §4 is the real
budget item.

---

## 6. What this buys the paper

DIRECTION.md already says it: one non-NVIDIA backend would be transformative
(P5 — "fit on architecture A, predict on architecture B"). MI300X is not just
"a fourth GPU": it is the first backend where the hierarchy *shape* differs
(disjoint L2s + a memory-side LLC NVIDIA doesn't have), so it tests whether
C_eff is a property of the method or of NVIDIA L2s. The audit above says the
port is one small patch plus one smoke test away — the "hipify" sentence in the
harness docstring should not survive contact with the actual mechanism, and
after the session it won't have to.
