"""
Kernel-level bridge for the engine-level 896->1024 prefill band step
(2026-08-02; follows profiling/served/run_band_prefill.sh, which measured a
reproducible +24% total-throughput step between chunk caps 896 and 1024 on
served Qwen3.6-27B fp8).

Question: is the step predicted by wave quantization of the model's dominant
prefill GEMMs at M = chunk size?

Method: graph-time the four dominant per-layer GEMM shapes (qkv, o, gate_up,
down; dims from the checkpoint config) at M in {768..1152}, fp8 W8A8 via
torch._scaled_mm (the engine's cutlass path) and bf16 torch.mm as reference.
Report per-token time (t/M) per shape and the M-weighted sum; capture grid
sizes via the profiler for CTA/wave arithmetic on 132 SMs.

Clock-locked. Output: results_prefill_band_bridge.json
"""

import json
import os
import statistics

import torch

_D = os.path.dirname(os.path.abspath(__file__))

SHAPES = {           # K, N  (weight [K,N]) — Qwen3.6-27B config
    "qkv":     (5120, 8192),
    "o_proj":  (6144, 5120),
    "gate_up": (5120, 34816),
    "down":    (17408, 5120),
}
M_SWEEP = [768, 832, 896, 960, 1024, 1088, 1152]

GRAPH_INNER, GRAPH_REPS = 10, 30


def graph_time_us(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_INNER):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(GRAPH_REPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / GRAPH_INNER * 1000


def bench_cell(M, K, N):
    a8 = (torch.randn(M, K, device="cuda") / 8).to(torch.float8_e4m3fn)
    b8 = (torch.randn(N, K, device="cuda").t() / 8).to(torch.float8_e4m3fn)
    sa = torch.ones(M, 1, device="cuda")
    sb = torch.ones(1, N, device="cuda")
    out8 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    fn8 = lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb,  # noqa: E731
                                   out_dtype=torch.bfloat16, out=out8)
    t8 = graph_time_us(fn8)

    ab = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    bb = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    cb = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    tb = graph_time_us(lambda: torch.mm(ab, bb, out=cb))

    # grid capture (fp8 kernel)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn8()
        torch.cuda.synchronize()
    kern = ""
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA and "scaled" not in e.key.lower():
            if e.self_device_time_total > 0:
                kern = e.key[:70]
    del a8, b8, out8, ab, bb, cb
    torch.cuda.empty_cache()
    return t8, tb, kern


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    clocks = os.popen(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}  SM clock: {clocks}")

    results = []
    print(f"{'M':>5} " + "".join(f"{s+'(f8/bf)':>22}" for s in SHAPES)
          + f" {'sum f8 ns/tok':>14}")
    for M in M_SWEEP:
        row = {"M": M, "shapes": {}}
        tot8 = totb = 0.0
        cells = []
        for name, (K, N) in SHAPES.items():
            t8, tb, kern = bench_cell(M, K, N)
            row["shapes"][name] = {
                "fp8_us": round(t8, 2), "bf16_us": round(tb, 2),
                "fp8_ns_per_tok": round(t8 * 1000 / M, 2),
                "kernel": kern}
            tot8 += t8
            totb += tb
            cells.append(f"{t8:7.1f}/{tb:7.1f}")
        row["sum_fp8_us"] = round(tot8, 2)
        row["sum_bf16_us"] = round(totb, 2)
        row["sum_fp8_ns_per_tok"] = round(tot8 * 1000 / M, 2)
        row["sum_bf16_ns_per_tok"] = round(totb * 1000 / M, 2)
        results.append(row)
        print(f"{M:>5} " + " ".join(cells) + f" {row['sum_fp8_ns_per_tok']:>14}",
              flush=True)

    out = {"gpu": gpu, "sm_clock": clocks,
           "shapes": {k: list(v) for k, v in SHAPES.items()},
           "timing": f"CUDA graphs, {GRAPH_INNER}/graph, median of {GRAPH_REPS}",
           "results": results}
    with open(os.path.join(_D, "results_prefill_band_bridge.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results_prefill_band_bridge.json")


if __name__ == "__main__":
    main()
