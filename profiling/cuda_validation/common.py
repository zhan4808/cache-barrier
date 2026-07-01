"""Shared helpers for the CUDA-operator validation runs (Experiments A/B/C).

Methodology is copied verbatim from profiling/fused_moe/bench_fused_moe_extended.py
(graph_med_us: 10 launches/graph, median of 40 replays, warm cache) so the CUDA
numbers are directly comparable to the existing FlagGems Triton JSONs. Nothing in
here writes to the Triton result files.
"""

import json
import os
import statistics

import torch

N_GRAPH = 10


def graph_med_us(fn, n_inner=N_GRAPH, reps=40, warmup=10):
    """Median per-call latency (us) inside a captured CUDA graph.

    fn must operate on pre-allocated static buffers (no fresh allocation inside
    the timed region), matching profiling/w8a8/bench_w8a8.py.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000.0


def eager_med_us(fn, reps=50, warmup=20):
    """Median per-call latency (us), event-timed, no graph. Fallback when a
    kernel is not graph-capturable; flagged in the result row when used."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1000.0


def env_versions():
    """Record exact versions for the report. Best-effort on optional libs."""
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda}
    for mod in ("triton", "vllm", "flash_mla", "flashmla"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"<absent: {type(exc).__name__}>"
    return out


def gpu_key():
    """Map the live device to a CARM param key (carm.py CARM_PARAMS).

    Keys off compute capability (robust across SKUs/marketing names) with a
    device-name tiebreak for the 90GB vs 180GB Blackwell parts. Added for the
    B200/B100 leg so the same bench self-selects params on any GPU.

      (9,0)  -> h100        (8,0) -> a100
      (10,x) -> b200 / b100 (name tiebreak; both SM100, native FP4)
      (12,x) -> b200        (consumer/other Blackwell fall back to b200 params)
    """
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0).lower()
    if cap == (9, 0):
        return "h100"
    if cap == (8, 0):
        return "a100"
    if cap[0] == 10:
        # SM100 family: GB200/B200 vs B100. Name tiebreak; default b200.
        if "b100" in name:
            return "b100"
        return "b200"
    if cap[0] >= 10:
        return "b200"
    raise RuntimeError(f"No CARM param key for {name!r} cap={cap}; add one to carm.py")


def native_low_precisions():
    """Precisions with native tensor-core MMA on the live device (no dequant).

    Drives the FP4 leg: on H100 FP4 is EMULATED (Marlin dequant->bf16); on
    SM100 (B200/B100) FP4 MMA is native. Kept as a runtime probe so the bench
    labels EMU vs native correctly wherever it runs."""
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 10:      # Blackwell SM100+: int8, fp8, fp4 all native
        return ("int8", "fp8", "fp4")
    if cap == (9, 0):     # Hopper: int8, fp8 native; NO fp4
        return ("int8", "fp8")
    if cap == (8, 0):     # Ampere: int8 native
        return ("int8",)
    return ()


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


_D = os.path.dirname(os.path.abspath(__file__))
