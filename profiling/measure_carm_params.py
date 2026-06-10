"""
Measure Cache-Aware Roofline Model (CARM) parameters on the current GPU.

Produces carm_params.json with:
  - hbm_read_tbs:   streaming read bandwidth (sum-reduction over 1 GB)
  - l2_read_tbs:    read bandwidth for an L2-resident 16 MB buffer (graph-timed)
  - kernel_floor_us: per-kernel fixed cost inside a CUDA graph (tiny bmm)
  - eager_floor_us:  per-launch event-timed floor (repo's original methodology)
  - recon_points:    graph-timed MLA reconstruction BMM1 (d_lora=512) across
                     batch sizes, FP16 + INT4, with AI and achieved TFLOPS
  - fp16_size_sweep / int4_size_sweep: graph-timed kernel latency vs weight size
    (copied from validation/ runs when present, else re-measured)

All bandwidths are measured, not vendor numbers: the CARM ceilings should
reflect what these access patterns can actually achieve.
"""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4

H, K_DIM, D_LORA = 128, 128, 512
DEV = "cuda"
_D = os.path.dirname(os.path.abspath(__file__))


def graph_time_us(build_fn, n_inner, n_rep=50):
    """Median time per inner op inside a captured CUDA graph."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000


def _sum_time_us(mbytes, n_inner):
    n = mbytes * 1024 * 1024 // 4
    x = torch.randn(n, dtype=torch.float32, device=DEV)
    for _ in range(10):
        x.sum()
    torch.cuda.synchronize()
    us = graph_time_us(lambda: [x.sum() for _ in range(n_inner)], n_inner=n_inner)
    del x
    torch.cuda.empty_cache()
    return us


def measure_hbm_read_tbs():
    """Differential slope between two HBM-sized reductions cancels fixed cost."""
    t_small = _sum_time_us(256, 8)
    t_large = _sum_time_us(1024, 4)
    return (1024 - 256) * 1024 * 1024 / ((t_large - t_small) * 1e-6) / 1e12


def measure_l2_read_tbs():
    """Differential slope between two L2-resident reductions cancels fixed cost."""
    t_small = _sum_time_us(8, 20)
    t_large = _sum_time_us(28, 20)
    return (28 - 8) * 1024 * 1024 / ((t_large - t_small) * 1e-6) / 1e12


def measure_kernel_floor_us():
    x = torch.randn(H, 1, K_DIM, dtype=torch.float16, device=DEV)
    w = torch.randn(H, K_DIM, 16, dtype=torch.float16, device=DEV)  # 0.5 MB
    for _ in range(10):
        torch.bmm(x, w)
    torch.cuda.synchronize()
    return graph_time_us(lambda: [torch.bmm(x, w) for _ in range(20)], n_inner=20)


def measure_eager_floor_us():
    x = torch.randn(H, 1, K_DIM, dtype=torch.float16, device=DEV)
    w = torch.randn(H, K_DIM, 16, dtype=torch.float16, device=DEV)
    for _ in range(50):
        torch.bmm(x, w)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.bmm(x, w)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1000


def recon_points():
    """Graph-timed BMM1 across batch sizes; FP16 and INT4."""
    out = []
    for bs in [1, 4, 16, 64, 128, 256, 512]:
        x = torch.randn(H, bs, K_DIM, dtype=torch.float16, device=DEV)
        w = torch.randn(H, K_DIM, D_LORA, dtype=torch.float16, device=DEV)
        wp, sc = quantize_weights_int4(w)

        for _ in range(10):
            torch.bmm(x, w)
            batched_int4_gemm(x, wp, sc, K_DIM)
        torch.cuda.synchronize()

        fp16_us = graph_time_us(lambda: [torch.bmm(x, w) for _ in range(20)], n_inner=20)
        int4_us = graph_time_us(
            lambda: [batched_int4_gemm(x, wp, sc, K_DIM) for _ in range(20)], n_inner=20)

        flops = 2 * H * bs * K_DIM * D_LORA
        wt_b = H * K_DIM * D_LORA * 2
        act_b = H * bs * K_DIM * 2
        o_b = H * bs * D_LORA * 2
        ai_fp16 = flops / (wt_b + act_b + o_b)
        ai_int4 = flops / (wt_b // 4 + H * D_LORA * 2 + act_b + o_b)
        out.append({
            "bs": bs,
            "fp16_us": round(fp16_us, 3), "int4_us": round(int4_us, 3),
            "ai_fp16": round(ai_fp16, 3), "ai_int4": round(ai_int4, 3),
            "fp16_tflops": round(flops / (fp16_us * 1e-6) / 1e12, 4),
            "int4_tflops": round(flops / (int4_us * 1e-6) / 1e12, 4),
            "flops": flops,
            "fp16_bytes": wt_b + act_b + o_b,
        })
        del x, w, wp, sc
        torch.cuda.empty_cache()
    return out


def main():
    torch.cuda.init()
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    hbm = measure_hbm_read_tbs()
    l2 = measure_l2_read_tbs()
    floor = measure_kernel_floor_us()
    eager_floor = measure_eager_floor_us()
    pts = recon_points()

    params = {
        "gpu": gpu,
        "peak_fp16_tflops": 989.4 if "H100" in gpu.upper() else 312.0,
        "hbm_read_tbs": round(hbm, 3),
        "l2_read_tbs": round(l2, 3),
        "kernel_floor_us": round(floor, 3),
        "eager_floor_us": round(eager_floor, 3),
        "recon_points": pts,
    }

    # Attach graph-timed size sweeps from the validation runs when available.
    vdir = os.path.join(_D, "validation")
    try:
        with open(os.path.join(vdir, "diag_results.json")) as f:
            params["fp16_size_sweep"] = json.load(f)["d3_graph"]
        with open(os.path.join(vdir, "graph_sweep_int4.json")) as f:
            params["int4_size_sweep"] = json.load(f)
    except FileNotFoundError:
        print("validation sweeps not found; run validation/diag_l2_residency.py first")

    out = os.path.join(_D, "carm_params.json")
    with open(out, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps({k: v for k, v in params.items() if k not in
                      ("recon_points", "fp16_size_sweep", "int4_size_sweep")}, indent=2))
    print(f"recon_points: {len(pts)} entries")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
