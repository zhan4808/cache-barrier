"""
P1 — The capacity-gate figure (DIRECTION.md §6 P1).

2D sweep: weight working set W x token count T, three precisions, one figure:
speedup vs W / C_eff with the sign flip at the measured effective capacity.

Shape family: the batched-BMM family every CARM parameter was fit on
(H=128 heads, K=128, N swept to set the weight working set; M=T tokens).
Weight bytes (bf16-equivalent) = H*K*N*2, so N = 32 * W_MB.

Precisions:
  bf16   — cuBLAS torch.bmm (tuned CUDA baseline, guardrail 3/5). fp16 storage;
           fp16 and bf16 tensor-core paths are rate-identical on H100.
  w4a16  — Triton in-core dequant kernel (bench_l2_barrier.py). Carries the
           dequant ceiling (r_dequant = 0.496 TB/s packed bytes, re-dequanted
           per M-tile). CARM predicts it NEVER reaches the MARLIN far field on
           this kernel — the ceiling-free far field is the W8A8 line (and the
           CUDA Marlin validation in profiling/cuda_validation/, ceiling 423 TF).
  w8a8   — INT8 IMMA kernel (profiling/w8a8/w8a8_bmm.py), dynamic act quant
           INSIDE the timed region. No in-core dequant -> the constructive gate.

Methodology (guardrails): CUDA-graph timing only (10 launches/graph, median of
30 replays); clocks locked externally via `nvidia-smi -lgc 1755`; per-cell
mini-autotune over Triton block configs; warm-loop = weights L2-warm across
replays, i.e. the ISOLATED-microbenchmark C_eff (serving contention kills this
regime entirely — profiling/dense_qwen/results_contention_h100.json).

CARM overlay: t = t0_graph + max(B/BW(W_operand), F/ceiling) per precision,
BW capacity-gated at each precision's OWN operand bytes (operand-aware gate,
dense_qwen finding). MAPE reported per regime (guardrail 7).
"""

import argparse
import json
import os
import statistics
import sys

import torch
import triton

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, ".."))          # profiling/
sys.path.insert(0, os.path.join(_D, "..", "w8a8"))  # profiling/w8a8/

from bench_l2_barrier import kernel_batched_w4a16_simple, quantize_weights_int4  # noqa: E402
from w8a8_bmm import quantize_weights_w8, quantize_acts_w8, w8a8_bmm, _pick_config  # noqa: E402

# Cross-architecture: --params selects the CARM params file. Accepts either
# the carm_model.json key names (H100 fit) or the portable-harness names
# (params_<gpu-slug>.json from measure_params.py).
_ap = argparse.ArgumentParser()
_ap.add_argument("--params", default=os.path.join(_D, "..", "carm_model.json"))
_ap.add_argument("--tsweep", default="1,16,32,64,128,256,512",
                 help="comma-separated token counts")
_ap.add_argument("--out", default=os.path.join(_D, "results_capacity_gate.json"))
ARGS = _ap.parse_args()

with open(ARGS.params) as f:
    P = json.load(f)


def _p(*keys):
    for k in keys:
        if k in P and P[k] is not None:
            return P[k]
    raise KeyError(keys)


C_EFF = _p("effective_l2_capacity_mb") * 1024 * 1024
BW_L2 = _p("bw_l2_gemm_tbs", "bw_l2_tbs") * 1e12
BW_HBM = _p("bw_hbm_tbs") * 1e12
PEAK = _p("peak_tflops", "peak_fp16_tflops") * 1e12
PEAK_INT8 = PEAK  # conservative: Triton IMMA does not reach the 2x INT8 rate
T0_US = _p("t0_graph_us")
R_DQ = _p("r_dequant_tbs") * 1e12
BW_ACTQ = 1.6e12  # measured act-quant kernel bandwidth (dense_qwen)

H, K = 128, 128
W_MB_SWEEP = [8, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128]
T_SWEEP = [int(t) for t in ARGS.tsweep.split(",")]

GRAPH_INNER = 10
GRAPH_REPS = 30
TUNE_REPS = 8


def graph_time_us(fn, reps=GRAPH_REPS):
    """Median time per launch under CUDA graphs (guardrail 2)."""
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
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0 / GRAPH_INNER)
    return statistics.median(times)


def block_m_for(t):
    return max(16, min(128, triton.next_power_of_2(t)))


def make_w4a16_fn(x, wp, ws, c, T, N, BLOCK_N):
    BLOCK_M = block_m_for(T)
    grid = (H, triton.cdiv(T, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    def fn():
        kernel_batched_w4a16_simple[grid](
            x, wp, ws, c, T, N, K,
            x.stride(0), x.stride(1), x.stride(2),
            wp.stride(0), wp.stride(1), wp.stride(2),
            ws.stride(0), ws.stride(1),
            c.stride(0), c.stride(1), c.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=K,
        )
    return fn


def make_w8a8_fn(x, wq, wscale, T, N, cfg):
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty(H * T, dtype=torch.float32, device="cuda")
    out = torch.empty(H, T, N, dtype=torch.float16, device="cuda")

    def fn():
        quantize_acts_w8(x, q, s)
        w8a8_bmm(q, wq, s, wscale, out, **cfg)
    return fn


def predict_us(precision, T, N, block_m=None, split_mem=False):
    """CARM: t0 + max(mem, compute), operand-aware capacity gate.

    split_mem=False: lumped form (kernel-compass carm.py) — all bytes at the
    weight-gated bandwidth. split_mem=True: weights at gated BW, activations
    and outputs streamed at HBM BW (physical for large T where the output,
    H*T*N*2 bytes, dominates traffic and never re-reads)."""
    E = H * K * N
    flops = 2.0 * H * T * K * N
    act = H * T * K * 2.0
    out = H * T * N * 2.0
    if precision == "bf16":
        wbytes = 2.0 * E
        comp = flops / PEAK
        extra = 0.0
    elif precision == "w4a16":
        wbytes = 0.5 * E + 2.0 * H * N
        tiles = -(-T // (block_m or block_m_for(T)))
        comp = max(flops / PEAK, 0.5 * E * tiles / R_DQ)  # re-dequant per M-tile
        extra = 0.0
    else:  # w8a8
        wbytes = 1.0 * E + 4.0 * H * N
        comp = flops / PEAK_INT8
        # dynamic act quant: fp16 read + int8 write, own launch
        extra = T0_US * 1e-6 + (3.0 * H * T * K) / BW_ACTQ
    bw = BW_L2 if wbytes < C_EFF else BW_HBM
    if split_mem:
        mem = wbytes / bw + (act + out) / BW_HBM
    else:
        mem = (wbytes + act + out) / bw
    return (T0_US * 1e-6 + max(mem, comp) + extra) * 1e6


def run_cell(w_mb, T):
    N = 32 * w_mb
    x = torch.randn(H, T, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
    c = torch.empty(H, T, N, dtype=torch.float16, device="cuda")

    # bf16 baseline (cuBLAS)
    t_bf16 = graph_time_us(lambda: torch.bmm(x, w, out=c))

    # w4a16: mini-tune BLOCK_N
    wp, wsc = quantize_weights_int4(w)
    best = None
    for bn in (64, 128):
        fn = make_w4a16_fn(x, wp, wsc, c, T, N, bn)
        t = graph_time_us(fn, reps=TUNE_REPS)
        if best is None or t < best[0]:
            best = (t, bn, fn)
    t_w4a16 = graph_time_us(best[2])
    w4_bn = best[1]

    # w8a8: mini-tune BLOCK_N around the pre-tuned config
    wq, wscale = quantize_weights_w8(w)
    cfg0 = _pick_config(T)
    best = None
    for bn in (64, 128):
        cfg = dict(cfg0, BLOCK_N=bn)
        fn = make_w8a8_fn(x, wq, wscale, T, N, cfg)
        t = graph_time_us(fn, reps=TUNE_REPS)
        if best is None or t < best[0]:
            best = (t, bn, fn)
    t_w8a8 = graph_time_us(best[2])
    w8_bn = best[1]

    del x, w, c, wp, wsc, wq, wscale
    torch.cuda.empty_cache()

    return {
        "w_mb": w_mb, "tokens": T, "N": N,
        "bf16_us": round(t_bf16, 3),
        "w4a16_us": round(t_w4a16, 3),
        "w8a8_us": round(t_w8a8, 3),
        "w4a16_block_n": w4_bn, "w8a8_block_n": w8_bn,
        "sp_w4a16": round(t_bf16 / t_w4a16, 4),
        "sp_w8a8": round(t_bf16 / t_w8a8, 4),
        "pred_bf16_us": round(predict_us("bf16", T, N), 3),
        "pred_w4a16_us": round(predict_us("w4a16", T, N), 3),
        "pred_w8a8_us": round(predict_us("w8a8", T, N), 3),
    }


def sanity_check():
    """Numerical correctness at one mid cell before trusting any timing."""
    T, N = 32, 32 * 16
    x = torch.randn(H, T, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
    ref = torch.bmm(x, w).float()

    # w4a16 kernel vs an explicit-dequant reference (isolates kernel bugs from
    # inherent INT4 quantization noise, which is ~0.1 rel on gaussian weights)
    wp, wsc = quantize_weights_int4(w)
    lo = (wp & 0x0F).to(torch.int16)
    hi = ((wp >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    w_dq = torch.empty(H, K, N, dtype=torch.float16, device="cuda")
    w_dq[:, 0::2, :] = lo.half()
    w_dq[:, 1::2, :] = hi.half()
    w_dq *= wsc[:, None, :]
    ref4 = torch.bmm(x, w_dq).float()
    c = torch.empty(H, T, N, dtype=torch.float16, device="cuda")
    make_w4a16_fn(x, wp, wsc, c, T, N, 64)()
    r4 = ((c.float() - ref4).norm() / ref4.norm()).item()

    wq, wscale = quantize_weights_w8(w)
    q, s = quantize_acts_w8(x)
    o8 = w8a8_bmm(q, wq, s, wscale).float()
    r8 = ((o8 - ref).norm() / ref.norm()).item()
    print(f"sanity rel_err: w4a16-vs-dequant={r4:.4f} w8a8-vs-fp16={r8:.4f}", flush=True)
    assert r4 < 0.02 and r8 < 0.02, "quantized kernel numerically wrong"


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    clocks = os.popen(
        "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader").read().strip()
    print(f"GPU: {gpu}  SM clock: {clocks} (lock externally: nvidia-smi -lgc <max>)")
    print(f"params: {ARGS.params}")
    print(f"torch {torch.__version__}  triton {triton.__version__}")
    sanity_check()

    results = []
    for T in T_SWEEP:
        print(f"\n=== T={T} ===")
        print(f"{'W(MB)':>6} {'W/Ceff':>7} {'bf16':>9} {'w4a16':>9} {'w8a8':>9} "
              f"{'sp4':>6} {'sp8':>6}")
        for w_mb in W_MB_SWEEP:
            r = run_cell(w_mb, T)
            results.append(r)
            print(f"{w_mb:>6} {w_mb/P['effective_l2_capacity_mb']:>7.2f} "
                  f"{r['bf16_us']:>9.2f} {r['w4a16_us']:>9.2f} {r['w8a8_us']:>9.2f} "
                  f"{r['sp_w4a16']:>6.2f} {r['sp_w8a8']:>6.2f}", flush=True)

    # Regime-separated MAPE (guardrail 7). Gate at each precision's operand bytes.
    mape = {}
    for prec, wfrac in (("bf16", 2.0), ("w4a16", 0.5), ("w8a8", 1.0)):
        below, above = [], []
        for r in results:
            err = abs(r[f"pred_{prec}_us"] - r[f"{prec}_us"]) / r[f"{prec}_us"]
            operand = r["w_mb"] * 1024 * 1024 * wfrac / 2.0
            (below if operand < C_EFF else above).append(err)
        mape[prec] = {
            "below_gate_mape_pct": round(100 * sum(below) / max(len(below), 1), 1),
            "above_gate_mape_pct": round(100 * sum(above) / max(len(above), 1), 1),
            "n_below": len(below), "n_above": len(above),
        }
    print("\nRegime-separated MAPE (operand-aware gate):")
    print(json.dumps(mape, indent=2))

    out = {
        "gpu": gpu, "sm_clock": clocks,
        "torch": torch.__version__, "triton": triton.__version__,
        "H": H, "K": K, "c_eff_mb": P["effective_l2_capacity_mb"],
        "timing": f"CUDA graphs, {GRAPH_INNER} launches/graph, median of {GRAPH_REPS} replays",
        "params_file": ARGS.params,
        "carm_params": {
            "peak_tflops": PEAK / 1e12, "bw_hbm_tbs": BW_HBM / 1e12,
            "bw_l2_tbs": BW_L2 / 1e12,
            "effective_l2_capacity_mb": C_EFF / 1048576,
            "t0_graph_us": T0_US, "r_dequant_tbs": R_DQ / 1e12},
        "mape_regime_separated": mape,
        "results": results,
    }
    path = ARGS.out
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
