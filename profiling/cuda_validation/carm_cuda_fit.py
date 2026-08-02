"""Experiment C -- re-fit the cache-aware roofline (CARM) on the CUDA data and
make it GPU-parameterized.

Two parts:

1) CUDA-env base params: re-measure HBM/L2 read bandwidth and the graphed per-
   launch floor on the torch-2.11/CUDA-13 stack (the Triton numbers were torch
   2.7). Standalone (no triton/flag_gems) so it runs in the vLLM env.

2) GPU-parameterized CARM:  t = t0 + max(B/BW(WS), F/P)
   - BW(WS) capacity-gated on effective L2 capacity (per-GPU input).
   - P is the *kernel* compute ceiling, which depends on the precision regime
     AND on whether the GPU has native MMA for that precision:
       * weight-only quant (W8A16 / W4A16, bf16 compute): P = in-core dequant
         ceiling (fit from CUDA data) -- BELOW bf16 peak, so a crossover EXISTS.
       * matched quant (W8A8 / W4A4): P = native low-precision tensor-core peak
         IF the precision is native on the GPU, else the emulated dequant ceiling.
   The model is fit on the measured H100 CUDA MoE sweep, validated by MAPE, and
   then evaluated for a *projected* B200 parameter set (native FP4 + bigger L2)
   as the Blackwell hook. Every B200 number is labelled PROJECTED.

Writes carm_cuda_params.json. Prints MAPE (vs the Triton-based 10-18%) and the
per-operator / per-GPU quant-vs-dense crossover token counts.
"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json  # noqa: E402

_D = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"

# ---- Mixtral MoE shape (matches Experiment A) ----
E, H, I, TOPK = 8, 4096, 14336, 2


def moe_flops(T):
    return 2.0 * T * TOPK * (2 * H * (2 * I) + I * H)


def moe_weight_bytes(bytes_per_elem, e_touched=E):
    # w1 (2I*H) + w2 (H*I) per expert = 3*H*I elements
    return e_touched * 3 * H * I * bytes_per_elem


def moe_act_bytes(T):
    # bf16 activations: input + output + gemm1-out + gemm2-in
    return 2.0 * (2 * T * H + TOPK * T * (2 * I) + TOPK * T * I)


# ======================================================================
# Part 1 -- re-measure base params on the CUDA-13 / torch-2.11 stack
# ======================================================================
def graph_time_us(build_fn, n_inner, n_rep=50):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000


def _sum_us(mb, n_inner):
    n = mb * 1024 * 1024 // 4
    x = torch.randn(n, dtype=torch.float32, device=DEV)
    for _ in range(10):
        x.sum()
    torch.cuda.synchronize()
    us = graph_time_us(lambda: [x.sum() for _ in range(n_inner)], n_inner=n_inner)
    del x; torch.cuda.empty_cache()
    return us


def measure_base_params():
    hbm = (1024 - 256) * 1024 * 1024 / ((_sum_us(1024, 4) - _sum_us(256, 8)) * 1e-6) / 1e12
    l2 = (28 - 8) * 1024 * 1024 / ((_sum_us(28, 20) - _sum_us(8, 20)) * 1e-6) / 1e12
    x = torch.randn(128, 1, 128, dtype=torch.float16, device=DEV)
    w = torch.randn(128, 128, 16, dtype=torch.float16, device=DEV)
    for _ in range(10):
        torch.bmm(x, w)
    torch.cuda.synchronize()
    floor = graph_time_us(lambda: [torch.bmm(x, w) for _ in range(20)], n_inner=20)
    return {"hbm_read_tbs": round(hbm, 3), "l2_read_tbs": round(l2, 3),
            "kernel_floor_us": round(floor, 3)}


# ======================================================================
# Part 2 -- GPU-parameterized CARM
# ======================================================================
# native_mma: precisions with native tensor-core MMA (matched-precision compute).
# native_peak_mult: tensor-core peak relative to bf16 for a matched kernel.
GPU_PARAMS = {
    "h100": {  # measured (cache-barrier + this run)
        "measured": True,
        "c_eff_mb": 36.0, "bw_l2_tbs": 6.3, "bw_hbm_tbs": 3.146,
        "peak_bf16_tflops": None,        # fit below from the CUDA bf16 sweep
        "dequant_ceiling_tflops": None,  # fit below from the CUDA fp8 sweep
        "t0_us": 2.8,
        "native_mma": ["int8", "fp8"],   # H100: NO native fp4
        "native_peak_mult": {"int8": 2.0, "fp8": 2.0},
    },
    "b200": {  # PROJECTED (Blackwell hook) -- not measured here
        "measured": False,
        "c_eff_mb": 96.0,                # 64-126 MB range; midpoint
        "bw_l2_tbs": 10.0, "bw_hbm_tbs": 8.0,   # HBM3e ~8 TB/s
        "peak_bf16_tflops": 2250.0,      # ~2.25 PFLOPS dense bf16
        "dequant_ceiling_tflops": 960.0, # scaled ~ same fraction of peak as H100
        "t0_us": 2.5,
        "native_mma": ["int8", "fp8", "fp4"],   # B200: native FP4
        "native_peak_mult": {"int8": 2.0, "fp8": 2.0, "fp4": 4.0},
    },
    "b100": {  # PROJECTED lower-power SM100 sibling (native FP4, ~0.8x compute)
        "measured": False,
        "c_eff_mb": 96.0,
        "bw_l2_tbs": 9.6, "bw_hbm_tbs": 7.7,     # HBM3e, slightly below B200
        "peak_bf16_tflops": 1800.0,              # ~0.8x B200 dense bf16
        "dequant_ceiling_tflops": 770.0,         # same fraction of peak as B200
        "t0_us": 2.5,
        "native_mma": ["int8", "fp8", "fp4"],    # B100: native FP4
        "native_peak_mult": {"int8": 2.0, "fp8": 2.0, "fp4": 4.0},
    },
}

# quant modes: (weight bytes/elem, compute regime, precision tag)
QUANT_MODES = {
    "w8a16":      {"belem": 1.0, "matched": False, "prec": "fp8"},   # weight-only fp8
    "w4a16_mxfp4":{"belem": 0.5, "matched": False, "prec": "fp4"},   # weight-only fp4 (mxfp4)
    "w8a8":       {"belem": 1.0, "matched": True,  "prec": "fp8"},   # matched fp8/int8
    "w4a4_mxfp4": {"belem": 0.5, "matched": True,  "prec": "fp4"},   # matched fp4
}


def ceiling_tflops(gp, mode):
    """Compute ceiling (TFLOPS) for a quant mode on a GPU."""
    m = QUANT_MODES[mode]
    if not m["matched"]:
        return gp["dequant_ceiling_tflops"]          # bf16 compute -> dequant ceiling
    if m["prec"] in gp["native_mma"]:
        return gp["peak_bf16_tflops"] * gp["native_peak_mult"][m["prec"]]
    return gp["dequant_ceiling_tflops"]              # matched but emulated -> dequant ceiling


def bw_ws(ws_bytes, gp):
    return gp["bw_l2_tbs"] if ws_bytes < gp["c_eff_mb"] * 2**20 else gp["bw_hbm_tbs"]


def predict_moe_us(T, gp, belem):
    """CARM latency for a MoE call at weight bytes/elem = belem (2 bf16, 1 fp8...)."""
    F = moe_flops(T)
    wbytes = moe_weight_bytes(belem)
    B = wbytes + moe_act_bytes(T)
    ws = 3 * H * I * belem                      # per-expert weight working set
    peak = (gp["peak_bf16_tflops"] if belem >= 2 else None)
    t_mem = B / (bw_ws(ws, gp) * 1e12) * 1e6
    t_comp = F / (peak * 1e12) * 1e6
    return gp["t0_us"] + max(t_mem, t_comp)


def predict_moe_quant_us(T, gp, mode):
    F = moe_flops(T)
    m = QUANT_MODES[mode]
    B = moe_weight_bytes(m["belem"]) + moe_act_bytes(T)
    ws = 3 * H * I * m["belem"]
    P = ceiling_tflops(gp, mode)
    t_mem = B / (bw_ws(ws, gp) * 1e12) * 1e6
    t_comp = F / (P * 1e12) * 1e6
    return gp["t0_us"] + max(t_mem, t_comp)


def moe_crossover(gp, mode, t_max=4096):
    """Smallest T where quant latency >= bf16 latency (quant stops winning).
    None => quant wins across the whole range."""
    won = False
    for T in range(8, t_max + 1, 2):
        q = predict_moe_quant_us(T, gp, mode)
        d = predict_moe_us(T, gp, 2.0)
        if q < d:
            won = True
        elif won:
            return T
    return None


def fit_ceilings(key, moe_rows):
    """Fit bf16 peak and dequant ceiling from the measured CUDA MoE sweep
    (achieved TFLOPS at the most compute-bound point). Generalized from the
    H100-only version for the B200 leg (2026-08-02)."""
    t0 = GPU_PARAMS[key]["t0_us"]
    peak_bf16 = max(moe_flops(r["T"]) / ((r["bf16"] - t0) * 1e-6) / 1e12 for r in moe_rows)
    deq = max(moe_flops(r["T"]) / ((r["fp8_w8a16"] - t0) * 1e-6) / 1e12 for r in moe_rows)
    GPU_PARAMS[key]["peak_bf16_tflops"] = round(peak_bf16, 1)
    GPU_PARAMS[key]["dequant_ceiling_tflops"] = round(deq, 1)
    return peak_bf16, deq


def fit_h100_ceilings(moe_rows):
    return fit_ceilings("h100", moe_rows)


def mape_moe(moe_rows, key="h100"):
    gp = GPU_PARAMS[key]
    errs_b, errs_q = [], []
    rows = []
    for r in moe_rows:
        T = r["T"]
        pb = predict_moe_us(T, gp, 2.0)
        pq = predict_moe_quant_us(T, gp, "w8a16")
        errs_b.append(abs(pb - r["bf16"]) / r["bf16"])
        errs_q.append(abs(pq - r["fp8_w8a16"]) / r["fp8_w8a16"])
        rows.append({"T": T, "bf16_meas": r["bf16"], "bf16_pred": round(pb, 1),
                     "fp8_meas": r["fp8_w8a16"], "fp8_pred": round(pq, 1)})
    return (round(sum(errs_b) / len(errs_b) * 100, 1),
            round(sum(errs_q) / len(errs_q) * 100, 1), rows)


def measured_moe_crossover(moe_rows):
    """Measured crossover: interpolate where fp8/bf16 ratio crosses 1.0."""
    prev = None
    for r in sorted(moe_rows, key=lambda x: x["T"]):
        ratio = r["fp8_vs_bf16"]
        if prev and prev[1] >= 1.0 > ratio:
            (T0, r0), (T1, r1) = prev, (r["T"], ratio)
            return int(round(T0 + (T1 - T0) * (r0 - 1.0) / (r0 - r1)))
        prev = (r["T"], ratio)
    return None


def main():
    from common import gpu_key
    key = gpu_key()
    print(f"GPU: {torch.cuda.get_device_name(0)} ({key})  torch={torch.__version__}\n")
    print("== Part 1: base CARM params on this CUDA stack ==")
    base = measure_base_params()
    print(json.dumps(base, indent=2))

    # 2026-08-02 B200 leg: replace the PROJECTED b200 row with measured values.
    # c_eff / gemm-class bw_l2 come from the goal-1 harness (same box, clock
    # note recorded there); bw_hbm / reduction bw_l2 / t0 from this stack.
    harness = os.path.join(_D, "..", "portable", f"params_nvidia-{key}.json")
    if key != "h100" and os.path.exists(harness):
        with open(harness) as f:
            hp = json.load(f)
        GPU_PARAMS[key]["c_eff_mb"] = hp["effective_l2_capacity_mb"]
        GPU_PARAMS[key]["bw_l2_tbs"] = hp["bw_l2_tbs"]
        print(f"  {key} c_eff/bw_l2 from harness: {hp['effective_l2_capacity_mb']} MB, "
              f"{hp['bw_l2_tbs']} TB/s")
    GPU_PARAMS[key]["bw_hbm_tbs"] = base["hbm_read_tbs"]
    GPU_PARAMS[key]["bw_l2_tbs_reduction"] = base["l2_read_tbs"]
    GPU_PARAMS[key]["t0_us"] = base["kernel_floor_us"]

    moe_path = os.path.join(_D, f"results_cuda_moe_{key}.json")
    if not os.path.exists(moe_path):
        moe_path = os.path.join(_D, "results_cuda_moe.json")
    with open(moe_path) as f:
        moe = json.load(f)
    moe_rows = moe["rows"]
    print(f"  MoE sweep: {os.path.basename(moe_path)}")

    print(f"\n== Part 2: fit {key} ceilings from measured CUDA MoE sweep ==")
    peak, deq = fit_ceilings(key, moe_rows)
    print(f"  fit peak_bf16 = {peak:.1f} TFLOPS")
    print(f"  fit dequant_ceiling (fp8 W8A16) = {deq:.1f} TFLOPS")

    # Native-FP4 peak from the measured W4A4 sweep (B200_RUNBOOK section 4):
    # replaces the projected native_peak_mult["fp4"] = 4.0.
    nv_path = os.path.join(_D, f"results_moe_nvfp4_native_{key}.json")
    fp4_fit = None
    if "fp4" in GPU_PARAMS[key]["native_mma"] and os.path.exists(nv_path):
        with open(nv_path) as f:
            nv = json.load(f)
        if "rows" in nv:
            t0 = GPU_PARAMS[key]["t0_us"]
            fp4_peak = max(moe_flops(r["T"]) / ((r["nvfp4_w4a4_us"] - t0) * 1e-6) / 1e12
                           for r in nv["rows"])
            mult = round(fp4_peak / GPU_PARAMS[key]["peak_bf16_tflops"], 2)
            GPU_PARAMS[key]["native_peak_mult"]["fp4"] = mult
            fp4_fit = {"fp4_peak_tflops": round(fp4_peak, 1),
                       "native_peak_mult_fp4": mult,
                       "note": "fit from measured W4A4 sweep; replaces projected 4.0"}
            print(f"  fit native fp4 peak = {fp4_peak:.1f} TFLOPS "
                  f"(mult {mult}x vs fitted bf16 peak; projected was 4.0x)")
    if key != "h100":
        GPU_PARAMS[key]["measured"] = True

    mb, mq, mrows = mape_moe(moe_rows, key)
    print(f"\n== MAPE (CARM vs measured CUDA MoE, {key}) ==")
    print(f"  bf16 MoE MAPE = {mb:.1f}%   fp8 W8A16 MoE MAPE = {mq:.1f}%")
    for r in mrows:
        print(f"    T={r['T']:4d}  bf16 meas={r['bf16_meas']:7.0f} pred={r['bf16_pred']:7.0f}   "
              f"fp8 meas={r['fp8_meas']:7.0f} pred={r['fp8_pred']:7.0f}")

    print(f"\n== MoE quant-vs-dense crossover (tokens) ==")
    meas_x = measured_moe_crossover(moe_rows)
    print(f"  {key} measured (fp8 W8A16):           T* ~= {meas_x}")
    for gpu in ("h100", "b200", "b100"):
        gp = GPU_PARAMS[gpu]
        if gp["peak_bf16_tflops"] is None:
            continue  # not fit in this run
        tag = "measured" if gp["measured"] else "PROJECTED"
        line = []
        for mode in QUANT_MODES:
            x = moe_crossover(gp, mode)
            line.append(f"{mode}:{'none(wins)' if x is None else f'T*={x}'}")
        print(f"  {gpu.upper():5s} [{tag}] predicted:  " + "  ".join(line))

    out = {
        "experiment": "C_carm_cuda_refit_gpu_parameterized",
        "gpu": torch.cuda.get_device_name(0), "gpu_key": key,
        "torch": torch.__version__,
        "base_params_cuda_stack": base,
        "model": "t = t0 + max(B/BW(WS), F/P); P=dequant-ceiling for weight-only, "
                 "native-peak for matched-precision iff precision in GPU native_mma",
        f"fit_{key}": {"peak_bf16_tflops": GPU_PARAMS[key]["peak_bf16_tflops"],
                       "dequant_ceiling_tflops": GPU_PARAMS[key]["dequant_ceiling_tflops"],
                       "native_fp4": fp4_fit},
        "mape": {"bf16_moe_pct": mb, "fp8_w8a16_moe_pct": mq, "rows": mrows},
        "moe_crossover_tokens": {
            f"{key}_measured_fp8_w8a16": meas_x,
            "predicted": {gpu: {m: moe_crossover(GPU_PARAMS[gpu], m) for m in QUANT_MODES}
                          for gpu in ("h100", "b200", "b100")
                          if GPU_PARAMS[gpu]["peak_bf16_tflops"] is not None},
        },
        "gpu_params": GPU_PARAMS,
        "clock_note": "B200 instance: clock lock unavailable (lgc denied w/ sudo); "
                      "sustained-load SM band 1237-1320 MHz (+/-3%)",
    }
    save_json(os.path.join(_D, f"carm_cuda_params_{key}.json" if key != "h100"
                           else "carm_cuda_params.json"), out)


if __name__ == "__main__":
    main()
