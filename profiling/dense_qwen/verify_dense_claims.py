"""Red-team the two headline dense claims (clock-locked, rotated order, median-of-3).

Claim A (boundary super-win): q/o_proj (63 MB bf16 -> 31 MB fp8), M<=16, warm:
  w8a8 mm-only speedup > byte-ratio 2.0x, driven by an L2 BW-tier jump
  (fp8 ~4.2-4.7 TB/s vs bf16 ~2.8); compresses to <2.0x under rotation.
Claim B (left-edge flip): kv_proj (21 MB) warm quant loses at M<=16, but
  W8A16 flips to a 1.1-1.2x WIN under rotation (eviction makes it HBM-streamed).
Also: recheck the kv_proj w8a16 M=64 warm anomaly (0.53x).

Method: SM clock LOCKED (caller: sudo nvidia-smi -lgc 1755,1755), three
interleaved rounds with rotated path order, median per cell. Same kernels and
prep as bench_dense_proj.py.
"""

import os
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, save_json  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    marlin_quant_fp8_torch, marlin_make_workspace_new, apply_fp8_marlin_linear,
)

DEV, DT = "cuda", torch.bfloat16
C_EFF_MB = 36.0
T0 = 2.78


def clock():
    import subprocess
    o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    return o


def setup(K, N, rotate):
    n_cop = 1 if not rotate else max(2, int(2 * C_EFF_MB * 1e6 / (K * N * 2)) + 1)
    g = torch.Generator(device=DEV).manual_seed(K * 7 + N)
    ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(n_cop)]
    marl = [marlin_quant_fp8_torch(w, -1) for w in ws]
    fp8 = []
    for w in ws:
        wq, wsc = ops.scaled_fp8_quant(w)
        fp8.append((wq.t(), wsc))
    return ws, marl, fp8, marlin_make_workspace_new(DEV), n_cop


def time_cell(K, N, M, path, rotate):
    ws, marl, fp8, wspace, n_cop = setup(K, N, rotate)
    x = torch.randn(M, K, device=DEV, dtype=DT) / 32
    xq, xs = ops.scaled_fp8_quant(x)
    it = {"i": 0}
    if path == "bf16":
        fn = lambda: (torch.mm(x, ws[it.__setitem__("i", it["i"] + 1) or it["i"] % n_cop].t()))
        def fn():
            torch.mm(x, ws[it["i"] % n_cop].t()); it["i"] += 1
    elif path == "w8a16":
        def fn():
            _, qw, s = marl[it["i"] % n_cop]
            apply_fp8_marlin_linear(x, qw, s, wspace, N, K, None); it["i"] += 1
    else:  # w8a8_mm
        def fn():
            wq, wsc = fp8[it["i"] % n_cop]
            ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    us = graph_med_us(fn)
    del ws, marl, fp8
    torch.cuda.empty_cache()
    return us


def med3(K, N, M, paths, rotate):
    """Three rounds, path order rotated each round; median per path."""
    acc = {p: [] for p in paths}
    for r in range(3):
        order = paths[r % len(paths):] + paths[:r % len(paths)]
        for p in order:
            acc[p].append(time_cell(K, N, M, p, rotate))
    return {p: round(statistics.median(v), 2) for p, v in acc.items()}


def main():
    print(f"SM clock: {clock()} (expect locked 1755 MHz)\n")
    out = {"clock": clock()}

    print("== Claim A: boundary super-win (w8a8 mm-only vs bf16; byte ratio = 2.0) ==")
    A = []
    for name, K, N in [("q_proj", 5120, 6144), ("o_proj", 6144, 5120)]:
        for M in (1, 16):
            w = med3(K, N, M, ["bf16", "w8a8_mm"], rotate=False)
            r = med3(K, N, M, ["bf16", "w8a8_mm"], rotate=True)
            sw, sr = w["bf16"] / w["w8a8_mm"], r["bf16"] / r["w8a8_mm"]
            bw_w = (K * N / 1e6) / max(w["w8a8_mm"] - T0, .05)
            bw_r = (K * N / 1e6) / max(r["w8a8_mm"] - T0, .05)
            A.append({"shape": name, "M": M, "warm": w, "rot": r,
                      "speedup_warm": round(sw, 2), "speedup_rot": round(sr, 2),
                      "fp8_bw_warm_tbs": round(bw_w, 2), "fp8_bw_rot_tbs": round(bw_r, 2)})
            print(f"  {name} M={M:2d}: warm {sw:.2f}x (fp8 BW {bw_w:.2f} TB/s)  "
                  f"rotated {sr:.2f}x (BW {bw_r:.2f})  super={'YES' if sw > 2.1 else 'no'}")
    out["claimA_super_win"] = A

    print("\n== Claim B: kv_proj left-edge flip (w8a16 vs bf16) ==")
    B = []
    K, N = 5120, 2048
    for M in (1, 4, 16, 64):
        w = med3(K, N, M, ["bf16", "w8a16"], rotate=False)
        r = med3(K, N, M, ["bf16", "w8a16"], rotate=True)
        xw, xr = w["bf16"] / w["w8a16"], r["bf16"] / r["w8a16"]
        B.append({"M": M, "warm": w, "rot": r,
                  "w8a16_x_warm": round(xw, 2), "w8a16_x_rot": round(xr, 2)})
        print(f"  M={M:3d}: warm {xw:.2f}x   rotated {xr:.2f}x   "
              f"flip={'YES' if xw < 1.0 <= xr else 'no'}")
    out["claimB_flip"] = B

    save_json(os.path.join(_D, "results_verify_dense.json"), out)


if __name__ == "__main__":
    main()
