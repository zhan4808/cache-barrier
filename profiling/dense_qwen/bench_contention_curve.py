"""Contention-degree sweep — the continuous residency factor for CARM v2 (stretch).

Warm vs rotated is a binary; real serving sits between (a layer's weight is
re-read every forward, competing with the OTHER layers' weights + KV traffic).
Sweep the number of distinct weight copies R cycled per graph (co-tenancy
degree) and measure effective weight BW. The curve BW(R x size / C_eff) is the
contention factor CARM needs to interpolate microbenchmark -> serving.

Shapes: kv_proj (21 MB, resident @R=1) and q_proj fp8 side (31.5 MB, the
boundary case). Paths: bf16 mm + w8a8 mm-only. M=16.
Also: lm_head (5120x248k ~ 2.5 GB) single point -- the extreme streamed case.

Output: results_contention_h100.json
"""

import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, "..", "cuda_validation"))
from common import graph_med_us, env_versions, save_json  # noqa: E402

import vllm._custom_ops as ops  # noqa: E402

DEV, DT = "cuda", torch.bfloat16
T0 = 2.78
M = 16
COPIES = [1, 2, 3, 4, 6, 8, 12]
SHAPES = [("kv_proj", 5120, 2048), ("q_proj", 5120, 6144)]


def run(K, N, R, path):
    g = torch.Generator(device=DEV).manual_seed(N + R)
    ws = [torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5 for _ in range(R)]
    x = torch.randn(M, K, device=DEV, dtype=DT) / 32
    if path == "bf16":
        it = {"i": 0}
        def fn():
            torch.mm(x, ws[it["i"] % R].t()); it["i"] += 1
    else:
        fp8 = []
        for w in ws:
            wq, wsc = ops.scaled_fp8_quant(w)
            fp8.append((wq.t(), wsc))
        xq, xs = ops.scaled_fp8_quant(x)
        it = {"i": 0}
        def fn():
            wq, wsc = fp8[it["i"] % R]
            ops.cutlass_scaled_mm(xq, wq, xs, wsc, DT); it["i"] += 1
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    us = graph_med_us(fn)
    del ws
    torch.cuda.empty_cache()
    return us


def main():
    ver = env_versions()
    print(f"GPU {ver['gpu']}  M={M}  contention sweep R={COPIES}\n")
    rows = []
    for name, K, N in SHAPES:
        for path, bytes_per in [("bf16", 2), ("w8a8_mm", 1)]:
            mb = K * N * bytes_per / 1e6
            print(f"== {name} {path}  operand {mb:.1f} MB ==")
            for R in COPIES:
                us = run(K, N, R, path)
                bw = mb / max(us - T0, .05)
                rows.append({"shape": name, "path": path, "operand_mb": round(mb, 1),
                             "R": R, "total_ws_mb": round(mb * R, 1),
                             "us": round(us, 2), "bw_tbs": round(bw, 2)})
                print(f"  R={R:2d}  total WS {mb*R:7.1f} MB   {us:7.2f} us   {bw:5.2f} TB/s")
    # lm_head extreme point
    K, N = 5120, 248832
    g = torch.Generator(device=DEV).manual_seed(1)
    w = torch.randn(N, K, device=DEV, dtype=DT, generator=g) / K**0.5
    x = torch.randn(M, K, device=DEV, dtype=DT) / 32
    fn = lambda: torch.mm(x, w.t())
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    us = graph_med_us(fn)
    mb = K * N * 2 / 1e6
    rows.append({"shape": "lm_head", "path": "bf16", "operand_mb": round(mb, 1),
                 "R": 1, "total_ws_mb": round(mb, 1), "us": round(us, 2),
                 "bw_tbs": round(mb / (us - T0), 2)})
    print(f"\nlm_head 5120x248832 ({mb:.0f} MB): {us:.1f} us  {mb/(us-T0):.2f} TB/s")
    save_json(os.path.join(_D, "results_contention_h100.json"), {
        "experiment": "contention_degree_sweep", "gpu": ver["gpu"], "M": M,
        "rows": rows})


if __name__ == "__main__":
    main()
