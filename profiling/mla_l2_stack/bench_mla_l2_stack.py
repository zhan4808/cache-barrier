"""
Multi-layer MLA reconstruction stacking: how L2 residency changes with depth.

Simulates N sequential reconstruction BMMs (same 16 MB/layer MLA shape) in one
CUDA graph. Total weight working set = N * 16 MB; CARM predicts the cliff at
C_eff ≈ 36 MB (between 2 and 3 layers).

Outputs: results_mla_l2_stack.json
"""

import json
import os
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
_PROF = os.path.join(_D, "..")
sys.path.insert(0, _PROF)
sys.path.insert(0, os.path.join(_PROF, "w8a8"))
sys.path.insert(0, os.path.join(_D, "..", "..", "..", "kernel-compass"))

from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4  # noqa: E402
from w8a8_bmm import quantize_acts_w8, quantize_weights_w8, w8a8_bmm  # noqa: E402
from profiling.carm import predict_fp16_recon_us  # noqa: E402

H, K, N = 128, 128, 512
LAYER_MB = H * K * N * 2 / 2**20
N_GRAPH = 10


def graph_med_us(fn, n_inner=N_GRAPH, reps=40):
    for _ in range(5):
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
    return statistics.median(ts) / n_inner * 1000


def bench_layers(n_layers: int):
    a = torch.randn(H, 1, K, dtype=torch.float16, device="cuda") / 4
    weights = [torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8 for _ in range(n_layers)]
    wq_ws = [quantize_weights_w8(w) for w in weights]
    wp_sc = [quantize_weights_int4(w) for w in weights]
    qbuf = torch.empty_like(a, dtype=torch.int8)
    sbuf = torch.empty(H, dtype=torch.float32, device="cuda")
    obuf = torch.empty(H, 1, N, dtype=torch.float16, device="cuda")

    def fp16():
        for w in weights:
            torch.bmm(a, w)

    def w8a8():
        quantize_acts_w8(a, qbuf, sbuf)
        for wq, ws in wq_ws:
            w8a8_bmm(qbuf, wq, sbuf, ws, obuf)

    def w4a16():
        for wp, sc in wp_sc:
            batched_int4_gemm(a, wp, sc, K)

    for fn in (fp16, w8a8, w4a16):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

    fp16_us = graph_med_us(fp16)
    w8a8_us = graph_med_us(w8a8)
    w4_us = graph_med_us(w4a16)

    flops = 2 * H * 1 * K * N * n_layers
    fp16_bytes = sum(H * K * N * 2 + H * K * 2 + H * N * 2 for _ in range(n_layers))
    carm_pred = predict_fp16_recon_us(flops, fp16_bytes)

    row = {
        "n_layers": n_layers,
        "weight_mb": round(LAYER_MB * n_layers, 1),
        "fp16_us": round(fp16_us, 2),
        "w8a8_us": round(w8a8_us, 2),
        "w4a16_us": round(w4_us, 2),
        "w8a8_speedup": round(fp16_us / w8a8_us, 3),
        "carm_fp16_pred_us": round(carm_pred, 2),
        "l2_served": LAYER_MB * n_layers < 36,
    }
    del a, weights, wq_ws, wp_sc, qbuf, sbuf, obuf
    torch.cuda.empty_cache()
    return row


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  layer={LAYER_MB:.1f} MB")
    rows = []
    for n in [1, 2, 3, 4, 5, 6]:
        r = bench_layers(n)
        rows.append(r)
        print(
            f"L={n} ({r['weight_mb']:5.1f} MB)  fp16={r['fp16_us']:6.1f}  "
            f"w8a8={r['w8a8_us']:6.1f} ({r['w8a8_speedup']:.2f}x)  "
            f"w4a16={r['w4a16_us']:6.1f}  carm={r['carm_fp16_pred_us']:.1f}  "
            f"L2={'yes' if r['l2_served'] else 'no'}"
        )
    out = {"gpu": torch.cuda.get_device_name(0), "layer_mb": LAYER_MB, "rows": rows}
    path = os.path.join(_D, "results_mla_l2_stack.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
