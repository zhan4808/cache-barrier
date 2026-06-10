"""Graph-timed FlagGems MoE sweep at high T + warm-state token crossover (H100)."""

import json
import os
import statistics

import torch
import flag_gems  # noqa: F401
from flag_gems.fused.fused_moe import fused_experts_impl
from bench_fused_moe_mxq import make_case, quant_int8_per_channel

_D = os.path.dirname(os.path.abspath(__file__))
E, H, I, TOPK = 8, 4096, 14336, 2
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


def bench_t(T):
    x, w1, w2, tw, ti = make_case(T, E, H, I, TOPK, seed=T)
    w1_q, w1_s = quant_int8_per_channel(w1)
    w2_q, w2_s = quant_int8_per_channel(w2)

    def bf16():
        fused_experts_impl(x.clone(), w1, w2, tw, ti)

    def w816():
        fused_experts_impl(
            x.clone(), w1_q, w2_q, tw, ti,
            use_int8_w8a16=True, per_channel_quant=True,
            w1_scale=w1_s, w2_scale=w2_s,
        )

    def w88():
        fused_experts_impl(
            x.clone(), w1_q, w2_q, tw, ti,
            use_int8_w8a8=True, per_channel_quant=True,
            w1_scale=w1_s, w2_scale=w2_s,
        )

    row = {"T": T}
    for name, fn in [("bf16_us", bf16), ("w8a16_us", w816), ("w8a8_us", w88)]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        row[name.replace("_us", "")] = round(graph_med_us(fn), 1)
    row["w8a16_vs_bf16"] = round(row["bf16"] / row["w8a16"], 3)
    row["w8a8_vs_bf16"] = round(row["bf16"] / row["w8a8"], 3)
    del x, w1, w2, w1_q, w2_q
    torch.cuda.empty_cache()
    return row


def main():
    tokens = [16, 64, 128, 256, 512, 1024, 2048]
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)}  graph inner={N_GRAPH}")
    for T in tokens:
        r = bench_t(T)
        rows.append(r)
        print(
            f"T={T:4d}  bf16={r['bf16']:7.0f}u  w8a16={r['w8a16']:7.0f}u "
            f"({r['w8a16_vs_bf16']:.2f}x)  w8a8={r['w8a8']:7.0f}u ({r['w8a8_vs_bf16']:.2f}x)"
        )
    out = {"gpu": torch.cuda.get_device_name(0), "shape": {"E": E, "H": H, "I": I, "topk": TOPK}, "rows": rows}
    path = os.path.join(_D, "results_fused_moe_extended.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
