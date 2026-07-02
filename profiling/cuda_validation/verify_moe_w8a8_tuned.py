"""Rigor pass: W8A8 MoE vs a TUNED bf16 baseline, clock-locked (night item 8).

The bench_moe_w8a8 headline (Mixtral 3.15x @T=512) is measured against STOCK
vLLM bf16, whose GROUP_SIZE_M=1 default is under-tuned in the T=256-512 band
(REPORT S7). Re-measure w8a8 against bf16 with GROUP_SIZE_M forced to 16 (the
fair baseline), SM clock locked, median of 3 interleaved rounds.

Expected: w8a8 win compresses at mid-T (3.15x -> ~1.9-2.0x) but the "wins at
every T, no cliff" claim survives.
"""

import contextlib
import os
import statistics
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from common import graph_med_us, save_json  # noqa: E402

import vllm.model_executor.layers.fused_moe.fused_moe as VF  # noqa: E402
from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config  # noqa: E402
import bench_moe_w8a8 as W  # noqa: E402  (reuse quant + routing helpers)

E, H, I, TOPK = 8, 4096, 14336, 2
GRID = [128, 256, 512, 768, 1024, 2048]


@contextlib.contextmanager
def force_group_m(g):
    orig = VF.try_get_optimal_moe_config

    def patched(*a, **k):
        cfg = dict(orig(*a, **k)); cfg["GROUP_SIZE_M"] = g; return cfg
    VF.try_get_optimal_moe_config = patched
    try:
        yield
    finally:
        VF.try_get_optimal_moe_config = orig


def main():
    print(f"GPU {torch.cuda.get_device_name(0)} — W8A8 vs TUNED bf16 (clock-locked)")
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16, generator=g) / H**0.5
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) / I**0.5
    w1q, w1s, _ = W.quant_experts(w1)
    w2q, w2s, _ = W.quant_experts(w2)
    qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s)

    acc = {T: {"bf16_def": [], "bf16_g16": [], "w8a8": []} for T in GRID}
    for rnd in range(3):
        for T in GRID:
            x, tw, ti = W.routing(T, E, H, TOPK, seed=T)
            runs = [("bf16_def", lambda: fused_experts(x, w1, w2, tw, ti), None),
                    ("bf16_g16", lambda: fused_experts(x, w1, w2, tw, ti), 16),
                    ("w8a8", lambda: fused_experts(x, w1q, w2q, tw, ti, quant_config=qc), None)]
            order = runs[rnd % 3:] + runs[:rnd % 3]
            for tag, fn, gm in order:
                ctx = force_group_m(gm) if gm else contextlib.nullcontext()
                with ctx:
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    acc[T][tag].append(graph_med_us(fn))
            del x, tw, ti
            torch.cuda.empty_cache()
        print(f"  round {rnd+1}/3 done")

    rows = []
    print(f"\n{'T':>6} {'bf16_def':>9} {'bf16_g16':>9} {'w8a8':>8} {'vs_def':>7} {'vs_g16':>7}")
    for T in GRID:
        m = {k: statistics.median(v) for k, v in acc[T].items()}
        r = {"T": T, **{k: round(v, 1) for k, v in m.items()},
             "w8a8_vs_stock": round(m["bf16_def"] / m["w8a8"], 2),
             "w8a8_vs_tuned": round(min(m["bf16_def"], m["bf16_g16"]) / m["w8a8"], 2)}
        rows.append(r)
        print(f"{T:>6} {m['bf16_def']:>9.0f} {m['bf16_g16']:>9.0f} {m['w8a8']:>8.0f} "
              f"{r['w8a8_vs_stock']:>6.2f}x {r['w8a8_vs_tuned']:>6.2f}x")

    save_json(os.path.join(_D, "results_moe_w8a8_tuned.json"), {
        "experiment": "w8a8_vs_tuned_bf16_mixtral_clock_locked",
        "gpu": torch.cuda.get_device_name(0),
        "method": "SM 1755 locked; 3 interleaved rounds rotated order; median; "
                  "bf16_g16 = GROUP_SIZE_M=16 forced (fair baseline per REPORT S7)",
        "rows": rows})


if __name__ == "__main__":
    main()
