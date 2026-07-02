"""Matched-precision W8A8 (fp8) MoE — completes Dr. Xiao todo #3 on the MoE path.

Prediction (CARM + the dense result): matched-precision fp8 (native FP8 MMA,
no in-kernel dequant-to-bf16) has NO dequant cliff, so unlike Marlin W8A16
(crosses over ~300-600 tok and loses 0.6x at T=2048 on Mixtral) it should hold
its win or tie at large T. This is the same W8A16-vs-W8A8 contrast the dense
GEMMs showed (dequant ceiling 334 TFLOPS vs fp8-MMA 1335 TFLOPS), now on the
MoE operator itself.

Path: vLLM triton fused_experts with fp8_w8a8_moe_quant_config (per-tensor
weight scales, dynamic activation quant inside the kernel) -- the same family
as vLLM's shipped fp8_w8a8 MoE configs. bf16 baseline = stock fused_experts
(same baseline as results_cuda_moe.json; W8A16 numbers for comparison come
from that committed reference).

Shapes: Mixtral (coarse, the crossover case) + Qwen3.6-35B-A3B (fine-grained
target, the always-quantize case).

Output: results_moe_w8a8_<gpu_key>.json
"""

import os
import sys

import torch

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from common import graph_med_us, env_versions, save_json, gpu_key  # noqa: E402

from vllm.model_executor.layers.fused_moe import fused_experts  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config  # noqa: E402

DEV, DT = "cuda", torch.bfloat16
FP8 = torch.float8_e4m3fn
TOKENS = [16, 64, 128, 256, 512, 1024, 2048]
SHAPES = [("mixtral", 8, 4096, 14336, 2),
          ("qwen3.6-35B-A3B", 256, 2048, 512, 8)]


def quant_experts(w):
    """Per-expert per-tensor e4m3 quant. w [E,N,K] bf16 -> (w_fp8, scales[E], ref)."""
    E = w.shape[0]
    qs, ss, refs = [], [], []
    for e in range(E):
        amax = w[e].abs().amax().clamp(min=1e-6)
        s = (amax / torch.finfo(FP8).max).float()
        q = (w[e].float() / s).clamp(torch.finfo(FP8).min, torch.finfo(FP8).max).to(FP8)
        qs.append(q); ss.append(s); refs.append((q.float() * s).to(DT))
    return torch.stack(qs), torch.stack(ss).view(E), torch.stack(refs)


def routing(T, E, H, topk, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=DT, generator=g) / 10
    gate = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=g)
    tw, ti = torch.topk(torch.softmax(gate, -1), topk, -1)
    return x, (tw / tw.sum(-1, keepdim=True)).float().contiguous(), ti.to(torch.int32).contiguous()


def main():
    ver = env_versions()
    key = gpu_key()
    print(f"GPU {ver['gpu']} ({key})  vllm={ver['vllm']}\n")
    all_res = []
    for name, E, H, I, topk in SHAPES:
        g = torch.Generator(device=DEV).manual_seed(0)
        w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DT, generator=g) / H**0.5
        w2 = torch.randn(E, H, I, device=DEV, dtype=DT, generator=g) / I**0.5
        w1q, w1s, w1r = quant_experts(w1)
        w2q, w2s, w2r = quant_experts(w2)
        qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s)

        # correctness @T=128 vs bf16 fused_experts on dequantized weights
        xc, twc, tic = routing(128, E, H, topk, seed=128)
        out8 = fused_experts(xc, w1q, w2q, twc, tic, quant_config=qc)
        ref = fused_experts(xc, w1r, w2r, twc, tic)
        rel = ((out8.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"== {name}  E={E} H={H} I={I} topk={topk}  w8a8 relerr={rel:.4f} ==")

        rows = []
        for T in TOKENS:
            x, tw, ti = routing(T, E, H, topk, seed=T)
            f_bf = lambda: fused_experts(x, w1, w2, tw, ti)
            f_88 = lambda: fused_experts(x, w1q, w2q, tw, ti, quant_config=qc)
            for f in (f_bf, f_88):
                for _ in range(3):
                    f()
            torch.cuda.synchronize()
            bf = round(graph_med_us(f_bf), 1)
            q8 = round(graph_med_us(f_88), 1)
            rows.append({"T": T, "bf16": bf, "w8a8": q8,
                         "w8a8_vs_bf16": round(bf / q8, 3)})
            print(f"  T={T:5d}  bf16={bf:8.0f}u  w8a8={q8:8.0f}u  ({bf/q8:.2f}x)")
            del x, tw, ti
            torch.cuda.empty_cache()
        all_res.append({"shape": name, "E": E, "H": H, "I": I, "topk": topk,
                        "relerr": round(rel, 4), "rows": rows})
        del w1, w2, w1q, w2q, w1r, w2r
        torch.cuda.empty_cache()
        print()

    save_json(os.path.join(_D, f"results_moe_w8a8_{key}.json"), {
        "experiment": "moe_w8a8_matched_precision",
        "gpu": ver["gpu"], "gpu_key": key, "versions": ver,
        "method": "vllm triton fused_experts + fp8_w8a8_moe_quant_config "
                  "(per-tensor weight scales, dynamic act quant in-kernel); "
                  "graph_med_us 10/graph median-of-40; bf16 baseline = stock "
                  "fused_experts (same as results_cuda_moe.json; W8A16 "
                  "comparison numbers live there)",
        "results": all_res,
    })


if __name__ == "__main__":
    main()
