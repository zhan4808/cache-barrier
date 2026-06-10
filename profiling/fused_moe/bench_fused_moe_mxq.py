"""
FlagGems PR #2336 (fused_moe_mxq) analysis on H100 — correctness + token scaling.

Compares, on Mixtral-like MoE shapes (E=8, H=4096, I=14336, topk=2):
  bf16        FlagGems main fused_experts_impl (vLLM-adapted tiled kernel, tl.dot)
  w8a16       same path with use_int8_w8a16 (tiled GPTQ/AWQ-style dequant + tl.dot)
  w4a16       same path with use_int4_w4a16 (INT4 in INT8 containers)
  mxq_w8a16   PR kernel: fused_moe_kernel_gptq_awq, grid=(T*topk,), BLOCK_M=1, no tl.dot
  mxq_w4a16   PR kernel, W4A16 (packed) config

Correctness is checked against a torch reference before timing; kernels that
produce wrong results are still timed (the PR's own benchmark times them) but
flagged, since their latency reflects less work than a correct kernel would do.

Outputs: results_fused_moe_mxq.json
"""

import json
import os
import statistics

import torch

import flag_gems  # noqa: F401  (registers device)
from flag_gems.fused.fused_moe import fused_experts_impl
from flag_gems.fused_moe_mxq import (
    QuantConfig,
    QuantMode,
    fused_moe as mxq_fused_moe,
    quantize_weights_moe,
)

DEV = "cuda"
# fp16, not bf16: the mxq kernel hardcodes compute_type=tl.float16 and
# tl.atomic_add into the output, which fails to compile against bf16 buffers
# (the PR's own bf16 benchmark config cannot have run this path as written).
DTYPE = torch.float16
_D = os.path.dirname(os.path.abspath(__file__))


def make_case(T, E, H, I, topk, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=DTYPE, generator=g) / 10
    w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DTYPE, generator=g) / H**0.5
    w2 = torch.randn(E, H, I, device=DEV, dtype=DTYPE, generator=g) / I**0.5
    gating = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=g)
    tw, ti = torch.topk(torch.softmax(gating, -1), topk, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(DTYPE)
    return x, w1, w2, tw, ti


def ref_moe(x, w1, w2, tw, ti):
    """Reference SwiGLU MoE in fp32."""
    T, H = x.shape
    I = w2.shape[2]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    for t in range(T):
        for k in range(ti.shape[1]):
            e = int(ti[t, k])
            h = w1f[e] @ xf[t]                       # [2I]
            act = torch.nn.functional.silu(h[: I]) * h[I:]
            out[t] += float(tw[t, k]) * (w2f[e] @ act)
    return out


def quant_int8_per_channel(w):
    """Symmetric per-output-channel INT8: w ~ q * scale."""
    s = w.float().abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(w.float() / s).clamp(-128, 127).to(torch.int8)
    return q, s.squeeze(-1)


def quant_int4_unpacked(w):
    """Symmetric per-channel INT4 stored in INT8 containers (FlagGems main path)."""
    s = w.float().abs().amax(-1, keepdim=True) / 7.0
    q = torch.round(w.float() / s).clamp(-8, 7).to(torch.int8)
    return q, s.squeeze(-1)


def build_impls(x, w1, w2, tw, ti, E, topk):
    w1_q8, w1_s8 = quant_int8_per_channel(w1)
    w2_q8, w2_s8 = quant_int8_per_channel(w2)
    w1_q4, w1_s4 = quant_int4_unpacked(w1)
    w2_q4, w2_s4 = quant_int4_unpacked(w2)

    mxq8 = QuantConfig(mode=QuantMode.W8A16, has_zero_point=False)
    mxq4 = QuantConfig(mode=QuantMode.W4A16, has_zero_point=False)
    m8_w1, m8_w1s, _ = quantize_weights_moe(w1, E, mxq8)
    m8_w2, m8_w2s, _ = quantize_weights_moe(w2, E, mxq8)
    m4_w1, m4_w1s, _ = quantize_weights_moe(w1, E, mxq4)
    m4_w2, m4_w2s, _ = quantize_weights_moe(w2, E, mxq4)

    return {
        "bf16": lambda: fused_experts_impl(x.clone(), w1, w2, tw, ti),
        "w8a16": lambda: fused_experts_impl(
            x.clone(), w1_q8, w2_q8, tw, ti,
            use_int8_w8a16=True, per_channel_quant=True,
            w1_scale=w1_s8, w2_scale=w2_s8),
        "w4a16": lambda: fused_experts_impl(
            x.clone(), w1_q4, w2_q4, tw, ti,
            use_int4_w4a16=True, per_channel_quant=True,
            w1_scale=w1_s4, w2_scale=w2_s4),
        "mxq_w8a16": lambda: mxq_fused_moe(
            x, w1=None, w2=None, w3=None, topk_weights=tw, topk_ids=ti,
            quant_config=mxq8, num_experts=E, top_k=topk,
            w1_q=m8_w1, w1_scales=m8_w1s, w2_q=m8_w2, w2_scales=m8_w2s),
        "mxq_w4a16": lambda: mxq_fused_moe(
            x, w1=None, w2=None, w3=None, topk_weights=tw, topk_ids=ti,
            quant_config=mxq4, num_experts=E, top_k=topk,
            w1_q=m4_w1, w1_scales=m4_w1s, w2_q=m4_w2, w2_scales=m4_w2s),
    }


def check_correctness():
    """Small shape so the torch reference is fast: T=4, E=4, H=256, I=512."""
    T, E, H, I, topk = 4, 4, 256, 512, 2
    x, w1, w2, tw, ti = make_case(T, E, H, I, topk)
    ref = ref_moe(x, w1, w2, tw, ti)
    impls = build_impls(x, w1, w2, tw, ti, E, topk)
    report = {}
    for name, fn in impls.items():
        try:
            out = fn().float()
            if out.shape != ref.shape:
                report[name] = f"WRONG SHAPE {tuple(out.shape)} vs {tuple(ref.shape)}"
                continue
            rel = ((out - ref).norm() / ref.norm()).item()
            report[name] = f"rel_err={rel:.4f}" + ("  [FAIL >0.05]" if rel > 0.05 else "  [ok]")
        except Exception as exc:
            report[name] = f"EXCEPTION: {type(exc).__name__}: {str(exc)[:120]}"
    return report


def time_us(fn, iters=30, reps=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters * 1000)
    return statistics.median(ts)


def perf_sweep():
    E, H, I, topk = 8, 4096, 14336, 2
    tokens = [1, 4, 16, 64, 128, 256, 512]
    rows = []
    for T in tokens:
        x, w1, w2, tw, ti = make_case(T, E, H, I, topk, seed=T)
        impls = build_impls(x, w1, w2, tw, ti, E, topk)
        row = {"tokens": T}
        flops = 2 * T * topk * (2 * H * 2 * I + H * I)  # gemm1(2I)+gemm3(I), x2 mul-add
        for name, fn in impls.items():
            iters = 30 if T <= 128 else 10
            try:
                us = time_us(fn, iters=iters)
                row[name + "_us"] = round(us, 1)
                row[name + "_tflops"] = round(flops / (us * 1e-6) / 1e12, 2)
            except Exception as exc:
                row[name + "_us"] = None
                row[name + "_err"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        rows.append(row)
        print(row)
        del x, w1, w2, impls
        torch.cuda.empty_cache()
    return rows


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n== correctness (T=4, E=4, H=256, I=512, topk=2) ==")
    corr = check_correctness()
    for k, v in corr.items():
        print(f"  {k:12s} {v}")

    print("\n== perf sweep: Mixtral shape (E=8, H=4096, I=14336, topk=2) ==")
    rows = perf_sweep()

    out = os.path.join(_D, "results_fused_moe_mxq.json")
    with open(out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__,
                   "correctness": corr, "sweep": rows}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
