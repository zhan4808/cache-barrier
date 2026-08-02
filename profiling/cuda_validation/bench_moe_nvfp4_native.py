"""Experiment A-native — matched-precision NATIVE FP4 MoE (Blackwell leg).

This is the thesis-completing leg that CANNOT run on H100. The MXFP4 path in
bench_cuda_moe.py uses vLLM Marlin, which dequantizes FP4->bf16 in-kernel and
runs bf16 tensor cores on EVERY GPU (EMU is intrinsic to Marlin). The cache-aware
roofline predicts that a *matched-precision* W4A4 kernel using NATIVE FP4 tensor
cores breaks the in-core dequant ceiling and therefore keeps winning in the
compute-bound regime where the weight-only (Marlin) path crosses over and loses.

Native FP4 MMA exists only on SM100+ (B200/B100). This script:
  * on non-SM100 hardware: prints why it is skipped and records a stub (so it is
    safe to commit/run from the H100 box);
  * on SM100+: quantizes to NVFP4 and runs vLLM's cutlass FP4 MoE
    (run_cutlass_moe_fp4), graph-timed with the SAME methodology as
    bench_cuda_moe.py, so the native curve is directly comparable to the bf16 /
    fp8-W8A16 / mxfp4-W4A16-Marlin curves at the same Mixtral shape.

Comparison target: results_cuda_moe_<gpu>.json (bf16, fp8 W8A16, mxfp4 W4A16
Marlin) produced by bench_cuda_moe.py on the same box.

>>> Blackwell-only code below is marked `TODO(B200): validate on-device`. It is
    written against the vLLM 0.20.2 API surface (run_cutlass_moe_fp4,
    scaled_fp4_quant) but is UNTESTED on Hopper by construction; finalize it on
    the first B200 run.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    graph_med_us, env_versions, save_json, gpu_key, native_low_precisions,
)

DEV, DT = "cuda", torch.bfloat16
E, H, I, TOPK = 8, 4096, 14336, 2          # Mixtral shape (matches bench_cuda_moe.py)
TOKENS = [16, 64, 128, 256, 512, 640, 768, 896, 1024, 1536, 2048]
_D = os.path.dirname(os.path.abspath(__file__))


def _skip(reason):
    key = gpu_key()
    out = {
        "experiment": "A_native_nvfp4_w4a4_moe",
        "gpu": torch.cuda.get_device_name(0), "gpu_key": key,
        "status": "SKIPPED", "reason": reason,
        "note": "Native FP4 MMA requires SM100+ (B200/B100). Run this on Blackwell; "
                "on Hopper the matched-FP4 leg is physically impossible (no FP4 tensor "
                "cores) and the Marlin W4A16 path in bench_cuda_moe.py is the EMU stand-in.",
    }
    save_json(os.path.join(_D, f"results_moe_nvfp4_native_{key}.json"), out)
    print(f"SKIPPED: {reason}")


def main():
    ver = env_versions()
    key = gpu_key()
    print(f"GPU: {ver['gpu']} ({key})  torch={ver['torch']} vllm={ver['vllm']}")

    if "fp4" not in native_low_precisions():
        _skip(f"{key} has no native FP4 tensor cores (cap="
              f"{torch.cuda.get_device_capability(0)}).")
        return

    # ---- SM100+ native FP4 path -------------------------------------------
    # TODO(B200): validate on-device. Entry points confirmed present in vLLM
    # 0.20.2 source; weight/activation prep + workspace sizing to be pinned on
    # the first Blackwell run.
    try:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            run_cutlass_moe_fp4,
        )
        from vllm.model_executor.layers.fused_moe.config import MoEActivation
        import vllm._custom_ops as ops  # scaled_fp4_quant
    except Exception as exc:  # noqa: BLE001
        _skip(f"native FP4 MoE ops unavailable in this build: {type(exc).__name__}: {exc}. "
              "Blackwell vLLM must be built with cutlass FP4 ops (see B200_RUNBOOK.md).")
        return

    print("Native FP4 MoE ops present. Building NVFP4 weights (once, outside timed region)...")
    # Finalized 2026-08-02 on the B200 box against vLLM 0.26.0.
    # Recipe mirrors vllm/quantization/online/nvfp4.py (_quantize_moe_weight_to_nvfp4)
    # + oracle/nvfp4 VLLM_CUTLASS prep (swizzle_blockscale, neutral activation
    # gscales, alphas = w_scale_2 * a_input_scale with a_input_scale = 1).
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    F4_MAX, F8_MAX = 6.0, 448.0

    def quant_nvfp4_expertwise(w):
        """w [E,N,K] bf16 -> (fp4 [E,N,K//2] u8, swizzled sf, linear sf, scale_2 [E])."""
        e, nn, kk = w.shape
        amax = w.abs().amax(dim=(1, 2)).to(torch.float32).clamp_min(1e-8)
        gscale = (F4_MAX * F8_MAX) / amax
        scale_2 = (1.0 / gscale).to(torch.float32)
        scaled = (w.float() * gscale[:, None, None]).to(w.dtype).reshape(-1, kk)
        one = torch.ones((), device=w.device, dtype=torch.float32)
        qw, sf = ops.scaled_fp4_quant(scaled, one, is_sf_swizzled_layout=False)
        qw = qw.reshape(e, nn, kk // 2)
        sf = sf.reshape(e, nn, kk // 16)
        return qw, swizzle_blockscale(sf), sf, scale_2

    torch.manual_seed(0)
    g = torch.Generator(device=DEV).manual_seed(0)
    w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DT, generator=g) / H**0.5
    w2 = torch.randn(E, H, I, device=DEV, dtype=DT, generator=g) / I**0.5

    w1_fp4, w1_sf_sw, w1_sf, w1_s2 = quant_nvfp4_expertwise(w1)
    w2_fp4, w2_sf_sw, w2_sf, w2_s2 = quant_nvfp4_expertwise(w2)
    ones_e = torch.ones(E, device=DEV, dtype=torch.float32)
    # a_input_scale = 1 (neutral): alphas reduce to the weight scale_2s.
    a1_gscale, a2_gscale = ones_e, ones_e.clone()
    g1_alphas, g2_alphas = w1_s2.clone(), w2_s2.clone()
    print(f"  w1_fp4 {tuple(w1_fp4.shape)}  sf {tuple(w1_sf_sw.shape)}  "
          f"w2_fp4 {tuple(w2_fp4.shape)}  sf {tuple(w2_sf_sw.shape)}")

    # bf16 reference weights = dequantized NVFP4 (isolates kernel error).
    w1_ref = torch.stack([
        dequantize_to_dtype(w1_fp4[e], w1_sf[e], w1_s2[e], DT, swizzle=False)
        for e in range(E)])
    w2_ref = torch.stack([
        dequantize_to_dtype(w2_fp4[e], w2_sf[e], w2_s2[e], DT, swizzle=False)
        for e in range(E)])

    def make_routing(T, seed):
        gg = torch.Generator(device=DEV).manual_seed(seed)
        x = torch.randn(T, H, device=DEV, dtype=DT, generator=gg) / 10
        gating = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=gg)
        tw, ti = torch.topk(torch.softmax(gating, -1), TOPK, -1)
        tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
        return x, tw.contiguous(), ti.to(torch.int32).contiguous()

    def run_native(x, tw, ti, out, ws13, ws2):
        T = x.shape[0]
        run_cutlass_moe_fp4(
            output=out, a=x,
            a1_gscale=a1_gscale, w1_fp4=w1_fp4, w1_blockscale=w1_sf_sw,
            w1_alphas=g1_alphas,
            a2_gscale=a2_gscale, w2_fp4=w2_fp4, w2_blockscale=w2_sf_sw,
            w2_alphas=g2_alphas,
            topk_weights=tw, topk_ids=ti,
            activation=MoEActivation.SILU,
            workspace13=ws13, workspace2=ws2,
            m=T, n=I, k=H, e=E, device=x.device)
        return out

    # ---- correctness @T=128 vs bf16 fused_experts on dequantized weights ----
    xc, twc, tic = make_routing(128, seed=128)
    outc = torch.empty(128, H, device=DEV, dtype=DT)
    ws13 = torch.empty(128 * TOPK, max(2 * I, H), device=DEV, dtype=DT)
    ws2 = torch.empty(128 * TOPK, I, device=DEV, dtype=DT)
    run_native(xc, twc, tic, outc, ws13, ws2)
    ref = fused_experts(xc, w1_ref, w2_ref, twc, tic)
    rel = ((outc.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"Correctness @T=128 (rel-err vs bf16 on deq nvfp4 weights): {rel:.4f}")
    del w1_ref, w2_ref, ref
    torch.cuda.empty_cache()

    # ---- comparison bf16 curve from bench_cuda_moe.py results (same box) ----
    import json
    ref_path = os.path.join(_D, f"results_cuda_moe_{key}.json")
    bf16_ref = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            for r in json.load(f)["rows"]:
                bf16_ref[r["T"]] = r["bf16"]

    from common import graph_med_us as _g  # timing identical to bench_cuda_moe
    rows = []
    for T in TOKENS:
        x, tw, ti = make_routing(T, seed=T)
        out = torch.empty(T, H, device=DEV, dtype=DT)
        ws13 = torch.empty(T * TOPK, max(2 * I, H), device=DEV, dtype=DT)
        ws2 = torch.empty(T * TOPK, I, device=DEV, dtype=DT)
        ti64 = ti

        def fn():
            run_cutlass_moe_fp4(
                output=out, a=x,
                a1_gscale=a1_gscale, w1_fp4=w1_fp4, w1_blockscale=w1_sf_sw,
                w1_alphas=g1_alphas,
                a2_gscale=a2_gscale, w2_fp4=w2_fp4, w2_blockscale=w2_sf_sw,
                w2_alphas=g2_alphas,
                topk_weights=tw, topk_ids=ti,
                activation=MoEActivation.SILU,
                workspace13=ws13, workspace2=ws2,
                m=T, n=I, k=H, e=E, device=x.device)

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            us, timing = round(graph_med_us(fn), 1), "graph"
        except Exception as exc:  # noqa: BLE001
            from common import eager_med_us
            print(f"    [graph capture failed: {type(exc).__name__}: "
                  f"{str(exc)[:80]}; eager]")
            us, timing = round(eager_med_us(fn), 1), "eager"

        row = {"T": T, "nvfp4_w4a4_us": us, "timing": timing}
        if T in bf16_ref:
            row["bf16_us_ref"] = bf16_ref[T]
            row["nvfp4_vs_bf16"] = round(bf16_ref[T] / us, 3)
        rows.append(row)
        vs = f"  vs bf16 {row.get('nvfp4_vs_bf16', 'n/a')}x" if T in bf16_ref else ""
        print(f"T={T:>5}  nvfp4_w4a4={us:>8}u [{timing}]{vs}", flush=True)
        del x, tw, ti, ti64, out, ws13, ws2
        torch.cuda.empty_cache()

    out_obj = {
        "experiment": "A_native_nvfp4_w4a4_moe",
        "gpu": ver["gpu"], "gpu_key": key, "versions": ver,
        "shape": {"E": E, "H": H, "I": I, "topk": TOPK},
        "method": "graph_med_us 10 launches/graph, median of 40 replays "
                  "(identical to bench_cuda_moe.py); run_cutlass_moe_fp4 "
                  "(vLLM 0.26.0 cutlass SM100 grouped GEMM, W4A4 NVFP4, "
                  "block-16 fp8 scales, per-expert fp32 alphas)",
        "clock_note": "clock lock UNAVAILABLE on this instance; "
                      "sustained-load SM band 1237-1320 MHz (+/-3%)",
        "correctness_rel_err_T128": round(rel, 5),
        "bf16_reference": f"results_cuda_moe_{key}.json (same box/session)",
        "rows": rows,
    }
    save_json(os.path.join(_D, f"results_moe_nvfp4_native_{key}.json"), out_obj)


if __name__ == "__main__":
    main()
