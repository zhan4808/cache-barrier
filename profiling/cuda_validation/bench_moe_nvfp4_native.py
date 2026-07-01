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
    # NVFP4 block scale = 16 elems/scale; weights [E, N, K] -> fp4 uint8 [E, N, K//2]
    # plus per-block fp8_e4m3 blockscales and per-expert fp32 alphas/gscales.
    #
    # Canonical prep (finalize on B200):
    #   w1_fp4, w1_blockscale, w1_alpha, a1_gscale = quant_nvfp4_expertwise(w1)
    #   w2_fp4, w2_blockscale, w2_alpha, a2_gscale = quant_nvfp4_expertwise(w2)
    # using ops.scaled_fp4_quant for the activation path and the modelopt/
    # compressed-tensors nvfp4 weight packer for weights. See B200_RUNBOOK.md §3.
    #
    # run_cutlass_moe_fp4(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alpha,
    #                     a2_gscale, w2_fp4, w2_blockscale, w2_alpha,
    #                     topk_weights, topk_ids, MoEActivation.SiLU (gated),
    #                     workspace13, workspace2, m, n, k, e, device)
    #
    # Timing loop mirrors bench_cuda_moe.py exactly:
    #   for T in TOKENS: nvfp4_us = graph_med_us(lambda: run_cutlass_moe_fp4(...))
    #   row = {"T": T, "nvfp4_w4a4_us": nvfp4_us,
    #          "nvfp4_vs_bf16": bf16_from_ref_json / nvfp4_us}
    #
    # Correctness: rel-err vs bf16 fused_experts on dequantized nvfp4 weights.
    raise NotImplementedError(
        "SM100 detected: finalize the native NVFP4 weight prep + run_cutlass_moe_fp4 "
        "call on the B200 box (structure and entry points are in place above; "
        "see B200_RUNBOOK.md §3). Left as NotImplementedError so it fails loudly "
        "rather than silently emitting wrong numbers.")


if __name__ == "__main__":
    main()
