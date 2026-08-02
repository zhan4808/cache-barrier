"""
B200_RUNBOOK section 2 — sanity probe: does this build expose native FP4?

Written against vLLM 0.26.0 (fresh Blackwell env, torch 2.11+cu130) — the
runbook's import paths were speculated from 0.20.2 and are probed defensively
here; every check prints rather than asserts so the output is a report.
"""
import importlib
import json

report = {}

import torch
report["torch"] = torch.__version__
report["device"] = torch.cuda.get_device_name(0)
report["capability"] = list(torch.cuda.get_device_capability(0))

import vllm
report["vllm"] = vllm.__version__

from vllm.platforms import current_platform as p
report["platform_capability"] = str(p.get_device_capability())
try:
    report["is_family_100"] = bool(p.is_device_capability_family(100))
except Exception as e:
    report["is_family_100"] = f"api missing: {e}"

import vllm._custom_ops as ops
fp4_ops = sorted(n for n in dir(ops) if "fp4" in n.lower() or "nvfp4" in n.lower())
report["fp4_custom_ops"] = fp4_ops
report["cutlass_moe_fp4"] = hasattr(ops, "cutlass_moe_fp4")
report["scaled_fp4_quant"] = hasattr(ops, "scaled_fp4_quant")

try:
    from vllm import _custom_ops
    report["cutlass_scaled_fp4_mm"] = hasattr(_custom_ops, "cutlass_scaled_fp4_mm")
except Exception:
    pass

# oracle module paths moved between versions; try candidates
for mod, name in [
    ("vllm.model_executor.layers.fused_moe.oracle.mxfp4", "select_mxfp4_moe_backend"),
    ("vllm.model_executor.layers.quantization.mxfp4", "Mxfp4Backend"),
    ("vllm.model_executor.layers.fused_moe.cutlass_moe", "run_cutlass_moe_fp4"),
]:
    try:
        m = importlib.import_module(mod)
        report[f"{mod}.{name}"] = hasattr(m, name)
    except Exception as e:
        report[f"{mod}"] = f"import failed: {type(e).__name__}"

try:
    import flashinfer
    report["flashinfer"] = flashinfer.__version__
except Exception as e:
    report["flashinfer"] = f"missing: {type(e).__name__}"
try:
    import deep_gemm
    report["deep_gemm"] = getattr(deep_gemm, "__version__", "present")
except Exception as e:
    report["deep_gemm"] = f"missing: {type(e).__name__}"

print(json.dumps(report, indent=1))
