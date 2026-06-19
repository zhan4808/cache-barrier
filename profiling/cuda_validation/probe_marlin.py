"""De-risking probe: build a tiny Marlin MoE (fp8_w8a16 + mxfp4_w4a16) end-to-end
and check rel-err vs a bf16 fused_experts reference on the SAME dequantized weights.

If this passes, the full sweep (bench_cuda_moe.py) is the same flow at the real shape.
"""
import torch
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import marlin_quant_fp8_torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like

DEV, DT = "cuda", torch.bfloat16
FP8_ID = scalar_types.float8_e4m3fn.id
FP4_ID = scalar_types.float4_e2m1f.id


def make_case(T, E, H, I, topk, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=DT, generator=g) / 10
    w1 = torch.randn(E, 2 * I, H, device=DEV, dtype=DT, generator=g) / H**0.5   # [E,N=2I,K=H]
    w2 = torch.randn(E, H, I, device=DEV, dtype=DT, generator=g) / I**0.5       # [E,N=H,K=I]
    gating = torch.randn(T, E, device=DEV, dtype=torch.float32, generator=g)
    tw, ti = torch.topk(torch.softmax(gating, -1), topk, -1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
    return x, w1, w2, tw, ti.to(torch.int32)


def repack_moe(w, quanter, *args):
    """Per-expert repack. Returns (qw[E,...], scale[E,...], ref_bf16[E,N,K])."""
    qs, ss, refs = [], [], []
    for e in range(w.shape[0]):
        ref_kn, qw, s = quanter(w[e], *args)   # ref is [K,N]
        qs.append(qw); ss.append(s); refs.append(ref_kn.T.contiguous())  # ref -> [N,K]
    return (torch.stack(qs).contiguous(), torch.stack(ss).contiguous(),
            torch.stack(refs).to(DT).contiguous())


def run(tag, T=16, E=8, H=256, I=512, topk=2):
    x, w1, w2, tw, ti = make_case(T, E, H, I, topk)
    print(f"\n[{tag}] T={T} E={E} H={H} I={I}  x{tuple(x.shape)} w1{tuple(w1.shape)} w2{tuple(w2.shape)}")

    for name, quanter, qid, gs in [
        ("fp8_w8a16", marlin_quant_fp8_torch, FP8_ID, -1),
        ("mxfp4_w4a16", rand_marlin_weight_mxfp4_like, FP4_ID, 32),
    ]:
        w1q, w1s, w1ref = repack_moe(w1, quanter, gs)
        w2q, w2s, w2ref = repack_moe(w2, quanter, gs)
        print(f"  {name}: w1q{tuple(w1q.shape)} w1s{tuple(w1s.shape)} w2q{tuple(w2q.shape)} w2s{tuple(w2s.shape)}")
        # marlin output, and bf16 reference on the SAME dequantized weights
        out_q = fused_marlin_moe(x, w1q, w2q, None, None, w1s, w2s, tw, ti, qid, global_num_experts=E)
        out_ref = fused_experts(x, w1ref, w2ref, tw, ti)
        rel = ((out_q.float() - out_ref.float()).norm() / out_ref.float().norm()).item()
        print(f"  {name}: out{tuple(out_q.shape)}  rel_err_vs_bf16(deq)={rel:.4f}  {'OK' if rel < 0.06 else 'CHECK'}")


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    run("tiny")
