import sys
import torch
import flag_gems
from flag_gems.fused.fused_moe import fused_experts_impl
from bench_fused_moe_mxq import make_case, quant_int8_per_channel

T = int(sys.argv[1]); mode = sys.argv[2]
E, H, I, topk = 8, 4096, 14336, 2
x, w1, w2, tw, ti = make_case(T, E, H, I, topk, seed=T)
w1_q, w1_s = quant_int8_per_channel(w1); w2_q, w2_s = quant_int8_per_channel(w2)
if mode == "bf16":
    fn = lambda: fused_experts_impl(x.clone(), w1, w2, tw, ti)
else:
    fn = lambda: fused_experts_impl(x.clone(), w1_q, w2_q, tw, ti,
        use_int8_w8a16=True, per_channel_quant=True, w1_scale=w1_s, w2_scale=w2_s)
for _ in range(30): fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
