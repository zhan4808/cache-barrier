"""NCU target: fp16 cuBLAS BMM vs W8A8 full path for MLA reconstruction."""
import sys
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from w8a8_bmm import quantize_acts_w8, quantize_weights_w8, w8a8_bmm

H, K, bs = 128, 128, 1
d_lora = int(sys.argv[1])
mode = sys.argv[2]

a = torch.randn(H, bs, K, dtype=torch.float16, device="cuda") / 4
w = torch.randn(H, K, d_lora, dtype=torch.float16, device="cuda") / 8

if mode == "fp16":
    fn = lambda: torch.bmm(a, w)
else:
    wq, ws = quantize_weights_w8(w)
    q = torch.empty_like(a, dtype=torch.int8)
    s = torch.empty(H * bs, dtype=torch.float32, device="cuda")
    o = torch.empty(H, bs, d_lora, dtype=torch.float16, device="cuda")

    def fn():
        quantize_acts_w8(a, q, s)
        w8a8_bmm(q, wq, s, ws, o)

for _ in range(30):
    fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
