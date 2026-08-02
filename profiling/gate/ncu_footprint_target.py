"""NCU target: warm-loop bmm at (W_MB, T) from argv; steady-state launch
profiled via --launch-skip. Usage: ncu ... python3 ncu_footprint_target.py W T"""
import sys
import torch

w_mb, T = float(sys.argv[1]), int(sys.argv[2])
H = K = 128
N = int(w_mb * 1048576 / (H * K * 2))
x = torch.randn(H, T, K, dtype=torch.float16, device="cuda") / 4
w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
c = torch.empty(H, T, N, dtype=torch.float16, device="cuda")
for _ in range(40):
    torch.bmm(x, w, out=c)
torch.cuda.synchronize()
