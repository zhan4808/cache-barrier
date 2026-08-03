"""NCU target for sawtooth attribution. Usage: ncu_sawtooth_target.py <shape> <M>"""
import sys
import torch

SHAPES = {"qkv": (5120, 8192), "o_proj": (6144, 5120),
          "gate_up": (5120, 34816), "down": (17408, 5120)}

def main():
    name, m = sys.argv[1], int(sys.argv[2])
    K, N = SHAPES[name]
    torch.manual_seed(0)
    x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        torch.matmul(x, w, out=o)
    torch.cuda.synchronize()
    print("done", name, m)

if __name__ == "__main__":
    main()
