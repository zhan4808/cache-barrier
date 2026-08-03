"""NCU target for the two-capacities probe. Usage: ncu_target.py {gemm|sum} <mb>

Runs 20 warm iterations of either a T=1 bf16 GEMM (weight = <mb> MB) or a
pure fp32 sum-reduce over an <mb> MB buffer. Profile the tail launches with
--cache-control none --launch-skip so counters see the warm state.
"""

import sys

import torch

K = 8192


def main():
    mode, mb = sys.argv[1], float(sys.argv[2])
    torch.manual_seed(0)
    if mode == "gemm":
        n = max(256, int(mb * 1048576 / K / 2 / 128) * 128)
        x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(K, n, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
        fn = lambda: torch.matmul(x, w, out=o)
        real_mb = K * n * 2 / 1048576
    else:
        buf = torch.randn(int(mb * 1048576 // 4), dtype=torch.float32,
                          device="cuda")
        fn = lambda: buf.sum()
        real_mb = buf.numel() * 4 / 1048576
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    print(f"done {mode} {real_mb:.1f} MB")


if __name__ == "__main__":
    main()
