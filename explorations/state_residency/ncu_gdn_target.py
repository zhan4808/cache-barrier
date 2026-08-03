"""NCU target: warm GDN decode, ours vs fla, at a given state MB.
Usage: ncu_gdn_target.py {ours|fla} <mb>"""
import sys
import torch

sys.path.insert(0, "/lambda/nfs/robert-nfs/cache-barrier-project/repos/cache-barrier/explorations/state_residency")
DK = DV = 128
H = 16

def main():
    mode, mb = sys.argv[1], float(sys.argv[2])
    torch.manual_seed(0)
    b = max(1, int(mb * 1048576 / (H * DK * DV * 4)))
    n = b * H
    S = torch.randn(n, DK, DV, dtype=torch.float32, device="cuda")
    q = torch.randn(n, DK, dtype=torch.float32, device="cuda")
    k = torch.nn.functional.normalize(torch.randn(n, DK, dtype=torch.float32, device="cuda"), dim=-1)
    v = torch.randn(n, DV, dtype=torch.float32, device="cuda")
    g = torch.full((n,), -0.1, dtype=torch.float32, device="cuda")
    beta = torch.rand(n, dtype=torch.float32, device="cuda")
    o = torch.empty(n, DV, dtype=torch.float32, device="cuda")
    if mode == "ours":
        from gdn_l2_kernel import gdn_step
        fn = lambda: gdn_step(S, q, k, v, g, beta, o)
    else:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        q4 = q.view(b, 1, H, DK).to(torch.bfloat16)
        k4 = k.view(b, 1, H, DK).to(torch.bfloat16)
        v4 = v.view(b, 1, H, DV).to(torch.bfloat16)
        g4 = g.view(b, 1, H)
        b4 = beta.view(b, 1, H).to(torch.bfloat16)
        S4 = S.view(b, H, DK, DV)
        fn = lambda: fused_recurrent_gated_delta_rule(
            q4, k4, v4, g=g4, beta=b4, initial_state=S4, output_final_state=True)
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    print("done", mode, mb)

if __name__ == "__main__":
    main()
