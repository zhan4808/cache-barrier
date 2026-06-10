"""Autotune W8A8 BMM block configs at large batch sizes (graph-timed)."""
import itertools, json, os, sys
import torch
_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from bench_w8a8 import graph_med_us
from w8a8_bmm import quantize_acts_w8, quantize_weights_w8, w8a8_bmm
H, K, N = 128, 128, 512

def autotune_bs(bs):
    a = torch.randn(H, bs, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
    wq, ws = quantize_weights_w8(w)
    q = torch.empty_like(a, dtype=torch.int8)
    s = torch.empty(H * bs, dtype=torch.float32, device="cuda")
    o = torch.empty(H, bs, N, dtype=torch.float16, device="cuda")
    quantize_acts_w8(a, q, s)
    for _ in range(10): torch.bmm(a, w)
    torch.cuda.synchronize()
    fp16_us = graph_med_us(lambda: torch.bmm(a, w), reps=30)
    best = None
    for bm, bn, bk, nw, ns in itertools.product([16,32,64,128],[64,128,256],[64,128],[4,8],[3,4]):
        try:
            fn = lambda: w8a8_bmm(q, wq, s, ws, o, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)
            for _ in range(3): fn()
            torch.cuda.synchronize()
            us = graph_med_us(fn, reps=20)
            if best is None or us < best[0]: best = (us, bm, bn, bk, nw, ns)
        except Exception: pass
    del a,w,wq,q,s,o; torch.cuda.empty_cache()
    return {"bs":bs,"fp16_us":round(fp16_us,2),"w8a8_mm_us":round(best[0],2),"ratio":round(fp16_us/best[0],3),
            "cfg":{"BLOCK_M":best[1],"BLOCK_N":best[2],"BLOCK_K":best[3],"num_warps":best[4],"num_stages":best[5]}}

if __name__ == "__main__":
    out = {str(bs): autotune_bs(bs) for bs in [64,128,256,512]}
    with open(os.path.join(_D,"autotune_bs.json"),"w") as f: json.dump(out,f,indent=2)
    for bs,r in out.items(): print(f"bs={bs}: fp16={r['fp16_us']} mm={r['w8a8_mm_us']} ({r['ratio']}x)")
