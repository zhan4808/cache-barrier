"""D1 exploration — recurrent-state residency and the capacity gate (session 9+).

Hybrid linear-attention models (Qwen 3.5 / Nemotron 3 / Mamba-3 convergence:
~75% linear layers + 25% softmax attention) replace the growing KV cache with
a CONSTANT-size per-request recurrent state S [d_k, d_v] per head. Decode
re-reads and rewrites the full state every step:

    S <- a*S + k v^T        (read S, write S)
    o  = q^T S              (read S)

so per-step traffic ~= 3x the state footprint, and the batch's total state
footprint (B x H x d_k x d_v x 2B) is a warm re-read working set — exactly
the operand class the capacity gate governs. Prediction (pre-registered
here): per-step effective bandwidth cliffs from the L2 tier to the HBM tier
when total state footprint crosses C_eff (H100 fine-grid onset 39.8 MB),
i.e. hybrid-model decode throughput has a BATCH-SIZE cliff at
B* = C_eff / (H * d_k * d_v * 2). A rotated-copies control (footprint forced
out of residency) bounds the far field.

Shapes: d_k = d_v = 128, fp16 state = 32 KB/head; a Qwen3-Next-like layer
(16 linear-attention heads) carries 0.5 MB/request/layer.

Graph-timed (10 steps/graph, median of 30 replays), same discipline as the
gate benches. Output: results_state_residency_<gpu>.json next to this file.
"""

import json
import os
import re
import statistics

import torch

_D = os.path.dirname(os.path.abspath(__file__))
DEV, DT = "cuda", torch.float16
DK = DV = 128
DECAY = 0.99
GRAPH_INNER = 10
GRAPH_REPS = 30


def sm_clock_loaded(samples=3, iters=40):
    a = torch.randn(8192, 8192, device=DEV, dtype=DT)
    b = torch.randn(8192, 8192, device=DEV, dtype=DT)
    vals = []
    for _ in range(samples):
        for _ in range(iters):
            a @ b
        v = os.popen("nvidia-smi --query-gpu=clocks.sm "
                     "--format=csv,noheader,nounits").read().strip().splitlines()[0]
        torch.cuda.synchronize()
        vals.append(int(v))
    del a, b
    torch.cuda.empty_cache()
    vals.sort()
    return f"{vals[0]}-{vals[-1]} MHz sampled under load (median {vals[len(vals) // 2]})"


def graph_time_us(fn, n_inner=GRAPH_INNER, n_rep=GRAPH_REPS):
    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_inner):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000


def one(n_heads_total, n_cop):
    """n_heads_total batched state heads; n_cop copies cycled across steps.

    n_cop=1 -> warm decode (state re-read every step, residency possible).
    n_cop>1 -> rotated control: each copy touched every n_cop steps, total
    footprint n_cop x state, forcing the far field.
    """
    S = [torch.randn(n_heads_total, DK, DV, dtype=DT, device=DEV)
         for _ in range(n_cop)]
    k = torch.randn(n_heads_total, DK, 1, dtype=DT, device=DEV)
    v = torch.randn(n_heads_total, 1, DV, dtype=DT, device=DEV)
    q = torch.randn(n_heads_total, 1, DK, dtype=DT, device=DEV)
    out = torch.empty(n_heads_total, 1, DV, dtype=DT, device=DEV)

    def step(i):
        s = S[i % n_cop]
        # S <- a*S + k v^T in ONE pass (decay folded into beta): read+write S
        torch.baddbmm(s, k, v, beta=DECAY, alpha=1.0, out=s)
        torch.bmm(q, s, out=out)  # o = q^T S: read S

    us = graph_time_us(step)
    for s in S:
        del s
    del S, k, v, q, out
    torch.cuda.empty_cache()
    return us


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    clocks = sm_clock_loaded()
    state_kb = DK * DV * 2 / 1024
    print(f"GPU {gpu}  clocks {clocks}")
    print(f"state/head {state_kb:.0f} KB  traffic model 3x footprint/step "
          f"(read,write,read)")
    print(f"{'MB':>7} {'heads':>7} {'warm us':>9} {'warm TB/s':>9} "
          f"{'rot us':>9} {'rot TB/s':>9} {'ratio':>6}")

    # footprint sweep: 8..96 MB total state (H100 C_eff onset 39.8 MB)
    targets_mb = [8, 12, 16, 20, 24, 28, 32, 36, 38, 40, 42, 44, 48, 56, 64, 80, 96]
    rows = []
    for mb in targets_mb:
        n_heads = max(1, int(mb * 1048576 / (DK * DV * 2)))
        fp = n_heads * DK * DV * 2
        warm_us = one(n_heads, 1)
        # rotated: total footprint >= max(2x C_eff, 3x fp) to force far field
        n_cop = max(2, int(96 * 1048576 / fp) + 1)
        rot_us = one(n_heads, n_cop)
        bw = lambda u: 3 * fp / (u * 1e-6) / 1e12
        r = {"target_mb": mb, "n_heads": n_heads,
             "footprint_mb": round(fp / 1048576, 2),
             "warm_us": round(warm_us, 3), "rot_us": round(rot_us, 3),
             "warm_bw_tbs": round(bw(warm_us), 3),
             "rot_bw_tbs": round(bw(rot_us), 3),
             "rot_copies": n_cop,
             "warm_over_rot": round(rot_us / warm_us, 3)}
        rows.append(r)
        print(f"{r['footprint_mb']:>7.1f} {n_heads:>7} {warm_us:>9.2f} "
              f"{r['warm_bw_tbs']:>9.2f} {rot_us:>9.2f} {r['rot_bw_tbs']:>9.2f} "
              f"{r['warm_over_rot']:>6.2f}")

    # Contended mode: warm 24 MB state, but interleave a streaming read of
    # X MB between state steps (emulates the other layers' weight+state
    # traffic between two touches of this layer's state in real decode).
    # Serving-contention precedent says residency collapses once total hot
    # set crosses C_eff.
    print("\ncontended (24 MB warm state + X MB interleaved stream):")
    print(f"{'X MB':>6} {'us':>9} {'TB/s':>6} {'vs uncontended':>14}")
    n_heads = int(24 * 1048576 / (DK * DV * 2))
    fp = n_heads * DK * DV * 2
    base_us = one(n_heads, 1)
    contended = []
    for x_mb in [0, 8, 16, 24, 32, 64, 128]:
        if x_mb == 0:
            us = base_us
        else:
            S = torch.randn(n_heads, DK, DV, dtype=DT, device=DEV)
            k = torch.randn(n_heads, DK, 1, dtype=DT, device=DEV)
            v = torch.randn(n_heads, 1, DV, dtype=DT, device=DEV)
            q = torch.randn(n_heads, 1, DK, dtype=DT, device=DEV)
            o = torch.empty(n_heads, 1, DV, dtype=DT, device=DEV)
            w = torch.randn(int(x_mb * 1048576 // 4), dtype=torch.float32,
                            device=DEV)

            def step(i):
                torch.baddbmm(S, k, v, beta=DECAY, alpha=1.0, out=S)
                torch.bmm(q, S, out=o)
                w.sum()  # the other layers' streaming traffic

            us_tot = graph_time_us(step)
            # subtract the stream's own cost measured alone
            us_stream = graph_time_us(
                lambda i: w.sum())
            us = us_tot - us_stream
            del S, k, v, q, o, w
            torch.cuda.empty_cache()
        contended.append({"interleave_mb": x_mb, "state_us": round(us, 3),
                          "state_bw_tbs": round(3 * fp / (us * 1e-6) / 1e12, 3)})
        print(f"{x_mb:>6} {us:>9.2f} {contended[-1]['state_bw_tbs']:>6.2f} "
              f"{us / base_us:>14.2f}")

    out = {
        "experiment": "recurrent_state_residency_gate",
        "gpu": gpu, "sm_clock_loaded": clocks,
        "torch": torch.__version__,
        "dk": DK, "dv": DV, "dtype": "fp16", "decay": DECAY,
        "traffic_model": "3x footprint per step (S read+write in update, read in readout)",
        "timing": f"CUDA graphs, {GRAPH_INNER} steps/graph, median of {GRAPH_REPS} replays",
        "prediction": "warm BW cliffs L2->HBM tier as footprint crosses C_eff "
                      "(h100 fine-grid onset 39.8 MB); rotated flat at HBM tier; "
                      "implies batch-size cliff B* = C_eff/(H*dk*dv*2) for "
                      "hybrid-model decode",
        "rows": rows,
        "contended_24mb": contended,
    }
    path = os.path.join(_D, f"results_state_residency_{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
