"""D2+D5 exploration — sparse-attention KV hot sets and quantized KV vs the gate.

DSA-class sparse attention selects ~2K tokens per query, shrinking the HOT
KV working set per request to single-digit MB — an operand class the
original study's dense-KV null (Exp B: "KV is not L2-limited") never saw.
The batch's hot set is B x hot-KV bytes, so it crosses C_eff at modest B:
the same batch-size-knee structure as recurrent state (D1).

Three questions, one instrument (graph-timed decode GQA attention,
q [B,32,1,128], KV [B,8,T,128], T=2048 selected tokens = 8 MB/request fp16):

  Q1 (D2): does warm re-read attention over the selected hot set show the
      residency advantage below C_eff and lose it above? (warm vs
      rotated-copies, B sweep 8->96 MB)
  Q2 (D5, byte-governance): is the cliff located in BYTES, not tokens?
      Re-run at half the per-token bytes (Hkv=4): cliff should sit at the
      same MB, 2x the batch. If yes, KV quantization arithmetic follows:
      fp8 doubles, 4-bit quadruples the resident batch/context.
  Q3 (D5, spot check): store KV as e4m3 with inline dequant->sdpa at a
      token count where fp16 is above the gate but fp8 bytes are below —
      does quantized storage recover the warm advantage that fp16 lost?

Output: results_kv_hotset_<gpu>.json
"""

import json
import os
import re
import statistics

import torch
import torch.nn.functional as F

_D = os.path.dirname(os.path.abspath(__file__))
DEV, DT = "cuda", torch.float16
HQ, D = 32, 128
T_SEL = 2048
GRAPH_INNER = 10
GRAPH_REPS = 30


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


def kv_bytes(b, hkv, t=T_SEL, elt=2):
    return 2 * b * hkv * t * D * elt


def one(b, hkv, n_cop):
    q = torch.randn(b, HQ, 1, D, dtype=DT, device=DEV)
    KV = [(torch.randn(b, hkv, T_SEL, D, dtype=DT, device=DEV),
           torch.randn(b, hkv, T_SEL, D, dtype=DT, device=DEV))
          for _ in range(n_cop)]

    def step(i):
        k, v = KV[i % n_cop]
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    us = graph_time_us(step)
    del KV, q
    torch.cuda.empty_cache()
    return us


def one_fp8(b, hkv, n_cop):
    """e4m3 KV storage, inline dequant then sdpa (dequant-on-read serving)."""
    q = torch.randn(b, HQ, 1, D, dtype=DT, device=DEV)
    KV = [(torch.randn(b, hkv, T_SEL, D, dtype=DT, device=DEV).to(torch.float8_e4m3fn),
           torch.randn(b, hkv, T_SEL, D, dtype=DT, device=DEV).to(torch.float8_e4m3fn))
          for _ in range(n_cop)]

    def step(i):
        k, v = KV[i % n_cop]
        F.scaled_dot_product_attention(q, k.to(DT), v.to(DT), enable_gqa=True)

    us = graph_time_us(step)
    del KV, q
    torch.cuda.empty_cache()
    return us


def sweep(hkv, batches, tag, rows):
    print(f"\n[{tag}] Hkv={hkv} ({kv_bytes(1, hkv) / 1048576:.0f} MB/req)")
    print(f"{'B':>4} {'MB':>7} {'warm us':>9} {'rot us':>9} {'rot/warm':>8}")
    for b in batches:
        mb = kv_bytes(b, hkv) / 1048576
        warm = one(b, hkv, 1)
        n_cop = max(2, int(96 * 1048576 / kv_bytes(b, hkv)) + 1)
        rot = one(b, hkv, n_cop)
        r = {"mode": tag, "hkv": hkv, "B": b, "hot_mb": round(mb, 1),
             "warm_us": round(warm, 3), "rot_us": round(rot, 3),
             "rot_copies": n_cop, "rot_over_warm": round(rot / warm, 3)}
        rows.append(r)
        print(f"{b:>4} {mb:>7.1f} {warm:>9.2f} {rot:>9.2f} "
              f"{r['rot_over_warm']:>8.2f}")


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    print(f"GPU {gpu}  T_sel={T_SEL} (DSA-class selection)  d={D} Hq={HQ}")

    rows = []
    # Q1: 8 MB/request, B 1..12 -> 8..96 MB
    sweep(8, [1, 2, 3, 4, 5, 6, 8, 10, 12], "fp16_hkv8", rows)
    # Q2: half the bytes/token, B 2..24 -> same MB range
    sweep(4, [2, 4, 6, 8, 10, 12, 16, 20, 24], "fp16_hkv4", rows)

    # Q3: fp8 spot check at B=8: fp16 hot set 64 MB (above gate),
    # fp8 hot set 32 MB (below gate)
    b = 8
    spot = {}
    for tag, fn, elt in (("fp16", one, 2), ("fp8_dequant", one_fp8, 1)):
        warm = fn(b, 8, 1)
        n_cop = max(2, int(96 * 1048576 / kv_bytes(b, 8, elt=elt)) + 1)
        rot = fn(b, 8, n_cop)
        spot[tag] = {"hot_mb": round(kv_bytes(b, 8, elt=elt) / 1048576, 1),
                     "warm_us": round(warm, 3), "rot_us": round(rot, 3),
                     "rot_over_warm": round(rot / warm, 3)}
        print(f"\n[spot B={b} {tag}] hot {spot[tag]['hot_mb']} MB  "
              f"warm {warm:.2f}  rot {rot:.2f}  rot/warm {rot / warm:.2f}")

    out = {
        "experiment": "sparse_kv_hotset_gate",
        "gpu": gpu, "torch": torch.__version__,
        "shape": {"Hq": HQ, "d": D, "t_selected": T_SEL},
        "timing": f"CUDA graphs, {GRAPH_INNER}/graph, median of {GRAPH_REPS}",
        "clock_note": "locked cap 1755 MHz; memory-bound cells near the cap, "
                      "see session-9 journal on lock semantics",
        "prediction": "warm advantage below C_eff bytes regardless of Hkv "
                      "(byte-governed); fp8 storage recovers the advantage "
                      "fp16 loses at the same token count",
        "rows": rows, "fp8_spot_B8": spot,
    }
    path = os.path.join(_D, f"results_kv_hotset_{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
