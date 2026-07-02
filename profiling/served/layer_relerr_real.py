"""Layer-level rel-err on REAL Qwen3.6-27B weights — outliers vs the Gaussian floor.

Loads real projection weights straight from safetensors (no engine), applies
each quant path, and measures output rel-err against bf16 on
realistically-scaled activations. Answers: does the 0.037 Gaussian-floor
rel-err transfer to real (outlier-bearing) weights, and how much do
per-channel / blockwise scales buy back?

Paths: W8A16 marlin-fp8 style (weight-only), W8A8 per-tensor, W8A8 per-channel
(weight) + per-token (act), W8A8 blockwise 128x128 (triton reference math).
Layers sampled: q/kv/o/gate_up/down from an early, middle, late block.

Output: results_layer_relerr_h100.json
"""

import glob
import json
import os

import torch
from safetensors import safe_open

MODEL = "/home/ubuntu/models/Qwen3.6-27B"
_D = os.path.dirname(os.path.abspath(__file__))
DEV, DT = "cuda", torch.bfloat16
FP8 = torch.float8_e4m3fn
BLK = 128
LAYERS = [3, 31, 63]   # FULL-ATTENTION layers (every 4th: 3,7,..); others are linear_attn
PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
         "linear_attn.in_proj_qkv", "linear_attn.out_proj"]  # last two exist on 3k+1.. layers only


def load_weight(name):
    for f in glob.glob(os.path.join(MODEL, "*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).to(DEV, DT)
    return None


def q_pt(w):        # per-tensor
    s = w.abs().amax().float().clamp(min=1e-6) / torch.finfo(FP8).max
    return ((w.float() / s).clamp(-448, 448).to(FP8).float() * s).to(DT)


def q_pc(w):        # per-output-channel
    s = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6) / torch.finfo(FP8).max
    return ((w.float() / s).clamp(-448, 448).to(FP8).float() * s).to(DT)


def q_blk(w):       # 128x128 blockwise
    N, K = w.shape
    pn, pk = (BLK - N % BLK) % BLK, (BLK - K % BLK) % BLK
    wp = torch.nn.functional.pad(w.float(), (0, pk, 0, pn))
    v = wp.view((N + pn) // BLK, BLK, (K + pk) // BLK, BLK).permute(0, 2, 1, 3)
    s = v.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-6) / torch.finfo(FP8).max
    q = ((v / s).clamp(-448, 448).to(FP8).float() * s).permute(0, 2, 1, 3)
    return q.reshape(N + pn, K + pk)[:N, :K].to(DT)


def a_pt(x):
    s = x.abs().amax().float().clamp(min=1e-6) / torch.finfo(FP8).max
    return ((x.float() / s).clamp(-448, 448).to(FP8).float() * s).to(DT)


def a_ptok(x):
    s = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6) / torch.finfo(FP8).max
    return ((x.float() / s).clamp(-448, 448).to(FP8).float() * s).to(DT)


def main():
    torch.manual_seed(0)
    rows = []
    print(f"{'layer':>5} {'proj':>22} {'kurt':>7} | {'w8a16':>7} {'w8a8pt':>7} "
          f"{'w8a8pc':>7} {'w8a8blk':>8}")
    for li in LAYERS:
        for pj in PROJS:
            name = f"model.language_model.layers.{li}.{pj}.weight"
            w = load_weight(name)
            if w is None:
                w = load_weight(name.replace(".language_model", ""))
            if w is None:
                print(f"  (missing {name})")
                continue
            N, K = w.shape
            x = (torch.randn(64, K, device=DEV, dtype=DT) * 0.05)
            ref = x @ w.t()
            k = torch.kurtosis(w.float().flatten()) if hasattr(torch, "kurtosis") else \
                (((w.float() - w.float().mean()) ** 4).mean() /
                 (w.float().var() ** 2)).item()
            outs = {
                "w8a16": x @ q_pt(w).t(),                      # weight-only, per-tensor
                "w8a8_pt": a_pt(x) @ q_pt(w).t(),
                "w8a8_pc": a_ptok(x) @ q_pc(w).t(),
                "w8a8_blk": a_ptok(x) @ q_blk(w).t(),
            }
            rel = {t: round(((o.float() - ref.float()).norm() /
                             ref.float().norm()).item(), 4) for t, o in outs.items()}
            rows.append({"layer": li, "proj": pj, "N": N, "K": K,
                         "kurtosis": round(float(k), 1), **rel})
            print(f"{li:>5} {pj:>22} {float(k):>7.1f} | {rel['w8a16']:>7.4f} "
                  f"{rel['w8a8_pt']:>7.4f} {rel['w8a8_pc']:>7.4f} {rel['w8a8_blk']:>8.4f}")
            del w
            torch.cuda.empty_cache()
    json.dump({"experiment": "real_weight_layer_relerr", "model": MODEL,
               "note": "reference math (quant-dequant + bf16 mm), 64 realistic acts; "
                       "gaussian-floor baseline was 0.037 per-tensor",
               "rows": rows}, open(os.path.join(_D, "results_layer_relerr_h100.json"), "w"),
              indent=1)
    print("saved")


if __name__ == "__main__":
    main()
