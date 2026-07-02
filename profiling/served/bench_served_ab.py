"""Served A/B — turn the decode-speedup projection into a measurement.

Serves the REAL Qwen3.6-27B (the model from Dr. Xiao's profile) via vLLM
offline engine and measures decode throughput:
  bf16          (baseline)
  fp8           (vLLM quantization="fp8": W8A8 per-tensor/channel cutlass with
                 fused dynamic act quant — the exact deployed path our
                 projection modeled at ~1.6x)

Workload: decode-heavy (short prompts, long generations) + a prefill-heavy
variant. One config per process invocation (weights are 55 GB; run
sequentially and let the process exit free memory):

  python bench_served_ab.py bf16   [--prefill]
  python bench_served_ab.py fp8    [--prefill]

Appends one row per run to results_served_ab_h100.json.
"""

import argparse
import json
import os
import time

MODEL = "/home/ubuntu/models/Qwen3.6-27B"
_D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_D, "results_served_ab_h100.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bf16", "fp8"])
    ap.add_argument("--prefill", action="store_true",
                    help="prefill-heavy workload (long prompts, 1 gen token)")
    ap.add_argument("--max-batched", type=int, default=None,
                    help="max_num_batched_tokens override (band-aware demo)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kw = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=4096, enable_prefix_caching=False,
              disable_log_stats=True)
    if args.mode == "fp8":
        kw["quantization"] = "fp8"
    if args.max_batched:
        kw["max_num_batched_tokens"] = args.max_batched
        kw["enable_chunked_prefill"] = True

    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0

    base = ("The history of computing begins with mechanical calculation. " * 4).strip()
    if args.prefill:
        prompts = [base * 12 + f" Variant {i}." for i in range(64)]   # ~3k tok prompts
        sp = SamplingParams(max_tokens=1, temperature=0)
        tag = "prefill"
    else:
        prompts = [base[:200] + f" Case {i}: continue the story." for i in range(64)]
        sp = SamplingParams(max_tokens=256, temperature=0, ignore_eos=True)
        tag = "decode"

    # warmup
    llm.generate(prompts[:4], sp)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    gen_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    in_toks = sum(len(o.prompt_token_ids) for o in outs)
    row = {"mode": args.mode, "workload": tag,
           "max_batched": args.max_batched,
           "load_s": round(load_s, 1), "wall_s": round(dt, 2),
           "prompt_toks": in_toks, "gen_toks": gen_toks,
           "gen_tok_per_s": round(gen_toks / dt, 1),
           "total_tok_per_s": round((in_toks + gen_toks) / dt, 1)}
    hist = []
    if os.path.exists(OUT):
        hist = json.load(open(OUT))
    hist.append(row)
    json.dump(hist, open(OUT, "w"), indent=1)
    print("RESULT:", json.dumps(row))


if __name__ == "__main__":
    main()
