"""Real-model perplexity — closes the Gaussian-floor accuracy half-claim.

Scores wikitext-2-raw test windows with vLLM prompt_logprobs, using the SAME
engine/kernels as the served path, for:
  bf16   |   fp8 (vLLM W8A8, fused act quant)

Usage (one mode per process): python ppl_eval.py bf16 | fp8
Appends to results_ppl_h100.json.

Note: this evaluates the deployed per-tensor/channel fp8 path. Blockwise and
W8A16 accuracy are covered separately by layer-level real-weight rel-err
(layer_relerr_real.py) since vLLM needs pre-quantized checkpoints for those.
"""

import argparse
import json
import math
import os

MODEL = "/home/ubuntu/models/Qwen3.6-27B"
_D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_D, "results_ppl_h100.json")
WIN, N_WIN = 2048, 24   # ~49k tokens scored


def get_text():
    cache = os.path.join(_D, "wikitext2_test.txt")
    if not os.path.exists(cache):
        import urllib.request
        # wikitext-2-raw-v1 test split via the HF datasets-server parquet API is
        # heavier than needed; use the canonical raw file mirror on the Hub.
        url = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
               "wikitext-2-raw-v1/test-00000-of-00001.parquet")
        urllib.request.urlretrieve(url, cache + ".parquet")
        import pandas as pd
        txt = "\n\n".join(pd.read_parquet(cache + ".parquet")["text"].tolist())
        open(cache, "w").write(txt)
    return open(cache).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bf16", "fp8"])
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(get_text(), return_tensors=None)["input_ids"]
    print(f"wikitext-2 test: {len(ids)} tokens; scoring {N_WIN} x {WIN}")

    kw = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=WIN + 16, enable_prefix_caching=False,
              disable_log_stats=True)
    if args.mode == "fp8":
        kw["quantization"] = "fp8"
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)

    windows = [ids[i * WIN:(i + 1) * WIN] for i in range(N_WIN)]
    outs = llm.generate([{"prompt_token_ids": w} for w in windows], sp)
    nll, cnt = 0.0, 0
    for o in outs:
        for lp in o.prompt_logprobs or []:
            if lp is None:
                continue
            v = list(lp.values())[0].logprob
            nll -= v
            cnt += 1
    ppl = math.exp(nll / cnt)
    row = {"mode": args.mode, "tokens_scored": cnt, "nll_per_tok": round(nll / cnt, 5),
           "ppl": round(ppl, 4)}
    hist = json.load(open(OUT)) if os.path.exists(OUT) else []
    hist.append(row)
    json.dump(hist, open(OUT, "w"), indent=1)
    print("RESULT:", json.dumps(row))


if __name__ == "__main__":
    main()
