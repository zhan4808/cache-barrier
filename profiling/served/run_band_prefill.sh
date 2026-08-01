#!/usr/bin/env bash
# Mechanism-B follow-up (2026-08-02): band-aware PREFILL batch shaping.
# The 2026-08-01 band demo ran the decode workload and closed negative;
# mechanism B's prediction lives in prefill batch shaping, unexercised.
# Sweep: fp8, prefill-heavy workload (27k-token prompts, 1 gen token),
# chunked prefill with max_num_batched_tokens at wave-band-aligned vs
# misaligned values at three chunk scales, plus an uncapped baseline.
# Clocks at default (serving-realistic; A/B is relative).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/ubuntu/vllm-env/bin/python
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "1/7 prefill fp8 baseline (no cap)"
$PY bench_served_ab.py fp8 --prefill || log "baseline FAILED ($?)"
for mb in 512 460 1024 896 2048 1792; do
  log "prefill fp8 --max-batched $mb"
  $PY bench_served_ab.py fp8 --prefill --max-batched $mb || log "mb=$mb FAILED ($?)"
done
log "BAND_PREFILL_DONE"
