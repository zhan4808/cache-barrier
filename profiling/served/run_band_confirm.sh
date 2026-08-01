#!/usr/bin/env bash
# Repeat pass for the pivotal band-prefill points (single-run caveat).
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/ubuntu/vllm-env/bin/python
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
log(){ echo "[$(date +%H:%M:%S)] $*"; }
log "confirm: uncapped"
$PY bench_served_ab.py fp8 --prefill || log "uncapped FAILED ($?)"
for mb in 896 1024 1792; do
  log "confirm: mb=$mb"
  $PY bench_served_ab.py fp8 --prefill --max-batched $mb || log "mb=$mb FAILED ($?)"
done
log "BAND_CONFIRM_DONE"
