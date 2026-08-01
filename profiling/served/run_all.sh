#!/usr/bin/env bash
# Served pipeline, handoff order (docs/HANDOFF_2026-07-02.md §2).
# Each engine in its own process; sequential; clocks left at default (serving-
# realistic; A/B is relative). Appends to results_*.json per harness.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/ubuntu/vllm-env/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "1/7 layer_relerr (no engine)"
$PY layer_relerr_real.py || log "layer_relerr FAILED ($?)"

log "2/7 served decode bf16"
$PY bench_served_ab.py bf16 || log "decode bf16 FAILED ($?)"
log "3/7 served decode fp8"
$PY bench_served_ab.py fp8 || log "decode fp8 FAILED ($?)"

log "4/7 ppl bf16"
$PY ppl_eval.py bf16 || log "ppl bf16 FAILED ($?)"
log "5/7 ppl fp8"
$PY ppl_eval.py fp8 || log "ppl fp8 FAILED ($?)"

log "6/7 served prefill pair"
$PY bench_served_ab.py bf16 --prefill || log "prefill bf16 FAILED ($?)"
$PY bench_served_ab.py fp8  --prefill || log "prefill fp8 FAILED ($?)"

log "7/7 band demo (fp8, chunked prefill, aligned vs misaligned batch cap)"
$PY bench_served_ab.py fp8 --max-batched 512 || log "band 512 FAILED ($?)"
$PY bench_served_ab.py fp8 --max-batched 460 || log "band 460 FAILED ($?)"

log "PIPELINE DONE"
