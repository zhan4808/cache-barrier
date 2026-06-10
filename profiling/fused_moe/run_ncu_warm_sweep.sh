#!/usr/bin/env bash
# Warm-state NCU for fused_moe_kernel (cache-control none). Requires ncu + sudo.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/ncu_warm"
mkdir -p "$OUT"
PY="${PYTHON:-python3}"
NCU="$(which ncu)"
export PYTHONPATH="${PYTHONPATH:-/home/ubuntu/.local/lib/python3.10/site-packages}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_ncu}"

for T in 16 128 512; do
  for mode in bf16 w8a16; do
    base="$OUT/T${T}_${mode}"
    echo "== NCU T=$T mode=$mode =="
    sudo -E env PYTHONPATH="$PYTHONPATH" TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
      "$NCU" --target-processes all --cache-control none \
      --replay-mode kernel \
      -k "regex:fused_moe" \
      --metrics "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum" \
      --csv --log-file "$base" \
      "$PY" "$DIR/ncu_target_w8a16.py" "$T" "$mode" 2>/dev/null || true
    if [[ -f "$base" ]]; then
      echo "  -> $base"
    else
      echo "  FAILED"
    fi
  done
done
