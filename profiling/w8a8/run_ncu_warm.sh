#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/ncu_warm"
mkdir -p "$OUT"
PY="${PYTHON:-python3}"
NCU="$(which ncu)"
export PYTHONPATH="${PYTHONPATH:-}"

for d in 512 1536; do
  for mode in fp16 w8a8; do
    base="$OUT/d${d}_${mode}"
    echo "== NCU d_lora=$d mode=$mode =="
    sudo -E env PYTHONPATH="$PYTHONPATH" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_ncu}" \
      "$NCU" --target-processes all --cache-control none --replay-mode kernel \
      -k "regex:nvjet|_w8a8|_quant_act" \
      --metrics "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum" \
      --csv --log-file "$base" \
      "$PY" "$DIR/ncu_target_w8a8.py" "$d" "$mode" 2>/dev/null || true
    [[ -f "$base" ]] && echo "  -> $base" || echo "  FAILED"
  done
done
