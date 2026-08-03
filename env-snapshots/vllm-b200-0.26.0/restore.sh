#!/bin/bash
# Restore the B200 native-FP4 env (2026-08-02 session).
# Target: Blackwell (SM100), driver >= 595. The H100 byte-exact venv will NOT
# work for the FP4 leg (no SM100 cutlass ops) - see B200_RUNBOOK.md section 1.
set -e
VENV="${1:-$HOME/vllm-b200-env}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
# Fast path: plain vllm pulls torch cu130 + flashinfer as deps (verified 0.26.0).
"$VENV/bin/pip" install vllm==0.26.0
# Byte-exact path (if the fast path drifts):
#   "$VENV/bin/pip" install -r "$(dirname "$0")/requirements-frozen.txt"
"$VENV/bin/python" -c "import vllm, torch; assert torch.cuda.get_device_capability()[0] >= 10, 'not a Blackwell GPU'; import vllm._custom_ops as ops; assert hasattr(ops, 'cutlass_fp4_moe_mm'), 'FP4 ops missing'; print('B200 env OK:', vllm.__version__, torch.__version__)"
