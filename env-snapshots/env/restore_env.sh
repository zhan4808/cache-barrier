#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
pip3 install -r "$DIR/requirements-frozen.txt"
echo "Done. Verify: python3 -c 'import torch, triton; print(torch.__version__, triton.__version__)'"
