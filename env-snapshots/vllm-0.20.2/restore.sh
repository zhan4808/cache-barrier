#!/usr/bin/env bash
# Restore the vLLM 0.20.2 env used for the cache-barrier CUDA validation.
# Captured 2026-06-19 on H100 80GB, driver 580.105.08 / CUDA 13.0 (see versions.txt).
#
# Strategy: copy the byte-exact venv from NFS to a LOCAL path (default
# /home/ubuntu/vllm-env). Same absolute path => the venv's hard-coded shebangs
# work unchanged, and it runs off fast local disk (not NFS). If the byte-exact
# copy fails to import (e.g. a different driver/CUDA on the new instance), it
# falls back to recreating from requirements-frozen.txt using the cached wheels
# (offline-ish, no big re-download).
set -euo pipefail
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-/home/ubuntu/vllm-env}"

echo "== restore vLLM 0.20.2 env -> $DEST =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

if [ -e "$DEST" ]; then
  echo "NOTE: $DEST already exists. Verifying it instead of overwriting."
  "$DEST/bin/python" -c "import torch,vllm;print('existing OK torch',torch.__version__,'vllm',vllm.__version__)" && exit 0
  echo "existing env broken; remove it and re-run, or pass a fresh target path."; exit 1
fi

# ── 2026-08-01 addendum: deps discovered missing on a fresh instance ──
# (serving needs ninja for GDN JIT and pandas/pyarrow for ppl_eval;
#  matplotlib for figures. Also export VLLM_USE_DEEP_GEMM=0 when serving —
#  deep_gemm is not in this venv and vLLM 0.20.2's eligibility scan raises.)
# The 2026-06 NFS snapshot predates these, so they must be installed on
# EVERY restore path, including byte-exact — hence a function called before
# each successful exit (the original addendum sat after an early exit 0 and
# never ran on the byte-exact path).
install_extras() {
  "$DEST/bin/pip" install -q ninja pandas pyarrow matplotlib || true
  echo "== 2026-08 extras installed (ninja pandas pyarrow matplotlib) =="
}

if [ -d "$NFS/vllm-env" ]; then
  echo "[1/2] byte-exact copy from NFS (~7.8G, ~1-2 min on local NVMe)..."
  cp -a "$NFS/vllm-env" "$DEST"
  if "$DEST/bin/python" -c "import torch,vllm;print('OK (byte-exact) torch',torch.__version__,'vllm',vllm.__version__,'cuda_avail',torch.cuda.is_available())" 2>/dev/null; then
    install_extras
    echo "== restore OK (byte-exact copy) =="; exit 0
  fi
  echo "byte-exact venv did not import (driver/CUDA mismatch?). Falling back to recreate."
  rm -rf "$DEST"
fi

echo "[2/2] recreate from freeze + cached wheels..."
python3 -m venv "$DEST"
"$DEST/bin/pip" install --upgrade pip wheel >/dev/null
"$DEST/bin/pip" install --cache-dir "$NFS/pip-cache" -r "$NFS/requirements-frozen.txt"
"$DEST/bin/python" -c "import torch,vllm;print('OK (recreated) torch',torch.__version__,'vllm',vllm.__version__)"
install_extras
echo "== restore OK (recreated from freeze) =="
