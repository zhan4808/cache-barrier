"""Validate CARM predict_us MAPE against measured carm_params.json operating points."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../kernel-compass"))
from profiling.carm import predict_fp16_recon_us, predict_int4_recon_us, validate_recon_mape  # noqa: E402

_D = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_D, "carm_params.json")) as f:
    P = json.load(f)

mape = validate_recon_mape(P["recon_points"], gpu="h100")
print(f"FP16 MAPE: {mape['fp16_mape_pct']:.1f}%")
print(f"INT4 MAPE: {mape['int4_mape_pct']:.1f}%")
for row in mape["rows"]:
    print(
        f"  bs={row['bs']:3d}  fp16 meas={row['fp16_us']:6.2f} pred={row['fp16_pred']:.2f}  "
        f"int4 meas={row['int4_us']:6.2f} pred={row['int4_pred']:.2f}"
    )
if mape["fp16_mape_pct"] > 25 or mape["int4_mape_pct"] > 30:
    sys.exit(1)
