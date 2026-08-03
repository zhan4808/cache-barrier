# Prefill sawtooth attribution (session 12, H100, NCU)

Open since session 7: real prefill GEMM shapes show per-token cost
sawtooth in M (896 pays +17% vs 1024 in the graph-timed fp8 sum of four
shapes). NCU on the gate_up shape (5120x34816 bf16, launch-skip 12):

| M | kernel variant | CTAs | occ | dur us | per-token ns |
|---|---|---|---|---|---|
| 832  | nvjet 320x128 2x1_v | 132 | 14.7% | 462.0 | 0.555 |
| 896  | nvjet 320x128 2x1_v | 132 | 14.7% | 444.8 | 0.496 |
| 960  | nvjet 320x128 2x1_v | 132 | 14.7% | 447.4 | 0.466 |
| 1024 | nvjet 320x128 **1x2_h** | 132 | 14.7% | 498.3 | 0.487 |
| 1088 | nvjet 320x128 2x1_v | 132 | 14.7% | 569.2 | 0.523 |

Attribution, two mechanisms, neither of which is grid-level wave
quantization:

1. **nvjet is persistent**: exactly 132 CTAs (1/SM) at every M — the
   classic waves-of-CTAs picture does not apply. Quantization lives in
   the INTERNAL tile schedule: M-tiles = ceil(M/320), so 832/896/960 all
   run 3 M-tiles (identical total tile work — the per-token cost falls
   as M fills the third tile), and 1024/1088 run 4.
2. **Per-M kernel-variant selection**: M=1024 uniquely draws the 1x2_h
   cooperative-split variant; it executes 4 M-tiles in 498 us where the
   2x1_v variant at 1088 needs 569 — the variant switch, not tile count,
   is why 1024 is a per-token local MINIMUM (0.487) while the tile-ceil
   model alone would make it the maximum (predicted +17% vs 896 — the
   naive model gets the session-7 magnitude right but the SIGN of the
   winner wrong without the variant term).

Consequence for the model: mechanism B is macro-tile ceil (320-row
units on H100 nvjet) x variant selection — both per-kernel properties,
consistent with the paper's per-kernel-predicate framing; a wave-count
term would be the wrong parameterization.

Scope: one shape (gate_up), bf16, NCU-clocked (absolute us not
comparable to graph-timed); the fp8 path and remaining shapes unprofiled.
Target: `profiling/gate/ncu_sawtooth_target.py`.
