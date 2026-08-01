"""
P6 — dispatch cost model (DIRECTION.md §6 P6). Analysis; one measured input.

Runtime precision dispatch needs the weights available in whichever format the
dispatcher picks. Three storage policies, costed for a dense model on one GPU:

  A. DUAL-RESIDENT   both formats in HBM. Cost: quantized copy's bytes come
                     straight out of the KV-cache budget -> concurrency ->
                     decode throughput (throughput ~ concurrency until
                     compute-bound).
  B. REPACK          one format resident, convert on dispatch switch. Cost:
                     critical-path repack latency, amortized over the switch
                     period. Measured on H100 (unfused torch, int8->bf16,
                     graph-timed): 0.133 T elems/s; fused lower bound =
                     bw_hbm / 3 bytes-moved/elem = 1.09 T elems/s.
  C. JIT DEQUANT     quantized-primary, dequantize on use.
                     - via HBM scratch: weight traffic per use becomes
                       read packed (0.5x) + write bf16 (2x) + read bf16 (2x)
                       = 4.5x per int4-elem-byte vs 2x for resident bf16 ->
                       2.25x WORSE than not quantizing. Never pays. Closed.
                     - in-kernel (registers/SMEM): this is exactly the W4A16
                       in-core path CARM already prices via r_dequant. Not a
                       new option — it is the ceiling term.

Conclusion the numbers force: with bf16-primary storage, dispatch is either
memory-infeasible (A, on 80 GB) or switch-period-limited (B). The only policy
with zero marginal memory and zero switch latency is QUANTIZED-PRIMARY storage
with per-shape choice between (i) quantized-compute kernels (W8A8/W4A4) and
(ii) dequant-in-kernel to bf16 compute, priced by r_dequant. This matches the
2026 reality that checkpoints ship natively quantized (DIRECTION.md §1): the
dispatcher's job is not "when to quantize" but "when to pay the dequant
ceiling vs the quantized-compute path" — plus per-kernel predicates
(mechanisms A–E). Couples back to P3: policy A also *shrinks* KV traffic,
which raises GEMM-visible L2 capacity; second-order, dominated by the
concurrency loss.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_D = os.path.dirname(os.path.abspath(__file__))

# ── measured on H100 2026-08-01 (graph-timed, clock-locked) ────────────────
REPACK_UNFUSED_TELEMS = 0.133e12   # int8->bf16, unfused torch (temps incl.)
BW_HBM = 3.26e12
REPACK_FUSED_TELEMS = BW_HBM / 3.0  # 1 B read + 2 B write per elem, fused bound

# ── workload/model assumptions (edit per deployment; Qwen3.6-27B-ish) ──────
ASSUMPTIONS = {
    "n_params": 27e9,
    "bytes_primary": 2.0,          # bf16-primary
    "bytes_secondary": 1.0,        # int8 copy
    "kv_bytes_per_token": 4096 * 16,  # 8 KV heads x 128 dim x 2(K,V) x 2B x 16
                                      # full-attn layers (every 4th of 64;
                                      # linear-attn layers hold no KV)
    "tokens_per_seq": 32768 + 1024,   # p32768d1024
    "runtime_overhead_gb": 6.0,
}


def policy_a(gpu_mem_gb, a=ASSUMPTIONS, kv_scale=1.0):
    """Dual-residency: concurrency with and without the second copy.

    kv_scale: KV-cache bytes multiplier (0.5 = fp8 KV). KV quantization enters
    the dispatch problem HERE — as a memory/concurrency lever — not as a
    bandwidth lever: kv_serving/ showed KV reads are not L2-limited and the
    end-to-end decode-speed ceiling of fp8 KV is <=0.2% on the Qwen3.6-27B
    profile (full attention is 2.67% of runtime, and the fp8 decode kernel is
    BW-ceiling-bound at ~1.6 vs bf16's ~3.0 TB/s). Orthogonal to the gate;
    coupled through the KV budget."""
    kv_seq = a["kv_bytes_per_token"] * a["tokens_per_seq"] * kv_scale
    base_free = gpu_mem_gb * 1e9 - a["n_params"] * a["bytes_primary"] \
        - a["runtime_overhead_gb"] * 1e9
    dual_free = base_free - a["n_params"] * a["bytes_secondary"]
    c0 = max(int(base_free / kv_seq), 0)
    c1 = max(int(dual_free / kv_seq), 0)
    return {
        "gpu_mem_gb": gpu_mem_gb,
        "kv_gb_per_seq": round(kv_seq / 1e9, 2),
        "concurrency_single": c0,
        "concurrency_dual": c1,
        "throughput_cost_pct": round(100 * (1 - c1 / c0), 1) if c0 else None,
        "feasible": c1 > 0,
    }


def policy_b(a=ASSUMPTIONS):
    """Repack latency and the switch period needed to amortize it."""
    t_unfused = a["n_params"] / REPACK_UNFUSED_TELEMS
    t_fused = a["n_params"] / REPACK_FUSED_TELEMS
    def amort(t):  # switch period for <=1% overhead
        return round(t / 0.01, 1)
    return {
        "repack_s_unfused_measured": round(t_unfused, 3),
        "repack_s_fused_bound": round(t_fused, 3),
        "switch_period_s_for_1pct_overhead_unfused": amort(t_unfused),
        "switch_period_s_for_1pct_overhead_fused": amort(t_fused),
        "verdict": "feasible only at engine-mode granularity (seconds-minutes);"
                   " infeasible per-phase (~100 ms) or per-layer",
    }


def policy_c():
    """JIT dequant via HBM scratch: traffic ratio vs resident bf16 (int4 case)."""
    per_use = 0.5 + 2.0 + 2.0   # read packed + write bf16 + read bf16, per elem
    resident = 2.0
    return {
        "traffic_ratio_vs_resident_bf16": per_use / resident,  # 2.25
        "verdict": "never pays; in-kernel JIT is the r_dequant ceiling CARM "
                   "already prices",
    }


def main():
    out = {
        "assumptions": ASSUMPTIONS,
        "measured": {
            "repack_unfused_telems_per_s": REPACK_UNFUSED_TELEMS / 1e12,
            "repack_fused_bound_telems_per_s": round(REPACK_FUSED_TELEMS / 1e12, 3),
            "bw_hbm_tbs": BW_HBM / 1e12,
        },
        "policy_a_dual_resident": {
            "h100_80gb": policy_a(80),
            "h200_141gb": policy_a(141),
        },
        "kv_precision_lever": {
            "note": "fp8 KV halves bytes/token. It is a CONCURRENCY lever "
                    "(memory), not a speed lever: e2e decode-speed ceiling "
                    "<=0.2% (kv_serving/, full attn = 2.67% of runtime), but "
                    "it roughly doubles max concurrency at fixed memory and "
                    "re-opens dual-residency headroom on large-memory parts.",
            "h100_80gb_kv_fp8": policy_a(80, kv_scale=0.5),
            "h200_141gb_kv_fp8": policy_a(141, kv_scale=0.5),
        },
        "policy_b_repack": policy_b(),
        "policy_c_jit_dequant": policy_c(),
        "conclusion": "quantized-primary storage + per-shape kernel choice is "
                      "the only zero-marginal-cost dispatch; bf16-primary "
                      "dispatch is memory-infeasible (80 GB) or "
                      "switch-period-limited",
    }
    with open(os.path.join(_D, "results_cost_model.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["policy_a_dual_resident"], indent=1))
    print(json.dumps(out["policy_b_repack"], indent=1))

    # Figure: left = concurrency under storage policies; right = repack amortization
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    labels, single, dual = [], [], []
    for name, gb in (("H100 80GB", 80), ("H200 141GB", 141)):
        r = policy_a(gb)
        labels.append(name)
        single.append(r["concurrency_single"])
        dual.append(r["concurrency_dual"])
    xs = range(len(labels))
    axes[0].bar([x - 0.18 for x in xs], single, 0.36, label="single format", color="#2171B5")
    axes[0].bar([x + 0.18 for x in xs], dual, 0.36, label="dual resident", color="#E6550D")
    axes[0].set_xticks(list(xs)); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("max concurrent sequences")
    axes[0].set_title("Policy A: the second format is paid in KV budget", fontsize=10)
    axes[0].legend(fontsize=8)
    for x, v in zip(xs, dual):
        if v == 0:
            axes[0].annotate("infeasible", (x + 0.18, 0.5), ha="center", fontsize=8,
                             color="#E6550D")

    periods = [0.1, 1, 10, 60, 600]
    for t, lab, c in ((policy_b()["repack_s_unfused_measured"], "unfused (measured)", "#E6550D"),
                      (policy_b()["repack_s_fused_bound"], "fused bound", "#2171B5")):
        axes[1].plot(periods, [100 * t / p for p in periods], "o-", label=lab, color=c)
    axes[1].axhline(1, color="gray", lw=0.8, ls="--")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("dispatch switch period (s)")
    axes[1].set_ylabel("repack overhead (%)")
    axes[1].set_title("Policy B: repack amortizes only at engine-mode scale", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.suptitle("P6: dispatch storage policies — quantized-primary is the only free one",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(_D, "fig_cost_model.png"), dpi=180)
    print("saved results_cost_model.json, fig_cost_model.png")


if __name__ == "__main__":
    main()
