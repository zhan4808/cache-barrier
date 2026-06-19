"""Render the methodology-validation figures (profiling/validation) as clean
vector PDFs for the paper. Reads the committed JSON measurements (H100, bs=1,
torch 2.7/triton 3.3 audit stack); no GPU required. The baked-in suptitle from
the audit's make_figures.py is dropped so LaTeX owns the figure numbering."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PAPER = os.path.dirname(os.path.abspath(__file__))
_D = os.path.normpath(os.path.join(_PAPER, "..", "..", "profiling", "validation"))
_OUT = _PAPER

HBM_TBS = 3.35
PEAK_TF = 989.4
H, K, BS = 128, 128, 1

with open(os.path.join(_D, "results_l2_barrier_repro_torch27.json")) as f:
    event = [r for r in json.load(f)["results"] if r["batch_size"] == 1]
with open(os.path.join(_D, "diag_results.json")) as f:
    diag = json.load(f)
with open(os.path.join(_D, "graph_sweep_int4.json")) as f:
    g_int4 = json.load(f)
with open(os.path.join(_D, "graph_rotation.json")) as f:
    rot = json.load(f)

mb = np.array([r["weight_mb"] for r in event])
fp16_ev = np.array([r["fp16_ms"] for r in event]) * 1000           # us
int4_ev = np.array([r["int4_ms"] for r in event]) * 1000
dloras = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
fp16_gr = np.array([diag["d3_graph"][str(d)]["us_per_bmm"] for d in dloras])
int4_gr = np.array([g_int4[str(d)]["us_per_kernel"] for d in dloras])

dlora = mb * 1024 * 1024 / (H * K * 2)
flops = 2 * H * BS * K * dlora                                     # per launch
wt_bytes = mb * 1024 * 1024
io_bytes_fp16 = wt_bytes + H * BS * K * 2 + H * BS * dlora * 2
io_bytes_int4 = wt_bytes / 4 + H * dlora * 2 + H * BS * K * 2 + H * BS * dlora * 2

ncu_mb = np.array([8, 16, 32, 40, 48, 56, 64, 128])
ncu_warm = np.array([8438016, 4011776, 2159360, 38172416, 49979136, 58773504, 67165952, 134273792]) / 1e6
ncu_flush = np.array([8438528, 16828416, 33609984, 41996800, 50387712, 58772480, 67167232, 134273536]) / 1e6

C_FP, C_I4, C_GR, C_EV = "#1565C0", "#C62828", "#2E7D32", "#9E9E9E"
plt.rcParams.update({"font.size": 11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), bbox_inches="tight")
    plt.close(fig)


# ── val_latency: latency (repo vs true) + effective compute throughput ────────
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
a.plot(mb, fp16_ev, "o--", color=C_FP, alpha=0.45, label="FP16 — per-launch events (repo method)")
a.plot(mb, fp16_gr, "o-", color=C_FP, label="FP16 — CUDA-graph (true kernel)")
a.plot(mb, int4_ev, "^--", color=C_I4, alpha=0.45, label="INT4 — per-launch events (repo method)")
a.plot(mb, int4_gr, "^-", color=C_I4, label="INT4 — CUDA-graph (true kernel)")
a.axhline(15.5, color=C_EV, ls=":", lw=1.5)
a.text(60, 16.3, "~15.5 us launch/eventing floor\n(= repo's 'flat region')", fontsize=9, color="#616161")
a.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
a.text(50.8, 80, "nominal L2 (50 MB)", rotation=90, fontsize=9, alpha=0.6)
a.axvspan(32, 40, color="orange", alpha=0.15)
a.text(33, 80, "measured\nresidency\ncliff", fontsize=9, color="#E65100")
a.set_xlabel("FP16 weight size (MB)"); a.set_ylabel("Latency per BMM (us)")
a.set_title("Latency: repo methodology vs true kernel time")
a.legend(fontsize=9); a.grid(alpha=0.3)

tf_fp = flops / (fp16_gr * 1e-6) / 1e12
tf_i4 = flops / (int4_gr * 1e-6) / 1e12
b.plot(mb, tf_fp, "o-", color=C_FP, label="FP16 (graph-timed)")
b.plot(mb, tf_i4, "^-", color=C_I4, label="INT4 (graph-timed)")
b.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
b.axvspan(32, 40, color="orange", alpha=0.15)
b.set_xlabel("FP16 weight size (MB)"); b.set_ylabel("Effective compute (TFLOPS)")
b.set_title(f"Useful FLOP throughput: max ~4.6 TFLOPS = 0.5% of peak ({PEAK_TF:.0f} TF)\n"
            "neither kernel is compute-throughput bound; INT4's high SM% is dequant overhead")
b.legend(fontsize=9); b.grid(alpha=0.3)
save(fig, "val_latency.pdf")

# ── val_bw: effective serving bandwidth + NCU warm vs cold DRAM reads ─────────
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
bw_fp = io_bytes_fp16 / (fp16_gr * 1e-6) / 1e12
bw_i4 = io_bytes_int4 / (int4_gr * 1e-6) / 1e12
a.plot(mb, bw_fp, "o-", color=C_FP, label="FP16: logical bytes / true kernel time")
a.plot(mb, bw_i4, "^-", color=C_I4, label="INT4: logical bytes / true kernel time")
a.axhline(HBM_TBS, color="k", ls="-", lw=1.5, alpha=0.6)
a.text(2, 3.42, "HBM peak 3.35 TB/s — anything above this line MUST be L2-served", fontsize=9)
a.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
a.axvspan(32, 40, color="orange", alpha=0.15)
a.annotate("L2-fed regime:\n3.5-4.4 TB/s effective\n(not 12 TB/s)", xy=(24, 3.95), xytext=(40, 4.4),
           arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
a.annotate("HBM regime: ~2.5-2.7 TB/s", xy=(96, 2.45), xytext=(70, 1.6),
           arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
a.set_xlabel("FP16 weight size (MB)"); a.set_ylabel("Effective bandwidth (TB/s)")
a.set_title("Effective serving bandwidth (graph-timed)")
a.legend(fontsize=9); a.grid(alpha=0.3); a.set_ylim(0, 5)

w = 0.36
xi = np.arange(len(ncu_mb))
b.bar(xi - w / 2, ncu_warm, w, color=C_GR, label="warm loop, --cache-control none (true steady state)")
b.bar(xi + w / 2, ncu_flush, w, color=C_EV, label="--cache-control all (repo's NCU method = forced cold)")
b.plot(xi, ncu_mb, "k_", markersize=18, label="weight size (full re-read)")
b.set_xticks(xi); b.set_xticklabels([f"{m:g}" for m in ncu_mb])
b.set_xlabel("FP16 weight size (MB)"); b.set_ylabel("DRAM bytes read per launch (MB)")
b.set_title("NCU dram__bytes_read: cache-flushed profiling is blind to L2 residency;\n"
            "true warm reads collapse at 16-32 MB and snap to 100% at >=40 MB (not 50 MB)")
b.legend(fontsize=9); b.grid(alpha=0.3, axis="y")
save(fig, "val_bw.pdf")

# ── val_ratio: INT4/FP16 ratio (event vs graph) + rotation decomposition ─────
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
a.plot(mb, int4_ev / fp16_ev, "s--", color=C_EV, label="event-timed (repo method, incl. launch overhead)")
a.plot(mb, int4_gr / fp16_gr, "s-", color="#6A1B9A", label="graph-timed (true kernel ratio)")
a.axhline(1.0, color="k", ls=":", lw=1)
a.axvline(50, color="k", ls="--", lw=1, alpha=0.5)
a.axvspan(32, 40, color="orange", alpha=0.15)
a.text(70, 1.02, "1x parity — INT4 never wins", fontsize=9)
a.set_xlabel("FP16 weight size (MB)"); a.set_ylabel("INT4 / FP16 time ratio")
a.set_title("Knee is real but sits at 32-40 MB (effective L2), not 50 MB;\nabove it INT4 only reaches parity")
a.legend(fontsize=9); a.grid(alpha=0.3)

bars = {
    "FP16\nL2-resident": rot["fp16_copies1"],
    "FP16\nL2 destroyed\n(6 weight copies)": rot["fp16_copies6"],
    "INT4\nL2-resident": rot["int4_copies1"],
    "INT4\nL2 destroyed": rot["int4_copies6"],
    "INT4 roofline\nentitlement\n(4 MB @ 2.7 TB/s)": 4.2 / 2.7 + 1.8,
}
cols = [C_FP, "#90CAF9", C_I4, "#EF9A9A", "#A5D6A7"]
xb = np.arange(len(bars))
a2 = b.bar(xb, list(bars.values()), color=cols)
for r, v in zip(a2, bars.values()):
    b.text(r.get_x() + r.get_width() / 2, v + 0.12, f"{v:.1f}", ha="center", fontsize=10)
b.set_xticks(xb); b.set_xticklabels(list(bars.keys()), fontsize=9)
b.set_ylabel("us per kernel (graph-timed)")
b.set_title("Rotation intervention, fixed 16 MB shape:\nwith FP16 forced to HBM, INT4 is STILL 1.12x slower\n"
            "-> L2 residency is NOT sufficient to explain INT4's failure")
b.grid(alpha=0.3, axis="y")
save(fig, "val_ratio.pdf")

print("validation PDFs written to", _OUT)
