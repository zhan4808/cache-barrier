import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/lambda/nfs/robert-nfs/cache-barrier-project/repos/cache-barrier"
h_full = json.load(open(f"{R}/explorations/state_residency/results_gdn_full_h100.json"))
b_fla = json.load(open(f"{R}/explorations/state_residency/results_fla_gdn_b300.json"))
b_ours = json.load(open(f"{R}/explorations/state_residency/results_gdn_l2_kernel_b300.json"))

fig, ax = plt.subplots(figsize=(6.2, 3.4))
# H100: epilogue-complete chain speedup
xs = [r["state_mb"] for r in h_full["rows"]]
ys = [r["speedup_warm"] for r in h_full["rows"]]
ax.plot(xs, ys, "o-", color="#2171B5", label="H100: fused step vs fla chain")
# B300: single-op speedup
bx = [r["state_mb"] for r in b_ours["rows"]]
fla_by_b = {r["state_mb"]: r["warm_us"] for r in b_fla["rows"]}
by = [fla_by_b[m] / r["warm_us"] for m, r in zip(bx, b_ours["rows"]) if m in fla_by_b]
bx = [m for m in bx if m in fla_by_b]
ax.plot(bx, by, "s-", color="#E6550D", label="B300: fused step vs fla op")
ax.axvline(39.8, color="#2171B5", ls="--", lw=1)
ax.axvline(91.6, color="#E6550D", ls="--", lw=1)
ax.text(39.8, 2.42, " H100 $C_{\\mathrm{eff}}$ 39.8", color="#2171B5", fontsize=8)
ax.text(90.6, 2.42, "B300 $C_{\\mathrm{eff}}$ 91.6 ", color="#E6550D", fontsize=8, ha="right")
ax.axhline(1.0, color="gray", lw=0.8)
ax.set_xlabel("total recurrent-state footprint (MB)")
ax.set_ylabel("speedup vs flash-linear-attention")
ax.set_xlim(0, 165)
ax.set_ylim(0.9, 2.55)
ax.legend(fontsize=8, loc="center right")
ax.set_title("Residency-aware GDN decode: the speedup window tracks measured $C_{\\mathrm{eff}}$",
             fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/paper/figures/gdn_window.{ext}", dpi=180)
print("saved gdn_window.png/pdf")
