import numpy as np
import matplotlib.pyplot as plt
import csv
# =========================
# Figure 6 data
# =========================
metrics_left = ["Misclass.↑", "Fidelity↑", "ΔE↓", "Plausibility↑"]
time_label = "Time(sec)↓"

# No pruning vs with pruning

with open("figure6_verified_metrics.csv", newline="") as f:
    rows = {
        row["pruning"]: row
        for row in csv.DictReader(f)
    }

def plot_metrics(row):
    return {
        "Misclass.↑": float(row["misclassification"]),
        "Fidelity↑": float(row["fidelity"]),
        "ΔE↓": float(row["explanation_size"]),
        "Plausibility↑": float(row["plausibility"]),
        "Time(sec)↓": float(row["runtime_s_per_node"]),
    }

no_pruning = plot_metrics(rows["off"])
with_pruning = plot_metrics(rows["on"])

print("Pruning OFF:", no_pruning)
print("Pruning ON:", with_pruning)


# =========================
# Plot
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13
})

fig, ax1 = plt.subplots(figsize=(12, 4.8))
ax2 = ax1.twinx()

labels = metrics_left + [time_label]
x = np.arange(len(labels))
width = 0.16

blue = "#0b57ff"
red = "#ff3b1f"

# Left-axis bars
vals_no_left = [no_pruning[m] for m in metrics_left]
vals_yes_left = [with_pruning[m] for m in metrics_left]

bars1 = ax1.bar(x[:4] - width/1.2, vals_no_left, width=width, color=blue,
                label="ATEX-CF (No Pruning)")
bars2 = ax1.bar(x[:4] + width/1.2, vals_yes_left, width=width, color=red,
                label="ATEX-CF (With Pruning)")

# Right-axis bars for time
bars3 = ax2.bar(x[4] - width/1.2, no_pruning[time_label], width=width, color=blue)
bars4 = ax2.bar(x[4] + width/1.2, with_pruning[time_label], width=width, color=red)

# Axes
ax1.set_ylabel("Metrics")
ax2.set_ylabel("Time (sec)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

ax1.set_ylim(0, 2.1)   # enough for ΔE
ax2.set_ylim(0, max(no_pruning[time_label], with_pruning[time_label]) * 1.35)

ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.set_axisbelow(True)

# Annotate bars
def annotate(ax, bars, is_time=False):
    for b in bars:
        h = b.get_height()
        if is_time:
            txt = f"{h:.2f}".rstrip("0").rstrip(".")
            offset = 0.04 * ax.get_ylim()[1]
        else:
            precision = 3 if abs(h - no_pruning["Plausibility↑"]) < 1e-6 or abs(h - with_pruning["Plausibility↑"]) < 1e-6 else 2
            txt = f"{h:.{precision}f}".rstrip("0").rstrip(".")
            offset = 0.03 * ax.get_ylim()[1]
        ax.text(
            b.get_x() + b.get_width()/2,
            h + offset,
            txt,
            ha="center",
            va="bottom",
            fontsize=11
        )

annotate(ax1, bars1)
annotate(ax1, bars2)
annotate(ax2, bars3, is_time=True)
annotate(ax2, bars4, is_time=True)

# Legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc="upper center", bbox_to_anchor=(0.5, -0.15),
           ncol=2, frameon=False)

# Caption-like title
plt.title("Figure 6: Effectiveness of Post-Hoc Pruning (Cora + GCN, seed 104)")

plt.tight_layout()
plt.savefig("figure6_posthoc_pruning_final.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: figure6_posthoc_pruning_final.png")