
import csv
import numpy as np
import matplotlib.pyplot as plt

with open("table7_audit.csv", newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: int(r["seed"]))

metrics = [
    ("Misclassification", "misclassification", "0.72"),
    ("Fidelity", "fidelity_valid", "0.234"),
    ("Explanation size", "explanation_size", "1.63"),
    ("Plausibility", "plausibility", "0.75"),
    ("Runtime (s/node)", "runtime", "7.26"),
]

headers = [
    "Metric", "Paper", "Seed 102", "Seed 103",
    "Seed 104", "Mean +/- SD"
]

table_data = []

for label, key, paper in metrics:
    values = np.array([float(r[key]) for r in rows])
    mean = values.mean()
    std = values.std(ddof=1)

    table_data.append([
        label,
        paper,
        *[f"{v:.4f}" for v in values],
        f"{mean:.4f} \u00b1 {std:.4f}"
    ])

fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc="center",
    loc="center",
    colWidths=[0.22, 0.12, 0.15, 0.15, 0.15, 0.21],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.9)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#233752")
        cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f3f7")

ax.set_title(
    "Table 7: Cora + GCN Reproduction (Budget = 5)",
    fontsize=15,
    pad=18
)

plt.savefig(
    "table7_comparison_final.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Saved: table7_comparison_final.png")
