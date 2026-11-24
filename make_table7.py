import re
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

seeds = [102, 103, 104]
results = []

def read_log(path):
    raw = path.read_bytes()

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    elif raw.count(b"\x00") > len(raw) // 10:
        return raw.decode("utf-16-le")
    else:
        return raw.decode("utf-8-sig", errors="replace")

def extract(text, pattern):
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find: {pattern}")
    return float(match.group(1))

for seed in seeds:
    candidates = [
        Path(f"cora_seed{seed}_eval_check.log"),
        Path(f"cora_seed{seed}_eval.log")
    ]

    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(f"Evaluation log missing for seed {seed}")

    text = read_log(path)
    print(f"Reading seed {seed}: {path}")

    targets = extract(text, r"Num of target nodes:\s*(\d+)")
    flips = extract(text, r"Num of misclassification:\s*(\d+)")

    results.append([
        flips / targets,
        extract(text, r"Metric 2 - Fidelity:\s*([\d.]+)"),
        extract(text, r"Metric 3 - Average Explanation Size:\s*([\d.]+)"),
        extract(text, r"Metric 4 - Average Plausibility:\s*([\d.]+)"),
        extract(text, r"Metric 5 - Average Time Cost:\s*([\d.]+)")
    ])

metrics = [
    "Misclassification Rate",
    "Fidelity",
    "Explanation Size (\u0394E)",
    "Plausibility",
    "Runtime (s/node)"
]

paper = ["0.72", "0.234", "1.63", "0.75", "7.26"]
formats = [".3f", ".4f", ".2f", ".2f", ".2f"]

pm = "\u00b1"
rows = []

for i, metric in enumerate(metrics):
    values = [results[j][i] for j in range(3)]

    mean = statistics.mean(values)
    std = statistics.stdev(values)

    rows.append([
        metric,
        paper[i],
        *[format(v, formats[i]) for v in values],
        f"{format(mean, formats[i])} {pm} {format(std, formats[i])}"
    ])

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=[
        "Metric",
        "Original Paper",
        "Seed 102",
        "Seed 103",
        "Seed 104",
        f"Mean {pm} Std"
    ],
    colWidths=[0.28, 0.15, 0.12, 0.12, 0.12, 0.21],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.4)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")

    if row == 0:
        cell.set_facecolor("#174A7E")
        cell.set_text_props(color="white", weight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EAF2FA")

ax.set_title(
    "ATEX-CF | Table 7 Reproduction (Cora + GCN)",
    fontsize=17,
    fontweight="bold",
    pad=20
)

plt.tight_layout()
plt.savefig("table7_comparison.png", dpi=200, bbox_inches="tight")

print("Saved: table7_comparison.png")
plt.show()
