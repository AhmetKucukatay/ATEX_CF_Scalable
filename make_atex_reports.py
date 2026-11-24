"""Build ATEX-CF Figure 3 and Table 7 reports from existing local results.
Put this file in the ATEX_CF repository root and run with its .venv Python.
No training or counterfactual optimization is performed.
"""
import csv
import pickle
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from deeprobust.graph.data import Dataset

from config.config import (
    DATA_NAME, TEST_MODEL, HIDDEN_CHANNELS, DROPOUT, LEARNING_RATE,
    WEIGHT_DECAY, WITH_BIAS, GCN_LAYER, DEVICE, TAU_C,
    α2, α3, k as PLAU_K,
)
from model.GCN import load_GCN_model
from utilty.utils import normalize_adj, compute_deg_diff, compute_motif_viol

ROOT = Path(__file__).resolve().parent
BUDGETS = [1, 3, 5, 7, 10, 15]
EXPECTED_SUCCESS = {1: 248, 3: 340, 5: 369, 7: 376, 10: 379, 15: 381}
SEED = 104

assert DATA_NAME == "cora" and TEST_MODEL == "GCN", "Expected Cora + GCN settings"


def log_text(path):
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.count(b"\x00") > len(data) // 10:
        return data.decode("utf-16-le")
    return data.decode("utf-8-sig", errors="replace")


def metric(log, regex):
    match = re.search(regex, log)
    if not match:
        raise ValueError(f"Missing {regex} in log")
    return float(match.group(1))


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path.name)


def make_table7():
    paper = {
        "Misclassification Rate": 0.72,
        "Fidelity": 0.234,
        "Explanation Size": 1.63,
        "Plausibility": 0.75,
        "Runtime (s/node)": 7.26,
    }
    three_seeds = {}
    for seed in (102, 103, 104):
        candidates = [ROOT / f"cora_seed{seed}_eval_check.log",
                      ROOT / f"cora_seed{seed}_eval.log"]
        file = next((f for f in candidates if f.exists()), None)
        if file is None:
            raise FileNotFoundError(f"Seed {seed} evaluation log missing")
        s = log_text(file)
        count = metric(s, r"Num of target nodes:\s*(\d+)")
        flips = metric(s, r"Num of misclassification:\s*(\d+)")
        three_seeds[seed] = [
            flips / count,
            metric(s, r"Metric 2 - Fidelity:\s*([\d.]+)"),
            metric(s, r"Metric 3 - Average Explanation Size:\s*([\d.]+)"),
            metric(s, r"Metric 4 - Average Plausibility:\s*([\d.]+)"),
            metric(s, r"Metric 5 - Average Time Cost:\s*([\d.]+)"),
        ]
    rows = []
    for i, (name, reported) in enumerate(paper.items()):
        vals = [three_seeds[seed][i] for seed in (102, 103, 104)]
        rows.append({
            "metric": name, "paper": reported,
            "seed_102": vals[0], "seed_103": vals[1], "seed_104": vals[2],
            "mean": statistics.mean(vals), "sample_std": statistics.stdev(vals),
        })
    write_csv(ROOT / "table7_results.csv", list(rows[0]), rows)


def find_examples(budget):
    folders = [p for p in (ROOT / "results").glob("*/counterfactual_subgraph_GCN/*")
               if p.is_dir() and p.name.endswith(f"_cora_budget[{budget}]-{SEED}")]
    files = [f for folder in folders for f in folder.iterdir()
             if f.is_file() and f.name.startswith("cora_cf_examples_gcnlayer")
             and f.name.endswith(f"_seed{SEED}")]
    if not files:
        raise FileNotFoundError(f"Saved CF results missing for budget {budget}")
    return max(files, key=lambda f: f.stat().st_mtime)


def make_figure3():
    dataset = Dataset(root=str(ROOT / "dataset"), name="cora")
    adj, features, labels = dataset.adj, dataset.features, dataset.labels
    model_dir = ROOT / "model_save" / "GCN" / "cora" / f"{GCN_LAYER}-layer"
    model_path = model_dir / "gcn_model_seed104.pth"
    if not model_path.exists():
        model_path = model_dir / "gcn_model.pth"
        print("WARNING: Using active gcn_model.pth; verify it is seed 104.")
    if not model_path.exists():
        raise FileNotFoundError("Seed 104 GCN model not found")
    print("GCN checkpoint:", model_path)
    gcn = load_GCN_model(str(model_path), features, labels, HIDDEN_CHANNELS,
                         DROPOUT, DEVICE, LEARNING_RATE, WEIGHT_DECAY,
                         WITH_BIAS, GCN_LAYER)
    gcn.eval()
    with torch.no_grad():
        global_logits = gcn.forward(torch.tensor(features.toarray()),
                                    normalize_adj(torch.tensor(adj.toarray())))
        global_probs = global_logits.exp()

    rows = []
    for budget in BUDGETS:
        file = find_examples(budget)
        print(f"Evaluating k={budget}: {file}", flush=True)
        with file.open("rb") as f:
            examples = pickle.load(f)
        n = len(examples)
        valid = 0
        extra_flips_within_budget = 0
        flips_over_budget = 0
        invalid_flagged_success = 0
        edit_sum = 0
        plausibility_sum = 0.0
        fidelity_sum = 0.0
        times = []
        with torch.no_grad():
            for item in examples:
                d = item["data"]
                times.append(float(item["time_cost"]))
                orig = torch.as_tensor(d["extended_adj"]).detach()
                edited = torch.as_tensor(d["cf_adj"]).detach()
                edits = int(torch.triu(edited.ne(orig), diagonal=1).sum().item())
                node = int(d["target_node"])
                original_class = int(global_logits[node].argmax().item())
                new_logits = gcn.forward(d["extended_feat"], normalize_adj(edited))
                flipped = original_class != int(new_logits[int(d["new_idx"])].argmax().item())
                flagged = bool(d["success"])
                if flipped and edits > budget:
                    flips_over_budget += 1
                if flipped and edits <= budget and not flagged:
                    extra_flips_within_budget += 1
                if flagged and (not flipped or edits > budget):
                    invalid_flagged_success += 1
                # Consistent conservative definition: explainer success AND
                # independently verified flip AND no more than k edge edits.
                if flagged and flipped and edits <= budget:
                    valid += 1
                    edit_sum += edits
                    fidelity_sum += float(global_probs[node, original_class] -
                                          new_logits[int(d["new_idx"])].exp()[original_class])
                    loss = (α2 * compute_deg_diff(orig, edited) +
                            α3 * compute_motif_viol(orig, edited, TAU_C))
                    plausibility_sum += float(2 * (1 - 1 / (1 + torch.exp(-PLAU_K * loss))))

        rows.append({
            "budget": budget, "seed": SEED, "target_nodes": n,
            "valid_success": valid, "misclassification_rate": valid / n,
            "fidelity": fidelity_sum / n,
            "explanation_size": edit_sum / valid if valid else 0,
            "plausibility": plausibility_sum / valid if valid else 0,
            "runtime_s_per_node": statistics.mean(times),
            "extra_flips_within_budget_not_flagged": extra_flips_within_budget,
            "flips_over_budget_excluded": flips_over_budget,
            "invalid_flagged_success": invalid_flagged_success,
        })
        print(f"  valid {valid}/{n}; over-budget flips excluded: {flips_over_budget}; "
              f"extra valid flips not flagged: {extra_flips_within_budget}")
        if valid != EXPECTED_SUCCESS[budget]:
            raise ValueError(f"Budget {budget}: {valid} valid successes; expected "
                             f"{EXPECTED_SUCCESS[budget]}. Verify seed 104 model and result files.")
        del examples

    write_csv(ROOT / "figure3_budget_sweep.csv", list(rows[0]), rows)
    names = [
        ("misclassification_rate", "Misclassification Rate"),
        ("fidelity", "Fidelity"),
        ("explanation_size", "Explanation Size (edge edits)"),
        ("plausibility", "Plausibility"),
        ("runtime_s_per_node", "Runtime (s/node)"),
    ]
    xs = [r["budget"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, label) in zip(axes.flat, names):
        ax.plot(xs, [r[key] for r in rows], marker="o", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Perturbation budget (k)")
        ax.set_xticks(xs)
        ax.grid(alpha=0.25)
        # Save each metric separately for slides.
        single_fig, single_ax = plt.subplots(figsize=(7, 4.2))
        single_ax.plot(xs, [r[key] for r in rows], marker="o", linewidth=2)
        single_ax.set_title(label)
        single_ax.set_xlabel("Perturbation budget (k)")
        single_ax.set_xticks(xs)
        single_ax.grid(alpha=0.25)
        single_fig.tight_layout()
        single_fig.savefig(ROOT / f"figure3_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(single_fig)
    axes.flat[-1].axis("off")
    fig.suptitle("ATEX-CF — Cora + GCN, seed 104 (budget-valid results)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figure3_atex_cf_5_metrics.png", dpi=200, bbox_inches="tight")
    print("Saved: figure3_atex_cf_5_metrics.png and 5 individual plot PNGs")
    plt.show()


if __name__ == "__main__":
    make_table7()
    make_figure3()
    print("DONE: table7_results.csv, figure3_budget_sweep.csv, figure3_atex_cf_5_metrics.png")
