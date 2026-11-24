
import pickle
import statistics
from pathlib import Path

import torch

ROOT = Path("results/2026-11-24/counterfactual_subgraph_GCN")
FILENAME = "cora_cf_examples_gcnlayer2_lr0.01_epochs200_seed104"
BUDGET = 5


def load_results(pruning):
    folder = ROOT / (
        f"Evasion_GOttack_counterfactual_ACExplainer_"
        f"cora_budget[5]-104_pruning_{pruning}_backup"
    )
    path = folder / FILENAME

    with open(path, "rb") as f:
        examples = pickle.load(f)

    results = {}

    for example in examples:
        d = example["data"]

        original = torch.as_tensor(d["extended_adj"]).detach().cpu()
        edited = torch.as_tensor(d["cf_adj"]).detach().cpu()

        actual_edits = int(
            torch.triu(original != edited, diagonal=1).sum().item()
        )

        results[int(d["target_node"])] = {
            "success": bool(d["success"]),
            "actual_edits": actual_edits,
            "recorded_edits": int(d["explanation_size"]),
        }

    return results


on = load_results("on")
off = load_results("off")

valid_results = {}

for name, results in [("ON", on), ("OFF", off)]:
    successful = [r for r in results.values() if r["success"]]

    valid = {
        node: r
        for node, r in results.items()
        if r["success"] and r["actual_edits"] <= BUDGET
    }

    invalid = len(successful) - len(valid)

    print(f"\nPruning {name}")
    print("Original successful CFs:", len(successful))
    print("Over-budget successful CFs:", invalid)
    print("Budget-valid successful CFs:", len(valid))

    print(
        "Original reported explanation size:",
        round(statistics.mean(r["recorded_edits"] for r in successful), 4)
    )
    print(
        "Corrected budget-valid explanation size:",
        round(statistics.mean(r["actual_edits"] for r in valid.values()), 4)
    )

    valid_results[name] = valid


# Fair comparison: only nodes valid under BOTH conditions
common = set(valid_results["ON"]) & set(valid_results["OFF"])

on_size = statistics.mean(
    valid_results["ON"][n]["actual_edits"] for n in common
)
off_size = statistics.mean(
    valid_results["OFF"][n]["actual_edits"] for n in common
)

print("\nPAIRED COMPARISON")
print("Common valid nodes:", len(common))
print("Pruning OFF average edits:", round(off_size, 4))
print("Pruning ON average edits:", round(on_size, 4))
print("Average edits removed by pruning:", round(off_size - on_size, 4))
