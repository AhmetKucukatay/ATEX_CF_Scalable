
import csv
import pickle
from pathlib import Path

import torch
from deeprobust.graph.data import Dataset

import config.config as cfg
from model.GCN import load_GCN_model
from utilty.utils import (
    normalize_adj, compute_deg_diff, compute_motif_viol
)

torch.set_num_threads(1)

ROOT = Path("results/2026-11-24/counterfactual_subgraph_GCN")
BUDGET = 5

data = Dataset(root="dataset", name="cora")
features = torch.tensor(data.features.toarray(), dtype=torch.float32)
full_adj = torch.tensor(data.adj.toarray(), dtype=torch.float32)
norm_full_adj = normalize_adj(full_adj)

results = []

for seed in (102, 103, 104):
    print(f"\n=== SEED {seed} ===")

    model = load_GCN_model(
        f"model_save/GCN/cora/2-layer/gcn_model_seed{seed}.pth",
        data.features, data.labels,
        cfg.HIDDEN_CHANNELS, cfg.DROPOUT, "cpu",
        cfg.LEARNING_RATE, cfg.WEIGHT_DECAY,
        cfg.WITH_BIAS, cfg.GCN_LAYER
    )
    model.eval()

    with torch.no_grad():
        original_outputs = model(features, norm_full_adj)

    suffix = "_pruning_on_backup" if seed == 104 else ""
    folder = ROOT / (
        "Evasion_GOttack_counterfactual_ACExplainer_"
        f"cora_budget[5]-{seed}{suffix}"
    )
    filename = (
        f"cora_cf_examples_gcnlayer2_lr0.01_"
        f"epochs200_seed{seed}"
    )

    with open(folder / filename, "rb") as f:
        examples = pickle.load(f)

    raw_flips = 0
    valid_flips = 0
    recorded_successes = 0
    invalid_successes = 0
    false_flag_flips = 0
    false_flag_overbudget = 0

    fidelity_valid = 0.0
    fidelity_raw = 0.0
    size_sum = 0
    plausibility_sum = 0.0
    time_list = []

    with torch.no_grad():
        for example in examples:
            d = example.get("data")
            time_list.append(float(example["time_cost"]))

            if not d:
                continue

            node = int(d["target_node"])
            original = torch.as_tensor(
                d["extended_adj"], dtype=torch.float32
            )
            cf_adj = torch.as_tensor(
                d["cf_adj"], dtype=torch.float32
            )
            sub_features = torch.as_tensor(
                d["extended_feat"], dtype=torch.float32
            )

            edits = int(
                torch.triu(original != cf_adj, diagonal=1).sum()
            )

            output = model(sub_features, normalize_adj(cf_adj))
            original_class = int(original_outputs[node].argmax())
            new_idx = int(d["new_idx"])
            new_class = int(output[new_idx].argmax())

            flip = new_class != original_class
            success = bool(d["success"])

            contribution = (
                original_outputs[node, original_class].exp()
                - output[new_idx, original_class].exp()
            ).item()

            fidelity_raw += contribution

            if flip:
                raw_flips += 1

            if success:
                recorded_successes += 1

            if success and edits > BUDGET:
                invalid_successes += 1

            if not success and flip:
                false_flag_flips += 1
                if edits > BUDGET:
                    false_flag_overbudget += 1

            if success and flip and edits <= BUDGET:
                valid_flips += 1
                fidelity_valid += contribution
                size_sum += edits

                loss = (
                    cfg.α2 * compute_deg_diff(original, cf_adj)
                    + cfg.α3 * compute_motif_viol(
                        original, cf_adj, cfg.TAU_C
                    )
                )
                plausibility_sum += (
                    2 * torch.sigmoid(-cfg.k * loss)
                ).item()

    n = len(examples)

    row = {
        "seed": seed,
        "total_nodes": n,
        "recorded_successes": recorded_successes,
        "raw_flips": raw_flips,
        "valid_flips": valid_flips,
        "invalid_successes": invalid_successes,
        "false_flag_flips": false_flag_flips,
        "false_flag_overbudget": false_flag_overbudget,
        "misclassification": valid_flips / n,
        "fidelity_valid": fidelity_valid / n,
        "fidelity_raw": fidelity_raw / n,
        "explanation_size": size_sum / valid_flips,
        "plausibility": plausibility_sum / valid_flips,
        "runtime": sum(time_list) / len(time_list),
    }
    results.append(row)

    for key, value in row.items():
        print(f"{key}: {value}")

with open("table7_audit.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved: table7_audit.csv")
