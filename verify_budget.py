
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from deeprobust.graph.data import Dataset

from model.GCN import load_GCN_model
from utilty.utils import normalize_adj, compute_deg_diff, compute_motif_viol
import config.config as cfg
import csv

torch.set_num_threads(1)

# Load original Cora graph and saved seed-104 GCN.
data = Dataset(root="dataset", name="cora")
features = torch.tensor(data.features.toarray(), dtype=torch.float32)

model = load_GCN_model(
    "model_save/GCN/cora/2-layer/gcn_model_seed104.pth",
    data.features, data.labels,
    cfg.HIDDEN_CHANNELS, cfg.DROPOUT, "cpu",
    cfg.LEARNING_RATE, cfg.WEIGHT_DECAY,
    cfg.WITH_BIAS, cfg.GCN_LAYER
)
model.eval()
with torch.no_grad():
    full_adj = torch.tensor(data.adj.toarray(), dtype=torch.float32)
    original_outputs = model(features, normalize_adj(full_adj))

# Identify global node IDs from their feature vectors.
feature_lookup = defaultdict(list)
for node, row in enumerate(features.numpy()):
    feature_lookup[row.tobytes()].append(node)

root = Path("results/2026-11-24/counterfactual_subgraph_GCN")
filename = "cora_cf_examples_gcnlayer2_lr0.01_epochs200_seed104"
verified_rows = []
for variant in ("off", "on"):
    folder = root / (
        "Evasion_GOttack_counterfactual_ACExplainer_"
        f"cora_budget[5]-104_pruning_{variant}_backup"
    )

    with open(folder / filename, "rb") as f:
        examples = pickle.load(f)

    fidelity_sum = 0.0
    plausibility_sum = 0.0
    explanation_size_sum = 0

    success_count = 0
    verified_flips = 0
    over_budget = 0
    saved_matrix_over_budget = 0
    unresolved = []
    not_flipped = []
    prediction_mismatches = []

    with torch.no_grad():
        bounded_successes = 0
        for example in examples:
            d = example["data"]
            if not d or not bool(d["success"]):
                continue

            recorded_edits = len(d["added_edges"]) + len(d["removed_edges"])
            if (
                recorded_edits <= 5
                and int(d["new_pred"]) != int(d["original_pred"])
            ):
                bounded_successes += 1
            success_count += 1
            node = int(d["target_node"])

            original = torch.as_tensor(
                d["extended_adj"], dtype=torch.float32
            ).detach().clone()
            saved = torch.as_tensor(
                d["cf_adj"], dtype=torch.float32
            ).detach()

            saved_edits = int(
                torch.triu(original != saved, diagonal=1).sum()
            )
            if saved_edits > 5:
                saved_matrix_over_budget += 1

            local_features = torch.as_tensor(
                d["extended_feat"], dtype=torch.float32
            )


            # Recover only the endpoints of modified edges.
            changed_edges = (
                list(d["added_edges"]) + list(d["removed_edges"])
            )
            changed_nodes = {
                int(v) for edge in changed_edges for v in edge
            }

            local_by_feature = defaultdict(list)
            for i, row in enumerate(local_features.numpy()):
                local_by_feature[row.tobytes()].append(i)

            # Nodes with unique features are reliable anchors.
            anchors = {}
            for i, row in enumerate(local_features.numpy()):
                matches = feature_lookup.get(row.tobytes(), [])
                if len(matches) == 1:
                    anchors[i] = matches[0]

            # We also know the target node's exact position.
            anchors[int(d["new_idx"])] = node

            local_index = {}
            remaining = set(changed_nodes)

            while remaining:
                progress = False

                for global_id in list(remaining):
                    key = features[global_id].numpy().tobytes()
                    candidates = local_by_feature.get(key, [])

                    valid_candidates = []

                    for local_id in candidates:
                        valid = True

                        for anchor_local, anchor_global in anchors.items():
                            if local_id == anchor_local:
                                if global_id != anchor_global:
                                    valid = False
                                continue

                            if global_id == anchor_global:
                                valid = False
                                break

                            # extended_nodes is sorted by global ID.
                            if ((local_id < anchor_local) !=
                                    (global_id < anchor_global)):
                                valid = False
                                break

                            # Original graph connectivity must match.
                            if int(original[local_id, anchor_local]) != int(
                                data.adj[global_id, anchor_global]
                            ):
                                valid = False
                                break

                        if valid:
                            valid_candidates.append(local_id)

                    if len(valid_candidates) == 1:
                        local_id = valid_candidates[0]
                        local_index[global_id] = local_id
                        anchors[local_id] = global_id
                        remaining.remove(global_id)
                        progress = True

                if not progress:
                    break

            if remaining:
                unresolved.append(node)
                continue

            # Verify original connectivity between changed endpoints.
            changed_list = sorted(changed_nodes)
            consistent = all(
                int(original[local_index[u], local_index[v]])
                == int(data.adj[u, v])
                for u in changed_list
                for v in changed_list
            )

            if not consistent:
                unresolved.append(node)
                continue


            # Reconstruct CF from budget-constrained edge lists.
            rebuilt = original.clone()

            for u, v in d["added_edges"]:
                i, j = local_index[int(u)], local_index[int(v)]
                rebuilt[i, j] = rebuilt[j, i] = 1

            for u, v in d["removed_edges"]:
                i, j = local_index[int(u)], local_index[int(v)]
                rebuilt[i, j] = rebuilt[j, i] = 0

            actual_edits = int(
                torch.triu(original != rebuilt, diagonal=1).sum()
            )
            if actual_edits > 5:
                over_budget += 1

            output = model(local_features, normalize_adj(rebuilt))
            new_pred = int(output[int(d["new_idx"])].argmax())
            original_pred = int(d["original_pred"])
            recorded_new_pred = int(d["new_pred"])

            if new_pred != recorded_new_pred:
                prediction_mismatches.append(node)


            if new_pred != original_pred and actual_edits <= 5:
                verified_flips += 1
                explanation_size_sum += actual_edits

                # Fidelity: change in original-class probability.
                original_class = int(original_outputs[node].argmax())

                fidelity_sum += (
                    original_outputs[node, original_class].exp()
                    - output[int(d["new_idx"]), original_class].exp()
                ).item()

                # Plausibility: same formula as repository evaluator.
                plau_loss = (
                    cfg.α2 * compute_deg_diff(original, rebuilt)
                    + cfg.α3 * compute_motif_viol(
                        original, rebuilt, cfg.TAU_C
                    )
                )

                plausibility_sum += (
                    2.0 * torch.sigmoid(-cfg.k * plau_loss)
                ).item()

            else:
                not_flipped.append((node, actual_edits))


        print("Recorded budget-constrained successes:", bounded_successes)

    print(f"\n=== PRUNING {variant.upper()} ===")
    print("Recorded successes:", success_count)
    print("Saved matrices exceeding budget:", saved_matrix_over_budget)
    print("Reconstructed CFs exceeding budget:", over_budget)
    print("Verified budget-valid flips:", verified_flips)
    print("Prediction mismatches:", prediction_mismatches[:20])
    print("Not flipped:", not_flipped[:20])
    print("Unresolved node mappings:", len(unresolved))
    print("First unresolved nodes:", unresolved[:10])
    print("\nCORRECTED METRICS")
    print("Misclassification:", round(verified_flips / len(examples), 4))
    print("Fidelity:", round(fidelity_sum / len(examples), 4))
    print("Explanation size:", round(explanation_size_sum / verified_flips, 4))
    print("Plausibility:", round(plausibility_sum / verified_flips, 4))

    verified_rows.append({
        "pruning": variant,
        "seed": 104,
        "budget": 5,
        "total_nodes": len(examples),
        "verified_flips": verified_flips,
        "misclassification": verified_flips / len(examples),
        "fidelity": fidelity_sum / len(examples),
        "explanation_size": explanation_size_sum / verified_flips,
        "plausibility": plausibility_sum / verified_flips,
        "runtime_s_per_node": sum(
            float(ex["time_cost"]) for ex in examples
        ) / len(examples),
        "saved_matrices_over_budget": saved_matrix_over_budget,
        "reconstructed_over_budget": over_budget,
        "prediction_mismatches": len(prediction_mismatches),
        "unresolved_mappings": len(unresolved),
    })


with open("figure6_verified_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=verified_rows[0].keys())
    writer.writeheader()
    writer.writerows(verified_rows)

print("\nSaved: figure6_verified_metrics.csv")
