
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import orca

from utilty.utils import CPU_Unpickler, BAShapesDataset
from attack.GOttack.orbit_table_generator import (
    generate_orbit_tables_from_count,
)

# Load the exact BA-SHAPES graph used for GCN training.
with open("dataset/BAShapes.pickle", "rb") as f:
    pyg_data = CPU_Unpickler(f).load()

data = BAShapesDataset(pyg_data)
adj = data.adj.tocsr()

assert (adj != adj.T).nnz == 0, "Graph is not undirected"

# Keep each undirected edge exactly once.
upper = sp.triu(adj, k=1).tocoo()
edges = np.column_stack(
    (upper.row, upper.col)
).astype(np.int32)

n_nodes = adj.shape[0]
print("Nodes:", n_nodes)
print("Edges:", len(edges))

# Count the 73 node orbits for graphlets up to size 5.
counts = orca.count_node_orbits(
    edges, n_nodes, graphlet_size=5
)

assert counts.shape == (n_nodes, 73)

# Use the repository's original table-construction function.
df = generate_orbit_tables_from_count(
    counts, list(range(n_nodes))
)

path = Path("dataset/orbit/BAShapes_orbit_df.csv")
df.to_csv(path)

print("Saved:", path)
print("Orbit counts:")
print(df["two_Orbit_type"].value_counts().head(10))
print("1518:", (df["two_Orbit_type"] == "1518").sum())
print("1519:", (df["two_Orbit_type"] == "1519").sum())
