# Random DAG generation with topological order
# We assume we already know the topological order

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# O(N^2) DAG Generation Algorithm (Erdős–Rényi with fixed topological order)
def synthesize_dag(n: int = 10, edge_prob: float = 0.3, show_graph: bool = False)->nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Topological order is simply 0, 1, ..., n-1
    # Only add edges from i -> j where i < j (guarantees acyclicity)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < edge_prob:
                G.add_edge(i, j)

    if show_graph:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                node_size=500, arrowsize=15, font_size=10)
        plt.title(f"Random DAG (n={n}, p={edge_prob}, edges={G.number_of_edges()})")
        plt.show()

    return G

# Noise types
GAUSSIAN = "gaussian"
UNIFORM = "uniform"
LAPLACE = "laplace"
EXPONENTIAL = "exponential"
NOISE_TYPES = [GAUSSIAN, UNIFORM, LAPLACE, EXPONENTIAL]


def _sample_noise(noise_type: str, n_samples: int) -> np.ndarray:
    """Sample noise from the specified distribution."""
    if noise_type == GAUSSIAN:
        return np.random.randn(n_samples)
    elif noise_type == UNIFORM:
        return np.random.uniform(-1, 1, n_samples)
    elif noise_type == LAPLACE:
        return np.random.laplace(0, 1, n_samples)
    elif noise_type == EXPONENTIAL:
        return np.random.exponential(1, n_samples) - 1  # centered
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


# Dataset generation based on causal diagram (Linear SEM)
def generate_data_from_dag(
    diagram: nx.DiGraph,
    n_samples: int = 1000,
    noise_type: str = GAUSSIAN,
    noise_scale: float = 1.0,
    weight_range: tuple = (0.5, 2.0),
) -> np.ndarray:
    """
    Generate observational data from a DAG using a linear Structural Equation Model.

    X_j = sum_{i in parents(j)} w_ij * X_i + noise_j

    Parameters
    ----------
    diagram : nx.DiGraph — the causal DAG
    n_samples : int — number of data points to generate
    noise_type : str — one of GAUSSIAN, UNIFORM, LAPLACE, EXPONENTIAL
    noise_scale : float — scale factor for noise
    weight_range : tuple — (min, max) for absolute edge weight (sign is random)

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_nodes)
    """
    nodes = list(nx.topological_sort(diagram))
    n_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Assign random edge weights
    W = np.zeros((n_nodes, n_nodes))
    for u, v in diagram.edges():
        sign = np.random.choice([-1, 1])
        weight = np.random.uniform(*weight_range)
        W[node_to_idx[u], node_to_idx[v]] = sign * weight

    # Generate data following topological order
    X = np.zeros((n_samples, n_nodes))
    for node in nodes:
        j = node_to_idx[node]
        # Sum of weighted parent values
        parents = list(diagram.predecessors(node))
        for parent in parents:
            i = node_to_idx[parent]
            X[:, j] += W[i, j] * X[:, i]
        # Add noise
        X[:, j] += noise_scale * _sample_noise(noise_type, n_samples)

    return X

if __name__ == "__main__":
    seed: int = int(input('Enter Random Seed: '))
    np.random.seed(42)

    dag = synthesize_dag(n=10, edge_prob=0.3, show_graph=True)
    print(f"Nodes: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(dag)}")

    X = generate_data_from_dag(dag, n_samples=1000, noise_type=GAUSSIAN)
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    print(f"\nGenerated dataset shape: {df.shape}")
    print(df.head(5))
    print(df.describe())