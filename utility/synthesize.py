# Random DAG generation with topological order
# We assume we already know the topological order

import networkx as nx
import matplotlib.pyplot as plt
import random

# O(N^2) DAG Generation Algorithm (Erdős–Rényi with fixed topological order)
def synthesize_dag(n: int = 10, edge_prob: float = 0.3, show_graph: bool = False):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Topological order is simply 0, 1, ..., n-1
    # Only add edges from i -> j where i < j (guarantees acyclicity)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_prob:
                G.add_edge(i, j)

    if show_graph:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                node_size=500, arrowsize=15, font_size=10)
        plt.title(f"Random DAG (n={n}, p={edge_prob}, edges={G.number_of_edges()})")
        plt.show()

    return G


if __name__ == "__main__":
    dag = synthesize_dag(n=10, edge_prob=0.3, show_graph=True)
    print(f"Nodes: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(dag)}")