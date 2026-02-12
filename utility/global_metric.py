# Measuring Performance for Causal Discovery Algorithms (Global Causal Discovery)
# Metrics: Accuracy, Precision, Recall, F1, SHD, NSHD, SID
#
# Uses causal-learn's GeneralGraph internally.
#
# GeneralGraph endpoint matrix convention:
#   graph[j,i] =  1, graph[i,j] = -1  →  directed  i --> j
#   graph[i,j] = -1, graph[j,i] = -1  →  undirected i --- j
#   graph[i,j] =  1, graph[j,i] =  1  →  bidirected i <-> j
#   graph[i,j] =  0, graph[j,i] =  0  →  no edge

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _validate_graphs(truth: GeneralGraph, est: GeneralGraph) -> None:
    """Ensure both graphs have the same node set."""
    truth_names = set(n.get_name() for n in truth.get_nodes())
    est_names = set(n.get_name() for n in est.get_nodes())
    if truth_names != est_names:
        raise ValueError(
            f"Node sets differ.\n"
            f"  True graph nodes: {sorted(truth_names)}\n"
            f"  Est. graph nodes: {sorted(est_names)}"
        )


def _adj_confusion_counts(truth: GeneralGraph, est: GeneralGraph):
    """
    Compute TP, FP, FN, TN at the skeleton level (adjacency).

    For each unordered pair (i, j), two nodes are "adjacent" if any
    edge (directed, undirected, or bidirected) exists between them.
    This is direction-agnostic.
    """
    _validate_graphs(truth, est)
    n = truth.get_num_nodes()

    TP = FP = FN = TN = 0
    truth_names = [node.get_name() for node in truth.get_nodes()]

    for i in range(n):
        for j in range(i + 1, n):
            t_adj = truth.is_adjacent_to(
                truth.get_node(truth_names[i]),
                truth.get_node(truth_names[j])
            )
            e_adj = est.is_adjacent_to(
                est.get_node(truth_names[i]),
                est.get_node(truth_names[j])
            )

            if t_adj and e_adj:
                TP += 1
            elif not t_adj and e_adj:
                FP += 1
            elif t_adj and not e_adj:
                FN += 1
            else:
                TN += 1

    return TP, FP, FN, TN


def _get_descendants(G: GeneralGraph, node_idx: int) -> set:
    """
    Get all descendants of node at index node_idx using the dpath matrix.
    dpath[j, i] == 1 means i is an ancestor of j (i.e., j is a descendant of i).
    """
    n = G.get_num_nodes()
    desc = set()
    for j in range(n):
        if j != node_idx and G.dpath[j, node_idx] == 1:
            desc.add(j)
    return desc


def _get_parent_indices(G: GeneralGraph, node_idx: int) -> set:
    """
    Get parent indices of node at node_idx.
    Parent: graph[i, j] = -1 and graph[j, i] = 1  means  node_i --> node_j
    So for node_j, parent is node_i where graph[node_idx, i] = 1 and graph[i, node_idx] = -1.
    """
    n = G.get_num_nodes()
    parents = set()
    for i in range(n):
        if G.graph[node_idx, i] == 1 and G.graph[i, node_idx] == -1:
            parents.add(i)
    return parents


# ──────────────────────────────────────────────────────────────
# Classification-Based Metrics (Skeleton / Adjacency level)
# ──────────────────────────────────────────────────────────────

def accuracy(truth: GeneralGraph, est: GeneralGraph) -> float:
    """
    Accuracy = (TP + TN) / (TP + FP + FN + TN)

    Evaluated on the skeleton (adjacency, direction-agnostic).
    """
    TP, FP, FN, TN = _adj_confusion_counts(truth, est)
    total = TP + FP + FN + TN
    return (TP + TN) / total if total > 0 else 0.0


def precision(truth: GeneralGraph, est: GeneralGraph) -> float:
    """
    Precision = TP / (TP + FP)

    Among edges the algorithm predicted (skeleton), how many are correct?
    """
    TP, FP, _, _ = _adj_confusion_counts(truth, est)
    denom = TP + FP
    return TP / denom if denom > 0 else 0.0


def recall(truth: GeneralGraph, est: GeneralGraph) -> float:
    """
    Recall = TP / (TP + FN)

    Among actual edges (skeleton), how many did the algorithm find?
    """
    TP, _, FN, _ = _adj_confusion_counts(truth, est)
    denom = TP + FN
    return TP / denom if denom > 0 else 0.0


def f1_score(truth: GeneralGraph, est: GeneralGraph) -> float:
    """
    F1 = 2 * Precision * Recall / (Precision + Recall)

    Harmonic mean of precision and recall (skeleton level).
    """
    p = precision(truth, est)
    r = recall(truth, est)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Structure-Based Metrics
# ──────────────────────────────────────────────────────────────

def shd(truth: GeneralGraph, est: GeneralGraph) -> int:
    """
    Structural Hamming Distance (SHD).

    For each node pair (i, j) where i <= j, compare endpoint pairs:
        (truth.graph[i,j], truth.graph[j,i])  vs  (est.graph[i,j], est.graph[j,i])

    Any mismatch counts as +1. This matches causal-learn's own SHD class.
    """
    _validate_graphs(truth, est)
    n = truth.get_num_nodes()

    truth_map = {node.get_name(): node_id for node, node_id in truth.node_map.items()}
    est_map = {node.get_name(): node_id for node, node_id in est.node_map.items()}

    distance = 0
    for name_i, t_i in truth_map.items():
        for name_j, t_j in truth_map.items():
            if t_j < t_i:
                continue  # only upper triangle (i <= j), including self-loops

            e_i, e_j = est_map[name_i], est_map[name_j]

            truth_endpoints = (truth.graph[t_i, t_j], truth.graph[t_j, t_i])
            est_endpoints = (est.graph[e_i, e_j], est.graph[e_j, e_i])

            if truth_endpoints != est_endpoints:
                distance += 1

    return distance


def nshd(truth: GeneralGraph, est: GeneralGraph) -> float:
    """
    Normalized SHD.

    NSHD = SHD / (n choose 2)

    Normalized by the total number of node pairs.
    Returns a value in [0, 1].
    """
    _validate_graphs(truth, est)
    n = truth.get_num_nodes()
    max_pairs = n * (n - 1) // 2
    if max_pairs == 0:
        return 0.0
    return shd(truth, est) / max_pairs


def sid(truth: GeneralGraph, est: GeneralGraph) -> int:
    """
    Structural Intervention Distance (SID).
    (Peters and Buhlmann, 2015)

    SID is defined for DAG vs DAG. If est is a CPDAG, undirected edges
    are oriented by node index (smaller -> larger) as a consistent extension.

    For each ordered pair (i, j) where i != j:
        error(i, j) = 1  if  pa_true(j) & (desc_true(i) | {i})
                              != pa_est(j) & (desc_est(i) | {i})

    SID = sum of error(i, j) over all ordered pairs.
    """
    _validate_graphs(truth, est)

    # Orient undirected edges in est for consistent DAG extension
    est_dag = _cpdag_to_dag(est)

    n = truth.get_num_nodes()

    # Precompute descendants and parents
    desc_truth = {i: _get_descendants(truth, i) for i in range(n)}
    desc_est = {i: _get_descendants(est_dag, i) for i in range(n)}

    distance = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pa_truth_j = _get_parent_indices(truth, j)
            pa_est_j = _get_parent_indices(est_dag, j)

            desc_truth_i = desc_truth[i] | {i}
            desc_est_i = desc_est[i] | {i}

            # Causal parents of j reachable from i
            relevant_truth = pa_truth_j & desc_truth_i
            relevant_est = pa_est_j & desc_est_i

            if relevant_truth != relevant_est:
                distance += 1

    return distance


def _cpdag_to_dag(G: GeneralGraph) -> GeneralGraph:
    """
    Convert a CPDAG to a consistent DAG extension.

    Undirected edges (graph[i,j] == -1 and graph[j,i] == -1)
    are oriented by node index: i --> j where i < j.

    If the graph is already a DAG, returns a copy.
    """
    nodes = G.get_nodes()
    new_nodes = [GraphNode(node.get_name()) for node in nodes]
    dag = GeneralGraph(new_nodes)

    n = G.get_num_nodes()
    for i in range(n):
        for j in range(i + 1, n):
            e_ij = G.graph[i, j]
            e_ji = G.graph[j, i]

            if e_ij == 0 and e_ji == 0:
                # No edge
                continue
            elif e_ij == -1 and e_ji == -1:
                # Undirected -> orient as i --> j (i < j)
                dag.graph[j, i] = 1    # arrow at j
                dag.graph[i, j] = -1   # tail at i
            else:
                # Keep original orientation
                dag.graph[i, j] = e_ij
                dag.graph[j, i] = e_ji

    # Reconstitute directed paths
    dag.reconstitute_dpath(dag.get_graph_edges())

    return dag


# ──────────────────────────────────────────────────────────────
# Convenience: compute all metrics at once
# ──────────────────────────────────────────────────────────────

def evaluate_all(truth: GeneralGraph, est: GeneralGraph) -> dict:
    """
    Compute all global evaluation metrics and return as a dict.

    Parameters
    ----------
    truth : GeneralGraph -- the ground-truth DAG
    est   : GeneralGraph -- the estimated graph (DAG or CPDAG)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, shd, nshd, sid
    """
    return {
        "accuracy":  accuracy(truth, est),
        "precision": precision(truth, est),
        "recall":    recall(truth, est),
        "f1":        f1_score(truth, est),
        "shd":       shd(truth, est),
        "nshd":      nshd(truth, est),
        "sid":       sid(truth, est),
    }


# ──────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _make_graph(n: int, directed_edges: list, undirected_edges: list = None) -> GeneralGraph:
        """Helper to build a GeneralGraph for testing."""
        nodes = [GraphNode(f"X{i}") for i in range(n)]
        G = GeneralGraph(nodes)
        for i, j in directed_edges:
            G.graph[j, i] = 1    # arrow at j
            G.graph[i, j] = -1   # tail at i
        if undirected_edges:
            for i, j in undirected_edges:
                G.graph[i, j] = -1
                G.graph[j, i] = -1
        G.reconstitute_dpath(G.get_graph_edges())
        return G

    # ── Test 1: DAG vs DAG ──
    print("=" * 55)
    print("Test 1: DAG vs DAG")
    print("=" * 55)

    # True DAG:  X0 -> X1 -> X2 -> X3
    true_g = _make_graph(4, [(0, 1), (1, 2), (2, 3)])

    # Estimated DAG:  X0 -> X1, X2 -> X1 (reversed), X2 -> X3, X0 -> X3 (extra)
    est_g = _make_graph(4, [(0, 1), (2, 1), (2, 3), (0, 3)])

    print(f"  True graph matrix:\n{true_g.graph}")
    print(f"  Est. graph matrix:\n{est_g.graph}")

    results = evaluate_all(true_g, est_g)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"    {metric:>10s} = {value:.4f}")
        else:
            print(f"    {metric:>10s} = {value}")

    # ── Test 2: DAG vs CPDAG ──
    print()
    print("=" * 55)
    print("Test 2: DAG vs CPDAG (estimated has undirected edge)")
    print("=" * 55)

    # True DAG:  X0 -> X1 -> X2 -> X3
    # Estimated CPDAG:  X0 -> X1,  X1 --- X2 (undirected),  X2 -> X3
    est_cpdag = _make_graph(4,
        directed_edges=[(0, 1), (2, 3)],
        undirected_edges=[(1, 2)]
    )

    print(f"  True graph matrix:\n{true_g.graph}")
    print(f"  Est. CPDAG matrix:\n{est_cpdag.graph}")

    results = evaluate_all(true_g, est_cpdag)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"    {metric:>10s} = {value:.4f}")
        else:
            print(f"    {metric:>10s} = {value}")

    # ── Test 3: Perfect match ──
    print()
    print("=" * 55)
    print("Test 3: Perfect match")
    print("=" * 55)

    results = evaluate_all(true_g, true_g)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"    {metric:>10s} = {value:.4f}")
        else:
            print(f"    {metric:>10s} = {value}")
