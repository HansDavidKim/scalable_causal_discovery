# Utility functions for scalable causal discovery

import networkx as nx
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def nx_to_general_graph(G: nx.DiGraph) -> GeneralGraph:
    """
    Convert a networkx DiGraph (DAG) to causal-learn's GeneralGraph.

    Endpoint convention:
        Directed edge i --> j:
            graph[j, i] =  1  (ARROW at j)
            graph[i, j] = -1  (TAIL at i)

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph (typically a DAG from synthesize.py).

    Returns
    -------
    GeneralGraph
        Equivalent causal-learn graph with directed edges.
    """
    nodes_sorted = sorted(G.nodes())
    node_map = {node: idx for idx, node in enumerate(nodes_sorted)}

    graph_nodes = [GraphNode(f"X{node}") for node in nodes_sorted]
    gg = GeneralGraph(graph_nodes)

    for u, v in G.edges():
        i, j = node_map[u], node_map[v]
        gg.graph[j, i] = 1    # arrow at j
        gg.graph[i, j] = -1   # tail at i

    gg.reconstitute_dpath(gg.get_graph_edges())
    return gg


def general_graph_to_nx(G: GeneralGraph) -> nx.DiGraph:
    """
    Convert a causal-learn GeneralGraph to a networkx DiGraph.

    Directed edges (i --> j) are preserved.
    Undirected edges (i --- j) become two directed edges (i->j and j->i).
    Bidirected edges (i <-> j) become two directed edges (i->j and j->i).

    Parameters
    ----------
    G : GeneralGraph
        A causal-learn graph (DAG, CPDAG, or PAG).

    Returns
    -------
    nx.DiGraph
        Networkx directed graph.
    """
    n = G.get_num_nodes()
    nodes = G.get_nodes()
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i --> j means graph[j, i] = 1 and graph[i, j] = -1
            # But also capture undirected (-1, -1) and bidirected (1, 1)
            if G.graph[j, i] == 1 and G.graph[i, j] == -1:
                nx_graph.add_edge(i, j)
            elif G.graph[i, j] == -1 and G.graph[j, i] == -1:
                # Undirected: add both directions
                nx_graph.add_edge(i, j)
            elif G.graph[i, j] == 1 and G.graph[j, i] == 1:
                # Bidirected: add both directions
                nx_graph.add_edge(i, j)

    return nx_graph
