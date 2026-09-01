"""Global clone-lineage inference from an undirected similarity graph.

The optimizer treats consensus calls as fixed observations.  An oriented edge
is penalized only when it would produce an MT-to-WT transition, weighted by the
confidence of both endpoint calls.  A rooted spanning tree supplies guaranteed
reachability during the deterministic fallback optimization; remaining edges
are oriented jointly by a global topological order rather than independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _EdgeCost:
    u: Any
    v: Any
    similarity: float
    edge_weight: float
    forward_cost: float
    reverse_cost: float
    forward_loss: float
    reverse_loss: float
    informative_disagreement: float
    forward_error_rate: float
    reverse_error_rate: float


def _as_status_matrix(values: Any, nodes: list[Any], name: str) -> np.ndarray:
    """Convert WT/MT encodings to a numeric matrix with WT=0 and MT=1."""
    if isinstance(values, pd.DataFrame):
        if not values.index.is_unique or set(nodes) - set(values.index):
            raise ValueError(f"{name} DataFrame index must contain every graph node.")
        values = values.loc[nodes].to_numpy()
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != len(nodes):
        raise ValueError(f"{name} must have shape (n_nodes, n_mutations).")

    if array.dtype.kind in "OUS":
        normalized = np.char.upper(array.astype(str))
        if not np.isin(normalized, ("WT", "MT")).all():
            raise ValueError(f"{name} string values must be only WT or MT.")
        return (normalized == "MT").astype(np.int8)

    numeric = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    unique = set(np.unique(numeric).tolist())
    if unique <= {-1.0, 1.0}:
        return (numeric == 1.0).astype(np.int8)
    if unique <= {0.0, 1.0}:
        return numeric.astype(np.int8)
    raise ValueError(
        f"{name} must use WT/MT, -1/1, or 0/1 encoding; got {sorted(unique)}."
    )


def _as_confidence_matrix(values: Any, nodes: list[Any], shape: tuple[int, int]) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if not values.index.is_unique or set(nodes) - set(values.index):
            raise ValueError("confidence DataFrame index must contain every graph node.")
        values = values.loc[nodes].to_numpy()
    confidence = np.asarray(values, dtype=float)
    if confidence.shape != shape:
        raise ValueError("confidence must have the same shape as status.")
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence values must be finite and in [0, 1].")
    return confidence


def _edge_costs(
    graph: nx.Graph,
    status: np.ndarray,
    confidence: np.ndarray,
    nodes: list[Any],
    node_index: dict[Any, int],
    normal_node: Any,
    similarity_attr: str,
    normalize_edge_error: bool,
    uncertainty_penalty: float,
    similarity_power: float,
    eps: float = 1e-12,
) -> list[_EdgeCost]:
    similarities = []
    raw_edges = []
    for u, v, data in graph.edges(data=True):
        similarity = float(data.get(similarity_attr, 1.0))
        if not np.isfinite(similarity) or similarity < 0.0:
            raise ValueError(f"Edge ({u!r}, {v!r}) has an invalid similarity.")
        raw_edges.append((u, v, similarity))
        similarities.append(similarity)
    mean_similarity = float(np.mean(similarities)) if similarities else 1.0
    if mean_similarity <= 0.0:
        raise ValueError("The graph must contain at least one positive similarity.")

    result = []
    for u, v, raw_similarity in raw_edges:
        similarity = raw_similarity / mean_similarity
        edge_weight = similarity**similarity_power
        iu, iv = node_index[u], node_index[v]
        su, sv = status[iu], status[iv]
        cu, cv = confidence[iu], confidence[iv]
        product = cu * cv
        disagreement = float(np.sum(product * (su != sv)))
        forward_loss = float(np.sum(product * ((su == 1) & (sv == 0))))
        reverse_loss = float(np.sum(product * ((sv == 1) & (su == 0))))
        if normalize_edge_error:
            forward_error = forward_loss / (disagreement + eps)
            reverse_error = reverse_loss / (disagreement + eps)
        else:
            forward_error = forward_loss
            reverse_error = reverse_loss
        uncertainty = uncertainty_penalty / np.sqrt(disagreement + eps)
        result.append(
            _EdgeCost(
                u=u,
                v=v,
                similarity=similarity,
                edge_weight=edge_weight,
                forward_cost=edge_weight * (forward_error + uncertainty),
                reverse_cost=edge_weight * (reverse_error + uncertainty),
                forward_loss=forward_loss,
                reverse_loss=reverse_loss,
                informative_disagreement=disagreement,
                forward_error_rate=float(forward_error),
                reverse_error_rate=float(reverse_error),
            )
        )
    return result


def _virtual_root_edges(
    graph: nx.Graph,
    normal_node: Any,
    confidence: np.ndarray,
    node_index: dict[Any, int],
) -> list[tuple[Any, Any]]:
    """Connect disconnected components to normal for optimization only."""
    virtual = []
    for component in nx.connected_components(graph):
        if normal_node in component:
            continue
        source = max(
            component,
            key=lambda node: (
                float(np.mean(confidence[node_index[node]])),
                repr(node),
            ),
        )
        virtual.append((normal_node, source))
    return virtual


def _rooted_tree(graph: nx.Graph, normal_node: Any, virtual_edges: list[tuple[Any, Any]]) -> nx.DiGraph:
    augmented = graph.copy()
    augmented.add_edges_from(virtual_edges)
    tree = nx.bfs_tree(augmented, normal_node)
    if len(tree) != len(augmented):
        raise ValueError("Unable to connect every clone to the artificial normal.")
    return tree


def _topological_order(tree: nx.DiGraph, priority: dict[Any, tuple[float, str]]) -> list[Any]:
    indegree = dict(tree.in_degree())
    ready = [node for node, degree in indegree.items() if degree == 0]
    order = []
    while ready:
        ready.sort(key=lambda node: priority[node])
        node = ready.pop(0)
        order.append(node)
        for child in tree.successors(node):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(tree):
        raise ValueError("The rooted optimization tree contains a cycle.")
    return order


def _evaluate_order(
    edge_costs: list[_EdgeCost],
    order: list[Any],
    tree: nx.DiGraph,
    normal_node: Any,
    allow_unresolved: bool,
    unresolved_penalty: float,
) -> tuple[float, list[dict[str, Any]]]:
    position = {node: index for index, node in enumerate(order)}
    tree_edges = {(u, v) for u, v in tree.edges()}
    rows = []
    objective = 0.0
    for edge in edge_costs:
        uv = (edge.u, edge.v)
        vu = (edge.v, edge.u)
        forced_forward = uv in tree_edges or edge.u == normal_node
        forced_reverse = vu in tree_edges or edge.v == normal_node
        if forced_forward:
            selected = "forward"
        elif forced_reverse:
            selected = "reverse"
        elif allow_unresolved and unresolved_penalty * edge.edge_weight <= min(
            edge.forward_cost, edge.reverse_cost
        ):
            selected = "unresolved"
        else:
            selected = "forward" if position[edge.u] < position[edge.v] else "reverse"
        if selected == "forward":
            objective += edge.forward_cost
        elif selected == "reverse":
            objective += edge.reverse_cost
        else:
            objective += unresolved_penalty * edge.edge_weight
        rows.append({"edge": edge, "selected": selected})
    return objective, rows


def _directed_from_rows(
    nodes: list[Any],
    rows: list[dict[str, Any]],
    normal_node: Any,
) -> nx.DiGraph:
    directed = nx.DiGraph()
    directed.add_nodes_from(nodes)
    for row in rows:
        edge = row["edge"]
        if row["selected"] == "forward":
            directed.add_edge(edge.u, edge.v)
        elif row["selected"] == "reverse":
            directed.add_edge(edge.v, edge.u)
    if not nx.is_directed_acyclic_graph(directed):
        raise ValueError("Global orientation produced a directed cycle.")
    if directed.in_degree(normal_node) != 0:
        raise ValueError("The artificial normal cannot have incoming edges.")
    return directed


def infer_global_lineage(
    graph: nx.Graph,
    status: Any,
    confidence: Any,
    *,
    normal_node: Any,
    similarity_attr: str = "weight",
    allow_unresolved: bool = False,
    unresolved_penalty: float = 0.25,
    normalize_edge_error: bool = True,
    uncertainty_penalty: float = 0.0,
    similarity_power: float = 1.0,
    solver: str = "auto",
    time_limit: float | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Infer a globally consistent directed lineage from clone similarities.

    Consensus status is fixed. For each candidate direction, only MT-to-WT
    transitions contribute mutation loss, weighted by the confidence of both
    calls. All edges are oriented jointly by a rooted DAG construction rather
    than by independent pairwise decisions. The artificial normal is fixed at
    depth zero and supplies the unique optimization root. Disconnected
    components receive temporary virtual root edges; these are used only to
    construct a common rooted partial order and are excluded from the returned
    biological graph and error totals.

    ``solver='auto'`` currently selects the deterministic rooted-order
    optimizer when optional MILP/CP-SAT packages are unavailable. The returned
    ``optimal`` flag is therefore false for the heuristic solution.
    """
    del time_limit
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("graph must be an undirected networkx.Graph.")
    if normal_node not in graph:
        raise ValueError("normal_node must identify a graph node.")
    if not isinstance(allow_unresolved, (bool, np.bool_)):
        raise TypeError("allow_unresolved must be boolean.")
    if unresolved_penalty < 0.0 or uncertainty_penalty < 0.0 or similarity_power < 0.0:
        raise ValueError("Penalties and similarity_power must be nonnegative.")
    if solver not in {"auto", "heuristic", "gurobi", "ortools", "scip"}:
        raise ValueError("solver must be auto, heuristic, gurobi, ortools, or scip.")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer.")

    nodes = list(graph.nodes())
    node_index = {node: index for index, node in enumerate(nodes)}
    status_matrix = _as_status_matrix(status, nodes, "status")
    confidence_matrix = _as_confidence_matrix(confidence, nodes, status_matrix.shape)
    # The artificial normal is fixed as ancestral WT with fully reliable calls.
    normal_index = node_index[normal_node]
    status_matrix = status_matrix.copy()
    confidence_matrix = confidence_matrix.copy()
    status_matrix[normal_index] = 0
    confidence_matrix[normal_index] = 1.0

    costs = _edge_costs(
        graph,
        status_matrix,
        confidence_matrix,
        nodes,
        node_index,
        normal_node,
        similarity_attr,
        normalize_edge_error,
        uncertainty_penalty,
        similarity_power,
    )
    virtual_edges = _virtual_root_edges(graph, normal_node, confidence_matrix, node_index)
    tree = _rooted_tree(graph, normal_node, virtual_edges)
    burden = np.sum(status_matrix * confidence_matrix, axis=1)
    mean_confidence = np.mean(confidence_matrix, axis=1)

    # Deterministic restarts use alternative priorities while preserving the
    # rooted-tree partial order.
    priorities = []
    for mode in range(3):
        if mode == 0:
            key = {node: (float(burden[node_index[node]]), repr(node)) for node in nodes}
        elif mode == 1:
            key = {node: (-float(burden[node_index[node]]), repr(node)) for node in nodes}
        else:
            key = {
                node: (-float(mean_confidence[node_index[node]]), repr(node))
                for node in nodes
            }
        key[normal_node] = (-np.inf, repr(normal_node))
        priorities.append(key)

    best_objective = np.inf
    best_rows = None
    best_order = None
    for priority in priorities:
        order = _topological_order(tree, priority)
        objective, rows = _evaluate_order(
            costs, order, tree, normal_node, bool(allow_unresolved), unresolved_penalty
        )
        if objective < best_objective - 1e-12:
            best_objective, best_rows, best_order = objective, rows, order
    assert best_rows is not None and best_order is not None
    directed = _directed_from_rows(nodes, best_rows, normal_node)

    depths = {node: 0 for node in nodes}
    optimization_graph = nx.DiGraph()
    optimization_graph.add_nodes_from(nodes)
    optimization_graph.add_edges_from(virtual_edges)
    for row in best_rows:
        edge = row["edge"]
        if row["selected"] == "forward":
            optimization_graph.add_edge(edge.u, edge.v)
        elif row["selected"] == "reverse":
            optimization_graph.add_edge(edge.v, edge.u)
    if not nx.is_directed_acyclic_graph(optimization_graph):
        raise ValueError("Optimization graph is cyclic.")
    for node in nx.topological_sort(optimization_graph):
        if node == normal_node:
            depths[node] = 0
        else:
            depths[node] = max(
                (depths[parent] + 1 for parent in optimization_graph.predecessors(node)),
                default=0,
            )

    edge_rows = []
    selected_mass = 0.0
    informative_mass = 0.0
    error_mass = 0.0
    for row in best_rows:
        edge = row["edge"]
        selected = row["selected"]
        local = "forward" if edge.forward_cost < edge.reverse_cost else "reverse"
        if np.isclose(edge.forward_cost, edge.reverse_cost):
            local = "undetermined"
        if selected == "forward":
            parent, child = edge.u, edge.v
            selected_cost = edge.forward_cost
            selected_loss = edge.forward_loss
            selected_error = edge.forward_error_rate
            reverse_cost = edge.reverse_cost
            agrees = local == "forward"
        elif selected == "reverse":
            parent, child = edge.v, edge.u
            selected_cost = edge.reverse_cost
            selected_loss = edge.reverse_loss
            selected_error = edge.reverse_error_rate
            reverse_cost = edge.forward_cost
            agrees = local == "reverse"
        else:
            parent = child = None
            selected_cost = unresolved_penalty * edge.edge_weight
            selected_loss = 0.0
            selected_error = np.nan
            reverse_cost = min(edge.forward_cost, edge.reverse_cost)
            agrees = False
        if selected != "unresolved":
            selected_mass += edge.edge_weight
            informative_mass += edge.edge_weight * edge.informative_disagreement
            error_mass += edge.edge_weight * selected_loss
        edge_rows.append(
            {
                "node_u": edge.u,
                "node_v": edge.v,
                "selected_direction": (
                    f"{edge.u}->{edge.v}" if selected == "forward" else
                    f"{edge.v}->{edge.u}" if selected == "reverse" else "unresolved"
                ),
                "selected_parent": parent,
                "selected_child": child,
                "selected_cost": float(selected_cost),
                "reverse_cost": float(reverse_cost),
                "mutation_loss": float(selected_loss),
                "reverse_mutation_loss": float(
                    edge.reverse_loss if selected == "forward" else edge.forward_loss
                ),
                "informative_disagreement": edge.informative_disagreement,
                "selected_error_rate": float(selected_error),
                "reverse_error_rate": float(
                    edge.reverse_error_rate if selected == "forward" else edge.forward_error_rate
                ),
                "similarity": edge.similarity,
                "edge_weight": edge.edge_weight,
                "local_preferred_direction": (
                    f"{edge.u}->{edge.v}" if local == "forward" else
                    f"{edge.v}->{edge.u}" if local == "reverse" else "undetermined"
                ),
                "agrees_with_local_preference": bool(agrees),
                "direction_margin": float(reverse_cost - selected_cost),
                "parent_depth": depths[parent] if parent is not None else np.nan,
                "child_depth": depths[child] if child is not None else np.nan,
            }
        )
    edge_table = pd.DataFrame(edge_rows)

    node_rows = []
    for node in nodes:
        parents = sorted(directed.predecessors(node), key=repr)
        children = sorted(directed.successors(node), key=repr)
        incoming_error = float(sum(edge_table.loc[edge_table.selected_child == node, "selected_cost"]))
        outgoing_error = float(sum(edge_table.loc[edge_table.selected_parent == node, "selected_cost"]))
        node_rows.append(
            {
                "node": node,
                "depth": int(depths[node]),
                "indegree": int(directed.in_degree(node)),
                "outdegree": int(directed.out_degree(node)),
                "parents": parents,
                "children": children,
                "shortest_distance_from_normal": (
                    nx.shortest_path_length(optimization_graph, normal_node, node)
                    if nx.has_path(optimization_graph, normal_node, node) else np.inf
                ),
                "weighted_mt_burden": float(burden[node_index[node]]),
                "incoming_error": incoming_error,
                "outgoing_error": outgoing_error,
                "total_error": incoming_error + outgoing_error,
            }
        )
        directed.nodes[node].update(node_rows[-1])
    node_table = pd.DataFrame(node_rows)

    global_error_rate = error_mass / (informative_mass + 1e-12)
    return {
        "directed_graph": directed,
        "edge_table": edge_table,
        "node_table": node_table,
        "global_error_rate": float(global_error_rate),
        "global_error_mass": float(error_mass),
        "informative_mass": float(informative_mass),
        "objective": float(best_objective),
        "solver_status": "heuristic",
        "optimal": False,
        "optimality_gap": np.nan,
        "virtual_edges": list(virtual_edges),
        "normal_node": normal_node,
        "random_state": int(random_state),
    }
