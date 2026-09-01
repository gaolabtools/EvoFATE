"""Single-cell lineage projection tools.

The projection uses only a precomputed cell-cell connectivity matrix. Any
integrative embedding contributes upstream, when that matrix is constructed.
"""

from __future__ import annotations

import logging

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx

import numpy as np

import pandas as pd

from scipy import sparse
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.stats import chi2, f as f_distribution
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer
from sklearn.isotonic import IsotonicRegression

def _connectivity_from_embedding(embedding: np.ndarray, n_neighbors: int = 30) -> sparse.csr_matrix:
    """Build a sparse adaptive-affinity KNN graph without modifying AnnData."""
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or not np.all(np.isfinite(embedding)):
        raise ValueError("The embedding must be a finite two-dimensional array.")
    n_cells = embedding.shape[0]
    if n_cells < 2:
        raise ValueError("At least two cells are required to construct connectivity.")
    k = min(max(int(n_neighbors), 1), n_cells - 1)
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(embedding)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    sigma = np.maximum(distances[:, -1], 1e-12)
    rows = np.repeat(np.arange(n_cells), k)
    cols = indices.ravel()
    values = np.exp(-(distances.ravel() ** 2) / (sigma[rows] * sigma[cols] + 1e-12))
    graph = sparse.coo_matrix((values, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    graph = ((graph + graph.T) * 0.5).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

def _trajectory_graph(lineage: Any) -> nx.DiGraph:
    """Normalize common lineage inputs to a string-labeled DiGraph."""
    if lineage is None:
        raise KeyError(
            "No lineage was provided and `adata.uns['Lineage_tree']` is missing."
        )
    graph = nx.DiGraph()
    if isinstance(lineage, nx.DiGraph):
        graph.add_nodes_from((str(node), dict(data)) for node, data in lineage.nodes(data=True))
        graph.add_edges_from((str(parent), str(child), dict(data)) for parent, child, data in lineage.edges(data=True))
        return graph
    if hasattr(lineage, "get_edgelist"):
        graph.add_nodes_from(str(node) for node in range(int(lineage.vcount())))
        graph.add_edges_from((str(parent), str(child)) for parent, child in lineage.get_edgelist())
        return graph
    if isinstance(lineage, Mapping):
        edges = lineage.get("edges", [])
    else:
        edges = lineage
    try:
        graph.add_edges_from((str(edge[0]), str(edge[1])) for edge in edges)
    except (TypeError, IndexError):
        raise TypeError("`lineage` must be a directed NetworkX graph or edge list.") from None
    return graph

def _evofate_analysis_graph(adata, lineage=None) -> nx.DiGraph:
    """Return a string-labeled directed lineage graph for downstream modules."""
    value = lineage if lineage is not None else adata.uns.get("Lineage_tree")
    graph = _trajectory_graph(value)
    normalized = nx.DiGraph()
    normalized.add_nodes_from(str(node) for node in graph.nodes)
    normalized.add_edges_from((str(source), str(target)) for source, target in graph.edges)
    return normalized

def _evofate_cell_graph(adata, state, connectivity_key, n_neighbors):
    """Load the EvoFATE graph or construct it from lineage-state coordinates."""
    if connectivity_key in adata.obsp:
        raw = adata.obsp[connectivity_key]
        graph = raw.tocsr().astype(float) if sparse.issparse(raw) else sparse.csr_matrix(raw, dtype=float)
        if graph.shape != (adata.n_obs, adata.n_obs):
            raise ValueError("EvoFATE connectivity must have shape (n_obs, n_obs).")
        graph = graph.copy()
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph
    return _connectivity_from_embedding(state, n_neighbors=n_neighbors)

def _evofate_lineage_paths(graph, observed):
    """Return stable directed root-to-terminal paths containing observed clones."""
    roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
    terminals = [node for node in graph.nodes if graph.out_degree(node) == 0]
    paths = []
    for terminal in terminals:
        candidates = [
            nx.shortest_path(graph, root, terminal)
            for root in roots
            if nx.has_path(graph, root, terminal)
        ]
        if candidates:
            path = max(candidates, key=lambda value: (len(value), tuple(map(str, value))))
            path = [node for node in path if node in observed]
            if len(path) >= 2:
                paths.append(path)
    return paths

def cal_clonal_evofate(
    adata,
    state_key: str = "X_integrated",
    clone_key: str = "ordered_clone",
    lineage=None,
    key_added: str = "evofate_clonal",
    copy: bool = False,
):
    """Quantify clone heterogeneity and directed lineage state changes."""
    if copy:
        adata = adata.copy()
    if state_key not in adata.obsm:
        raise KeyError(f"`adata.obsm['{state_key}']` is missing.")
    if clone_key not in adata.obs:
        raise KeyError(f"`adata.obs['{clone_key}']` is missing.")
    state = np.asarray(adata.obsm[state_key], dtype=float)
    if state.ndim != 2 or state.shape[0] != adata.n_obs or not np.isfinite(state).all():
        raise ValueError("EvoFATE state must be finite with one row per cell.")
    labels = adata.obs[clone_key].astype(str).to_numpy()
    if np.any(pd.isna(labels)):
        raise ValueError("Clone labels cannot contain missing values.")
    clone_ids = np.unique(labels)
    centroids = {
        clone: state[labels == clone].mean(axis=0)
        for clone in clone_ids
    }
    intra = {
        clone: float(np.sqrt(np.mean(np.sum((state[labels == clone] - centroids[clone]) ** 2, axis=1))))
        for clone in clone_ids
    }
    graph = _evofate_analysis_graph(adata, lineage=lineage)
    edges = {}
    for parent, child in graph.edges:
        if parent in centroids and child in centroids:
            edges[f"{parent}->{child}"] = float(np.linalg.norm(centroids[child] - centroids[parent]))
    adata.uns[key_added] = {
        "clone_centroid": {clone: value.astype(float) for clone, value in centroids.items()},
        "intra_plasticity": intra,
        "edge_state_change": edges,
    }
    return adata if copy else None

def _connectivity_gated_clone_time(
    adjacency,
    clone_labels,
    cell_indices,
    clone_times,
    adjacent_clones,
):
    """Shift clone time only toward topology-valid adjacent clone support."""
    cells = np.asarray(cell_indices, dtype=int)
    labels = np.asarray(clone_labels, dtype=str)
    local_labels = labels[cells]
    graph_rows = adjacency[cells].tocsr().astype(float)
    eps = np.finfo(float).eps
    own_support = np.zeros(cells.size, dtype=float)
    out_support = np.zeros(cells.size, dtype=float)
    target_numerator = np.zeros(cells.size, dtype=float)
    for clone in np.unique(local_labels):
        clone_mask = labels == clone
        local_mask = local_labels == clone
        own_support[local_mask] = np.asarray(
            graph_rows[local_mask][:, clone_mask].sum(axis=1)
        ).ravel()
        for adjacent in adjacent_clones.get(clone, ()):
            adjacent_mask = labels == str(adjacent)
            support = np.asarray(
                graph_rows[local_mask][:, adjacent_mask].sum(axis=1)
            ).ravel()
            out_support[local_mask] += support
            target_numerator[local_mask] += support * float(clone_times[adjacent])
    target_time = np.divide(
        target_numerator,
        out_support,
        out=np.full(cells.size, np.nan, dtype=float),
        where=out_support > eps,
    )
    base_time = np.asarray([clone_times.get(clone, np.nan) for clone in local_labels], dtype=float)
    out_fraction = np.divide(
        out_support,
        own_support + out_support,
        out=np.zeros(cells.size, dtype=float),
        where=(own_support + out_support) > eps,
    )
    progression = base_time.copy()
    has_external = np.isfinite(target_time) & np.isfinite(base_time)
    progression[has_external] = (
        base_time[has_external]
        + out_fraction[has_external]
        * (target_time[has_external] - base_time[has_external])
    )
    return progression, {
        "own_support": own_support,
        "out_support": out_support,
        "out_fraction": out_fraction,
        "target_time": target_time,
    }

def cal_single_cell_evofate(
    adata,
    integrated_key: str = "X_integrated",
    connectivity_key: str = "evofate_lineage_connectivities",
    clone_key: str = "ordered_clone",
    embedding_key: str = "X_evofate_projection",
    n_neighbors: int | None = None,
    smoothing_strength: float = 0.25,
    inter_weight: float = 0.5,
    min_effective_neighbors: float = 5.0,
    min_lineage_support: float = 0.20,
    min_lineage_margin: float = 0.10,
    min_outward_lineage_support: float = 1e-6,
    min_anchor_labels: int = 5,
    lineage_propagation_alpha: float = 0.5,
    lineage_propagation_iter: int = 10,
    use_tmb_time: bool = True,
    enforce_clone_monotonicity: bool = True,
    store_diagnostics: bool = False,
    copy: bool = False,
):
    """Build the unified single-cell EvoFATE map.

    The authoritative single-cell order is stored in
    ``adata.obs['evofate_progression']``.  The rendered flow curves
    in ``adata.uns['evofate_trajectory']`` are fitted afterward as geometry
    from that fixed order and never redefine it. Every cell receives one hard
    lineage assignment; the support and margin parameters are retained for
    API compatibility but do not filter assignments.
    """
    if copy:
        adata = adata.copy()
    for key, location_name, location in (
        (integrated_key, "obsm", adata.obsm),
        (clone_key, "obs", adata.obs),
    ):
        if key not in location:
            raise KeyError(f"`adata.{location_name}['{key}']` is missing.")
    state = np.asarray(adata.obsm[integrated_key], dtype=float)
    labels = adata.obs[clone_key].astype(str).to_numpy()
    if state.ndim != 2 or state.shape[0] != adata.n_obs or not np.isfinite(state).all():
        raise ValueError("Integrated state must be finite with one row per cell.")
    if not 0.0 <= float(smoothing_strength) <= 1.0:
        raise ValueError("`smoothing_strength` must be between 0 and 1.")
    if float(inter_weight) < 0.0 or float(min_effective_neighbors) <= 0.0:
        raise ValueError("`inter_weight` must be nonnegative and minimum support positive.")
    if n_neighbors is None:
        n_neighbors = min(max(round(0.01 * adata.n_obs), 10), 200)
    if int(n_neighbors) < 1:
        raise ValueError("`n_neighbors` must be positive or None.")
    if not 0.0 <= float(min_lineage_support) <= 1.0:
        raise ValueError("`min_lineage_support` must be between 0 and 1.")
    clone_ids = np.unique(labels)
    centroids = {clone: state[labels == clone].mean(axis=0) for clone in clone_ids}
    graph = _evofate_analysis_graph(adata)
    neighbor_clones = {clone: set() for clone in clone_ids}
    for source, target in graph.edges:
        if source in neighbor_clones and target in neighbor_clones:
            neighbor_clones[source].add(target)
            neighbor_clones[target].add(source)
    # Always rebuild this graph from X_integrated so n_neighbors has a
    # deterministic effect. The graph is stored for inspection/reuse by
    # downstream routines, but is not treated as a stale input here.
    cell_graph = _evofate_cell_graph(adata, state, None, n_neighbors)
    adata.obsp[connectivity_key] = cell_graph.astype(np.float32)
    local_state = np.zeros_like(state)
    intra_shift = np.zeros(adata.n_obs, dtype=float)
    inter_affinity = np.zeros(adata.n_obs, dtype=float)
    local_support = np.zeros(adata.n_obs, dtype=float)
    eps = 1e-12
    for cell in range(adata.n_obs):
        start, stop = cell_graph.indptr[cell], cell_graph.indptr[cell + 1]
        neighbors = cell_graph.indices[start:stop]
        weights = cell_graph.data[start:stop]
        if not neighbors.size or weights.sum() <= eps:
            local_state[cell] = state[cell]
            continue
        center = np.average(state[neighbors], axis=0, weights=weights)
        distances = np.linalg.norm(state[neighbors] - center, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        threshold = median_distance + 2.5 * max(mad, eps)
        keep = distances <= threshold
        if not np.any(keep):
            keep[np.argmin(distances)] = True
        retained_weights = weights[keep]
        retained_states = state[neighbors[keep]]
        local_state[cell] = np.average(retained_states, axis=0, weights=retained_weights)
        sum_w = float(retained_weights.sum())
        n_eff = sum_w**2 / (float(np.sum(retained_weights**2)) + eps)
        local_support[cell] = min(1.0, n_eff / float(min_effective_neighbors))
    for cell, clone in enumerate(labels):
        intra_shift[cell] = np.linalg.norm(local_state[cell] - centroids[clone])
        neighbor_distances = [
            np.linalg.norm(local_state[cell] - centroids[neighbor])
            for neighbor in neighbor_clones.get(clone, ())
        ]
        if neighbor_distances:
            inter_affinity[cell] = intra_shift[cell] - min(neighbor_distances)
    local_field_score = intra_shift + float(inter_weight) * np.maximum(inter_affinity, 0.0)
    supported_score = local_field_score * local_support
    row_sum = np.asarray(cell_graph.sum(axis=1)).ravel()
    normalized_cell_graph = sparse.diags(
        np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    ).dot(cell_graph).tocsr()
    neighbor_score = normalized_cell_graph.dot(supported_score)
    score = (
        (1.0 - float(smoothing_strength)) * supported_score
        + float(smoothing_strength) * neighbor_score
    )
    adata.obs["state_variation"] = score
    if store_diagnostics:
        adata.obs["evofate_local_intra_shift"] = intra_shift
        adata.obs["evofate_local_inter_affinity"] = inter_affinity
        adata.obs["evofate_local_support"] = local_support
        adata.obsm["X_state_variation"] = np.asarray(
            [local_state[cell] - centroids[clone] for cell, clone in enumerate(labels)],
            dtype=np.float32,
        )
    if int(min_anchor_labels) < 5:
        raise ValueError("min_anchor_labels must be at least 5.")
    if not 0.0 <= float(lineage_propagation_alpha) <= 1.0:
        raise ValueError("lineage_propagation_alpha must be between 0 and 1.")
    if int(lineage_propagation_iter) < 1:
        raise ValueError("lineage_propagation_iter must be positive.")
    if (
        float(min_lineage_support) < 0.0
        or float(min_lineage_margin) < 0.0
        or float(min_outward_lineage_support) < 0.0
    ):
        raise ValueError("Lineage support, margin, and outward-support thresholds must be nonnegative.")

    # Build lineage paths, then assign branch identity locally while walking
    # backward through the directed clone graph.  This deliberately avoids
    # global terminal-clone diffusion.
    lineage_graph = _evofate_analysis_graph(adata)
    lineage_paths = _evofate_lineage_paths(lineage_graph, set(labels))
    if not lineage_paths:
        raise ValueError("No observed root-to-terminal lineage is available for single-cell assignment.")
    lineage_names = [f"lineage_{index + 1}" for index in range(len(lineage_paths))]
    n_lineages = len(lineage_names)
    observed = set(labels)
    # Lineage identity is inferred by graph label propagation. Connectivity
    # weights determine the strength of neighboring lineage evidence.
    label_graph = cell_graph.tocsr()
    row_sum = np.asarray(label_graph.sum(axis=1)).ravel()
    P = sparse.diags(
        np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > eps)
    ).dot(label_graph).tocsr()
    clone_to_cells = {clone: np.flatnonzero(labels == clone) for clone in np.unique(labels)}
    clone_to_lineages = {clone: [] for clone in observed}
    for lineage_index, path in enumerate(lineage_paths):
        for clone in path:
            clone_to_lineages.setdefault(clone, []).append(lineage_index)

    support_matrix = np.zeros((adata.n_obs, n_lineages), dtype=float)
    final_lineage = np.full(adata.n_obs, -1, dtype=int)
    membership = np.zeros((adata.n_obs, n_lineages), dtype=bool)
    clone_assignment_state = {}
    resolution_order = []
    topo = [node for node in nx.topological_sort(lineage_graph) if node in observed]
    for clone in reversed(topo):
        cells = clone_to_cells.get(clone, np.array([], dtype=int))
        candidates = clone_to_lineages.get(clone, [])
        if cells.size == 0 or not candidates:
            continue
        observed_children = [
            child for child in lineage_graph.successors(clone)
            if child in observed and child in clone_to_cells
        ]
        if any(child not in clone_assignment_state for child in observed_children):
            # Reverse topological order resolves most dependencies; retaining
            # this check makes the parent-splitting dependency explicit.
            continue
        outer_neighbors = np.unique(cell_graph[cells].indices)
        outer_neighbors = outer_neighbors[labels[outer_neighbors] != clone]
        if len(candidates) == 1:
            final_lineage[cells] = candidates[0]
            membership[cells, candidates[0]] = True
            support_matrix[cells, candidates[0]] = 1.0
            clone_assignment_state[clone] = {
                "state": "assigned",
                "lineages": [int(candidates[0])],
            }
            resolution_order.append(clone)
            continue

        # Collapse memberships inherited from one shared child into a single
        # composite candidate.  Thus {L1, L2} versus L3 is treated as two
        # branch choices, not three independent choices.
        grouped_lineages = []
        grouped_anchors = []
        grouped_children = []
        group_keys = {}
        for lineage_index in candidates:
            path = lineage_paths[lineage_index]
            position = path.index(clone)
            if position + 1 >= len(path):
                key = (lineage_index,)
                child_cells = np.array([], dtype=int)
            else:
                child = path[position + 1]
                child_state = clone_assignment_state.get(child, {})
                if child_state.get("state") == "shared":
                    key = tuple(sorted(int(value) for value in child_state.get("lineages", [lineage_index])))
                else:
                    key = (int(lineage_index),)
                child_cells = clone_to_cells.get(child, np.array([], dtype=int))
            # Use every directly connected outer cell carrying this lineage
            # identity, including cells from a shared child clone.  The
            # topology-constrained membership matrix prevents unrelated
            # branches from becoming evidence.
            anchors = outer_neighbors[
                membership[outer_neighbors, lineage_index]
            ]
            if key not in group_keys:
                group_keys[key] = len(grouped_lineages)
                grouped_lineages.append(key)
                grouped_anchors.append(anchors)
                grouped_children.append(child_cells)
            else:
                index = group_keys[key]
                grouped_anchors[index] = np.unique(np.concatenate([grouped_anchors[index], anchors]))
                grouped_children[index] = np.unique(np.concatenate([grouped_children[index], child_cells]))

        n_groups = len(grouped_lineages)
        scores = np.zeros((cells.size, n_groups), dtype=float)
        # First label current-clone cells from their direct outer connections.
        # These labels are fixed seeds; propagation below is restricted to the
        # current clone only.
        direct_scores = np.zeros((cells.size, n_groups), dtype=float)
        current_to_outer = label_graph[cells][:, outer_neighbors].tocsr()
        for group_index, group in enumerate(grouped_lineages):
            outer_mask = np.any(
                membership[outer_neighbors][:, np.asarray(group, dtype=int)],
                axis=1,
            )
            if np.any(outer_mask):
                direct_scores[:, group_index] = np.asarray(
                    current_to_outer[:, outer_mask].sum(axis=1)
                ).ravel()
        fixed_seed = np.zeros_like(direct_scores)
        direct_valid = direct_scores.sum(axis=1) > eps
        direct_index = np.argmax(direct_scores, axis=1)
        fixed_seed[np.arange(cells.size)[direct_valid], direct_index[direct_valid]] = 1.0
        # Prepare a minimum number of fixed labels for every group before
        # propagation. Directly connected cells are preferred; missing labels
        # are filled by nearest X_integrated cells to the group's child cells.
        for group_index, child_cells in enumerate(grouped_children):
            selected = np.flatnonzero(fixed_seed[:, group_index] > 0)
            if selected.size >= int(min_anchor_labels):
                continue
            if child_cells.size:
                distances = np.linalg.norm(
                    state[cells, None, :] - state[child_cells][None, :, :],
                    axis=2,
                ).min(axis=1)
            else:
                distances = np.full(cells.size, np.inf)
            order = np.argsort(distances, kind="stable")
            needed = int(min_anchor_labels) - selected.size
            # Do not reuse a cell fixed to another group unless no unlabelled
            # cells remain; the anchor check enforces minimum support.
            unlabelled = np.flatnonzero(fixed_seed.sum(axis=1) == 0)
            if unlabelled.size:
                available = unlabelled[np.argsort(distances[unlabelled], kind="stable")]
            else:
                available = np.array([], dtype=int)
            fallback = available[:needed]
            fixed_seed[fallback, group_index] = 1.0
        fixed_valid = fixed_seed.sum(axis=1) > 0
        # Count labels per group directly.  ``argmax`` would undercount a
        # group whenever a prepared seed accidentally carried >1 labels.
        anchor_counts = np.sum(fixed_seed > 0, axis=0)
        effective = (
            n_groups >= 2
            and np.all(anchor_counts >= int(min_anchor_labels))
        )
        if effective:
            within_graph = label_graph[cells][:, cells].tocsr()
            within_sum = np.asarray(within_graph.sum(axis=1)).ravel()
            within_P = sparse.diags(
                np.divide(1.0, within_sum, out=np.zeros_like(within_sum), where=within_sum > eps)
            ).dot(within_graph).tocsr()
            propagated = fixed_seed.copy()
            fixed_rows = fixed_valid
            for _ in range(int(lineage_propagation_iter)):
                propagated = within_P.dot(propagated)
                propagated[fixed_rows] = fixed_seed[fixed_rows]
            scores = propagated
        else:
            scores = direct_scores
        # Normalize the propagated evidence across candidate lineages.
        totals = scores.sum(axis=1)
        valid = totals > eps
        scores[valid] /= totals[valid, None]
        for group_index, group in enumerate(grouped_lineages):
            for lineage_index in group:
                support_matrix[cells, lineage_index] = scores[:, group_index]
        if not effective:
            # Stop upstream tracing when at most one downstream identity remains;
            # shared cells retain the compatible descendant memberships.
            membership[np.ix_(cells, np.asarray(candidates, dtype=int))] = True
            support_matrix[np.ix_(cells, np.asarray(candidates, dtype=int))] = 1.0 / len(candidates)
            clone_assignment_state[clone] = {
                "state": "shared",
                "lineages": [int(candidate) for candidate in candidates],
                "anchor_counts": anchor_counts.astype(int).tolist(),
                "n_groups": int(n_groups),
            }
            resolution_order.append(clone)
            continue
        effective_indices = np.arange(n_groups, dtype=int)
        best_local = np.argmax(scores[:, effective_indices], axis=1)
        best_group = effective_indices[best_local]
        for group_index in np.unique(best_group):
            selected_cells = cells[best_group == group_index]
            group = np.asarray(grouped_lineages[group_index], dtype=int)
            membership[np.ix_(selected_cells, group)] = True
            if group.size == 1:
                final_lineage[selected_cells] = group[0]
        clone_assignment_state[clone] = {
            "state": "split",
            "lineages": [
                int(lineage)
                for index in effective_indices
                for lineage in grouped_lineages[index]
            ],
            "groups": [list(map(int, grouped_lineages[index])) for index in effective_indices],
            "anchor_counts": anchor_counts.astype(int).tolist(),
            "n_groups": int(n_groups),
        }
        resolution_order.append(clone)
        # Cells in composite/shared groups retain multi-lineage memberships;
        # their group identity informs the next parent assignment.

    # Convert lineage evidence into one exhaustive hard assignment per cell;
    # weak and tied evidence uses the deterministic first argmax lineage.
    best_lineage = np.argmax(support_matrix, axis=1)
    cells = np.arange(adata.n_obs)
    membership[:, :] = False
    membership[cells, best_lineage] = True
    # Soft compatibility is the local neighborhood composition of the hard
    # labels, not the intermediate propagation score.  Weighted graph edges
    # contribute proportionally to the neighboring lineage composition.
    composition_graph = label_graph + sparse.eye(adata.n_obs, format="csr")
    composition_sum = np.asarray(composition_graph.sum(axis=1)).ravel()
    composition_P = sparse.diags(
        np.divide(1.0, composition_sum, out=np.zeros_like(composition_sum), where=composition_sum > eps)
    ).dot(composition_graph).tocsr()
    support_matrix = np.asarray(
        composition_P.dot(membership.astype(float)),
        dtype=float,
    )
    final_lineage[:] = best_lineage

    # Build coarse clone time from path order, using TMB values when a stable
    # clone-to-TMB mapping is available.
    tmb_by_clone = {}
    tmb_values = np.asarray(adata.uns.get("TMB_clone", []), dtype=float).ravel()
    tmb_labels = np.asarray(adata.uns.get("consensus_clone_labels", []), dtype=str).ravel()
    if tmb_values.size and tmb_values.size == tmb_labels.size:
        tmb_by_clone = {str(label): float(value) for label, value in zip(tmb_labels, tmb_values)}
    progression = np.full((adata.n_obs, n_lineages), np.nan, dtype=float)
    shared_progression = np.full(adata.n_obs, np.nan, dtype=float)
    progression_out_fraction = np.full((adata.n_obs, n_lineages), np.nan, dtype=float)
    progression_target_time = np.full((adata.n_obs, n_lineages), np.nan, dtype=float)
    progression_own_support = np.full((adata.n_obs, n_lineages), np.nan, dtype=float)
    progression_out_support = np.full((adata.n_obs, n_lineages), np.nan, dtype=float)
    clone_time_by_lineage = {}
    for column, path_value in enumerate(lineage_paths):
        path = [str(clone) for clone in path_value]
        path_values = np.asarray([tmb_by_clone.get(clone, np.nan) for clone in path])
        if use_tmb_time and np.isfinite(path_values).all() and np.ptp(path_values) > eps:
            distances = np.r_[0.0, np.cumsum(np.abs(np.diff(path_values)))]
            clone_time = distances / max(distances[-1], eps)
        else:
            edge_distances = []
            for source, target in zip(path[:-1], path[1:]):
                edge_data = lineage_graph.get_edge_data(source, target, {}) or {}
                edge_value = edge_data.get("length", edge_data.get("weight", 1.0))
                try:
                    edge_value = float(edge_value)
                except (TypeError, ValueError):
                    edge_value = 1.0
                edge_distances.append(edge_value if np.isfinite(edge_value) and edge_value > 0 else 1.0)
            if edge_distances:
                distances = np.r_[0.0, np.cumsum(edge_distances)]
                clone_time = distances / max(distances[-1], eps)
            else:
                clone_time = np.linspace(0.0, 1.0, len(path))
        clone_time_by_name = dict(zip(path, clone_time))
        clone_time_by_lineage[column] = clone_time_by_name
        lineage_cells = np.flatnonzero(
            (support_matrix[:, column] > 0)
            & np.isfinite(support_matrix[:, column])
            & np.isin(labels, path)
        )
        if lineage_cells.size == 0:
            continue
        adjacent_clones = {}
        for index, clone in enumerate(path):
            adjacent = []
            if index > 0:
                adjacent.append(path[index - 1])
            if index + 1 < len(path):
                adjacent.append(path[index + 1])
            adjacent_clones[clone] = adjacent
        predicted, diagnostics = _connectivity_gated_clone_time(
            label_graph, labels, lineage_cells, clone_time_by_name, adjacent_clones
        )
        if predicted.size:
            progression[lineage_cells, column] = predicted
            progression_out_fraction[lineage_cells, column] = diagnostics["out_fraction"]
            progression_target_time[lineage_cells, column] = diagnostics["target_time"]
            progression_own_support[lineage_cells, column] = diagnostics["own_support"]
            progression_out_support[lineage_cells, column] = diagnostics["out_support"]

        if enforce_clone_monotonicity:
            clone_names = []
            clone_medians = []
            clone_targets = []
            for clone in path:
                clone_cells = np.flatnonzero(
                    (labels == clone)
                    & np.isfinite(progression[:, column])
                )
                if clone_cells.size:
                    clone_names.append(clone)
                    clone_medians.append(float(np.median(progression[clone_cells, column])))
                    clone_targets.append(float(clone_time_by_name[clone]))
            if len(clone_medians) >= 2:
                order = np.argsort(clone_targets, kind="stable")
                corrected = IsotonicRegression(
                    increasing=True,
                    out_of_bounds="clip",
                ).fit_transform(
                    np.asarray(clone_targets)[order],
                    np.asarray(clone_medians)[order],
                )
                for index, clone in enumerate(np.asarray(clone_names)[order]):
                    clone_cells = (labels == clone) & np.isfinite(progression[:, column])
                    progression[clone_cells, column] += corrected[index] - np.asarray(clone_medians)[order][index]
        finite_lineage = np.isfinite(progression[:, column])
        if np.any(finite_lineage):
            progression[finite_lineage, column] = np.clip(
                progression[finite_lineage, column], 0.0, 1.0
            )
    for cell, clone in enumerate(labels):
        values = [mapping[clone] for mapping in clone_time_by_lineage.values() if clone in mapping]
        if values:
            shared_progression[cell] = float(np.mean(values))
    assigned_mask = final_lineage >= 0
    final_support = support_matrix.max(axis=1)
    lineage_margin = np.ones(adata.n_obs, dtype=float)
    adata.obsm["evofate_lineage_support"] = support_matrix.astype(np.float32)
    adata.obsm["evofate_lineage_progression"] = progression.astype(np.float32)
    adata.obsm["evofate_lineage_membership"] = membership.astype(bool)
    adata.obsm["evofate_progression_out_fraction"] = progression_out_fraction.astype(np.float32)
    adata.obsm["evofate_progression_target_time"] = progression_target_time.astype(np.float32)
    adata.obsm["evofate_progression_own_support"] = progression_own_support.astype(np.float32)
    adata.obsm["evofate_progression_out_support"] = progression_out_support.astype(np.float32)
    adata.uns["evofate_lineage_assignment"] = {
        "clone_states": clone_assignment_state,
        "resolution_order": resolution_order,
        "connectivity_key": connectivity_key,
        "n_neighbors": int(n_neighbors),
        "min_anchor_labels": int(min_anchor_labels),
        "lineage_propagation_alpha": float(lineage_propagation_alpha),
        "lineage_propagation_iter": int(lineage_propagation_iter),
        "progression_method": "connectivity_gated_clone_time_interpolation",
        "use_tmb_time": bool(use_tmb_time),
        "enforce_clone_monotonicity": bool(enforce_clone_monotonicity),
        "min_outward_lineage_support": float(min_outward_lineage_support),
    }
    adata.obs["evofate_lineage"] = pd.Categorical(
        [lineage_names[index] if index >= 0 else "shared" for index in final_lineage],
        categories=lineage_names + ["shared"],
    )
    adata.obs["evofate_lineage_support"] = final_support.astype(np.float32)
    adata.obs["evofate_lineage_margin"] = lineage_margin.astype(np.float32)
    assigned_progression = np.full(adata.n_obs, np.nan, dtype=float)
    assigned_mask = final_lineage >= 0
    assigned_progression[assigned_mask] = progression[
        np.flatnonzero(assigned_mask), final_lineage[assigned_mask]
    ]
    shared_mask = ~assigned_mask
    if np.any(shared_mask):
        shared_progression_values = progression[shared_mask]
        finite_shared = np.isfinite(shared_progression_values)
        shared_count = finite_shared.sum(axis=1)
        shared_sum = np.nansum(shared_progression_values, axis=1)
        assigned_progression[shared_mask] = np.divide(
            shared_sum,
            shared_count,
            out=np.zeros_like(shared_sum),
            where=shared_count > 0,
        )
    adata.obs["evofate_progression"] = assigned_progression.astype(np.float32)
    lineage_cell_order = {}
    lineage_cell_order_names = {}
    for column, lineage in enumerate(lineage_names):
        ordered = np.flatnonzero(
            membership[:, column] & np.isfinite(progression[:, column])
        )
        ordered = ordered[
            np.argsort(progression[ordered, column], kind="stable")
        ]
        lineage_cell_order[lineage] = ordered.astype(int).tolist()
        lineage_cell_order_names[lineage] = [
            str(adata.obs_names[index]) for index in ordered
        ]
    adata.uns["evofate_lineage_order"] = lineage_cell_order
    adata.uns["evofate_lineage_order_names"] = lineage_cell_order_names
    _evofate_trajectory(
        adata,
        embedding_key=embedding_key,
        clone_key=clone_key,
        lineage=lineage_graph,
        membership_matrix=membership,
        progression_matrix=progression,
        copy=False,
    )
    return adata if copy else None

def _evofate_trajectory(
    adata,
    embedding_key: str = "X_evofate_projection",
    clone_key: str = "ordered_clone",
    lineage=None,
    n_control_points: int = 5,
    n_render_points: int = 100,
    min_cells_per_control: int = 10,
    min_lineage_support: float = 0.20,
    n_curve_points: int | None = None,
    max_curve_iter: int = 20,
    membership_matrix=None,
    progression_matrix=None,
    smooth_control_points: bool = True,
    control_smooth_strength: float = 0.25,
    control_smooth_iter: int = 1,
    control_point_min_separation: float = 1e-6,
    key_added: str = "evofate_trajectory",
    copy: bool = False,
):
    """Fit lineage-guided coarse principal trajectories through the EvoFATE map."""
    if copy:
        adata = adata.copy()
    for key, location_name, location in (
        (embedding_key, "obsm", adata.obsm),
        (clone_key, "obs", adata.obs),
    ):
        if key not in location:
            raise KeyError(f"`adata.{location_name}['{key}']` is missing.")
    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)[:, :2]
    labels = adata.obs[clone_key].astype(str).to_numpy()
    if embedding.shape != (adata.n_obs, 2) or not np.isfinite(embedding).all():
        raise ValueError("EvoFATE embedding must be finite with shape (n_obs, 2).")
    if n_curve_points is not None:
        n_control_points = int(n_curve_points)
    if int(n_control_points) < 2 or int(n_render_points) < 2 or int(min_cells_per_control) < 1:
        raise ValueError("Control points, render points, and minimum support must be positive.")
    if not 0.0 <= float(control_smooth_strength) <= 1.0:
        raise ValueError("`control_smooth_strength` must be between 0 and 1.")
    if int(control_smooth_iter) < 0:
        raise ValueError("`control_smooth_iter` must be nonnegative.")
    graph = _evofate_analysis_graph(adata, lineage=lineage)
    observed = set(labels)
    if not observed.issubset(set(graph.nodes)):
        raise ValueError("The lineage graph is missing observed clone labels.")
    clone_ids = np.unique(labels)
    clone_centers = {clone: np.median(embedding[labels == clone], axis=0) for clone in clone_ids}
    lineage_paths = _evofate_lineage_paths(graph, observed)
    if not lineage_paths:
        raise ValueError("No observed root-to-terminal lineage contains at least two clones.")

    def clone_controls(cell_indices, path):
        """Return one robust control point for each observed lineage clone.

        Curve complexity is therefore determined by the clonal structure,
        not by arbitrary equal-cell progression bins.  ``progression_values``
        is used only to provide a stable order for observed clones; the
        directed ``path`` remains the primary ordering constraint.
        """
        clone_position = {
            str(clone): index for index, clone in enumerate(path)
        }
        controls = []
        control_clones = []
        for clone in path:
            clone_mask = labels[cell_indices] == str(clone)
            selected = cell_indices[clone_mask]
            if selected.size == 0:
                continue
            # Median coordinates keep individual cells from pulling the
            # clone-level flow point into an unsupported excursion.
            controls.append(np.median(embedding[selected], axis=0))
            control_clones.append(str(clone))
        if len(controls) < 2:
            return np.empty((0, 2), dtype=float), control_clones
        return np.asarray(controls, dtype=float), control_clones

    def prepare_controls(controls):
        if controls.shape[0] < 2:
            return controls
        scale = max(float(np.linalg.norm(controls[-1] - controls[0])), 1e-12)
        keep = [0]
        for index in range(1, controls.shape[0]):
            if np.linalg.norm(controls[index] - controls[keep[-1]]) >= scale * float(control_point_min_separation):
                keep.append(index)
        controls = controls[np.asarray(keep, dtype=int)]
        if smooth_control_points and controls.shape[0] > 2:
            for _ in range(int(control_smooth_iter)):
                updated = controls.copy()
                updated[1:-1] = (
                    (1.0 - float(control_smooth_strength)) * controls[1:-1]
                    + 0.5 * float(control_smooth_strength) * (controls[:-2] + controls[2:])
                )
                controls = updated
        return controls

    def render_controls(controls):
        separation = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        scale = max(float(np.linalg.norm(controls[-1] - controls[0])), 1e-12)
        keep = np.r_[True, separation > scale * 1e-6]
        controls = controls[keep]
        if controls.shape[0] < 2:
            return np.repeat(controls[:1], int(n_render_points), axis=0), controls
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(controls, axis=0), axis=1))]
        arc /= max(arc[-1], 1e-12)
        render_t = np.linspace(0.0, 1.0, int(n_render_points))
        curve = np.column_stack([
            PchipInterpolator(arc, controls[:, axis])(render_t)
            for axis in range(2)
        ])
        return curve, controls

    records = {}
    if membership_matrix is None or progression_matrix is None:
        raise ValueError("Flow fitting requires fixed lineage support and progression matrices.")
    membership_matrix = np.asarray(membership_matrix, dtype=bool)
    progression_matrix = np.asarray(progression_matrix, dtype=float)
    if membership_matrix.shape != progression_matrix.shape or membership_matrix.shape != (adata.n_obs, len(lineage_paths)):
        raise ValueError("Lineage support/progression matrices do not match the trajectory paths.")
    for lineage_index, path in enumerate(lineage_paths):
        cell_indices = np.flatnonzero(
            membership_matrix[:, lineage_index]
            & np.isfinite(progression_matrix[:, lineage_index])
        )
        if cell_indices.size < 2:
            continue
        order = np.argsort(progression_matrix[cell_indices, lineage_index], kind="stable")
        cell_indices = cell_indices[order]
        points = embedding[cell_indices]
        controls, _ = clone_controls(cell_indices, path)
        controls = prepare_controls(controls)
        if controls.shape[0] < 2:
            continue
        curve, controls = render_controls(controls)
        nearest_distance = NearestNeighbors(n_neighbors=1).fit(points).kneighbors(curve, return_distance=True)[0].ravel()
        directions = np.diff(curve, axis=0)
        direction_norm = np.linalg.norm(directions, axis=1)
        cosines = np.sum(directions[:-1] * directions[1:], axis=1) / (
            direction_norm[:-1] * direction_norm[1:] + 1e-12
        )
        records[f"lineage_{lineage_index + 1}"] = {
            "clone_path": path,
            "curve_xy": curve.astype(np.float32),
            "control_points": controls.astype(np.float32),
            "cell_indices": cell_indices.astype(int),
            "max_curve_to_cell_distance": float(np.max(nearest_distance)),
            "median_curve_to_cell_distance": float(np.median(nearest_distance)),
            "backtracking_fraction": float(np.mean(cosines < 0.0)) if cosines.size else 0.0,
        }
    adata.uns[key_added] = {
        "lineages": records,
        "params": {
            "n_control_points": int(n_control_points),
            "n_render_points": int(n_render_points),
            "min_cells_per_control": int(min_cells_per_control),
            "control_structure": "one_median_control_point_per_observed_clone",
            "smooth_control_points": bool(smooth_control_points),
            "control_smooth_strength": float(control_smooth_strength),
            "control_smooth_iter": int(control_smooth_iter),
            "control_point_min_separation": float(control_point_min_separation),
        },
    }
    # Preserve lineage membership and progression while storing rendered geometry.
    lineage_names = [f"lineage_{index + 1}" for index in range(len(lineage_paths))]
    adata.uns["evofate_lineages"] = {
        "names": lineage_names,
        "clone_paths": {
            name: list(path)
            for name, path in zip(lineage_names, lineage_paths)
        },
    }
    return adata if copy else None

def _require_lineage_progression(adata):
    if "evofate_lineage_support" not in adata.obsm or "evofate_lineage_progression" not in adata.obsm:
        raise ValueError("Lineage progression is not available. Run tl.cal_single_cell_evofate() first.")
    support = np.asarray(adata.obsm["evofate_lineage_support"], dtype=float)
    progression = np.asarray(adata.obsm["evofate_lineage_progression"], dtype=float)
    membership = np.asarray(
        adata.obsm.get("evofate_lineage_membership", support > 0),
        dtype=bool,
    )
    names = list(adata.uns.get("evofate_lineages", {}).get("names", []))
    if support.ndim != 2 or progression.shape != support.shape or membership.shape != support.shape or support.shape[0] != adata.n_obs:
        raise ValueError("Stored lineage support and progression matrices are invalid.")
    if not names or len(names) != support.shape[1]:
        raise ValueError("Stored lineage names do not match support/progression matrices.")
    included = (support > 0) & np.isfinite(progression)
    finite_progression = progression[included]
    if np.any(~np.isfinite(finite_progression)) or np.any((finite_progression < 0) | (finite_progression > 1)):
        raise ValueError("Lineage progression must be finite and in [0, 1] for supported cells.")
    # Downstream progression analysis uses soft lineage support as weights;
    # hard membership remains a structural annotation only.
    return support, progression, names

def _bh_fdr(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.nan_to_num(values, nan=1.0))
    adjusted = np.full(values.size, np.nan)
    running = 1.0
    for rank in range(values.size - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * values.size / max(rank + 1, 1))
        adjusted[index] = min(running, 1.0)
    return adjusted

def _spline_gam(values, progression_values, n_knots, grid, smoothing=0.1, weights=None):
    """Fit a compact spline GAM approximation and its null model."""
    n_unique = np.unique(progression_values).size
    if values.size < 2 or n_unique < 2:
        fitted = np.full(values.shape, np.mean(values) if values.size else 0.0)
        grid_fitted = np.full(grid.shape, np.mean(values) if values.size else 0.0)
        return fitted, grid_fitted, 0.0, 0.0, 1
    degree = min(3, max(1, n_unique - 1))
    n_knots = min(
        max(int(n_knots), 2),
        max(2, n_unique - degree + 1),
    )
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots="uniform",
        include_bias=True,
    )
    basis = transformer.fit_transform(progression_values[:, None])
    grid_basis = transformer.transform(grid[:, None])
    weights = np.ones(values.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    sqrt_weights = np.sqrt(np.maximum(weights, 0.0))
    weighted_basis = basis * sqrt_weights[:, None]
    weighted_values = values * sqrt_weights
    penalty = np.eye(basis.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    if float(smoothing) == 0.0:
        coefficients = np.linalg.lstsq(weighted_basis, weighted_values, rcond=None)[0]
    else:
        coefficients = np.linalg.solve(
            weighted_basis.T @ weighted_basis + float(smoothing) * penalty,
            weighted_basis.T @ weighted_values,
        )
    fitted = basis @ coefficients
    grid_fitted = grid_basis @ coefficients
    null_value = np.average(values, weights=weights)
    rss_full = float(np.sum(weights * (values - fitted) ** 2))
    rss_null = float(np.sum(weights * (values - null_value) ** 2))
    df_full = max(1, basis.shape[1])
    return fitted, grid_fitted, rss_full, rss_null, df_full

def cal_progression_features(
    adata_snv,
    adata_exp,
    layer=None,
    use_raw=False,
    n_knots=2,
    smoothing=0.0,
    min_effect=0.0,
    fdr_threshold=0.05,
    feature_columns=None,
    min_support=0.0,
    result_key="progression_genes",
):
    """Identify globally lineage/progression-structured genes with GAMs.

    ``adata_exp`` must be a pandas DataFrame with cells as rows and genes as
    columns. Values must already be library-size-normalized and ``log1p``
    transformed; this function does not normalize or transform them. Regression
    uses single-cell progression observations. The DataFrame index must match
    ``adata_snv.obs_names`` exactly.
    """
    lineage_support, progression, lineages = _require_lineage_progression(adata_snv)
    if float(smoothing) < 0.0:
        raise ValueError("smoothing must be nonnegative.")
    if not adata_snv.obs_names.is_unique:
        raise ValueError("adata_snv.obs_names must be unique.")
    if not isinstance(adata_exp, pd.DataFrame):
        raise TypeError(
            "adata_exp must be a pandas DataFrame with cells as rows and "
            "genes as columns."
        )
    if layer is not None or use_raw:
        raise ValueError("layer= and use_raw= are not applicable to DataFrame expression input.")
    if feature_columns is not None:
        raise ValueError(
            "feature_columns is not applicable to DataFrame expression input; "
            "gene names are taken directly from the DataFrame index."
        )
    if not adata_exp.index.is_unique or not adata_exp.columns.is_unique:
        raise ValueError("The expression DataFrame index and columns must be unique.")
    if not adata_snv.obs_names.equals(pd.Index(adata_exp.index)):
        raise ValueError(
            "adata_exp.index must contain exactly the same cells as "
            "adata_snv.obs_names, in the same order."
        )
    expression = np.asarray(adata_exp.to_numpy(dtype=float), dtype=float)
    feature_names = np.asarray(adata_exp.columns, dtype=str)
    grid = np.linspace(0.0, 1.0, 100)
    lineage_results = {}
    per_gene = {str(feature): {"lineages": {}, "statistics": []} for feature in feature_names}

    for lineage_index, lineage in enumerate(lineages):
        mask = (
            (lineage_support[:, lineage_index] > 0)
            & np.isfinite(lineage_support[:, lineage_index])
            & np.isfinite(progression[:, lineage_index])
        )
        ordered_cells = np.flatnonzero(mask)
        ordered_cells = ordered_cells[
            np.argsort(progression[ordered_cells, lineage_index], kind="stable")
        ]
        record = {
            "ordered_cells": ordered_cells,
            "progression": progression[ordered_cells, lineage_index].astype(np.float32),
            "fitted_values": {},
            "feature_stats": {},
        }
        base_cells = np.flatnonzero(mask)
        t_all = progression[base_cells, lineage_index]
        w_all = np.maximum(lineage_support[base_cells, lineage_index], 0.0)
        if base_cells.size < 2 or np.unique(t_all).size < 2:
            lineage_results[lineage] = record
            continue

        def store_feature(feature_index, cells, y, fitted, grid_values, statistic, pvalue):
            feature = str(feature_names[feature_index])
            model_values = y
            grid_output = grid_values
            fitted_output = fitted
            record["fitted_values"][feature] = {
                "cell_indices": np.asarray(cells, dtype=int),
                "values": np.asarray(fitted_output, dtype=np.float32),
                "observed_values": np.asarray(y, dtype=np.float32),
                "progression_grid": grid.astype(np.float32),
                "grid_values": np.asarray(grid_output, dtype=np.float32),
            }
            early = float(np.mean(grid_output[grid <= 0.15]))
            late = float(np.mean(grid_output[grid >= 0.85]))
            effect = float(np.quantile(grid_output, 0.90) - np.quantile(grid_output, 0.10))
            derivative = np.gradient(grid_output, grid)
            peak = int(np.argmax(grid_output))
            max_change = int(np.argmax(np.abs(derivative)))
            endpoint_delta = float(grid_output[-1] - grid_output[0])
            threshold = 0.15 * max(abs(effect), 1e-12)
            if endpoint_delta > threshold:
                pattern = "increasing"
            elif endpoint_delta < -threshold:
                pattern = "decreasing"
            elif peak < 0.35 * grid.size and grid_output[0] > grid_output[-1] + threshold:
                pattern = "early-high"
            elif peak > 0.65 * grid.size and grid_output[-1] > grid_output[0] + threshold:
                pattern = "late-high"
            elif peak not in {0, grid.size - 1}:
                pattern = "transient"
            else:
                pattern = "complex"
            stability = float(np.clip(
                1.0
                - np.mean(np.abs(model_values - fitted))
                / (np.std(model_values) + np.finfo(float).eps),
                0.0,
                1.0,
            ))
            per_gene[feature]["lineages"][lineage] = {
                "effect_size": effect,
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "early_expression": early,
                "late_expression": late,
                "peak_progression": float(grid[peak]),
                "max_change_progression": float(grid[max_change]),
                "pattern": pattern,
                "stability": stability,
            }
            record["feature_stats"][feature] = {
                "effect_size": effect,
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            }
            per_gene[feature]["statistics"].append((float(statistic), float(pvalue)))

        finite = np.isfinite(expression[base_cells])
        fully_observed = finite.all(axis=0)

        # Fit all fully observed genes in one batched single-cell
        # least-squares solve.
        if np.any(fully_observed):
            model_t = t_all
            model_y = expression[base_cells][:, fully_observed]
            n_unique = np.unique(model_t).size
            degree = min(3, max(1, n_unique - 1))
            effective_knots = min(
                max(int(n_knots), 2),
                max(2, n_unique - degree + 1),
            )
            transformer = SplineTransformer(
                n_knots=effective_knots,
                degree=degree,
                knots="uniform",
                include_bias=True,
            )
            basis = transformer.fit_transform(model_t[:, None])
            grid_basis = transformer.transform(grid[:, None])
            sqrt_weights = np.sqrt(np.maximum(w_all, 0.0))
            weighted_basis = basis * sqrt_weights[:, None]
            weighted_model_y = model_y * sqrt_weights[:, None]
            penalty = np.eye(basis.shape[1], dtype=float)
            penalty[0, 0] = 0.0
            if float(smoothing) == 0.0:
                coefficients = np.linalg.lstsq(weighted_basis, weighted_model_y, rcond=None)[0]
            else:
                coefficients = np.linalg.solve(
                    weighted_basis.T @ weighted_basis + float(smoothing) * penalty,
                    weighted_basis.T @ weighted_model_y,
                )
            fitted_matrix = basis @ coefficients
            grid_matrix = grid_basis @ coefficients
            fitted_original = transformer.transform(t_all[:, None]) @ coefficients
            null_values = np.average(model_y, axis=0, weights=w_all)
            rss_full = np.sum(w_all[:, None] * (model_y - fitted_matrix) ** 2, axis=0)
            rss_null = np.sum(w_all[:, None] * (model_y - null_values[None, :]) ** 2, axis=0)
            df_full = max(1, basis.shape[1])
            df1 = max(1, df_full - 1)
            df2 = max(1, base_cells.size - df_full)
            statistic = np.maximum(0.0, (rss_null - rss_full) / df1) / np.maximum(
                rss_full / df2, 1e-12
            )
            pvalues = f_distribution.sf(statistic, df1, df2)
            feature_indices = np.flatnonzero(fully_observed)
            for matrix_index, feature_index in enumerate(feature_indices):
                store_feature(
                    feature_index,
                    base_cells,
                    expression[base_cells, feature_index],
                    fitted_original[:, matrix_index],
                    grid_matrix[:, matrix_index],
                    statistic[matrix_index],
                    pvalues[matrix_index],
                )

        # Handle genes with missing values using the per-feature solver.
        fallback_indices = np.flatnonzero(~fully_observed)
        for feature_index in fallback_indices:
            valid_cells = base_cells[finite[:, feature_index]]
            t = progression[valid_cells, lineage_index]
            y = expression[valid_cells, feature_index]
            fitted, grid_values, rss_full, rss_null, df_full = _spline_gam(
                y,
                t,
                n_knots,
                grid,
                smoothing,
                w_all[finite[:, feature_index]],
            )
            df1 = max(1, df_full - 1)
            df2 = max(1, valid_cells.size - df_full)
            statistic = max(0.0, (rss_null - rss_full) / df1) / max(rss_full / df2, 1e-12)
            store_feature(
                feature_index,
                valid_cells,
                y,
                np.interp(
                    progression[valid_cells, lineage_index],
                    grid,
                    grid_values,
                ),
                grid_values,
                statistic,
                f_distribution.sf(statistic, df1, df2),
            )
        lineage_results[lineage] = record

    rows = []
    for feature in map(str, feature_names):
        info = per_gene[feature]
        modeled = info["lineages"]
        if not modeled:
            continue
        pvalues = np.asarray([item["pvalue"] for item in modeled.values()], dtype=float)
        statistics = np.asarray([item["statistic"] for item in modeled.values()], dtype=float)
        effects = np.asarray([item["effect_size"] for item in modeled.values()], dtype=float)
        # Combine only lineage-specific progression tests. Lineage baseline
        # differences are context/control, not the biological target.
        global_statistic = float(-2.0 * np.sum(np.log(np.clip(pvalues, 1e-300, 1.0))))
        global_pvalue = float(chi2.sf(global_statistic, 2 * len(pvalues)))
        effect = float(np.quantile(
            np.concatenate([
                lineage_results[lineage]["fitted_values"][feature]["grid_values"]
                for lineage in modeled
            ]),
            0.90,
        ) - np.quantile(
            np.concatenate([
                lineage_results[lineage]["fitted_values"][feature]["grid_values"]
                for lineage in modeled
            ]),
            0.10,
        ))
        association_strength = float(
            np.mean(statistics) * max(effect, 0.0)
        )
        strongest = max(modeled.values(), key=lambda item: item["effect_size"])
        rows.append({
            "gene": feature,
            "statistic": global_statistic,
            "pvalue": global_pvalue,
            "effect_size": effect,
            "association_strength": association_strength,
            "lineage_effect_strength": float(np.std(effects)),
            "progression_strength": float(np.mean(effects)),
            "early_expression": float(strongest["early_expression"]),
            "late_expression": float(strongest["late_expression"]),
            "peak_progression": float(strongest["peak_progression"]),
            "max_change_progression": float(strongest["max_change_progression"]),
            "pattern": strongest["pattern"],
            "stability": float(np.mean([item["stability"] for item in modeled.values()])),
        })
    result_table = pd.DataFrame(rows)
    if not result_table.empty:
        result_table["fdr"] = _bh_fdr(result_table["pvalue"].to_numpy())
        result_table["significant"] = (
            (result_table["fdr"] <= float(fdr_threshold))
            & (result_table["effect_size"] >= float(min_effect))
        )
        result_table["importance"] = (
            result_table["association_strength"]
            * np.maximum(result_table["effect_size"], 0.0)
            * np.maximum(result_table["stability"], 1e-6)
        )
        result_table = result_table.sort_values(
            ["importance", "association_strength", "effect_size"],
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        result_table["rank"] = np.arange(1, len(result_table) + 1)

    # Store compact observed values and ranking statistics for plotting.
    for lineage, record in lineage_results.items():
        fitted_records = record.pop("fitted_values", {})
        record.pop("progression", None)
        compact_features = list(fitted_records)
        n_cells = len(record["ordered_cells"])
        observed_matrix = np.full(
            (len(compact_features), n_cells), np.nan, dtype=np.float32
        )
        cell_positions = {
            int(cell): position
            for position, cell in enumerate(np.asarray(record["ordered_cells"], dtype=int))
        }
        for row, feature in enumerate(compact_features):
            item = fitted_records[feature]
            positions = [cell_positions.get(int(cell)) for cell in np.asarray(item["cell_indices"], dtype=int)]
            valid_positions = [index for index, position in enumerate(positions) if position is not None]
            target_positions = [positions[index] for index in valid_positions]
            if target_positions:
                observed_matrix[row, target_positions] = np.asarray(item["observed_values"], dtype=np.float32)[valid_positions]
        record["feature_names"] = np.asarray(compact_features, dtype=str)
        record["observed_matrix"] = observed_matrix
        record["feature_stats"] = pd.DataFrame(
            [
                [
                    record["feature_stats"].get(feature, {}).get("effect_size", np.nan),
                    record["feature_stats"].get(feature, {}).get("statistic", np.nan),
                    record["feature_stats"].get(feature, {}).get("pvalue", np.nan),
                ]
                for feature in compact_features
            ],
            index=pd.Index(compact_features, name="gene"),
            columns=["effect_size", "statistic", "pvalue"],
            dtype=np.float32,
        )
        if not record["feature_stats"].empty:
            record["feature_stats"]["fdr"] = _bh_fdr(
                record["feature_stats"]["pvalue"].to_numpy(dtype=float)
            ).astype(np.float32)

    metadata = {}
    adata_snv.uns[result_key] = {
        "results": result_table,
        "lineages": lineage_results,
        "params": {
            "n_knots": int(n_knots),
            "smoothing": float(smoothing),
            "min_effect": float(min_effect),
            "fdr_threshold": float(fdr_threshold),
            "min_support": float(min_support),
            "layer": layer,
            "use_raw": bool(use_raw),
            "model": "spline_gam",
            "global_test": "Fisher_combined_lineage_progression_tests",
        },
        "feature_metadata": metadata,
    }
    return adata_snv


def select_progression_features(
    adata_snv,
    n_features=50,
    genes=None,
    significant_only=False,
    fdr_threshold=None,
    min_effect=None,
    pvalue_threshold=None,
    include_global_rank=True,
    result_key="progression_genes",
):
    """Select and save important progression features without plotting."""
    if result_key not in adata_snv.uns:
        raise KeyError(
            f"`adata.uns['{result_key}']` is missing. "
            "Run tl.cal_progression_features() first."
        )
    result = adata_snv.uns[result_key]
    table = result.get("results", pd.DataFrame())
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    available = set(table["gene"].astype(str)) if "gene" in table else set()
    if genes is not None:
        selected = [str(gene) for gene in genes if str(gene) in available]
    else:
        selected = []
        use_fdr_filter = significant_only or fdr_threshold is not None
        fdr_cutoff = (
            float(result.get("params", {}).get("fdr_threshold", 0.05))
            if fdr_threshold is None
            else float(fdr_threshold)
        )
        lineages = list(adata_snv.uns["evofate_lineages"]["names"])
        for lineage in lineages:
            record = result.get("lineages", {}).get(lineage, {})
            names = np.asarray(record.get("feature_names", []), dtype=str)
            stats = record.get("feature_stats", pd.DataFrame())
            if not isinstance(stats, pd.DataFrame):
                stats = pd.DataFrame(stats, index=names)
            candidates = [
                gene for gene in names
                if gene in available and gene in stats.index.astype(str)
            ]
            eligible = []
            for gene in candidates:
                row = stats.loc[gene]
                if use_fdr_filter and float(row.get("fdr", 1.0)) > fdr_cutoff:
                    continue
                if min_effect is not None and float(row.get("effect_size", 0.0)) < float(min_effect):
                    continue
                if pvalue_threshold is not None and float(row.get("pvalue", 1.0)) > float(pvalue_threshold):
                    continue
                eligible.append(gene)
            eligible.sort(key=lambda gene: (-float(stats.loc[gene, "statistic"]), gene))
            selected.extend(eligible[:int(n_features)])
        if include_global_rank:
            ranked = table.sort_values("rank", kind="stable") if "rank" in table.columns else table
            global_count = 0
            for _, row in ranked.iterrows():
                if use_fdr_filter and float(row.get("fdr", 1.0)) > fdr_cutoff:
                    continue
                if min_effect is not None and float(row.get("effect_size", 0.0)) < float(min_effect):
                    continue
                if pvalue_threshold is not None and float(row.get("pvalue", 1.0)) > float(pvalue_threshold):
                    continue
                selected.append(str(row["gene"]))
                global_count += 1
                if global_count >= int(n_features):
                    break
        selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("No progression features are available for selection.")
    result["important_features"] = [str(gene) for gene in selected]
    return result["important_features"]


from ._clones import define_clones
from ._consensus import cal_consensus_profile
from ._connectivities import cal_genetic_connectivities
from ._lineage import cal_clone_connectivity, cal_tree_layout
from ._integration import cal_evofate_embedding
from ._projection import cal_linear_projection, cal_lineage_guided_projection

__all__ = [
    "cal_genetic_connectivities", "define_clones", "cal_consensus_profile",
    "cal_clone_connectivity", "cal_tree_layout", "cal_evofate_embedding",
    "cal_linear_projection", "cal_lineage_guided_projection",
    "cal_clonal_evofate", "cal_single_cell_evofate",
    "cal_progression_features", "select_progression_features",
]
