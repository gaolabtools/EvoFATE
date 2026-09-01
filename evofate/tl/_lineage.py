"""Clone lineage connectivity helpers for EvoFATE."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags, issparse
from scipy.stats import rankdata, spearmanr

from ._consensus import infer_edge_direction, score_clone_direction
from ._genetic_utils import _as_string_array, _get_uns_graph, _set_uns_graph

if TYPE_CHECKING:
    from anndata import AnnData




def cal_clone_connectivity(
    adata_mut: AnnData,
    label_key: str = "clone",
    connectivity_key: str = "genetic_lineage_connectivity",
    mutation_order_method: str = "average",
    mutation_order_metric: str = "hamming",
    prediction_branch_protection: float = 1.0,
    prediction_mode: Literal["local", "global"] = "global",
    prediction_protected_error: float = 0.1,
    prediction_transition_penalty: float = 0.25,
    undirected_weight: float = 0.1,
    directed_weight: float = 1.0,
    initial_parent_neighbors: int = 3,
    parent_neighbor_step: int = 1,
    minimum_forward_proportion: float = 0.30,
    expanded_minimum_forward_proportion: float = 0.50,
    direction_contrast_weight: float = 0.25,
    parent_score_temperature: float = 0.10,
) -> None:
    """
    Calculate clone connectivity and Edmonds-defined lineage directions.

        Hard clone statuses rank possible parents by similarity. The parent
        neighborhood expands only when no nearby parent passes the directional
        threshold. The artificial all-WT normal participates in the same search.
    The original undirected graph remains available in ``Clone_connectivity``.
    TMB is calculated as the number of predicted MT calls in
    `consensus_profile_predict`.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with `.uns['consensus_profile']`.
    label_key : str, default='clone'
        Column in `.obs` containing clone labels used to assign
        predicted-TMB-ordered `ordered_clone` cell labels after lineage connectivity
        is calculated.
    connectivity_key : str, default='genetic_lineage_connectivity'
        Retained for API compatibility. It is not used by lineage inference.
    prediction_branch_protection : float, default=1.0
        Protects a clone consensus from being overturned solely because the
        clone has many children. Its flip cost is multiplied by
        ``1 + prediction_branch_protection * outdegree`` in the global
        prediction step.
    prediction_mode : {"local", "global"}, default="global"
        Global minimum-cut prediction is used by default. It allows
        confidence-supported WT-to-MT acquisition while enforcing irreversible
        MT-to-WT prevention. ``"local"`` remains available for comparison.
    prediction_protected_error : float, default=0.1
        In local mode this is retained for compatibility. In global mode,
        MT calls at or below this error rate are hard-protected from flipping.
    prediction_transition_penalty : float, default=0.25
        Per-mutation penalty for an otherwise allowed WT-to-MT transition
        along a directed lineage edge. This discourages unnecessary gains
        without forbidding well-supported acquisitions.
    undirected_weight : float, default=0.1
        Weight stored for precomputed undirected clone-connectivity edges.
    directed_weight : float, default=1.0
        Weight stored for directed lineage edges.
    mutation_order_method : str, default='average'
        Linkage method used to order filtered mutation columns after
        `consensus_profile_predict` is calculated.
    mutation_order_metric : str, default='hamming'
        Distance metric used to order filtered mutation columns after
        `consensus_profile_predict` is calculated.

    Returns
    -------
    None
        Modifies `adata_mut` in place:
        - `.uns['Clone_connectivity']`: Serialized undirected clone graph
        - `.uns['Lineage_tree']`: Serialized directed lineage tree
        - `.uns['consensus_profile']`: Consensus profile sorted by predicted TMB
          and prediction-based mutation order
        - `.uns['consensus_profile_predict']`: Predicted trinary consensus
          profile sorted by predicted TMB and prediction-based mutation order
        - `.uns['Clone_connectivity_similarity']`: Profile-based candidate
          similarities in lineage-node order
        - `.uns['TMB_clone']`: Predicted MT counts in lineage-node order
        - `.obs['ordered_clone']`: Cell labels matching lineage-node indices
        - `.obs['TMB']`: Per-cell TMB from the ordered clone profile
    """
    for obsolete_key in (
        "clone_child_error_rate",
        "clone_child_contrast",
        "clone_child_error_rate_labels",
    ):
        adata_mut.uns.pop(obsolete_key, None)

    if not np.isfinite(undirected_weight) or undirected_weight < 0:
        raise ValueError("`undirected_weight` must be finite and nonnegative.")
    if not np.isfinite(directed_weight) or directed_weight <= 0:
        raise ValueError("`directed_weight` must be finite and positive.")
    if int(initial_parent_neighbors) <= 0 or int(parent_neighbor_step) <= 0:
        raise ValueError("Parent neighborhood sizes and step must be positive.")
    for name, value in (
        ("initial_parent_neighbors", initial_parent_neighbors),
        ("parent_neighbor_step", parent_neighbor_step),
        ("direction_contrast_weight", direction_contrast_weight),
        ("expanded_minimum_forward_proportion", expanded_minimum_forward_proportion),
        ("minimum_forward_proportion", minimum_forward_proportion),
        ("parent_score_temperature", parent_score_temperature),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not 0.0 <= minimum_forward_proportion <= 1.0:
        raise ValueError("minimum_forward_proportion must be in [0, 1].")
    if not 0.0 <= expanded_minimum_forward_proportion <= 1.0:
        raise ValueError("expanded_minimum_forward_proportion must be in [0, 1].")
    if parent_score_temperature <= 0.0:
        raise ValueError("parent_score_temperature must be positive.")
    if not np.isfinite(prediction_transition_penalty) or prediction_transition_penalty < 0:
        raise ValueError(
            "`prediction_transition_penalty` must be finite and nonnegative."
        )
    consensus_data_available = (
        "consensus_profile" in adata_mut.uns
        and "filtered_mutations" in adata_mut.uns
        and (
            isinstance(adata_mut.uns.get("clone_consensus"), dict)
            and all(
                key in adata_mut.uns["clone_consensus"]
                for key in (
                    "status",
                    "posterior_p_mt",
                    "posterior_p_wt",
                    "posterior_reliability",
                )
            )
        )
    )
    if not consensus_data_available:
        raise KeyError(
            "Consensus data are unavailable. Run "
            "`cal_consensus_profile(adata_mut)` before "
            "`cal_clone_connectivity(adata_mut)`."
        )
    if connectivity_key not in adata_mut.obsp:
        raise KeyError(
            f"`adata_mut.obsp['{connectivity_key}']` is missing. Run "
            "`cal_genetic_connectivities(adata_mut)` first."
        )
    if label_key not in adata_mut.obs:
        raise KeyError(
            f"`adata_mut.obs['{label_key}']` is missing. "
            "Run `define_clones(adata_mut)` first or provide a valid "
            "`label_key`."
        )
    input_digest = hashlib.sha256()
    input_digest.update(np.asarray(
        adata_mut.obs[label_key].astype(str), dtype="U"
    ).tobytes())
    for key in (
        "consensus_profile",
        "consensus_posterior_p_mt",
        "consensus_posterior_p_wt",
        "consensus_posterior_reliability",
    ):
        value = adata_mut.uns.get(key)
        if value is not None:
            array = np.asarray(value)
            input_digest.update(str(array.shape).encode())
            input_digest.update(np.ascontiguousarray(array).tobytes())
    connectivity_input = adata_mut.obsp[connectivity_key]
    if issparse(connectivity_input):
        connectivity_input = connectivity_input.tocsr()
        for value in (
            connectivity_input.data,
            connectivity_input.indices,
            connectivity_input.indptr,
        ):
            input_digest.update(np.ascontiguousarray(value).tobytes())
    else:
        input_digest.update(np.ascontiguousarray(
            np.asarray(connectivity_input)
        ).tobytes())
    input_signature = input_digest.hexdigest()
    cached_params = adata_mut.uns.get("clone_connectivity_direction_params", {})
    if not isinstance(cached_params, dict):
        cached_params = {}
    if (
        adata_mut.uns.get("clone_connectivity_input_signature") == input_signature
        and "Clone_connectivity" in adata_mut.uns
        and "Lineage_tree" in adata_mut.uns
        and "clone_connectivity_graph" in adata_mut.uns
        and "clone_similarity_graph" in adata_mut.uns
        and adata_mut.uns.get("clone_connectivity_direction_method", "edmonds")
        == "edmonds"
        and cached_params.get("method")
        == "adaptive_status_similarity_parent_search"
        and int(cached_params.get(
            "initial_parent_neighbors", 3
        )) == int(initial_parent_neighbors)
        and int(cached_params.get(
            "parent_neighbor_step", 1
        )) == int(parent_neighbor_step)
        and float(cached_params.get(
            "minimum_forward_proportion", 0.30
        )) == float(minimum_forward_proportion)
        and float(cached_params.get(
            "direction_contrast_weight", 0.25
        )) == float(direction_contrast_weight)
        and float(cached_params.get(
            "expanded_minimum_forward_proportion", 0.50
        )) == float(expanded_minimum_forward_proportion)
        and float(cached_params.get(
            "parent_score_temperature", 0.10
        )) == float(parent_score_temperature)
        and cached_params.get(
            "prediction_mode", "global"
        ) == str(prediction_mode)
        and float(cached_params.get(
            "prediction_transition_penalty", 0.25
        )) == float(prediction_transition_penalty)
    ):
        # Edge weights parameterize layout geometry; the clone graph remains
        # unchanged when these values are updated.
        cached_clone_graph = _get_uns_graph(
            adata_mut, "Clone_connectivity", directed=False, required=True
        )
        cached_lineage_graph = _get_uns_graph(
            adata_mut, "Lineage_tree", directed=True, required=True
        )
        adata_mut.uns["Clone_connectivity_edge_weights"] = pd.DataFrame(
            [
                {"source": int(source), "target": int(target), "weight": float(undirected_weight)}
                for source, target in cached_clone_graph.get_edgelist()
                if int(source) != int(target)
            ],
            columns=["source", "target", "weight"],
        )
        adata_mut.uns["Lineage_tree_edge_weights"] = pd.DataFrame(
            [
                {"source": int(source), "target": int(target), "weight": float(directed_weight)}
                for source, target in cached_lineage_graph.get_edgelist()
                if int(source) != int(target)
            ],
            columns=["source", "target", "weight"],
        )
        adata_mut.uns["clone_connectivity_weight_params"] = {
            "undirected_weight": float(undirected_weight),
            "directed_weight": float(directed_weight),
        }
        return None
    consensus_snv = np.asarray(adata_mut.uns["consensus_profile"], dtype=float)
    (
        direction_snv,
        direction_p_mt,
        direction_p_wt,
        direction_reliability,
    ) = _get_full_consensus_for_lineage(
        adata_mut=adata_mut,
        consensus_snv=consensus_snv,
        label_key=label_key,
    )
    if direction_snv.shape[1] != consensus_snv.shape[1]:
        lineage_mutation_names = np.asarray(adata_mut.var_names, dtype=str)
        consensus_snv = direction_snv.copy()
        adata_mut.uns["filtered_mutations"] = _as_string_array(lineage_mutation_names)
    else:
        lineage_mutation_names = np.asarray(
            adata_mut.uns.get("filtered_mutations", []), dtype=str
        )
        # The active consensus profile is the raw status source for
        # prediction. The full-width lineage cache may contain different
        # inferred statuses even when its selected mutation count happens to
        # match the active profile shape.
        direction_snv = consensus_snv.astype(int, copy=True)
    n_nodes = consensus_snv.shape[0]
    direction_prediction_error = np.clip(1.0 - direction_reliability, 0.0, 1.0)

    clone_connectivity_raw, clone_connectivity_normalized, clone_sizes = (
        _aggregate_clone_connectivity_density(
            adata_mut=adata_mut,
            label_key=label_key,
            connectivity_key=connectivity_key,
            n_nodes=n_nodes,
        )
    )

    profile_similarity = _cal_profile_similarity_from_status(
        consensus_status=direction_snv,
    )
    biological_order = np.arange(1, n_nodes, dtype=int)
    _, g, profile_graph_info = _build_clone_graph_from_profile_neighbors(
        profile_similarity=profile_similarity,
        n_profile_neighbors=int(initial_parent_neighbors),
    )
    biological_order = np.arange(1, n_nodes, dtype=int)
    biological_clone_labels = np.asarray(
        adata_mut.uns.get("consensus_clone_labels", biological_order),
        dtype=object,
    ).astype(str)
    selected_edges = []
    for source, target in g.get_edgelist():
        source, target = int(source), int(target)
        if source == 0 or target == 0:
            continue
        edge = tuple(sorted((source, target)))
        selected_edges.append({
            "clone_a": source,
            "clone_b": target,
            "profile_similarity": float(profile_similarity[source, target]),
            "selected_by_profile_neighborhood": True,
        })
    adata_mut.uns["clone_connectivity_graph"] = {
        "clone_order": biological_clone_labels.tolist(),
        "clone_sizes": clone_sizes[biological_order],
        "clone_connectivity_raw": clone_connectivity_raw[
            np.ix_(biological_order, biological_order)
        ],
        "clone_connectivity_normalized": clone_connectivity_normalized[
            np.ix_(biological_order, biological_order)
        ],
        "selected_edges": pd.DataFrame(selected_edges),
        "profile_neighborhood_size": int(initial_parent_neighbors),
        "profile_neighborhood_edges": profile_graph_info["selected_by_neighbor_union"],
        "profile_similarity": profile_similarity[
            np.ix_(biological_order, biological_order)
        ],
        "candidate_graph": "full_directed_biological_parent_matrix",
    }
    # Preserve the public alias while selecting candidates by profile similarity.
    adata_mut.uns["clone_similarity_graph"] = adata_mut.uns[
        "clone_connectivity_graph"
    ]
    g2, edmonds_metadata = _build_adaptive_status_lineage(
        consensus_status=direction_snv,
        posterior_reliability=direction_reliability,
        initial_parent_neighbors=int(initial_parent_neighbors),
        parent_neighbor_step=int(parent_neighbor_step),
        clone_connectivity_normalized=clone_connectivity_normalized,
        direction_contrast_weight=float(direction_contrast_weight),
        minimum_forward_proportion=float(minimum_forward_proportion),
        expanded_minimum_forward_proportion=float(expanded_minimum_forward_proportion),
        parent_score_temperature=float(parent_score_temperature),
    )
    adata_mut.uns["Lineage_tree_edmonds_candidates"] = edmonds_metadata

    # Use the same retained biological candidate pairs for similarity-edge
    # plotting that were actually scored and passed to Edmonds. Artificial
    # normal edges are directional root candidates, not similarity edges.
    retained_similarity_pairs: dict[tuple[int, int], dict[str, float]] = {}
    retained_candidates = edmonds_metadata.get("retained_candidates")
    if isinstance(retained_candidates, pd.DataFrame):
        retained_records = retained_candidates.to_dict("records")
    else:
        retained_records = retained_candidates or []
    for record in retained_records:
        parent_value = record.get("parent")
        child_value = record.get("child")
        if str(parent_value) == "__normal__":
            continue
        try:
            parent = int(parent_value)
            child = int(child_value)
        except (TypeError, ValueError):
            continue
        if parent == child or parent <= 0 or child <= 0:
            continue
        pair = tuple(sorted((parent, child)))
        retained_similarity_pairs[pair] = {
            "status_similarity": float(record.get("similarity", 0.0)),
        }
    g = ig.Graph(
        n=n_nodes,
        edges=sorted(retained_similarity_pairs),
        directed=False,
    )
    if g.ecount():
        g.es["status_similarity"] = [
            retained_similarity_pairs[edge]["status_similarity"]
            for edge in g.get_edgelist()
        ]
    adata_mut.uns["clone_connectivity_graph"]["selected_edges"] = pd.DataFrame(
        [
            {
                "clone_a": source,
                "clone_b": target,
                "status_similarity": values["status_similarity"],
                "selected_by_edmonds": True,
            }
            for (source, target), values in sorted(retained_similarity_pairs.items())
        ]
    )
    adata_mut.uns["clone_connectivity_graph"][
        "candidate_graph"
    ] = "retained_edmonds_biological_candidates"
    g.simplify()
    g2.simplify()
    consensus_profile_predict_original = _predict_consensus_profile_from_lineage(
        direction_snv,
        g2,
        error_rate=direction_prediction_error,
        branch_protection=prediction_branch_protection,
        mode=prediction_mode,
        protected_error=prediction_protected_error,
        transition_penalty=prediction_transition_penalty,
    )
    tmb_clone = _cal_tmb_from_predict(consensus_profile_predict_original)

    (
        consensus_snv,
        g,
        g2,
        tmb_clone,
        labels_snv,
        profile_labels,
    ) = _order_lineage_by_tmb(
        adata_mut=adata_mut,
        consensus_snv=consensus_snv,
        clone_graph=g,
        lineage_tree=g2,
        tmb_clone=tmb_clone,
        label_key=label_key,
    )
    # Preserve non-lineage relationships, such as sibling/shortcut edges, in
    # the undirected clone graph. Only selected staged edges are directed in
    # ``Lineage_tree``.
    g = _merge_lineage_edges_into_undirected_graph(g, g2)
    direction_p_mt = direction_p_mt[profile_labels["order"]]
    direction_p_wt = direction_p_wt[profile_labels["order"]]
    direction_reliability = direction_reliability[profile_labels["order"]]
    direction_prediction_error = np.clip(1.0 - direction_reliability, 0.0, 1.0)
    direction_snv = direction_snv[profile_labels["order"]]
    consensus_profile_predict = _predict_consensus_profile_from_lineage(
        direction_snv,
        g2,
        error_rate=direction_prediction_error,
        branch_protection=prediction_branch_protection,
        mode=prediction_mode,
        protected_error=prediction_protected_error,
        transition_penalty=prediction_transition_penalty,
    )
    tmb_clone = _cal_tmb_from_predict(consensus_profile_predict)
    mutation_order = _get_clustermap_column_order(
        consensus_profile_predict,
        method=mutation_order_method,
        metric=mutation_order_metric,
    )
    consensus_snv = consensus_snv[:, mutation_order]
    consensus_profile_predict = consensus_profile_predict[:, mutation_order]
    active_mutation_names = np.asarray(adata_mut.uns["filtered_mutations"]).astype(str)
    if active_mutation_names.shape[0] != mutation_order.shape[0]:
        if consensus_snv.shape[1] == adata_mut.n_vars:
            active_mutation_names = np.asarray(adata_mut.var_names).astype(str)
        else:
            raise ValueError(
                "Filtered mutation names do not match the active consensus "
                "profile columns before lineage ordering. Re-run "
                "cal_consensus_profile."
            )
    adata_mut.uns["filtered_mutations"] = _as_string_array(
        active_mutation_names[mutation_order]
    )
    adata_mut.uns["consensus_profile"] = consensus_snv
    adata_mut.uns.pop("consensus_profile_high_confidence", None)
    adata_mut.uns["consensus_profile_predict"] = consensus_profile_predict
    adata_mut.uns.pop("consensus_profile_prob", None)
    adata_mut.uns.pop("Lineage_stage", None)
    adata_mut.uns["TMB_clone"] = tmb_clone
    adata_mut.uns["clone_connectivity_input_signature"] = input_signature
    adata_mut.uns["consensus_profile_labels"] = _as_string_array(
        profile_labels["labels"]
    )
    adata_mut.uns["consensus_clone_labels"] = _as_string_array(
        profile_labels["labels"][1:]
    )
    adata_mut.obs["ordered_clone"] = labels_snv.astype(str)
    adata_mut.obs["TMB"] = tmb_clone[labels_snv]
    _set_uns_graph(adata_mut, "Clone_connectivity", g)
    # Store layout weights explicitly so serialized igraph edge lists retain
    # the precomputed undirected clone-graph weights.
    adata_mut.uns["Clone_connectivity_edge_weights"] = pd.DataFrame(
        [
            {
                "source": int(source),
                "target": int(target),
                "weight": float(undirected_weight),
            }
            for source, target in g.get_edgelist()
            if int(source) != int(target)
        ],
        columns=["source", "target", "weight"],
    )
    adata_mut.uns["Lineage_tree_edge_weights"] = pd.DataFrame(
        [
            {
                "source": int(source),
                "target": int(target),
                "weight": float(directed_weight),
            }
            for source, target in g2.get_edgelist()
            if int(source) != int(target)
        ],
        columns=["source", "target", "weight"],
    )
    adata_mut.uns["clone_connectivity_weight_params"] = {
        "undirected_weight": float(undirected_weight),
        "directed_weight": float(directed_weight),
    }
    adata_mut.uns["clone_connectivity_direction_method"] = "edmonds"
    adata_mut.uns["clone_connectivity_direction_params"] = {
        "method": "adaptive_status_similarity_parent_search",
        "initial_parent_neighbors": int(initial_parent_neighbors),
        "parent_neighbor_step": int(parent_neighbor_step),
        "minimum_forward_proportion": float(minimum_forward_proportion),
        "expanded_minimum_forward_proportion": float(expanded_minimum_forward_proportion),
        "direction_contrast_weight": float(direction_contrast_weight),
        "parent_score_temperature": float(parent_score_temperature),
        "prediction_mode": str(prediction_mode),
        "prediction_transition_penalty": float(prediction_transition_penalty),
    }
    lineage_inference = adata_mut.uns.get("lineage_inference", {})
    if not isinstance(lineage_inference, dict):
        lineage_inference = {}
    lineage_inference["selected_roots"] = edmonds_metadata["selected_roots"]
    lineage_inference["n_normal_children"] = edmonds_metadata[
        "n_normal_children"
    ]
    lineage_inference["candidate_root_scores"] = edmonds_metadata[
        "candidate_root_scores"
    ]
    lineage_inference["candidate_root_results"] = edmonds_metadata[
        "candidate_root_results"
    ]
    lineage_inference["expected_mt_burden"] = edmonds_metadata[
        "expected_mt_burden"
    ]
    lineage_inference["selected_total_mt_to_wt_loss"] = edmonds_metadata[
        "selected_total_mt_to_wt_loss"
    ]
    lineage_inference["selected_total_wt_to_mt_gain"] = edmonds_metadata[
        "selected_total_wt_to_mt_gain"
    ]
    lineage_inference["selected_total_edmonds_score"] = edmonds_metadata[
        "selected_total_edmonds_score"
    ]
    lineage_inference["selected_total_tree_cost"] = edmonds_metadata[
        "selected_total_tree_cost"
    ]
    lineage_inference["selected_normal_edge_cost"] = edmonds_metadata[
        "selected_normal_edge_cost"
    ]
    lineage_inference["selected_biological_tree_cost"] = edmonds_metadata[
        "selected_biological_tree_cost"
    ]
    lineage_inference["edge_score"] = {
        "method": "adaptive_status_similarity_parent_search",
        "initial_parent_neighbors": int(initial_parent_neighbors),
        "parent_neighbor_step": int(parent_neighbor_step),
        "minimum_forward_proportion": float(minimum_forward_proportion),
        "expanded_minimum_forward_proportion": float(expanded_minimum_forward_proportion),
        "direction_contrast_weight": float(direction_contrast_weight),
        "parent_score_temperature": float(parent_score_temperature),
    }
    adata_mut.uns["lineage_inference"] = lineage_inference
    adata_mut.uns["Clone_connectivity_similarity"] = profile_similarity[
        np.ix_(profile_labels["order"], profile_labels["order"])
    ]
    _set_uns_graph(adata_mut, "Lineage_tree", g2)




def _aggregate_clone_connectivity_density(
    adata_mut: AnnData,
    label_key: str,
    connectivity_key: str,
    n_nodes: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate cell connectivity as size-normalized clone-pair density."""
    n_cells = adata_mut.n_obs
    graph = (
        adata_mut.obsp[connectivity_key].tocsr()
        if issparse(adata_mut.obsp[connectivity_key])
        else csr_matrix(adata_mut.obsp[connectivity_key])
    )
    if graph.shape != (n_cells, n_cells):
        raise ValueError("Cell connectivity must be square with one row per cell.")
    graph = graph.maximum(graph.T).tocoo()
    if graph.data.size and (
        not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0)
    ):
        raise ValueError("Cell connectivity weights must be finite and nonnegative.")
    clone_labels = np.asarray(
        adata_mut.uns.get("consensus_clone_labels", []), dtype=object
    )
    if clone_labels.shape[0] != n_nodes - 1:
        raise ValueError("Consensus clone labels do not match consensus rows.")
    label_to_node = {
        _normalize_label_key(label): index + 1
        for index, label in enumerate(clone_labels)
    }
    cell_to_node = np.asarray([
        label_to_node[_normalize_label_key(label)]
        for label in adata_mut.obs[label_key].to_numpy()
    ], dtype=int)
    sizes = np.bincount(cell_to_node, minlength=n_nodes).astype(float)
    raw = np.zeros((n_nodes, n_nodes), dtype=float)
    for row, col, value in zip(graph.row, graph.col, graph.data):
        source = cell_to_node[int(row)]
        target = cell_to_node[int(col)]
        if source != target and source > 0 and target > 0:
            raw[source, target] += float(value)
    raw = (raw + raw.T) * 0.5
    normalized = np.divide(
        raw,
        sizes[:, None] * sizes[None, :] + eps,
        out=np.zeros_like(raw),
        where=(sizes[:, None] > 0) & (sizes[None, :] > 0),
    )
    normalized = (normalized + normalized.T) * 0.5
    np.fill_diagonal(raw, 0.0)
    np.fill_diagonal(normalized, 0.0)
    raw[0, :] = raw[:, 0] = 0.0
    normalized[0, :] = normalized[:, 0] = 0.0
    return raw, normalized, sizes






def _cal_profile_similarity_from_status(
    consensus_status: np.ndarray,
) -> np.ndarray:
    """Calculate symmetric profile similarity from hard clone statuses."""
    status = np.asarray(consensus_status)
    if status.ndim != 2:
        raise ValueError("`consensus_status` must be a 2D matrix.")
    n_nodes, n_mutations = status.shape
    similarity = np.zeros((n_nodes, n_nodes), dtype=float)
    if n_mutations:
        similarity = np.mean(
            status[:, None, :] == status[None, :, :],
            axis=2,
            dtype=float,
        )
    np.fill_diagonal(similarity, 0.0)
    if n_nodes:
        similarity[0, :] = 0.0
        similarity[:, 0] = 0.0
    return np.clip(similarity, 0.0, 1.0)


def _build_clone_graph_from_profile_neighbors(
    profile_similarity: np.ndarray,
    n_profile_neighbors: int = 2,
) -> tuple[np.ndarray, ig.Graph, dict[str, object]]:
    """Select profile neighbors and minimally repair the candidate graph."""
    profile = np.asarray(profile_similarity, dtype=float)
    if profile.ndim != 2 or profile.shape[0] != profile.shape[1]:
        raise ValueError("Profile similarity must be a square matrix.")
    if int(n_profile_neighbors) < 0:
        raise ValueError("`n_profile_neighbors` must be nonnegative.")
    profile = np.clip(np.nan_to_num(profile, nan=0.0), 0.0, 1.0)
    profile = np.maximum(profile, profile.T)
    np.fill_diagonal(profile, 0.0)
    biological_nodes = list(range(1, profile.shape[0]))
    graph = nx.Graph()
    graph.add_nodes_from(biological_nodes)
    selected_by_neighbor_union: set[tuple[int, int]] = set()
    minimum_degree_edges: set[tuple[int, int]] = set()
    connectivity_repair_edges: set[tuple[int, int]] = set()

    for clone in biological_nodes:
        profile_neighbors = sorted(
            (other for other in biological_nodes if other != clone),
            key=lambda other: (-float(profile[clone, other]), int(other)),
        )[: min(int(n_profile_neighbors), max(len(biological_nodes) - 1, 0))]
        for other in profile_neighbors:
            edge = tuple(sorted((clone, other)))
            graph.add_edge(*edge)
            selected_by_neighbor_union.add(edge)

    for clone in biological_nodes:
        if graph.degree(clone) > 0 or len(biological_nodes) <= 1:
            continue
        candidates = [other for other in biological_nodes if other != clone]
        neighbor = max(
            candidates,
            key=lambda other: (float(profile[clone, other]), -int(other)),
        )
        edge = tuple(sorted((clone, neighbor)))
        graph.add_edge(*edge)
        minimum_degree_edges.add(edge)

    n_components_before_repair = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    while graph.number_of_nodes() and not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        component_index = {
            node: index for index, component in enumerate(components)
            for node in component
        }
        candidates = [
            (float(profile[first, second]), first, second)
            for index, first in enumerate(biological_nodes)
            for second in biological_nodes[index + 1:]
            if component_index[first] != component_index[second]
            and not graph.has_edge(first, second)
        ]
        if not candidates:
            raise RuntimeError("Unable to connect the biological clone candidate graph.")
        _, first, second = max(
            candidates,
            key=lambda item: (item[0], -int(item[1]), -int(item[2])),
        )
        edge = tuple(sorted((first, second)))
        graph.add_edge(*edge)
        connectivity_repair_edges.add(edge)

    n_components_after_repair = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    retained_edges = sorted(graph.edges())
    clone_graph = ig.Graph(n=profile.shape[0], edges=retained_edges, directed=False)
    clone_graph["minimum_degree_edges"] = sorted(minimum_degree_edges)
    clone_graph["repaired_edges"] = sorted(connectivity_repair_edges)
    clone_graph["n_components_before_repair"] = n_components_before_repair
    clone_graph["n_components_after_repair"] = n_components_after_repair
    clone_graph.es["profile_similarity"] = [
        float(profile[first, second]) for first, second in retained_edges
    ]
    clone_graph.es["selected_by_neighbor_union"] = [
        edge in selected_by_neighbor_union for edge in retained_edges
    ]
    clone_graph.es["added_for_minimum_degree"] = [
        edge in minimum_degree_edges for edge in retained_edges
    ]
    clone_graph.es["added_for_connectivity_repair"] = [
        edge in connectivity_repair_edges for edge in retained_edges
    ]
    return profile, clone_graph, {
        "n_components_before_repair": n_components_before_repair,
        "n_components_after_repair": n_components_after_repair,
        "selected_by_neighbor_union": sorted(selected_by_neighbor_union),
        "minimum_degree_edges": sorted(minimum_degree_edges),
        "connectivity_repair_edges": sorted(connectivity_repair_edges),
    }




















def _calculate_status_similarity_matrix(status_with_normal: np.ndarray) -> np.ndarray:
    """Return the full hard-status agreement matrix."""
    status = np.asarray(status_with_normal)
    if status.ndim != 2 or status.shape[1] == 0:
        raise ValueError("Status matrix must be 2D with at least one mutation.")
    similarity = np.mean(
        status[:, None, :] == status[None, :, :], axis=2, dtype=float
    )
    return np.clip((similarity + similarity.T) * 0.5, 0.0, 1.0)


def _calculate_direction_statistics(
    parent_status: np.ndarray,
    child_status: np.ndarray,
    *,
    parent_reliability: np.ndarray | None = None,
    child_reliability: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Calculate hard-status transitions with reliability-weighted direction."""
    parent = np.asarray(parent_status)
    child = np.asarray(child_status)
    if parent.shape != child.shape:
        raise ValueError("Parent and child status vectors must have matching shapes.")
    gain_mask = (parent == -1) & (child == 1)
    loss_mask = (parent == 1) & (child == -1)
    gain_count = float(np.sum(gain_mask))
    loss_count = float(np.sum(loss_mask))
    transition_count = gain_count + loss_count
    if parent_reliability is None:
        parent_rel = np.ones(parent.shape, dtype=float)
    else:
        parent_rel = np.asarray(parent_reliability, dtype=float)
    if child_reliability is None:
        child_rel = np.ones(child.shape, dtype=float)
    else:
        child_rel = np.asarray(child_reliability, dtype=float)
    if parent_rel.shape != parent.shape or child_rel.shape != child.shape:
        raise ValueError("Reliability vectors must match the status vectors.")
    pair_reliability = np.clip(
        np.sqrt(np.clip(parent_rel, 0.0, 1.0) * np.clip(child_rel, 0.0, 1.0)),
        0.0,
        1.0,
    )
    trusted_gain = float(np.sum(pair_reliability[gain_mask]))
    trusted_loss = float(np.sum(pair_reliability[loss_mask]))
    trusted_transition_mass = trusted_gain + trusted_loss
    forward_proportion = (
        trusted_gain / trusted_transition_mass
        if trusted_transition_mass > eps
        else 0.5
    )
    return {
        "gain": gain_count,
        "loss": loss_count,
        "transition_count": transition_count,
        "trusted_gain": trusted_gain,
        "trusted_loss": trusted_loss,
        "trusted_transition_mass": trusted_transition_mass,
        "forward_proportion": float(np.clip(forward_proportion, 0.0, 1.0)),
    }




def _normalize_retained_parent_scores(
    raw_scores: dict[int, float],
    *,
    parent_score_temperature: float = 0.10,
) -> dict[int, float]:
    """Normalize retained parent scores relative to the child's best parent."""
    if not raw_scores:
        raise ValueError("Every biological child must retain a parent.")
    best = max(raw_scores.values())
    return {
        parent: float(np.exp((score - best) / parent_score_temperature))
        for parent, score in raw_scores.items()
    }


def _build_adaptive_status_lineage(
    consensus_status: np.ndarray,
    posterior_reliability: np.ndarray,
    *,
    initial_parent_neighbors: int = 3,
    parent_neighbor_step: int = 1,
    minimum_forward_proportion: float = 0.30,
    expanded_minimum_forward_proportion: float = 0.50,
    clone_connectivity_normalized: np.ndarray,
    direction_contrast_weight: float = 0.25,
    parent_score_temperature: float = 0.10,
    eps: float = 1e-12,
) -> tuple[ig.Graph, dict[str, object]]:
    """Build Edmonds candidates from a symmetrized ranked neighborhood."""
    status = np.asarray(consensus_status)
    if status.ndim != 2 or status.shape[0] < 2 or status.shape[1] == 0:
        raise ValueError("Consensus status must contain biological clones and mutations.")
    if not np.all(np.isin(status, (-1, 1))):
        raise ValueError(
            "Lineage construction requires hard MT/WT statuses only; missing "
            "statuses must be resolved before lineage inference."
        )
    reliability = np.asarray(posterior_reliability, dtype=float)
    if reliability.shape != status.shape:
        raise ValueError("Posterior reliability must match consensus status shape.")
    if not np.all(np.isfinite(reliability)):
        raise ValueError("Posterior reliability must be finite.")
    reliability = np.clip(reliability, 0.0, 1.0)
    biological_status = status[1:]
    biological_reliability = reliability[1:]
    normal_status = np.full(status.shape[1], -1, dtype=status.dtype)
    normal_reliability = np.ones(status.shape[1], dtype=float)
    status_with_normal = np.vstack([normal_status, biological_status])
    reliability_with_normal = np.vstack([normal_reliability, biological_reliability])
    similarity = _calculate_status_similarity_matrix(status_with_normal)
    connectivity = np.asarray(clone_connectivity_normalized, dtype=float)
    if connectivity.shape != similarity.shape:
        raise ValueError("Clone connectivity and status similarity must have matching shapes.")
    connectivity = np.clip(np.nan_to_num(connectivity, nan=0.0), 0.0, None)
    connectivity[0, :] = 0.0
    connectivity[:, 0] = 0.0
    normal = "__normal__"
    biological_nodes = list(range(1, status_with_normal.shape[0]))
    graph = nx.DiGraph()
    graph.add_nodes_from([normal, *biological_nodes])
    all_pair_records: list[dict[str, object]] = []
    retained_records: list[dict[str, object]] = []
    child_diagnostics: list[dict[str, object]] = []
    normal_records_by_child: dict[int, dict[str, object]] = {}
    ranked_by_child: dict[int, list[int]] = {}
    selected_parents_by_child: dict[int, set[int]] = {}

    # Select the requested neighborhood first, then symmetrize it. Edmonds
    # receives both orientations of every retained biological pair, regardless
    # of which endpoint selected the pair in its local ranking.
    for child in biological_nodes:
        possible = [parent for parent in biological_nodes if parent != child]
        possible.append(0)
        connected_parents = sorted(
            (parent for parent in possible if connectivity[parent, child] > 0.0),
            key=lambda parent: (
                -float(connectivity[parent, child]),
                -float(similarity[parent, child]),
                int(parent),
            ),
        )
        unconnected_parents = sorted(
            (parent for parent in possible if connectivity[parent, child] <= 0.0),
            key=lambda parent: (-float(similarity[parent, child]), int(parent)),
        )
        ranked = connected_parents + unconnected_parents
        ranked_by_child[child] = ranked
        window_size = min(max(int(initial_parent_neighbors), 0), len(ranked))
        selected = set(ranked[:window_size])
        if not selected:
            selected.add(0)
        selected_parents_by_child[child] = selected

    for child, selected in selected_parents_by_child.items():
        for parent in sorted(selected):
            if parent == 0:
                continue
            selected_parents_by_child[parent].add(child)

    for child in biological_nodes:
        possible = [parent for parent in biological_nodes if parent != child]
        possible.append(0)
        connected_parents = sorted(
            (parent for parent in possible if connectivity[parent, child] > 0.0),
            key=lambda parent: (
                -float(connectivity[parent, child]),
                -float(similarity[parent, child]),
                int(parent),
            ),
        )
        unconnected_parents = sorted(
            (parent for parent in possible if connectivity[parent, child] <= 0.0),
            key=lambda parent: (-float(similarity[parent, child]), int(parent)),
        )
        ranked = connected_parents + unconnected_parents
        window_size = min(max(int(initial_parent_neighbors), 0), len(ranked))
        allowed = sorted(
            selected_parents_by_child[child],
            key=lambda parent: ranked.index(parent),
        )
        if not allowed:
            allowed = [0]
        stats_by_parent: dict[int, dict[str, float]] = {}
        normal_stats = _calculate_direction_statistics(
            status_with_normal[0],
            status_with_normal[child],
            parent_reliability=reliability_with_normal[0],
            child_reliability=reliability_with_normal[child],
            eps=eps,
        )
        normal_records_by_child[child] = {
            "parent": normal,
            "child": child,
            "similarity": float(similarity[0, child]),
            "distance": float(1.0 - similarity[0, child]),
            **normal_stats,
            "similarity_rank": ranked.index(0) + 1,
            "clone_connectivity": 0.0,
            "has_positive_connectivity": False,
            "ranking_source": "zero_connectivity_profile",
            "raw_parent_score": float(
                similarity[0, child]
                + direction_contrast_weight
                * (normal_stats["forward_proportion"] - 0.5)
            ),
            "edge_allowed": True,
            "direction_pass": True,
            "direction_threshold": None,
            "is_artificial_normal": True,
        }
        for parent in allowed:
            stats = _calculate_direction_statistics(
                status_with_normal[parent],
                status_with_normal[child],
                parent_reliability=reliability_with_normal[parent],
                child_reliability=reliability_with_normal[child],
                eps=eps,
            )
            stats_by_parent[parent] = stats
            raw_score = float(
                similarity[parent, child]
                + direction_contrast_weight
                * (stats["forward_proportion"] - 0.5)
            )
            direction_threshold = None
            all_pair_records.append({
                "parent": normal if parent == 0 else parent,
                "child": child,
                "similarity": float(similarity[parent, child]),
                "distance": float(1.0 - similarity[parent, child]),
                **stats,
                "similarity_rank": ranked.index(parent) + 1,
                "clone_connectivity": float(connectivity[parent, child]),
                "has_positive_connectivity": connectivity[parent, child] > 0.0,
                "ranking_source": (
                    "positive_connectivity"
                    if connectivity[parent, child] > 0.0
                    else "zero_connectivity_profile"
                ),
                "raw_parent_score": raw_score,
                "edge_allowed": True,
                "direction_pass": True,
                "direction_threshold": direction_threshold,
                "is_artificial_normal": parent == 0,
            })
        raw_scores = {
            parent: float(
                similarity[parent, child]
                + direction_contrast_weight
                * (stats_by_parent[parent]["forward_proportion"] - 0.5)
            )
            for parent in allowed
        }
        local_weights = _normalize_retained_parent_scores(
            raw_scores, parent_score_temperature=parent_score_temperature
        )
        best_parent = max(raw_scores, key=raw_scores.get)
        child_diagnostics.append({
            "child": child,
            "connected_parents_ranked": connected_parents,
            "unconnected_parents_ranked": [
                normal if parent == 0 else parent for parent in unconnected_parents
            ],
            "ranked_possible_parents": [normal if p == 0 else p for p in ranked],
            "initial_parent_neighbors": initial_parent_neighbors,
            "final_window_size": window_size,
            "search_expanded": False,
            "retained_parents": [normal if p == 0 else p for p in allowed],
        })
        for parent in allowed:
            stats = stats_by_parent[parent]
            record = {
                "parent": normal if parent == 0 else parent,
                "child": child,
                "similarity": float(similarity[parent, child]),
                "distance": float(1.0 - similarity[parent, child]),
                **stats,
                "similarity_rank": ranked.index(parent) + 1,
                "raw_parent_score": raw_scores[parent],
                "local_parent_weight": local_weights[parent],
                "direction_pass": True,
                "edge_weight": local_weights[parent],
                "weight": local_weights[parent],
                "was_local_best_parent": parent == best_parent,
                "is_artificial_normal": parent == 0,
                "edge_type": "artificial_normal" if parent == 0 else "biological",
            }
            retained_records.append(record)
            graph.add_edge(
                record["parent"], child,
                **{key: value for key, value in record.items()
                   if key not in {"parent", "child", "edge_type"}},
                edge_type=record["edge_type"],
            )

    from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
    try:
        arborescence = maximum_spanning_arborescence(
            graph, attr="weight", default=0.0, preserve_attrs=True
        )
    except nx.NetworkXException:
        # A locally valid parent window can still form closed biological
        # components. Preserve the retained biological candidates and anchor
        # each weak component with exactly one normal edge, choosing the
        # mutation-poor clone in that component while preserving the remaining
        # parent assignments.
        retained_before_repair = list(retained_records)
        biological_components = nx.Graph()
        biological_components.add_nodes_from(biological_nodes)
        for record in retained_before_repair:
            parent = record["parent"]
            child = int(record["child"])
            if parent == normal:
                continue
            biological_components.add_edge(int(parent), child)
        component_roots = {}
        for component in nx.connected_components(biological_components):
            root = min(
                component,
                key=lambda node: (
                    int(np.sum(status_with_normal[int(node)] == 1)),
                    int(node),
                ),
            )
            for node in component:
                component_roots[int(node)] = int(root)
        graph.remove_edges_from(list(graph.edges()))
        retained_records.clear()
        for child in biological_nodes:
            options = [
                record for record in retained_before_repair
                if int(record["child"]) == child
                and record["parent"] != normal
            ]
            if component_roots[child] == child:
                normal_record = normal_records_by_child.get(child)
                if normal_record is not None:
                    options.append(normal_record)
            if not options:
                raise nx.NetworkXException(
                    f"No repaired parent candidate remains for child {child}."
                )
            raw_scores = {
                record["parent"]: float(record["raw_parent_score"])
                for record in options
            }
            local_weights = _normalize_retained_parent_scores(
                raw_scores, parent_score_temperature=parent_score_temperature
            )
            best_parent = max(raw_scores, key=raw_scores.get)
            for record in options:
                updated = dict(record)
                parent = record["parent"]
                updated["local_parent_weight"] = local_weights[parent]
                updated["edge_weight"] = local_weights[parent]
                updated["weight"] = local_weights[parent]
                updated["was_local_best_parent"] = parent == best_parent
                updated["edge_type"] = (
                    "artificial_normal" if parent == normal else "biological"
                )
                retained_records.append(updated)
                graph.add_edge(
                    parent, child,
                    **{key: value for key, value in updated.items()
                       if key not in {"parent", "child", "edge_type"}},
                    edge_type=updated["edge_type"],
                )
        arborescence = maximum_spanning_arborescence(
            graph, attr="weight", default=0.0, preserve_attrs=True
        )

    # A deterministic representative is still needed because downstream
    # prediction expects one arborescence, but equal-scoring parents are not a
    # biological resolution. Preserve those alternatives explicitly.
    ambiguous_parent_candidates = []
    records_by_child: dict[int, list[dict[str, object]]] = {}
    for record in retained_records:
        records_by_child.setdefault(int(record["child"]), []).append(record)
    for child, records in records_by_child.items():
        best_score = max(float(record["raw_parent_score"]) for record in records)
        tied = [
            record for record in records
            if np.isclose(
                float(record["raw_parent_score"]),
                best_score,
                rtol=1e-12,
                atol=1e-12,
            )
        ]
        if len(tied) > 1:
            ambiguous_parent_candidates.append({
                "child": child,
                "selected_score": best_score,
                "candidate_parents": sorted(
                    (record["parent"] for record in tied),
                    key=lambda parent: (
                        0 if parent == normal else 1,
                        -1 if parent == normal else int(parent),
                    ),
                ),
            })
    expected_nodes = {normal, *biological_nodes}
    if set(arborescence.nodes) != expected_nodes:
        raise ValueError("Edmonds result is not spanning.")
    if arborescence.number_of_edges() != len(biological_nodes):
        raise ValueError("Edmonds result is not a rooted arborescence.")
    if arborescence.in_degree(normal) != 0 or any(
        arborescence.in_degree(node) != 1 for node in biological_nodes
    ):
        raise ValueError("Edmonds result has invalid parent assignments.")
    selected_roots = sorted(
        int(child) for parent, child in arborescence.edges() if parent == normal
    )
    selected = []
    lineage_tree = ig.Graph(n=status.shape[0], directed=True)
    for parent, child, data in sorted(
        arborescence.edges(data=True),
        key=lambda edge: (
            0 if edge[0] == normal else 1,
            int(edge[1]),
            0 if edge[0] == normal else int(edge[0]),
        ),
    ):
        lineage_tree.add_edge(0 if parent == normal else int(parent), int(child))
        selected.append({
            "parent": parent,
            "child": int(child),
            **dict(data),
            "edge_type": "artificial_normal" if parent == normal else "biological",
        })
    total_score = float(sum(float(data.get("weight", 0.0))
                            for _, _, data in arborescence.edges(data=True)))
    def diagnostics_frame(records: list[dict[str, object]]) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: repr(value)
                    if isinstance(value, (list, tuple, dict, set))
                    else str(value)
                )
        return frame
    metadata = {
        "artificial_normal": 0,
        "normal_edges_scored": True,
        "algorithm": "edmonds_maximum_spanning_arborescence",
        "method": "ranked_status_similarity_neighborhood",
        "status_similarity": similarity,
        "clone_connectivity_normalized": connectivity,
        "initial_parent_neighbors": initial_parent_neighbors,
        "parent_neighbor_step": parent_neighbor_step,
        "minimum_forward_proportion": minimum_forward_proportion,
        "expanded_minimum_forward_proportion": expanded_minimum_forward_proportion,
        "direction_contrast_weight": direction_contrast_weight,
        "parent_score_temperature": parent_score_temperature,
        "selected_roots": selected_roots,
        "n_normal_children": len(selected_roots),
        "ambiguous_parent_candidates": pd.DataFrame(
            ambiguous_parent_candidates,
            columns=["child", "selected_score", "candidate_parents"],
        ),
        "tie_break_is_biological": False,
        "tie_break_note": (
            "The selected tree is a deterministic representative. Equal-scoring "
            "parent alternatives are listed in ambiguous_parent_candidates."
        ),
        "selected_total_edmonds_score": total_score,
        "selected_total_tree_cost": -total_score,
        "selected_normal_edge_cost": -sum(
            float(data.get("weight", 0.0))
            for parent, _, data in arborescence.edges(data=True)
            if parent == normal
        ),
        "selected_biological_tree_cost": -sum(
            float(data.get("weight", 0.0))
            for parent, _, data in arborescence.edges(data=True)
            if parent != normal
        ),
        "selected_total_mt_to_wt_loss": float(sum(
            float(data.get("loss", 0.0)) for parent, _, data in arborescence.edges(data=True)
            if parent != normal
        )),
        "selected_total_wt_to_mt_gain": float(sum(
            float(data.get("gain", 0.0)) for parent, _, data in arborescence.edges(data=True)
            if parent != normal
        )),
        "expected_mt_burden": {
            str(node): float(np.sum(status_with_normal[node] == 1))
            for node in biological_nodes
        },
        "candidate_root_scores": {"adaptive_status_search": total_score},
        "candidate_tree_scores": {"adaptive_status_search": total_score},
        "candidate_root_results": pd.DataFrame([{
            "valid": True,
            "selected_roots": repr(selected_roots),
            "tree_score": total_score,
        }]),
        # Pass only scored candidate edges to Edmonds.
        "candidates": diagnostics_frame(retained_records),
        "retained_candidates": diagnostics_frame(retained_records),
        "child_diagnostics": diagnostics_frame(child_diagnostics),
        "selected": diagnostics_frame(selected),
    }
    return lineage_tree, metadata








































def _merge_lineage_edges_into_undirected_graph(
    clone_graph: ig.Graph,
    lineage_tree: ig.Graph,
) -> ig.Graph:
    """Return an undirected graph containing both local and directed edges."""
    n_nodes = max(clone_graph.vcount(), lineage_tree.vcount())
    edges = set()
    for source, target in clone_graph.get_edgelist():
        if source == target:
            continue
        edges.add(tuple(sorted((int(source), int(target)))))
    for source, target in lineage_tree.get_edgelist():
        if source == target:
            continue
        edges.add(tuple(sorted((int(source), int(target)))))

    graph = ig.Graph(n=n_nodes, edges=sorted(edges), directed=False)
    graph.simplify()
    return graph


def _get_clustermap_column_order(
    matrix: np.ndarray,
    method: str = "average",
    metric: str = "hamming",
) -> np.ndarray:
    """
    Return the column order produced by Seaborn-style hierarchical clustering.

    Seaborn clusters heatmap columns by passing the transposed matrix to
    `scipy.cluster.hierarchy.linkage`, then reading the dendrogram leaf order.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("`matrix` must be a 2D array.")
    n_columns = matrix.shape[1]
    if n_columns <= 1:
        return np.arange(n_columns)
    column_linkage = linkage(matrix.T, method=method, metric=metric)
    return leaves_list(column_linkage)


def _predict_consensus_profile_local(
    profile: np.ndarray,
    errors: np.ndarray,
    lineage_tree: ig.Graph,
    protected_error: float,
) -> np.ndarray:
    """Predict statuses with top-down two-state dynamic programming.

    Each clone is optimized only after its parent state is available. A
    descendant therefore cannot rewrite an ancestor. For a child, the two
    candidate states are evaluated from its raw call and error rate subject to
    irreversible mutation accumulation. With multiple parents, MT is required
    when any already-optimized parent is MT.
    """
    predicted = np.asarray(profile, dtype=int).copy()
    n_nodes, n_mutations = predicted.shape
    try:
        topo = [int(node) for node in lineage_tree.topological_sorting(mode="OUT")]
    except Exception as exc:  # pragma: no cover - guarded by caller validation
        raise ValueError("Lineage tree must be acyclic for local prediction.") from exc
    parents = {
        int(node): [int(parent) for parent in lineage_tree.predecessors(node)]
        for node in range(n_nodes)
    }
    for mutation in range(n_mutations):
        predicted[0, mutation] = -1
        for node in topo:
            if node == 0:
                continue
            raw_state = int(profile[node, mutation])
            flip_cost = -np.log(
                np.clip(errors[node, mutation], 1e-12, 1.0)
            )
            mt_required = any(
                predicted[parent, mutation] == 1
                for parent in parents[node]
            )
            candidate_states = (1,) if mt_required else (-1, 1)
            candidate_costs = {
                state: (0.0 if raw_state == state else flip_cost)
                for state in candidate_states
            }
            best_cost = min(candidate_costs.values())
            tied = [
                state
                for state, cost in candidate_costs.items()
                if np.isclose(cost, best_cost, rtol=0.0, atol=1e-12)
            ]
            predicted[node, mutation] = (
                raw_state if raw_state in tied else tied[0]
            )
    if np.any((profile == 1) & (predicted != 1)):
        raise RuntimeError("Local lineage prediction changed an MT consensus call.")
    return predicted


def _predict_consensus_profile_from_lineage(
    consensus_profile: np.ndarray,
    lineage_tree: ig.Graph,
    error_rate: np.ndarray,
    branch_protection: float = 1.0,
    mode: Literal["local", "global"] = "local",
    protected_error: float = 0.1,
    transition_penalty: float = 0.25,
) -> np.ndarray:
    """Resolve lineage disagreements with local repair or a global cut.

    WT and MT are encoded as 0 and 1 internally. In global mode, biological
    directed edges impose ``parent <= child``: an MT parent cannot transition
    to a WT child. A WT-to-MT transition remains allowed, but receives a
    finite acquisition penalty. If an observed MT-to-WT conflict exists, the
    optimizer chooses whether to flip the parent or child using their
    confidence-derived flip costs. The artificial normal is fixed at WT and
    is excluded from the acquisition penalty. Each mutation is solved jointly
    across the directed lineage.
    """
    profile = np.asarray(consensus_profile, dtype=int)
    errors = np.asarray(error_rate, dtype=float)
    if mode not in {"local", "global"}:
        raise ValueError("`mode` must be 'local' or 'global'.")
    if not np.isfinite(protected_error) or not 0.0 <= protected_error <= 1.0:
        raise ValueError("`protected_error` must be finite and in [0, 1].")
    if not np.isfinite(branch_protection) or branch_protection < 0.0:
        raise ValueError("`branch_protection` must be finite and nonnegative.")
    if not np.isfinite(transition_penalty) or transition_penalty < 0.0:
        raise ValueError("`transition_penalty` must be finite and nonnegative.")
    if profile.ndim != 2 or errors.shape != profile.shape:
        raise ValueError("Consensus profile and error_rate must have matching 2D shapes.")
    if profile.shape[0] != lineage_tree.vcount():
        raise ValueError("Consensus profile rows must match lineage tree vertices.")
    if not np.isin(profile, (-1, 1)).all():
        raise ValueError("Consensus profile must contain only -1 and 1.")
    if not np.all(np.isfinite(errors)) or np.any((errors < 0.0) | (errors > 1.0)):
        raise ValueError("Consensus error_rate must be finite and in [0, 1].")
    if not lineage_tree.is_dag():
        raise ValueError("Lineage tree must be acyclic for profile prediction.")

    n_nodes, n_mutations = profile.shape
    predicted = np.full_like(profile, -1)
    edges = [(int(parent), int(child)) for parent, child in lineage_tree.get_edgelist()]
    if not edges:
        return profile.copy()

    if mode == "local":
        return _predict_consensus_profile_local(
            profile,
            errors,
            lineage_tree,
            protected_error=protected_error,
        )

    outdegree = np.zeros(n_nodes, dtype=float)
    for parent, _ in edges:
        outdegree[parent] += 1.0
    node_flip_multiplier = 1.0 + float(branch_protection) * outdegree

    for mutation in range(n_mutations):
        source, sink = "__source__", "__sink__"
        flow = nx.DiGraph()
        flow.add_nodes_from([source, sink, *range(n_nodes)])
        flip_cost = (
            -np.log(np.clip(errors[:, mutation], 1e-12, 1.0))
            * node_flip_multiplier
        )
        biological_edge_count = sum(parent != 0 for parent, _ in edges)
        big_m = float(
            np.sum(flip_cost)
            + float(transition_penalty) * biological_edge_count
            + 1.0
        )
        for node in range(n_nodes):
            cost_wt = float(flip_cost[node]) if profile[node, mutation] == 1 else 0.0
            cost_mt = float(flip_cost[node]) if profile[node, mutation] == -1 else 0.0
            flow.add_edge(source, node, capacity=cost_wt)
            flow.add_edge(node, sink, capacity=cost_mt)
            if (
                profile[node, mutation] == 1
                and errors[node, mutation] <= protected_error
            ):
                # Preserve every high-confidence MT call in the optional
                # global mode; only WT calls can be promoted to MT.
                flow.add_edge(source, node, capacity=big_m)
        for parent, child in edges:
            # This hard constraint forbids MT -> WT, including from the
            # artificial normal. The reverse capacity is only a soft penalty
            # for biological WT -> MT acquisition.
            flow.add_edge(parent, child, capacity=big_m)
            if parent != 0 and transition_penalty > 0.0:
                flow.add_edge(
                    child,
                    parent,
                    capacity=float(transition_penalty),
                )
        flow.add_edge(0, sink, capacity=big_m)
        _, partition = nx.minimum_cut(flow, source, sink, capacity="capacity")
        mt_side, _ = partition
        for node in range(n_nodes):
            predicted[node, mutation] = 1 if node in mt_side else -1
    return predicted


def _cal_tmb_from_predict(consensus_profile_predict: np.ndarray) -> np.ndarray:
    """Calculate clone TMB as the number of predicted MT calls."""
    predict = np.asarray(consensus_profile_predict)
    if predict.ndim != 2:
        raise ValueError("`consensus_profile_predict` must be a 2D matrix.")
    return (predict == 1).sum(axis=1).astype(float)


def _order_lineage_by_tmb(
    adata_mut: AnnData,
    consensus_snv: np.ndarray,
    clone_graph: ig.Graph,
    lineage_tree: ig.Graph,
    tmb_clone: np.ndarray,
    label_key: str,
) -> tuple[
    np.ndarray,
    ig.Graph,
    ig.Graph,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Order consensus rows and lineage graph nodes by clone TMB."""
    n_nodes = consensus_snv.shape[0]
    if n_nodes == 0:
        return (
            consensus_snv,
            clone_graph,
            lineage_tree,
            np.zeros(0, dtype=float),
            np.zeros(adata_mut.n_obs, dtype=int),
            {"order": np.zeros(0, dtype=int), "labels": np.asarray([], dtype=object)},
        )

    tmb_clone = np.asarray(tmb_clone, dtype=float)
    if tmb_clone.shape[0] != n_nodes:
        tmb_clone = (consensus_snv == 1).sum(axis=1).astype(float)

    current_clone_labels = np.asarray(
        adata_mut.uns.get("consensus_clone_labels", np.arange(n_nodes - 1)),
        dtype=object,
    )
    stable_clone_labels = np.asarray(
        adata_mut.uns.get(
            "clone_consensus", {}
        ).get("clone_names", current_clone_labels),
        dtype=object,
    )
    stable_rank = {
        _normalize_label_key(label): rank
        for rank, label in enumerate(stable_clone_labels)
    }
    tie_order = np.asarray(
        [stable_rank.get(_normalize_label_key(label), index)
         for index, label in enumerate(current_clone_labels)],
        dtype=int,
    )
    clone_order = np.concatenate(
        [np.asarray([0], dtype=int), 1 + np.lexsort((tie_order, tmb_clone[1:]))]
    )
    old_to_new = np.empty(n_nodes, dtype=int)
    old_to_new[clone_order] = np.arange(n_nodes)

    clone_graph = _remap_graph_vertices(clone_graph, old_to_new, directed=False)
    lineage_tree = _remap_graph_vertices(lineage_tree, old_to_new, directed=True)

    labels = np.asarray(
        adata_mut.uns.get(
            "consensus_profile_labels",
            np.arange(n_nodes, dtype=object),
        ),
        dtype=object,
    )
    if labels.shape[0] != n_nodes:
        labels = np.arange(n_nodes, dtype=object)
    labels = labels[clone_order]

    clone_labels = np.asarray(
        adata_mut.uns.get("consensus_clone_labels", labels[1:]),
        dtype=object,
    )
    if clone_labels.shape[0] != max(n_nodes - 1, 0):
        clone_labels = labels[1:]
    original_label_to_node = {
        _normalize_label_key(label): index + 1
        for index, label in enumerate(clone_labels)
    }
    cell_labels = adata_mut.obs[label_key].to_numpy()
    labels_snv = np.empty(cell_labels.shape[0], dtype=int)
    for cell_index, label in enumerate(cell_labels):
        key = _normalize_label_key(label)
        if key not in original_label_to_node:
            raise KeyError(
                f"Clone label {label!r} from `adata_mut.obs['{label_key}']` "
                "is missing from `adata_mut.uns['consensus_clone_labels']`."
            )
        labels_snv[cell_index] = old_to_new[original_label_to_node[key]]

    return (
        consensus_snv[clone_order],
        clone_graph,
        lineage_tree,
        tmb_clone[clone_order],
        labels_snv,
        {"order": clone_order, "labels": labels},
    )


def _remap_graph_vertices(
    graph: ig.Graph,
    old_to_new: np.ndarray,
    directed: bool,
) -> ig.Graph:
    """Return a graph with vertex ids remapped from old indices to new ones."""
    edges = [
        (int(old_to_new[source]), int(old_to_new[target]))
        for source, target in graph.get_edgelist()
    ]
    remapped = ig.Graph(n=old_to_new.shape[0], edges=edges, directed=directed)
    remapped.simplify()
    return remapped


def _normalize_label_key(label) -> object:
    """Normalize clone labels for robust matching between pandas and NumPy."""
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, bytes):
        try:
            label = label.decode()
        except UnicodeDecodeError:
            return label
    if isinstance(label, str):
        text = label.strip()
        if text == "":
            return text
        signless = text[1:] if text[0] in ("+", "-") else text
        if signless.isdigit():
            return int(text)
        try:
            numeric = float(text)
        except ValueError:
            return text
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return text
    return label


def cal_tree_layout(
    adata_mut: AnnData,
    method: str = "fr",
    tmb_strength: float = 1.0,
    relax_iter: int = 300,
    n_outer: int = 80,
    force_niter: int = 8,
    thred: float = 0.1,
    graph_override=None,
    lineage_override=None,
    initial_layout=None,
    output_view_key: str = "Lineage_tree_coords_view_genetic",
    store_canonical: bool = True,
) -> None:
    """
    Calculate lineage-tree coordinates with soft TMB-guided horizontal layout.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with serialized `.uns['Lineage_tree']` and
        `.uns['TMB_clone']`. `Clone_connectivity` is not used for layout.
    method : str, default='fr'
        igraph layout method used for local force-layout updates.
    tmb_strength : float, default=1.0
        Strength of the horizontal pull toward normalized clone TMB.
    relax_iter : int, default=300
        Number of final view-layout overlap relaxation iterations.
    n_outer : int, default=300
        Number of alternating force-layout and TMB-gravity updates.
    force_niter : int, default=20
        Number of force-directed iterations performed in each outer update.
        Larger values give igraph more opportunity to repel overlapping
        clones while preserving the weighted lineage edges.
    thred : float, default=0.1
        Minimum node spacing target for the relaxed view layout.

    Returns
    -------
    None
        Modifies `adata_mut` in place:
        - `.uns['Lineage_tree_coords']`: Original normalized coordinates
        - `.uns['Lineage_tree_coords_original']`: Copy of original coordinates
        - `.uns['Lineage_tree_coords_view_genetic']`: Genetic relaxed view coordinates
        - `.uns['Lineage_tree_layout_info']`: Layout metadata
    """
    if int(n_outer) < 0 or int(relax_iter) < 0 or int(force_niter) <= 0:
        raise ValueError("`n_outer` and `relax_iter` must be non-negative; `force_niter` must be positive.")
    lineage_tree = (
        lineage_override
        if lineage_override is not None
        else _get_uns_graph(
            adata_mut,
            "Lineage_tree",
            directed=True,
            required=True,
        )
    )
    # Use the inferred directed lineage as the layout graph.
    clone_graph = graph_override if graph_override is not None else lineage_tree
    n_graph_nodes = max(clone_graph.vcount(), lineage_tree.vcount())
    edge_weights_by_pair: dict[tuple[int, int], float] = {}
    stored_lineage_weights = {}
    if graph_override is None:
        stored_edges = adata_mut.uns.get("Lineage_tree_edge_weights")
        if isinstance(stored_edges, pd.DataFrame):
            for edge in stored_edges.itertuples(index=False):
                pair = tuple(sorted((int(edge.source), int(edge.target))))
                stored_lineage_weights[pair] = float(edge.weight)
        else:
            for edge in stored_edges or []:
                if isinstance(edge, dict):
                    pair = tuple(sorted((int(edge["source"]), int(edge["target"]))))
                    stored_lineage_weights[pair] = float(edge.get("weight", 1.0))
    for edge_index, (source, target) in enumerate(clone_graph.get_edgelist()):
        if source != target:
            if "weight" in clone_graph.es.attributes():
                clone_edge_weight = float(clone_graph.es[edge_index]["weight"])
            else:
                clone_edge_weight = 1.0
            edge_weights_by_pair[tuple(sorted((int(source), int(target))))] = clone_edge_weight
    for source, target in lineage_tree.get_edgelist():
        if source != target:
            pair = tuple(sorted((int(source), int(target))))
            if graph_override is not None and "weight" in lineage_tree.es.attributes():
                lineage_edge_index = lineage_tree.get_eid(source, target)
                lineage_edge_weight = float(lineage_tree.es[lineage_edge_index]["weight"])
            else:
                lineage_edge_weight = stored_lineage_weights.get(pair, 1.0)
            edge_weights_by_pair[pair] = lineage_edge_weight
    edge_pairs = sorted(edge_weights_by_pair)
    g = ig.Graph(n=n_graph_nodes, edges=edge_pairs, directed=False)
    g.simplify()
    tmb = np.asarray(adata_mut.uns["TMB_clone"], dtype=float)
    if tmb.shape != (g.vcount(),):
        raise ValueError("`TMB_clone` must match the clone graph node count.")

    edge_pairs = [
        (int(source), int(target))
        for source, target in g.get_edgelist()
        if source != target
    ]
    # Represent directed lineage edges as undirected springs for layout.
    weighted_graph = ig.Graph(n=g.vcount(), edges=edge_pairs, directed=False)
    weighted_graph.es["weight"] = [
        edge_weights_by_pair[tuple(sorted(edge))] for edge in edge_pairs
    ]

    # Initialize the layout from clone-level spectral components.
    if initial_layout is None:
        layout, spectral_info = _spectral_tree_layout_seed(
            adata_mut=adata_mut,
            n_nodes=g.vcount(),
            tmb=tmb,
            force_seed=np.asarray(weighted_graph.layout("kk"), dtype=float),
            weighted_adjacency=np.asarray(weighted_graph.get_adjacency(attribute="weight").data, dtype=float),
        )
    else:
        layout = np.asarray(initial_layout, dtype=float)
        if layout.shape != (g.vcount(), 2) or not np.all(np.isfinite(layout)):
            raise ValueError("`initial_layout` must be finite with shape (n_nodes, 2).")
        spectral_info = {
            "source": "provided_initial_layout",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": np.nan,
        }
    # Preserve the spectral coordinates as the canonical biological layout.
    layout_original = _normalize_layout_2d(layout)
    layout_view_seed = layout_original.copy()
    finite_tmb = np.isfinite(tmb)
    biological_tmb = finite_tmb.copy()
    if biological_tmb.size:
        biological_tmb[0] = False
    if np.count_nonzero(biological_tmb) >= 2 and np.ptp(tmb[biological_tmb]) > 1e-12:
        # Apply rank-normalized TMB as a one-sided rightward force.
        tmb_rank = np.zeros_like(tmb, dtype=float)
        tmb_rank[biological_tmb] = rankdata(
            tmb[biological_tmb], method="average"
        )
        tmb_rank[biological_tmb] = (
            tmb_rank[biological_tmb] - np.min(tmb_rank[biological_tmb])
        ) / (np.ptp(tmb_rank[biological_tmb]) + 1e-12)
        tmb_gravity_available = True
    else:
        tmb_rank = np.zeros_like(tmb, dtype=float)
        tmb_gravity_available = False

    for _ in range(max(int(n_outer), 1)):
        layout_view_seed = np.asarray(
            weighted_graph.layout(
                method,
                seed=layout_view_seed,
                weights="weight",
            start_temp=0.05,
            niter=int(force_niter),
        ),
        dtype=float,
        )
        if tmb_gravity_available and float(tmb_strength) > 0:
            x_min = float(np.min(layout_view_seed[:, 0]))
            x_max = float(np.max(layout_view_seed[:, 0]))
            x_range = max(x_max - x_min, 1e-12)
            # Apply the TMB displacement after spring-layout refinement.
            gravity_step = min(0.12, 0.025 * float(tmb_strength))
            layout_view_seed[:, 0] += gravity_step * tmb_rank * x_range

        # Enforce parent-before-child ordering along the x-axis.
        x_range = max(float(np.ptp(layout_view_seed[:, 0])), 1e-12)
        order_margin = 0.01 * x_range
        try:
            topological_nodes = lineage_tree.topological_sorting(mode="OUT")
        except Exception:
            topological_nodes = list(range(lineage_tree.vcount()))
        for child in topological_nodes:
            parents = lineage_tree.neighbors(int(child), mode="IN")
            if not parents:
                continue
            minimum_child_x = max(
                layout_view_seed[int(parent), 0] + order_margin
                for parent in parents
            )
            if layout_view_seed[int(child), 0] < minimum_child_x:
                layout_view_seed[int(child), 0] = minimum_child_x

        # Compact sibling branches while preserving their relative positions.
        layout_view_seed[:, 1] *= 0.92

    n_nodes = layout_original.shape[0]
    view_min_dist = max(
        thred,
        min(0.18, 0.7 / max(np.sqrt(max(n_nodes, 1)), 1.0)),
    )

    layout_view = _relax_lineage_tree_layout(
        _normalize_layout_2d(layout_view_seed),
        min_dist=view_min_dist,
        iterations=relax_iter,
        padding=0.08,
    )

    if store_canonical:
        adata_mut.uns["Lineage_tree_coords"] = layout_original
        adata_mut.uns["Lineage_tree_coords_original"] = layout_original.copy()
    adata_mut.uns[output_view_key] = layout_view
    adata_mut.uns["Lineage_tree_layout_info"] = {
        "method": f"{method}_with_tmb_gravity",
        "edge_source": "Lineage_tree_only" if graph_override is None else "provided_directed_lineage",
        "edge_weight_source": "directed_lineage_only",
        "directed_edge_weight": 1.0,
        "undirected_edge_weight": 0.0,
        "initialization": spectral_info["source"],
        "spectral_x_component": spectral_info["x_component"],
        "spectral_y_component": spectral_info["y_component"],
        "spectral_x_tmb_spearman": spectral_info["x_tmb_spearman"],
        "spectral_component_source": spectral_info["source"],
        "tmb_strength": float(tmb_strength),
        "tmb_gravity_mode": "one_sided_rightward_rank_force",
        "tmb_gravity_available": bool(tmb_gravity_available),
        "force_niter": int(force_niter),
        "force_outer_iterations": int(max(int(n_outer), 1)),
        "min_distance": float(view_min_dist),
        "relax_iter": int(relax_iter),
        "output_view_key": output_view_key,
        "store_canonical": bool(store_canonical),
    }


def _spectral_tree_layout_seed(
    adata_mut: AnnData,
    n_nodes: int,
    tmb: np.ndarray,
    force_seed: np.ndarray,
    weighted_adjacency: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Build a clone-level spectral seed for the tree layout.

    ``X_genetic`` is a cell-level spectral embedding. Its rows are
    averaged by lineage node, then the component with the strongest rank
    association with clone TMB is selected as x. A second component, after
    removing its linear association with x, seeds y. The normal node has no
    cells, so it is placed just before the real-clone spectral range.
    """
    force_seed = np.asarray(force_seed, dtype=float)
    if force_seed.shape != (n_nodes, 2):
        raise ValueError("`force_seed` must have shape (n_nodes, 2).")

    # Use the clone-graph Laplacian and TMB to orient the spectral seed.
    adjacency = np.asarray(
        weighted_adjacency
        if weighted_adjacency is not None
        else adata_mut.uns.get("_evofate_tree_weighted_adjacency", np.zeros((n_nodes, n_nodes))),
        dtype=float,
    )
    if adjacency.shape != (n_nodes, n_nodes):
        adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    if not np.any(adjacency):
        # Use a zero adjacency when no weighted graph is available.
        adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    if np.any(adjacency):
        degree = adjacency.sum(axis=1)
        inv_degree = np.zeros_like(degree)
        mask = degree > eps
        inv_degree[mask] = 1.0 / np.sqrt(degree[mask])
        laplacian = np.eye(n_nodes) - inv_degree[:, None] * adjacency * inv_degree[None, :]
        values, vectors = np.linalg.eigh((laplacian + laplacian.T) * 0.5)
        valid = np.flatnonzero(values > 1e-10)
        if valid.size:
            basis = vectors[:, valid]
            basis = (basis - basis.mean(axis=0)) / (basis.std(axis=0) + eps)
            tmb_reference = np.asarray(tmb, dtype=float)
            finite = np.isfinite(tmb_reference)
            correlations = np.full(basis.shape[1], np.nan, dtype=float)
            for component in range(basis.shape[1]):
                if np.count_nonzero(finite) > 1:
                    correlations[component] = spearmanr(
                        basis[finite, component], tmb_reference[finite]
                    ).statistic
            x_component = int(np.nanargmax(np.abs(correlations))) if np.any(np.isfinite(correlations)) else 0
            timing = basis[:, x_component].copy()
            if np.isfinite(correlations[x_component]) and correlations[x_component] < 0:
                timing *= -1.0
            timing = (timing - timing.mean()) / (timing.std() + eps)
            residual_basis = basis - timing[:, None] * (
                timing @ basis
            )[None, :] / (timing @ timing + eps)
            residual_variance = np.var(residual_basis, axis=0)
            residual_variance[x_component] = -np.inf
            branch_index = int(np.argmax(residual_variance)) if np.any(np.isfinite(residual_variance)) else None
            branch = residual_basis[:, branch_index] if branch_index is not None else force_seed[:, 1]
            branch = (branch - branch.mean()) / (branch.std() + eps)
            if branch[np.argmax(np.abs(branch))] < 0:
                branch *= -1.0
            seed = _normalize_layout_2d(np.column_stack((timing, branch)))
            return seed, {
                "source": "clone_graph_normalized_laplacian_spectral_basis",
                "x_component": x_component,
                "y_component": branch_index,
                "x_tmb_spearman": float(spearmanr(timing[finite], tmb_reference[finite]).statistic),
            }

    n_real = max(n_nodes - 1, 0)
    if n_real == 0:
        return force_seed.copy(), {
            "source": "force_fallback",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": None,
        }

    embedding = adata_mut.obsm.get("X_genetic")
    if embedding is None:
        return force_seed.copy(), {
            "source": "force_fallback_missing_X_genetic",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": None,
        }
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[0] != adata_mut.n_obs:
        return force_seed.copy(), {
            "source": "force_fallback_invalid_X_genetic",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": None,
        }

    # Map cells to lineage node indices. ordered_clone is preferred because it
    # is written in the same node order as Lineage_tree; original clone labels
    # are the fallback before cal_clone_connectivity has been run.
    codes = np.full(adata_mut.n_obs, -1, dtype=int)
    if "ordered_clone" in adata_mut.obs:
        values = np.asarray(adata_mut.obs["ordered_clone"], dtype=object)
        for i, value in enumerate(values):
            try:
                code = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= code < n_nodes:
                codes[i] = code
    if not np.any(codes >= 1) and "clone" in adata_mut.obs:
        labels = np.asarray(
            adata_mut.uns.get("consensus_clone_labels", []),
            dtype=object,
        )
        label_to_code = {
            _normalize_label_key(label): index + 1
            for index, label in enumerate(labels[:n_real])
        }
        for i, value in enumerate(np.asarray(adata_mut.obs["clone"], dtype=object)):
            code = label_to_code.get(_normalize_label_key(value))
            if code is not None:
                codes[i] = code

    means = np.full((n_nodes, embedding.shape[1]), np.nan, dtype=float)
    for node in range(1, n_nodes):
        selected = codes == node
        if np.any(selected):
            means[node] = np.nanmean(embedding[selected], axis=0)

    valid_nodes = np.all(np.isfinite(means[1:]), axis=1)
    valid_tmb = np.isfinite(tmb[1:])
    valid = valid_nodes & valid_tmb
    if np.count_nonzero(valid) < 2:
        return force_seed.copy(), {
            "source": "force_fallback_insufficient_clone_spectral_means",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": None,
        }

    component_scores = np.full(embedding.shape[1], -np.inf, dtype=float)
    component_correlations = np.full(embedding.shape[1], np.nan, dtype=float)
    for component in range(embedding.shape[1]):
        values = means[1:, component][valid]
        if np.ptp(values) <= eps:
            continue
        correlation = spearmanr(values, tmb[1:][valid]).statistic
        if np.isfinite(correlation):
            component_correlations[component] = float(correlation)
            component_scores[component] = abs(float(correlation))

    if not np.any(np.isfinite(component_scores)):
        return force_seed.copy(), {
            "source": "force_fallback_constant_spectral_components",
            "x_component": None,
            "y_component": None,
            "x_tmb_spearman": None,
        }

    x_component = int(np.nanargmax(component_scores))
    x = means[:, x_component].copy()
    x_real = x[1:]
    x_real = (x_real - np.mean(x_real)) / (np.std(x_real) + eps)
    if component_correlations[x_component] < 0:
        x_real *= -1.0
    x[1:] = x_real
    x[0] = float(np.min(x_real) - 1.0)

    # Choose the highest-variance remaining component and residualize it
    # against x, so branches are visible without duplicating the x axis.
    variances = np.nanvar(means[1:, :], axis=0)
    variances[x_component] = -np.inf
    y_component = int(np.argmax(variances)) if np.any(np.isfinite(variances)) else None
    if y_component is None or not np.isfinite(variances[y_component]):
        y = force_seed[:, 1].copy()
    else:
        y = means[:, y_component].copy()
        y_real = y[1:]
        y_real = y_real - np.mean(y_real)
        x_centered = x_real - np.mean(x_real)
        y_real = y_real - (
            np.dot(y_real, x_centered)
            / (np.dot(x_centered, x_centered) + eps)
        ) * x_centered
        y_real /= np.std(y_real) + eps
        y[1:] = y_real
        y[0] = 0.0

    seed = np.column_stack((x, y))
    seed = _normalize_layout_2d(seed)
    return seed, {
        "source": "adata.obsm['X_genetic']_clone_means",
        "x_component": x_component,
        "y_component": y_component,
        "x_tmb_spearman": float(component_correlations[x_component]),
    }


def _normalize_layout_2d(
    layout: np.ndarray,
    padding: float = 0.0,
) -> np.ndarray:
    """Normalize a 2D layout to [padding, 1 - padding] on each axis."""
    layout = np.asarray(layout, dtype=float)
    if layout.ndim != 2 or layout.shape[1] != 2:
        raise ValueError("`layout` must have shape (n_nodes, 2).")
    if layout.shape[0] == 0:
        return layout.copy()

    span = np.ptp(layout, axis=0)
    normalized = np.zeros_like(layout, dtype=float)
    for axis in range(2):
        if span[axis] > 0:
            normalized[:, axis] = (
                layout[:, axis] - np.min(layout[:, axis])
            ) / span[axis]
        else:
            normalized[:, axis] = 0.5
    if padding > 0:
        normalized = padding + normalized * (1.0 - 2.0 * padding)
    return normalized


def _relax_lineage_tree_layout(
    layout: np.ndarray,
    min_dist: float,
    iterations: int = 300,
    padding: float = 0.08,
    anchor_strength: float = 0.035,
) -> np.ndarray:
    """Spread nearby nodes while keeping the layout close to its original shape."""
    view = _normalize_layout_2d(layout, padding=padding)
    anchors = view.copy()
    n_nodes = view.shape[0]
    if n_nodes <= 1:
        return view

    min_dist = float(max(min_dist, 1e-6))
    for _ in range(max(int(iterations), 0)):
        displacement = np.zeros_like(view)
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                delta = view[j] - view[i]
                dist = float(np.linalg.norm(delta))
                if dist >= min_dist:
                    continue
                if dist <= 1e-12:
                    angle = (i * 131 + j * 17) * np.pi / 180.0
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / dist
                push = 0.5 * (min_dist - dist) * direction
                push[0] *= 0.55
                displacement[i] -= push
                displacement[j] += push

        displacement += anchor_strength * (anchors - view)
        if np.max(np.linalg.norm(displacement, axis=1)) < 1e-5:
            break
        view += displacement
        view = np.clip(view, padding, 1.0 - padding)

    return _normalize_layout_2d(view, padding=padding)


    # Canonical probability-based lineage helpers.
def _get_full_consensus_for_lineage(
    adata_mut: AnnData,
    consensus_snv: np.ndarray,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load status, posterior MT/WT probabilities, and reliability."""
    stored = adata_mut.uns.get("clone_consensus")
    required = (
        "status",
        "posterior_p_mt",
        "posterior_p_wt",
        "posterior_reliability",
        "clone_names",
    )
    if not isinstance(stored, dict) or any(key not in stored for key in required):
        raise KeyError(
            "Graph-evidence posterior outputs are missing. Run "
            "cal_consensus_profile before cal_clone_connectivity."
        )
    status = np.asarray(stored["status"], dtype=int)
    p_mt = np.asarray(stored["posterior_p_mt"], dtype=float)
    p_wt = np.asarray(stored["posterior_p_wt"], dtype=float)
    reliability = np.asarray(stored["posterior_reliability"], dtype=float)
    source_labels = np.asarray(stored["clone_names"], dtype=object)
    if any(
        value.ndim != 2
        or value.shape != status.shape
        for value in (p_mt, p_wt, reliability)
    ):
        raise ValueError(
            "Consensus status, posterior, and reliability matrices must have "
            "matching shapes."
        )
    if not np.isin(status, (-1, 0, 1)).all():
        raise ValueError("Consensus status must contain only -1, 0, and 1.")
    if (
        not np.all(np.isfinite(p_mt))
        or not np.all(np.isfinite(p_wt))
        or np.any((p_mt < 0.0) | (p_mt > 1.0))
        or np.any((p_wt < 0.0) | (p_wt > 1.0))
        or not np.allclose(p_mt + p_wt, 1.0, atol=1e-6)
        or not np.all(np.isfinite(reliability))
        or np.any((reliability < 0.0) | (reliability > 1.0))
    ):
        raise ValueError(
            "Posterior probabilities must sum to one and reliability values "
            "must be finite in [0, 1]."
        )

    current_labels = np.asarray(
        adata_mut.uns.get(
            "consensus_clone_labels",
            np.sort(adata_mut.obs[label_key].astype(str).unique()),
        ),
        dtype=object,
    )
    source_row = {
        _normalize_label_key(label): i for i, label in enumerate(source_labels)
    }
    try:
        row_order = [source_row[_normalize_label_key(label)] for label in current_labels]
    except KeyError as exc:
        raise KeyError("Consensus clone labels do not match clone_consensus.") from exc

    status = status[row_order]
    p_mt = p_mt[row_order]
    p_wt = p_wt[row_order]
    reliability = reliability[row_order]
    status = np.vstack((np.full((1, status.shape[1]), -1, dtype=int), status))
    p_mt = np.vstack((np.zeros((1, p_mt.shape[1])), p_mt))
    p_wt = np.vstack((np.ones((1, p_wt.shape[1])), p_wt))
    reliability = np.vstack((np.ones((1, reliability.shape[1])), reliability))
    if status.shape[1] == 0:
        shape = (consensus_snv.shape[0], 1)
        status = np.full(shape, -1, dtype=int)
        p_mt = np.zeros(shape, dtype=float)
        p_wt = np.ones(shape, dtype=float)
        reliability = np.zeros(shape, dtype=float)
        reliability[0, 0] = 1.0
    return status, p_mt, p_wt, reliability
