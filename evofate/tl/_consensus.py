"""Clone-level mutation consensus estimation for EvoFATE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scipy.special import logsumexp

from ._genetic_utils import _as_dense_array, _as_string_array

if TYPE_CHECKING:
    from anndata import AnnData


def _cal_theta_graph_mle(
    matrix: np.ndarray,
    connectivity,
    clusters,
    graph_strength: float = 1.0,
    missing_wt_lambda: float = 10.0,
    max_iter: int = 50,
    tol: float = 1e-5,
    eps: float = 1e-12,
    fill_no_info: float = np.nan,
) -> dict:
    """
    Estimate cluster-level true MT probabilities with graph-informed MLE.

    Parameters
    ----------
    matrix : np.ndarray
        Mutation matrix with values 1 (observed MT), -1 (observed WT),
        and 0 (missing). Shape is (n_cells, n_mutations).
    connectivity : scipy sparse matrix or array-like
        Cell-cell connectivity graph. Row i contains 1-hop neighbors for
        target cell i, and edge weights are used as local support weights.
    clusters : array-like
        Cluster/clone labels for each cell.
    graph_strength : float, default=1.0
        Multiplier applied to weighted graph pseudo-counts.
    missing_wt_lambda : float, default=10.0
        Relative likelihood that a missing observation belongs to true WT
        rather than true MT when estimating local observation probabilities.
    max_iter : int, default=50
        Maximum Newton-Raphson iterations.
    tol : float, default=1e-5
        Convergence tolerance for theta updates.
    eps : float, default=1e-12
        Numerical stability constant.
    fill_no_info : float, default=np.nan
        Value returned for cluster/mutation pairs with no observed MT/WT count.

    Returns
    -------
    dict
        Dictionary containing theta, coverage, and purity. Coverage is the raw
        MT/WT non-missing fraction and does not include graph pseudo-counts.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("`matrix` must be a 2D array.")

    n_cells, _ = matrix.shape
    clusters_array = np.asarray(clusters).reshape(-1)
    if clusters_array.shape[0] != n_cells:
        raise ValueError("`clusters` must have one label per matrix row.")

    graph = (
        connectivity.tocsr().copy()
        if issparse(connectivity)
        else csr_matrix(connectivity)
    )
    if graph.shape != (n_cells, n_cells):
        raise ValueError(
            "`connectivity` must have shape (n_cells, n_cells); "
            f"got {graph.shape} for {n_cells} cells."
        )
    if graph_strength < 0:
        raise ValueError("`graph_strength` must be non-negative.")
    if missing_wt_lambda < 0:
        raise ValueError("`missing_wt_lambda` must be non-negative.")

    graph = graph.astype(float, copy=False)
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    cluster_labels, cluster_index = np.unique(clusters_array, return_inverse=True)
    n_clusters = cluster_labels.shape[0]
    cluster_membership = csr_matrix(
        (
            np.ones(n_cells, dtype=float),
            (cluster_index, np.arange(n_cells)),
        ),
        shape=(n_clusters, n_cells),
    )

    mt_obs = (matrix == 1).astype(float, copy=False)
    wt_obs = (matrix == -1).astype(float, copy=False)
    missing_counts = (matrix == 0).astype(float, copy=False)

    raw_MT = np.asarray(cluster_membership @ mt_obs, dtype=float)
    raw_WT = np.asarray(cluster_membership @ wt_obs, dtype=float)
    raw_missing = np.asarray(cluster_membership @ missing_counts, dtype=float)
    raw_count = raw_MT + raw_WT
    cluster_sizes = np.bincount(cluster_index, minlength=n_clusters).astype(float)
    coverage = np.divide(
        raw_count,
        cluster_sizes[:, None],
        out=np.zeros_like(raw_count, dtype=float),
        where=cluster_sizes[:, None] > eps,
    )

    # Aggregate observed neighbor states using the connectivity weights.
    neighbor_MT_support = graph @ mt_obs
    neighbor_WT_support = graph @ wt_obs

    # Use graph support to estimate local observation probabilities.
    pseudo_MT = np.asarray(
        cluster_membership @ neighbor_MT_support,
        dtype=float,
    )
    pseudo_WT = np.asarray(
        cluster_membership @ neighbor_WT_support,
        dtype=float,
    )

    n_MT_eff = raw_MT + graph_strength * pseudo_MT
    n_WT_eff = raw_WT + graph_strength * pseudo_WT
    p_obs_MT, p_obs_WT = _estimate_p_obs(
        raw_MT=raw_MT,
        raw_WT=raw_WT,
        raw_missing=raw_missing,
        split_MT=n_MT_eff,
        split_WT=n_WT_eff,
        missing_wt_lambda=missing_wt_lambda,
        eps=eps,
    )

    # Estimate theta from clone-level observations and inferred observation probabilities.
    theta = np.full(raw_MT.shape, fill_no_info, dtype=float)
    info_mask = raw_count > eps
    if np.any(info_mask):
        t = raw_MT[info_mask] / (raw_count[info_mask] + eps)
        t = np.clip(t, 0.01, 0.99)

        m_MT = raw_MT[info_mask]
        m_WT = raw_WT[info_mask]
        m_missing = raw_missing[info_mask]

        drop_MT = 1.0 - p_obs_MT[info_mask]
        drop_WT = 1.0 - p_obs_WT[info_mask]
        delta_drop = drop_MT - drop_WT

        for _ in range(max_iter):
            p_missing = t * drop_MT + (1.0 - t) * drop_WT + eps

            gradient = (
                m_MT / (t + eps)
                - m_WT / (1.0 - t + eps)
                + m_missing * delta_drop / p_missing
            )
            hessian = (
                -m_MT / ((t + eps) ** 2)
                - m_WT / (((1.0 - t) + eps) ** 2)
                - m_missing * (delta_drop**2) / (p_missing**2)
            )

            step = np.divide(
                gradient,
                hessian,
                out=np.zeros_like(gradient),
                where=np.abs(hessian) > eps,
            )
            t_new = np.clip(t - step, 1e-4, 1.0 - 1e-4)

            if np.max(np.abs(t_new - t)) < tol:
                t = t_new
                break
            t = t_new

        theta[info_mask] = t

    raw_mt_fraction = np.divide(
        raw_MT,
        raw_count + eps,
        out=np.full_like(raw_MT, 0.5, dtype=float),
        where=raw_count > eps,
    )
    purity = np.clip(
        2.0 * np.abs(raw_mt_fraction - 0.5),
        0.0,
        1.0,
    )

    return {
        "theta": theta,
        "coverage": coverage,
        "purity": purity,
    }



def _estimate_p_obs(
    raw_MT: np.ndarray,
    raw_WT: np.ndarray,
    raw_missing: np.ndarray,
    split_MT: np.ndarray,
    split_WT: np.ndarray,
    missing_wt_lambda: float = 10.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate local p_obs_MT and p_obs_WT from raw MT/WT/missing counts.

    `missing_wt_lambda` controls how much more likely a missing observation is
    assigned to true WT than true MT. The missing split is guided by effective
    MT/WT support, while p_obs values are calculated only from raw observed
    MT/WT counts plus the split of raw missing observations.

    Returns
    -------
    p_obs_MT : np.ndarray
        P(observed MT | true MT) for each cluster/mutation pair.
    p_obs_WT : np.ndarray
        P(observed WT | true WT) for each cluster/mutation pair.
    """
    raw_MT = np.asarray(raw_MT, dtype=float)
    raw_WT = np.asarray(raw_WT, dtype=float)
    raw_missing = np.asarray(raw_missing, dtype=float)
    split_MT = np.asarray(split_MT, dtype=float)
    split_WT = np.asarray(split_WT, dtype=float)
    if (
        raw_MT.shape != raw_WT.shape
        or raw_MT.shape != raw_missing.shape
        or raw_MT.shape != split_MT.shape
        or raw_MT.shape != split_WT.shape
    ):
        raise ValueError("Count arrays must have matching shapes.")
    if missing_wt_lambda < 0:
        raise ValueError("`missing_wt_lambda` must be non-negative.")

    mt_weight = split_MT
    wt_weight = split_WT
    missing_weight_denominator = mt_weight + missing_wt_lambda * wt_weight
    missing_to_MT = np.divide(
        raw_missing * mt_weight,
        missing_weight_denominator,
        out=np.zeros_like(raw_missing, dtype=float),
        where=missing_weight_denominator > eps,
    )
    missing_to_WT = np.divide(
        raw_missing * wt_weight * missing_wt_lambda,
        missing_weight_denominator,
        out=np.zeros_like(raw_missing, dtype=float),
        where=missing_weight_denominator > eps,
    )
    mt_denominator = raw_MT + missing_to_MT
    wt_denominator = raw_WT + missing_to_WT

    p_obs_MT = np.divide(
        raw_MT,
        mt_denominator,
        out=np.zeros_like(raw_MT, dtype=float),
        where=mt_denominator > eps,
    )
    p_obs_WT = np.divide(
        raw_WT,
        wt_denominator,
        out=np.zeros_like(raw_WT, dtype=float),
        where=wt_denominator > eps,
    )

    return p_obs_MT, p_obs_WT



def _validate_mutation_values(matrix) -> None:
    """Validate mutation calls without densifying sparse matrices."""
    if issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix).ravel()
    if values.size == 0:
        return
    valid = np.isin(values, [-1, 0, 1])
    if not np.all(valid):
        invalid_values = np.unique(values[~valid])
        raise ValueError(
            "`adata_mut.X` must contain only -1, 0, and 1 mutation calls; "
            f"found invalid values {invalid_values[:10]!r}."
        )


def _prepare_connectivity(connectivities) -> csr_matrix:
    """
    Return a clean CSR connectivity matrix for deterministic graph evidence.

    The returned graph is a copy. Self-loops and explicit zeros are removed.
    Weights must be finite and nonnegative. If any positive weight is above 1,
    all weights are divided by the maximum weight so external evidence remains
    on a comparable [0, 1] scale.
    """
    graph = (
        connectivities.tocsr().copy()
        if issparse(connectivities)
        else csr_matrix(connectivities)
    )
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("`connectivities` must be a square cell-cell matrix.")
    graph = graph.astype(float, copy=False)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if graph.data.size == 0:
        return graph
    if not np.all(np.isfinite(graph.data)):
        raise ValueError("Connectivity weights must be finite.")
    if np.any(graph.data < 0):
        raise ValueError("Connectivity weights must be nonnegative.")
    max_weight = float(graph.data.max())
    if max_weight > 1.0:
        graph.data = graph.data / max_weight
    graph.eliminate_zeros()
    return graph


def _get_clone_graph_weights(
    graph: csr_matrix,
    anchor_mask: np.ndarray,
    use_one_hop: bool = True,
    use_two_hop: bool = True,
    external_weight_scale: float = 1.0,
    external_weight_cap: float | None = 0.5,
    hop_decay_rate: float = 0.7,
    anchor_top_k: int = 10,
    clone_top_k: int = 10,
    selection_frequency_power: float = 1.0,
    return_hops: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """
    Return one graph-supported evidence weight per cell for one clone.

    Anchor cells receive weight 1. Non-anchor 1-hop cells receive the strongest
    direct edge from any anchor. Exclusively 2-hop cells receive the strongest
    max-product two-edge path from any anchor. A cell is assigned to only its
    nearest available hop.
    """
    if not issparse(graph):
        graph = csr_matrix(graph)
    graph = graph.tocsr()
    anchor_mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    if graph.shape[0] != anchor_mask.shape[0]:
        raise ValueError("`anchor_mask` length must match graph size.")
    if external_weight_scale < 0:
        raise ValueError("`external_weight_scale` must be nonnegative.")
    if external_weight_cap is not None and external_weight_cap < 0:
        raise ValueError("`external_weight_cap` must be nonnegative or None.")
    if hop_decay_rate < 0:
        raise ValueError("`hop_decay_rate` must be nonnegative.")
    if anchor_top_k <= 0 or clone_top_k <= 0:
        raise ValueError("Top-k values must be positive integers.")
    if selection_frequency_power <= 0:
        raise ValueError("`selection_frequency_power` must be positive.")

    n_cells = graph.shape[0]
    weights = np.zeros(n_cells, dtype=float)
    anchor_weights = np.zeros(n_cells, dtype=float)
    anchor_weights[anchor_mask] = 1.0
    weights[anchor_mask] = 1.0
    anchor_indices = np.flatnonzero(anchor_mask)
    if anchor_indices.size == 0:
        raise ValueError("Each clone must contain at least one anchor cell.")

    one_hop_sum_before_cap = 0.0
    two_hop_sum_before_cap = 0.0
    one_hop_strength = np.zeros(n_cells, dtype=float)
    two_hop_strength = np.zeros(n_cells, dtype=float)

    def _keep_top_k(values: np.ndarray, k: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        positive = np.flatnonzero(values > eps)
        if positive.size == 0:
            return np.zeros_like(values, dtype=bool)
        if positive.size <= k:
            mask = np.zeros_like(values, dtype=bool)
            mask[positive] = True
            return mask
        order = np.lexsort((positive, -values[positive]))
        keep = positive[order[:k]]
        mask = np.zeros_like(values, dtype=bool)
        mask[keep] = True
        return mask

    if use_one_hop and anchor_indices.size > 0:
        one_hop_sums = np.zeros(n_cells, dtype=float)
        one_hop_counts = np.zeros(n_cells, dtype=float)
        for anchor in anchor_indices:
            row_start = graph.indptr[anchor]
            row_end = graph.indptr[anchor + 1]
            candidates = graph.indices[row_start:row_end]
            candidate_weights = graph.data[row_start:row_end]
            if candidates.size == 0:
                continue
            valid = (candidate_weights > eps) & (~anchor_mask[candidates])
            if not np.any(valid):
                continue
            candidates = candidates[valid]
            candidate_weights = candidate_weights[valid]
            if candidates.size > anchor_top_k:
                order = np.lexsort((candidates, -candidate_weights))
                take = order[:anchor_top_k]
                candidates = candidates[take]
                candidate_weights = candidate_weights[take]
            np.add.at(one_hop_sums, candidates, candidate_weights)
            np.add.at(one_hop_counts, candidates, 1.0)
        one_hop_mask = one_hop_counts > 0
        one_hop_strength[one_hop_mask] = (
            (one_hop_sums[one_hop_mask] / one_hop_counts[one_hop_mask])
            * np.power(
                one_hop_counts[one_hop_mask] / max(anchor_indices.size, 1),
                float(selection_frequency_power),
            )
        )
        one_hop_mask = _keep_top_k(one_hop_strength, clone_top_k)
        one_hop_strength[~one_hop_mask] = 0.0
        one_hop_sum_before_cap = float(one_hop_strength[one_hop_mask].sum())
        weights[one_hop_mask] = one_hop_strength[one_hop_mask]
    else:
        one_hop_mask = np.zeros(n_cells, dtype=bool)

    if use_two_hop and anchor_indices.size > 0:
        two_hop_sums = np.zeros(n_cells, dtype=float)
        two_hop_counts = np.zeros(n_cells, dtype=float)
        for anchor in anchor_indices:
            first_start = graph.indptr[anchor]
            first_end = graph.indptr[anchor + 1]
            intermediates = graph.indices[first_start:first_end]
            first_weights = graph.data[first_start:first_end]
            if intermediates.size == 0:
                continue
            anchor_candidate_strength = np.zeros(n_cells, dtype=float)
            for intermediate, first_weight in zip(intermediates, first_weights):
                if first_weight <= eps or anchor_mask[intermediate]:
                    continue
                second_start = graph.indptr[intermediate]
                second_end = graph.indptr[intermediate + 1]
                candidates = graph.indices[second_start:second_end]
                second_weights = graph.data[second_start:second_end]
                if candidates.size == 0:
                    continue
                valid = second_weights > eps
                if not np.any(valid):
                    continue
                candidates = candidates[valid]
                second_weights = second_weights[valid]
                path_weights = float(first_weight) * second_weights
                np.maximum.at(anchor_candidate_strength, candidates, path_weights)
            anchor_candidate_strength[anchor_mask] = 0.0
            if not use_one_hop:
                anchor_candidate_strength[one_hop_mask] = 0.0
            candidate_indices = np.flatnonzero(anchor_candidate_strength > eps)
            if candidate_indices.size == 0:
                continue
            candidate_weights = anchor_candidate_strength[candidate_indices]
            if candidate_indices.size > anchor_top_k:
                order = np.lexsort((candidate_indices, -candidate_weights))
                take = order[:anchor_top_k]
                candidate_indices = candidate_indices[take]
                candidate_weights = candidate_weights[take]
            np.add.at(two_hop_sums, candidate_indices, candidate_weights)
            np.add.at(two_hop_counts, candidate_indices, 1.0)
        two_hop_mask = two_hop_counts > 0
        two_hop_strength[two_hop_mask] = (
            (two_hop_sums[two_hop_mask] / two_hop_counts[two_hop_mask])
            * np.power(
                two_hop_counts[two_hop_mask] / max(anchor_indices.size, 1),
                float(selection_frequency_power),
            )
            * float(np.exp(-float(hop_decay_rate)))
        )
        two_hop_mask = _keep_top_k(two_hop_strength, clone_top_k)
        two_hop_strength[~two_hop_mask] = 0.0
        two_hop_sum_before_cap = float(two_hop_strength[two_hop_mask].sum())
    else:
        two_hop_mask = np.zeros(n_cells, dtype=bool)

    weights = np.maximum(one_hop_strength, two_hop_strength)
    weights[anchor_mask] = 1.0
    one_hop_weights = np.zeros(n_cells, dtype=float)
    one_hop_weights[one_hop_mask] = one_hop_strength[one_hop_mask]
    two_hop_weights = np.zeros(n_cells, dtype=float)
    two_hop_weights[two_hop_mask] = two_hop_strength[two_hop_mask]

    external_mask = ~anchor_mask
    weights[external_mask] *= float(external_weight_scale)
    one_hop_sum_before_cap *= float(external_weight_scale)
    two_hop_sum_before_cap *= float(external_weight_scale)
    external_mass = float(weights[external_mask].sum())
    anchor_mass = float(anchor_mask.sum())
    cap_scale = 1.0
    if external_weight_cap is not None and external_mass > eps:
        cap_limit = float(external_weight_cap) * anchor_mass
        cap_scale = min(1.0, cap_limit / (external_mass + eps))
        weights[external_mask] *= cap_scale
        one_hop_weights[~anchor_mask] *= cap_scale
        two_hop_weights[~anchor_mask] *= cap_scale

    summary = {
        "n_anchor": int(anchor_mask.sum()),
        "n_one_hop": int(np.count_nonzero(one_hop_mask)),
        "n_two_hop": int(np.count_nonzero(two_hop_mask)),
        "anchor_weight_sum": float(weights[anchor_mask].sum()),
        "one_hop_weight_sum_before_cap": one_hop_sum_before_cap,
        "two_hop_weight_sum_before_cap": two_hop_sum_before_cap,
        "external_weight_sum_after_cap": float(weights[external_mask].sum()),
        "external_scale_applied": float(cap_scale),
    }
    if return_hops:
        hop_weights = {
            "anchor": anchor_weights,
            "one_hop": one_hop_weights,
            "two_hop": two_hop_weights,
            "combined": weights,
        }
        return weights, summary, hop_weights
    return weights, summary


def _calculate_weighted_state_counts(
    X,
    clone_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted MT, WT, and missing counts for every mutation."""
    weights = np.asarray(clone_weights, dtype=float).reshape(-1)
    if issparse(X):
        matrix = X.tocsr().copy()
        if matrix.shape[0] != weights.shape[0]:
            raise ValueError("`clone_weights` must have one value per cell.")
        _validate_mutation_values(matrix)
        matrix.eliminate_zeros()

        mt_matrix = matrix.copy()
        mt_matrix.data = (mt_matrix.data == 1).astype(float)
        mt_matrix.eliminate_zeros()
        wt_matrix = matrix.copy()
        wt_matrix.data = (wt_matrix.data == -1).astype(float)
        wt_matrix.eliminate_zeros()

        weighted_mt = np.asarray(weights @ mt_matrix).ravel().astype(float)
        weighted_wt = np.asarray(weights @ wt_matrix).ravel().astype(float)
        total_weight = float(weights.sum())
        weighted_missing = total_weight - weighted_mt - weighted_wt
        weighted_missing = np.maximum(weighted_missing, 0.0)
        return weighted_mt, weighted_wt, weighted_missing

    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise ValueError("`X` must be a 2D mutation matrix.")
    if matrix.shape[0] != weights.shape[0]:
        raise ValueError("`clone_weights` must have one value per cell.")
    _validate_mutation_values(matrix)
    weighted_mt = weights @ (matrix == 1).astype(float, copy=False)
    weighted_wt = weights @ (matrix == -1).astype(float, copy=False)
    weighted_missing = weights @ (matrix == 0).astype(float, copy=False)
    return (
        np.asarray(weighted_mt, dtype=float),
        np.asarray(weighted_wt, dtype=float),
        np.asarray(weighted_missing, dtype=float),
    )


def _estimate_theta_from_weighted_counts(
    weighted_mt: np.ndarray,
    weighted_wt: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Calculate Beta-smoothed mutant probabilities from weighted evidence."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("`theta_alpha` and `theta_beta` must be positive.")
    weighted_mt = np.asarray(weighted_mt, dtype=float)
    weighted_wt = np.asarray(weighted_wt, dtype=float)
    if weighted_mt.shape != weighted_wt.shape:
        raise ValueError("`weighted_mt` and `weighted_wt` must match.")
    denominator = weighted_mt + weighted_wt + float(alpha) + float(beta)
    theta = (weighted_mt + float(alpha)) / (denominator + eps)
    return np.clip(theta, 0.0, 1.0)


def _calculate_zero_hop_call(
    mt_count: np.ndarray,
    wt_count: np.ndarray,
    missing_count: np.ndarray,
    support_tau: float,
    coverage_power: float,
    purity_power: float,
    support_power: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return initial binary status, confidence, coverage, and purity."""
    mt_count = np.asarray(mt_count, dtype=float)
    wt_count = np.asarray(wt_count, dtype=float)
    missing_count = np.asarray(missing_count, dtype=float)
    if mt_count.shape != wt_count.shape or mt_count.shape != missing_count.shape:
        raise ValueError("Input count arrays must have matching shapes.")
    if support_tau <= 0:
        raise ValueError("`zero_hop_support_tau` must be positive.")
    if coverage_power <= 0 or purity_power <= 0 or support_power <= 0:
        raise ValueError("Zero-hop powers must be positive.")

    observed = mt_count + wt_count
    total = observed + missing_count
    coverage = np.divide(
        observed,
        total + eps,
        out=np.zeros_like(observed, dtype=float),
        where=total > eps,
    )
    purity = np.divide(
        np.maximum(mt_count, wt_count),
        observed + eps,
        out=np.zeros_like(observed, dtype=float),
        where=observed > eps,
    )
    agreement = np.divide(
        np.abs(mt_count - wt_count),
        observed + eps,
        out=np.zeros_like(observed, dtype=float),
        where=observed > eps,
    )
    coverage_term = np.clip(coverage / 0.1, 0.0, 1.0)
    support = 1.0 - np.exp(-observed / float(support_tau))
    confidence = np.zeros_like(observed, dtype=float)
    info_mask = observed > eps
    if np.any(info_mask):
        confidence[info_mask] = np.power(
            np.clip(
                coverage_term[info_mask] ** float(coverage_power)
                * agreement[info_mask] ** float(purity_power)
                * support[info_mask] ** float(support_power),
                0.0,
                1.0,
            ),
            1.0 / (float(coverage_power) + float(purity_power) + float(support_power)),
        )
    status = np.where(mt_count > wt_count, 1, -1).astype(int)
    confidence[~info_mask] = 0.0
    status[~info_mask] = -1
    return status, np.clip(confidence, 0.0, 1.0), coverage, purity


def _calculate_zero_hop_probability(
    mt_count: np.ndarray,
    wt_count: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the zero-hop MT probability, independent of confidence."""
    mt_count = np.asarray(mt_count, dtype=float)
    wt_count = np.asarray(wt_count, dtype=float)
    if mt_count.shape != wt_count.shape:
        raise ValueError("MT and WT count arrays must have matching shapes.")
    observed = mt_count + wt_count
    probability = np.divide(
        mt_count,
        observed + eps,
        out=np.full_like(observed, 0.5, dtype=float),
        where=observed > eps,
    )
    return np.clip(probability, 0.0, 1.0)


def _calculate_weighted_neighbor_vote(
    X,
    cell_weights: np.ndarray,
    support_tau: float,
    hop_decay: float,
    support_power: float,
    agreement_power: float,
    graph_strength_power: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    For every mutation, return signed weighted WT/MT vote and reliability.

    Missing observations are ignored. No coverage is calculated.
    """
    if support_tau <= 0:
        raise ValueError("`support_tau` must be positive.")
    if hop_decay < 0:
        raise ValueError("`hop_decay` must be nonnegative.")
    if support_power <= 0 or agreement_power <= 0 or graph_strength_power <= 0:
        raise ValueError("Neighbor vote powers must be positive.")

    weights = np.asarray(cell_weights, dtype=float).reshape(-1)
    if issparse(X):
        matrix = X.tocsr().copy()
        if matrix.shape[0] != weights.shape[0]:
            raise ValueError("`cell_weights` must have one value per cell.")
        _validate_mutation_values(matrix)
        matrix.eliminate_zeros()
        mt_matrix = matrix.copy()
        mt_matrix.data = (mt_matrix.data == 1).astype(float)
        mt_matrix.eliminate_zeros()
        wt_matrix = matrix.copy()
        wt_matrix.data = (wt_matrix.data == -1).astype(float)
        wt_matrix.eliminate_zeros()
        obs_matrix = matrix.copy()
        obs_matrix.data = (obs_matrix.data != 0).astype(float)
        obs_matrix.eliminate_zeros()

        weighted_mt = np.asarray(weights @ mt_matrix).ravel().astype(float)
        weighted_wt = np.asarray(weights @ wt_matrix).ravel().astype(float)
        observed_weight = np.asarray(weights @ obs_matrix).ravel().astype(float)
        observed_count = np.asarray(obs_matrix.sum(axis=0)).ravel().astype(float)
    else:
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError("`X` must be a 2D mutation matrix.")
        if matrix.shape[0] != weights.shape[0]:
            raise ValueError("`cell_weights` must have one value per cell.")
        _validate_mutation_values(matrix)
        mt_mask = (matrix == 1).astype(float, copy=False)
        wt_mask = (matrix == -1).astype(float, copy=False)
        obs_mask = (matrix != 0).astype(float, copy=False)
        weighted_mt = weights @ mt_mask
        weighted_wt = weights @ wt_mask
        observed_weight = weights @ obs_mask
        observed_count = np.sum(obs_mask, axis=0)
    weighted_mt = np.asarray(weighted_mt, dtype=float)
    weighted_wt = np.asarray(weighted_wt, dtype=float)
    observed_weight = np.asarray(observed_weight, dtype=float)
    observed_count = np.asarray(observed_count, dtype=float)
    total = weighted_mt + weighted_wt
    vote = np.divide(
        weighted_mt - weighted_wt,
        total + eps,
        out=np.zeros_like(weighted_mt, dtype=float),
        where=total > eps,
    )
    agreement = np.abs(vote)
    support = 1.0 - np.exp(-total / float(support_tau))
    graph_strength = np.divide(
        observed_weight,
        observed_count + eps,
        out=np.zeros_like(observed_weight, dtype=float),
        where=observed_count > eps,
    )
    reliability = np.zeros_like(total, dtype=float)
    info_mask = total > eps
    if np.any(info_mask):
        reliability[info_mask] = hop_decay * np.power(
            np.clip(
                support[info_mask] ** float(support_power)
                * agreement[info_mask] ** float(agreement_power)
                * graph_strength[info_mask] ** float(graph_strength_power),
                0.0,
                1.0,
            ),
            1.0 / (
                float(support_power)
                + float(agreement_power)
                + float(graph_strength_power)
            ),
        )
    stats = {
        "weighted_mt": weighted_mt,
        "weighted_wt": weighted_wt,
        "total_weight": total,
        "observed_weight": observed_weight,
        "observed_count": observed_count,
        "support": support,
        "agreement": agreement,
        "graph_strength": graph_strength,
    }
    return vote, np.clip(reliability, 0.0, float(hop_decay)), stats


def _update_signed_consensus(
    signed_evidence: np.ndarray,
    neighbor_vote: np.ndarray,
    neighbor_reliability: np.ndarray,
    update_strength: float,
) -> np.ndarray:
    """Add neighbor evidence only to the unresolved fraction of inference."""
    signed_evidence = np.asarray(signed_evidence, dtype=float)
    neighbor_vote = np.asarray(neighbor_vote, dtype=float)
    neighbor_reliability = np.asarray(neighbor_reliability, dtype=float)
    if (
        signed_evidence.shape != neighbor_vote.shape
        or signed_evidence.shape != neighbor_reliability.shape
    ):
        raise ValueError("Consensus arrays must have matching shapes.")
    if update_strength < 0:
        raise ValueError("`update_strength` must be nonnegative.")
    updated = signed_evidence + float(update_strength) * (
        1.0 - np.abs(signed_evidence)
    ) * neighbor_reliability * neighbor_vote
    return np.clip(updated, -1.0, 1.0)


def _update_mutation_probability(
    probability: np.ndarray,
    neighbor_vote: np.ndarray,
    neighbor_reliability: np.ndarray,
    update_strength: float,
) -> np.ndarray:
    """Update MT probability using neighbor evidence, never confidence."""
    probability = np.asarray(probability, dtype=float)
    neighbor_vote = np.asarray(neighbor_vote, dtype=float)
    neighbor_reliability = np.asarray(neighbor_reliability, dtype=float)
    if (
        probability.shape != neighbor_vote.shape
        or probability.shape != neighbor_reliability.shape
    ):
        raise ValueError("Probability arrays must have matching shapes.")
    if update_strength < 0:
        raise ValueError("`update_strength` must be nonnegative.")
    neighbor_probability = 0.5 * (np.clip(neighbor_vote, -1.0, 1.0) + 1.0)
    update = np.clip(
        float(update_strength) * np.clip(neighbor_reliability, 0.0, 1.0),
        0.0,
        1.0,
    )
    updated = probability + update * (neighbor_probability - probability)
    return np.clip(updated, 0.0, 1.0)


def _calculate_raw_clone_coverage_purity(
    X,
    labels: np.ndarray,
    clone_labels: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve raw clone-member coverage and purity semantics."""
    coverage_rows = []
    purity_rows = []
    for clone_label in clone_labels:
        anchor_mask = labels == clone_label
        anchor_weights = anchor_mask.astype(float)
        raw_mt, raw_wt, _ = _calculate_weighted_state_counts(X, anchor_weights)
        raw_count = raw_mt + raw_wt
        cluster_size = float(np.count_nonzero(anchor_mask))
        coverage = np.divide(
            raw_count,
            cluster_size,
            out=np.zeros_like(raw_count, dtype=float),
            where=cluster_size > eps,
        )
        raw_mt_fraction = np.divide(
            raw_mt,
            raw_count + eps,
            out=np.full_like(raw_mt, 0.5, dtype=float),
            where=raw_count > eps,
        )
        purity = np.clip(2.0 * np.abs(raw_mt_fraction - 0.5), 0.0, 1.0)
        coverage_rows.append(coverage)
        purity_rows.append(purity)
    return np.vstack(coverage_rows), np.vstack(purity_rows)


def _get_mutation_matrix(adata_mut: AnnData, layer: str | None) -> Any:
    """Return and validate the configured mutation matrix."""
    matrix = adata_mut.X if layer is None else adata_mut.layers[layer]
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("The mutation matrix must be two-dimensional.")
    _validate_mutation_values(matrix)
    if matrix.shape[0] != adata_mut.n_obs:
        raise ValueError("The mutation matrix must have one row per cell.")
    return matrix


def _top_positive_indices(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Select positive values deterministically, breaking ties by index."""
    values = np.asarray(values, dtype=float).reshape(-1)
    indices = np.flatnonzero(values > 0.0)
    if indices.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    order = np.lexsort((indices, -values[indices]))[:k]
    selected = indices[order]
    return selected.astype(int), values[selected].astype(float)


def _build_cell_candidate_neighborhoods(
    connectivity,
    k_neighbors: int,
    two_hop_decay: float,
    edge_power: float,
    use_two_hop: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse top-k neighborhoods per cell.

    By default only direct neighbors are used. Set ``use_two_hop=True`` to
    add the approximate-candidate, exact max-product two-hop branch.
    """
    if k_neighbors <= 0:
        raise ValueError("`k_neighbors` must be positive.")
    if two_hop_decay < 0:
        raise ValueError("`two_hop_decay` must be nonnegative.")
    if edge_power <= 0:
        raise ValueError("`edge_power` must be positive.")
    if not isinstance(use_two_hop, (bool, np.bool_)):
        raise TypeError("`use_two_hop` must be boolean.")
    graph = (
        connectivity.tocsr().astype(float, copy=True)
        if issparse(connectivity)
        else csr_matrix(np.asarray(connectivity, dtype=float))
    )
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Connectivity must be a square matrix.")
    graph.data[~np.isfinite(graph.data)] = 0.0
    graph.data[graph.data < 0.0] = 0.0
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    n_cells = graph.shape[0]
    neighbor_indices = np.empty(n_cells, dtype=object)
    neighbor_weights = np.empty(n_cells, dtype=object)
    hop_scale = float(np.exp(-two_hop_decay))
    two_hop_candidates = None
    candidate_limit = max(4 * int(k_neighbors), int(k_neighbors))
    if use_two_hop:
        # Use sparse multiplication only to shortlist candidates. The final
        # weights below are still exact max-products, not summed A @ A values.
        two_hop_candidates = (graph @ graph).tocsr()
        two_hop_candidates.eliminate_zeros()
    def build_row(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        start, end = graph.indptr[i], graph.indptr[i + 1]
        direct_idx, direct_weight = _top_positive_indices(
            graph.data[start:end], k_neighbors
        )
        direct_idx = graph.indices[start:end][direct_idx]
        direct_set = set(int(index) for index in direct_idx)

        two_idx = np.empty(0, dtype=int)
        two_weight = np.empty(0, dtype=float)
        if use_two_hop:
            candidate_start = two_hop_candidates.indptr[i]
            candidate_end = two_hop_candidates.indptr[i + 1]
            candidate_idx, _ = _top_positive_indices(
                two_hop_candidates.data[candidate_start:candidate_end],
                candidate_limit,
            )
            candidate_idx = two_hop_candidates.indices[
                candidate_start:candidate_end
            ][candidate_idx]
            candidate_set = set(
                int(index)
                for index in candidate_idx
                if int(index) != i and int(index) not in direct_set
            )

            # Evaluate exact max-product paths only for the sparse candidate
            # set, and skip cells already represented by a direct edge.
            path_max: dict[int, float] = {}
            row_indices = graph.indices[start:end]
            row_weights = graph.data[start:end]
            for intermediate, first_weight in zip(row_indices, row_weights):
                l_start, l_end = graph.indptr[intermediate], graph.indptr[intermediate + 1]
                for candidate, second_weight in zip(
                    graph.indices[l_start:l_end], graph.data[l_start:l_end]
                ):
                    candidate = int(candidate)
                    if candidate not in candidate_set or second_weight <= 0.0:
                        continue
                    path_strength = float(first_weight * second_weight * hop_scale)
                    if path_strength > path_max.get(candidate, 0.0):
                        path_max[candidate] = path_strength
            if path_max:
                path_idx = np.fromiter(path_max.keys(), dtype=int)
                path_values = np.fromiter(path_max.values(), dtype=float)
                selected_order = np.lexsort((path_idx, -path_values))[:k_neighbors]
                two_idx = path_idx[selected_order]
                two_weight = path_values[selected_order]

        combined: dict[int, float] = {
            int(index): float(weight) for index, weight in zip(direct_idx, direct_weight)
        }
        for index, weight in zip(two_idx, two_weight):
            combined[int(index)] = max(combined.get(int(index), 0.0), float(weight))
        if combined:
            indices = np.fromiter(combined.keys(), dtype=int)
            weights = np.fromiter(combined.values(), dtype=float)
            order = np.lexsort((indices, -weights))
            neighbor_indices[i] = indices[order]
            neighbor_weights[i] = np.power(weights[order], edge_power)
        else:
            return i, np.empty(0, dtype=int), np.empty(0, dtype=float)
        return i, neighbor_indices[i], neighbor_weights[i]

    for i in range(n_cells):
        i, indices, weights = build_row(i)
        neighbor_indices[i] = indices
        neighbor_weights[i] = weights
    return neighbor_indices, neighbor_weights


def infer_cell_mutation_profiles(
    adata_mut: AnnData,
    connectivity_key: str = "genetic_lineage_connectivity",
    layer: str | None = None,
    k_neighbors: int = 15,
    use_two_hop: bool = True,
    two_hop_decay: float = 0.5,
    edge_power: float = 2.0,
    conflict_top_k: int = 3,
    confidence_top_k: int = 3,
    observed_conflict_penalty: float = 0.8,
    agreement_power: float = 1.5,
    graph_strength_power: float = 1.0,
    missing_self_correct_prob_cap: float = 0.9,
    random_state: int = 0,
) -> dict[str, Any]:
    """Infer cell-level binary mutation calls and correctness probabilities.

    Observed focal calls are immutable. Missing calls are graph-imputed from
    selected neighbors. The returned correctness probability is probability-
    like evidence strength, not a calibrated posterior without validation.
    ``random_state`` is accepted for API reproducibility; this deterministic
    implementation does not draw random numbers.
    """
    del random_state
    if connectivity_key not in adata_mut.obsp:
        raise KeyError(f"`adata_mut.obsp['{connectivity_key}']` is missing.")
    if conflict_top_k <= 0 or confidence_top_k <= 0:
        raise ValueError("Conflict and confidence top-k values must be positive.")
    if not 0.0 <= observed_conflict_penalty <= 1.0:
        raise ValueError("`observed_conflict_penalty` must be in [0, 1].")
    if agreement_power <= 0 or graph_strength_power <= 0:
        raise ValueError("Information powers must be positive.")
    if not 0.5 <= missing_self_correct_prob_cap <= 1.0:
        raise ValueError("`missing_self_correct_prob_cap` must be in [0.5, 1].")

    matrix = _get_mutation_matrix(adata_mut, layer)
    n_cells, n_mutations = matrix.shape
    neighbor_indices, neighbor_weights = _build_cell_candidate_neighborhoods(
        adata_mut.obsp[connectivity_key],
        k_neighbors=k_neighbors,
        use_two_hop=use_two_hop,
        two_hop_decay=two_hop_decay,
        edge_power=edge_power,
    )
    dense_matrix = None if issparse(matrix) else np.asarray(matrix)

    def infer_row(i: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(neighbor_indices[i], dtype=int)
        weights = np.asarray(neighbor_weights[i], dtype=float)
        if issparse(matrix):
            values = matrix[indices].toarray() if indices.size else np.empty((0, n_mutations))
            focal = matrix.getrow(i).toarray().ravel()
        else:
            values = dense_matrix[indices] if indices.size else np.empty((0, n_mutations))
            focal = dense_matrix[i]
        row_status = np.full(n_mutations, -1, dtype=np.int8)
        row_correct_prob = np.full(n_mutations, 0.5, dtype=float)
        row_source = np.full(n_mutations, 2, dtype=np.int8)
        observed_focal = np.isin(focal, (-1, 1))
        row_status[observed_focal] = focal[observed_focal].astype(np.int8)
        row_correct_prob[observed_focal] = 1.0
        row_source[observed_focal] = 0

        if indices.size == 0:
            return i, row_status, row_correct_prob, row_source
        observed_neighbors = values != 0
        mt_weight = weights @ (values == 1)
        wt_weight = weights @ (values == -1)
        observed_weight = weights @ observed_neighbors
        conflict_mask = (
            observed_neighbors
            & (values == -focal[None, :])
            & observed_focal[None, :]
        )
        conflict_values = conflict_mask * weights[:, None]
        conflict_count = conflict_mask.sum(axis=0)
        conflict_k = min(int(conflict_top_k), conflict_values.shape[0])
        if conflict_k:
            conflict_top = np.partition(
                conflict_values,
                conflict_values.shape[0] - conflict_k,
                axis=0,
            )[-conflict_k:].sum(axis=0)
            conflict_denominator = np.minimum(conflict_count, conflict_k)
            strong_conflict = np.divide(
                conflict_top,
                conflict_denominator,
                out=np.zeros(n_mutations, dtype=float),
                where=conflict_denominator > 0,
            )
        else:
            strong_conflict = np.zeros(n_mutations, dtype=float)
        conflict_fraction = np.divide(
            conflict_values.sum(axis=0),
            observed_weight + 1e-12,
        )
        row_correct_prob[observed_focal] = np.clip(
            1.0
            - observed_conflict_penalty
            * conflict_fraction[observed_focal]
            * strong_conflict[observed_focal],
            0.5,
            1.0,
        )
        missing = ~observed_focal
        usable = (mt_weight + wt_weight) > 1e-12
        imputed = missing & usable
        row_status[imputed] = np.where(
            mt_weight[imputed] > wt_weight[imputed], 1, -1
        ).astype(np.int8)
        row_source[imputed] = 1
        total = mt_weight[imputed] + wt_weight[imputed]
        agreement = np.abs(mt_weight[imputed] - wt_weight[imputed]) / (total + 1e-12)
        observed_strength = observed_neighbors * weights[:, None]
        observed_count = observed_neighbors.sum(axis=0)
        strength_k = min(int(confidence_top_k), observed_strength.shape[0])
        if strength_k:
            strength_top = np.partition(
                observed_strength,
                observed_strength.shape[0] - strength_k,
                axis=0,
            )[-strength_k:].sum(axis=0)
            strength_denominator = np.minimum(observed_count, strength_k)
            top_strength = np.divide(
                strength_top,
                strength_denominator,
                out=np.zeros(n_mutations, dtype=float),
                where=strength_denominator > 0,
            )
        else:
            top_strength = np.zeros(n_mutations, dtype=float)
        information = agreement**agreement_power * top_strength[imputed]**graph_strength_power
        row_correct_prob[imputed] = np.minimum(
            0.5 + 0.5 * np.clip(information, 0.0, 1.0),
            missing_self_correct_prob_cap,
        )
        return i, row_status, row_correct_prob, row_source

    status = np.empty((n_cells, n_mutations), dtype=np.int8)
    correct_prob = np.empty((n_cells, n_mutations), dtype=float)
    source = np.empty((n_cells, n_mutations), dtype=np.int8)
    for i in range(n_cells):
        i, row_status, row_correct_prob, row_source = infer_row(i)
        status[i] = row_status
        correct_prob[i] = row_correct_prob
        source[i] = row_source

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_weights": neighbor_weights,
        "cell_status": status,
        "cell_correct_prob": np.clip(correct_prob, 0.5, 1.0),
        "cell_source": source,
        "source_labels": np.asarray(["direct", "imputed", "unsupported"], dtype=str),
        "parameters": {
            "connectivity_key": connectivity_key,
            "layer": layer,
            "k_neighbors": int(k_neighbors),
            "use_two_hop": bool(use_two_hop),
            "two_hop_decay": float(two_hop_decay),
            "edge_power": float(edge_power),
        },
    }


def infer_clone_consensus(
    adata_mut: AnnData,
    cell_profile: dict[str, Any],
    clone_key: str = "clone",
    imputed_cell_weight: float = 0.35,
    prior_mt: float = 0.5,
    store_key: str = "clone_consensus",
) -> dict[str, Any]:
    """Infer clone status and probability-like status error rates."""
    if clone_key not in adata_mut.obs:
        raise KeyError(f"`adata_mut.obs['{clone_key}']` is missing.")
    if not 0.0 <= imputed_cell_weight <= 1.0:
        raise ValueError("`imputed_cell_weight` must be in [0, 1].")
    if not 0.0 < prior_mt < 1.0:
        raise ValueError("`prior_mt` must be strictly between 0 and 1.")
    status_cells = np.asarray(cell_profile["cell_status"], dtype=int)
    correct_prob = np.asarray(cell_profile["cell_correct_prob"], dtype=float)
    source = np.asarray(cell_profile["cell_source"], dtype=int)
    if status_cells.ndim != 2 or correct_prob.shape != status_cells.shape or source.shape != status_cells.shape:
        raise ValueError("Cell profile arrays must have matching 2D shapes.")
    if not np.isin(status_cells, (-1, 1)).all():
        raise ValueError("Cell status must contain only -1 and 1.")
    labels = np.asarray(adata_mut.obs[clone_key].to_numpy())
    if labels.shape[0] != status_cells.shape[0]:
        raise ValueError("Clone labels must match the cell profile rows.")
    clone_names, clone_codes = np.unique(labels, return_inverse=True)
    n_clones, n_mutations = clone_names.size, status_cells.shape[1]
    error_rate = np.full((n_clones, n_mutations), 0.5, dtype=float)
    status = np.full((n_clones, n_mutations), -1, dtype=np.int8)
    posterior_mt = np.full_like(error_rate, prior_mt)
    posterior_wt = np.full_like(error_rate, 1.0 - prior_mt)
    log_odds = np.zeros_like(error_rate)
    source_weights = np.array([1.0, float(imputed_cell_weight), 0.0])
    for clone_index in range(n_clones):
        members = np.flatnonzero(clone_codes == clone_index)
        q = np.clip(correct_prob[members], 1e-12, 1.0 - 1e-12)
        reliability = source_weights[np.clip(source[members], 0, 2)]
        reliability_total = reliability.sum(axis=0)
        # Consensus status is a reliability-weighted vote. Observed MT/WT
        # calls contribute 1.0, while imputed calls contribute
        # ``imputed_cell_weight``.
        mt_support = np.sum(
            reliability * (status_cells[members] == 1), axis=0
        )
        wt_support = np.sum(
            reliability * (status_cells[members] == -1), axis=0
        )
        vote_total = mt_support + wt_support
        vote_p_mt = np.divide(
            mt_support + float(prior_mt),
            vote_total + 1.0,
            out=np.full_like(mt_support, float(prior_mt)),
            where=vote_total > 0,
        )
        consensus_status = np.where(vote_p_mt > 0.5, 1, -1)
        status[clone_index] = consensus_status

        # Estimate error from confidence-aware expected evidence separately for
        # MT-supporting and WT-supporting calls; the hard status vote remains
        # defined by the reliability-weighted support above.
        mt_likelihood = np.where(status_cells[members] == 1, q, 1.0 - q)
        wt_likelihood = 1.0 - mt_likelihood
        expected_mt_support = np.sum(reliability * mt_likelihood, axis=0)
        expected_wt_support = np.sum(reliability * wt_likelihood, axis=0)
        expected_total = expected_mt_support + expected_wt_support
        p_mt = np.divide(
            expected_mt_support,
            expected_total,
            out=np.full_like(vote_p_mt, float(prior_mt)),
            where=expected_total > 0,
        )
        p_mt = np.clip(p_mt, 0.0, 1.0)
        p_wt = 1.0 - p_mt
        posterior_mt[clone_index] = p_mt
        posterior_wt[clone_index] = p_wt
        log_odds_clone = np.log(np.clip(p_mt, 1e-12, 1.0)) - np.log(
            np.clip(p_wt, 1e-12, 1.0)
        )
        log_odds[clone_index] = log_odds_clone
        expected_consensus_support = np.where(
            consensus_status == 1,
            expected_mt_support,
            expected_wt_support,
        )
        consensus_support_fraction = np.divide(
            expected_consensus_support,
            expected_total,
            out=np.full_like(expected_total, float(prior_mt)),
            where=expected_total > 0,
        )
        error_rate[clone_index] = np.maximum(
            np.clip(1.0 - consensus_support_fraction, 0.0, 1.0),
            0.0,
        )

    result = {
        "clone_names": np.asarray(clone_names, dtype=str),
        "mutation_names": np.asarray(adata_mut.var_names, dtype=str),
        "status": status,
        "error_rate": np.clip(error_rate, 0.0, 1.0),
        "posterior_mt": posterior_mt,
        "posterior_wt": posterior_wt,
        "log_odds": log_odds,
        "parameters": {
            "clone_key": clone_key,
            "imputed_cell_weight": float(imputed_cell_weight),
            "prior_mt": float(prior_mt),
            "status_rule": "reliability_weighted_MT_WT_vote",
            "likelihood_role": "error_estimation_only",
        },
    }
    adata_mut.uns[store_key] = result
    return result


def score_clone_direction(
    parent_status: np.ndarray,
    parent_error_rate: np.ndarray,
    child_status: np.ndarray,
    child_error_rate: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Score the error cost of one irreversible parent-to-child direction."""
    parent_status = np.asarray(parent_status, dtype=int)
    child_status = np.asarray(child_status, dtype=int)
    parent_error_rate = np.asarray(parent_error_rate, dtype=float)
    child_error_rate = np.asarray(child_error_rate, dtype=float)
    arrays = (parent_status, child_status, parent_error_rate, child_error_rate)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Direction inputs must be one-dimensional.")
    if not all(array.shape == parent_status.shape for array in arrays):
        raise ValueError("Direction inputs must have identical shapes.")
    if not np.isin(parent_status, (-1, 1)).all() or not np.isin(child_status, (-1, 1)).all():
        raise ValueError("Statuses must contain only -1 and 1.")
    if not all(np.all(np.isfinite(error)) and np.all((error >= 0) & (error <= 1.0)) for error in (parent_error_rate, child_error_rate)):
        raise ValueError("Error rates must be finite and in [0, 1].")
    violations = (parent_status == 1) & (child_status == -1)
    # Correcting a violation means correcting one endpoint. Compare the two
    # endpoint log-error terms directly and choose the cheaper correction.
    parent_log_error = np.log(np.clip(parent_error_rate, eps, 1.0))
    child_log_error = np.log(np.clip(child_error_rate, eps, 1.0))
    site_cost = np.zeros(parent_status.shape, dtype=float)
    site_cost[violations] = -np.maximum(
        parent_log_error[violations],
        child_log_error[violations],
    )
    violation_indices = np.flatnonzero(violations)
    return {
        "score": float(-site_cost.sum()),
        "total_cost": float(site_cost.sum()),
        "n_violations": int(violations.sum()),
        "mean_violation_cost": float(site_cost[violations].mean()) if np.any(violations) else 0.0,
        "max_violation_cost": float(site_cost[violations].max()) if np.any(violations) else 0.0,
        "violating_mutation_indices": violation_indices,
        "site_cost": site_cost,
        "n_informative_mutations": int(parent_status.size),
        "normalized_cost": float(site_cost.sum() / max(parent_status.size, 1)),
    }


def infer_edge_direction(
    clone_a,
    clone_b,
    consensus_result: dict[str, Any],
    absolute_direction_margin: float = 0.5,
    relative_direction_margin: float = 0.1,
) -> dict[str, Any]:
    """Compare both orientations of an existing clone edge."""
    if absolute_direction_margin < 0 or relative_direction_margin < 0:
        raise ValueError("Direction margins must be nonnegative.")
    names = np.asarray(consensus_result["clone_names"]).astype(str)
    lookup = {name: index for index, name in enumerate(names)}
    if str(clone_a) in lookup:
        a = lookup[str(clone_a)]
    elif isinstance(clone_a, (int, np.integer)):
        a = int(clone_a)
    else:
        a = -1
    if str(clone_b) in lookup:
        b = lookup[str(clone_b)]
    elif isinstance(clone_b, (int, np.integer)):
        b = int(clone_b)
    else:
        b = -1
    if not (0 <= a < len(names) and 0 <= b < len(names)):
        raise KeyError("Both clone identifiers must exist in consensus_result.")
    status = np.asarray(consensus_result["status"], dtype=int)
    error = np.asarray(consensus_result["error_rate"], dtype=float)
    score_ab = score_clone_direction(status[a], error[a], status[b], error[b])
    score_ba = score_clone_direction(status[b], error[b], status[a], error[a])
    cost_ab, cost_ba = score_ab["total_cost"], score_ba["total_cost"]
    difference = cost_ba - cost_ab
    absolute_difference = abs(difference)
    relative_difference = absolute_difference / max(cost_ab, cost_ba, 1e-12)
    accepted = absolute_difference >= absolute_direction_margin and relative_difference >= relative_direction_margin
    if not accepted:
        direction = "ambiguous"
    elif cost_ab < cost_ba:
        direction = "A->B"
    else:
        direction = "B->A"
    return {
        "source": clone_a,
        "target": clone_b,
        "direction": direction,
        "cost_ab": float(cost_ab),
        "cost_ba": float(cost_ba),
        "cost_difference": float(difference),
        "relative_cost_difference": float(relative_difference),
        "n_violations_ab": score_ab["n_violations"],
        "n_violations_ba": score_ba["n_violations"],
        "A_to_B": score_ab,
        "B_to_A": score_ba,
    }


def _consensus_matrix_from_key(adata_mut, mutation_key: str | None):
    """Resolve the configured mutation matrix source."""
    if mutation_key is None or mutation_key == "X":
        matrix = adata_mut.X
    elif mutation_key in adata_mut.layers:
        matrix = adata_mut.layers[mutation_key]
    elif mutation_key in adata_mut.obsm:
        matrix = adata_mut.obsm[mutation_key]
    else:
        raise KeyError(
            f"Mutation matrix key {mutation_key!r} was not found in "
            "adata.X, adata.layers, or adata.obsm."
        )
    _validate_mutation_values(matrix)
    if getattr(matrix, "ndim", None) != 2 or matrix.shape[0] != adata_mut.n_obs:
        raise ValueError("The mutation matrix must have one row per cell.")
    return matrix


def cal_consensus_profile(
    adata_mut: AnnData,
    clone_key: str = "clone",
    mutation_key: str | None = None,
    connectivity_key: str = "genetic_lineage_connectivity",
    consensus_method: str = "clone_boundary",
    outside_evidence_weight: float = 0.25,
    internal_evidence_scale: float = 2.0,
    outside_evidence_scale: float = 5.0,
    internal_reliability_weight: float = 0.7,
    indirect_reliability_discount: float = 0.5,
    mt_prior_alpha: float | None = None,
    wt_prior_beta: float | None = None,
    minimum_prior_observed_evidence: float = 2.0,
    prior_weight_scale: float = 2.0,
    prior_mt_fraction_threshold: float = 0.7,
    global_mt_prevalence: float = 0.10,
    prevalence_shrinkage: float = 20.0,
    prior_strength: float = 2.0,
    mt_probability_threshold: float = 0.70,
    minimum_posterior_reliability: float = 0.20,
    allow_uncertain_state: bool = False,
    wt_probability_threshold: float = 0.50,
    artificial_normal: bool = True,
    normal_label: str = "0",
    store_intermediate: bool = True,
    strong_internal_evidence_threshold: float = 3.0,
    reliability_evidence_threshold: float = 3.0,
    outside_reliability_penalty: float = 0.50,
    agreement_weight: float = 0.30,
    prior_mean_shrinkage_strength: float = 10.0,
    prior_probability_floor: float = 1e-3,
    mutation_baseline_shrinkage_strength: float = 5.0,
    minimum_mt_log_odds_enrichment: float = 0.0,
    copy: bool = False,
):
    """Infer clone mutation states with a graph-evidence Beta posterior.

    MT/WT/missing calls are encoded as 1, -1, and 0. For each clone, direct
    calls from clone cells and one-hop calls from directly connected outside
    boundary cells are aggregated into positive Beta evidence. Outside cells
    are counted once per clone using their capped aggregate boundary weight.
    No multihop diffusion or imputation is performed. The posterior probability
    determines the reported state, while reliability separately measures the
    amount and agreement of internal and boundary evidence.
    """
    if copy:
        adata_mut = adata_mut.copy()
    if clone_key not in adata_mut.obs:
        raise KeyError(f"adata.obs[{clone_key!r}] is missing.")
    if connectivity_key not in adata_mut.obsp:
        raise KeyError(f"adata.obsp[{connectivity_key!r}] is missing.")
    if consensus_method != "clone_boundary":
        raise ValueError("consensus_method must be 'clone_boundary'.")
    if not 0.0 <= float(outside_evidence_weight) <= 1.0:
        raise ValueError("outside_evidence_weight must be in [0, 1].")
    if float(strong_internal_evidence_threshold) <= 0.0:
        raise ValueError("strong_internal_evidence_threshold must be positive.")
    if float(reliability_evidence_threshold) <= 0.0:
        raise ValueError("reliability_evidence_threshold must be positive.")
    if not 0.0 <= float(outside_reliability_penalty) <= 1.0:
        raise ValueError("outside_reliability_penalty must be in [0, 1].")
    if not 0.0 <= float(agreement_weight) <= 1.0:
        raise ValueError("agreement_weight must be in [0, 1].")
    if float(internal_evidence_scale) <= 0.0 or float(outside_evidence_scale) <= 0.0:
        raise ValueError("Evidence scales must be positive.")
    if not 0.0 <= float(internal_reliability_weight) <= 1.0:
        raise ValueError("internal_reliability_weight must be in [0, 1].")
    if not 0.0 <= float(indirect_reliability_discount) <= 1.0:
        raise ValueError("indirect_reliability_discount must be in [0, 1].")
    if float(minimum_prior_observed_evidence) < 0.0:
        raise ValueError("minimum_prior_observed_evidence must be nonnegative.")
    if float(prior_weight_scale) <= 0.0:
        raise ValueError("prior_weight_scale must be positive.")
    if not 0.0 <= float(prior_mt_fraction_threshold) <= 1.0:
        raise ValueError("prior_mt_fraction_threshold must be in [0, 1].")
    if not 0.0 < float(global_mt_prevalence) < 1.0:
        raise ValueError("global_mt_prevalence must be in (0, 1).")
    if float(prevalence_shrinkage) < 0.0:
        raise ValueError("prevalence_shrinkage must be nonnegative.")
    if float(prior_strength) <= 0.0:
        raise ValueError("prior_strength must be positive.")
    if float(prior_mean_shrinkage_strength) < 0.0:
        raise ValueError("prior_mean_shrinkage_strength must be nonnegative.")
    if not 0.0 < float(prior_probability_floor) < 0.5:
        raise ValueError("prior_probability_floor must be in (0, 0.5).")
    if float(mutation_baseline_shrinkage_strength) < 0.0:
        raise ValueError(
            "mutation_baseline_shrinkage_strength must be nonnegative."
        )
    if (mt_prior_alpha is None) != (wt_prior_beta is None):
        raise ValueError(
            "mt_prior_alpha and wt_prior_beta must both be provided or both be None."
        )
    for prior_name, prior_value in (
        ("mt_prior_alpha", mt_prior_alpha),
        ("wt_prior_beta", wt_prior_beta),
    ):
        if prior_value is None:
            continue
        prior_array = np.asarray(prior_value, dtype=float)
        if prior_array.ndim != 0 or not np.all(np.isfinite(prior_array)):
            raise ValueError(f"{prior_name} must be a finite scalar.")
        if np.any(prior_array <= 0.0):
            raise ValueError(f"{prior_name} must be positive when provided.")
    if not 0.5 <= float(mt_probability_threshold) <= 1.0:
        raise ValueError("mt_probability_threshold must be in [0.5, 1].")
    if not 0.0 <= float(minimum_posterior_reliability) <= 1.0:
        raise ValueError("minimum_posterior_reliability must be in [0, 1].")
    if not 0.0 <= float(wt_probability_threshold) <= 0.5:
        raise ValueError("wt_probability_threshold must be in [0, 0.5].")
    if float(wt_probability_threshold) >= float(mt_probability_threshold):
        raise ValueError("wt_probability_threshold must be below mt_probability_threshold.")

    labels = adata_mut.obs[clone_key]
    if labels.isna().any():
        raise ValueError("Clone labels cannot contain missing values.")
    if isinstance(labels.dtype, pd.CategoricalDtype):
        present = {str(value) for value in labels.tolist()}
        clone_values = [
            str(value) for value in labels.cat.categories if str(value) in present
        ]
    else:
        clone_values = sorted({str(value) for value in labels.tolist()})
        if clone_values and all(value.lstrip("-").isdigit() for value in clone_values):
            clone_values = sorted(clone_values, key=lambda value: int(value))
    matrix = _consensus_matrix_from_key(adata_mut, mutation_key)
    n_cells, n_mutations = matrix.shape
    if n_cells != adata_mut.n_obs:
        raise ValueError("Mutation matrix must have one row per cell.")
    mutation_names = np.asarray(
        adata_mut.var_names
        if len(adata_mut.var_names) == n_mutations
        else [f"mutation_{i}" for i in range(n_mutations)],
        dtype=str,
    )

    # Establish clone order from raw mutation burden before any consensus
    # evidence is calculated. This is the stable biological order used to
    # resolve otherwise index-dependent downstream ties.
    if issparse(matrix):
        matrix = matrix.tocsr()
        raw_tmb_cells = np.bincount(
            np.repeat(np.arange(n_cells), np.diff(matrix.indptr)),
            weights=(matrix.data == 1).astype(float),
            minlength=n_cells,
        )
    else:
        raw_tmb_cells = np.sum(matrix == 1, axis=1).astype(float)
    preliminary_names = np.asarray(clone_values, dtype=str)
    preliminary_labels = np.asarray(labels.astype(str), dtype=str)
    preliminary_codes = np.asarray(
        [np.flatnonzero(preliminary_names == value)[0] for value in preliminary_labels],
        dtype=int,
    )
    preliminary_clone_sizes = np.bincount(
        preliminary_codes,
        minlength=preliminary_names.size,
    ).astype(float)
    raw_tmb_clone_sum = np.bincount(
        preliminary_codes,
        weights=raw_tmb_cells,
        minlength=preliminary_names.size,
    )
    raw_tmb_clone = np.divide(
        raw_tmb_clone_sum,
        preliminary_clone_sizes,
        out=np.zeros_like(raw_tmb_clone_sum),
        where=preliminary_clone_sizes > 0.0,
    )
    raw_order = np.argsort(raw_tmb_clone, kind="stable")
    clone_names = preliminary_names[raw_order]
    raw_tmb_clone = raw_tmb_clone[raw_order]
    label_values = preliminary_labels
    clone_codes = np.asarray(
        [np.flatnonzero(clone_names == value)[0] for value in label_values],
        dtype=int,
    )
    n_clones = clone_names.size

    if issparse(matrix):
        matrix = matrix.tocsr()
        if matrix.data.size and (
            not np.isfinite(matrix.data).all()
            or not np.isin(matrix.data, [-1, 0, 1]).all()
        ):
            raise ValueError("Mutation values must be finite and encoded as -1, 0, or 1.")
        mt_indicator = matrix.copy()
        mt_indicator.data = (mt_indicator.data == 1).astype(np.float32)
        mt_indicator.eliminate_zeros()
        wt_indicator = matrix.copy()
        wt_indicator.data = (wt_indicator.data == -1).astype(np.float32)
        wt_indicator.eliminate_zeros()
        observed_indicator = mt_indicator + wt_indicator
    else:
        matrix = np.asarray(matrix)
        if not np.isfinite(matrix).all() or not np.isin(matrix, [-1, 0, 1]).all():
            raise ValueError("Mutation values must be finite and encoded as -1, 0, or 1.")
        mt_indicator = (matrix == 1).astype(np.float32)
        wt_indicator = (matrix == -1).astype(np.float32)
        observed_indicator = mt_indicator + wt_indicator

    graph = csr_matrix(adata_mut.obsp[connectivity_key], dtype=float)
    if graph.shape != (n_cells, n_cells):
        raise ValueError("Connectivity must have shape (n_cells, n_cells).")
    if graph.data.size and (
        not np.isfinite(graph.data).all() or np.any(graph.data < 0.0)
    ):
        raise ValueError("Connectivity must contain finite nonnegative weights.")
    graph = ((graph + graph.T) * 0.5).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    def _dense(value):
        return np.asarray(value.toarray() if issparse(value) else value, dtype=float)

    clone_sizes = np.bincount(clone_codes, minlength=n_clones).astype(float)
    n_mt_observed = np.zeros((n_clones, n_mutations), dtype=float)
    n_wt_observed = np.zeros_like(n_mt_observed)
    outside_mt_evidence = np.zeros_like(n_mt_observed)
    outside_wt_evidence = np.zeros_like(n_mt_observed)

    for clone_code in range(n_clones):
        internal = np.flatnonzero(clone_codes == clone_code)
        if internal.size == 0:
            continue
        internal_mt = _dense(mt_indicator[internal]).sum(axis=0)
        internal_wt = _dense(wt_indicator[internal]).sum(axis=0)
        n_mt_observed[clone_code] = internal_mt
        n_wt_observed[clone_code] = internal_wt

        boundary_sum = np.asarray(graph[internal].sum(axis=0)).ravel()
        boundary_sum[clone_codes == clone_code] = 0.0
        outside = boundary_sum > 0.0
        if not np.any(outside):
            continue
        outside_weight = np.minimum(boundary_sum[outside], 1.0)
        outside_mt_evidence[clone_code] = np.asarray(
            mt_indicator[outside].T @ outside_weight
        ).ravel()
        outside_wt_evidence[clone_code] = np.asarray(
            wt_indicator[outside].T @ outside_weight
        ).ravel()

    internal_observed_evidence = n_mt_observed + n_wt_observed
    outside_observed_evidence = outside_mt_evidence + outside_wt_evidence
    # Estimate one weak global prior and a separate mutation-specific baseline
    # using only sufficiently observed direct internal clone evidence.
    eligible = internal_observed_evidence >= float(
        minimum_prior_observed_evidence
    )
    mutation_mt_evidence = np.sum(
        np.where(eligible, n_mt_observed, 0.0),
        axis=0,
    )
    mutation_wt_evidence = np.sum(
        np.where(eligible, n_wt_observed, 0.0),
        axis=0,
    )
    mutation_total_evidence = mutation_mt_evidence + mutation_wt_evidence
    prior_units = np.sum(eligible, axis=0).astype(float)
    global_internal_mt_evidence = float(np.sum(mutation_mt_evidence))
    global_internal_wt_evidence = float(np.sum(mutation_wt_evidence))
    global_evidence_total = (
        global_internal_mt_evidence + global_internal_wt_evidence
    )
    if global_evidence_total > 1e-12:
        global_mu = global_internal_mt_evidence / global_evidence_total
    else:
        global_mu = float(global_mt_prevalence)
    global_mu = float(np.clip(
        global_mu,
        prior_probability_floor,
        1.0 - prior_probability_floor,
    ))
    raw_mutation_mt_fraction = np.divide(
        mutation_mt_evidence,
        mutation_total_evidence,
        out=np.full(n_mutations, global_mu, dtype=float),
        where=mutation_total_evidence > 1e-12,
    )
    baseline_shrinkage_weight = prior_units / (
        prior_units + float(mutation_baseline_shrinkage_strength)
    )
    mutation_mt_baseline = np.clip(
        baseline_shrinkage_weight * raw_mutation_mt_fraction
        + (1.0 - baseline_shrinkage_weight) * global_mu,
        prior_probability_floor,
        1.0 - prior_probability_floor,
    )

    prior_estimated = mt_prior_alpha is None and wt_prior_beta is None
    if prior_estimated:
        mt_prior_alpha = float(prior_strength) * global_mu
        wt_prior_beta = float(prior_strength) * (1.0 - global_mu)
        print("Estimating empirical Bayes mutation prior...")
        print(f"Eligible clone-mutation pairs: {int(np.sum(eligible))}")
        print(f"Estimated global MT prevalence: {global_mu:.2f}")
        print(f"Using mt_prior_alpha = {mt_prior_alpha:.2f}")
        print(f"Using wt_prior_beta = {wt_prior_beta:.2f}")
    else:
        mt_prior_alpha = float(np.asarray(mt_prior_alpha, dtype=float))
        wt_prior_beta = float(np.asarray(wt_prior_beta, dtype=float))
    global_prior_mean = (
        global_mu
        if prior_estimated
        else mt_prior_alpha / (mt_prior_alpha + wt_prior_beta)
    )
    prior_metadata = {
        "estimated": bool(prior_estimated),
        "global_prior_mt_mean": global_prior_mean,
        "global_internal_mt_fraction": global_mu,
        "prior_concentration": float(prior_strength),
        "prior_probability_floor": float(prior_probability_floor),
        "minimum_prior_observed_evidence": float(
            minimum_prior_observed_evidence
        ),
        "mutation_baseline_shrinkage_strength": float(
            mutation_baseline_shrinkage_strength
        ),
        "mutation_prior_units": prior_units,
        "mutation_prior_mt_evidence": mutation_mt_evidence,
        "mutation_prior_wt_evidence": mutation_wt_evidence,
        "raw_mutation_mt_fraction": raw_mutation_mt_fraction,
        "mutation_mt_baseline": mutation_mt_baseline,
        "mt_prior_alpha": float(mt_prior_alpha),
        "wt_prior_beta": float(wt_prior_beta),
    }
    adata_mut.uns["consensus_prior"] = prior_metadata

    # Combine evidence internal-first. Strong internal evidence is authoritative
    # for the posterior; boundary evidence remains available for reliability.
    internal_total = internal_observed_evidence
    outside_total = outside_observed_evidence
    outside_weight = float(outside_evidence_weight)
    strong_internal = internal_total >= float(strong_internal_evidence_threshold)
    weak_internal = (internal_total > 0.0) & ~strong_internal
    no_internal = internal_total <= 0.0

    effective_internal_mt = n_mt_observed.copy()
    effective_internal_wt = n_wt_observed.copy()
    used_outside_mt = np.zeros_like(outside_mt_evidence)
    used_outside_wt = np.zeros_like(outside_wt_evidence)
    supplement_outside = weak_internal | no_internal
    used_outside_mt[supplement_outside] = outside_weight * outside_mt_evidence[supplement_outside]
    used_outside_wt[supplement_outside] = outside_weight * outside_wt_evidence[supplement_outside]
    effective_outside_mt = used_outside_mt
    effective_outside_wt = used_outside_wt
    effective_mt = effective_internal_mt + effective_outside_mt
    effective_wt = effective_internal_wt + effective_outside_wt
    effective_total = effective_mt + effective_wt

    # These aliases are retained for existing filtering and diagnostics.
    total_mt_evidence = effective_mt
    total_wt_evidence = effective_wt
    mt_effective_evidence = effective_mt.copy()
    wt_effective_evidence = effective_wt.copy()
    mt_positive_evidence = effective_mt.copy()
    wt_positive_evidence = effective_wt.copy()
    total_positive_evidence = mt_positive_evidence + wt_positive_evidence

    alpha = float(mt_prior_alpha) + effective_mt
    beta = float(wt_prior_beta) + effective_wt
    posterior_concentration = alpha + beta
    prior_concentration = float(prior_strength)
    prior_mean = float(mt_prior_alpha) / (
        float(mt_prior_alpha) + float(wt_prior_beta)
    )
    posterior_p_mt = np.divide(
        alpha,
        posterior_concentration,
        out=np.full_like(alpha, prior_mean),
        where=posterior_concentration > 0.0,
    )
    posterior_p_mt = np.clip(posterior_p_mt, 0.0, 1.0)
    posterior_p_wt = 1.0 - posterior_p_mt
    posterior_purity = np.clip(
        np.abs(posterior_p_mt - posterior_p_wt),
        0.0,
        1.0,
    )
    evidence_sufficiency = np.minimum(
        1.0,
        effective_total / float(reliability_evidence_threshold),
    )
    effective_outside_total = effective_outside_mt + effective_outside_wt
    outside_fraction = np.divide(
        effective_outside_total,
        effective_total,
        out=np.zeros_like(effective_total),
        where=effective_total > 1e-12,
    )
    source_quality = np.clip(
        1.0 - float(outside_reliability_penalty) * outside_fraction,
        0.0,
        1.0,
    )
    has_internal = internal_total > 1e-12
    has_outside = outside_total > 1e-12
    internal_mt_fraction = np.divide(
        n_mt_observed,
        internal_total,
        out=np.full_like(internal_total, np.nan),
        where=has_internal,
    )
    outside_mt_fraction = np.divide(
        outside_mt_evidence,
        outside_total,
        out=np.full_like(outside_total, np.nan),
        where=has_outside,
    )
    both_channels = has_internal & has_outside
    agreement = np.ones_like(effective_total)
    agreement[both_channels] = np.clip(
        1.0 - np.abs(
            internal_mt_fraction[both_channels]
            - outside_mt_fraction[both_channels]
        ),
        0.0,
        1.0,
    )
    agreement_factor = np.clip(
        1.0 - float(agreement_weight) * (1.0 - agreement),
        0.0,
        1.0,
    )
    posterior_reliability = np.clip(
        posterior_purity
        * evidence_sufficiency
        * source_quality
        * agreement_factor,
        0.0,
        1.0,
    )
    evidence_regime = np.full(
        effective_total.shape,
        "outside_only",
        dtype="<U32",
    )
    evidence_regime[strong_internal] = "strong_internal"
    evidence_regime[weak_internal] = "weak_internal_plus_outside"
    inference_source = np.full(
        effective_total.shape,
        "prior_only",
        dtype="<U20",
    )
    inference_source[strong_internal] = "internal_only"
    inference_source[weak_internal & has_outside] = "internal_and_outside"
    inference_source[weak_internal & ~has_outside] = "internal_only"
    inference_source[~has_internal & has_outside] = "outside_only"
    posterior_log_odds = np.log(
        np.clip(posterior_p_mt, 1e-12, 1.0 - 1e-12)
        / np.clip(posterior_p_wt, 1e-12, 1.0 - 1e-12)
    )
    baseline_log_odds = np.log(
        np.clip(mutation_mt_baseline, 1e-12, 1.0 - 1e-12)
        / np.clip(1.0 - mutation_mt_baseline, 1e-12, 1.0 - 1e-12)
    )
    mt_log_odds_enrichment = posterior_log_odds - baseline_log_odds[None, :]
    consensus_status = np.where(
        (
            (posterior_p_mt >= float(mt_probability_threshold))
            & (mt_log_odds_enrichment >= float(minimum_mt_log_odds_enrichment))
        ),
        1,
        -1,
    ).astype(np.int8)
    consensus_confidence = posterior_reliability.copy()
    consensus_error = 1.0 - consensus_confidence
    heterogeneity_support = (
        np.divide(
            2.0 * np.minimum(mt_positive_evidence, wt_positive_evidence),
            total_positive_evidence,
            out=np.zeros_like(total_positive_evidence),
            where=total_positive_evidence > 0.0,
        )
        * posterior_reliability
    )
    mt_call_strength = np.clip(
        (posterior_p_mt - float(mt_probability_threshold))
        / (1.0 - float(mt_probability_threshold) + 1e-12),
        0.0,
        1.0,
    )
    mt_call_confidence = posterior_reliability * mt_call_strength

    # Retain mutation columns with at least one reliable MT call. The complete
    # posterior and status arrays preserve low-reliability MT states for audit.
    reliable_mt_mask = (
        (posterior_p_mt >= float(mt_probability_threshold))
        & (posterior_reliability >= float(minimum_posterior_reliability))
    )
    filtered_mask = np.any(reliable_mt_mask, axis=0)
    filtered_mutations = mutation_names[filtered_mask]
    clone_index = pd.Index(clone_names, name="clone")
    filtered_index = pd.Index(filtered_mutations, name="mutation")
    filtered_profile = pd.DataFrame(
        consensus_status[:, filtered_mask], index=clone_index, columns=filtered_index
    )
    filtered_high_confidence = pd.DataFrame(
        np.where(
            consensus_confidence[:, filtered_mask]
            >= float(minimum_posterior_reliability),
            consensus_status[:, filtered_mask],
            0,
        ).astype(np.int8),
        index=clone_index,
        columns=filtered_index,
    )
    profile_labels = clone_names.copy()
    profile = filtered_profile
    if artificial_normal:
        normal_row = pd.DataFrame(
            -np.ones((1, filtered_index.size), dtype=np.int8),
            index=pd.Index([str(normal_label)], name="clone"),
            columns=filtered_index,
        )
        profile = pd.concat((normal_row, filtered_profile))
        filtered_high_confidence = pd.concat((normal_row, filtered_high_confidence))
        profile_labels = np.concatenate(([str(normal_label)], clone_names))

    for key in (
        "consensus_mt_probability", "consensus_wt_probability",
        "consensus_mt_probability_full", "consensus_wt_probability_full",
        "consensus_theta", "consensus_theta_full",
        "consensus_probability", "consensus_probability_full",
        "consensus_error_rate", "consensus_error_rate_full",
        "consensus_error_rate_threshold", "consensus_confidence_full",
        "consensus_mt_evidence_full", "consensus_wt_evidence_full",
    ):
        adata_mut.uns.pop(key, None)

    adata_mut.uns["consensus_profile"] = profile
    adata_mut.uns["consensus_profile_high_confidence"] = filtered_high_confidence
    adata_mut.uns["consensus_clone_labels"] = clone_names.astype(str)
    adata_mut.uns["consensus_profile_labels"] = profile_labels.astype(str)
    adata_mut.uns["raw_TMB_clone"] = raw_tmb_clone.astype(float)
    adata_mut.uns["filtered_mutations"] = filtered_mutations.astype(str)
    adata_mut.uns["consensus_profile_params"] = {
        "method": "clone_boundary_beta_posterior",
        "consensus_method": consensus_method,
        "clone_key": clone_key,
        "clone_order": "ascending_raw_TMB",
        "raw_TMB_clone": raw_tmb_clone.astype(float),
        "mutation_key": mutation_key or "X",
        "connectivity_key": connectivity_key,
        "outside_evidence_weight": float(outside_evidence_weight),
        "strong_internal_evidence_threshold": float(
            strong_internal_evidence_threshold
        ),
        "reliability_evidence_threshold": float(
            reliability_evidence_threshold
        ),
        "outside_reliability_penalty": float(outside_reliability_penalty),
        "agreement_weight": float(agreement_weight),
        "internal_evidence_scale": float(internal_evidence_scale),
        "outside_evidence_scale": float(outside_evidence_scale),
        "internal_reliability_weight": float(internal_reliability_weight),
        "indirect_reliability_discount": float(indirect_reliability_discount),
        "mt_prior_alpha": np.asarray(mt_prior_alpha, dtype=float),
        "wt_prior_beta": np.asarray(wt_prior_beta, dtype=float),
        "prior_mt_probability": prior_mean,
        "prior_estimated": bool(prior_estimated),
        "minimum_prior_observed_evidence": float(
            minimum_prior_observed_evidence
        ),
        "prior_weight_scale": float(prior_weight_scale),
        "prior_mt_fraction_threshold": float(prior_mt_fraction_threshold),
        "global_mt_prevalence": float(global_mt_prevalence),
        "prevalence_shrinkage": float(prevalence_shrinkage),
        "prior_strength": float(prior_strength),
        "prior_concentration": float(prior_strength),
        "prior_mean_shrinkage_strength": float(
            prior_mean_shrinkage_strength
        ),
        "prior_probability_floor": float(prior_probability_floor),
        "mutation_baseline_shrinkage_strength": float(
            mutation_baseline_shrinkage_strength
        ),
        "minimum_mt_log_odds_enrichment": float(
            minimum_mt_log_odds_enrichment
        ),
        "mt_probability_threshold": float(mt_probability_threshold),
        "minimum_posterior_reliability": float(minimum_posterior_reliability),
        "allow_uncertain_state": bool(allow_uncertain_state),
        "wt_probability_threshold": float(wt_probability_threshold),
        "posterior_evidence_mode": "positive_state_evidence_only",
        "cross_state_disagreement": False,
        "multihop_diffusion": False,
        "artificial_normal": bool(artificial_normal),
    }
    adata_mut.uns["clone_consensus"] = {
        "clone_names": clone_names.astype(str),
        "mutation_names": mutation_names.astype(str),
        "status": consensus_status,
        "posterior_p_mt": posterior_p_mt.astype(np.float32),
        "posterior_p_wt": posterior_p_wt.astype(np.float32),
        "posterior_reliability": posterior_reliability.astype(np.float32),
        "mutation_mt_baseline": mutation_mt_baseline.astype(np.float32),
        "mt_log_odds_enrichment": mt_log_odds_enrichment.astype(np.float32),
        "parameters": adata_mut.uns["consensus_profile_params"],
    }
    if copy:
        return adata_mut
    return None
