"""Genetic connectivity graph construction for EvoFATE."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph

if TYPE_CHECKING:
    from anndata import AnnData


def _resolve_kneighbors(n_cells: int, kneighbors: int | None = None) -> int:
    """Return adaptive or user-specified neighbor count as a safe integer."""
    if n_cells < 1:
        raise ValueError("Expected at least one cell.")
    if kneighbors is None:
        kneighbors = min(max(int(round(0.01 * n_cells)), 10), 200)
    return min(max(int(kneighbors), 1), n_cells)



def _resolve_workers(workers: int | None = 1) -> int:
    """Return a safe worker count; use -1 for all available CPUs."""
    if workers is None:
        return 1
    workers = int(workers)
    if workers == -1:
        return max(os.cpu_count() or 1, 1)
    if workers < 1:
        raise ValueError("`workers` must be a positive integer, None, or -1.")
    return workers



def cal_genetic_connectivities(
    adata_mut: AnnData,
    thred: float = 0.05,
    topology: str = "SNV",
    kneighbors: int | None = None,
    n_components: int = 50,
    snn_threshold: float = 0.05,
    weight_floor: float = 0.01,
    support_tau: float = 5.0,
    verbose: bool = True,
    dim: int = 8,
    workers: int | None = 1,
) -> None:
    """
    Calculate the genetic weighted connectivity graph.

    This function constructs a shared-nearest-neighbor-style cell-cell
    connectivity graph from the MT-based representation and stores it directly as
    `adata_mut.obsp['genetic_lineage_connectivity']`.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object containing mutation matrix in `.X`.
        Values should be: 1 (mutant), -1 (wildtype), 0 (missing).
    thred : float, default=0.05
        Threshold for selecting principal components based on explained variance ratio.
    topology : {'SNV', 'CNV'}, default='SNV'
        Data source used to build the local neighborhood topology. 'SNV' uses
        `adata_mut.X`; 'CNV' uses `adata_mut.obsm['CNV']`.
    kneighbors : int, optional
        Number of neighbors for the broad initial KNN graph. If omitted,
        defaults to ``min(max(round(0.01 * n_cells), 10), 200)``.
    n_components : int, default=50
        Maximum number of SVD components used to build the initial topology
        embedding before KNN graph construction.
    snn_threshold : float, default=0.05
        Minimum MT-Jaccard overlap required for a candidate edge. The
        parameter name is retained for API compatibility.
    weight_floor : float, default=0.01
        Minimum retained edge weight after MT-overlap threshold filtering.
    support_tau : float, default=5.0
        Shrinkage strength for MT-overlap weights.
    verbose : bool, default=True
        Whether to print training progress.
    dim : int, default=8
        Number of spectral graph components stored in
        `adata_mut.obsm['X_genetic']`.
    workers : int, optional, default=1
        Number of workers used for KNN construction when supported. Use -1 for
        all CPUs.
    Returns
    -------
    None
        Modifies `adata_mut` in place:
        - `.uns['kneighbors']`: Number of neighbors used
        - `.uns['dim']`: Default downstream embedding dimension
        - `.obsp['connectivities_initial']`: Initial connectivity matrix
        - `.obsp['genetic_lineage_connectivity']`: Weighted mutation-derived connectivity map
        - `.obsm['X_genetic']`: Spectral components of the active
          weighted connectivity graph
    """
    kneighbors_resolved = _resolve_kneighbors(adata_mut.shape[0], kneighbors)
    workers_resolved = _resolve_workers(workers)
    _cal_weighted_connectivities(
        adata_mut,
        topology=topology,
        thred=thred,
        kneighbors=kneighbors_resolved,
        n_components=n_components,
        snn_threshold=snn_threshold,
        weight_floor=weight_floor,
        support_tau=support_tau,
        verbose=verbose,
        workers=workers_resolved,
    )
    if verbose:
        graph = adata_mut.obsp["connectivities_initial"]
        print(
            "[Genetic] _cal_weighted_connectivities complete: "
            f"topology={topology.upper()}, k={kneighbors_resolved}, "
            f"edges={graph.nnz}, workers={workers_resolved}"
        )

    _cal_genetic_embedding_from_connectivities(
        adata_mut,
        dim=dim,
        connectivity_key="connectivities_initial",
    )
    if verbose:
        embedding = adata_mut.obsm["X_genetic"]
        print(
            "[Genetic] _cal_genetic_embedding_from_connectivities complete: "
            f"embedding_shape={embedding.shape}"
        )



def _cal_weighted_connectivities(
    adata_mut: AnnData,
    topology: str = "SNV",
    thred: float = 0.05,
    kneighbors: int | None = None,
    n_components: int = 50,
    snn_threshold: float = 0.05,
    weight_floor: float = 0.01,
    support_tau: float = 5.0,
    verbose: bool = True,
    workers: int | None = 1,
) -> None:
    """
    Calculate weighted SNV connectivities using either SNV or SNV+CNV topology.

    SNV components are always calculated from the MT-overlap representation of
    `adata_mut.X`. If topology is CNV, CNV components from `adata_mut.obsm['CNV']`
    are concatenated to the SNV components before KNN construction. Edge weights
    are shrunken MT-overlap scores between KNN neighbors.
    """
    kneighbors = _resolve_kneighbors(adata_mut.shape[0], kneighbors)
    workers = _resolve_workers(workers)
    topology = topology.upper()

    if topology == "CNV":
        if "CNV" not in adata_mut.obsm:
            raise KeyError("topology='CNV' requires `adata_mut.obsm['CNV']`.")
        cnv_matrix = adata_mut.obsm["CNV"]
        if cnv_matrix.shape[0] != adata_mut.shape[0]:
            raise ValueError(
                "`adata_mut.obsm['CNV']` must have one row per cell in "
                "`adata_mut.X`."
            )
        knn_graph, topology_embedding = _cal_knn_from_cnv(
            snv=adata_mut.X,
            cnv_matrix=cnv_matrix,
            kneighbors=kneighbors,
            thred=thred,
            n_components=n_components,
            verbose=verbose,
            workers=workers,
        )
    elif topology == "SNV":
        knn_graph, topology_embedding = _cal_knn_from_snv(
            adata_mut.X,
            kneighbors=kneighbors,
            thred=thred,
            n_components=n_components,
            verbose=verbose,
            workers=workers,
        )
    else:
        raise ValueError("topology must be either 'SNV' or 'CNV'.")

    connectivities_initial = _build_weighted_connectivity_graph(
        knn_graph=knn_graph,
        snv=adata_mut.X,
        snn_threshold=snn_threshold,
        weight_floor=weight_floor,
        support_tau=support_tau,
        workers=workers,
    )

    adata_mut.uns["kneighbors"] = kneighbors
    adata_mut.uns["connectivity_topology"] = topology
    adata_mut.uns["connectivity_workers"] = workers
    adata_mut.uns["connectivity_mt_overlap_threshold"] = float(snn_threshold)
    adata_mut.uns["connectivity_support_tau"] = float(support_tau)
    adata_mut.obsp["connectivities_initial"] = connectivities_initial
    _set_weighted_connectivities(adata_mut, connectivities_initial)
    adata_mut.obsm["X_genetic_initial"] = topology_embedding



def _cal_genetic_embedding_from_connectivities(
    adata_mut: AnnData,
    dim: int = 8,
    connectivity_key: str = "genetic_lineage_connectivity",
) -> None:
    """
    Use a precomputed weighted graph as the genetic connectivity map and
    calculate spectral graph components as `obsm['X_genetic']`.
    """
    if connectivity_key not in adata_mut.obsp:
        raise KeyError(
            f"`adata_mut.obsp['{connectivity_key}']` is missing. "
            "Provide a weighted graph in `.obsp` or run "
            "`_cal_weighted_connectivities` first."
        )

    adata_mut.uns["dim"] = dim
    _set_weighted_connectivities(adata_mut, adata_mut.obsp[connectivity_key])
    spectral_embedding = _get_spectral_embedding_from_connectivities(
        adata_mut.obsp["genetic_lineage_connectivity"],
        dim=dim,
    )
    adata_mut.obsm["X_genetic"] = spectral_embedding
    adata_mut.uns["genetic_embedding_method"] = "spectral_connectivity"



def _set_weighted_connectivities(
    adata_mut: AnnData,
    weighted_graph: spmatrix | np.ndarray,
) -> None:
    """Store a weighted graph as the active cell-cell connectivity map."""
    if hasattr(weighted_graph, "tocsr"):
        connectivities = weighted_graph.tocsr().copy()
    else:
        connectivities = csr_matrix(weighted_graph)
    connectivities.setdiag(0.0)
    connectivities.eliminate_zeros()

    distances = connectivities.copy()
    if distances.data.size > 0:
        max_weight = max(float(distances.data.max()), 1e-12)
        distances.data = 1.0 - distances.data / max_weight
        distances.eliminate_zeros()

    adata_mut.obsp["genetic_lineage_connectivity"] = connectivities
    adata_mut.obsp["distances"] = distances
    adata_mut.uns["neighbors"] = {
        "connectivities_key": "genetic_lineage_connectivity",
        "distances_key": "distances",
        "params": {
            "method": "evofate_weighted_graph",
            "metric": "shrunken_mt_overlap",
        },
    }



def _get_spectral_embedding_from_connectivities(
    weighted_graph: spmatrix | np.ndarray,
    dim: int = 8,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calculate spectral graph components from a weighted connectivity map.

    The embedding is based on the smallest non-trivial eigenvectors of the
    normalized graph Laplacian. Components are padded with zeros when the graph
    has fewer informative dimensions than requested.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("`dim` must be a positive integer.")

    if hasattr(weighted_graph, "tocsr"):
        graph = weighted_graph.tocsr().astype(float, copy=True)
    else:
        graph = csr_matrix(weighted_graph, dtype=float)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("`weighted_graph` must be square.")

    n_cells = graph.shape[0]
    if n_cells == 0:
        return np.zeros((0, dim), dtype=float)
    if n_cells == 1:
        return np.zeros((1, dim), dtype=float)

    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    if graph.nnz == 0:
        return np.zeros((n_cells, dim), dtype=float)

    laplacian = csgraph_laplacian(graph, normed=True).astype(float)
    n_components = min(dim, n_cells - 1)

    if n_cells <= dim + 1:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        order = np.argsort(eigenvalues)
        components = eigenvectors[:, order[1 : n_components + 1]]
    else:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=dim + 1,
            which="SM",
            tol=1e-4,
        )
        order = np.argsort(eigenvalues)
        components = eigenvectors[:, order[1 : dim + 1]]

    components = np.asarray(components, dtype=float)
    for component_index in range(components.shape[1]):
        column = components[:, component_index]
        anchor = int(np.argmax(np.abs(column)))
        if column[anchor] < 0:
            components[:, component_index] *= -1.0

    components = components - components.mean(axis=0, keepdims=True)
    scale = components.std(axis=0, keepdims=True)
    components = np.divide(
        components,
        scale + eps,
        out=np.zeros_like(components),
        where=scale > eps,
    )

    if components.shape[1] < dim:
        padding = np.zeros((n_cells, dim - components.shape[1]), dtype=float)
        components = np.hstack((components, padding))
    return components[:, :dim]



def _cal_knn_from_cnv(
    snv: np.ndarray,
    cnv_matrix: np.ndarray,
    kneighbors: int,
    thred: float = 0.05,
    n_components: int = 50,
    verbose: bool = True,
    workers: int | None = 1,
) -> tuple[spmatrix, np.ndarray]:
    """
    Calculate a binary k-NN graph from concatenated SNV and CNV components.
    """
    snv_overlap = _get_normalized_snv_overlap(snv)
    snv_embedding_initial = _get_topology_embedding(
        snv_overlap,
        thred=thred,
        n_components=n_components,
        verbose=verbose,
    )
    cnv_embedding_initial = _get_topology_embedding(
        cnv_matrix,
        thred=thred,
        n_components=n_components,
        verbose=verbose,
    )
    combined_embedding_initial = np.hstack(
        (snv_embedding_initial, cnv_embedding_initial)
    )
    knn_graph = _kneighbors_graph(
        combined_embedding_initial,
        kneighbors=kneighbors,
        workers=workers,
    )
    return knn_graph, combined_embedding_initial



def _cal_knn_from_snv(
    snv: np.ndarray,
    kneighbors: int,
    thred: float = 0.05,
    n_components: int = 50,
    verbose: bool = True,
    workers: int | None = 1,
) -> tuple[spmatrix, np.ndarray]:
    """
    Calculate a binary k-NN graph from SNV-derived structural embeddings.
    """
    snv_overlap = _get_normalized_snv_overlap(snv)
    genetic_embedding_initial = _get_topology_embedding(
        snv_overlap,
        thred=thred,
        n_components=n_components,
        verbose=verbose,
    )
    knn_graph = _kneighbors_graph(
        genetic_embedding_initial,
        kneighbors=kneighbors,
        workers=workers,
    )
    return knn_graph, genetic_embedding_initial



def _kneighbors_graph(
    features: np.ndarray,
    kneighbors: int,
    workers: int | None = 1,
) -> spmatrix:
    """Build a KNN graph, using parallel workers when sklearn supports it."""
    workers = _resolve_workers(workers)
    try:
        return kneighbors_graph(
            features,
            n_neighbors=kneighbors,
            mode="connectivity",
            include_self=True,
            n_jobs=workers,
        )
    except TypeError as exc:
        if "n_jobs" not in str(exc):
            raise
        return kneighbors_graph(
            features,
            n_neighbors=kneighbors,
            mode="connectivity",
            include_self=True,
        )



def _build_weighted_connectivity_graph(
    knn_graph: spmatrix,
    snv: np.ndarray,
    snn_threshold: float = 0.05,
    weight_floor: float = 0.01,
    support_tau: float = 5.0,
    workers: int | None = 1,
) -> csr_matrix:
    """
    Recover weighted cell-cell edges with shrunken MT overlap.

    SNV coding:
        1 = mutant
       -1 = wildtype
        0 = missing

    Design:
        The initial KNN graph comes from the broad MT-based representation.
        The graph topology is kept fixed to the 1-hop KNN neighbors. Edges are
        reweighted by MT overlap, then removed if their MT-Jaccard weight is
        below `snn_threshold`.
    """
    n_cells = snv.shape[0]
    eps = 1e-12

    neighbor_graph = _get_one_hop_candidate_graph(knn_graph, n_cells)
    neighbor_pairs = _get_candidate_pairs(
        n_cells=n_cells,
        candidate_graph=neighbor_graph,
        use_all_pairs=False,
    )
    if neighbor_pairs.size == 0:
        return csr_matrix((n_cells, n_cells))

    rows = neighbor_pairs[:, 0]
    cols = neighbor_pairs[:, 1]
    weights, jaccard = _cal_shrunken_mt_overlap_for_pairs(
        snv=snv,
        pairs=neighbor_pairs,
        support_tau=support_tau,
        eps=eps,
    )
    threshold = float(snn_threshold)
    if threshold < 0:
        raise ValueError("`snn_threshold` must be non-negative.")
    keep_mask = jaccard >= threshold
    rows = rows[keep_mask]
    cols = cols[keep_mask]
    weights = weights[keep_mask]
    if rows.size == 0:
        return csr_matrix((n_cells, n_cells))

    weights = np.maximum(weights, weight_floor)
    raw_graph = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))
    graph = raw_graph.maximum(raw_graph.T)

    graph.setdiag(0.0)
    graph.eliminate_zeros()

    return graph



def _cal_shrunken_mt_overlap_for_pairs(
    snv: np.ndarray,
    pairs: np.ndarray,
    support_tau: float = 5.0,
    eps: float = 1e-12,
    chunk_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate MT-Jaccard and support-shrunken MT weights for cell pairs."""
    if support_tau < 0 or not np.isfinite(support_tau):
        raise ValueError("`support_tau` must be finite and nonnegative.")
    if pairs.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty

    snv_array = _as_dense_array(snv)
    mt = csr_matrix((snv_array == 1).astype(np.float32, copy=False))
    mt_counts = np.asarray(mt.sum(axis=1)).ravel().astype(float)
    rows = np.asarray(pairs[:, 0], dtype=np.int64)
    cols = np.asarray(pairs[:, 1], dtype=np.int64)
    shared = np.zeros(rows.size, dtype=float)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        left = mt[rows[start:stop]]
        right = mt[cols[start:stop]]
        shared[start:stop] = np.asarray(
            left.multiply(right).sum(axis=1)
        ).ravel()

    union = mt_counts[rows] + mt_counts[cols] - shared
    jaccard = np.divide(
        shared,
        union,
        out=np.zeros_like(shared),
        where=union > eps,
    )
    weights = np.divide(
        jaccard * union,
        union + float(support_tau),
        out=np.zeros_like(shared),
        where=union > eps,
    )
    return weights, jaccard



def _get_one_hop_candidate_graph(
    knn_graph: spmatrix | np.ndarray,
    n_cells: int,
) -> csr_matrix:
    """Return sparse undirected 1-hop KNN candidate edges."""
    one_hop = _as_csr_square(knn_graph, n_cells).astype(float, copy=True)
    one_hop.data = np.ones_like(one_hop.data, dtype=float)
    one_hop.setdiag(0.0)
    one_hop.eliminate_zeros()
    one_hop = one_hop.maximum(one_hop.T).tocsr()
    one_hop.data = np.ones_like(one_hop.data, dtype=float)
    return one_hop



def _get_candidate_pairs(
    n_cells: int,
    candidate_graph: spmatrix | np.ndarray | None = None,
    candidate_pairs: np.ndarray | list[tuple[int, int]] | None = None,
    use_all_pairs: bool = True,
    fallback_graph: spmatrix | np.ndarray | None = None,
) -> np.ndarray:
    """Return unique unordered candidate cell pairs."""
    if candidate_pairs is not None:
        pairs = np.asarray(candidate_pairs, dtype=int)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        pairs = pairs.reshape(-1, 2)
    elif candidate_graph is not None:
        graph = _as_csr_square(candidate_graph, n_cells)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        rows, cols = graph.nonzero()
        pairs = np.column_stack((rows, cols))
    elif use_all_pairs:
        rows, cols = np.triu_indices(n_cells, k=1)
        pairs = np.column_stack((rows, cols))
    elif fallback_graph is not None:
        graph = _as_csr_square(fallback_graph, n_cells)
        rows, cols = graph.nonzero()
        pairs = np.column_stack((rows, cols))
    else:
        return np.empty((0, 2), dtype=int)

    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)

    valid = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < n_cells)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < n_cells)
        & (pairs[:, 0] != pairs[:, 1])
    )
    pairs = pairs[valid]
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)



def _as_dense_array(matrix: np.ndarray) -> np.ndarray:
    """Convert dense or sparse matrix-like input to a NumPy array."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)



def _as_csr_square(matrix: spmatrix | np.ndarray, n_cells: int) -> csr_matrix:
    """Convert a graph-like input to CSR and validate cell dimensions."""
    if hasattr(matrix, "tocsr"):
        graph = matrix.tocsr().copy()
    else:
        graph = csr_matrix(matrix)
    if graph.shape != (n_cells, n_cells):
        raise ValueError(
            "Graph shape must match the number of cells: "
            f"expected {(n_cells, n_cells)}, got {graph.shape}."
        )
    return graph



def _get_topology_embedding(
    matrix: np.ndarray,
    thred: float = 0.05,
    n_components: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate a low-dimensional KNN topology embedding with shared logic for
    CNV matrices and SNV overlap matrices.
    """
    eps = 1e-12
    max_comps = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_comps < 1:
        return np.ones((matrix.shape[0], 1))

    # Reduce topology features with truncated SVD.
    model = TruncatedSVD(n_components=max_comps, random_state=42)
    pc = model.fit_transform(matrix)
    v = model.explained_variance_ratio_

    # Normalize and order components by explained variance.
    scale = v[1] if v.shape[0] > 1 else v[0]
    v = v / (scale + eps)
    sorted_idx = np.argsort(-v)
    v = v[sorted_idx]
    pc = pc[:, sorted_idx]

    # Retain components above the threshold and the strongest components.
    threshold_mask = v > thred
    top_5_indices = np.argsort(v)[-min(5, v.shape[0]):]
    fallback_mask = np.zeros_like(v, dtype=bool)
    fallback_mask[top_5_indices] = True
    final_mask = threshold_mask | fallback_mask

    # Remove the background component and retain a valid embedding dimension.
    topology_embedding = pc[:, final_mask][:, 1:]
    if topology_embedding.shape[1] == 0:
        topology_embedding = pc[:, :1]

    if verbose:
        print(f"[Topology] Active PCs preserved: {final_mask.sum()}")
    return topology_embedding



def _get_embedding_from_cnv(
    cnv_matrix: np.ndarray,
    thred: float = 0.05,
    n_components: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate a low-dimensional neighborhood topology embedding from CNV data.
    """
    return _get_topology_embedding(
        cnv_matrix,
        thred=thred,
        n_components=n_components,
        verbose=verbose,
    )



def _get_normalized_snv_overlap(
    snv: np.ndarray,
) -> np.ndarray:
    """Calculate normalized MT-only Jaccard overlap."""
    eps = 1e-12
    snv = _as_dense_array(snv)

    mt_matrix = (snv == 1).astype(float)

    mt_intersection = np.dot(mt_matrix, mt_matrix.T)
    mt_union = (
        mt_matrix.sum(axis=1).reshape(-1, 1)
        + mt_matrix.sum(axis=1).reshape(1, -1)
        - mt_intersection
    )

    P = mt_intersection / (mt_union + eps)
    row_sums = np.sqrt(P.sum(1)).reshape(-1, 1) + eps
    col_sums = np.sqrt(P.sum(1)).reshape(1, -1) + eps
    P = P / row_sums / col_sums

    return P
