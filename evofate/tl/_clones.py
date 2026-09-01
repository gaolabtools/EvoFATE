"""Clone definition helpers for EvoFATE genetic analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csr_matrix

from ._genetic_utils import _as_dense_array

if TYPE_CHECKING:
    from anndata import AnnData


def define_clones(
    adata_mut: AnnData,
    resolution: float | Literal["automatic"] = "automatic",
    random_seed: int = 42,
    auto_resolution_min: float = 0.1,
    auto_resolution_max: float = 2.0,
    auto_resolution_steps: float = 0.2,
    key_added: str = "clone",
    min_mt_fraction: float = 0.05,
    min_clone_size: int = 20,
    outlier_label: str = "outlier",
    plot_resolution: bool = True,
) -> None:
    """
    Define clonal populations from mutation data using Leiden clustering.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with mutation matrix in `.X`.
        Values: 1 (mutant), -1 (wildtype), 0 (missing).
    resolution : float or {'automatic'}, default='automatic'
        Resolution parameter for Leiden clustering. If 'automatic', scan a
        resolution grid, calculate average cluster mutation purity at each
        resolution, and choose the elbow point of the purity curve.
    random_seed : int, default=42
        Random seed for reproducibility.
    auto_resolution_min : float, default=0.1
        Minimum resolution considered when `resolution='automatic'`.
    auto_resolution_max : float, default=2.0
        Maximum resolution considered when `resolution='automatic'`.
    auto_resolution_steps : float, default=0.2
        Step size between candidate resolutions in the automatic scan.
    key_added : str, default='clone'
        Column in `.obs` where final Leiden clone labels are stored.
    min_mt_fraction : float, default=0.05
        Minimum fraction of all cells in a clone that must be MT for a
        mutation to be considered a candidate clonal mutation during the
        automatic purity scan. The comparison is strict: a mutation must
        be present in more than this fraction of clone cells.
    min_clone_size : int, default=20
        Minimum clone size after Leiden clustering. Smaller clones are merged
        into the most strongly connected eligible clone using
        ``genetic_lineage_connectivity``. If all candidate supports are zero,
        their cells are marked with ``outlier_label`` instead of being merged
        arbitrarily.
    outlier_label : str, default='outlier'
        Label assigned to unsupported undersized clones.
    plot_resolution : bool, default=True
        Draw the automatic resolution-versus-purity scan and highlight the
        selected elbow resolution. This applies only when `resolution` is
        ``"automatic"``.

    Returns
    -------
    None
        Modifies `adata_mut` in place:
        - `.obs[key_added]`: Leiden cluster assignments from
          weighted `.obsp['genetic_lineage_connectivity']`
        - `.uns['clone_resolution_selection']`: Resolution used and automatic
          diagnostics when `resolution='automatic'`
    """
    import scanpy as sc

    if not 0.0 <= float(min_mt_fraction) < 1.0:
        raise ValueError("`min_mt_fraction` must be in [0, 1).")
    if int(min_clone_size) < 1:
        raise ValueError("`min_clone_size` must be a positive integer.")

    if "genetic_lineage_connectivity" not in adata_mut.obsp:
        raise KeyError(
            "`adata_mut.obsp['genetic_lineage_connectivity']` is missing. "
            "Run `cal_genetic_connectivities(adata_mut)` before "
            "`define_clones(adata_mut)`."
        )

    temporary_prefix = "_evofate_leiden_auto_"
    if str(key_added).startswith(temporary_prefix):
        raise ValueError(
            f"`key_added` cannot start with the temporary prefix "
            f"`{temporary_prefix}`."
        )
    _remove_obs_columns_with_prefix(adata_mut, temporary_prefix)

    if resolution == "automatic":
        if auto_resolution_min <= 0 or auto_resolution_max <= 0:
            raise ValueError("Automatic resolution bounds must be positive.")
        if auto_resolution_max < auto_resolution_min:
            raise ValueError(
                "`auto_resolution_max` must be greater than or equal to "
                "`auto_resolution_min`."
            )
        if auto_resolution_steps <= 0:
            raise ValueError("`auto_resolution_steps` must be positive.")

        matrix = _as_dense_array(adata_mut.X)
        candidate_resolutions = np.arange(
            auto_resolution_min,
            auto_resolution_max + auto_resolution_steps * 0.5,
            auto_resolution_steps,
        )
        candidate_resolutions = candidate_resolutions[
            candidate_resolutions <= auto_resolution_max + 1e-12
        ]
        mean_purity = []
        try:
            for index, candidate_resolution in enumerate(candidate_resolutions):
                temporary_key = f"{temporary_prefix}{index}"
                sc.tl.leiden(
                    adata_mut,
                    resolution=float(candidate_resolution),
                    random_state=random_seed,
                    flavor="igraph",
                    obsp="genetic_lineage_connectivity",
                    use_weights=True,
                    key_added=temporary_key,
                )
                mean_purity.append(
                    _cal_cluster_mean_mutation_purity(
                        matrix=matrix,
                        clusters=adata_mut.obs[temporary_key].to_numpy(),
                        min_mt_fraction=float(min_mt_fraction),
                        min_cluster_size=int(min_clone_size),
                    )
                )
        finally:
            _remove_obs_columns_with_prefix(adata_mut, temporary_prefix)

        mean_purity = np.asarray(mean_purity, dtype=float)
        selected_index = _find_elbow_index(candidate_resolutions, mean_purity)
        resolution = float(candidate_resolutions[selected_index])
        adata_mut.uns["clone_resolution_selection"] = {
            "method": "candidate_mutation_purity_elbow",
            "selected_resolution": resolution,
            "candidate_resolutions": np.asarray(candidate_resolutions, dtype=float),
            "mean_purity": np.asarray(mean_purity, dtype=float),
            "selected_index": int(selected_index),
        }
        if plot_resolution:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                candidate_resolutions,
                mean_purity,
                marker="o",
                color="0.25",
                linewidth=1.5,
            )
            ax.scatter(
                [resolution],
                [mean_purity[selected_index]],
                s=90,
                color="tab:red",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                label=f"selected: {resolution:.3g}",
            )
            ax.axvline(resolution, color="tab:red", linestyle="--", alpha=0.7)
            ax.set_xlabel("Leiden resolution")
            ax.set_ylabel("Mean mutation purity")
            ax.set_title("Clone resolution selection")
            ax.legend(frameon=False)
            fig.tight_layout()
            plt.show()
    elif not isinstance(resolution, (int, float, np.floating)):
        raise ValueError("`resolution` must be a positive float or 'automatic'.")
    else:
        resolution = float(resolution)
        adata_mut.uns["clone_resolution_selection"] = {
            "method": "manual",
            "selected_resolution": resolution,
        }

    if float(resolution) <= 0:
        raise ValueError("`resolution` must be positive.")

    sc.tl.leiden(
        adata_mut,
        resolution=float(resolution),
        random_state=random_seed,
        flavor="igraph",
        obsp="genetic_lineage_connectivity",
        use_weights=True,
        key_added=key_added,
    )
    merge_history = _merge_small_clones(
        adata_mut,
        key_added=key_added,
        connectivity_key="genetic_lineage_connectivity",
        min_clone_size=int(min_clone_size),
        outlier_label=str(outlier_label),
    )
    marked = [
        entry for entry in merge_history
        if entry.get("target") == str(outlier_label)
    ]
    final_outlier_count = int(
        np.sum(np.asarray(adata_mut.obs[key_added].astype(str)) == str(outlier_label))
    )
    if final_outlier_count > 0:
        isolated_clone_count = max(len(marked), 1)
        print(
            "define_clones isolated "
            f"{isolated_clone_count} clone(s), {final_outlier_count} cell(s)."
        )
        print(
            "Run:\n"
            "adata_snv = adata_snv[\n"
            "    adata_snv.obs[\"clone\"].astype(str) != \"outlier\"\n"
            "].copy()"
        )
    _remove_obs_columns_with_prefix(adata_mut, temporary_prefix)
    adata_mut.uns["clone_resolution"] = float(resolution)
    # Merge history is used only for the immediate outlier report and is not
    # retained in `.uns` to keep the AnnData object compact and H5AD-safe.
    adata_mut.uns.pop("clone_merge_history", None)
    adata_mut.uns["clone_merge_params"] = {
        "min_clone_size": int(min_clone_size),
        "connectivity_key": "genetic_lineage_connectivity",
        "outlier_label": str(outlier_label),
        "outlier_count": int(sum(
            int(entry.get("source_size", 0))
            for entry in merge_history
            if entry.get("target") == str(outlier_label)
        )),
    }
    print(f"define_clones selected resolution: {float(resolution):.4g}")

def _remove_obs_columns_with_prefix(adata_mut: AnnData, prefix: str) -> None:
    """Remove temporary Scanpy `.obs` and `.uns` entries by prefix."""
    for column in list(adata_mut.obs.columns):
        if str(column).startswith(prefix):
            del adata_mut.obs[column]
    for key in list(adata_mut.uns.keys()):
        if str(key).startswith(prefix):
            del adata_mut.uns[key]


def _merge_small_clones(
    adata_mut: AnnData,
    key_added: str,
    connectivity_key: str,
    min_clone_size: int,
    outlier_label: str = "outlier",
) -> list[dict[str, object]]:
    """Merge undersized clones into their strongest connected clone."""
    graph = csr_matrix(adata_mut.obsp[connectivity_key], dtype=float)
    if graph.shape != (adata_mut.n_obs, adata_mut.n_obs):
        raise ValueError("Clone connectivity must match the number of cells.")
    if graph.data.size and (
        not np.isfinite(graph.data).all() or np.any(graph.data < 0)
    ):
        raise ValueError("Clone connectivity must be finite and nonnegative.")
    graph = graph.tocsr()
    labels = np.asarray(adata_mut.obs[key_added].astype(str), dtype=object).copy()
    history: list[dict[str, object]] = []

    while True:
        clone_names = np.unique(labels)
        clone_names = clone_names[clone_names != str(outlier_label)]
        if clone_names.size == 0:
            break
        sizes = np.asarray([(labels == value).sum() for value in clone_names], dtype=int)
        small_indices = np.flatnonzero(sizes < int(min_clone_size))
        if small_indices.size == 0:
            break

        merged = False
        for small_index in small_indices:
            source = str(clone_names[small_index])
            source_cells = np.flatnonzero(labels == source)
            if source_cells.size == 0:
                continue

            candidate_indices = [
                int(index)
                for index, size in enumerate(sizes)
                if index != small_index and size >= int(min_clone_size)
            ]
            if not candidate_indices:
                candidate_indices = [
                    int(index)
                    for index in range(clone_names.size)
                    if index != small_index
                ]
            if not candidate_indices:
                continue

            edge_mass = np.asarray(graph[source_cells].sum(axis=0)).ravel()
            scored = [
                (float(edge_mass[np.flatnonzero(labels == clone_names[index])].sum()), index)
                for index in candidate_indices
            ]
            scored.sort(
                key=lambda item: (-item[0], -int(sizes[item[1]]), str(clone_names[item[1]]))
            )
            support, target_index = scored[0]
            if support <= 0.0:
                labels[source_cells] = str(outlier_label)
                history.append(
                    {
                        "source": source,
                        "target": str(outlier_label),
                        "source_size": int(source_cells.size),
                        "target_size_before": 0,
                        "connectivity_support": support,
                        "reason": "unsupported_small_clone",
                    }
                )
                merged = True
                continue
            target = str(clone_names[target_index])
            labels[source_cells] = target
            history.append(
                {
                    "source": source,
                    "target": target,
                    "source_size": int(source_cells.size),
                    "target_size_before": int(sizes[target_index]),
                    "connectivity_support": support,
                }
            )
            merged = True

        if not merged:
            break

    adata_mut.obs[key_added] = labels
    return history


def _cal_cluster_mean_mutation_purity(
    matrix: np.ndarray,
    clusters,
    min_mt_fraction: float = 0.05,
    min_cluster_size: int = 1,
    eps: float = 1e-12,
) -> float:
    """Calculate purity for candidate clonal mutations.

    A mutation is a candidate for a cluster only when its MT call is present
    in more than ``min_mt_fraction`` of all cells in that cluster. Clusters
    smaller than ``min_cluster_size`` are excluded from the scan. Purity is
    then calculated from the observed MT/WT calls for those candidates;
    missing calls do not count as either allele.
    """
    matrix = np.asarray(matrix)
    clusters_array = np.asarray(clusters).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("`matrix` must be a 2D array.")
    if clusters_array.shape[0] != matrix.shape[0]:
        raise ValueError("`clusters` must have one label per matrix row.")
    if not 0.0 <= float(min_mt_fraction) < 1.0:
        raise ValueError("`min_mt_fraction` must be in [0, 1).")
    if int(min_cluster_size) < 1:
        raise ValueError("`min_cluster_size` must be positive.")

    _, cluster_index = np.unique(clusters_array, return_inverse=True)
    cluster_membership = csr_matrix(
        (
            np.ones(matrix.shape[0], dtype=float),
            (cluster_index, np.arange(matrix.shape[0])),
        ),
        shape=(int(cluster_index.max()) + 1, matrix.shape[0]),
    )
    raw_MT = np.asarray(
        cluster_membership @ (matrix == 1).astype(float, copy=False),
        dtype=float,
    )
    raw_WT = np.asarray(
        cluster_membership @ (matrix == -1).astype(float, copy=False),
        dtype=float,
    )
    cluster_size = np.asarray(
        cluster_membership.sum(axis=1),
        dtype=float,
    ).reshape(-1, 1)
    mt_fraction_all_cells = np.divide(
        raw_MT,
        cluster_size,
        out=np.zeros_like(raw_MT, dtype=float),
        where=cluster_size > eps,
    )
    candidate = mt_fraction_all_cells > float(min_mt_fraction)
    eligible_cluster = cluster_size >= int(min_cluster_size)
    raw_count = raw_MT + raw_WT
    raw_mt_fraction = np.divide(
        raw_MT,
        raw_count + eps,
        out=np.full_like(raw_MT, 0.5, dtype=float),
        where=raw_count > eps,
    )
    purity = 2.0 * np.abs(raw_mt_fraction - 0.5)
    supported_candidates = candidate & (raw_count > eps) & eligible_cluster
    if not np.any(supported_candidates):
        return 0.0
    return float(np.mean(purity[supported_candidates]))


def _find_elbow_index(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index farthest from the line joining curve endpoints."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must be 1D arrays with matching length.")
    if x.shape[0] <= 2 or np.allclose(y, y[0]):
        return 0

    x_range = np.ptp(x)
    y_range = np.ptp(y)
    if x_range == 0 or y_range == 0:
        return 0

    points = np.column_stack(((x - x.min()) / x_range, (y - y.min()) / y_range))
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return 0
    delta = start - points
    distances = np.abs(line[0] * delta[:, 1] - line[1] * delta[:, 0]) / line_norm
    return int(np.argmax(distances))
