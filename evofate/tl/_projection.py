"""
Utility functions for EvoFATE.

This module provides functions for:
- Lineage tree construction and layout
- Embedding projections and timing calculations
- Visualization utilities
"""

from __future__ import annotations

import importlib

import warnings

from dataclasses import dataclass

from typing import TYPE_CHECKING, Literal

from collections.abc import Mapping, Sequence

import numpy as np

import pandas as pd

import networkx as nx

from scipy import sparse, spatial, stats

from scipy.linalg import eigh, orthogonal_procrustes, svd

from scipy.optimize import LinearConstraint, minimize

from scipy.sparse.linalg import eigsh, spsolve

from sklearn.cross_decomposition import CCA

from sklearn.isotonic import IsotonicRegression

from sklearn.decomposition import PCA

from sklearn.cluster import SpectralClustering

from sklearn.linear_model import Ridge

from sklearn.neighbors import NearestNeighbors

def cal_linear_projection(
    adata_mut,
    embedding_key: str = "X_integrated",
    clone_key: str = "ordered_clone",
    lineage_tree_key: str = "Lineage_tree",
    layout_key: str = "Lineage_tree_coords_view_integrative",
    layout_source: Literal["genetic", "evofate"] = "genetic",
    output_key: str = "X_lineage_linear",
    n_residual_components: int = 5,
    residual_ridge: float = 1e-3,
    lineage_weight: float = 1.0,
    residual_weight: float = 0.2,
    residual_smoothing: float = 0.0,
    random_state: int = 0,
    copy: bool = False,
    **kwargs,
):
    """Build a native CCA scaffold for lineage-guided projection.

    The weighted CCA-plus-residual UMAP target is constructed by
    ``cal_lineage_guided_projection`` so its weights are applied at the stage
    that runs supervised UMAP.

    ``layout_source="genetic"`` uses the genetic clonal layout stored in
    ``Lineage_tree_coords_view_genetic``. ``layout_source="evofate"`` uses the
    EvoFATE-refined layout stored in
    ``Lineage_tree_coords_view_integrative``. A non-default ``layout_key``
    continues to override this selection.
    """
    if copy:
        adata_mut = adata_mut.copy()
    if "key" in kwargs:
        embedding_key = kwargs.pop("key")
    if "label_key" in kwargs:
        clone_key = kwargs.pop("label_key")
    if layout_source not in {"genetic", "evofate"}:
        raise ValueError("`layout_source` must be 'genetic' or 'evofate'.")
    selected_layout_key = layout_key
    if layout_key == "Lineage_tree_coords_view_integrative":
        selected_layout_key = (
            "Lineage_tree_coords_view_genetic"
            if layout_source == "genetic"
            else "Lineage_tree_coords_view_integrative"
        )
    if int(n_residual_components) < 0 or float(residual_ridge) < 0.0:
        raise ValueError("Residual component count and ridge must be nonnegative.")
    if float(lineage_weight) < 0.0 or float(residual_weight) < 0.0:
        raise ValueError("Representation block weights must be nonnegative.")
    if float(lineage_weight) == 0.0 and float(residual_weight) == 0.0:
        raise ValueError("At least one target block weight must be positive.")
    if not 0.0 <= float(residual_smoothing) <= 1.0:
        raise ValueError("`residual_smoothing` must be in [0, 1].")
    for key, container, kind in (
        (embedding_key, adata_mut.obsm, "obsm"),
        (clone_key, adata_mut.obs, "obs"),
        (selected_layout_key, adata_mut.uns, "uns"),
    ):
        if key not in container:
            raise KeyError(f"`adata.{kind}['{key}']` is missing.")
    X = np.asarray(adata_mut.obsm[embedding_key], dtype=float)
    n_cells = int(adata_mut.n_obs)
    if X.ndim != 2 or X.shape[0] != n_cells or not np.isfinite(X).all():
        raise ValueError(f"`adata.obsm['{embedding_key}']` must be finite with one row per cell.")
    labels = np.asarray(adata_mut.obs[clone_key].astype(str), dtype=str)

    layout_value = adata_mut.uns[selected_layout_key]
    if isinstance(layout_value, Mapping) and "coordinates" in layout_value:
        coordinates = np.asarray(layout_value["coordinates"], dtype=float)
        order = np.asarray(layout_value.get("clone_order", np.arange(coordinates.shape[0])), dtype=str)
        layout = {str(clone): coordinates[index, :2] for index, clone in enumerate(order)}
    elif isinstance(layout_value, Mapping):
        layout = {str(clone): np.asarray(value, dtype=float)[:2] for clone, value in layout_value.items()}
    else:
        coordinates = np.asarray(layout_value, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(
                f"`adata.uns['{selected_layout_key}']` must contain 2D coordinates."
            )
        layout = {str(index): coordinates[index, :2] for index in range(coordinates.shape[0])}
    missing = sorted(set(labels) - set(layout), key=str)
    if missing:
        raise KeyError(f"The clone layout is missing observed clone(s): {missing[:5]}")

    def standardize(values):
        values = np.asarray(values, dtype=float)
        centered = values - values.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)

    X_scaled = standardize(X)
    clone_target = np.asarray([layout[label] for label in labels], dtype=float)
    target_scaled = standardize(clone_target)
    cca = CCA(n_components=2, max_iter=500, scale=False)
    Z_cca, _ = cca.fit_transform(X_scaled, target_scaled)
    Z_cca = np.asarray(Z_cca, dtype=float)
    cca_min = np.min(Z_cca, axis=0, keepdims=True)
    cca_range = np.max(Z_cca, axis=0, keepdims=True) - cca_min
    Z_cca = np.divide(
        Z_cca - cca_min,
        cca_range,
        out=np.zeros_like(Z_cca),
        where=cca_range > 1e-12,
    )
    clone_ids = np.unique(labels)
    masks = [labels == clone for clone in clone_ids]
    # Keep the CCA scaffold in its native coordinate system.
    aligned = Z_cca.copy()

    # Orient the final projection so increasing clone TMB runs toward +x.
    # This changes only the global orientation and preserves pairwise distances.
    clone_tmb = None
    if "TMB" in adata_mut.obs:
        observed_tmb = pd.to_numeric(adata_mut.obs["TMB"], errors="coerce")
        observed_tmb_values = observed_tmb.to_numpy(dtype=float)
        if np.isfinite(observed_tmb_values).all():
            clone_tmb = np.asarray(
                [observed_tmb_values[labels == clone].mean() for clone in clone_ids],
                dtype=float,
            )
    if clone_tmb is None:
        stored_tmb = np.asarray(adata_mut.uns.get("TMB_clone", []), dtype=float).reshape(-1)
        numeric_clone_ids = []
        valid_numeric_ids = True
        for clone in clone_ids:
            try:
                numeric_clone_ids.append(int(clone))
            except (TypeError, ValueError):
                valid_numeric_ids.append(False)
                break
        if valid_numeric_ids and stored_tmb.size:
            if all(0 <= index < stored_tmb.size for index in numeric_clone_ids):
                clone_tmb = stored_tmb[numeric_clone_ids]

    tmb_rotation_angle = 0.0
    if clone_tmb is not None and np.isfinite(clone_tmb).all():
        aligned_centers = np.vstack(
            [aligned[labels == clone].mean(axis=0) for clone in clone_ids]
        )
        clone_weights = np.asarray(
            [np.sum(labels == clone) for clone in clone_ids],
            dtype=float,
        )
        centered_centers = aligned_centers - np.average(
            aligned_centers,
            axis=0,
            weights=clone_weights,
        )
        centered_tmb = clone_tmb - np.average(clone_tmb, weights=clone_weights)
        covariance = np.sum(
            clone_weights[:, None] * centered_centers * centered_tmb[:, None],
            axis=0,
        )
        if np.linalg.norm(covariance) > 1e-12 and np.std(clone_tmb) > 1e-12:
            tmb_rotation_angle = float(np.arctan2(covariance[1], covariance[0]))
            c, s = np.cos(tmb_rotation_angle), np.sin(tmb_rotation_angle)
            tmb_rotation = np.array([[c, -s], [s, c]], dtype=float)
            aligned_center = np.average(
                aligned_centers,
                axis=0,
                weights=clone_weights,
            )
            aligned = (
                (aligned - aligned_center[None, :]) @ tmb_rotation
                + aligned_center[None, :]
            )

    adata_mut.obsm[output_key] = aligned.astype(np.float32)
    for key in (
        "lineage_edge", "lineage_edge_progress", "lineage_edge_transverse",
        "lineage_edge_assignment_distance", "lineage_assigned_parent",
        "lineage_assigned_child", "lineage_assigned_edge",
        "lineage_edge_t_raw", "lineage_edge_fraction", "lineage_edge_cost",
    ):
        if key in adata_mut.obs:
            del adata_mut.obs[key]
    adata_mut.uns["linear_projection"] = {
        "method": "cca_native_tmb_oriented_multidimensional_umap_target",
        "embedding_key": embedding_key,
        "clone_key": clone_key,
        "lineage_tree_key": lineage_tree_key,
        "layout_key": selected_layout_key,
        "layout_source": layout_source,
        "tmb_rotation_angle": tmb_rotation_angle,
        "tmb_x_axis_oriented": bool(clone_tmb is not None),
    }
    return adata_mut if copy else None

def cal_lineage_guided_projection(
    adata_mut,
    integrated_key: str = "X_integrated",
    clone_key: str = "ordered_clone",
    clone_layout_key: str = "Lineage_tree_coords_view_genetic",
    linear_projection_key: str = "X_lineage_linear",
    n_spectral_components: int | None = None,
    n_lineage_components: int = 5,
    cca_target_weight: float = 0.30,
    umap_init: Literal["spectral", "linear", "cca"] = "linear",
    target_key: str = "X_lineage_umap_target",
    output_key: str = "X_evofate_umap",
    n_neighbors: int = 30,
    min_dist: float = 0.2,
    spread: float = 1.0,
    use_final_procrustes: bool = False,
    n_epochs: int | None = None,
    random_state: int = 0,
    store_intermediate: bool = True,
    copy: bool = False,
):
    """Project ``X_integrated`` into a linear-scaffold UMAP."""
    if copy:
        adata_mut = adata_mut.copy()
    if not 0.0 <= float(cca_target_weight) <= 1.0:
        raise ValueError("`cca_target_weight` must be between 0 and 1.")
    if umap_init not in {"spectral", "linear", "cca"}:
        raise ValueError("`umap_init` must be 'spectral' or 'linear'.")
    for key, container, kind in (
        (integrated_key, adata_mut.obsm, "obsm"),
        (clone_key, adata_mut.obs, "obs"),
    ):
        if key not in container:
            raise KeyError(f"`adata.{kind}['{key}']` is missing.")
    X = np.asarray(adata_mut.obsm[integrated_key], dtype=float)
    n_cells = int(adata_mut.n_obs)
    if X.ndim != 2 or X.shape[0] != n_cells:
        raise ValueError(f"`adata.obsm['{integrated_key}']` must have one row per cell.")
    if not np.isfinite(X).all():
        raise ValueError(f"`adata.obsm['{integrated_key}']` contains non-finite values.")
    clone_values = adata_mut.obs[clone_key]
    if hasattr(clone_values, "isna") and bool(clone_values.isna().any()):
        raise ValueError(
            f"`adata.obs['{clone_key}']` contains missing clone assignments; "
            "a categorical clone target is required."
        )
    labels = np.asarray(clone_values.astype(str), dtype=str)
    clone_ids_for_validation = np.unique(labels)
    if clone_ids_for_validation.size < 2:
        raise ValueError(
            f"`adata.obs['{clone_key}']` must contain at least two biological "
            "clones; a categorical clone target is required."
        )

    if linear_projection_key not in adata_mut.obsm:
        raise KeyError(
            f"`adata.obsm['{linear_projection_key}']` is missing. "
            "Run cal_linear_projection first."
        )
    linear_scaffold = np.asarray(
        adata_mut.obsm[linear_projection_key], dtype=float
    )
    if linear_scaffold.shape != (n_cells, 2) or not np.isfinite(linear_scaffold).all():
        raise ValueError(
            f"`adata.obsm['{linear_projection_key}']` must be a finite (n_cells, 2) "
            "linear projection."
        )

    clone_ids = np.unique(labels)
    masks = [labels == clone for clone in clone_ids]
    X_linear_2d = linear_scaffold
    adata_mut.obsm["X_evofate_linear"] = X_linear_2d.astype(np.float32)
    X_umap = X.astype(np.float32, copy=False)
    adata_mut.obsm["X_lineage_umap_input"] = X_umap
    # Store the continuous supervision target separately from the UMAP features.
    adata_mut.obsm[target_key] = X_linear_2d.astype(np.float32)

    UMAP = _import_umap_model()
    neighbors = min(max(int(n_neighbors), 2), n_cells - 1)
    procrustes_transform = None
    init_value = (
        "spectral"
        if umap_init == "spectral"
        else X_linear_2d.astype(np.float32)
    )
    model = UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        spread=float(spread),
        metric="euclidean",
        init=init_value,
        target_metric="euclidean",
        target_weight=float(cca_target_weight),
        n_epochs=n_epochs,
        random_state=int(random_state),
        low_memory=True,
        verbose=False,
    )
    result = np.asarray(model.fit_transform(X_umap, y=X_linear_2d), dtype=float)
    if result.shape != (n_cells, 2) or not np.isfinite(result).all():
        raise ValueError("UMAP returned invalid coordinates.")
    if hasattr(model, "graph_"):
        adata_mut.obsp["evofate_umap_connectivities"] = model.graph_.tocsr().astype(np.float32)

    if use_final_procrustes:
        procrustes_transform = _fit_similarity_transform(
            result,
            X_linear_2d,
            allow_reflection=False,
        )
        result = _apply_similarity_transform(result, procrustes_transform)
    final_centers = np.vstack([result[mask].mean(axis=0) for mask in masks])
    linear_centers = np.vstack([X_linear_2d[mask].mean(axis=0) for mask in masks])
    linear_pairwise = spatial.distance.pdist(linear_centers)
    final_pairwise = spatial.distance.pdist(final_centers)
    if (
        linear_pairwise.size >= 2
        and np.std(linear_pairwise) > 1e-12
        and np.std(final_pairwise) > 1e-12
    ):
        clone_geometry_correlation = float(
            np.corrcoef(linear_pairwise, final_pairwise)[0, 1]
        )
    else:
        clone_geometry_correlation = np.nan
    adata_mut.obsm[output_key] = result.astype(np.float32)
    adata_mut.obsm["X_lineage_umap"] = result.astype(np.float32)
    # Synchronize the project-wide projection key for downstream plotting.
    adata_mut.obsm["X_evofate_projection"] = result.astype(np.float32)
    adata_mut.uns["lineage_guided_projection"] = {
        "method": "integrated_linear_scaffold_supervised_umap",
        "params": {
            "integrated_key": integrated_key,
            "clone_key": clone_key,
            "clone_layout_key": clone_layout_key,
            "linear_projection_key": linear_projection_key,
            "target_key": target_key,
            "cca_target_weight": float(cca_target_weight),
            "umap_init": umap_init,
            "use_final_procrustes": bool(use_final_procrustes),
            "n_neighbors": int(neighbors),
            "min_dist": float(min_dist),
            "spread": float(spread),
            "random_state": int(random_state),
        },
        "X_linear_scaffold": X_linear_2d,
        "linear_clone_centroids": linear_centers,
        "final_clone_centroids": final_centers,
        "clone_centroid_labels": np.asarray(clone_ids, dtype=str),
        "linear_clone_distance_matrix": spatial.distance.squareform(linear_pairwise),
        "final_clone_distance_matrix": spatial.distance.squareform(final_pairwise),
        "clone_geometry_correlation": clone_geometry_correlation,
        "procrustes_transform": None if procrustes_transform is None else {
            "scale": float(procrustes_transform["scale"]),
            "rotation": np.asarray(procrustes_transform["rotation"], dtype=float).tolist(),
            "translation": np.asarray(procrustes_transform["translation"], dtype=float).tolist(),
            "allow_reflection": False,
        },
    }
    return adata_mut if copy else None

def _import_umap_model():
    """Import the UMAP model class, recovering from numba cache-dir failures."""
    try:
        from umap import UMAP

        return UMAP
    except Exception as exc:
        if "cannot cache function" not in str(exc):
            raise ImportError(
                "`cal_guided_umap_projection` requires `umap-learn` to be "
                "installed and importable."
            ) from exc

    import os
    import sys
    import tempfile

    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "evofate_numba_cache"),
    )
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    for module_name in list(sys.modules):
        if (
            module_name == "umap"
            or module_name.startswith("umap.")
            or module_name == "pynndescent"
            or module_name.startswith("pynndescent.")
        ):
            sys.modules.pop(module_name, None)

    try:
        from umap import UMAP

        return UMAP
    except Exception as retry_exc:
        raise ImportError(
            "`cal_guided_umap_projection` requires `umap-learn` to be "
            "installed and importable."
        ) from retry_exc

def _fit_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_reflection: bool = True,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Fit one global similarity transform from source to target points."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Similarity source and target must both have shape (n, 2).")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_ss = float(np.sum(source_centered**2))
    if source_ss <= eps:
        rotation = np.eye(2)
        scale = 1.0
    else:
        u, singular_values, vt = np.linalg.svd(
            source_centered.T @ target_centered,
            full_matrices=False,
        )
        rotation = u @ vt
        if not allow_reflection and np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        scale = float(np.sum(singular_values) / source_ss)

    translation = target_mean - scale * source_mean @ rotation
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "allow_reflection": bool(allow_reflection),
    }

def _apply_similarity_transform(
    source: np.ndarray,
    transform: dict[str, object],
) -> np.ndarray:
    """Apply a similarity transform fitted by `_fit_similarity_transform`."""
    return (
        float(transform["scale"])
        * np.asarray(source, dtype=float)
        @ np.asarray(transform["rotation"], dtype=float)
        + np.asarray(transform["translation"], dtype=float)
    )
