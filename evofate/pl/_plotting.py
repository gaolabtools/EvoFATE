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

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Patch
import seaborn as sns

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

def _ensure_matplotlib_cache_dir() -> None:
    """Point Matplotlib at a writable cache dir before lazy plotting imports."""
    import os
    import tempfile

    cache_dir = os.path.join(tempfile.gettempdir(), "evofate_matplotlib_cache")
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

def _get_uns_graph_lazy(*args, **kwargs):
    """Import igraph-backed graph helpers only when graph behavior is needed."""
    _ensure_matplotlib_cache_dir()
    from ._genetic_utils import _get_uns_graph

    return _get_uns_graph(*args, **kwargs)

def _get_lineage_tree_plot_layout(
    adata_mut: AnnData,
    layout_key: str | None = None,
) -> np.ndarray:
    """Return the preferred lineage-tree layout for plotting."""
    if layout_key is None:
        layout_key = (
            "Lineage_tree_coords_view_genetic"
            if "Lineage_tree_coords_view_genetic" in adata_mut.uns
            else "Lineage_tree_coords"
        )
    if layout_key not in adata_mut.uns:
        raise KeyError(
            f"`adata_mut.uns['{layout_key}']` is missing. "
            "Run `tl.cal_tree_layout(adata_mut)` first."
        )
    return np.asarray(adata_mut.uns[layout_key], dtype=float)

def _to_hex_color(color: tuple[float, float, float, float]) -> str:
    """Convert an RGBA tuple to hex."""
    r, g, b, _ = color
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def _get_plot_color_map(
    values: pd.Series | np.ndarray,
    palette: str,
    label: str | None,
    adata_mut: AnnData,
    include_normal: bool = False,
) -> tuple[np.ndarray, dict[object, str]]:
    """Return stable categorical colors shared by all plotting functions."""
    if label in {"SNV_clone", "ordered_clone"}:
        if "consensus_profile" not in adata_mut.uns:
            raise KeyError(
                "`adata_mut.uns['consensus_profile']` is missing. "
                "Run `cal_consensus_profile(adata_mut)` first."
            )

        num_nodes = adata_mut.uns["consensus_profile"].shape[0]
        base_cmap = cm.get_cmap("Spectral_r")
        colors = base_cmap(np.linspace(0, 0.8, num_nodes))
        colors[0] = [0.5, 0.5, 0.5, 1]
        start = 0 if include_normal else 1
        categories = np.asarray([str(i) for i in range(start, num_nodes)])
        color_dict = {
            category: _to_hex_color(colors[int(category)])
            for category in categories
        }
        return categories, color_dict

    raw_categories = pd.Series(values).dropna().unique()

    def sort_key(value: object) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    categories = np.array(sorted(raw_categories, key=sort_key))
    base_cmap = cm.get_cmap(palette)
    colors = base_cmap(np.linspace(0, 1, len(categories)))
    color_dict = {
        category: _to_hex_color(color)
        for category, color in zip(categories, colors)
    }
    return categories, color_dict

def _as_dense_array(matrix: object) -> np.ndarray:
    """Return a dense NumPy array from dense or sparse matrix-like input."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)

def _get_obs_label_row_order_and_colors(
    adata_mut: AnnData,
    label: str,
    palette: str,
    label_cmap: str,
    row_cluster: bool,
) -> tuple[
    np.ndarray,
    list[tuple[float, float, float, float]] | np.ndarray,
    list[Patch] | None,
    Normalize | None,
    object | None,
]:
    """Return row order and annotation colors for numeric or categorical obs labels."""
    if label not in adata_mut.obs:
        raise KeyError(f"`adata_mut.obs['{label}']` is missing.")

    label_values = adata_mut.obs[label]
    if pd.api.types.is_numeric_dtype(label_values):
        numeric_values = pd.to_numeric(label_values)
        row_order = (
            np.arange(adata_mut.n_obs)
            if row_cluster
            else np.argsort(numeric_values.to_numpy(), kind="stable")
        )
        vmin = float(np.nanmin(numeric_values))
        vmax = float(np.nanmax(numeric_values))
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)
        color_map = cm.get_cmap(label_cmap)
        row_colors = [
            color_map(norm(value)) if pd.notna(value) else (0.8, 0.8, 0.8, 1.0)
            for value in numeric_values.iloc[row_order]
        ]
        return row_order, row_colors, None, norm, color_map

    label_values = label_values.astype(str)
    categories, color_dict = _get_plot_color_map(
        values=label_values,
        palette=palette,
        label=label,
        adata_mut=adata_mut,
    )
    label_colors = label_values.map(color_dict)
    if row_cluster:
        row_order = np.arange(adata_mut.n_obs)
    else:
        category_rank = {category: rank for rank, category in enumerate(categories)}
        row_order = np.array(
            sorted(
                range(adata_mut.n_obs),
                key=lambda index: (category_rank[label_values.iloc[index]], index),
            )
        )
    row_colors = label_colors.iloc[row_order].to_numpy()
    legend_handles = [
        Patch(color=color_dict[category], label=str(category))
        for category in categories
    ]
    return row_order, row_colors, legend_handles, None, None

def _add_obs_label_legend(
    grid: sns.matrix.ClusterGrid,
    label: str,
    show_legend: bool,
    legend_handles: list[Patch] | None,
    norm: Normalize | None,
    color_map: object | None,
    bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    colorbar_pad: float = 0.02,
) -> None:
    """Add categorical legend or numeric colorbar for obs row annotations."""
    if not show_legend:
        return

    if legend_handles is not None:
        grid.ax_heatmap.legend(
            handles=legend_handles,
            title=label,
            loc="center left",
            bbox_to_anchor=bbox_to_anchor,
            borderaxespad=0,
            fontsize="small",
            frameon=False,
        )
        return

    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=color_map)
    scalar_mappable.set_array([])
    grid.fig.colorbar(
        scalar_mappable,
        ax=grid.ax_heatmap,
        label=label,
        fraction=0.03,
        pad=colorbar_pad,
    )

def _save_plot_if_requested(
    plot_obj: Figure | sns.matrix.ClusterGrid,
    filename: str | None,
) -> None:
    """Save a Matplotlib figure or Seaborn ClusterGrid when a filename is provided."""
    if filename is None:
        return
    fig = plot_obj.fig if hasattr(plot_obj, "fig") else plot_obj
    fig.savefig(filename, bbox_inches="tight")

def _get_trinary_colormap(cmap: str) -> ListedColormap:
    """Return the fixed WT/NA/MT colors used by mutation plots."""
    return ListedColormap(
        ["#c1daf2", "#fafbfc", "#ea6e70"],
        name="evofate_wt_na_mt",
    )

def plot_consensus_profile(
    adata_mut: AnnData,
    profile: Literal["binary", "call", "high_confidence", "predict"] = "binary",
    consensus_source: Literal["raw", "predicted"] = "raw",
    show_germline: bool = False,
    show_wt: bool = False,
    cmap: str | None = None,
    palette: str = "Spectral_r",
    metric: str = "hamming",
    row_cluster: bool = False,
    col_cluster: bool = False,
    figsize: tuple[float, float] | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    filename: str | None = None,
) -> sns.matrix.ClusterGrid:
    """
    Plot heatmap of consensus mutation profiles across clones.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with consensus profiles in `.uns`.
    profile : {'binary', 'call', 'high_confidence', 'predict'}, default='binary'
        Which consensus profile to plot. 'binary' is retained as a
        compatibility alias for the ternary call profile in
        `.uns['consensus_profile']`; 'high_confidence' uses
        `.uns['consensus_profile_high_confidence']`; 'predict' uses
        `.uns['consensus_profile_predict']`.
    cmap : str, optional
        Colormap for heatmap.
    palette : str, default='Spectral_r'
        Color palette for row colors.
    metric : str, default='hamming'
        Distance metric for clustering.
    row_cluster : bool, default=False
        Whether to cluster rows.
    col_cluster : bool, default=False
        Whether to cluster columns. By default, the mutation order calculated
        by `cal_consensus_profile` is preserved.
    consensus_source : {'raw', 'predicted'}, default='raw'
        Use the raw consensus profile or the lineage-predicted consensus
        profile. This applies to the compatibility ``binary``/``call``
        profile selection.
    show_germline : bool, default=False
        Include columns that are MT in every biological clone. These are
        treated as germline-like mutations for visualization.
    show_wt : bool, default=False
        Include columns that are WT in every biological clone. Mixed clonal
        mutations remain visible regardless of this option.
    figsize : tuple, optional
        Figure size.
    filename : str, optional
        Path to save the figure. If None, the figure is not saved.
    vmin, vmax : float, optional
        Explicit continuous color limits. If omitted, the 2nd and 98th
        percentiles are used.

    Returns
    -------
    ClusterGrid
        Seaborn clustermap object.
    """
    if consensus_source not in {"raw", "predicted"}:
        raise ValueError("`consensus_source` must be 'raw' or 'predicted'.")
    if profile in {"binary", "call"}:
        profile_key = (
            "consensus_profile_predict"
            if consensus_source == "predicted"
            else "consensus_profile"
        )
    elif profile == "high_confidence":
        profile_key = "consensus_profile_high_confidence"
    elif profile == "predict":
        profile_key = "consensus_profile_predict"
    else:
        raise ValueError(
            "profile must be 'binary', 'call', 'high_confidence', or 'predict'."
        )
    if profile_key not in adata_mut.uns:
        raise KeyError(
            f"`adata_mut.uns['{profile_key}']` is missing. "
            "Run `cal_consensus_profile(adata_mut)` first."
        )

    consensus_profile = np.asarray(adata_mut.uns[profile_key]).copy()
    if consensus_profile.ndim != 2:
        raise ValueError("The consensus profile must be a 2D matrix.")
    # Exclude row zero, which represents the artificial normal, from germline tests.
    biological_profile = consensus_profile[1:]
    germline_columns = (
        np.all(biological_profile == 1, axis=0)
        if biological_profile.shape[0]
        else np.zeros(consensus_profile.shape[1], dtype=bool)
    )
    wt_columns = (
        np.all(biological_profile == -1, axis=0)
        if biological_profile.shape[0]
        else np.zeros(consensus_profile.shape[1], dtype=bool)
    )
    keep_columns = np.ones(consensus_profile.shape[1], dtype=bool)
    if not show_germline:
        keep_columns &= ~germline_columns
    if not show_wt:
        keep_columns &= ~wt_columns
    if not np.any(keep_columns):
        raise ValueError("No consensus mutations remain after plot filtering.")
    consensus_profile = consensus_profile[:, keep_columns]
    is_trinary_profile = profile_key in {
        "consensus_profile",
        "consensus_profile_high_confidence",
        "consensus_profile_predict",
    }
    heatmap_norm = None
    cbar_kws = {}
    if is_trinary_profile:
        cmap_name = "vlag" if cmap is None else cmap
        cmap = _get_trinary_colormap(cmap_name)
        heatmap_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        heatmap_center = None
        heatmap_vmin = None
        heatmap_vmax = None
        cbar_kws = {"ticks": [-1, 0, 1]}
    else:
        cmap = "Blues" if cmap is None else cmap
        heatmap_center = None
        heatmap_vmin = None
        heatmap_vmax = 1 if vmax is None else vmax

    num_nodes = consensus_profile.shape[0]
    _, color_dict = _get_plot_color_map(
        values=np.asarray([str(i) for i in range(num_nodes)]),
        palette=palette,
        label="ordered_clone",
        adata_mut=adata_mut,
        include_normal=True,
    )
    hex_colors = [color_dict[str(i)] for i in range(num_nodes)]

    pl = sns.clustermap(
        consensus_profile,
        cmap=cmap,
        metric=metric,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        center=heatmap_center,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
        row_colors=hex_colors,
        figsize=figsize,
        norm=heatmap_norm,
        cbar_kws=cbar_kws,
        cbar_pos=None if not show_colorbar else (0.02, 0.80, 0.05, 0.18),
    )
    pl.ax_heatmap.set_xlabel("Mutations", fontsize=12)
    pl.ax_heatmap.set_ylabel("Clones", fontsize=12)
    if is_trinary_profile and pl.ax_heatmap.collections:
        colorbar = pl.ax_heatmap.collections[0].colorbar
        if colorbar is not None:
            colorbar.set_ticks([-1, 0, 1])
            colorbar.set_ticklabels(["WT", "NA", "MT"])
    _save_plot_if_requested(pl, filename)
    return pl

def plot_filtered_mutations(
    adata_mut: AnnData,
    label: str = "ordered_clone",
    profile: Literal["raw", "predict"] = "raw",
    show_germline: bool = False,
    show_wt: bool = False,
    cmap: str = "vlag",
    palette: str = "Spectral_r",
    label_cmap: str = "plasma",
    row_cluster: bool = False,
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    show_colorbar: bool = True,
    rasterized: bool = True,
    filename: str | None = None,
) -> sns.matrix.ClusterGrid:
    """
    Plot filtered mutation calls sorted and colored by an obs label.

    Mutation columns use the ordered names in
    `adata_mut.uns['filtered_mutations']`.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object after `cal_consensus_profile`.
    label : str, default='ordered_clone'
        Column in `.obs` used to sort and color cells.
    profile : {'raw', 'predict'}, default='raw'
        Selects the raw or predicted clone consensus used to choose mutation
        columns. The plotted values always come from the single-cell matrix
        in `.X`.
    show_germline : bool, default=False
        Include mutations that are MT in every biological clone. The raw or
        predicted consensus selected by `profile` determines this category.
    show_wt : bool, default=False
        Include mutations that are WT in every biological clone. Mixed clonal
        mutations are always retained.
    cmap : str, default='vlag'
        Colormap for mutation calls.
    palette : str, default='Spectral_r'
        Color palette for categorical row labels.
    label_cmap : str, default='plasma'
        Colormap for numeric row labels.
    row_cluster : bool, default=False
        Whether to cluster cells. By default cells are sorted by `label`.
    figsize : tuple, optional
        Figure size.
    show_legend : bool, default=True
        Whether to show the clone color legend.
    rasterized : bool, default=True
        Whether to rasterize the heatmap body for safer rendering of large
        single-cell mutation matrices.
    filename : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    ClusterGrid
        Seaborn clustermap object.
    """
    if "filtered_mutations" not in adata_mut.uns:
        raise KeyError(
            "`adata_mut.uns['filtered_mutations']` is missing. "
            "Run `cal_consensus_profile(adata_mut)` first."
        )
    if label not in adata_mut.obs:
        raise KeyError(f"`adata_mut.obs['{label}']` is missing.")

    mutation_names = np.asarray(adata_mut.uns["filtered_mutations"]).astype(str)
    if profile not in {"raw", "predict"}:
        raise ValueError("profile must be 'raw' or 'predict'.")

    var_names = np.asarray(adata_mut.var_names.astype(str))
    var_index = pd.Index(var_names)
    missing_mutations = mutation_names[~np.isin(mutation_names, var_names)]
    if missing_mutations.size > 0:
        raise KeyError(
            "`adata_mut.uns['filtered_mutations']` contains names not found "
            f"in `adata_mut.var_names`: {missing_mutations[:5].tolist()}"
        )
    mutation_indices = var_index.get_indexer(mutation_names)
    # Display single-cell calls; use `profile` only for mutation-column selection.
    matrix = _as_dense_array(adata_mut.X)[:, mutation_indices]

    consensus_key = (
        "consensus_profile_predict" if profile == "predict" else "consensus_profile"
    )
    if consensus_key not in adata_mut.uns:
        raise KeyError(
            f"`adata_mut.uns['{consensus_key}']` is missing. "
            "Run consensus generation first."
        )
    consensus = np.asarray(adata_mut.uns[consensus_key], dtype=int)
    if consensus.ndim != 2 or consensus.shape[1] != mutation_names.size:
        raise ValueError(
            f"`{consensus_key}` columns must match `filtered_mutations`."
        )
    biological_consensus = consensus[1:]
    germline_columns = np.all(biological_consensus == 1, axis=0)
    wt_columns = np.all(biological_consensus == -1, axis=0)
    keep_columns = np.ones(mutation_names.size, dtype=bool)
    if not show_germline:
        keep_columns &= ~germline_columns
    if not show_wt:
        keep_columns &= ~wt_columns
    if not np.any(keep_columns):
        raise ValueError("No mutations remain after germline/WT plot filtering.")
    matrix = matrix[:, keep_columns]
    mutation_names = mutation_names[keep_columns]

    row_order, row_colors, legend_handles, norm, label_color_map = (
        _get_obs_label_row_order_and_colors(
            adata_mut=adata_mut,
            label=label,
            palette=palette,
            label_cmap=label_cmap,
            row_cluster=row_cluster,
        )
    )

    matrix = matrix[row_order]
    row_index = adata_mut.obs_names[row_order]

    data = pd.DataFrame(matrix, index=row_index, columns=mutation_names)
    trinary_cmap = _get_trinary_colormap(cmap)
    trinary_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], trinary_cmap.N)
    pl = sns.clustermap(
        data,
        cmap=trinary_cmap,
        metric="hamming",
        row_cluster=row_cluster,
        col_cluster=False,
        row_colors=row_colors,
        norm=trinary_norm,
        cbar_kws={"ticks": [-1, 0, 1]},
        cbar_pos=None if not show_colorbar else (0.02, 0.80, 0.05, 0.18),
        xticklabels=False,
        yticklabels=False,
        rasterized=rasterized,
        figsize=figsize,
    )
    pl.ax_heatmap.set_xlabel("Filtered mutations", fontsize=12)
    pl.ax_heatmap.set_ylabel("Cells", fontsize=12)
    pl.ax_heatmap.yaxis.set_label_position("right")
    pl.ax_heatmap.yaxis.set_ticks_position("left")
    if pl.ax_heatmap.collections:
        colorbar = pl.ax_heatmap.collections[0].colorbar
        if colorbar is not None:
            colorbar.set_ticks([-1, 0, 1])
            colorbar.set_ticklabels(["WT", "NA", "MT"])

    _add_obs_label_legend(
        grid=pl,
        label=label,
        show_legend=show_legend,
        legend_handles=legend_handles,
        norm=norm,
        color_map=label_color_map,
        bbox_to_anchor=(1.22, 0.5),
        colorbar_pad=0.12,
    )

    _save_plot_if_requested(pl, filename)
    return pl

def _get_lineage_tree_node_sizes(
    adata_mut: AnnData,
    n_nodes: int,
    show_clone_size: bool,
    default_size: float,
    size_range: tuple[float, float],
) -> np.ndarray:
    """Return fixed or clone-size-scaled display sizes for lineage nodes."""
    if n_nodes <= 0:
        return np.array([], dtype=float)
    if not show_clone_size:
        return np.full(n_nodes, float(default_size), dtype=float)
    if "ordered_clone" not in adata_mut.obs:
        raise KeyError(
            "`adata_mut.obs['ordered_clone']` is missing. "
            "Run `tl.cal_clone_connectivity(adata_mut)` first or set "
            "`show_clone_size=False`."
        )

    min_size, max_size = map(float, size_range)
    if min_size <= 0 or max_size <= 0:
        raise ValueError("`size_range` values must be positive.")
    if max_size < min_size:
        raise ValueError("`size_range` must be ordered as (min, max).")

    labels = adata_mut.obs["ordered_clone"].astype(int).to_numpy()
    valid = (labels >= 0) & (labels < n_nodes)
    counts = np.bincount(labels[valid], minlength=n_nodes).astype(float)
    sizes = np.full(n_nodes, min_size, dtype=float)
    positive = counts > 0
    if not np.any(positive):
        return sizes

    positive_counts = counts[positive]
    count_min = float(np.min(positive_counts))
    count_max = float(np.max(positive_counts))
    if np.isclose(count_min, count_max):
        sizes[positive] = max_size
        return sizes

    scaled = (counts - count_min) / (count_max - count_min)
    sizes[positive] = min_size + scaled[positive] * (max_size - min_size)
    return sizes

def plot_lineage_tree(
    adata_mut: AnnData,
    palette: str = "Spectral_r",
    node_values=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    colorbar_label: str = "node_value",
    figsize: tuple[float, float] = (5, 5),
    return_fig: bool = False,
    layout_key: str | None = None,
    filename: str | None = None,
    show_clone_size: bool = False,
    node_size: float = 430,
    node_size_range: tuple[float, float] = (260, 820),
    similarity_edges: bool = False,
) -> Figure | None:
    """
    Plot the directed clonal lineage as arrows.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with tree data.
    palette : str, default='Spectral_r'
        Color palette for nodes.
    node_values : array-like, optional
        Continuous value for each clone node. When provided, node colors are
        drawn from ``cmap`` instead of the categorical ``palette``. Values
        must have one entry per lineage node.
    cmap : str, default='viridis'
        Continuous colormap used when ``node_values`` is provided.
    vmin, vmax : float, optional
        Color limits for ``node_values``.
    show_colorbar : bool, default=True
        Whether to show a colorbar for continuous node values.
    colorbar_label : str, default='node_value'
        Label displayed next to the continuous node-value colorbar.
    figsize : tuple, default=(5, 5)
        Figure size.
    return_fig : bool, default=False
        Whether to return the figure object.
    layout_key : str, optional
        Layout key in `.uns` to plot. By default, uses
        `Lineage_tree_coords_view_genetic` when available and falls back to
        `Lineage_tree_coords`.
    filename : str, optional
        Path to save the figure. If None, the figure is not saved.
    show_clone_size : bool, default=False
        If True, scale node marker area by the number of cells assigned to
        each `ordered_clone`.
    node_size : float, default=430
        Marker area used when `show_clone_size=False`.
    node_size_range : tuple, default=(260, 820)
        Minimum and maximum marker areas used when `show_clone_size=True`.

    similarity_edges : bool, default=False
        If True, draw profile-neighborhood edges from ``Clone_connectivity``
        that are not already present in the directed ``Lineage_tree`` as
        dashed lines beneath the lineage arrows.

    Returns
    -------
    Figure or None
        Figure object if return_fig is True.
    """
    layout = _get_lineage_tree_plot_layout(adata_mut, layout_key=layout_key)
    g2 = _get_uns_graph_lazy(adata_mut, "Lineage_tree", directed=True)
    num_nodes = layout.shape[0]

    if node_values is None:
        _, color_dict = _get_plot_color_map(
            values=np.asarray([str(i) for i in range(num_nodes)]),
            palette=palette,
            label="ordered_clone",
            adata_mut=adata_mut,
            include_normal=True,
        )
        hex_colors = [color_dict[str(i)] for i in range(num_nodes)]
        value_norm = None
        value_mappable = None
    else:
        values = np.asarray(node_values, dtype=float).reshape(-1)
        if values.size != num_nodes or not np.isfinite(values).all():
            raise ValueError("node_values must contain one finite value per clone node.")
        value_norm = Normalize(
            vmin=float(np.min(values) if vmin is None else vmin),
            vmax=float(np.max(values) if vmax is None else vmax),
        )
        if value_norm.vmax <= value_norm.vmin:
            value_norm = Normalize(vmin=value_norm.vmin, vmax=value_norm.vmin + 1.0)
        value_cmap = cm.get_cmap(cmap)
        hex_colors = value_cmap(value_norm(values))
        value_mappable = cm.ScalarMappable(norm=value_norm, cmap=value_cmap)
        value_mappable.set_array(values)
    node_sizes = _get_lineage_tree_node_sizes(
        adata_mut=adata_mut,
        n_nodes=num_nodes,
        show_clone_size=show_clone_size,
        default_size=float(node_size),
        size_range=node_size_range,
    )
    arrow_shrink = np.sqrt(node_sizes) * 0.85

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if similarity_edges:
        similarity_graph = _get_uns_graph_lazy(
            adata_mut,
            "Clone_connectivity",
            directed=False,
            required=False,
        )
        directed_pairs = {
            tuple(sorted((int(source), int(target))))
            for source, target in g2.get_edgelist()
            if int(source) != int(target)
        }
        if similarity_graph is not None:
            similarity_edge_list = similarity_graph.get_edgelist()
        else:
            similarity_edge_list = []
        for source, target in similarity_edge_list:
            source, target = int(source), int(target)
            if (
                source == target
                or source < 0
                or target < 0
                or source >= num_nodes
                or target >= num_nodes
                or tuple(sorted((source, target))) in directed_pairs
            ):
                continue
            ax.plot(
                [layout[source, 0], layout[target, 0]],
                [layout[source, 1], layout[target, 1]],
                color="0.30",
                linestyle=(0, (4, 3)),
                linewidth=1.8,
                alpha=0.95,
                zorder=1,
            )

    for source, target in g2.get_edgelist():
        arrow = FancyArrowPatch(
            posA=tuple(layout[source]),
            posB=tuple(layout[target]),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="0.22",
            shrinkA=float(arrow_shrink[source]),
            shrinkB=float(arrow_shrink[target]),
            zorder=2,
        )
        ax.add_patch(arrow)

    ax.scatter(
        layout[:, 0],
        layout[:, 1],
        s=node_sizes,
        c=hex_colors[: layout.shape[0]],
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    if value_mappable is not None and show_colorbar:
        fig.colorbar(
            value_mappable,
            ax=ax,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
        )
    for node_id, (x_coord, y_coord) in enumerate(layout):
        ax.text(
            x_coord,
            y_coord,
            str(node_id),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)

    _save_plot_if_requested(fig, filename)
    if return_fig:
        return fig
    return None

def plot_lineage_tree_w_piechart(
    adata_mut: AnnData,
    label: str,
    palette: str = "Spectral",
    figsize: tuple[float, float] = (5, 5),
    return_fig: bool = False,
    layout_key: str | None = None,
    filename: str | None = None,
    show_clone_size: bool = False,
    pie_radius: float | None = None,
    pie_radius_range: tuple[float, float] = (0.035, 0.085),
    show_legend: bool = True,
) -> Figure | None:
    """
    Plot lineage tree with pie charts, all backbone edges, and directed arrows.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object with tree data.
    label : str
        Column in `.obs` to use for pie chart composition.
    palette : str, default='Spectral'
        Color palette for categories.
    figsize : tuple, default=(5, 5)
        Figure size.
    return_fig : bool, default=False
        Whether to return the figure object.
    layout_key : str, optional
        Layout key in `.uns` to plot. By default, uses
        `Lineage_tree_coords_view_genetic` when available and falls back to
        `Lineage_tree_coords`.
    filename : str, optional
        Path to save the figure. If None, the figure is not saved.
    show_clone_size : bool, default=False
        If True, scale each pie radius by the number of cells assigned to its
        `ordered_clone`.
    pie_radius : float, optional
        Fixed pie radius used when `show_clone_size=False`. If None, a
        layout-aware default is used.
    pie_radius_range : tuple, default=(0.035, 0.085)
        Minimum and maximum pie radii used when `show_clone_size=True`.

    Returns
    -------
    Figure or None
        Figure object if return_fig is True.
    """
    layout = _get_lineage_tree_plot_layout(adata_mut, layout_key=layout_key)
    g2 = _get_uns_graph_lazy(adata_mut, "Lineage_tree", directed=True)
    type_list, color_dict = _get_plot_color_map(
        values=adata_mut.obs[label],
        palette=palette,
        label=label,
        adata_mut=adata_mut,
    )
    labels_snv = adata_mut.obs["ordered_clone"].astype(int)

    fig, base_ax = plt.subplots(figsize=figsize)
    base_ax.set_aspect("equal", adjustable="box")
    base_ax.axis("off")

    default_pie_radius = min(
        0.055,
        max(0.035, 0.22 / max(np.sqrt(layout.shape[0]), 3.0)),
    )
    if pie_radius is None:
        pie_radius = default_pie_radius
    pie_radii = _get_lineage_tree_node_sizes(
        adata_mut=adata_mut,
        n_nodes=layout.shape[0],
        show_clone_size=show_clone_size,
        default_size=float(pie_radius),
        size_range=pie_radius_range,
    )
    arrow_shrink = 18.0 * pie_radii / max(default_pie_radius, 1e-12)
    custom_legend = [
        Patch(color=color_dict[t], label=str(t))
        for t in type_list
    ]
    if custom_legend and show_legend:
        base_ax.legend(
            handles=custom_legend,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=False,
            fontsize="small",
        )

    for source, target in g2.get_edgelist():
        if source >= layout.shape[0] or target >= layout.shape[0]:
            continue
        arrow = FancyArrowPatch(
            posA=tuple(layout[source]),
            posB=tuple(layout[target]),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="0.22",
            shrinkA=float(arrow_shrink[source]),
            shrinkB=float(arrow_shrink[target]),
            zorder=2,
        )
        base_ax.add_patch(arrow)

    # Root node (normal)
    base_ax.pie(
        [1],
        colors=["grey"],
        center=tuple(layout[0]),
        radius=float(pie_radii[0]),
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    base_ax.text(
        layout[0, 0],
        layout[0, 1],
        "0",
        ha="center",
        va="center",
        fontsize=8,
        zorder=4,
    )

    # Clone nodes
    clone_nodes = range(1, layout.shape[0])
    for i in clone_nodes:
        clone_mask = labels_snv == i
        if not np.any(clone_mask):
            base_ax.pie(
                [1],
                colors=["white"],
                center=tuple(layout[i]),
                radius=float(pie_radii[i]),
                wedgeprops={"edgecolor": "0.6", "linewidth": 1},
            )
            base_ax.text(
                layout[i, 0],
                layout[i, 1],
                str(i),
                ha="center",
                va="center",
                fontsize=8,
                color="0.35",
                zorder=4,
            )
            continue
        labels_in_clone, counts = np.unique(
            adata_mut.obs[label][clone_mask], return_counts=True
        )
        base_ax.pie(
            counts,
            colors=[color_dict[name] for name in labels_in_clone],
            center=tuple(layout[i]),
            radius=float(pie_radii[i]),
            wedgeprops={"edgecolor": "black", "linewidth": 1},
        )
        base_ax.text(
            layout[i, 0],
            layout[i, 1],
            str(i),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    plot_padding = max(0.04, float(np.max(pie_radii)) * 1.35)
    base_ax.set_xlim(-plot_padding, 1.0 + plot_padding)
    base_ax.set_ylim(-plot_padding, 1.0 + plot_padding)
    base_ax.set_aspect("equal", adjustable="box")
    base_ax.set_title(str(label), pad=12)
    base_ax.axis("off")

    _save_plot_if_requested(fig, filename)
    if return_fig:
        return fig
    return None

def plot_embedding(
    adata_mut: AnnData,
    basis: str,
    labels: list[str],
    palette: str = "Spectral",
    cmap: str = "plasma",
    figsize: tuple[float, float] = (5, 5),
    return_fig: bool = False,
    s: float = 15,
    alpha: float = 0.8,
    filename: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    axis_off: bool = False,
    show_legend: bool = True,
    show_colorbar: bool = True,
) -> Figure | None:
    """
    Plot scatterplots of embeddings colored by specified labels.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object.
    basis : str
        Key in `.obsm` containing 2D coordinates.
    labels : list of str
        List of keys in `.obs` to color by.
    palette : str, default='Spectral'
        Color palette for categorical labels.
    cmap : str, default='plasma'
        Colormap for continuous labels.
    figsize : tuple, default=(5, 5)
        Size of each subplot.
    return_fig : bool, default=False
        Whether to return the figure object.
    s : float, default=15
        Marker size.
    alpha : float, default=0.8
        Marker transparency, from 0 (transparent) to 1 (opaque).
    filename : str, optional
        Path to save the figure. If None, the figure is not saved.
    vmin, vmax : float, optional
        Explicit continuous color limits. If omitted, robust percentile limits
        are used.
    axis_off : bool, default=False
        Hide plot axes and ticks when True.

    Returns
    -------
    Figure or None
        Figure object if return_fig is True.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("`alpha` must be between 0 and 1.")
    coords = adata_mut.obsm[basis]
    if coords.shape[1] != 2:
        raise ValueError(
            f"Expected 2D coordinates in `adata_mut.obsm['{basis}']`, "
            f"got shape {coords.shape}"
        )

    n_labels = len(labels)
    side = max(float(figsize[0]), float(figsize[1]))
    fig, axes = plt.subplots(
        1,
        n_labels,
        figsize=(side * n_labels, side),
        squeeze=False,
        constrained_layout=True,
    )

    df_coords = pd.DataFrame(
        coords, columns=["x", "y"], index=adata_mut.obs_names
    )

    for i, label in enumerate(labels):
        ax = axes[0, i]

        if label not in adata_mut.obs.columns:
            raise ValueError(f"Label '{label}' not found in adata_mut.obs")

        label_values = adata_mut.obs[label]
        df_plot = df_coords.copy()
        df_plot[label] = label_values

        if pd.api.types.is_numeric_dtype(label_values):
            # Use robust limits from the central 96% of finite values.
            numeric_values = pd.to_numeric(label_values, errors="coerce").to_numpy(dtype=float)
            finite_values = numeric_values[np.isfinite(numeric_values)]
            if finite_values.size:
                plot_vmin = float(vmin) if vmin is not None else float(np.quantile(finite_values, 0.02))
                plot_vmax = float(vmax) if vmax is not None else float(np.quantile(finite_values, 0.98))
                if not np.isfinite(plot_vmin) or not np.isfinite(plot_vmax):
                    plot_vmin, plot_vmax = float(np.min(finite_values)), float(np.max(finite_values))
                if plot_vmax <= plot_vmin:
                    plot_vmax = plot_vmin + 1.0
            else:
                plot_vmin, plot_vmax = 0.0, 1.0
            sc = ax.scatter(
                df_plot["x"], df_plot["y"], c=numeric_values, cmap=cmap,
                vmin=plot_vmin, vmax=plot_vmax, s=s, alpha=float(alpha)
            )
            if show_colorbar:
                plt.colorbar(sc, ax=ax, label=label)
        else:
            # Use a categorical palette.
            unique_cats, color_dict = _get_plot_color_map(
                values=df_plot[label],
                palette=palette,
                label=label,
                adata_mut=adata_mut,
            )
            for cat in unique_cats:
                sub = df_plot[df_plot[label] == cat]
                ax.scatter(
                    sub["x"],
                    sub["y"],
                    color=color_dict[cat],
                    label=str(cat),
                    s=s,
                    alpha=float(alpha),
                )
            if show_legend:
                ax.legend(
                    title=label,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0,
                    fontsize="small",
                    frameon=False,
                )
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="box")
        if axis_off:
            ax.axis("off")

    _save_plot_if_requested(fig, filename)
    plt.show()

    if return_fig:
        return fig
    return None
