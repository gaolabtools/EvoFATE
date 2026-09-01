"""EvoFATE plotting functions."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib.patches import FancyArrowPatch

from matplotlib.lines import Line2D

from matplotlib.colors import to_hex

from ._plotting import _get_plot_color_map

def _lineage_graph(value) -> nx.DiGraph:
    """Convert supported lineage representations to a directed graph."""
    if isinstance(value, nx.DiGraph):
        return value.copy()
    graph = nx.DiGraph()
    if value is None:
        raise KeyError("`adata.uns['Lineage_tree']` is missing.")
    if hasattr(value, "get_edgelist"):
        graph.add_nodes_from(range(int(value.vcount())))
        graph.add_edges_from(value.get_edgelist())
        return graph
    if isinstance(value, Mapping):
        nodes = value.get("nodes")
        if nodes is not None:
            graph.add_nodes_from(nodes)
        edges = np.asarray(value.get("edges", np.empty((0, 2))), dtype=object)
        if edges.size:
            graph.add_edges_from(edges.reshape(-1, 2).tolist())
        return graph
    raise TypeError("Lineage must be a NetworkX, igraph, or serialized graph object.")

def _evofate_layout_positions(adata, layout_key, graph):
    value = adata.uns.get(layout_key)
    if value is None:
        raise KeyError(f"`adata.uns['{layout_key}']` is missing.")
    if isinstance(value, Mapping) and "coordinates" in value:
        coords = np.asarray(value["coordinates"], dtype=float)
        order = [str(item) for item in value.get("clone_order", list(graph.nodes))]
        return {clone: coords[index, :2] for index, clone in enumerate(order)}
    if isinstance(value, Mapping):
        return {str(clone): np.asarray(point, dtype=float)[:2] for clone, point in value.items()}
    coords = np.asarray(value, dtype=float)
    nodes = list(graph.nodes)
    numeric = all(str(node).lstrip("-").isdigit() for node in nodes)
    if numeric:
        nodes = sorted(nodes, key=lambda node: int(node))
    if coords.ndim != 2 or coords.shape[0] < len(nodes) or coords.shape[1] < 2:
        raise ValueError("Clone layout must contain one 2D coordinate per lineage node.")
    return {str(node): coords[index, :2] for index, node in enumerate(nodes)}

def plot_clonal_evofate(
    adata,
    layout_key: str = "Lineage_tree_coords_view_genetic",
    lineage=None,
    result_key: str = "evofate_clonal",
    node_size: float = 180.0,
    edge_width_range: tuple[float, float] = (0.8, 4.0),
    cmap: str = "plasma",
    ax=None,
    figsize: tuple[float, float] = (6, 5),
    show: bool = True,
    filename: str | None = None,
    dpi: int = 300,
    show_legend: bool = True,
    show_colorbar: bool = True,
):
    """Plot clonal state dispersion and directed state change on the clone layout."""
    if result_key not in adata.uns:
        raise KeyError(f"`adata.uns['{result_key}']` is missing. Run tl.cal_clonal_evofate first.")
    graph = _lineage_graph(lineage if lineage is not None else adata.uns.get("Lineage_tree"))
    graph = nx.relabel_nodes(graph, lambda node: str(node))
    positions = _evofate_layout_positions(adata, layout_key, graph)
    result = adata.uns[result_key]
    plasticity = {str(key): float(value) for key, value in result["intra_plasticity"].items()}
    changes = {str(key): float(value) for key, value in result["edge_state_change"].items()}
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    finite = np.asarray(list(plasticity.values()), dtype=float)
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size and np.nanmax(finite) > vmin else vmin + 1.0
    edge_values = np.asarray(list(changes.values()), dtype=float)
    edge_min = float(np.nanmin(edge_values)) if edge_values.size else 0.0
    edge_max = float(np.nanmax(edge_values)) if edge_values.size and np.nanmax(edge_values) > edge_min else edge_min + 1.0
    lo, hi = edge_width_range
    for parent, child in graph.edges:
        key = f"{parent}->{child}"
        if parent not in positions or child not in positions:
            continue
        value = changes.get(key, edge_min)
        width = lo + (hi - lo) * (value - edge_min) / (edge_max - edge_min)
        ax.annotate("", xy=positions[child], xytext=positions[parent],
                    arrowprops={"arrowstyle": "-|>", "color": "0.55", "lw": width,
                                "shrinkA": 8, "shrinkB": 8})
    if edge_values.size and show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color="0.55", lw=lo, label="Small state change"),
                Line2D([0], [0], color="0.55", lw=hi, label="Large state change"),
            ],
            title="Edge width",
            loc="best",
            frameon=False,
        )
    nodes = [node for node in graph.nodes if node in positions and node in plasticity]
    scatter = ax.scatter([positions[node][0] for node in nodes], [positions[node][1] for node in nodes],
                         c=[plasticity[node] for node in nodes], cmap=cmap, vmin=vmin, vmax=vmax,
                         s=node_size, edgecolors="none", zorder=3)
    if show_colorbar:
        ax.figure.colorbar(scatter, ax=ax, label="State dispersion")
    ax.set_axis_off(); ax.set_aspect("equal", adjustable="datalim")
    if filename is not None: ax.figure.savefig(filename, bbox_inches="tight", dpi=dpi)
    if show: plt.show()
    return ax.figure

def _progression_heatmap(
    adata,
    n_features,
    features,
    feature_label,
    cmap,
    figsize,
    filename,
    show,
    return_fig,
    dpi,
    cluster_genes,
    cluster_method,
    cluster_metric,
    include_global_rank,
    significant_only,
    fdr_threshold,
    min_effect,
    pvalue_threshold,
    show_legend,
    show_colorbar,
    standard_scale,
    cbar_pos,
    show_lineage_labels,
    feature_fontsize,
    vmin,
    vmax,
    result_key,
):
    if result_key not in adata.uns:
        raise KeyError(f"`adata.uns['{result_key}']` is missing. Run tl.cal_progression_features() first.")
    if "evofate_lineage_support" not in adata.obsm or "evofate_lineage_progression" not in adata.obsm:
        raise ValueError("Lineage progression is not available. Run tl.cal_single_cell_evofate() first.")
    support = np.asarray(adata.obsm["evofate_lineage_support"], dtype=float)
    progression = np.asarray(adata.obsm["evofate_lineage_progression"], dtype=float)
    membership = np.asarray(
        adata.obsm.get("evofate_lineage_membership", support > 0),
        dtype=bool,
    )
    lineages = list(adata.uns["evofate_lineages"]["names"])
    if support.ndim != 2 or progression.shape != support.shape or membership.shape != support.shape or support.shape[0] != adata.n_obs or len(lineages) != support.shape[1]:
        raise ValueError(
            "Stored EvoFATE lineage support and progression matrices are invalid."
        )
    result = adata.uns[result_key]
    if fdr_threshold is not None and not 0.0 <= float(fdr_threshold) <= 1.0:
        raise ValueError("fdr_threshold must be between 0 and 1 or None.")
    if pvalue_threshold is not None and not 0.0 <= float(pvalue_threshold) <= 1.0:
        raise ValueError("pvalue_threshold must be between 0 and 1 or None.")
    if min_effect is not None and float(min_effect) < 0.0:
        raise ValueError("min_effect must be nonnegative or None.")
    clone_paths = adata.uns.get("evofate_lineages", {}).get("clone_paths", {})
    stored_cell_order = adata.uns.get("evofate_lineage_order", {})
    plot_clone_key = result.get("params", {}).get("clone_key")
    if plot_clone_key not in adata.obs:
        plot_clone_key = next(
            (candidate for candidate in ("ordered_clone", "clone", "SNV_clone", "clone_id")
             if candidate in adata.obs),
            None,
        )
    if features is None:
        selected = result.get("important_features")
        if selected is None:
            raise ValueError(
                "No saved important features found. Run "
                "tl.select_progression_features() first or pass features explicitly."
            )
        selected = [str(gene) for gene in selected]
    else:
        selected = [str(gene) for gene in features]
    metadata = result.get("feature_metadata", {})
    metadata_values = metadata.get(feature_label, {}) if feature_label else {}
    blocks = []
    block_lineages = []
    ordered_cell_indices = []
    ordered_cells_by_lineage = {}
    for column, lineage in enumerate(lineages):
        if lineage not in stored_cell_order:
            raise ValueError(
                "Stored lineage cell order is missing. Run tl.cal_single_cell_evofate() first."
            )
        cells = np.asarray(stored_cell_order[lineage], dtype=int)
        cells = cells[
            membership[cells, column] & np.isfinite(progression[cells, column])
        ]
        if cells.size == 0:
            continue
        ordered_cell_indices.extend(cells.tolist())
        ordered_cells_by_lineage[str(lineage)] = cells.astype(int).tolist()
        block = np.full((len(selected), cells.size), np.nan, dtype=float)
        lineage_record = result["lineages"].get(lineage, {})
        compact_names = [str(name) for name in lineage_record.get("feature_names", [])]
        compact_index = {name: index for index, name in enumerate(compact_names)}
        compact_matrix = lineage_record.get("observed_matrix")
        record_cells = np.asarray(lineage_record.get("ordered_cells", []), dtype=int)
        record_positions = {int(cell): index for index, cell in enumerate(record_cells)}
        fitted_record = lineage_record.get("fitted_values", {})
        for row, gene in enumerate(selected):
            if compact_matrix is not None and gene in compact_index:
                source_row = compact_index[gene]
                values = np.asarray(compact_matrix[source_row], dtype=float)
                positions = [record_positions.get(int(cell)) for cell in cells]
                if all(position is not None for position in positions):
                    block[row] = values[np.asarray(positions, dtype=int)]
                continue
            item = fitted_record.get(gene)
            if item is None:
                continue
            index = np.asarray(item["cell_indices"], dtype=int)
            values = np.asarray(item.get("observed_values", []), dtype=float)
            if index.size == 0 or values.size == 0:
                continue
            order = np.argsort(progression[index, column], kind="stable")
            t = progression[index[order], column]
            values = values[order]
            unique_t, unique_index = np.unique(t, return_index=True)
            if unique_t.size == 1:
                block[row] = values[unique_index[0]]
            else:
                block[row] = np.interp(
                    progression[cells, column],
                    unique_t,
                    values[unique_index],
                )
        blocks.append(block)
        block_lineages.append(lineage)
    if not blocks:
        raise ValueError("No lineage blocks contain included cells.")
    result["cell_order"] = np.asarray(ordered_cell_indices, dtype=int)
    result["cell_order_names"] = [
        str(adata.obs_names[index]) for index in ordered_cell_indices
    ]
    result["cell_order_by_lineage"] = ordered_cells_by_lineage
    matrix = np.concatenate(blocks, axis=1)
    if standard_scale in {"gene", "row"}:
        center = np.nanmean(matrix, axis=1, keepdims=True)
        scale = np.nanstd(matrix, axis=1, keepdims=True)
        matrix = (matrix - center) / np.where(scale > 1e-12, scale, 1.0)
    elif standard_scale not in {None, "none"}:
        raise ValueError("standard_scale must be 'gene', 'row', None, or 'none'.")
        # Use zero for features unavailable in a lineage block.
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    row_labels = [
        str(metadata_values.get(gene, gene))
        for gene in selected
    ]
    palette = {
        lineage: plt.get_cmap("tab20")(index % 20)
        for index, lineage in enumerate(lineages)
    }
    col_labels = np.concatenate([
        np.repeat(lineage, block.shape[1])
        for lineage, block in zip(block_lineages, blocks)
    ])
    clone_key = plot_clone_key
    saved_clone_colors = None
    saved_clone_labels = None
    if clone_key not in adata.obs:
        clone_key = next(
            (candidate for candidate in ("ordered_clone", "clone", "SNV_clone", "clone_id")
             if candidate in adata.obs),
            None,
        )
    if clone_key is not None:
        cell_clones = adata.obs[clone_key].astype(str).to_numpy()
        try:
            _, clone_palette = _get_plot_color_map(
                values=cell_clones,
                palette="tab20",
                label=clone_key,
                adata_mut=adata,
                include_normal=True,
            )
        except KeyError:
            # Use a standard categorical palette when no consensus palette exists.
            clone_palette = {
                clone: plt.get_cmap("tab20")(index % 20)
                for index, clone in enumerate(np.unique(cell_clones))
            }
        clone_colors = []
        clone_labels_ordered = []
        for lineage, block in zip(block_lineages, blocks):
            column = lineages.index(lineage)
            cells = np.asarray(stored_cell_order[lineage], dtype=int)
            cells = cells[
                membership[cells, column] & np.isfinite(progression[cells, column])
            ]
            clone_colors.extend(clone_palette[cell_clones[cell]] for cell in cells)
            clone_labels_ordered.extend(str(cell_clones[cell]) for cell in cells)
        saved_clone_colors = [to_hex(color) for color in clone_colors]
        saved_clone_labels = clone_labels_ordered
        col_colors = pd.DataFrame({
            "lineage": [to_hex(palette[lineage]) for lineage in col_labels],
            "clone": clone_colors,
        })
    else:
        col_colors = pd.DataFrame({
            "lineage": [to_hex(palette[lineage]) for lineage in col_labels],
        })
    block_ranges = {}
    start = 0
    for lineage, block in zip(block_lineages, blocks):
        stop = start + block.shape[1]
        block_ranges[str(lineage)] = [int(start), int(stop)]
        start = stop
    result["plotting"] = {
        "cell_order": [int(index) for index in ordered_cell_indices],
        "cell_order_names": [str(adata.obs_names[index]) for index in ordered_cell_indices],
        "cell_order_by_lineage": ordered_cells_by_lineage,
        "block_lineages": [str(lineage) for lineage in block_lineages],
        "block_ranges": block_ranges,
        "block_colors": {
            str(lineage): to_hex(palette[lineage]) for lineage in block_lineages
        },
        "column_lineage": [str(lineage) for lineage in col_labels],
        "clone_key": clone_key,
        "column_clone": saved_clone_labels,
        "clone_colors": saved_clone_colors,
        "parameters": {
            "cmap": str(cmap),
            "figsize": None if figsize is None else [float(value) for value in figsize],
            "cluster_genes": bool(cluster_genes),
            "cluster_method": str(cluster_method),
            "cluster_metric": str(cluster_metric),
            "standard_scale": standard_scale,
            "vmin": None if vmin is None else float(vmin),
            "vmax": None if vmax is None else float(vmax),
            "feature_fontsize": float(feature_fontsize),
            "show_lineage_labels": bool(show_lineage_labels),
            "show_colorbar": bool(show_colorbar),
            "cbar_pos": [float(value) for value in cbar_pos],
        },
    }
    row_colors = None
    matrix_for_plot = pd.DataFrame(
        matrix,
        index=np.arange(matrix.shape[0]),
        columns=np.arange(matrix.shape[1]),
    )
    col_colors.index = matrix_for_plot.columns
    grid = sns.clustermap(
        matrix_for_plot,
        row_cluster=bool(cluster_genes),
        method=cluster_method,
        metric=cluster_metric,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=row_labels,
        figsize=figsize or (8, 8),
        cbar_pos=cbar_pos if show_colorbar else None,
        cbar_kws={
            "label": "Observed expression"
        },
    )
    if not show_colorbar:
        # Preserve the clustermap layout when the colorbar is hidden.
        for axis in grid.fig.axes:
            if axis is not grid.ax_heatmap and axis is not getattr(grid, "ax_row_dendrogram", None) and axis is not getattr(grid, "ax_col_dendrogram", None):
                axis.set_visible(False)
    grid.ax_heatmap.set_xlabel("Lineage blocks ordered by progression (early to late)")
    grid.ax_heatmap.tick_params(axis="y", labelsize=feature_fontsize)
    row_dendrogram = getattr(grid, "dendrogram_row", None)
    row_order = getattr(row_dendrogram, "reordered_ind", None)
    if row_order is None:
        row_order = list(range(len(selected)))
    result["important_features"] = [
        str(selected[index]) for index in row_order
    ]
    offset = 0
    color_axis = getattr(grid, "ax_col_colors", None)
    for block, lineage in zip(blocks, block_lineages):
        path = clone_paths.get(lineage, [])
        # Normalize lineage paths and identify populated blocks.
        path_values = np.asarray(path).reshape(-1) if path is not None else np.array([])
        lineage_label = (
            "-".join(str(clone) for clone in path_values)
            if path_values.size > 0
            else str(lineage)
        )
        if color_axis is not None and show_lineage_labels and show_legend:
            color_axis.text(
                offset + block.shape[1] / 2.0,
                1.15,
                lineage_label,
                transform=color_axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                clip_on=False,
            )
        offset += block.shape[1]
        if offset < matrix.shape[1]:
            grid.ax_heatmap.axvline(offset - 0.5, color="white", linewidth=1.5, zorder=5)
    fig = grid.fig
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    return grid if return_fig else None

def plot_single_cell_evofate(*args, **kwargs):
    """Plot state variation and stored lineage progression together."""
    adata = args[0] if args else kwargs.pop("adata_snv")
    filename = kwargs.pop("filename", None)
    show = kwargs.pop("show", True)
    embedding_key = kwargs.pop("embedding_key", "X_evofate_projection")
    result_key = kwargs.pop("result_key", "evofate_trajectory")
    label = kwargs.pop("color", "state_variation")
    n_arrows = max(0, int(kwargs.pop("n_arrows", 3)))
    arrow_scale = float(kwargs.pop("arrow_scale", 10.0))
    line_color = kwargs.pop("line_color", "0.2")
    show_legend = kwargs.pop("show_legend", True)
    show_colorbar = kwargs.pop("show_colorbar", True)
    dpi = kwargs.pop("dpi", 300)
    kwargs.setdefault("cmap", "plasma")
    if label not in adata.obs:
        raise KeyError(f"`adata.obs['{label}']` is missing. Run tl.cal_single_cell_evofate first.")
    fig = plot_embedding(
        adata,
        basis=embedding_key,
        labels=[label],
        return_fig=True,
        filename=None,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        **kwargs,
    )
    result = adata.uns.get(result_key, {})
    axis = fig.axes[0]
    for record in result.get("lineages", {}).values():
        curve = np.asarray(record.get("curve_xy", []), dtype=float)
        if curve.ndim == 2 and curve.shape[0] > 1:
            axis.plot(curve[:, 0], curve[:, 1], color=line_color, lw=1.8, zorder=4)
            if n_arrows:
                fractions = np.linspace(0.25, 0.80, n_arrows)
                for fraction in fractions:
                    index = min(curve.shape[0] - 1, max(1, int(round(fraction * (curve.shape[0] - 1)))))
                    axis.annotate(
                        "",
                        xy=curve[index],
                        xytext=curve[index - 1],
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": line_color,
                            "lw": 1.2,
                            "mutation_scale": arrow_scale,
                        },
                    )
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    return fig

def plot_progression_features(
    adata_snv,
    features=None,
    min_lineage_support=0.0,
    feature_label=None,
    cmap="RdBu",
    figsize=None,
    filename=None,
    show=True,
    return_fig=True,
    dpi=300,
    max_cells_per_lineage=None,
    cluster_genes=True,
    cluster_method="average",
    cluster_metric="euclidean",
    show_legend=True,
    show_colorbar=True,
    standard_scale="gene",
    cbar_pos=(0.02, 0.80, 0.05, 0.18),
    show_lineage_labels=True,
    feature_fontsize=8,
    vmin=None,
    vmax=None,
    result_key="progression_genes",
):
    saved_features = adata_snv.uns.get(result_key, {}).get("important_features", [])
    feature_count = len(features) if features is not None else len(saved_features)
    return _progression_heatmap(
        adata_snv, feature_count, features, feature_label, cmap, figsize, filename,
        show, return_fig, dpi, bool(cluster_genes),
        cluster_method, cluster_metric, False, None, None, None, None, bool(show_legend), bool(show_colorbar), standard_scale, cbar_pos,
        bool(show_lineage_labels), feature_fontsize, vmin, vmax, result_key,
    )


from ._plotting import (
    plot_consensus_profile, plot_filtered_mutations, plot_lineage_tree,
    plot_lineage_tree_w_piechart, plot_embedding,
)

__all__ = [
    "plot_consensus_profile", "plot_filtered_mutations", "plot_lineage_tree",
    "plot_lineage_tree_w_piechart", "plot_embedding",
    "plot_clonal_evofate", "plot_single_cell_evofate",
    "plot_progression_features",
]
