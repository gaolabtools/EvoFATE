"""Shared helpers for EvoFATE genetic modules."""

from __future__ import annotations

import numpy as np
import igraph as ig


def _as_dense_array(matrix: object) -> np.ndarray:
    """Return a dense NumPy array from dense or sparse matrix-like input."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _as_string_array(values: object) -> np.ndarray:
    """Return an array with only Python string objects, preserving shape."""
    array = np.asarray(values, dtype=object)

    def stringify(value: object) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    return np.vectorize(stringify, otypes=[object])(array)


def _serialize_igraph_graph(graph: ig.Graph) -> dict[str, object]:
    """Convert an igraph graph into an AnnData/HDF5-safe edge-list payload."""
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        edges = edges.reshape(-1, 2)
    return {
        "format": "igraph_edgelist",
        "n_nodes": int(graph.vcount()),
        "directed": bool(graph.is_directed()),
        "edges": edges,
    }


def _set_uns_graph(adata_mut, key: str, graph: ig.Graph) -> None:
    """Store an igraph graph in `.uns` using only h5ad-safe values."""
    adata_mut.uns[key] = _serialize_igraph_graph(graph)


def _get_uns_graph(
    adata_mut,
    key: str,
    directed: bool | None = None,
    required: bool = True,
) -> ig.Graph | None:
    """
    Return an igraph graph from an h5ad-safe `.uns` graph payload.

    Older in-memory AnnData objects may still contain raw `igraph.Graph` values.
    When encountered, they are converted back to the safe payload in `.uns`.
    """
    if key not in adata_mut.uns:
        if required:
            raise KeyError(f"`adata_mut.uns['{key}']` is missing.")
        return None

    value = adata_mut.uns[key]
    if isinstance(value, ig.Graph):
        graph = value
        if directed is not None and graph.is_directed() != bool(directed):
            graph = graph.as_directed() if directed else graph.as_undirected()
        _set_uns_graph(adata_mut, key, graph)
        return graph

    if not isinstance(value, dict):
        raise TypeError(
            f"`adata_mut.uns['{key}']` must be an igraph graph payload."
        )

    edges = np.asarray(value.get("edges", np.empty((0, 2))), dtype=np.int64)
    if edges.size == 0:
        edge_list: list[tuple[int, int]] = []
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        edges = edges.reshape(-1, 2)
        edge_list = [(int(source), int(target)) for source, target in edges]

    if "n_nodes" in value:
        n_nodes = int(np.asarray(value["n_nodes"]).item())
    elif edge_list:
        n_nodes = max(max(edge) for edge in edge_list) + 1
    else:
        n_nodes = 0

    if directed is None:
        directed_value = value.get("directed", False)
        directed = bool(np.asarray(directed_value).item())

    graph = ig.Graph(n=n_nodes, edges=edge_list, directed=bool(directed))
    graph.simplify()

    safe_payload = _serialize_igraph_graph(graph)
    if (
        value.get("format") != safe_payload["format"]
        or not np.array_equal(edges, safe_payload["edges"])
        or int(np.asarray(value.get("n_nodes", n_nodes)).item()) != n_nodes
        or bool(np.asarray(value.get("directed", directed)).item()) != bool(directed)
    ):
        adata_mut.uns[key] = safe_payload
    return graph
