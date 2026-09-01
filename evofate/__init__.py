"""EvoFATE: evolutionary and fate trajectory estimation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "Yueying He"

_LAZY_IMPORTS = {
    "tl": ("evofate.tl", None),
    "pl": ("evofate.pl", None),
    "cal_genetic_connectivities": ("evofate.tl", "cal_genetic_connectivities"),
    "define_clones": ("evofate.tl", "define_clones"),
    "cal_consensus_profile": ("evofate.tl", "cal_consensus_profile"),
    "cal_clone_connectivity": ("evofate.tl", "cal_clone_connectivity"),
    "cal_tree_layout": ("evofate.tl", "cal_tree_layout"),
    "cal_evofate_embedding": ("evofate.tl", "cal_evofate_embedding"),
    "cal_linear_projection": ("evofate.tl", "cal_linear_projection"),
    "cal_lineage_guided_projection": ("evofate.tl", "cal_lineage_guided_projection"),
    "cal_clonal_evofate": ("evofate.tl", "cal_clonal_evofate"),
    "cal_single_cell_evofate": ("evofate.tl", "cal_single_cell_evofate"),
    "cal_progression_features": ("evofate.tl", "cal_progression_features"),
    "select_progression_features": ("evofate.tl", "select_progression_features"),
    "plot_consensus_profile": ("evofate.pl", "plot_consensus_profile"),
    "plot_filtered_mutations": ("evofate.pl", "plot_filtered_mutations"),
    "plot_lineage_tree": ("evofate.pl", "plot_lineage_tree"),
    "plot_lineage_tree_w_piechart": ("evofate.pl", "plot_lineage_tree_w_piechart"),
    "plot_embedding": ("evofate.pl", "plot_embedding"),
    "plot_clonal_evofate": ("evofate.pl", "plot_clonal_evofate"),
    "plot_single_cell_evofate": ("evofate.pl", "plot_single_cell_evofate"),
    "plot_progression_features": ("evofate.pl", "plot_progression_features"),
}

__all__ = ["__version__", "tl", "pl", *[name for name in _LAZY_IMPORTS if name not in {"tl", "pl"}]]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'evofate' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
