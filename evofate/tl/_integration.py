"""
EvoFATE Integration module.

This module integrates transcriptomic profiles with genetic structure using
BGRL (Bootstrapped Graph Representation Learning) with a GAT backbone.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from typing import TYPE_CHECKING

_CACHE_TMPDIR = tempfile.gettempdir()
for _env_key, _cache_path in {
    "NUMBA_CACHE_DIR": os.path.join(_CACHE_TMPDIR, "evofate_numba_cache"),
    "MPLCONFIGDIR": os.path.join(_CACHE_TMPDIR, "evofate_mpl_cache"),
}.items():
    if not os.environ.get(_env_key):
        os.environ[_env_key] = _cache_path
    os.makedirs(os.environ[_env_key], exist_ok=True)
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
for _thread_env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env_key, "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter

if TYPE_CHECKING:
    from anndata import AnnData


DEFAULT_H0_STRENGTH = 0.10
DEFAULT_H1_STRENGTH = 0.20
DEFAULT_H0_MODE = "trainable_linear"
DEFAULT_BRANCH_NORM = "layernorm"


def cal_evofate_embedding(
    adata_mut: AnnData,
    epochs: int = 1000,
    lr: float = 3e-4,
    encoder_lr: float | None = None,
    predictor_lr: float | None = None,
    heads: int = 2,
    dim: int | None = None,
    dropout: float = 0.2,
    momentum: float = 0.99,
    momentum_final: float | None = 0.999,
    edge_drop_prob: float = 0.05,
    noise_std: float = 0.01,
    rna_augmentation_alpha: float = 0.1,
    rna_relative_weight: float = 0.20,
    rna_relative_margin: float = 0.1,
    rna_positive_neighbors: int = 20,
    rna_positives_per_anchor: int = 2,
    rna_negative_candidates: int = 16,
    rna_loss_warmup_epochs: int = 10,
    rna_loss_ramp_epochs: int = 20,
    h0_strength: float = DEFAULT_H0_STRENGTH,
    h1_strength: float = DEFAULT_H1_STRENGTH,
    h0_mode: str = DEFAULT_H0_MODE,
    branch_norm: str = DEFAULT_BRANCH_NORM,
    fixed_h0: Tensor | np.ndarray | None = None,
    rna_n_neighbors: int | None = None,
    rna_metric: str = "euclidean",
    rna_connectivity_key: str = "rna_connectivities",
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    max_grad_norm: float | None = 1.0,
    lr_scheduler: str | None = "warmup_cosine",
    warmup_epochs: int = 10,
    print_interval: int = 100,
    drop_undirected_edges: bool = True,
    preserve_self_loops: bool = True,
    seed: int | None = 0,
    deterministic: bool = False,
    gat_concat: bool = False,
    encoder_layer_norm: bool = True,
    predictor_norm: str = "layer",
    diagnostics: bool = False,
    diagnostics_interval: int | None = None,
    diagnostics_key: str = "evofate_training_diagnostics",
    verbose: bool = True,
    continue_training: bool = False,
    training_state_key: str = "evofate_training_state",
    store_training_state: bool = False,
    device: str | torch.device | None = "auto",
) -> None:
    """
    Calculate EvoFATE embeddings by integrating genetic and transcriptomic data.

    This function uses BGRL (Bootstrapped Graph Representation Learning) with
    a GAT (Graph Attention Network) backbone to learn joint embeddings that
    capture both genetic relationships and transcriptomic profiles.

    Parameters
    ----------
    adata_mut : AnnData
        Annotated data object containing:
        - `.obsp['genetic_lineage_connectivity']`: Genetic connectivity matrix
        - `.obsm['X_rna']`: PCA-reduced expression features
        - `.uns['dim']`: Target embedding dimension
    epochs : int, default=1000
        Number of training epochs.
    lr : float, default=3e-4
        Base learning rate, used for the predictor by default.
    encoder_lr : float, optional
        Learning rate for the GAT encoder. Defaults to `min(lr, 1e-4)`.
    predictor_lr : float, optional
        Learning rate for the predictor. Defaults to `lr`.
    heads : int, default=2
        Number of parameter-matched attention heads in the graph-backbone GAT.
    dim : int, optional
        Total target embedding dimension. Defaults to `adata_mut.uns['dim']`.
    dropout : float, default=0.2
        Dropout rate for GAT attention coefficients.
    momentum : float, default=0.99
        Initial exponential moving average momentum for the target encoder.
    momentum_final : float, optional, default=0.999
        Final EMA momentum for cosine scheduling. Use None for constant
        momentum.
    edge_drop_prob : float, default=0.05
        Probability of dropping graph edges in the augmented online view.
    noise_std : float, default=0.01
        Standard deviation multiplier for PCA-scaled feature noise in the
        augmented online view.
    rna_augmentation_alpha : float, default=0.1
        Fraction of the sampled RNA-neighbor direction added to each online
        augmented feature, using `x_i + alpha * (x_j - x_i)`. Set to 0 to
        disable RNA-neighborhood feature augmentation.
    rna_relative_weight : float, default=0.20
        Final coefficient of the weak RNA-neighborhood ranking loss.
    rna_relative_margin : float, default=0.1
        Cosine-distance margin for RNA ranking triplets.
    rna_positive_neighbors : int, default=20
        Maximum number of RNA neighbors retained per anchor.
    rna_positives_per_anchor : int, default=2
        Number of RNA-positive samples per anchor and epoch.
    rna_negative_candidates : int, default=16
        Number of sampled negative candidates per RNA-positive pair.
    rna_loss_warmup_epochs : int, default=10
        Epochs before the RNA ranking loss is activated.
    rna_loss_ramp_epochs : int, default=20
        Number of epochs used to ramp the RNA ranking coefficient.
    h0_strength : float, default=0.10
        Fixed coefficient for the graph-independent RNA input skip.
    h1_strength : float, default=0.20
        Fixed coefficient for the first GAT layer skip.
    h0_mode : {'trainable_linear', 'fixed'}, default='trainable_linear'
        Whether h0 is computed by a trainable linear RNA projection or supplied
        through `fixed_h0`.
    branch_norm : {'layernorm', 'l2', 'none'}, default='layernorm'
        Normalization applied separately to h0, h1, and h2 before fusion.
    fixed_h0 : array-like or Tensor, optional
        Fixed graph-independent RNA representation with shape
        `(n_cells, embedding_dim)`, required when `h0_mode='fixed'`.
    rna_n_neighbors : int, optional
        Number of Scanpy neighbors used to build RNA connectivity from
        `.obsm['X_rna']`. Defaults to Scanpy's usual 15, clipped to
        the number of cells.
    rna_metric : str, default='euclidean'
        Distance metric passed to `scanpy.pp.neighbors` for RNA connectivity.
    rna_connectivity_key : str, default='rna_connectivities'
        `.obsp` key where the Scanpy RNA connectivity graph is stored.
    weight_decay : float, default=1e-4
        Weight decay for Adam/AdamW.
    optimizer_name : {'adamw', 'adam'}, default='adamw'
        Optimizer used for trainable online encoder and predictor parameters.
    max_grad_norm : float, optional, default=1.0
        If provided, clip trainable gradient norm before optimizer step.
    lr_scheduler : {'warmup_cosine', 'constant', None}, default='warmup_cosine'
        Per-step learning-rate scheduler.
    print_interval : int, default=100
        Number of epochs between progress messages.
    drop_undirected_edges : bool, default=True
        Whether reciprocal graph edges are dropped together.
    seed : int, optional, default=0
        Seed for model initialization and training stochasticity.
    gat_concat : bool, default=False
        Deprecated compatibility option. The GAT encoder keeps the
        final output dimension fixed and always concatenates internal heads.
    diagnostics : bool, default=False
        Whether to store h5ad-safe training diagnostics in `.uns`.
    verbose : bool, default=True
        Whether to print training progress.
    continue_training : bool, default=False
        If True, resume training from an in-memory, resumable PyTorch state in
        `adata_mut.uns[training_state_key]`. H5AD-safe metadata summaries cannot
        be used for continued training.
    training_state_key : str, default='evofate_training_state'
        Key in `.uns` used for optional training metadata.
    store_training_state : bool, default=False
        Whether to store an h5ad-safe training metadata summary in
        `.uns[training_state_key]`. Raw PyTorch model and optimizer states are
        never stored in AnnData because they are not h5ad-serializable.
    device : str, torch.device, or None, default='auto'
        Device used for model training. The default uses CUDA when available,
        then Apple MPS on Mac, and falls back to CPU. Use 'gpu' to require a
        GPU, 'mps' to force Mac GPU, or 'cpu' to force CPU.

    Returns
    -------
    None
        Modifies `adata_mut` in place:
        - `.obsm['X_integrated']`: Learned EvoFATE integrated embeddings
    """
    torch_device = _resolve_torch_device(device)
    if verbose:
        _print_torch_device_info(device, torch_device)

    # Extract aligned weighted genetic connectivity edges.
    edge_index, _ = _connectivity_to_edge_tensors(
        adata_mut.obsp["genetic_lineage_connectivity"]
    )

    # Extract transcriptomic features
    exp_features = torch.FloatTensor(adata_mut.obsm["X_rna"])

    rna_connectivities = _cal_scanpy_rna_connectivity(
        adata_mut,
        expression_key="X_rna",
        connectivity_key=rna_connectivity_key,
        n_neighbors=rna_n_neighbors,
        metric=rna_metric,
    )
    rna_edge_index, rna_edge_weight = _connectivity_to_edge_tensors(
        rna_connectivities
    )

    # Create PyTorch Geometric data object
    data = Data(
        x=exp_features,
        edge_index=edge_index,
        rna_edge_index=rna_edge_index,
        rna_edge_weight=rna_edge_weight,
    )
    data.rna_triplets = _prepare_rna_triplets(
        rna_edge_index=rna_edge_index,
        rna_edge_weight=rna_edge_weight,
        genetic_edge_index=edge_index,
        num_nodes=exp_features.size(0),
        positive_neighbors=rna_positive_neighbors,
        positives_per_anchor=rna_positives_per_anchor,
        negative_candidates=rna_negative_candidates,
        seed=seed,
    )

    initial_training_state = None
    if continue_training:
        if training_state_key not in adata_mut.uns:
            raise KeyError(
                f"`adata_mut.uns['{training_state_key}']` is missing. "
                "Run `cal_evofate_embedding` once with "
                "`store_training_state=True`, or set "
                "`continue_training=False` to start fresh."
            )
        initial_training_state = adata_mut.uns[training_state_key]
        if not _is_resumable_evofate_training_state(initial_training_state):
            raise ValueError(
                f"`adata_mut.uns['{training_state_key}']` does not contain a "
                "resumable PyTorch model/optimizer state. AnnData now stores "
                "only h5ad-safe EvoFATE training metadata; start fresh with "
                "`continue_training=False`."
            )
    else:
        adata_mut.uns.pop(training_state_key, None)

    # Train EvoFATE model
    train_result = train_evofate(
        data,
        embedding_dim=adata_mut.uns["dim"] if dim is None else dim,
        epochs=epochs,
        lr=lr,
        encoder_lr=encoder_lr,
        predictor_lr=predictor_lr,
        heads=heads,
        dropout=dropout,
        momentum=momentum,
        momentum_final=momentum_final,
        edge_drop_prob=edge_drop_prob,
        noise_std=noise_std,
        rna_augmentation_alpha=rna_augmentation_alpha,
        rna_relative_weight=rna_relative_weight,
        rna_relative_margin=rna_relative_margin,
        rna_loss_warmup_epochs=rna_loss_warmup_epochs,
        rna_loss_ramp_epochs=rna_loss_ramp_epochs,
        h0_strength=h0_strength,
        h1_strength=h1_strength,
        h0_mode=h0_mode,
        branch_norm=branch_norm,
        fixed_h0=fixed_h0,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        max_grad_norm=max_grad_norm,
        lr_scheduler=lr_scheduler,
        warmup_epochs=warmup_epochs,
        print_interval=print_interval,
        drop_undirected_edges=drop_undirected_edges,
        preserve_self_loops=preserve_self_loops,
        seed=seed,
        deterministic=deterministic,
        gat_concat=gat_concat,
        encoder_layer_norm=encoder_layer_norm,
        predictor_norm=predictor_norm,
        diagnostics=diagnostics,
        diagnostics_interval=diagnostics_interval,
        verbose=verbose,
        initial_state=initial_training_state,
        return_training_state=store_training_state,
        device=torch_device,
    )
    if store_training_state or diagnostics:
        embedding, training_state = train_result
    else:
        embedding = train_result
        training_state = None
    adata_mut.obsm["X_integrated"] = embedding
    if diagnostics:
        adata_mut.uns[diagnostics_key] = _make_h5ad_safe_diagnostics(
            training_state.get("diagnostics", [])
        )
    else:
        adata_mut.uns.pop(diagnostics_key, None)
    if store_training_state:
        adata_mut.uns[training_state_key] = _summarize_evofate_training_state(
            training_state
        )
    else:
        adata_mut.uns.pop(training_state_key, None)


class GATEncoder(nn.Module):
    """
    Graph attention encoder.

    The mutation graph defines all inter-cell edges. RNA/PCA features provide
    a graph-independent state representation while two shallow GAT layers learn
    attention and message refinements on mutation-graph edges.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Total output embedding dimension.
    heads : int
        Number of attention heads. `hidden_dim` must be divisible by `heads`.
    dropout : float, default=0.2
        Dropout rate for attention coefficients.
    concat : bool, default=False
        Deprecated compatibility argument. The encoder always concatenates
        parameter-matched heads to keep the total output dimension fixed.
    layer_norm : bool, default=True
        Deprecated compatibility flag. Use `branch_norm`.
    h0_strength : float, default=0.10
        Fixed coefficient for the graph-independent RNA projection.
    h1_strength : float, default=0.20
        Fixed coefficient for the first GAT layer residual contribution.
    h0_mode : {'trainable_linear', 'fixed'}, default='trainable_linear'
        Whether h0 is learned from RNA PCs through a linear projection or
        supplied as a fixed tensor during forward calls.
    branch_norm : {'layernorm', 'l2', 'none'}, default='layernorm'
        Normalization applied separately to h0, h1, and h2 before fusion.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float = 0.2,
        concat: bool = False,
        layer_norm: bool = True,
        h0_strength: float = DEFAULT_H0_STRENGTH,
        h1_strength: float = DEFAULT_H1_STRENGTH,
        h0_mode: str = DEFAULT_H0_MODE,
        branch_norm: str = DEFAULT_BRANCH_NORM,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("num_heads must be at least 1")
        if hidden_dim % heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        if float(h0_strength) < 0.0:
            raise ValueError("h0_strength must be non-negative")
        if float(h1_strength) < 0.0:
            raise ValueError("h1_strength must be non-negative")
        h0_mode = str(h0_mode).lower()
        if h0_mode not in {"trainable_linear", "fixed"}:
            raise ValueError("h0_mode must be 'trainable_linear' or 'fixed'.")
        branch_norm = str(branch_norm).lower()
        if branch_norm not in {"layernorm", "l2", "none"}:
            raise ValueError("branch_norm must be 'layernorm', 'l2', or 'none'.")

        head_dim = hidden_dim // heads
        self.rna_proj = nn.Linear(
            in_dim,
            hidden_dim,
            bias=False,
        )
        if h0_mode == "fixed" or float(h0_strength) == 0.0:
            for param in self.rna_proj.parameters():
                param.requires_grad = False

        self.gat1 = GATConv(
            in_dim,
            head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            fill_value=1.0,
        )
        self.gat2 = GATConv(
            hidden_dim,
            head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            fill_value=1.0,
        )
        self.out_dim = hidden_dim
        self.norm0 = nn.LayerNorm(self.out_dim)
        self.norm1 = nn.LayerNorm(self.out_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)
        self.final_norm = nn.LayerNorm(self.out_dim)
        self.num_layers = 2
        self.h0_strength = float(h0_strength)
        self.h1_strength = float(h1_strength)
        self.h0_mode = h0_mode
        self.branch_norm = "none" if not layer_norm else branch_norm

    def _normalize_branch(
        self,
        h: Tensor,
        norm_layer: nn.Module,
    ) -> Tensor:
        """Apply the configured branch normalization to one representation."""
        if self.branch_norm == "layernorm":
            return norm_layer(h)
        if self.branch_norm == "l2":
            return F.normalize(h, dim=1)
        return h

    def _resolve_h0(
        self,
        x: Tensor,
        fixed_h0: Tensor | None,
    ) -> Tensor:
        """Return graph-independent h0 from either RNA projection or fixed input."""
        if self.h0_mode == "fixed":
            if fixed_h0 is None:
                raise ValueError("fixed_h0 is required when h0_mode='fixed'.")
            expected_shape = (x.size(0), self.out_dim)
            if tuple(fixed_h0.shape) != expected_shape:
                raise ValueError(
                    "fixed_h0 has incorrect shape: expected "
                    f"{expected_shape}, got {tuple(fixed_h0.shape)}."
                )
            return fixed_h0.to(device=x.device, dtype=x.dtype)
        return self.rna_proj(x)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        fixed_h0: Tensor | None = None,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Forward pass through the GAT encoder."""
        h0 = self._resolve_h0(x, fixed_h0)
        h0 = self._normalize_branch(h0, self.norm0)

        h1 = self.gat1(x, edge_index)
        h1 = self._normalize_branch(h1, self.norm1)
        h1 = F.elu(h1)

        h2 = self.gat2(h1, edge_index)
        h2 = self._normalize_branch(h2, self.norm2)
        h2 = F.elu(h2)

        h_final = self.final_norm(
            h2
            + self.h1_strength * h1
            + self.h0_strength * h0
        )
        if return_layers:
            return h_final, {
                "h0": h0,
                "h1": h1,
                "h2": h2,
            }
        return h_final

    def attention_stats(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> dict[str, float]:
        """Return summary statistics for GAT attention coefficients."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            h1, (_, alpha1) = self.gat1(
                x,
                edge_index,
                return_attention_weights=True,
            )
            h1 = self._normalize_branch(h1, self.norm1)
            h1 = F.elu(h1)
            _, (_, alpha2) = self.gat2(
                h1,
                edge_index,
                return_attention_weights=True,
            )
            alpha1 = alpha1.detach().float().reshape(-1)
            alpha2 = alpha2.detach().float().reshape(-1)
            stats = {
                "attention_layer1_mean": float(alpha1.mean().item()),
                "attention_layer1_std": float(alpha1.std(unbiased=False).item()),
                "attention_layer1_min": float(alpha1.min().item()),
                "attention_layer1_max": float(alpha1.max().item()),
                "attention_layer2_mean": float(alpha2.mean().item()),
                "attention_layer2_std": float(alpha2.std(unbiased=False).item()),
                "attention_layer2_min": float(alpha2.min().item()),
                "attention_layer2_max": float(alpha2.max().item()),
            }
        self.train(was_training)
        return stats


class MLPPredictor(nn.Module):
    """
    MLP predictor for BGRL.

    A two-layer MLP with normalization and ReLU activation.

    Parameters
    ----------
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        norm = str(norm).lower()
        if norm == "batch":
            norm_layer: nn.Module = nn.BatchNorm1d(hidden_dim)
        elif norm == "layer":
            norm_layer = nn.LayerNorm(hidden_dim)
        elif norm in {"none", "identity"}:
            norm_layer = nn.Identity()
        else:
            raise ValueError("`predictor_norm` must be 'layer', 'batch', or 'none'.")
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            norm_layer,
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP predictor."""
        return self.layer(x)


def _prepare_rna_triplets(
    rna_edge_index: Tensor,
    rna_edge_weight: Tensor,
    genetic_edge_index: Tensor,
    num_nodes: int,
    positive_neighbors: int,
    positives_per_anchor: int,
    negative_candidates: int,
    seed: int | None,
) -> dict[str, Tensor]:
    """Prepare sparse RNA-positive and genetic-context negative samples."""
    rng = np.random.default_rng(seed)
    positive_neighbors = max(int(positive_neighbors), 1)
    positives_per_anchor = max(int(positives_per_anchor), 1)
    negative_candidates = max(int(negative_candidates), 1)
    rna_index = rna_edge_index.detach().cpu().numpy()
    rna_weight = rna_edge_weight.detach().cpu().numpy()
    genetic_index = genetic_edge_index.detach().cpu().numpy()
    positive_lists: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    genetic_lists: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target, weight in zip(rna_index[0], rna_index[1], rna_weight):
        source, target = int(source), int(target)
        if source != target and np.isfinite(weight) and weight > 0:
            positive_lists[source].append((target, float(weight)))
    for source, target in zip(genetic_index[0], genetic_index[1]):
        source, target = int(source), int(target)
        if source != target:
            genetic_lists[source].add(target)

    anchors: list[int] = []
    positives: list[int] = []
    candidates: list[list[int]] = []
    positive_weights: list[float] = []
    all_nodes = np.arange(num_nodes, dtype=np.int64)
    for anchor, values in enumerate(positive_lists):
        values.sort(key=lambda item: (-item[1], item[0]))
        values = values[:positive_neighbors]
        if not values:
            continue
        positive_ids = {target for target, _ in values}
        n_positive = min(positives_per_anchor, len(values))
        chosen = rng.choice(len(values), size=n_positive, replace=False)
        for chosen_index in np.atleast_1d(chosen):
            positive, positive_weight = values[int(chosen_index)]
            genetic_candidates = sorted(genetic_lists[anchor] - positive_ids)
            pool = np.asarray(genetic_candidates, dtype=np.int64)
            positive_array = np.asarray(sorted(positive_ids), dtype=np.int64)
            pool = pool[(pool != anchor) & ~np.isin(pool, positive_array)]
            if pool.size < negative_candidates:
                fallback = all_nodes[
                    (all_nodes != anchor) & ~np.isin(all_nodes, positive_array)
                ]
                if pool.size:
                    fallback = fallback[~np.isin(fallback, pool)]
                    pool = np.concatenate((pool, fallback))
                else:
                    pool = fallback
            if pool.size == 0:
                continue
            n_negative = min(negative_candidates, pool.size)
            negative = rng.choice(pool, size=n_negative, replace=False).tolist()
            negative.extend([-1] * (negative_candidates - n_negative))
            anchors.append(anchor)
            positives.append(positive)
            candidates.append([int(value) for value in negative])
            positive_weights.append(float(positive_weight))

    device = rna_edge_index.device
    empty_candidates = torch.empty(
        (0, negative_candidates), dtype=torch.long, device=device
    )
    return {
        "anchor": torch.as_tensor(anchors, dtype=torch.long, device=device),
        "positive": torch.as_tensor(positives, dtype=torch.long, device=device),
        "negative_candidates": (
            torch.as_tensor(candidates, dtype=torch.long, device=device)
            if candidates else empty_candidates
        ),
        "positive_weight": torch.as_tensor(
            positive_weights, dtype=torch.float32, device=device
        ),
    }


def _rna_relative_loss(
    z: Tensor,
    triplets: dict[str, Tensor] | None,
    margin: float,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Calculate weighted RNA-positive versus negative ranking loss."""
    if triplets is None or triplets["anchor"].numel() == 0:
        zero = z.sum() * 0.0
        empty_stats = {
            "mean_positive_distance": zero.detach(),
            "mean_negative_distance": zero.detach(),
            "active_triplet_fraction": zero.detach(),
            "weighted_rna_loss": zero.detach(),
        }
        return zero, empty_stats
    z = F.normalize(z, dim=1)
    anchor = triplets["anchor"]
    positive = triplets["positive"]
    negative_candidates = triplets["negative_candidates"]
    valid = negative_candidates >= 0
    safe_negative = negative_candidates.clamp_min(0)
    d_positive = 1.0 - (z[anchor] * z[positive]).sum(dim=1)
    d_negative = 1.0 - torch.einsum(
        "td,tkd->tk", z[anchor], z[safe_negative]
    )
    semi_hard = valid & (d_negative > d_positive[:, None]) & (
        d_negative < d_positive[:, None] + float(margin)
    )
    large = torch.full_like(d_negative, torch.inf)
    selected_semi = torch.where(semi_hard, d_negative, large).min(dim=1).values
    selected_any = torch.where(valid, d_negative, large).min(dim=1).values
    d_negative_selected = torch.where(
        torch.isfinite(selected_semi), selected_semi, selected_any
    )
    valid_triplet = torch.isfinite(d_negative_selected)
    if not torch.any(valid_triplet):
        zero = z.sum() * 0.0
        return zero, {
            "mean_positive_distance": zero.detach(),
            "mean_negative_distance": zero.detach(),
            "active_triplet_fraction": zero.detach(),
            "weighted_rna_loss": zero.detach(),
        }
    hinge = F.relu(
        d_positive[valid_triplet]
        - d_negative_selected[valid_triplet]
        + float(margin)
    )
    weights = triplets["positive_weight"][valid_triplet].to(dtype=z.dtype)
    weights = (weights / weights.mean().clamp_min(eps)).clamp(0.25, 2.0)
    loss = (weights * hinge).mean()
    return loss, {
        "mean_positive_distance": d_positive[valid_triplet].mean().detach(),
        "mean_negative_distance": d_negative_selected[valid_triplet].mean().detach(),
        "active_triplet_fraction": (hinge > 0).float().mean().detach(),
        "weighted_rna_loss": loss.detach(),
    }


class BGRL(nn.Module):
    """
    Bootstrapped Graph Representation Learning (BGRL) model.

    BGRL is a self-supervised learning method that learns node representations
    by predicting the representations of one augmented view from another.
    Uses an online-target encoder architecture with exponential moving average
    updates for the target encoder.

    Parameters
    ----------
    encoder : nn.Module
        Graph encoder network (e.g., GATEncoder).
    predictor : nn.Module
        Predictor network (e.g., MLPPredictor).
    momentum : float, default=0.99
        Momentum coefficient for target encoder EMA updates.

    References
    ----------
    Thakoor et al. "Bootstrapped Representation Learning on Graphs" (2021)
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.encoder_online = encoder
        self.encoder_target = copy.deepcopy(encoder)
        self.predictor = predictor
        self.momentum = momentum

        # Keep target encoder parameters fixed during the online update.
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        self.encoder_target.eval()

    def train(self, mode: bool = True):
        """Keep the EMA target encoder deterministic while training online parts."""
        super().train(mode)
        self.encoder_target.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Update target encoder using exponential moving average."""
        if momentum is not None:
            self.momentum = float(momentum)
        for param_online, param_target in zip(
            self.encoder_online.parameters(), self.encoder_target.parameters()
        ):
            param_target.mul_(self.momentum).add_(
                param_online,
                alpha=1.0 - self.momentum,
            )
        self._copy_target_buffers()

    @torch.no_grad()
    def _copy_target_buffers(self) -> None:
        """Keep target buffers aligned with online buffers."""
        online_buffers = dict(self.encoder_online.named_buffers())
        for name, target_buffer in self.encoder_target.named_buffers():
            online_buffer = online_buffers.get(name)
            if online_buffer is not None and target_buffer.shape == online_buffer.shape:
                target_buffer.copy_(online_buffer)

    def forward(
        self,
        x1: Tensor,
        edge_index1: Tensor,
        x2: Tensor,
        edge_index2: Tensor,
        fixed_h0: Tensor | None = None,
        rna_triplets: dict[str, Tensor] | None = None,
        rna_relative_weight: float = 0.0,
        rna_relative_margin: float = 0.1,
        return_details: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        Forward pass computing BGRL loss.

        Parameters
        ----------
        x1 : Tensor
            Node features for first augmented view.
        edge_index1 : Tensor
            Edge indices for first augmented view.
        x2 : Tensor
            Node features for the clean target view.
        edge_index2 : Tensor
            Edge indices for the clean target view.
        fixed_h0 : Tensor, optional
            Fixed graph-independent RNA representation used when the encoder
            is configured with `h0_mode='fixed'`.

        Returns
        -------
        loss : Tensor
            BGRL loss value.
        p1 : Tensor
            Predictions from first view (can be used as embeddings).
        """
        # Online encoder forward pass.
        z1_online = self.encoder_online(
            x1, edge_index1, fixed_h0=fixed_h0
        )
        z2_online = self.encoder_online(
            x2, edge_index2, fixed_h0=fixed_h0
        )

        # Predictor forward pass.
        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        # Target encoder forward pass without gradients.
        with torch.no_grad():
            z1_target = self.encoder_target(
                x1,
                edge_index1,
                fixed_h0=fixed_h0,
            ).detach()
            z2_target = self.encoder_target(
                x2,
                edge_index2,
                fixed_h0=fixed_h0,
            ).detach()

        # Compute the symmetric BGRL loss.
        node_loss_12 = self._bgrl_node_loss(p1, z2_target)
        node_loss_21 = self._bgrl_node_loss(p2, z1_target)
        bgrl_loss = node_loss_12.mean() + node_loss_21.mean()
        if float(rna_relative_weight) != 0.0:
            rna_loss_1, rna_stats_1 = _rna_relative_loss(
                z1_online, rna_triplets, rna_relative_margin
            )
            rna_loss_2, rna_stats_2 = _rna_relative_loss(
                z2_online, rna_triplets, rna_relative_margin
            )
            rna_loss = 0.5 * (rna_loss_1 + rna_loss_2)
        else:
            rna_loss = z1_online.sum() * 0.0
            rna_stats_1 = rna_stats_2 = {
                "mean_positive_distance": rna_loss.detach(),
                "mean_negative_distance": rna_loss.detach(),
                "active_triplet_fraction": rna_loss.detach(),
                "weighted_rna_loss": rna_loss.detach(),
            }
        loss = bgrl_loss + float(rna_relative_weight) * rna_loss
        if return_details:
            details = {
                "node_loss": 0.5 * (node_loss_12.detach() + node_loss_21.detach()),
                "bgrl_loss": bgrl_loss.detach(),
                "rna_loss": rna_loss.detach(),
                "rna_stats": {
                    key: 0.5 * (rna_stats_1[key] + rna_stats_2[key])
                    for key in rna_stats_1
                },
                "z1_online": z1_online,
                "z2_online": z2_online,
                "p1": p1.detach(),
                "p2": p2.detach(),
            }
            return loss, p1, details
        return loss, p1

    @staticmethod
    def _bgrl_loss(p_online: Tensor, z_target: Tensor) -> Tensor:
        """
        Compute BGRL loss between online predictions and target representations.

        Parameters
        ----------
        p_online : Tensor
            Online predictor output.
        z_target : Tensor
            Target encoder output.

        Returns
        -------
        Tensor
            Negative cosine similarity loss.
        """
        p_online = F.normalize(p_online, dim=1)
        z_target = F.normalize(z_target, dim=1)
        return 2 - 2 * (p_online * z_target).sum(dim=-1).mean()

    @staticmethod
    def _bgrl_node_loss(p_online: Tensor, z_target: Tensor) -> Tensor:
        """Return node-wise negative cosine similarity loss."""
        p_online = F.normalize(p_online, dim=1)
        z_target = F.normalize(z_target, dim=1)
        return 2 - 2 * (p_online * z_target).sum(dim=-1)


def augment_graph(
    x: Tensor,
    edge_index: Tensor,
    edge_drop_prob: float = 0.05,
    noise_std: float = 0.01,
    rna_edge_index: Tensor | None = None,
    rna_edge_weight: Tensor | None = None,
    rna_augmentation_alpha: float = 0.0,
    seed: int | None = None,
    feature_seed: int | None = None,
    edge_seed: int | None = None,
    rna_seed: int | None = None,
    drop_undirected_edges: bool = True,
    preserve_self_loops: bool = True,
    return_diagnostics: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, int]]:
    """
    Augmentation for PCA-based node features.

    Parameters
    ----------
    x : [N, D]
        PCA features.
    edge_index : [2, E]
        Graph edges.
    edge_drop_prob : float
        Probability of dropping each edge.
    noise_std : float
        Gaussian noise scale relative to PC std.
    rna_edge_index : Tensor, optional
        RNA-neighbor edges used to sample feature-space directions.
    rna_edge_weight : Tensor, optional
        RNA-connectivity weights used when sampling a neighbor for each cell.
    rna_augmentation_alpha : float, default=0.0
        Fraction of a sampled RNA-neighbor direction added to each cell:
        `x_i + alpha * (x_j - x_i)`.
    seed : int, optional
        Seed used to make this augmented view deterministic.
    feature_seed : int, optional
        Seed for feature-noise augmentation. Defaults to a value derived from
        `seed`.
    edge_seed : int, optional
        Seed for edge-drop augmentation. Defaults to a different value derived
        from `seed`.
    rna_seed : int, optional
        Seed for RNA-neighbor feature augmentation. Defaults to a value derived
        from `seed`.
    drop_undirected_edges : bool, default=True
        If True, drop both directions of an undirected edge together.
    preserve_self_loops : bool, default=True
        If True, retain any self-loop entries already present in `edge_index`.
    return_diagnostics : bool, default=False
        Whether to return isolated-node and degree-one counts after edge drop.

    Returns
    -------
    x_aug : [N, D]
    edge_index_aug : [2, E']
    """
    # Feature augmentation.
    feature_generator = None
    feature_random_device = x.device
    edge_generator = None
    edge_random_device = edge_index.device
    rna_generator = None
    rna_random_device = x.device
    if seed is not None:
        if feature_seed is None:
            feature_seed = int(seed) * 2 + 17
        if edge_seed is None:
            edge_seed = int(seed) * 2 + 29
        if rna_seed is None:
            rna_seed = int(seed) * 2 + 41
    if feature_seed is not None:
        feature_random_device = (
            x.device if x.device.type == "cuda" else torch.device("cpu")
        )
        feature_generator = torch.Generator(device=feature_random_device)
        feature_generator.manual_seed(int(feature_seed))
    if edge_seed is not None:
        edge_random_device = (
            edge_index.device
            if edge_index.device.type == "cuda"
            else torch.device("cpu")
        )
        edge_generator = torch.Generator(device=edge_random_device)
        edge_generator.manual_seed(int(edge_seed))
    if rna_seed is not None:
        rna_random_device = (
            x.device if x.device.type == "cuda" else torch.device("cpu")
        )
        rna_generator = torch.Generator(device=rna_random_device)
        rna_generator.manual_seed(int(rna_seed))

    # Add PC-scaled Gaussian noise.
    if float(noise_std) > 0.0:
        pc_std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=feature_random_device,
            generator=feature_generator,
        ).to(x.device)
        x_aug = x + float(noise_std) * pc_std * noise
    else:
        x_aug = x
    x_aug = _augment_features_along_rna_neighbors(
        x_aug=x_aug,
        reference_x=x,
        rna_edge_index=rna_edge_index,
        rna_edge_weight=rna_edge_weight,
        alpha=rna_augmentation_alpha,
        generator=rna_generator,
        random_device=rna_random_device,
    )

    # Edge augmentation.

    if float(edge_drop_prob) <= 0.0:
        keep_mask = torch.ones(
            edge_index.size(1),
            dtype=torch.bool,
            device=edge_index.device,
        )
    elif drop_undirected_edges:
        keep_mask = _undirected_edge_keep_mask(
            edge_index=edge_index,
            num_nodes=x.size(0),
            edge_drop_prob=edge_drop_prob,
            generator=edge_generator,
            random_device=edge_random_device,
            preserve_self_loops=preserve_self_loops,
        )
    else:
        keep_mask = torch.rand(
            edge_index.size(1),
            device=edge_random_device,
            generator=edge_generator,
        ).to(edge_index.device) > edge_drop_prob
        if preserve_self_loops:
            keep_mask = keep_mask | (edge_index[0] == edge_index[1])

    edge_index_aug = edge_index[:, keep_mask]

    if return_diagnostics:
        diagnostics = _edge_degree_diagnostics(
            edge_index_aug,
            num_nodes=x.size(0),
        )
        return x_aug, edge_index_aug, diagnostics
    return x_aug, edge_index_aug


def _augment_features_along_rna_neighbors(
    x_aug: Tensor,
    reference_x: Tensor,
    rna_edge_index: Tensor | None,
    rna_edge_weight: Tensor | None,
    alpha: float,
    generator: torch.Generator | None,
    random_device: torch.device,
) -> Tensor:
    """
    Move cells along directions observed in their RNA neighborhoods.

    For each source cell i with RNA neighbors, sample one neighbor j from
    `rna_edge_index` using RNA-connectivity weights, then apply:
    x_i <- x_i + alpha * (x_j - x_i).
    """
    alpha = float(alpha)
    if alpha <= 0.0 or rna_edge_index is None or rna_edge_index.numel() == 0:
        return x_aug
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("rna_augmentation_alpha must be in [0, 1].")

    rna_edge_index = rna_edge_index.to(reference_x.device)
    src = rna_edge_index[0].long()
    dst = rna_edge_index[1].long()
    if rna_edge_weight is None:
        edge_weight = torch.ones(
            src.size(0),
            dtype=reference_x.dtype,
            device=reference_x.device,
        )
    else:
        edge_weight = rna_edge_weight.to(
            device=reference_x.device,
            dtype=reference_x.dtype,
        )

    valid = (src != dst) & (edge_weight > 0.0)
    valid = valid & (src >= 0) & (dst >= 0)
    valid = valid & (src < reference_x.size(0)) & (dst < reference_x.size(0))
    if not torch.any(valid):
        return x_aug
    src = src[valid]
    dst = dst[valid]
    edge_weight = edge_weight[valid].clamp_min(1e-12)

    sampling_device = (
        torch.device("cpu") if reference_x.device.type == "mps" else reference_x.device
    )
    src_sample = src.to(sampling_device)
    dst_sample = dst.to(sampling_device)
    weight_sample = edge_weight.to(sampling_device)
    random_values = torch.rand(
        weight_sample.size(0),
        device=random_device,
        generator=generator,
    ).to(sampling_device)
    random_values = random_values.clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(random_values))
    scores = torch.log(weight_sample) + gumbel
    max_scores = scatter(
        scores,
        src_sample,
        dim=0,
        dim_size=reference_x.size(0),
        reduce="max",
    )
    chosen = scores == max_scores[src_sample]
    if not torch.any(chosen):
        return x_aug

    chosen_src = src_sample[chosen].to(reference_x.device)
    chosen_dst = dst_sample[chosen].to(reference_x.device)
    out = x_aug.clone()
    out[chosen_src] = (
        out[chosen_src]
        + alpha * (reference_x[chosen_dst] - reference_x[chosen_src])
    )
    return out


def _undirected_edge_keep_mask(
    edge_index: Tensor,
    num_nodes: int,
    edge_drop_prob: float,
    generator: torch.Generator | None,
    random_device: torch.device,
    preserve_self_loops: bool,
) -> Tensor:
    """Return an edge keep mask that treats reciprocal edges as one edge."""
    if edge_index.device.type == "mps":
        edge_index_cpu = edge_index.detach().cpu()
        src_cpu = edge_index_cpu[0]
        dst_cpu = edge_index_cpu[1]
        low_cpu = torch.minimum(src_cpu, dst_cpu)
        high_cpu = torch.maximum(src_cpu, dst_cpu)
        edge_key_cpu = low_cpu * int(num_nodes) + high_cpu
        unique_keys_cpu, inverse_cpu = torch.unique(
            edge_key_cpu,
            sorted=False,
            return_inverse=True,
        )
        keep_unique_cpu = (
            torch.rand(
                unique_keys_cpu.size(0),
                device=torch.device("cpu"),
                generator=generator,
            )
            > edge_drop_prob
        )
        if preserve_self_loops:
            self_loop_unique_cpu = torch.zeros_like(
                keep_unique_cpu,
                dtype=torch.bool,
            )
            self_loop_unique_cpu[inverse_cpu[src_cpu == dst_cpu]] = True
            keep_unique_cpu = keep_unique_cpu | self_loop_unique_cpu
        return keep_unique_cpu[inverse_cpu].to(edge_index.device)

    src = edge_index[0]
    dst = edge_index[1]
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    edge_key = low * int(num_nodes) + high
    unique_keys, inverse = torch.unique(edge_key, sorted=False, return_inverse=True)
    keep_unique = (
        torch.rand(
            unique_keys.size(0),
            device=random_device,
            generator=generator,
        ).to(edge_index.device)
        > edge_drop_prob
    )
    if preserve_self_loops:
        self_loop_unique = torch.zeros_like(keep_unique, dtype=torch.bool)
        self_loop_unique[inverse[src == dst]] = True
        keep_unique = keep_unique | self_loop_unique
    return keep_unique[inverse]


def _edge_degree_diagnostics(
    edge_index: Tensor,
    num_nodes: int,
) -> dict[str, int]:
    """Count isolated and degree-one nodes in an undirected view of edge_index."""
    edge_index = edge_index.detach().cpu()
    if edge_index.numel() == 0:
        return {
            "num_edges": 0,
            "num_isolated_nodes": int(num_nodes),
            "num_degree_one_nodes": 0,
        }
    src = edge_index[0]
    dst = edge_index[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    if src.numel() == 0:
        degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    else:
        low = torch.minimum(src, dst)
        high = torch.maximum(src, dst)
        edge_key = low * int(num_nodes) + high
        unique_edge_key = torch.unique(edge_key, sorted=False)
        unique_low = unique_edge_key.div(int(num_nodes), rounding_mode="floor")
        unique_high = unique_edge_key.remainder(int(num_nodes))
        degree = torch.bincount(unique_low, minlength=num_nodes)
        degree = degree + torch.bincount(unique_high, minlength=num_nodes)
    return {
        "num_edges": int(edge_index.size(1)),
        "num_isolated_nodes": int((degree == 0).sum().item()),
        "num_degree_one_nodes": int((degree == 1).sum().item()),
    }


def _resolve_rna_n_neighbors(
    n_cells: int,
    n_neighbors: int | None,
) -> int:
    """Resolve the Scanpy RNA-neighbor count for the current cell population."""
    if n_cells < 2:
        raise ValueError("At least two cells are required to build RNA neighbors.")
    if n_neighbors is None:
        n_neighbors = 15
    return min(max(int(n_neighbors), 1), int(n_cells) - 1)


def _prepare_scanpy_import_environment() -> None:
    """Ensure Scanpy's optional cache directories are writable."""
    tmp_dir = tempfile.gettempdir()
    cache_defaults = {
        "NUMBA_CACHE_DIR": os.path.join(tmp_dir, "evofate_numba_cache"),
        "MPLCONFIGDIR": os.path.join(tmp_dir, "evofate_mpl_cache"),
    }
    for env_key, path in cache_defaults.items():
        os.environ.setdefault(env_key, path)
        os.makedirs(os.environ[env_key], exist_ok=True)


def _cal_scanpy_rna_connectivity(
    adata_mut: AnnData,
    expression_key: str,
    connectivity_key: str,
    n_neighbors: int | None,
    metric: str,
):
    """Build and store RNA connectivity from expression PCA using Scanpy."""
    if expression_key not in adata_mut.obsm:
        raise KeyError(
            f"`adata_mut.obsm['{expression_key}']` is required to build RNA "
            "connectivity."
        )

    try:
        _prepare_scanpy_import_environment()
        import scanpy as sc
        from anndata import AnnData as _AnnData
    except ImportError as exc:
        raise ImportError(
            "Scanpy and AnnData are required to build RNA connectivity from "
            "expression PCA."
        ) from exc

    expression_pca = np.asarray(adata_mut.obsm[expression_key], dtype=np.float32)
    if expression_pca.ndim != 2:
        raise ValueError(f"`adata_mut.obsm['{expression_key}']` must be 2D.")
    if expression_pca.shape[0] != adata_mut.n_obs:
        raise ValueError(
            f"`adata_mut.obsm['{expression_key}']` has {expression_pca.shape[0]} "
            f"rows but `adata_mut` has {adata_mut.n_obs} cells."
        )

    resolved_neighbors = _resolve_rna_n_neighbors(
        adata_mut.n_obs,
        n_neighbors,
    )
    rna_adata = _AnnData(
        X=np.zeros((adata_mut.n_obs, 1), dtype=np.float32),
        obs=adata_mut.obs.iloc[:, :0].copy(),
    )
    rna_adata.obsm["X_pca"] = expression_pca
    sc.pp.neighbors(
        rna_adata,
        n_neighbors=resolved_neighbors,
        use_rep="X_pca",
        metric=str(metric),
    )
    connectivities = rna_adata.obsp["connectivities"].tocsr().astype(np.float32)
    connectivities.setdiag(0.0)
    connectivities.eliminate_zeros()
    if connectivities.nnz == 0:
        raise ValueError("Scanpy RNA connectivity graph has no positive edges.")

    adata_mut.obsp[str(connectivity_key)] = connectivities
    adata_mut.uns["evofate_rna_neighbors"] = {
        "connectivities_key": str(connectivity_key),
        "expression_key": str(expression_key),
        "n_neighbors": int(resolved_neighbors),
        "metric": str(metric),
        "method": "scanpy.pp.neighbors",
    }
    return connectivities


def _connectivity_to_edge_tensors(connectivity) -> tuple[Tensor, Tensor]:
    """Convert a sparse positive connectivity matrix to PyG edge tensors."""
    coo = connectivity.tocoo()
    row = np.asarray(coo.row, dtype=np.int64)
    col = np.asarray(coo.col, dtype=np.int64)
    data = np.asarray(coo.data, dtype=np.float32)
    mask = (data > 0.0) & (row != col)
    row = row[mask]
    col = col[mask]
    data = data[mask]
    if data.size == 0:
        raise ValueError("Connectivity graph has no positive non-self edges.")
    edge_index = torch.LongTensor(np.vstack((row, col)))
    edge_weight = torch.FloatTensor(data)
    return edge_index, edge_weight


def _prepare_fixed_h0_tensor(
    fixed_h0: Tensor | np.ndarray | None,
    h0_mode: str,
    num_nodes: int,
    embedding_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Validate and move fixed h0 to the active training device when requested."""
    if str(h0_mode).lower() != "fixed":
        return None
    if fixed_h0 is None:
        raise ValueError("fixed_h0 is required when h0_mode='fixed'.")
    if isinstance(fixed_h0, Tensor):
        fixed_h0_tensor = fixed_h0.detach()
    else:
        fixed_h0_tensor = torch.as_tensor(fixed_h0)
    expected_shape = (int(num_nodes), int(embedding_dim))
    if tuple(fixed_h0_tensor.shape) != expected_shape:
        raise ValueError(
            "fixed_h0 has incorrect shape: expected "
            f"{expected_shape}, got {tuple(fixed_h0_tensor.shape)}."
        )
    return fixed_h0_tensor.to(device=device, dtype=dtype)


def _clone_torch_state_to_cpu(value):
    """Clone a PyTorch state object to detached CPU tensors."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_torch_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_torch_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_torch_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _is_resumable_evofate_training_state(value: object) -> bool:
    """Return whether a value contains raw PyTorch state needed to resume."""
    return (
        isinstance(value, dict)
        and "model_state_dict" in value
        and "optimizer_state_dict" in value
        and "config" in value
    )


def _summarize_evofate_training_state(training_state: dict) -> dict[str, object]:
    """Return an h5ad-safe training summary without PyTorch objects."""
    if not isinstance(training_state, dict):
        raise TypeError("`training_state` must be a dictionary.")

    last_loss = training_state.get("last_loss")
    return {
        "format": "evofate_training_summary",
        "h5ad_safe": True,
        "resumable": False,
        "epochs_trained": int(training_state.get("epochs_trained", 0)),
        "last_loss": float("nan") if last_loss is None else float(last_loss),
        "device": str(training_state.get("device", "unknown")),
        "config": _make_h5ad_safe_metadata(training_state.get("config", {})),
        "note": (
            "Raw PyTorch model and optimizer state are intentionally not stored "
            "in AnnData."
        ),
    }


def _make_h5ad_safe_metadata(value: object) -> object:
    """Convert small metadata containers to h5ad-safe Python/NumPy values."""
    if isinstance(value, dict):
        return {
            str(key): _make_h5ad_safe_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_make_h5ad_safe_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Tensor):
        return f"torch.Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _make_h5ad_safe_diagnostics(records: list[dict[str, object]]) -> dict[str, object]:
    """Serialize nested training diagnostics into an h5ad-safe payload."""
    safe_records = [_make_h5ad_safe_metadata(record) for record in records]
    return {
        "format": "evofate_training_diagnostics",
        "encoding": "json_records",
        "n_records": int(len(safe_records)),
        "records_json": np.asarray(
            [
                json.dumps(record, sort_keys=True, allow_nan=True)
                for record in safe_records
            ],
            dtype=object,
        ),
    }


def _resolve_torch_device(device: str | torch.device | None) -> torch.device:
    """Resolve a user-facing CPU/GPU device option to a torch.device."""
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = str(device).strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            raise RuntimeError(
                "`device='gpu'` was requested, but no CUDA or Apple MPS device "
                "is available. Use `device='cpu'` to train on CPU."
            )
        resolved = torch.device(requested)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"`device='{resolved}'` was requested, but CUDA is not available."
        )
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            f"`device='{resolved}'` was requested, but MPS is not available."
        )
    return resolved


def _print_torch_device_info(
    requested_device: str | torch.device | None,
    resolved_device: torch.device,
) -> None:
    """Print a concise report of the available and selected torch device."""
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    parts = [
        "[EvoFATE] Device information:",
        f"requested={requested_device if requested_device is not None else 'auto'}",
        f"selected={resolved_device}",
        f"torch={torch.__version__}",
        f"cuda_available={cuda_available}",
        f"cuda_devices={cuda_count}",
        f"mps_available={mps_available}",
    ]
    if resolved_device.type == "cuda":
        cuda_index = resolved_device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        parts.append(f"cuda_device_name={torch.cuda.get_device_name(cuda_index)}")
    print(" | ".join(parts))


def _move_optimizer_state_to_device(
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    """Move any tensor state inside an optimizer to the active training device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _validate_evofate_training_state(
    training_state: dict,
    expected_config: dict,
) -> None:
    """Validate that a stored EvoFATE state can be loaded into this model."""
    if not isinstance(training_state, dict):
        raise TypeError("`initial_state` must be a dictionary.")
    for key in ("model_state_dict", "optimizer_state_dict", "config"):
        if key not in training_state:
            raise KeyError(f"`initial_state` is missing required key `{key}`.")

    stored_config = training_state["config"]
    for key in (
        "in_dim",
        "embedding_dim",
        "heads",
        "encoder_layers",
        "h0_strength",
        "h1_strength",
        "h0_mode",
        "branch_norm",
    ):
        if stored_config.get(key) != expected_config[key]:
            raise ValueError(
                "Stored EvoFATE training state is incompatible with the "
                f"requested model: `{key}` is {stored_config.get(key)!r} in "
                f"the stored state but {expected_config[key]!r} was requested."
            )


def _set_evofate_random_seed(
    seed: int | None,
    deterministic: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for EvoFATE training."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _make_evofate_optimizer(
    model: BGRL,
    lr: float,
    encoder_lr: float | None,
    predictor_lr: float | None,
    weight_decay: float,
    optimizer_name: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
) -> optim.Optimizer:
    """Create optimizer with separate online-encoder and predictor groups."""
    encoder_lr = float(min(float(lr), 1e-4) if encoder_lr is None else encoder_lr)
    predictor_lr = float(lr if predictor_lr is None else predictor_lr)
    param_groups = [
        {
            "params": [
                param
                for param in model.encoder_online.parameters()
                if param.requires_grad
            ],
            "lr": encoder_lr,
            "weight_decay": float(weight_decay),
            "name": "encoder",
        },
        {
            "params": [
                param
                for param in model.predictor.parameters()
                if param.requires_grad
            ],
            "lr": predictor_lr,
            "weight_decay": float(weight_decay),
            "name": "predictor",
        },
    ]
    _validate_optimizer_parameter_groups(model, param_groups)
    optimizer_name = str(optimizer_name).lower()
    optimizer_cls = {"adam": optim.Adam, "adamw": optim.AdamW}.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError("`optimizer_name` must be either 'adam' or 'adamw'.")
    return optimizer_cls(
        param_groups,
        betas=(float(adam_beta1), float(adam_beta2)),
        eps=float(adam_eps),
    )


def _validate_optimizer_parameter_groups(
    model: BGRL,
    param_groups: list[dict],
) -> None:
    """Check for duplicated or missing trainable optimizer parameters."""
    grouped_ids: list[int] = []
    for group in param_groups:
        grouped_ids.extend(id(param) for param in group["params"])
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("Optimizer parameter groups contain duplicated parameters.")

    expected_ids = {
        id(param)
        for param in model.parameters()
        if param.requires_grad
    }
    grouped_id_set = set(grouped_ids)
    missing = expected_ids - grouped_id_set
    extra = grouped_id_set - expected_ids
    if missing:
        raise RuntimeError("Optimizer is missing trainable EvoFATE parameters.")
    if extra:
        raise RuntimeError("Optimizer contains non-trainable EvoFATE parameters.")


def _make_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str | None,
    total_steps: int,
    warmup_steps: int,
):
    """Create a per-step learning-rate scheduler."""
    if scheduler_name is None or str(scheduler_name).lower() in {"none", "constant"}:
        return None
    scheduler_name = str(scheduler_name).lower()
    if scheduler_name != "warmup_cosine":
        raise ValueError("`lr_scheduler` must be 'warmup_cosine', 'constant', or None.")
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps + 1) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _scheduled_ema_momentum(
    momentum_start: float,
    momentum_final: float | None,
    step: int,
    total_steps: int,
) -> float:
    """Cosine-increase EMA momentum over training when requested."""
    if momentum_final is None:
        return float(momentum_start)
    total_steps = max(int(total_steps), 1)
    progress = min(max(float(step) / float(total_steps - 1 or 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(momentum_start + (momentum_final - momentum_start) * cosine)


def _current_lrs(optimizer: optim.Optimizer) -> dict[str, float]:
    """Return current learning rates by optimizer group name."""
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable model parameters."""
    return [param for param in model.parameters() if param.requires_grad]


def _count_trainable_parameters(module: nn.Module | None) -> int:
    """Return the number of trainable parameters in one module."""
    if module is None:
        return 0
    return int(
        sum(param.numel() for param in module.parameters() if param.requires_grad)
    )


def _parameter_count_report(model: BGRL) -> dict[str, int]:
    """Return exact trainable parameter counts for major online modules."""
    encoder = model.encoder_online
    rna_proj = getattr(encoder, "rna_proj", None)
    gat1 = getattr(encoder, "gat1", None)
    gat2 = getattr(encoder, "gat2", None)
    report = {
        "rna_proj": _count_trainable_parameters(rna_proj),
        "gat1": _count_trainable_parameters(gat1),
        "gat2": _count_trainable_parameters(gat2),
        "projector": 0,
        "predictor": _count_trainable_parameters(model.predictor),
        "total_online_encoder": _count_trainable_parameters(encoder),
    }
    report["total_online_model"] = int(
        report["total_online_encoder"]
        + report["projector"]
        + report["predictor"]
    )
    return report


def _module_gradient_norm(module: nn.Module) -> float:
    """Return the L2 norm of gradients in one module."""
    total_sq = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total_sq))


def _gradient_norm_report(model: BGRL) -> dict[str, float]:
    """Return total and per-module gradient norms."""
    encoder_norm = _module_gradient_norm(model.encoder_online)
    rna_proj_norm = (
        _module_gradient_norm(model.encoder_online.rna_proj)
        if hasattr(model.encoder_online, "rna_proj")
        else 0.0
    )
    gat1_norm = (
        _module_gradient_norm(model.encoder_online.gat1)
        if hasattr(model.encoder_online, "gat1")
        else 0.0
    )
    gat2_norm = (
        _module_gradient_norm(model.encoder_online.gat2)
        if hasattr(model.encoder_online, "gat2")
        else 0.0
    )
    predictor_norm = _module_gradient_norm(model.predictor)
    return {
        "total": float(math.sqrt(encoder_norm**2 + predictor_norm**2)),
        "encoder": encoder_norm,
        "rna_proj": rna_proj_norm,
        "gat1": gat1_norm,
        "gat2": gat2_norm,
        "projector": 0.0,
        "predictor": predictor_norm,
    }


def _summarize_values(values: list[float] | np.ndarray) -> dict[str, float]:
    """Return mean, std, min, and max for a sequence."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summarize_values_with_quantiles(
    values: Tensor | np.ndarray | list[float],
) -> dict[str, float]:
    """Return summary statistics including robust quantiles."""
    if isinstance(values, Tensor):
        array = values.detach().float().cpu().numpy()
    else:
        array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _embedding_diagnostics(
    embedding: Tensor,
    top_k: int = 10,
) -> dict[str, object]:
    """Summarize embedding norms, outliers, and singular spectrum."""
    embedding_cpu = embedding.detach().float().cpu()
    norms = torch.linalg.norm(embedding_cpu, dim=1).numpy()
    norm_stats = _summarize_values(norms)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    if mad > 1e-12:
        threshold = median + 6.0 * 1.4826 * mad
    else:
        q1, q3 = np.quantile(norms, [0.25, 0.75])
        threshold = float(q3 + 3.0 * (q3 - q1))
    outlier_mask = norms > threshold

    centered = embedding_cpu - embedding_cpu.mean(dim=0, keepdim=True)
    try:
        singular_values = torch.linalg.svdvals(centered).numpy()
    except RuntimeError:
        singular_values = np.asarray([], dtype=float)
    if singular_values.size > 0 and float(np.sum(singular_values)) > 0.0:
        probabilities = singular_values / np.sum(singular_values)
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        )
    else:
        effective_rank = 0.0

    top_k = min(int(top_k), norms.shape[0])
    top_norm_indices = np.argsort(-norms, kind="stable")[:top_k]
    return {
        "embedding_norm": norm_stats,
        "embedding_norm_outlier_threshold": float(threshold),
        "num_embedding_norm_outliers": int(np.count_nonzero(outlier_mask)),
        "effective_rank": effective_rank,
        "singular_values": [float(value) for value in singular_values[:20]],
        "top_embedding_norm_nodes": [
            {"node": int(index), "value": float(norms[index])}
            for index in top_norm_indices
        ],
    }


def _branch_diagnostics(
    layers: dict[str, Tensor],
    h0_strength: float,
    h1_strength: float,
) -> dict[str, object]:
    """Summarize h0/h1/h2 branch scale, contribution, and alignment."""
    h0 = layers["h0"].detach()
    h1 = layers["h1"].detach()
    h2 = layers["h2"].detach()
    h0_norm = torch.linalg.norm(h0, dim=1)
    h1_norm = torch.linalg.norm(h1, dim=1)
    h2_norm = torch.linalg.norm(h2, dim=1)
    h0_contribution = float(h0_strength) * h0_norm
    h1_contribution = float(h1_strength) * h1_norm
    h2_contribution = h2_norm
    contribution_total = (
        h0_contribution + h1_contribution + h2_contribution
    ).clamp_min(1e-12)
    cos_h0_h1 = F.cosine_similarity(h0, h1, dim=1)
    cos_h0_h2 = F.cosine_similarity(h0, h2, dim=1)
    cos_h1_h2 = F.cosine_similarity(h1, h2, dim=1)
    return {
        "branch_norm_h0": _summarize_values_with_quantiles(h0_norm),
        "branch_norm_h1": _summarize_values_with_quantiles(h1_norm),
        "branch_norm_h2": _summarize_values_with_quantiles(h2_norm),
        "branch_contribution_h0": _summarize_values_with_quantiles(
            h0_contribution
        ),
        "branch_contribution_h1": _summarize_values_with_quantiles(
            h1_contribution
        ),
        "branch_contribution_h2": _summarize_values_with_quantiles(
            h2_contribution
        ),
        "branch_fraction_h0": _summarize_values_with_quantiles(
            h0_contribution / contribution_total
        ),
        "branch_fraction_h1": _summarize_values_with_quantiles(
            h1_contribution / contribution_total
        ),
        "branch_fraction_h2": _summarize_values_with_quantiles(
            h2_contribution / contribution_total
        ),
        "branch_cosine_h0_h1": _summarize_values_with_quantiles(cos_h0_h1),
        "branch_cosine_h0_h2": _summarize_values_with_quantiles(cos_h0_h2),
        "branch_cosine_h1_h2": _summarize_values_with_quantiles(cos_h1_h2),
    }


def _node_loss_diagnostics(
    node_loss_sum: Tensor,
    count: int,
    top_k: int = 10,
) -> dict[str, object]:
    """Summarize node-wise BGRL loss distribution."""
    if count <= 0:
        return {}
    losses = (node_loss_sum.detach().float().cpu() / float(count)).numpy()
    top_k = min(int(top_k), losses.shape[0])
    top_loss_indices = np.argsort(-losses, kind="stable")[:top_k]
    return {
        "node_loss": _summarize_values(losses),
        "top_node_loss_nodes": [
            {"node": int(index), "value": float(losses[index])}
            for index in top_loss_indices
        ],
    }


def train_evofate(
    data: Data,
    embedding_dim: int,
    epochs: int,
    lr: float,
    heads: int = 2,
    encoder_lr: float | None = None,
    predictor_lr: float | None = None,
    dropout: float = 0.2,
    momentum: float = 0.99,
    momentum_final: float | None = 0.999,
    edge_drop_prob: float = 0.05,
    noise_std: float = 0.01,
    rna_relative_weight: float = 0.05,
    rna_relative_margin: float = 0.1,
    rna_loss_warmup_epochs: int = 10,
    rna_loss_ramp_epochs: int = 20,
    rna_edge_index: Tensor | None = None,
    rna_edge_weight: Tensor | None = None,
    rna_augmentation_alpha: float = 0.0,
    h0_strength: float = DEFAULT_H0_STRENGTH,
    h1_strength: float = DEFAULT_H1_STRENGTH,
    h0_mode: str = DEFAULT_H0_MODE,
    branch_norm: str = DEFAULT_BRANCH_NORM,
    fixed_h0: Tensor | np.ndarray | None = None,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    max_grad_norm: float | None = 1.0,
    lr_scheduler: str | None = "warmup_cosine",
    warmup_epochs: int = 10,
    print_interval: int = 100,
    drop_undirected_edges: bool = True,
    preserve_self_loops: bool = True,
    seed: int | None = 0,
    deterministic: bool = False,
    gat_concat: bool = False,
    encoder_layer_norm: bool = True,
    predictor_norm: str = "layer",
    diagnostics: bool = False,
    diagnostics_interval: int | None = None,
    diagnostics_top_k: int = 10,
    verbose: bool = True,
    initial_state: dict | None = None,
    return_training_state: bool = False,
    device: str | torch.device | None = "auto",
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Train EvoFATE model using BGRL.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object with node features and edge index.
    embedding_dim : int
        Target embedding dimension.
    epochs : int
        Number of training epochs.
    lr : float
        Base learning rate, used for the predictor by default.
    encoder_lr : float, optional
        Learning rate for the GAT encoder. Defaults to `min(lr, 1e-4)`.
    predictor_lr : float, optional
        Learning rate for the predictor. Defaults to `lr`.
    heads : int
        Number of GAT attention heads.
    dropout : float, default=0.2
        Dropout rate for GAT attention coefficients.
    momentum : float, default=0.99
        Initial exponential moving average momentum for the target encoder.
    momentum_final : float, optional, default=0.999
        Final EMA momentum for cosine scheduling. Use None for constant
        momentum.
    edge_drop_prob : float, default=0.05
        Probability of dropping graph edges in the augmented online view.
    noise_std : float, default=0.01
        Standard deviation multiplier for PCA-scaled feature noise in the
        augmented online view.
    rna_edge_index : Tensor, optional
        RNA-neighbor edges used by feature augmentation. If omitted,
        `data.rna_edge_index` is used when present.
    rna_edge_weight : Tensor, optional
        RNA-connectivity weights for `rna_edge_index`.
    rna_augmentation_alpha : float, default=0.0
        Fraction of the sampled RNA-neighbor direction added to each online
        augmented feature, using `x_i + alpha * (x_j - x_i)`.
    h0_strength : float, default=0.10
        Fixed coefficient for the graph-independent RNA input skip.
    h1_strength : float, default=0.20
        Fixed coefficient for the first GAT layer skip.
    h0_mode : {'trainable_linear', 'fixed'}, default='trainable_linear'
        Whether h0 is computed by a trainable linear RNA projection or supplied
        through `fixed_h0`.
    branch_norm : {'layernorm', 'l2', 'none'}, default='layernorm'
        Normalization applied separately to h0, h1, and h2 before fusion.
    fixed_h0 : array-like or Tensor, optional
        Fixed graph-independent RNA representation with shape
        `(n_nodes, embedding_dim)`, required when `h0_mode='fixed'`.
    weight_decay : float, default=1e-4
        Weight decay for Adam/AdamW.
    print_interval : int, default=100
        Number of epochs between progress messages.
    drop_undirected_edges : bool, default=True
        Whether reciprocal graph edges are dropped together.
    diagnostics : bool, default=False
        Whether to return structured training diagnostics.
    verbose : bool, default=True
        Whether to print training progress.
    initial_state : dict, optional
        Previously stored EvoFATE training state. If provided, model and
        optimizer states are restored before training.
    return_training_state : bool, default=False
        Whether to return the final model/optimizer state along with the
        embedding. This state can be passed back as `initial_state`.
    device : str, torch.device, or None, default='auto'
        Device used for model training. The default uses CUDA when available,
        then Apple MPS on Mac, and falls back to CPU. Use 'gpu' to require a
        GPU, 'mps' to force Mac GPU, or 'cpu' to force CPU.

    Returns
    -------
    embedding : np.ndarray
        Learned node embeddings of shape `(n_nodes, embedding_dim)`.
    """
    in_dim = data.x.shape[1]
    torch_device = _resolve_torch_device(device)
    _set_evofate_random_seed(seed, deterministic=deterministic)
    data = data.to(torch_device)
    print_interval = max(print_interval, 1)
    if not 0.0 <= float(edge_drop_prob) < 1.0:
        raise ValueError("edge_drop_prob must be in [0, 1).")
    if max_grad_norm is not None and float(max_grad_norm) <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")
    if float(noise_std) < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if float(rna_relative_weight) < 0.0 or float(rna_relative_margin) < 0.0:
        raise ValueError("RNA relative weight and margin must be nonnegative.")
    if int(rna_loss_warmup_epochs) < 0 or int(rna_loss_ramp_epochs) < 0:
        raise ValueError("RNA loss warmup and ramp epochs must be nonnegative.")
    if not 0.0 <= float(rna_augmentation_alpha) <= 1.0:
        raise ValueError("rna_augmentation_alpha must be in [0, 1].")
    if float(h0_strength) < 0.0:
        raise ValueError("h0_strength must be non-negative.")
    if float(h1_strength) < 0.0:
        raise ValueError("h1_strength must be non-negative.")
    h0_mode = str(h0_mode).lower()
    if h0_mode not in {"trainable_linear", "fixed"}:
        raise ValueError("h0_mode must be 'trainable_linear' or 'fixed'.")
    branch_norm = str(branch_norm).lower()
    if branch_norm not in {"layernorm", "l2", "none"}:
        raise ValueError("branch_norm must be 'layernorm', 'l2', or 'none'.")
    if not 0.0 < float(momentum) < 1.0:
        raise ValueError("momentum must be in (0, 1).")
    if (
        momentum_final is not None
        and not float(momentum) <= float(momentum_final) <= 1.0
    ):
        raise ValueError("momentum_final must be in [momentum, 1].")
    diagnostics_interval = (
        max(int(diagnostics_interval), 1)
        if diagnostics_interval is not None
        else max(print_interval, 1)
    )

    config = {
        "in_dim": int(in_dim),
        "embedding_dim": int(embedding_dim),
        "heads": int(heads),
        "encoder_layers": 2,
        "h0_strength": float(h0_strength),
        "h1_strength": float(h1_strength),
        "h0_mode": str(h0_mode),
        "branch_norm": str(branch_norm),
        "gat_concat": bool(gat_concat),
        "encoder_layer_norm": bool(encoder_layer_norm),
        "predictor_norm": str(predictor_norm),
        "rna_relative_margin": float(rna_relative_margin),
    }

    # Initialize model components.
    encoder = GATEncoder(
        in_dim=in_dim,
        hidden_dim=embedding_dim,
        heads=heads,
        dropout=dropout,
        concat=gat_concat,
        layer_norm=encoder_layer_norm,
        h0_strength=h0_strength,
        h1_strength=h1_strength,
        h0_mode=h0_mode,
        branch_norm=branch_norm,
    )
    predictor = MLPPredictor(
        hidden_dim=encoder.out_dim,
        out_dim=encoder.out_dim,
        norm=predictor_norm,
    )
    model = BGRL(encoder, predictor, momentum=momentum).to(torch_device)
    parameter_counts = _parameter_count_report(model)
    optimizer = _make_evofate_optimizer(
        model=model,
        lr=lr,
        encoder_lr=encoder_lr,
        predictor_lr=predictor_lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
    )
    total_steps = max(int(epochs), 1)
    scheduler = _make_lr_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        total_steps=total_steps,
        warmup_steps=int(warmup_epochs),
    )
    start_epoch = 0

    if initial_state is not None:
        _validate_evofate_training_state(initial_state, config)
        model.load_state_dict(initial_state["model_state_dict"])
        optimizer.load_state_dict(initial_state["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, torch_device)
        for param_group in optimizer.param_groups:
            if param_group.get("name") == "encoder":
                param_group["lr"] = float(
                    min(float(lr), 1e-4) if encoder_lr is None else encoder_lr
                )
            elif param_group.get("name") == "predictor":
                param_group["lr"] = float(lr if predictor_lr is None else predictor_lr)
            else:
                param_group["lr"] = float(lr)
            param_group["weight_decay"] = weight_decay
        start_epoch = int(initial_state.get("epochs_trained", 0))

    if rna_edge_index is None:
        rna_edge_index = getattr(data, "rna_edge_index", None)
    if rna_edge_weight is None:
        rna_edge_weight = getattr(data, "rna_edge_weight", None)
    if float(rna_augmentation_alpha) > 0.0 and rna_edge_index is None:
        raise ValueError(
            "rna_augmentation_alpha is positive, but no RNA connectivity graph "
            "was provided."
        )
    if rna_edge_index is not None:
        rna_edge_index = rna_edge_index.to(torch_device)
        if rna_edge_weight is not None:
            rna_edge_weight = rna_edge_weight.to(torch_device)
    if fixed_h0 is None:
        fixed_h0 = getattr(data, "fixed_h0", None)
    fixed_h0_tensor = _prepare_fixed_h0_tensor(
        fixed_h0=fixed_h0,
        h0_mode=h0_mode,
        num_nodes=data.x.size(0),
        embedding_dim=embedding_dim,
        device=torch_device,
        dtype=data.x.dtype,
    )
    config["fixed_h0_shape"] = (
        None
        if fixed_h0_tensor is None
        else tuple(int(v) for v in fixed_h0_tensor.shape)
    )

    # Train the online encoder and predictor.
    model.train()
    rna_triplets = getattr(data, "rna_triplets", None)
    if rna_triplets is not None:
        rna_triplets = {
            key: value.to(torch_device) for key, value in rna_triplets.items()
        }
    trainable_params = _trainable_parameters(model)
    total_epochs = start_epoch + epochs
    last_loss = None
    diagnostics_records: list[dict[str, object]] = []
    for epoch in range(start_epoch + 1, total_epochs + 1):
        is_first_epoch = epoch == start_epoch + 1
        should_print_epoch = verbose and (
            epoch % print_interval == 0 or is_first_epoch
        )
        should_record_diagnostics = diagnostics and (
            is_first_epoch
            or epoch == total_epochs
            or epoch % diagnostics_interval == 0
        )
        collect_grad_stats = should_print_epoch or should_record_diagnostics
        epoch_grad_norms: list[float] = []
        epoch_encoder_grad_norms: list[float] = []
        epoch_rna_proj_grad_norms: list[float] = []
        epoch_gat1_grad_norms: list[float] = []
        epoch_gat2_grad_norms: list[float] = []
        epoch_projector_grad_norms: list[float] = []
        epoch_predictor_grad_norms: list[float] = []
        epoch_isolated_counts: list[int] = []
        epoch_degree_one_counts: list[int] = []
        node_loss_sum = (
            torch.zeros(data.x.size(0), device=torch_device)
            if should_record_diagnostics
            else None
        )
        node_loss_count = 0
        step_index = epoch - start_epoch - 1
        current_momentum = _scheduled_ema_momentum(
            momentum_start=momentum,
            momentum_final=momentum_final,
            step=step_index,
            total_steps=total_steps,
        )
        if epoch < int(rna_loss_warmup_epochs):
            current_rna_weight = 0.0
        else:
            progress = (
                epoch - int(rna_loss_warmup_epochs) + 1
            ) / max(int(rna_loss_ramp_epochs), 1)
            current_rna_weight = float(rna_relative_weight) * min(
                max(progress, 0.0), 1.0
            )
        if float(rna_relative_weight) == 0.0:
            current_rna_weight = 0.0
        augmentation_seed = (
            None if seed is None else int(seed) + 1009 * int(epoch)
        )
        augmented_view = augment_graph(
            data.x,
            data.edge_index,
            edge_drop_prob=edge_drop_prob,
            noise_std=noise_std,
            rna_edge_index=rna_edge_index,
            rna_edge_weight=rna_edge_weight,
            rna_augmentation_alpha=rna_augmentation_alpha,
            seed=augmentation_seed,
            drop_undirected_edges=drop_undirected_edges,
            preserve_self_loops=preserve_self_loops,
            return_diagnostics=diagnostics,
        )
        if diagnostics:
            x1, edge_index1, aug_diag = augmented_view
            epoch_isolated_counts.append(aug_diag["num_isolated_nodes"])
            epoch_degree_one_counts.append(aug_diag["num_degree_one_nodes"])
        else:
            x1, edge_index1 = augmented_view

        # Keep the target view unmodified.
        x2 = data.x
        edge_index2 = data.edge_index

        # Compute the forward pass and loss.
        if should_record_diagnostics:
            loss, _, loss_details = model(
                x1,
                edge_index1,
                x2,
                edge_index2,
                fixed_h0=fixed_h0_tensor,
                rna_triplets=rna_triplets,
                rna_relative_weight=current_rna_weight,
                rna_relative_margin=rna_relative_margin,
                return_details=True,
            )
        else:
            loss, _ = model(
                x1,
                edge_index1,
                x2,
                edge_index2,
                fixed_h0=fixed_h0_tensor,
                rna_triplets=rna_triplets,
                rna_relative_weight=current_rna_weight,
                rna_relative_margin=rna_relative_margin,
            )
            loss_details = None

        # Backpropagate the online loss.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if collect_grad_stats:
            grad_report = _gradient_norm_report(model)
            epoch_grad_norms.append(grad_report["total"])
            epoch_encoder_grad_norms.append(grad_report["encoder"])
            epoch_rna_proj_grad_norms.append(grad_report["rna_proj"])
            epoch_gat1_grad_norms.append(grad_report["gat1"])
            epoch_gat2_grad_norms.append(grad_report["gat2"])
            epoch_projector_grad_norms.append(grad_report["projector"])
            epoch_predictor_grad_norms.append(grad_report["predictor"])
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(max_grad_norm),
            )
        optimizer.step()

        # Update the target encoder by exponential moving average.
        model.update_target_encoder(momentum=current_momentum)
        if scheduler is not None:
            scheduler.step()

        if should_record_diagnostics:
            node_loss_sum += loss_details["node_loss"]
            node_loss_count += 1

        mean_loss = float(loss.detach().item())
        last_loss = mean_loss

        if should_print_epoch:
            grad_mean = _summarize_values(epoch_grad_norms)["mean"]
            print(
                f"Epoch {epoch}/{total_epochs} - Loss: {mean_loss:.4f} "
                f"- GradNorm: {grad_mean:.4f}"
            )

        if should_record_diagnostics:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                epoch_embedding, epoch_layers = model.encoder_online(
                    data.x,
                    data.edge_index,
                    fixed_h0=fixed_h0_tensor,
                    return_layers=True,
                )
            if was_training:
                model.train()
            record = {
                "epoch": int(epoch),
                "loss": float(mean_loss),
                "bgrl_loss": float(loss_details["bgrl_loss"].item()),
                "rna_relative_loss": float(loss_details["rna_loss"].item()),
                "current_rna_weight": float(current_rna_weight),
                "rna_loss_fraction": float(
                    loss_details["rna_loss"].abs().item()
                    / (abs(mean_loss) + 1e-8)
                ),
                "mean_positive_distance": float(
                    loss_details["rna_stats"]["mean_positive_distance"].item()
                ),
                "mean_negative_distance": float(
                    loss_details["rna_stats"]["mean_negative_distance"].item()
                ),
                "active_triplet_fraction": float(
                    loss_details["rna_stats"]["active_triplet_fraction"].item()
                ),
                "parameter_counts": parameter_counts,
                "learning_rate": _current_lrs(optimizer),
                "ema_momentum": float(current_momentum),
                "gradient_norm": _summarize_values(epoch_grad_norms),
                "encoder_gradient_norm": _summarize_values(epoch_encoder_grad_norms),
                "rna_proj_gradient_norm": _summarize_values(
                    epoch_rna_proj_grad_norms
                ),
                "gat1_gradient_norm": _summarize_values(epoch_gat1_grad_norms),
                "gat2_gradient_norm": _summarize_values(epoch_gat2_grad_norms),
                "projector_gradient_norm": _summarize_values(
                    epoch_projector_grad_norms
                ),
                "predictor_gradient_norm": _summarize_values(
                    epoch_predictor_grad_norms
                ),
                "isolated_nodes_after_augmentation": _summarize_values(
                    epoch_isolated_counts
                ),
                "degree_one_nodes_after_augmentation": _summarize_values(
                    epoch_degree_one_counts
                ),
                **_embedding_diagnostics(
                    epoch_embedding,
                    top_k=diagnostics_top_k,
                ),
                **_branch_diagnostics(
                    epoch_layers,
                    h0_strength=h0_strength,
                    h1_strength=h1_strength,
                ),
                **_node_loss_diagnostics(
                    node_loss_sum,
                    node_loss_count,
                    top_k=diagnostics_top_k,
                ),
            }
            try:
                record.update(
                    model.encoder_online.attention_stats(
                        data.x,
                        data.edge_index,
                    )
                )
            except Exception:
                pass
            diagnostics_records.append(record)

    # Extract final node embeddings.
    model.eval()
    with torch.no_grad():
        final_embedding = model.encoder_online(
            data.x,
            data.edge_index,
            fixed_h0=fixed_h0_tensor,
        ).cpu().numpy()

    if return_training_state:
        training_state = {
            "model_state_dict": _clone_torch_state_to_cpu(model.state_dict()),
            "optimizer_state_dict": _clone_torch_state_to_cpu(
                optimizer.state_dict()
            ),
            "epochs_trained": int(total_epochs),
            "last_loss": None if last_loss is None else float(last_loss),
            "device": str(torch_device),
            "config": {
                **config,
                "dropout": float(dropout),
                "momentum": float(momentum),
                "momentum_final": (
                    None if momentum_final is None else float(momentum_final)
                ),
                "lr": float(lr),
                "encoder_lr": None if encoder_lr is None else float(encoder_lr),
                "predictor_lr": None if predictor_lr is None else float(predictor_lr),
                "weight_decay": float(weight_decay),
                "optimizer_name": str(optimizer_name),
                "max_grad_norm": (
                    None if max_grad_norm is None else float(max_grad_norm)
                ),
                "lr_scheduler": None if lr_scheduler is None else str(lr_scheduler),
                "warmup_epochs": int(warmup_epochs),
                "parameter_counts": parameter_counts,
                "edge_drop_prob": float(edge_drop_prob),
                "noise_std": float(noise_std),
                "rna_augmentation_alpha": float(rna_augmentation_alpha),
                "target_view": "undisturbed",
                "seed": None if seed is None else int(seed),
            },
            "diagnostics": diagnostics_records,
        }
        return final_embedding, training_state

    if diagnostics:
        return final_embedding, {"diagnostics": diagnostics_records}

    return final_embedding
