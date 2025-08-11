import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
import pandas as pd
import numpy as np
from torch_geometric.data import Data

# --- GAT Encoder ---
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=1):
        super().__init__()
        self.gat = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.2)
        self.prelu = nn.PReLU(hidden_dim * heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.prelu(x)
        return x

# --- MLP Predictor ---
class MLPPredictor(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layer(x)

# --- BGRL Model ---
class BGRL(nn.Module):
    def __init__(self, encoder, predictor, momentum=0.99):
        super().__init__()
        self.encoder_online = encoder
        self.encoder_target = copy.deepcopy(encoder)
        self.predictor = predictor
        self.momentum = momentum

        # Freeze target encoder params
        for param in self.encoder_target.parameters():
            param.requires_grad = False

    def update_target_encoder(self):
        # EMA update of target encoder parameters
        for param_online, param_target in zip(
            self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_target.data = (
                self.momentum * param_target.data + (1 - self.momentum) * param_online.data
            )

    def forward(self, x1, edge_index1, x2, edge_index2):
        z1_online = self.encoder_online(x1, edge_index1)
        z2_online = self.encoder_online(x2, edge_index2)

        p1 = self.predictor(z1_online)
        p2 = self.predictor(z2_online)

        with torch.no_grad():
            z1_target = self.encoder_target(x1, edge_index1).detach()
            z2_target = self.encoder_target(x2, edge_index2).detach()

        loss = self.bgrl_loss(p1, z2_target) + self.bgrl_loss(p2, z1_target)
        return loss, p1

    @staticmethod
    def bgrl_loss(p_online, z_target):
        p_online = F.normalize(p_online, dim=1)
        z_target = F.normalize(z_target, dim=1)
        return 2 - 2 * (p_online * z_target).sum(dim=-1).mean()

def augment_graph(x, edge_index, edge_drop_prob=0.2, feat_mask_prob=0.1):
    # Random edge dropout
    edge_index_aug, _ = dropout_edge(edge_index, p=edge_drop_prob)
    # Random feature masking
    x_aug = x.clone()
    mask = torch.rand(x.size()) < feat_mask_prob
    x_aug[mask] = 0
    return x_aug, edge_index_aug

def prepare_data(expression_csv, genetic_embedding, kneighbors=15):
    """
    Combine expression data and genetic embedding to build graph data.

    Returns:
        x_tensor: torch.FloatTensor node features (expression + genetic embedding concatenated)
        edge_index: torch.LongTensor edge index for graph connectivity
    """
    # Load expression matrix
    expr_df = pd.read_csv(expression_csv, index_col=0).fillna(0)
    
    # Check cell order alignment between expression and genetic embedding
    # genetic_embedding is assumed (N_cells, D_genetic)
    if expr_df.shape[0] != genetic_embedding.shape[0]:
        raise ValueError("Number of cells in expression and genetic embedding mismatch.")

    # Concatenate expression and genetic embedding features
    x = np.concatenate([expr_df.values, genetic_embedding], axis=1)
    x_tensor = torch.FloatTensor(x)

    # Build kNN graph on concatenated features
    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(x, n_neighbors=kneighbors, include_self=False)
    knn_graph = knn_graph + knn_graph.T  # symmetrize

    knn_graph_coo = knn_graph.tocoo()
    edge_index = torch.LongTensor(np.vstack((knn_graph_coo.row, knn_graph_coo.col)))

    return x_tensor, edge_index

