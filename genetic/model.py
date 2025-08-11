import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

def cal_cell_connectivity(snv, kneighbors=15, thred=0.05):
    """
    Calculate connectivity matrix from SNV matrix,
    using mutation overlap normalized by non-mutation,
    followed by dimensionality reduction with truncated SVD,
    and kneighbors graph on selected PCs.
    """
    mut_matrix = (snv == 1).astype(float)
    norm_matrix = (snv != 1).astype(float)

    A = np.dot(mut_matrix, mut_matrix.T)
    B = np.dot(norm_matrix, norm_matrix.T)
    B = snv.shape[1] - B

    P = A / B
    # Normalize matrix P
    P = P / np.sqrt(P.sum(1)).reshape(-1, 1) / np.sqrt(P.sum(1)).reshape(1, -1)

    # Truncated SVD for dimensionality reduction
    model = TruncatedSVD(n_components=50, random_state=42)
    pc = model.fit_transform(P)
    v = model.explained_variance_ratio_

    v = v / v[1]  # normalize variance by second component
    sorted_idx = np.argsort(-v)
    v = v[sorted_idx]
    pc = pc[:, sorted_idx]

    # Select PCs above threshold and skip first PC
    pc = pc[:, (v > thred)][:, 1:]

    # Construct symmetric k-nearest neighbor graph
    neighbor_graph = kneighbors_graph(pc, n_neighbors=kneighbors, include_self=False)
    connectivity = neighbor_graph + neighbor_graph.T

    return connectivity, int((v > thred).sum())

def run_node2vec_from_csv(csv_path, dim=64, kneighbors=15, thred=0.05, epochs=20, lr=0.05):
    """
    Given a path to SNV CSV file (cells x mutations),
    run the full Node2Vec pipeline and return embeddings.

    Returns:
        embedding (np.ndarray): N x dim embedding matrix
        cell_ids (list): list of cell IDs in order
        n_pcs (int): number of principal components used
    """
    # Load SNV matrix
    snv_df = pd.read_csv(csv_path, index_col=0).fillna(0)
    snv_matrix = snv_df.values

    # Calculate connectivity matrix
    connectivity, n_pcs = cal_cell_connectivity(snv_matrix, kneighbors, thred)

    # Convert to COO for edge_index
    connectivity = connectivity.tocoo()
    edge_index = np.vstack((connectivity.row, connectivity.col))
    edge_index_torch = torch.LongTensor(edge_index)

    # Create PyG Data object
    data = Data(edge_index=edge_index_torch)

    # Initialize Node2Vec model
    node2vec = Node2Vec(
        data.edge_index,
        embedding_dim=dim,
        walks_per_node=10,
        walk_length=40,
        context_size=5,
        p=1,
        q=1,
        num_negative_samples=5,
    )
    optimizer = torch.optim.Adam(node2vec.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        node2vec.train()
        total_loss = 0
        loader = node2vec.loader(batch_size=128, shuffle=True)
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

    # Get embeddings
    node2vec.eval()
    embedding = node2vec().detach().numpy()

    return embedding, snv_df.index.tolist(), n_pcs

