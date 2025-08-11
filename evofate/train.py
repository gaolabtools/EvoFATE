import argparse
import torch
import torch.optim as optim
from evofate.model import GATEncoder, MLPPredictor, BGRL, augment_graph, prepare_data
import numpy as np
import pandas as pd

def train_evofate(
    expression_csv, genetic_embedding_path, output_path,
    embedding_dim=64, hidden_dim=64, epochs=2000, lr=1e-3, kneighbors=15,
):
    # Load genetic embedding
    genetic_embedding = np.load(genetic_embedding_path)

    # Prepare graph data
    x, edge_index = prepare_data(expression_csv, genetic_embedding, kneighbors=kneighbors)

    # Setup model
    encoder = GATEncoder(in_dim=x.size(1), hidden_dim=hidden_dim, heads=1)
    predictor = MLPPredictor(hidden_dim=hidden_dim, out_dim=embedding_dim)
    model = BGRL(encoder, predictor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        x1, edge_index1 = augment_graph(x, edge_index)
        x2, edge_index2 = augment_graph(x, edge_index)

        loss, embedding = model(x1, edge_index1, x2, edge_index2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_encoder()

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Return final embedding tensor (detach and numpy)
    model.eval()
    with torch.no_grad():
        final_embedding = model.encoder_online(x, edge_index).cpu().numpy()

    # Save embedding
    np.save(output_path, final_embedding)
    print(f"EvoFATE embedding saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train EvoFATE embedding with BGRL")
    parser.add_argument("--expression_csv", type=str, required=True, help="Expression CSV file path")
    parser.add_argument("--genetic_embedding", type=str, required=True, help="Genetic embedding npy file path")
    parser.add_argument("--output", type=str, required=True, help="Output embedding npy file path")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size for GAT")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--kneighbors", type=int, default=15, help="k for kNN graph")

    args = parser.parse_args()

    train_evofate(
        args.expression_csv, args.genetic_embedding, args.output,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        epochs=args.epochs, lr=args.lr, kneighbors=args.kneighbors
    )

if __name__ == "__main__":
    main()

