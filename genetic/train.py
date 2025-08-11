import argparse
import numpy as np
import pandas as pd
from genetic.model import run_node2vec_from_csv

def main():
    parser = argparse.ArgumentParser(description="Train genetic embedding with Node2Vec")
    parser.add_argument("--snv_csv", type=str, required=True, help="Path to SNV CSV file (cells x mutations)")
    parser.add_argument("--output", type=str, required=True, help="Path to save genetic embedding npy file")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension size")
    parser.add_argument("--kneighbors", type=int, default=15, help="Number of neighbors in connectivity graph")
    parser.add_argument("--thred", type=float, default=0.05, help="Threshold for selecting PCs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of Node2Vec training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for Node2Vec optimizer")

    args = parser.parse_args()

    print("Loading SNV data and training Node2Vec embeddings...")
    embedding, cell_ids, n_pcs = run_node2vec_from_csv(
        args.snv_csv,
        dim=args.embedding_dim,
        kneighbors=args.kneighbors,
        thred=args.thred,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"Finished training. Number of PCs used: {n_pcs}")
    print(f"Saving embedding matrix to {args.output}")

    # Save embedding matrix as numpy file
    np.save(args.output, embedding)

    # Optional: Save cell IDs in the same order
    cell_id_path = args.output.replace(".npy", "_cell_ids.txt")
    with open(cell_id_path, "w") as f:
        for cid in cell_ids:
            f.write(f"{cid}\n")

    print("Done.")

if __name__ == "__main__":
    main()

