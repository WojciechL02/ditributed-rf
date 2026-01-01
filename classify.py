import os
import pickle
import argparse
import pandas as pd
import torch
import torch.distributed as dist
from rf import RandomForest
from utils import vectorize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_input", type=str)
    parser.add_argument("n_trees", type=int)
    parser.add_argument("query_input", type=str)
    parser.add_argument("predictions_output", type=str)
    args = parser.parse_args()

    # distributed setup
    # WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)
    if RANK == 0:
        print(f"Starting distributed classification with {WORLD_SIZE} processes.")

    # load data
    df = pd.read_csv(f"{args.query_input}.csv")
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    data_X = vectorize(df["text"], vocab)
    data_y = torch.tensor(df["label"].values)

    # load model
    rf = RandomForest(RANK, WORLD_SIZE, args.n_trees, max_depth=10, seed=42)
    rf.load_from_file(args.model_input)
    preds = rf.predict(data_X)

    if RANK == 0:
        correct = (data_y == preds).float()
        acc = correct.mean().item()
        print(f"NPROC={WORLD_SIZE} | Acc: {acc:.3f}")
        with open(f"{args.predictions_output}_{WORLD_SIZE}.txt", "w") as f:
            for pred in preds:
                f.write(f"{pred}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
