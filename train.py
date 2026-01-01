import os
import time
import pickle
import argparse
import pandas as pd
import torch
import torch.distributed as dist
from rf import RandomForest
from utils import vectorize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("model_output", type=str)
    parser.add_argument("n_trees", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # distributed setup
    # WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)
    if RANK == 0:
        print(f"Starting distributed training with {WORLD_SIZE} processes.")

    # load data
    df = pd.read_csv(f"{args.dataset_path}_{RANK}.csv")
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    data_X = vectorize(df["text"], vocab)
    data_y = torch.tensor(df["label"].values)

    # create and train model
    rf = RandomForest(RANK, WORLD_SIZE, args.n_trees, max_depth=10, seed=args.seed)
    dist.barrier()
    start = time.time()
    rf.fit(data_X, data_y)
    dist.barrier()
    end = time.time()
    rf.save_to_file(args.model_output)
    if RANK == 0:
        parallel_time = end - start
        print(f"NPROC={WORLD_SIZE} | NTrees={args.n_trees} | Time: {parallel_time:.4f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
