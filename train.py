import os
import time
import argparse
from mpi4py import MPI
import torch.distributed as dist
from rf import RandomForest
from utils import get_dataset_by_rank, get_local_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("model_output", type=str)
    parser.add_argument("n_trees", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # load data
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
    # if RANK == 0:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # RANK = int(os.environ['RANK'])
    df = get_dataset_by_rank(RANK, args.dataset_path)
    local_vocab = get_local_vocab(RANK, MPI.COMM_WORLD, df)

    dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)
    if RANK == 0:
        print(f"Starting distributed training with {WORLD_SIZE} processes.")

    # create and train model
    rf = RandomForest(RANK, WORLD_SIZE, args.n_trees, max_depth=10, seed=args.seed, vocab=local_vocab)
    dist.barrier()
    start = time.time()
    rf.fit(df)
    dist.barrier()
    end = time.time()
    rf.save_to_file(args.model_output)
    if RANK == 0:
        parallel_time = end - start
        print(f"NPROC={WORLD_SIZE} | NTrees={args.n_trees} | Time: {parallel_time:.4f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
