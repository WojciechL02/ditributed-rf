import pandas as pd
import torch
from torch import Tensor
import torch.distributed as dist
import numpy as np
from tree import DecisionTree
from typing import Tuple
from pandas.core.series import Series
from utils import count_vectorizer


class RandomForest:
    def __init__(self, rank: int, world_size: int, n_estimators: int, max_depth: int=None, seed: int=42, vocab: set=None) -> None:
        self.n_global_estimators = n_estimators
        split_ = self.n_global_estimators // world_size
        remainder_ = self.n_global_estimators % world_size
        self.n_estimators = split_ if rank < world_size - 1 else split_ + remainder_
        self.max_depth = max_depth
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.n_classes = 27
        self.vocab = None if vocab is None else sorted(list(vocab))
        self.estimators = []

    def fit(self, df: pd.DataFrame) -> None:
        if self.vocab is None:
            raise ValueError("Vocab is empty, call load_from_file first.")
        self.n_classes = len(df["label"].unique())
        # self.n_classes = len(y.unique())
        for i in range(self.n_estimators):
            self.estimators.append(DecisionTree(num_features=len(self.vocab), max_depth=self.max_depth))
            self._seed(i)
            df_sample = df.sample(n=len(df), replace=True)  # bootstrap
            X = count_vectorizer(df_sample["text"], self.vocab)
            y = torch.tensor(df_sample["label"].values)
            # X_sample, y_sample = self._bootstrap_sample(X, y, sample_size=1.0)
            # X_sample = X.to_sparse_csr()
            # self.estimators[-1].fit(X_sample, y_sample)
            self.estimators[-1].fit(X, y)

    def predict(self, df: pd.DataFrame) -> Tensor:
        if self.vocab is None:
            raise ValueError("Vocab is empty, call load_from_file first.")
        X = count_vectorizer(df["text"], self.vocab)
        predictions = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        all_classes = torch.zeros((X.size(0), self.n_classes))
        for c in range(self.n_classes):
            all_classes[:, c] = (predictions == c).sum(dim=0)

        dist.all_reduce(all_classes, op=dist.ReduceOp.SUM)
        global_decision = all_classes.argmax(dim=1)
        return global_decision

    # def _bootstrap_sample(self, X: Tensor, y: Tensor, sample_size: float) -> Tuple[Tensor, Tensor]:
    #     num_samples = int(sample_size * len(X))
    #     indices = torch.randint(0, len(X), size=(num_samples,))
    #     X_sample = X[indices]
    #     y_sample = y[indices]
    #     return X_sample, y_sample

    def _seed(self, i: int) -> None:
        s = self.seed * 1000 * self.rank + i
        np.random.seed(s)
        torch.manual_seed(s)

    def save_to_file(self, filename: str) -> None:
        with open(f"{filename}_{self.rank}.txt", "w") as f:
            f.write(" ".join(self.vocab))
            f.write("\n")
            f.write(str(self))

    def load_from_file(self, filename: str) -> None:
        with open(f"{filename}_{self.rank}.txt", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    self.vocab = line.split()
                    self.estimators = [DecisionTree(num_features=len(self.vocab), max_depth=self.max_depth) for _ in range(self.n_estimators)]
                else:
                    self.estimators[i - 1].load_from_string(line.strip())

    def __str__(self) -> str:
        result = ""
        for i, estimator in enumerate(self.estimators):
            if i > 0:
                result += "\n"
            result += str(estimator)
        return result


def main():
    from time import process_time
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    # data_X = torch.randint(0, 5, (10000, 100))
    # data_y = torch.randint(0, 2, (10000,))

    N = 30  # number of samples
    D = 100000  # number of binary features
    num_classes = 15

    # Generate random binary features (0/1)
    # X = torch.randint(0, 2, (N, D), dtype=torch.int32)
    N, D = 1000, 500
    X = torch.zeros((N, D), dtype=torch.int32)
    col_indices = torch.stack([torch.randperm(D)[:10] for _ in range(N)])
    row_indices = torch.arange(N).view(-1, 1).expand(-1, 10)
    X[row_indices, col_indices] = 1.0
    f5 = X[:, 5] == 1
    f10 = X[:, 10] == 1
    f50 = X[:, 50] == 1
    f15 = X[:, 15] == 1
    f30 = X[:, 30] == 1
    f499 = X[:, 499] == 1
    fmany = (X[:, [1, 3, 13, 14, 17, 477, 346, 245, 276, 123, 324, 399, 129, 199]] == 1).any(dim=1)
    y = ((f5 & f10) | f50 | (f15 & f30) | f499 | fmany).long()
    print(y.unique(return_counts=True))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for _ in range(10):
        x = 10 * 2

    rf = RandomForest(vocab={f"a{i}" for i in range(D)}, rank=0, world_size=1, n_estimators=5, max_depth=10)
    st = process_time()
    rf.fit(X_train, y_train)
    et = process_time()
    print(et - st)
    y_pred = rf.predict(X_test)
    print("score:", accuracy_score(y_test, y_pred))
    rf.save_to_file("model")

    rf1 = RandomForest(vocab=None, rank=0, world_size=1, n_estimators=5, max_depth=10)
    rf1.load_from_file("model")

    # srf = RandomForestClassifier(n_estimators=5, max_depth=10)
    # st = process_time()
    # srf.fit(X_train, y_train)
    # et = process_time()
    # print(et - st)
    # y_pred = srf.predict(X_test)
    # print("score:", accuracy_score(y_test, y_pred))
#
#
if __name__ == "__main__":
    main()