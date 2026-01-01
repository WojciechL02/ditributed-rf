import torch
from torch import Tensor
import torch.distributed as dist
import numpy as np
from tree import DecisionTree
from typing import Tuple


class RandomForest:
    def __init__(self, rank: int, world_size: int, n_estimators: int, max_depth: int=None, seed: int=42) -> None:
        self.n_global_estimators = n_estimators
        split_ = self.n_global_estimators // world_size
        remainder_ = self.n_global_estimators % world_size
        self.n_estimators = split_ if rank < world_size - 1 else split_ + remainder_
        self.max_depth = max_depth
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.n_classes = 27
        self.estimators = [DecisionTree(max_depth=self.max_depth) for _ in range(self.n_estimators)]

    def fit(self, X: Tensor, y: Tensor) -> None:
        self.n_classes = len(y.unique())
        for i, estimator in enumerate(self.estimators):
            self._seed(i)
            X_sample, y_sample = self._bootstrap_sample(X, y, sample_size=1.0)
            estimator.fit(X_sample, y_sample)

    def predict(self, X: Tensor) -> Tensor:
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

    def _bootstrap_sample(self, X: Tensor, y: Tensor, sample_size: float) -> Tuple[Tensor, Tensor]:
        num_samples = int(sample_size * len(X))
        indices = torch.randint(0, len(X), size=(num_samples,))
        X_sample = X[indices]
        y_sample = y[indices]
        return X_sample, y_sample

    def _seed(self, i: int) -> None:
        s = self.seed * 1000 * self.rank + i
        np.random.seed(s)
        torch.manual_seed(s)

    def save_to_file(self, filename: str) -> None:
        with open(f"{filename}_{self.rank}.txt", "w") as f:
            f.write(str(self))

    def load_from_file(self, filename: str) -> None:
        with open(f"{filename}_{self.rank}.txt", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if i > 0:
                self.estimators[i].load_from_string(line.strip())

    def __str__(self) -> str:
        result = ""
        for i, estimator in enumerate(self.estimators):
            if i > 0:
                result += "\n"
            result += str(estimator)
        return result


# def main():
#     # data_X = torch.randint(0, 5, (10000, 100))
#     # data_y = torch.randint(0, 2, (10000,))
#
#     N = 100  # number of samples
#     D = 20  # number of binary features
#     num_classes = 2
#
#     # Generate random binary features (0/1)
#     X = torch.randint(0, 2, (N, D), dtype=torch.float32)
#     y = torch.randint(0, num_classes, (N,), dtype=torch.long)
#
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     for _ in range(10):
#         x = 10 * 2
#
#     rf = RandomForest(rank=1, n_estimators=10, max_depth=10)
#     st = process_time()
#     rf.fit(X_train, y_train)
#     et = process_time()
#     print(et - st)
#     y_pred = rf.predict(X_test)
#     print("score:", accuracy_score(y_test, y_pred))
#     # rf.save_to_file("model.txt")
#
#     # srf = RandomForestClassifier(n_estimators=10, max_depth=10)
#     # st = process_time()
#     # srf.fit(X_train, y_train)
#     # et = process_time()
#     # print(et - st)
#     # y_pred = srf.predict(X_test)
#     # print("score:", accuracy_score(y_test, y_pred))
#
#
# if __name__ == "__main__":
#     main()