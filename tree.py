import torch
from torch import Tensor
from sklearn.tree import DecisionTreeClassifier
from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ast


class Node:
    def __init__(self, feature: int=None, label: int=None) -> None:
        self.feature = feature
        self.label = label
        self.children = {}

    def add_child(self, value: int) -> None:
        self.children[value] = Node()

    def __str__(self) -> str:
        if self.label is not None:
            return f"[{self.label}]"
        return f"[{self.feature}, {str(self.children[1])}, {str(self.children[0])}]"


class DecisionTree:
    def __init__(self, max_depth: int=None) -> None:
        self.max_depth = max_depth
        self.classes = None
        self.n_classes = None
        self.num_features = None
        self.root = Node()

    def fit(self, X: Tensor, y: Tensor) -> None:
        self.classes = torch.unique(y)
        self.n_classes = max(self.classes) + 1
        self.num_features = X.size(1)
        self._fit_recursive(self.root, X, y, set(), 0)

    def _fit_recursive(self, node: Node, X: Tensor, y: Tensor, used_features: set, depth=None) -> None:
        classes, counts = torch.unique(y, return_counts=True)
        parent_counts = torch.bincount(y, minlength=self.n_classes)

        if len(classes) == 1:
            node.label = classes[0]
            return

        if (self.max_depth is not None and depth == self.max_depth) or len(used_features) == self.num_features:
            node.label = classes[torch.argmax(counts)].item()
            return

        sampled_features = self._sample_features(used_features)
        feature_class_counts = self._compute_counts(X, y)

        gains = torch.zeros(len(sampled_features))
        for i, f in enumerate(sampled_features):
            gains[i] = self._information_gain(f, y, feature_class_counts, parent_counts)

        feature = sampled_features[torch.argmax(gains)].item()
        node.feature = feature

        if torch.isclose(torch.max(gains), torch.zeros(1), atol=1e-9):
            node.label = classes[torch.argmax(counts)].item()
            return

        new_used_features = used_features | {feature}
        next_depth = None if self.max_depth is None else depth + 1
        # presence of word
        node.add_child(1)
        split_indices = X[:, feature] > 0
        self._fit_recursive(node.children[1], X[split_indices], y[split_indices], new_used_features, depth=next_depth)

        # absence of word
        node.add_child(0)
        split_indices = X[:, feature] == 0
        self._fit_recursive(node.children[0], X[split_indices], y[split_indices], new_used_features, depth=next_depth)

    def _compute_counts(self, X: Tensor, y: Tensor):
        feature_class_counts = torch.zeros((self.num_features, self.n_classes), dtype=torch.long)
        for c in range(len(self.classes)):
            class_mask = (y == c)
            if class_mask.sum() > 0:
                feature_class_counts[:, c] = (X[class_mask] > 0).sum(dim=0)
        return feature_class_counts

    def _gini_index(self, counts: Tensor, size: int) -> float:
        if size == 0:
            return 0.0
        probs = counts / size
        return 1.0 - torch.sum(probs * probs).item()

    def _information_gain(self, feature: int, y: Tensor, feature_class_counts: Tensor, parent_counts: Tensor) -> float:
        left_counts = feature_class_counts[feature]
        right_counts = parent_counts - left_counts

        n = y.size(0)
        if n == 0:
            return 0.0
        n_left = left_counts.sum().item()
        n_right = right_counts.sum().item()

        gini_parent = self._gini_index(parent_counts, n)
        gini_left = self._gini_index(left_counts, n_left)
        gini_right = self._gini_index(right_counts, n_right)
        return gini_parent - (n_left / n) * gini_left - (n_right / n) * gini_right

    def _sample_features(self, used_features: set) -> Tensor:
        k = int(self.num_features ** 0.5)
        all_features = torch.arange(self.num_features)
        mask = torch.ones(self.num_features, dtype=torch.bool)
        mask[list(used_features)] = False
        available = all_features[mask]

        if len(available) <= k:
            return available
        return available[torch.randperm(len(available))][:k]

    def _predict_sample(self, sample: Tensor) -> int:
        node = self.root
        while (node.feature is not None) and (len(node.children.keys()) > 0):
            idx = 1 if sample[node.feature] > 0 else 0
            node = node.children[idx]
        return node.label

    def predict(self, X: Tensor) -> Tensor:
        preds = torch.zeros(X.size(0))
        for i in range(X.size(0)):
            preds[i] = self._predict_sample(X[i])
        return preds

    def load_from_string(self, text: str):
        data = ast.literal_eval(text)

        def build_tree(node_data):
            new_node = Node()
            if len(node_data) == 1:
                new_node.label = node_data[0]
                return new_node

            new_node.feature = node_data[0]

            # Create left child
            child_1_data = node_data[1]
            new_node.children[1] = build_tree(child_1_data)

            # Create right child
            child_0_data = node_data[2]
            new_node.children[0] = build_tree(child_0_data)
            return new_node

        self.root = build_tree(data)

    def __str__(self) -> str:
        return str(self.root)


def main():
    # data_X = torch.randint(0, 5, (10000, 100))
    # data_y = torch.randint(0, 2, (10000,))

    N = 100  # number of samples
    D = 20  # number of binary features
    num_classes = 2

    # Generate random binary features (0/1)
    X = torch.randint(0, 2, (N, D), dtype=torch.float32)
    y = torch.randint(0, num_classes, (N,), dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for _ in range(10):
        x = 10 * 2

    dt = DecisionTree(max_depth=10)
    st = process_time()
    dt.fit(X_train, y_train)
    et = process_time()
    print(et - st)
    y_pred = dt.predict(X_test)
    print("score:", accuracy_score(y_test, y_pred))

    srf = DecisionTreeClassifier(max_depth=10)
    st = process_time()
    srf.fit(X_train, y_train)
    et = process_time()
    print(et - st)
    y_pred = srf.predict(X_test)
    print("score:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()


# def _sample_features(self, used_features: set) -> Tensor:
    #     k = int(self.num_features ** 0.5)
    #     all_features = set(torch.arange(self.num_features).tolist())
    #     available = Tensor(list(all_features - used_features))
    #     if len(available) <= k:
    #         return available
    #     return available[torch.randperm(len(available))][:k]

    # def _gini_split(self, feature: int, X: Tensor, y: Tensor, num_classes: int) -> float:
    #     left_mask = X[:, feature] > 0
    #     right_mask = ~left_mask
    #     left_counts = torch.bincount(
    #         y[left_mask],
    #         minlength=num_classes
    #     )
    #     right_counts = torch.bincount(
    #         y[right_mask],
    #         minlength=num_classes
    #     )
    #     n = y.size(0)
    #     n_left = left_counts.sum()
    #     n_right = right_counts.sum()
    #     gini_left = self._gini_index(left_counts)
    #     gini_right = self._gini_index(right_counts)
    #     return float((n_left / n) * gini_left + (n_right / n) * gini_right)
    #     # left_gini = self._gini_index(y[left_split], classes, counts)
    #     # right_gini = self._gini_index(y[right_split], classes, counts)
    #     # return sum(left_split) / length * left_gini + sum(right_split) / length * right_gini
    #
    # def _information_gain(self, feature: int, X: Tensor, y: Tensor, parent_counts: Tensor) -> float:
    #     parent_gini = self._gini_index(parent_counts)
    #     split_gini = self._gini_split(feature, X, y, parent_counts.size(0))
    #     return parent_gini - split_gini