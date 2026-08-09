import numpy as np


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        samples, features = X.shape
        labels = len(np.unique(y))

        if depth >= self.max_depth or labels == 1 or samples < self.min_samples_split:
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)
        feat_idxs = np.random.choice(features, features, replace=False)
        best_feat, best_threshold = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thres = None, None
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self.__information_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thres = threshold

        return split_idx, split_thres

    def _information_gain(self, y, X_col, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self.split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)

        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_col, split_thres):
        left_idxs = np.argwhere(X_col <= split_thres).flatten()
        right_idxs = np.argwhere(X_col > split_thres).flatten()

        return left_idxs, right_idxs

    def most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, maxdepth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.maxdepth = maxdepth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape
        indices = np.rnadom.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                maxdepth=self.maxdepth, min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array(
            [np.bincount(tree_preds[:, i]).argmax() for i in range(X.shape[0])]
        )
