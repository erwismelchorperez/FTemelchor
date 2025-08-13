# ft4cip.py
import numpy as np
from split_utils import SplitUtils
from pruning import CostComplexityPruner
class FT4CIP:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        """Entrena recursivamente el árbol."""
        if depth == 0:
            self.tree = {}

        if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.mean(y)

        feature_indices = list(range(X.shape[1]))
        best_split, best_score = SplitUtils.mfl_d_split(X, y, feature_indices)

        if best_split is None:
            return np.mean(y)

        X_left, X_right, mask = SplitUtils.apply_univariate_split(X, best_split[0], best_split)
        left_branch = self.fit(X_left, y[mask], depth + 1)
        right_branch = self.fit(X_right, y[~mask], depth + 1)

        node = {
            'split': best_split,
            'score': best_score,
            'left': left_branch,
            'right': right_branch
        }

        if depth == 0:
            self.tree = node
        return node

    def predict_instance(self, x, node=None):
        """Predice una sola instancia."""
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            return node

        feature_index, threshold = node['split']
        if x[feature_index] <= threshold:
            return self.predict_instance(x, node['left'])
        else:
            return self.predict_instance(x, node['right'])

    def predict(self, X):
        """Predice para todo un conjunto."""
        return np.array([self.predict_instance(row) for row in X])
