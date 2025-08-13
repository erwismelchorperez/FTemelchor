# split_utils.py
import numpy as np

class SplitUtils:
    @staticmethod
    def twoing_score(y_left, y_right):
        """Calcula el puntaje Twoing para medir la separación entre dos particiones."""
        n_left = len(y_left)
        n_right = len(y_right)
        if n_left == 0 or n_right == 0:
            return 0

        p_left = np.mean(y_left)
        p_right = np.mean(y_right)
        return (n_left * n_right / (n_left + n_right) ** 2) * (abs(p_left - p_right) ** 2)

    @staticmethod
    def candidate_splits_univariate(X, feature_index, feature_values=None, n_thresholds=5):
        """Genera posibles splits para una característica numérica o categórica."""
        if feature_values is None:
            values = np.unique(X[:, feature_index])
        else:
            values = feature_values

        if len(values) <= n_thresholds:
            thresholds = values
        else:
            thresholds = np.percentile(values, np.linspace(0, 100, n_thresholds + 2)[1:-1])

        return [(feature_index, t) for t in thresholds]

    @staticmethod
    def apply_univariate_split(X, feature_index, split):
        """Divide el dataset en dos grupos usando un split univariado."""
        threshold = split[1]
        mask = X[:, feature_index] <= threshold
        return X[mask], X[~mask], mask

    @staticmethod
    def mfl_d_split(X, y, feature_indices):
        """Evalúa el mejor split basado en Twoing Score."""
        best_score = -np.inf
        best_split = None
        for feature_index in feature_indices:
            splits = SplitUtils.candidate_splits_univariate(X, feature_index)
            for split in splits:
                X_left, X_right, mask = SplitUtils.apply_univariate_split(X, feature_index, split)
                score = SplitUtils.twoing_score(y[mask], y[~mask])
                if score > best_score:
                    best_score = score
                    best_split = split
        return best_split, best_score

    @staticmethod
    def sequential_forward_selection(X, y, F_indices, evaluate_split_fn, split_generator_fn, max_features=None):
        """Selecciona secuencialmente las mejores características."""
        if max_features is None:
            max_features = len(F_indices)

        selected = []
        remaining = list(F_indices)
        best_score = -np.inf

        while len(selected) < max_features and remaining:
            candidate_best = None
            for f in remaining:
                split = split_generator_fn(X, f)
                score = evaluate_split_fn(split)
                if score > best_score:
                    best_score = score
                    candidate_best = f
            if candidate_best is None:
                break
            selected.append(candidate_best)
            remaining.remove(candidate_best)

        return selected, best_score
