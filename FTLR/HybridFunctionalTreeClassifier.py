from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("./"))
from FunctionalNode import FunctionalNode
class HybridFunctionalTreeClassifier:
    def __init__(self, max_depth=3, criterion='gini', alpha=0.5, beta=0.5, hybrid = True):
        self.max_depth = max_depth
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.tree = None
        self.feature_types = {}
        self.hybrid = hybrid
    def _detect_feature_types(self, X):
        self.feature_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types[col] = 'numerical'
            else:
                self.feature_types[col] = 'categorical'
    def _impurity(self, y):
        counts = y.value_counts(normalize=True)
        if self.criterion == 'gini':
            return 1 - np.sum(counts ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(counts * np.log2(counts + 1e-9))
        else:
            raise ValueError(f"Criterio desconocido: {self.criterion}")
    def _twoing_score(self, X_left, y_left, X_right, y_right):
        n_total = len(y_left) + len(y_right)
        if n_total == 0 or len(y_left) == 0 or len(y_right) == 0:
            return 0

        P_L = len(y_left) / n_total
        P_R = len(y_right) / n_total

        all_classes = list(set(y_left) | set(y_right))

        def class_probs(y, classes):
            counter = Counter(y)
            total = len(y)
            return np.array([counter.get(cls, 0) / total for cls in classes])

        p_left = class_probs(y_left, all_classes)
        p_right = class_probs(y_right, all_classes)

        diff = np.abs(p_left - p_right)
        score = P_L * P_R * (np.sum(diff))**2
        return score
    def _functional_score(self, X, y):
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression(max_iter=500, solver='liblinear')
            model.fit(X_scaled, y)
            return model.score(X_scaled, y)
        except:
            return 0
    def _score_split(self, X_left, y_left, X_right, y_right):
        n = len(y_left) + len(y_right)
        if self.criterion == 'twoing':
            # Queremos maximizar Twoing, así que lo transformamos para que sea un "costo" (minimizar)
            impurity_component = -self._twoing_score(X_left, y_left, X_right, y_right)
        else:
            impurity_left = self._impurity(y_left)
            impurity_right = self._impurity(y_right)
            #weighted_impurity = (len(y_left)/n)*impurity_left + (len(y_right)/n)*impurity_right
            impurity_component = (len(y_left)/n)*impurity_left + (len(y_right)/n)*impurity_right

        fs_left = self._functional_score(X_left, y_left)
        fs_right = self._functional_score(X_right, y_right)
        weighted_fs = (len(y_left)/n)*fs_left + (len(y_right)/n)*fs_right

        # Queremos minimizar este score
        # return self.alpha * (1 - weighted_fs) + (1 - self.alpha) * weighted_impurity
        return self.alpha * (1 - weighted_fs) + (1 - self.alpha) * impurity_component
    def _score_split_hybrid(self, X_left, y_left, X_right, y_right):
        n = len(y_left) + len(y_right)

        impurity_left = self._impurity(y_left)
        impurity_right = self._impurity(y_right)
        weighted_impurity = (len(y_left)/n)*impurity_left + (len(y_right)/n)*impurity_right

        fs_left = self._functional_score(X_left, y_left)
        fs_right = self._functional_score(X_right, y_right)
        weighted_fs = (len(y_left)/n)*fs_left + (len(y_right)/n)*fs_right

        twoing = self._twoing_score(X_left, y_left, X_right, y_right)
        twoing_penalty = -twoing  # Queremos maximizarlo, así que usamos su negativo

        # Nueva combinación con parámetro beta
        # self.alpha controla funcional, (1-alpha)*(1-beta) para impurity, y (1-alpha)*beta para twoing
        return self.alpha * (1 - weighted_fs) + (1 - self.alpha) * (
            self.beta * twoing_penalty + (1 - self.beta) * weighted_impurity
        )
    def _build_tree(self, X, y, depth):
        if depth == 0 or len(X) == 0 or y.nunique() == 1:
            pred = y.mode().iloc[0] if len(y) > 0 else "Desconocido"
            return FunctionalNode(prediction=pred, support=len(y))

        best_score = float('inf')
        best_split = None

        for col in X.columns:
            if self.feature_types[col] == 'numerical':
                threshold = X[col].median()
                idx_left = X[col] <= threshold
                idx_right = X[col] > threshold
                if idx_left.sum() == 0 or idx_right.sum() == 0:
                    continue
                if self.hybrid:
                    score = self._score_split_hybrid(X[idx_left], y[idx_left], X[idx_right], y[idx_right])
                else:
                    score = self._score_split(X[idx_left], y[idx_left], X[idx_right], y[idx_right])
                if score < best_score:
                    best_score = score
                    best_split = (col, 'numerical', threshold, idx_left, idx_right)

            else:  # categorical
                for val in X[col].unique():
                    idx_left = X[col] == val
                    idx_right = X[col] != val
                    if idx_left.sum() == 0 or idx_right.sum() == 0:
                        continue
                    
                    if self.hybrid:
                        score = self._score_split_hybrid(X[idx_left], y[idx_left], X[idx_right], y[idx_right])
                    else:
                        score = self._score_split(X[idx_left], y[idx_left], X[idx_right], y[idx_right])
                    if score < best_score:
                        best_score = score
                        best_split = (col, 'categorical', val, idx_left, idx_right)

        if best_split is None:
            pred = y.mode().iloc[0] if len(y) > 0 else "Desconocido"
            return FunctionalNode(prediction=pred, support=len(y))

        col, typ, val, idx_left, idx_right = best_split
        branches = []

        if typ == 'numerical':
            cond_str_true = f"{col} > {val:.2f}"
            cond_func_true = lambda row, c=col, t=val: row[c] > t
            cond_str_false = f"{col} <= {val:.2f}"
            cond_func_false = lambda row, c=col, t=val: row[c] <= t
        else:
            cond_str_true = f"{col} != {val}"
            cond_func_true = lambda row, c=col, v=val: row[c] != v
            cond_str_false = f"{col} == {val}"
            cond_func_false = lambda row, c=col, v=val: row[c] == v

        child_left = self._build_tree(X[idx_left], y[idx_left], depth - 1)
        child_right = self._build_tree(X[idx_right], y[idx_right], depth - 1)

        branches.append((cond_str_false, cond_func_false, child_left))
        branches.append((cond_str_true, cond_func_true, child_right))

        return FunctionalNode(branches=branches)
    def fit(self, X, y):
        self._detect_feature_types(X)
        self.tree = self._build_tree(X, y, self.max_depth)
    def predict(self, X):
        return X.apply(self.tree.predict, axis=1)
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
        }
    def extract_rules(self):
        return self.tree.extract_rules()
        #return self.tree.extract_rules(min_depth=3)  # Devuelve todas las reglas de longitud 2 hasta el largo máximo