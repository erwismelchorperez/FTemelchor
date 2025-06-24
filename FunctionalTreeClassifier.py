from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("./"))
from FunctionalNode import FunctionalNode

class FunctionalTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.feature_types = {}  # To store feature types

    def _detect_feature_types(self, X):
        self.feature_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types[col] = 'numerical'
            else:
                self.feature_types[col] = 'categorical'

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(X) == 0 or y.nunique() == 1:
            pred = y.mode().iloc[0] if len(y) > 0 else "Desconocido"
            return FunctionalNode(prediction=pred, support=len(y))

        branches = []
        for col in X.columns:
            if self.feature_types[col] == 'numerical':
                threshold = X[col].median()
                cond_str_true = f"{col} > {threshold:.2f}"
                cond_func_true = lambda row, c=col, t=threshold: row[c] > t
                cond_str_false = f"{col} <= {threshold:.2f}"
                cond_func_false = lambda row, c=col, t=threshold: row[c] <= t

                idx_true = X[col] > threshold
                idx_false = X[col] <= threshold

                child_true = self._build_tree(X[idx_true], y[idx_true], depth - 1)
                child_false = self._build_tree(X[idx_false], y[idx_false], depth - 1)

                branches.append((cond_str_true, cond_func_true, child_true))
                branches.append((cond_str_false, cond_func_false, child_false))

            else:  # categorical
                for val in X[col].unique():
                    cond_str = f"{col} == {val}"
                    cond_func = lambda row, c=col, v=val: row[c] == v
                    idx = X[col] == val
                    child = self._build_tree(X[idx], y[idx], depth - 1)
                    branches.append((cond_str, cond_func, child))

        return FunctionalNode(branches=branches)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self._detect_feature_types(X)
        else:
            raise ValueError("Input X must be a pandas DataFrame")

        self.tree = self._build_tree(X, y, self.max_depth)

    def predict(self, X):
        return X.apply(self.tree.predict, axis=1)


