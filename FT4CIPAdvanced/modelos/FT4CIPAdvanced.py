import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from joblib import Memory

memory = Memory(location="./cache_ft4cip", verbose=0)


class FT4CIPAdvanced(BaseEstimator, ClassifierMixin):
    """
    Functional Tree for Class Imbalance Problems (FT4CIPAdvanced)
    - Entrena un DecisionTree base (para particionar)
    - Ajusta modelos funcionales locales (LogisticRegression) en hojas no puras
    - Soporte nativo de categóricos (codificados internamente)
    - Método extract_rules() que devuelve un DataFrame con patrones para TODOS los nodos
    """

    def __init__(self, max_depth=5, alpha=0.5, beta=0.3, criterion="gini_auc", random_state=42):
        self.max_depth = max_depth
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.random_state = random_state
        self.tree_ = None
        self.functional_models_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.array(y)

        # detecta categóricas
        self.cat_features_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_features_ = [c for c in X.columns if c not in self.cat_features_]
        self.feature_names_ = list(X.columns)

        # codifica categóricas internamente (LabelEncoder)
        self.encoders_ = {}
        for col in self.cat_features_:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders_[col] = le

        # valores observados
        self.observed_values_ = {c: sorted(pd.Series(X[c]).unique().tolist()) for c in X.columns}

        # entrenar árbol
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            criterion="gini",
        )
        self.tree_.fit(X, y)

        # entrenar modelos funcionales en hojas
        self._fit_functional_models(X, y)

        return self

    def _fit_functional_models(self, X, y):
        self.functional_models_ = {}
        self.pure_class_ = {}
        self.pure_rules_ = []

        leaves = np.unique(self.tree_.apply(X))

        for leaf_id in leaves:
            mask_leaf = self.tree_.apply(X) == leaf_id
            X_leaf = X[mask_leaf]
            y_leaf = y[mask_leaf]

            unique_classes = np.unique(y_leaf)
            if len(unique_classes) == 1:
                # nodo puro (aunque tenga 1-2 muestras)
                clase_pura = unique_classes[0]
                self.pure_class_[leaf_id] = clase_pura
                continue

            # si hay muy pocas muestras y clases mixtas, omitir entrenamiento funcional
            if len(X_leaf) < 3:
                continue

            try:
                selector = SequentialFeatureSelector(
                    LogisticRegression(max_iter=200, solver='liblinear'),
                    n_features_to_select=min(5, X_leaf.shape[1]),
                    direction='forward',
                    scoring='roc_auc',
                    cv=min(3, len(X_leaf))
                )
                # SFS solo en features numéricas (login: leave as numeric subset)
                selector.fit(X_leaf[self.num_features_], y_leaf)
                selected_mask = selector.get_support()
                selected_features = list(pd.Series(self.num_features_)[selected_mask])
                model = LogisticRegression(max_iter=300, solver='liblinear')
                model.fit(X_leaf[selected_features], y_leaf)

                self.functional_models_[leaf_id] = {
                    "model": model,
                    "features": selected_features
                }
            except Exception as e:
                # no bloquee proceso si falla en una hoja
                print(f"❌ Error en hoja {leaf_id}: {e}")

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = pd.DataFrame(X).copy()

        # aplicar codificadores
        for col, le in getattr(self, "encoders_", {}).items():
            if col in X.columns:
                X[col] = X[col].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        preds = []
        leaf_ids = self.tree_.apply(X)

        for i, leaf_id in enumerate(leaf_ids):
            if leaf_id in self.pure_class_:
                preds.append(self.pure_class_[leaf_id])
                continue
            if leaf_id in self.functional_models_:
                func_info = self.functional_models_[leaf_id]
                feats = func_info["features"]
                model = func_info["model"]
                xi = X.iloc[[i]][feats]
                prob = model.predict_proba(xi)[:, 1][0]
                preds.append(1 if prob >= 0.5 else 0)
            else:
                # fallback: usar árbol base
                preds.append(self.tree_.predict(X.iloc[[i]])[0])
        return np.array(preds)

    def predict_proba(self, X):
        check_is_fitted(self, "tree_")
        X = pd.DataFrame(X).copy()

        for col, le in getattr(self, "encoders_", {}).items():
            if col in X.columns:
                X[col] = X[col].astype(str).apply(lambda v: v if v in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        leaf_ids = self.tree_.apply(X)
        probs = []
        for i, leaf in enumerate(leaf_ids):
            if leaf in self.functional_models_:
                model = self.functional_models_[leaf]["model"]
                feats = self.functional_models_[leaf]["features"]
                xi = X.iloc[[i]][feats]
                probs.append(model.predict_proba(xi)[0, 1])
            else:
                probs.append(self.tree_.predict_proba(X.iloc[[i]])[0, 1])
        return np.column_stack((1 - np.array(probs), np.array(probs)))

    def extract_rules(self, X=None, y=None, export_path=None):
        """
        Extrae patrones (reglas) para TODOS los nodos del árbol (no solo hojas).
        Devuelve un DataFrame con columnas:
        pattern, n0, support_n0_pct, n1, support_n1_pct, prob, dominant_class, depth

        - X, y: opcionales, solo se usan para calcular soporte relativo (%) respecto a len(y)
        - export_path: si se pasa, guarda 'patterns_all_nodes.xlsx' ahí
        """
        from collections import deque
        check_is_fitted(self, "tree_")

        tree = self.tree_.tree_
        feature = tree.feature
        threshold = tree.threshold
        feature_names = self.feature_names_

        # total para %: preferir y si se da, sino usar root count
        if y is not None:
            n_total = len(y)
        else:
            n_total = int(tree.n_node_samples[0])

        def _observed_codes(colname):
            if colname in self.observed_values_:
                return sorted(self.observed_values_[colname])
            if colname in self.encoders_:
                return list(range(len(self.encoders_[colname].classes_)))
            return None

        def _codes_left_labels(colname, thr):
            codes = _observed_codes(colname)
            if codes is None:
                return None
            left_codes = [c for c in codes if c <= thr]
            if colname in self.encoders_:
                encoder = self.encoders_[colname]
                labels = [str(encoder.inverse_transform([c])[0]) for c in left_codes]
            else:
                labels = [str(c) for c in left_codes]
            return labels

        rules = []
        stack = deque()
        stack.append((0, [], 0))  # node_id, conds, depth

        while stack:
            node_id, conds, depth = stack.pop()

            # obtain value counts from tree.value
            value = tree.value[node_id][0]
            total_node = float(value.sum())
            if total_node == 0:
                # no samples reached this node in training
                continue

            n0 = int(value[0]) if value.shape[0] > 0 else 0
            n1 = int(value[1]) if value.shape[0] > 1 else 0
            prob = (n1 / total_node) if total_node > 0 else 0.0
            dominant_class = "positive" if n1 >= n0 else "negative"

            cond_text = " AND ".join(conds) if conds else "(root)"
            rules.append({
                "pattern": cond_text,
                "n0": n0,
                "support_n0(%)": round((n0 / n_total) * 100, 6),
                "n1": n1,
                "support_n1(%)": round((n1 / n_total) * 100, 6),
                "prob": round(prob, 6),
                "dominant_class": dominant_class,
                "depth": depth
            })

            # si es interno, expandir children
            if tree.children_left[node_id] != tree.children_right[node_id]:
                feat_idx = feature[node_id]
                # en algunos casos feat_idx puede ser -2, safeguard:
                if feat_idx < 0 or feat_idx >= len(feature_names):
                    continue
                feat_name = feature_names[feat_idx]
                thr = threshold[node_id]

                if feat_name in getattr(self, "cat_features_", []):
                    left_labels = _codes_left_labels(feat_name, thr)
                    if left_labels is None:
                        cond_left = f"{feat_name} <= {thr:.6f}"
                        cond_right = f"{feat_name} > {thr:.6f}"
                    else:
                        lbls = ",".join([f"'{l}'" for l in left_labels])
                        cond_left = f"{feat_name} in {{{lbls}}}"
                        cond_right = f"{feat_name} not in {{{lbls}}}"
                else:
                    cond_left = f"{feat_name} <= {thr:.6f}"
                    cond_right = f"{feat_name} > {thr:.6f}"

                # push right then left (so left processed first)
                stack.append((tree.children_right[node_id], conds + [cond_right], depth + 1))
                stack.append((tree.children_left[node_id], conds + [cond_left], depth + 1))

        df_rules = pd.DataFrame(rules)
        # ordenar por support_n1 descendente
        if not df_rules.empty:
            df_rules = df_rules.sort_values(by="support_n1(%)", ascending=False).reset_index(drop=True)

        if export_path:
            out = export_path if export_path.endswith(".xlsx") else (export_path.rstrip("/") + "/patterns_all_nodes.xlsx")
            df_rules.to_excel(out, index=False)
        else:
            # default filename local
            df_rules.to_excel("patterns_all_nodes.xlsx", index=False)

        return df_rules
