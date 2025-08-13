"""
ft4cip_full.py

Implementación FT4cip (desde cero, aproximada) con:
- FT4CIPUtils: twoing, candidate splits, apply split, mfl_d_split (LDA), sequential_forward_selection
  — SFS ahora opera sobre atributos originales (nombres) y usa el mapeo a columnas transformadas.
- LogitBoostSimple: implementación aproximada de LogitBoost (Friedman-style) usando regresores lineales como base learners.
- Node, FT4CIP: árbol funcional que usa LogitBoostSimple en nodos/hojas.
- CostComplexityPruner: poda CCP optimizada por AUC.
- Ejemplo con Iris, guardando CSV temporal y ejecutando flujo entero.

Limitaciones / decisiones:
- La implementación de LogitBoost es una aproximación práctica (base learners = LinearRegression con sample_weight),
  más fiel que usar GradientBoostingClassifier directamente.
- MFLD se aproxima por LDA -> 1D -> búsqueda de umbral por percentiles (como en versiones previas).
"""

from copy import deepcopy
import numpy as np
import pandas as pd
import math
import warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# -------------------------
# Utilities
# -------------------------
class FT4CIPUtils:
    @staticmethod
    def twoing_score(y_left, y_right):
        total = len(y_left) + len(y_right)
        if total == 0:
            return 0.0
        classes = np.unique(np.concatenate([y_left, y_right]))
        p_left = np.array([np.mean(y_left == c) if len(y_left) > 0 else 0.0 for c in classes])
        p_right = np.array([np.mean(y_right == c) if len(y_right) > 0 else 0.0 for c in classes])
        diff = np.abs(p_left - p_right).sum()
        return (len(y_left) / total) * (len(y_right) / total) * (diff ** 2)

    @staticmethod
    def candidate_splits_univariate(X, feature_index, n_thresholds=10):
        col = X[:, feature_index]
        col = col[~np.isnan(col)]
        if col.size == 0:
            return []
        unique = np.unique(col)
        if unique.size <= 1:
            return []
        if unique.size <= n_thresholds:
            thresholds = unique[:-1]
        else:
            p = np.linspace(5, 95, n_thresholds)
            thresholds = np.percentile(unique, p)
            thresholds = np.unique(thresholds)
        return [("num", float(t)) for t in thresholds]

    @staticmethod
    def apply_univariate_split(X, feature_index, split):
        typ, val = split
        col = X[:, feature_index]
        if typ == "num":
            mask = np.where(np.isnan(col), False, col <= val)
        else:
            mask = col == val
        return mask

    @staticmethod
    def mfl_d_split(X, y, feature_indices, n_percentiles=25):
        if len(feature_indices) == 0:
            return None
        X_sub = X[:, feature_indices]
        X_sub = np.nan_to_num(X_sub)
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)
            proj = lda.fit_transform(X_sub, y).ravel()
        except Exception:
            return None
        percentiles = np.linspace(1, 99, n_percentiles)
        best = None
        best_score = -np.inf
        for p in percentiles:
            th = np.percentile(proj, p)
            left_idx = proj <= th
            if left_idx.sum() == 0 or (~left_idx).sum() == 0:
                continue
            score = FT4CIPUtils.twoing_score(y[left_idx], y[~left_idx])
            if score > best_score:
                best_score = score
                best = th
        if best is None:
            return None
        return {"feature_indices": list(feature_indices), "threshold": float(best), "score": float(best_score), "lda": lda, "proj": proj}

    @staticmethod
    def sequential_forward_selection_original_attrs(X, y, original_feature_names, feature_to_columns_map,
                                                   evaluate_split_fn, split_generator_fn, max_features=None):
        """
        SFS that selects ORIGINAL attributes (by name). feature_to_columns_map: dict{name -> list of transformed indices}.
        evaluate_split_fn(node_X_transformed, node_y, split_descriptor) -> numeric score
        split_generator_fn(node_X_transformed, node_y, list_of_transformed_cols) -> split_descriptor (MFLD)
        """
        if max_features is None:
            max_features = len(original_feature_names)
        candidate = set(original_feature_names)
        selected = []
        best_eval = -np.inf
        best_split = None
        best_attr = None

        # initial single-feature pass
        for attr in list(candidate):
            cols = feature_to_columns_map[attr]
            split = split_generator_fn(X, y, cols)
            if split is None:
                continue
            val = evaluate_split_fn(X, y, split)
            if val > best_eval:
                best_eval = val
                best_attr = attr
                best_split = split
        if best_attr is None:
            return None
        selected.append(best_attr)
        candidate.remove(best_attr)

        while candidate and len(selected) < max_features:
            best_candidate_eval = -np.inf
            best_candidate_attr = None
            best_candidate_split = None
            for attr in list(candidate):
                cols_try = []
                for s in selected:
                    cols_try += feature_to_columns_map[s]
                cols_try += feature_to_columns_map[attr]
                # unique sorted
                cols_try = sorted(set(cols_try))
                split = split_generator_fn(X, y, cols_try)
                if split is None:
                    continue
                val = evaluate_split_fn(X, y, split)
                if val > best_candidate_eval:
                    best_candidate_eval = val
                    best_candidate_attr = attr
                    best_candidate_split = split
            if best_candidate_eval > best_eval:
                selected.append(best_candidate_attr)
                candidate.remove(best_candidate_attr)
                best_eval = best_candidate_eval
                best_split = best_candidate_split
            else:
                break
        return best_split


# -------------------------
# LogitBoostSimple (approximate Friedman-style)
# base learners: linear regression per class (multinomial)
# -------------------------
class LogitBoostSimple:
    def __init__(self, max_iter=50, learning_rate=0.1, tol=1e-6, verbose=False):
        self.max_iter = int(max_iter)
        self.learning_rate = learning_rate
        self.tol = tol
        self.verbose = verbose
        self.K = None
        self.F = None  # n x K
        self.models = []  # list of lists? we'll store per iteration linear models per class
        self.n_iter_ = 0
        self.classes_ = None

    @staticmethod
    def _softmax(F):
        # stable softmax row-wise
        F = F - np.max(F, axis=1, keepdims=True)
        e = np.exp(F)
        s = np.sum(e, axis=1, keepdims=True)
        return e / s

    def fit(self, X, y):
        """
        X: numpy array (n, d)
        y: labels (n,)
        We implement a simplified LogitBoost:
         - Initialize F = 0
         - For m in 1..M:
             compute p = softmax(F)
             for each class k:
                 compute working response z and weights w (per Friedman eqn)
                 fit a linear regression with sample weights w predicting z
                 obtain predictions f_k (on training X)
             update F += learning_rate * ( predictions - mean(predictions) ) * (K-1)/K  (approx as in LMT)
        This is approximate and intended to emulate Weka's LogitBoost behavior.
        """
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.K = K
        Y = np.zeros((n, K))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, label in enumerate(y):
            Y[i, class_to_idx[label]] = 1.0

        F = np.zeros((n, K))
        self.models = []  # will store per-iteration a list of K linear models
        for m in range(self.max_iter):
            p = self._softmax(F)
            models_this_iter = []
            # compute pseudo-responses and weights, fit linear model per class
            preds = np.zeros((n, K))
            for k in range(K):
                # working response z = (y_k - p_k) / (p_k * (1 - p_k))  (but for multiclass p*(1-p) can be small)
                pk = p[:, k]
                yk = Y[:, k]
                # weights:
                w = pk * (1 - pk)
                # avoid zero weights
                w = np.clip(w, 1e-8, None)
                # z
                z = (yk - pk) / w
                # Fit weighted linear regression z ~ X with sample_weight = w
                lr = LinearRegression()
                try:
                    lr.fit(X, z, sample_weight=w)
                except Exception:
                    lr.fit(X, z)  # fallback
                f_k = lr.predict(X)
                preds[:, k] = f_k
                models_this_iter.append(lr)
            # normalize preds across classes as Weka does: subtract mean per-row
            pred_mean = preds.mean(axis=1, keepdims=True)
            update = (preds - pred_mean) * ((K - 1.0) / K)
            # update F
            F_new = F + self.learning_rate * update
            # check convergence by change in F
            delta = np.max(np.abs(F_new - F))
            F = F_new
            self.models.append(models_this_iter)
            self.n_iter_ = m + 1
            if self.verbose:
                print(f"[LogitBoostSimple] iter={m+1}, max delta F={delta:.6e}")
            if delta < self.tol:
                break
        self.F = F
        self._class_map = class_to_idx
        return self

    def predict_proba(self, X):
        # Predict via staged models: accumulate outputs from all iterations
        if len(self.models) == 0:
            # uniform probs
            n = X.shape[0]
            return np.ones((n, self.K)) / float(self.K)
        # compute Ftest by accumulating per-iteration predictions
        n = X.shape[0]
        Ftest = np.zeros((n, self.K))
        for models_iter in self.models:
            preds = np.zeros((n, self.K))
            for k, lr in enumerate(models_iter):
                preds[:, k] = lr.predict(X)
            pred_mean = preds.mean(axis=1, keepdims=True)
            update = (preds - pred_mean) * ((self.K - 1.0) / self.K)
            Ftest = Ftest + self.learning_rate * update
        # softmax
        probs = self._softmax(Ftest)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in idx])


# -------------------------
# Node and FT4cip tree (uses LogitBoostSimple in nodes/leaf)
# -------------------------
class Node:
    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent  # <-- aquí agregas el padre
        self.is_leaf = True
        self.model = None
        self.split = None
        self.left = None
        self.right = None
        self.n_samples = 0
        self.train_idx = None
        self.class_distribution = None

    def predict_proba_instance(self, x):
        if self.is_leaf:
            if self.model is not None:
                # modelo funcional entrenado (LogitBoost o LogisticRegression)
                # X debe tener forma (1, n_features)
                x_2d = x.reshape(1, -1)
                return self.model.predict_proba(x_2d).ravel()
            elif hasattr(self, "class_distribution"):
                # Nodo hoja sin modelo funcional, usamos distribución de clases
                return self.class_distribution
            else:
                raise RuntimeError("Leaf model missing and no class_distribution available.")
        else:
            # nodo interno: decidir a qué hijo ir según el split
            if self.split["type"] == "univariate":
                fi = self.split["feature_index"]
                s = self.split["split"]
                # Aplicar la condición del split univariado
                if FT4CIPUtils.apply_univariate_split(x.reshape(1, -1), fi, s)[0]:
                    return self.left.predict_proba_instance(x)
                else:
                    return self.right.predict_proba_instance(x)
            elif self.split["type"] == "mfl":
                mfl = self.split["mfl"]
                fi_list = mfl["feature_indices"]
                proj = mfl["lda"].transform(x[fi_list].reshape(1, -1)).ravel()
                if proj <= mfl["threshold"]:
                    return self.left.predict_proba_instance(x)
                else:
                    return self.right.predict_proba_instance(x)
            else:
                raise RuntimeError("Unknown split type in internal node.")

    def get_all_leaf_nodes_(self):
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_all_leaf_nodes())
        if self.right:
            leaves.extend(self.right.get_all_leaf_nodes())
        return leaves
    def get_all_leaf_nodes(self):
        leaves = []
        def recurse(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                if node.left is not None:
                    recurse(node.left)
                if node.right is not None:
                    recurse(node.right)
        recurse(self)
        return leaves

class FT4CIP:
    def __init__(self,
                 max_depth=10,
                 min_samples_split=10,
                 convert_nominal=True,
                 logitboost_params=None,
                 sfs_max_features=4,
                 random_state=None,
                 verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.convert_nominal = convert_nominal
        self.logitboost_params = logitboost_params or {"max_iter": 30, "learning_rate": 0.1}
        self.sfs_max_features = sfs_max_features
        self.random_state = random_state
        self.verbose = verbose

        # preprocessor and mapping original->transformed
        self.preprocessor = None
        self.orig_feature_names = None
        self.feature_to_columns = None

        self.root = None
        self._X_train_transformed = None
        self._y_train = None
        self._val_X = None
        self._val_y = None

    def _fit_preprocessor(self, X_df):
        self.orig_feature_names = X_df.columns.tolist()
        print(self.orig_feature_names)
        cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        #print(cat_cols)
        if self.convert_nominal and len(cat_cols) > 0:
            ct = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
                remainder="passthrough"
            )
            ct.fit(X_df)
            self.preprocessor = ct
            # build mapping from original attr to transformed column indices
            try:
                out_names = ct.get_feature_names_out()
            except Exception:
                # fallback generate own names
                out_names = ct.get_feature_names_out()
            # map original attribute to list of columns indices in transformed array
            fmap = {}
            for i, name in enumerate(out_names):
                # names for categorical -> format 'cat__<colname>_<value>' depending sklearn version
                # we'll find which original attribute this transformed col refers to:
                # if the original name appears as a prefix in the out_name, use that else match by token
                matched = None
                for orig in self.orig_feature_names:
                    if orig in name:
                        matched = orig
                        break
                if matched is None:
                    # as fallback, assign to last numeric attributes (passthrough)
                    # get list of passthrough columns
                    # we'll guess mapping by order: compute number of output columns and map last ones to numeric features in order
                    pass
                fmap.setdefault(matched, []).append(i)
            # for any original not in fmap (numeric passthrough), add mapping at end
            for orig in self.orig_feature_names:
                fmap.setdefault(orig, [])
            # Now fill passthrough numeric columns indices by detecting which out_names contain 'remainder' or not matched:
            # Simpler: if any fmap[orig] empty, assign remaining indices in order to those features (heuristic)
            remaining = [i for i in range(len(out_names)) if all(i not in cols for cols in fmap.values())]
            empties = [k for k, v in fmap.items() if len(v) == 0]
            for idx, orig in enumerate(empties):
                if remaining:
                    fmap[orig].append(remaining.pop(0))
            self.feature_to_columns = fmap
        else:
            self.preprocessor = None
            # identity mapping: each original feature maps to single column index
            fmap = {}
            for i, orig in enumerate(self.orig_feature_names):
                fmap[orig] = [i]
            self.feature_to_columns = fmap

    def _transform(self, X_df):
        #print(X_df.dtypes)
        if self.preprocessor is not None:
            arr = self.preprocessor.transform(X_df)
            return np.asarray(arr, dtype=float)
        else:
            return np.asarray(X_df.values, dtype=float)

    def fit(self, X_df, y, validation_fraction=0.15):

        self.n_classes_ = len(np.unique(y))
        self._fit_preprocessor(X_df)
        X = self._transform(X_df)
        #y = np.asarray(y)
        # Convertir etiquetas a enteros
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y = y_encoded
        # split validation if requested
        if validation_fraction and validation_fraction > 0:
            idx = np.arange(len(y))
            train_idx, val_idx = train_test_split(idx, test_size=validation_fraction, stratify=y, random_state=self.random_state)
            X_train = X[train_idx]; y_train = y[train_idx]
            self._val_X = X[val_idx]; self._val_y = y[val_idx]
        else:
            X_train = X; y_train = y
            self._val_X = None; self._val_y = None
        
        self._X_train_transformed = X_train
        self._y_train = y_train
        self.root = Node(depth=0)
        self._build_tree(self.root, X_train, y_train, np.arange(len(y_train)))
        # Aquí obtienes las hojas después de construir el árbol
        #leaves = self.root.get_all_leaf_nodes()
        # Ahora puedes iterar sobre las hojas para minar patrones o lo que necesites
        #for leaf in leaves:
            # Ejemplo: minar patrón y guardar
            #self.mine_patterns_and_save(leaf)
        #self.mine_patterns_and_save_all()
        

        # Extraer y guardar patrones de todas las hojas
        self.mine_patterns_and_save("patterns_all_leaves.txt")

    def _build_tree(self, node, X, y, indices):
        node.n_samples = len(indices)
        node.train_idx = indices
        # train LogitBoostSimple on node data
        Xn = X[indices]; yn = y[indices]
        # stop criteria
        if len(np.unique(yn)) == 1 or len(indices) < self.min_samples_split or node.depth >= self.max_depth:
            node.is_leaf = True
            # Si sólo hay una clase, asigna distribución de clases y retorna
            if len(np.unique(yn)) < 2:
                node.class_distribution = np.bincount(yn, minlength=self.n_classes_) / len(yn)
                node.model = None
                return
            # Si hay más de una clase, entrena regresión logística
            lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
            lr.fit(Xn, yn)
            node.model = lr
            return
        # build LogitBoost for node
        lb = LogitBoostSimple(max_iter=self.logitboost_params.get("max_iter", 30),
                            learning_rate=self.logitboost_params.get("learning_rate", 0.1),
                            tol=self.logitboost_params.get("tol", 1e-6),
                            verbose=False)
        try:
            lb.fit(Xn, yn)
            node.model = lb
            node.is_leaf = True  # default leaf unless we split below
        except Exception:
            # fallback logistic
            lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
            lr.fit(Xn, yn)
            node.model = lr
            node.is_leaf = True

        # generate univariate splits
        best_split = None
        best_eval = -np.inf
        n_features = X.shape[1]
        for fi in range(n_features):
            cand = FT4CIPUtils.candidate_splits_univariate(X, fi, n_thresholds=8)
            for s in cand:
                mask_full = FT4CIPUtils.apply_univariate_split(X, fi, s)
                mask = mask_full[indices]
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue
                left_idx = indices[mask]
                right_idx = indices[~mask]
                score = FT4CIPUtils.twoing_score(y[left_idx], y[right_idx])
                if score > best_eval:
                    best_eval = score
                    best_split = {"type": "univariate", "feature_index": fi, "split": s, "score": score}

        # try multivariate MFLD via SFS over original attributes
        if best_split is not None:
            try:
                mfl = FT4CIPUtils.sequential_forward_selection_original_attrs(
                    X[indices], y[indices],
                    self.orig_feature_names,
                    self.feature_to_columns,
                    evaluate_split_fn=lambda Xn, yn, split: FT4CIPUtils.twoing_score(
                        yn[split_mask := (np.nan_to_num(split["proj"]) <= split["threshold"])],
                        yn[~split_mask]) if isinstance(split, dict) and "proj" in split else split.get("score", -np.inf),
                    split_generator_fn=lambda Xn, yn, cols: FT4CIPUtils.mfl_d_split(Xn, yn, cols),
                    max_features=min(self.sfs_max_features, len(self.orig_feature_names)))
            except Exception:
                mfl = None
            # mfl is a dict with keys including 'score'
            if mfl is not None and isinstance(mfl, dict) and mfl.get("score", -np.inf) > best_eval:
                node.is_leaf = False
                node.split = {"type": "mfl", "mfl": mfl}
                fi_list = mfl["feature_indices"]
                proj = mfl["lda"].transform(np.nan_to_num(X[indices][:, fi_list])).ravel()
                mask_left = proj <= mfl["threshold"]
                left_idx = indices[mask_left]
                right_idx = indices[~mask_left]
                node.left = Node(depth=node.depth + 1, parent=node)
                node.right = Node(depth=node.depth + 1, parent=node)
                self._build_tree(node.left, X, y, left_idx)
                self._build_tree(node.right, X, y, right_idx)
                return

        if best_split is not None:
            node.is_leaf = False
            node.split = best_split
            fi = best_split["feature_index"]; s = best_split["split"]
            mask_full = FT4CIPUtils.apply_univariate_split(X, fi, s)
            mask = mask_full[indices]
            left_idx = indices[mask]; right_idx = indices[~mask]
            if left_idx.size == 0 or right_idx.size == 0:
                node.is_leaf = True
                # Entrena modelo en nodo hoja antes de retornar
                if not hasattr(node, 'model') or node.model is None:
                    Xn_leaf = X[left_idx if left_idx.size > 0 else right_idx]
                    yn_leaf = y[left_idx if left_idx.size > 0 else right_idx]
                    if len(np.unique(yn_leaf)) > 1:
                        lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
                        lr.fit(Xn_leaf, yn_leaf)
                        node.model = lr
                    else:
                        node.class_distribution = np.bincount(yn_leaf, minlength=self.n_classes_) / len(yn_leaf)
                        node.model = None
                return
            node.left = Node(depth=node.depth + 1, parent=node)
            node.right = Node(depth=node.depth + 1, parent=node)
            self._build_tree(node.left, X, y, left_idx)
            self._build_tree(node.right, X, y, right_idx)
            return

        # Si no hay split válido, asignar modelo funcional al nodo hoja
        node.is_leaf = True
        if not hasattr(node, 'model') or node.model is None:
            if len(np.unique(yn)) > 1:
                lr = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
                lr.fit(Xn, yn)
                node.model = lr
            else:
                node.class_distribution = np.bincount(yn, minlength=self.n_classes_) / len(yn)
                node.model = None
        return

    def predict_proba(self, X_df_or_np):
        if isinstance(X_df_or_np, pd.DataFrame):
            X = self._transform(X_df_or_np)
        else:
            X = np.asarray(X_df_or_np, dtype=float)
        probs = np.array([self.root.predict_proba_instance(x) for x in X])
        return probs

    def predict(self, X_df_or_np):
        probs = self.predict_proba(X_df_or_np)
        if probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)
        else:
            return probs.argmax(axis=1)

    def extract_patterns_from_leaf(self, node):
        path_conditions = []
        current = node

        if self.preprocessor is not None:
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"Feature_{i}" for i in range(len(self.orig_feature_names))]
        else:
            feature_names = self.orig_feature_names

        while current.parent is not None:
            split = current.parent.split
            if split is None:
                break

            split_type = split.get("type", "univariate")

            if split_type == "univariate":
                feature_idx = split.get("feature_index", None)
                split_val = split.get("split", None)
                if feature_idx is None or split_val is None:
                    break

                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"

                if current == current.parent.left:
                    cond = f"{feature_name} <= {split_val:.4f}"
                else:
                    cond = f"{feature_name} > {split_val:.4f}"

                path_conditions.append(cond)

            elif split_type == "mfl":
                mfl = split.get("mfl", None)
                if mfl is None:
                    break

                feature_indices = mfl.get("feature_indices", [])
                lda_coefs = mfl.get("lda", [])  # pesos de variables
                threshold = mfl.get("threshold", None)
                if threshold is None or len(feature_indices) != len(lda_coefs):
                    break

                terms = []
                for fi, coef in zip(feature_indices, lda_coefs):
                    fname = feature_names[fi] if fi < len(feature_names) else f"Feature_{fi}"
                    terms.append(f"({coef:.3f} * {fname})")

                expr = " + ".join(terms)
                if current == current.parent.left:
                    cond = f"{expr} <= {threshold:.4f}"
                else:
                    cond = f"{expr} > {threshold:.4f}"

                path_conditions.append(cond)

            else:
                break

            current = current.parent

        path_conditions.reverse()
        rule = " AND ".join(path_conditions)
        return rule

    def mine_patterns_and_save(self, filename="patterns_all_leaves.txt"):
        leaves = self.root.get_all_leaf_nodes_()
        
        with open(filename, "w") as f:
            for i, leaf in enumerate(leaves):
                #print(leaf)
                pattern = self.extract_patterns_from_leaf(leaf)
                #f.write(f"Leaf {i+1} (depth={leaf.depth}, samples={leaf.n_samples}):\n")
                #f.write(pattern + "\n\n")

        print(f"Patterns from {len(leaves)} leaves saved in '{filename}'.")
    
    
    # -------------------------
    # Cost Complexity Pruning optimized for AUC
    # -------------------------
    def _gather_internal_nodes(self, node, nodes=None):
        nodes = [] if nodes is None else nodes
        if node is None:
            return nodes
        if not node.is_leaf:
            nodes.append(node)
            self._gather_internal_nodes(node.left, nodes)
            self._gather_internal_nodes(node.right, nodes)
        return nodes

    def _compute_alpha_for_node(self, node):
        if node.train_idx is None:
            return None
        # error leaf
        idxs = node.train_idx
        Xn = self._X_train_transformed[idxs]; yn = self._y_train[idxs]
        try:
            probs_leaf = node.model.predict_proba(Xn)
            preds_leaf = probs_leaf.argmax(axis=1)
        except Exception:
            # fallback majority
            vals, counts = np.unique(yn, return_counts=True)
            maj = vals[np.argmax(counts)]
            preds_leaf = np.full_like(yn, maj)
        err_leaf = np.mean(preds_leaf != yn) * len(idxs)

        def subtree_error_and_leaves(n):
            if n.is_leaf:
                idxs_n = n.train_idx
                Xn2 = self._X_train_transformed[idxs_n]
                yn2 = self._y_train[idxs_n]
                if n.model is not None:
                    probs = n.model.predict_proba(Xn2)
                elif hasattr(n, "class_distribution"):
                    # Crear probabilidades constantes para todas las instancias en el nodo
                    probs = np.tile(n.class_distribution, (len(yn2), 1))
                else:
                    raise RuntimeError("Nodo hoja sin modelo ni distribución de clases")
                preds2 = probs.argmax(axis=1)
                return (np.mean(preds2 != yn2) * len(idxs_n), 1)
            else:
                e1, l1 = subtree_error_and_leaves(n.left)
                e2, l2 = subtree_error_and_leaves(n.right)
                return (e1 + e2, l1 + l2)
        err_subtree, leaves = subtree_error_and_leaves(node)
        if leaves <= 1:
            return None
        alpha = (err_leaf - err_subtree) / (leaves - 1)
        return float(alpha)

    def prune_ccp_optimize_auc(self):
        if self._val_X is None or self._val_y is None:
            raise RuntimeError("Fit with validation_fraction>0 to enable pruning.")
        internal = self._gather_internal_nodes(self.root)
        alpha_values = []
        for n in internal:
            a = self._compute_alpha_for_node(n)
            if a is not None and not math.isinf(a) and not math.isnan(a):
                alpha_values.append(a)
        alpha_values = sorted(set([0.0] + alpha_values))
        best_auc = -np.inf
        best_tree = deepcopy(self.root)
        for alpha in alpha_values:
            tree_copy = deepcopy(self.root)
            changed = True
            while changed:
                changed = False
                internals = self._gather_internal_nodes(tree_copy)
                node_alpha_pairs = []
                for n in internals:
                    a = self._compute_alpha_for_node(n)
                    if a is not None:
                        node_alpha_pairs.append((a, n))
                node_alpha_pairs = sorted(node_alpha_pairs, key=lambda x: x[0])
                if node_alpha_pairs and node_alpha_pairs[0][0] <= alpha:
                    node_to_prune = node_alpha_pairs[0][1]
                    node_to_prune.left = None
                    node_to_prune.right = None
                    node_to_prune.split = None
                    node_to_prune.is_leaf = True
                    changed = True
            probs = np.array([tree_copy.predict_proba_instance(x) for x in self._val_X])
            auc_val = self._auc_from_probs(probs, self._val_y)
            if auc_val > best_auc:
                best_auc = auc_val
                best_tree = tree_copy
        self.root = best_tree
        return best_auc

    @staticmethod
    def _auc_from_probs(probs, y):
        if probs.shape[1] == 2:
            return roc_auc_score(y, probs[:, 1])
        else:
            yb = label_binarize(y, classes=np.arange(probs.shape[1]))
            return roc_auc_score(yb, probs, average="macro")

# -------------------------
# Example / tests using Iris
# -------------------------
def run_example_with_iris():
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    # Filtramos solo 2 clases: setosa (0) y versicolor (1)
    mask = y != 2
    X = X[mask]
    y = y[mask]

    # Crear DataFrame para guardarlo como CSV
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    csv_path = "iris_binary.csv"
    df.to_csv(csv_path, index=False)
    print("Saved binary Iris CSV to", csv_path)

    # Train/Test split
    X_df = df.drop(columns=["target"])
    y = df["target"].values
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )

    # Ejemplo con tu FT4CIP (ajusta los parámetros si lo necesitas)
    clf = FT4CIP(
        max_depth=4,
        min_samples_split=5,
        convert_nominal=False,
        logitboost_params={"max_iter": 30, "learning_rate": 0.1},
        sfs_max_features=3,
        random_state=42,
        verbose=True
    )
    clf.fit(X_train_df, y_train, validation_fraction=0.15)

    probs_before = clf.predict_proba(X_test_df)
    auc_before = clf._auc_from_probs(probs_before, y_test)
    print("AUC before pruning:", auc_before)

    auc_val = clf.prune_ccp_optimize_auc()
    print("Best validation AUC during CCP:", auc_val)

    probs_after = clf.predict_proba(X_test_df)
    auc_after = clf._auc_from_probs(probs_after, y_test)
    print("AUC after pruning:", auc_after)

    print("Done example.")

def run_credit_risk():
    dataset = pd.read_csv("./../../datasets/australian.csv")

    X = dataset.drop(columns="target")
    y = dataset["target"]

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Ejemplo con tu FT4CIP (ajusta los parámetros si lo necesitas)
    clf = FT4CIP(
        max_depth=4,
        min_samples_split=5,
        convert_nominal=True,
        logitboost_params={"max_iter": 30, "learning_rate": 0.1},
        sfs_max_features=3,
        random_state=42,
        verbose=True
    )
    clf.fit(X_train_df, y_train, validation_fraction=0.15)

    probs_before = clf.predict_proba(X_test_df)
    auc_before = clf._auc_from_probs(probs_before, y_test)
    print("AUC before pruning:", auc_before)

    auc_val = clf.prune_ccp_optimize_auc()
    print("Best validation AUC during CCP:", auc_val)

    probs_after = clf.predict_proba(X_test_df)
    auc_after = clf._auc_from_probs(probs_after, y_test)
    print("AUC after pruning:", auc_after)

    

if __name__ == "__main__":
    #run_example_with_iris()
    run_credit_risk()
