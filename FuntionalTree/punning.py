# pruning.py
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

class CostComplexityPruner:
    """
    Poda iterativa orientada a optimizar AUC sobre un conjunto de validación.

    Requiere que tu árbol tenga nodos con la siguiente interfaz (como en la implementación
    FT4cip que entregué antes):
      - node.is_leaf (bool)
      - node.left, node.right (referencias a nodos o None)
      - node.split (descriptor del split)  # opcional
      - node.model (modelo entrenado para la hoja; puede usarse para predecir)
      - node.samples_idx (indices de entrenamiento en ese nodo)  # opcional pero útil
      - node.predict_proba_instance(x): método que devuelve probas para una sola instancia
    Alternativamente, puedes pasar predict_proba_fn(X) que reciba X_val y devuelva array(n, n_classes).
    """

    def __init__(self, min_improvement: float = 1e-4, multiclass_average: str = "macro", verbose: bool = False):
        """
        Args:
            min_improvement: mejora mínima en AUC para aceptar una poda (>=). 
                             Si lo pones 0 acepta igualar AUC; si negativo permite pequeñas pérdidas.
            multiclass_average: estrategia para AUC multiclass ('macro' or 'ovo' etc. for sklearn).
            verbose: imprime info de progreso.
        """
        self.min_improvement = min_improvement
        self.multiclass_average = multiclass_average
        self.verbose = verbose

    # -----------------------
    # Utilidades internas
    # -----------------------
    def _collect_internal_nodes(self, node, nodes=None):
        """Collect list of internal (non-leaf) nodes in the subtree rooted at node."""
        if nodes is None:
            nodes = []
        if node is None:
            return nodes
        if not node.is_leaf:
            nodes.append(node)
            # Recurse to children (if not None)
            if node.left is not None:
                self._collect_internal_nodes(node.left, nodes)
            if node.right is not None:
                self._collect_internal_nodes(node.right, nodes)
        return nodes

    def _predict_proba_with_callable(self, predict_proba_fn, X_val):
        """Use provided function to get probas."""
        return predict_proba_fn(X_val)

    def _predict_proba_via_node(self, root, X_val):
        """Fallback: predict by calling root.predict_proba_instance for each row."""
        probs = []
        for i in range(X_val.shape[0]):
            probs.append(root.predict_proba_instance(X_val[i]))
        return np.vstack(probs)

    def _auc_score(self, probs, y_val):
        """Compute AUC: binary or multiclass (macro)."""
        if probs.ndim == 1 or probs.shape[1] == 1:
            # fallback
            return roc_auc_score(y_val, probs)
        if probs.shape[1] == 2:
            # binary: use probability for positive class
            return roc_auc_score(y_val, probs[:, 1])
        else:
            # multiclass: label-binarize then compute macro AUC
            classes = np.arange(probs.shape[1])
            y_bin = label_binarize(y_val, classes=classes)
            return roc_auc_score(y_bin, probs, average=self.multiclass_average)

    # -----------------------
    # Método principal: prune
    # -----------------------
    def prune(self, root, X_val, y_val, predict_proba_fn=None):
        """
        Ejecuta poda iterativa sobre el árbol dado para optimizar AUC en (X_val,y_val).

        Args:
            root: nodo raíz del árbol (objeto Node).
            X_val: numpy array (n_samples, n_features) transformado exactamente como espera el árbol.
            y_val: array-like (n_samples,)
            predict_proba_fn: optional callable(X) -> probas; si None usa root.predict_proba_instance.
        Returns:
            history: list of tuples [(auc_before, best_delta, pruned_node_or_None), ...]
        """
        if predict_proba_fn is None:
            predict_proba_fn = lambda X: self._predict_proba_via_node(root, X)

        history = []
        improved_any = True
        iteration = 0

        # compute baseline
        probs_base = self._predict_proba_with_callable(predict_proba_fn, X_val)
        auc_base = self._auc_score(probs_base, y_val)
        if self.verbose:
            print(f"[Pruner] Iter {iteration}: base AUC = {auc_base:.6f}")

        while True:
            iteration += 1
            internal_nodes = self._collect_internal_nodes(root, [])
            if not internal_nodes:
                if self.verbose:
                    print("[Pruner] No more internal nodes to consider.")
                break

            best_delta = -np.inf
            best_node = None
            best_auc = auc_base

            # Try pruning each internal node and evaluate AUC
            for node in internal_nodes:
                # save state to revert later
                saved = {
                    "left": node.left,
                    "right": node.right,
                    "split": node.split,
                    "is_leaf": node.is_leaf,
                    "model": node.model
                }

                # perform tentative prune: make node a leaf (remove children & split)
                node.left = None
                node.right = None
                node.split = None
                node.is_leaf = True
                # note: we keep node.model as-is (assume it was fit for node training samples)

                # evaluate AUC
                probs_try = self._predict_proba_with_callable(predict_proba_fn, X_val)
                try:
                    auc_try = self._auc_score(probs_try, y_val)
                except ValueError:
                    # if AUC can't be computed, revert and skip
                    auc_try = -np.inf

                delta = auc_try - auc_base

                # revert node for now
                node.left = saved["left"]
                node.right = saved["right"]
                node.split = saved["split"]
                node.is_leaf = saved["is_leaf"]
                node.model = saved["model"]

                # track best candidate
                if self.verbose:
                    print(f"[Pruner] Tried prune node id={id(node)} => auc_try={auc_try:.6f}, delta={delta:.6f}")
                if delta > best_delta:
                    best_delta = delta
                    best_node = node
                    best_auc = auc_try

            # decide if accept best candidate
            if best_node is None:
                if self.verbose:
                    print("[Pruner] No candidate found.")
                break

            # Accept prune if improvement >= min_improvement
            if best_delta >= self.min_improvement:
                # apply prune permanently
                if self.verbose:
                    print(f"[Pruner] Accepting prune on node id={id(best_node)} with delta={best_delta:.6f} (auc {best_auc:.6f})")
                # drop children and split
                best_node.left = None
                best_node.right = None
                best_node.split = None
                best_node.is_leaf = True
                # update baseline
                auc_base = best_auc
                history.append((auc_base, best_delta, best_node))
                # continue to next iteration
            else:
                if self.verbose:
                    print(f"[Pruner] No prune accepted (best delta {best_delta:.6f} < min_improvement {self.min_improvement}). Stopping.")
                break

        if self.verbose:
            print(f"[Pruner] Finished after {iteration} iterations. Final AUC = {auc_base:.6f}")
        return history
