# -------------------------
# Node class
# -------------------------
class Node:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = True
        self.model = None  # LogisticModel instance
        self.split = None  # dict describing split {type:'univariate'/'mfl', ...}
        self.left = None
        self.right = None
        self.samples_idx = None  # indices of training samples at node
        self.prediction_cache = None

    def predict_proba_instance(self, x, feature_names=None):
        # given raw features array x (1d), route through tree and return probabilities
        if self.is_leaf:
            if self.model is None:
                raise RuntimeError("Leaf model missing")
            return self.model.predict_proba(x.reshape(1, -1))[0]
        else:
            # evaluate split
            s = self.split
            if s["type"] == "univariate":
                feat = s["feature_index"]
                typ = s["split"][0]
                val = s["split"][1]
                if typ == "num":
                    go_left = (x[feat] <= val) if not np.isnan(x[feat]) else False
                else:
                    go_left = (x[feat] == val)
            else:  # mfl
                # projection with lda
                fi = s["mfl"]["feature_indices"]
                lda = s["mfl"]["lda"]
                proj = lda.transform(x[fi].reshape(1, -1))[0,0]
                go_left = (proj <= s["mfl"]["threshold"])
            if go_left:
                return self.left.predict_proba_instance(x)
            else:
                return self.right.predict_proba_instance(x)

