from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
# -------------------------
# LogisticModel: wrapper para modelos en hojas
# -------------------------
class LogisticModel:
    def __init__(self, use_boosting=False, boosting_params=None, random_state=None):
        """
        use_boosting: si True usa GradientBoostingClassifier (aprox LogitBoost).
        boosting_params: dict de parámetros para GradientBoostingClassifier.
        """
        self.use_boosting = use_boosting
        self.random_state = random_state
        if boosting_params is None:
            boosting_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
        self.boosting_params = boosting_params

        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        if self.use_boosting:
            clf = GradientBoostingClassifier(**self.boosting_params)
            clf.fit(X, y)
            self.model = clf
            self.classes_ = clf.classes_
        else:
            clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
            clf.fit(X, y)
            self.model = clf
            self.classes_ = clf.classes_

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def score_auc(self, X, y):
        probs = self.predict_proba(X)
        # multiclass AUC averaged via one-vs-rest (if binary it's just AUC)
        if probs.shape[1] == 2:
            return roc_auc_score(y, probs[:,1])
        else:
            # one-vs-rest average
            y_bin = label_binarize(y, classes=self.classes_)
            return roc_auc_score(y_bin, probs, average="macro")