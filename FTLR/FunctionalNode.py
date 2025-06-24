class FunctionalNode:
    def __init__(self, condition_str=None, condition_func=None, branches=None, prediction=None, support=None):
        self.condition_str = condition_str
        self.condition_func = condition_func
        self.branches = branches or []
        self.prediction = prediction
        self.support = support

    def is_leaf(self):
        return self.prediction is not None

    def predict(self, x):
        if self.is_leaf():
            return self.prediction
        for cond_str, cond_func, child in self.branches:
            if cond_func(x):
                return child.predict(x)
        return None
    
    def extract_rules(self, path=None):
        if path is None:
            path = []

        if self.is_leaf():
            condition = " AND ".join(path) if path else "TRUE"
            return [{
                "condition": condition,
                "prediction": self.prediction,
                "support": self.support
            }]

        rules = []
        for cond_str, cond_func, child in self.branches:
            new_path = path + [cond_str]
            rules.extend(child.extract_rules(new_path))
        return rules
    """
    def extract_rules(self, path=None, min_depth=3):
        if path is None:
            path = []

        rules = []

        if self.is_leaf():
            # Extraer subreglas de longitud >= min_depth
            for i in range(min_depth, len(path)+1):
                condition = " AND ".join(path[:i]) if path[:i] else "TRUE"
                rules.append({
                    "condition": condition,
                    "prediction": self.prediction,
                    "support": self.support
                })
            return rules

        for cond_str, cond_func, child in self.branches:
            new_path = path + [cond_str]
            rules.extend(child.extract_rules(new_path, min_depth))
        return rules
    """

