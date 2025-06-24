import numpy as np
class FunctionalNode:
    def __init__(self, condition_str=None, condition_func=None, branches=None, prediction=None, support=None):
        self.condition_str = condition_str
        self.condition_func = condition_func
        self.branches = branches or []  # List of (condition_str, condition_func, child_node)
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

    def extract_rules(self, path=None, support_count=None):
        if path is None:
            path = []
        if support_count is None:
            support_count = {'count': 0}

        if self.is_leaf():
            condition = " AND ".join(path) if path else "TRUE"
            rule = f"IF {condition} THEN prediction = {self.prediction} (soporte: {self.support})"
            return [rule], [(condition, self.prediction, self.support)]

        rules = []
        details = []
        for cond_str, cond_func, child in self.branches:
            new_path = path + [cond_str]
            child_rules, child_details = child.extract_rules(new_path)
            rules.extend(child_rules)
            details.extend(child_details)
        return rules, details

