import pandas as pd
import numpy as np
import json
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import re
from typing import List, Dict


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

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
        }

class FunctionalRuleExtractor:
    def __init__(self, tree):
        self.tree = tree
        self.rules = None

    def extract_all_rules(self):
        if self.rules is not None:
            return self.rules
        self.rules = self.tree.tree.extract_rules()
        return self.rules

    def export_rules_txt(self, path: str):
        rules = self.extract_all_rules()
        with open(path, "w") as f:
            for rule in rules:
                f.write(f"IF {rule['condition']} THEN prediction = {rule['prediction']} (soporte: {rule['support']})\n")

    def export_rules_csv(self, path: str):
        rules = self.extract_all_rules()
        df = pd.DataFrame(rules)
        df.to_csv(path, index=False)

    def export_rules_json(self, path: str):
        rules = self.extract_all_rules()
        with open(path, "w") as f:
            json.dump(rules, f, indent=2)

    def _sanitize_condition(self, condition: str) -> str:
        condition = condition.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        tokens = condition.split()
        sanitized = []
        for i, token in enumerate(tokens):
            if i > 0 and tokens[i - 1] in ["==", "!=", "<=", ">=", "<", ">"]:
                if not re.match(r'^-?\d+(\.\d+)?$', token):
                    token = f'"{token}"'
            sanitized.append(token)
        return " ".join(sanitized)

    def predict_from_rules(self, X: pd.DataFrame, json_path: str, default_value: str = "Desconocido") -> pd.Series:
        with open(json_path, "r") as f:
            rules = json.load(f)

        predictions = []
        for idx, row in X.iterrows():
            predicted = False
            for rule in rules:
                condition = rule["condition"]
                try:
                    sanitized_condition = self._sanitize_condition(condition)
                    local_env = {col: row[col] for col in row.index}
                    if eval(sanitized_condition, {}, local_env):
                        predictions.append(rule["prediction"])
                        predicted = True
                        break
                except Exception as e:
                    print(f"\nError evaluando la condición: {condition}\nFila:\n{row.to_dict()}\nError: {e}")
                    break

            if not predicted:
                predictions.append(default_value)

        return pd.Series(predictions, index=X.index)

    def rules_covering_each_instance(self, X):
        if not hasattr(self, 'rules') or self.rules is None:
            raise ValueError("Primero debes extraer las reglas con extract_all_rules()")

        results = []
        for idx, row in X.iterrows():
            matched_rules = []
            for i, rule in enumerate(self.rules):
                condition = rule["condition"]
                try:
                    sanitized_condition = self._sanitize_condition(condition)
                    local_env = {col: row[col] for col in X.columns}
                    if eval(sanitized_condition, {}, local_env):
                        matched_rules.append(i)
                except Exception as e:
                    print(f"Error evaluando condición: {condition} en fila {idx}: {e}")

            results.append({
                "count": len(matched_rules),
                "rules": matched_rules,
            })

        return results

    def rules_covered_by_instance(self, X):
        if self.rules is None:
            raise ValueError("Primero debes extraer las reglas con extract_all_rules()")

        coverage_dict = {}
        for idx, row in X.iterrows():
            matched_rules = []
            for i, rule in enumerate(self.rules):
                condition = rule["condition"]
                try:
                    sanitized_condition = self._sanitize_condition(condition)
                    local_env = {col: row[col] for col in X.columns}
                    if eval(sanitized_condition, {}, local_env):
                        matched_rules.append(f"{i}: IF {rule['condition']} THEN prediction = {rule['prediction']}")
                except Exception as e:
                    print(f"Error evaluando condición: {condition} en fila {idx}: {e}")
            coverage_dict[idx] = matched_rules

        return coverage_dict

    def summarize_rule_stats(self, X: pd.DataFrame, y: pd.Series):
        if self.rules is None:
            raise ValueError("Primero debes extraer las reglas con extract_all_rules()")

        class_labels = sorted(y.unique())
        total_by_class = {label: (y == label).sum() for label in class_labels}

        for rule in self.rules:
            class_counts = {label: 0 for label in class_labels}
            condition = rule["condition"]
            for idx, row in X.iterrows():
                try:
                    sanitized = self._sanitize_condition(condition)
                    local_env = {col: row[col] for col in X.columns}
                    if eval(sanitized, {}, local_env):
                        label = y.loc[idx]
                        class_counts[label] += 1
                except Exception as e:
                    print(f"Error evaluando condición: {condition} en fila {idx}: {e}")
            class_supports = {label: round(class_counts[label] / total_by_class[label], 4)
                              if total_by_class[label] > 0 else 0
                              for label in class_labels}
            rule["class_counts"] = class_counts
            rule["class_supports"] = class_supports

    def export_rules_with_stats_json(self, path: str):
        if self.rules is None:
            raise ValueError("Primero ejecuta summarize_rule_stats(...)")
        with open(path, "w") as f:
            json.dump(self.rules, f, indent=2)

    def export_rules_with_stats_txt(self, path: str):
        if self.rules is None:
            raise ValueError("Primero ejecuta summarize_rule_stats(...)")
        with open(path, "w") as f:
            for rule in self.rules:
                line = f"IF {rule['condition']} THEN prediction = {rule['prediction']} (soporte total: {rule.get('support', '?')})\n"
                for label, count in rule.get("class_counts", {}).items():
                    support = rule["class_supports"].get(label, 0)
                    line += f"  - Clase {label}: {count} instancias ({support * 100:.2f}% del total de clase)\n"
                f.write(line + "\n")

    def export_rules_with_stats_csv(self, path: str):
        if self.rules is None:
            raise ValueError("Primero ejecuta summarize_rule_stats(...)")

        data = []
        for rule in self.rules:
            row = {
                "condition": rule["condition"],
                "prediction": rule["prediction"],
                "support": rule.get("support", 0)
            }
            for label, count in rule.get("class_counts", {}).items():
                row[f"{label}_count"] = count
                row[f"{label}_support"] = rule["class_supports"].get(label, 0)
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def rules_covering_test_instance(self, X_test: pd.DataFrame, instance_idx: int, rules_path: str = None):
        """
        Devuelve las reglas que cubren una instancia específica del conjunto de test.
        Si se pasa rules_path (json), las lee desde ahí. Si no, usa self.rules.
        """
        if rules_path:
            with open(rules_path, "r") as f:
                rules = json.load(f)
        else:
            if self.rules is None:
                raise ValueError("Primero debes extraer las reglas con extract_all_rules() o pasar un path json.")
            rules = self.rules

        row = X_test.iloc[instance_idx]
        matched_rules = []

        for i, rule in enumerate(rules):
            condition = rule["condition"]
            try:
                sanitized = self._sanitize_condition(condition)
                local_env = {col: row[col] for col in X_test.columns}
                if eval(sanitized, {}, local_env):
                    matched_rules.append({
                        "index": i,
                        "condition": rule["condition"],
                        "prediction": rule["prediction"],
                        "support": rule.get("support", "?"),
                        "class_counts": rule.get("class_counts", {}),
                        "class_supports": rule.get("class_supports", {})
                    })
            except Exception as e:
                print(f"Error evaluando condición: {condition} en fila {instance_idx}: {e}")

        return matched_rules

    def export_rules_for_instance(self, result, instance_idx, path_csv, path_json=None, path_txt=None):
        """
        Exporta las reglas que cubren una instancia específica con el formato:
        instancia, Pattern, negative Count, negative Support, positive Count, positive Support

        Parameters:
        - result: lista de dicts con las reglas que cubren la instancia y sus métricas.
        - instance_idx: índice de la instancia (int).
        - path_csv: ruta para exportar CSV.
        - path_json: ruta opcional para exportar JSON.
        - path_txt: ruta opcional para exportar TXT.
        """
        data = []
        for r in result:
            class_counts = r.get("class_counts", {})
            class_supports = r.get("class_supports", {})
            data.append({
                "instancia": instance_idx,
                "Pattern": r.get("condition", ""),
                "negative Count": class_counts.get("negative", 0),
                "negative Support": class_supports.get("negative", 0.0),
                "positive Count": class_counts.get("positive", 0),
                "positive Support": class_supports.get("positive", 0.0),
            })
        df = pd.DataFrame(data)
        df.to_csv(path_csv, index=False)

        if path_json:
            with open(path_json, "w") as f:
                json.dump(data, f, indent=2)

        if path_txt:
            with open(path_txt, "w") as f:
                for row in data:
                    f.write(f"instancia: {row['instancia']}, Pattern: {row['Pattern']}, "
                            f"negative Count: {row['negative Count']}, negative Support: {row['negative Support']}, "
                            f"positive Count: {row['positive Count']}, positive Support: {row['positive Support']}\n")


# ========== EJEMPLO DE USO ==========

if __name__ == "__main__":
    df = pd.read_csv("./../datasets/australian.csv")


    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
    # Entrenar árbol
    clf = FunctionalTreeClassifier(max_depth=2)
    clf.fit(X_train, y_train)


    metrics = clf.evaluate(X_test, y_test)
    print(metrics)

    extractor = FunctionalRuleExtractor(clf)
    rules = extractor.extract_all_rules()

    #print("Ejemplo de regla:", rules[0]) 
    #for r in rules:
        #print(r)

    # Exportar reglas
    extractor.export_rules_txt("reglas.txt")
    extractor.export_rules_csv("reglas.csv")
    extractor.export_rules_json("reglas.json")

    predictions = extractor.predict_from_rules(X_test, "reglas.json")

    print(classification_report(y_test, predictions))
    print(X_test.iloc[[10]])
    print(predictions)

    extractor.extract_all_rules()
    extractor.summarize_rule_stats(X_train, y_train)

    extractor.export_rules_with_stats_txt("reglas_con_estadisticas.txt")
    extractor.export_rules_with_stats_csv("reglas_con_estadisticas.csv")
    extractor.export_rules_with_stats_json("reglas_con_estadisticas.json")

    result = extractor.rules_covering_test_instance(X_test, instance_idx=3, rules_path="reglas_con_estadisticas.json")
    extractor.export_rules_for_instance(result, instance_idx=3,
                                    path_csv="reglas_instancia3.csv",
                                    path_json="reglas_instancia3.json",
                                    path_txt="reglas_instancia3.txt")

    """
        for i in range(len(X_test)):
        reglas = extractor.rules_covering_test_instance(X_test, i, rules_path="reglas.json")
        print(f"Instancia {i} cumple reglas:")
        for r in reglas:
            print(f" - Regla {r['index']}: IF {r['condition']} THEN {r['prediction']}")
    """

    
