import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("./"))
from FunctionalTreeClassifier import FunctionalTreeClassifier
from FunctionalRuleExtractor import FunctionalRuleExtractor
if __name__ == '__main__':
    # Dataset de ejemplo
    #df = pd.DataFrame({"temperatura": [36.5, 38.2, 39.0, 37.1, 38.8],"dolor_cabeza": [False, True, True, False, True],"diagnóstico": ["Sano", "Posible gripe", "Posible gripe", "Sano", "Posible gripe"]})
    df = pd.read_csv("./../datasets/australian.csv")


    X = df.drop(columns="target")
    y = df["target"]
    # Entrenar árbol
    clf = FunctionalTreeClassifier(max_depth=2)
    clf.fit(X, y)

    extractor = FunctionalRuleExtractor(clf)
    rules = extractor.extract_all_rules()
    for r in rules:
        print(r)

    # Exportar reglas
    extractor.export_rules_txt("patterns.txt")
    extractor.export_rules_csv("patterns.csv")
    extractor.export_rules_json("patterns.json")