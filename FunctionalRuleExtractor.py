import pandas as pd
import json
class FunctionalRuleExtractor:
    def __init__(self, tree):
        self.tree = tree

    def extract_all_rules(self):
        rules, _ = self.tree.tree.extract_rules()
        return rules

    def extract_rules_by_class(self):
        _, details = self.tree.tree.extract_rules()
        class_rules = {}
        for condition, pred_class, support in details:
            rule_str = f"IF {condition} THEN prediction = {pred_class} (soporte: {support})"
            class_rules.setdefault(pred_class, []).append(rule_str)
        return class_rules

    def export_rules_txt(self, filename):
        rules = self.extract_all_rules()
        with open(filename, "w") as f:
            for rule in rules:
                f.write(rule + "\n")

    def export_rules_csv(self, filename):
        _, details = self.tree.tree.extract_rules()
        df = pd.DataFrame(details, columns=["condition", "prediction", "support"])
        df.to_csv(filename, index=False)

    def export_rules_json(self, filename):
        _, details = self.tree.tree.extract_rules()
        rules_json = [
            {"condition": cond, "prediction": pred, "support": supp}
            for cond, pred, supp in details
        ]
        with open(filename, "w") as f:
            json.dump(rules_json, f, indent=2)

