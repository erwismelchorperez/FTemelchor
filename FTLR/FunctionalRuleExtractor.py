import json
import re
import csv
class FunctionalRuleExtractor:
    def __init__(self, tree_model):
        self.tree_model = tree_model
        self.rules = []
    def extract_all_rules(self, X, y):
        # Extraer reglas simples
        rules = self.tree_model.tree.extract_rules()
        # Calcular estadísticas adicionales por regla
        enhanced_rules = []
        for rule in rules:
            cond = rule['condition']
            prediction = rule['prediction']
            support = rule.get('support', None)

            # Evaluar la regla sobre X para obtener indices que cumplen condición
            mask = X.apply(lambda row: self.safe_eval_condition(cond, row.to_dict()), axis=1)

            # Conteos y soportes
            y_rule = y[mask]
            positive_count = sum((y_rule == prediction))
            negative_count = sum((y_rule != prediction))
            positive_support = positive_count / len(y) if len(y) > 0 else 0
            negative_support = negative_count / len(y) if len(y) > 0 else 0

            new_rule = {
                "condition": cond,
                "prediction": prediction,
                "support": support,
                "positive_count": positive_count,
                "positive_support": positive_support,
                "negative_count": negative_count,
                "negative_support": negative_support
            }
            enhanced_rules.append(new_rule)

        self.rules = enhanced_rules
        return self.rules
    def _add_quotes_to_categoricals(self, condition_str, X):
        categorical_features = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']
    
        def quote_match(m):
            val = m.group(2).strip('"\'' )
            return f"{m.group(1)}'{val}'"

        for feature in categorical_features:
            pattern_eq = re.compile(rf"({feature}\s*==\s*)([^\s()]+)")
            condition_str = pattern_eq.sub(quote_match, condition_str)

            pattern_neq = re.compile(rf"({feature}\s*!=\s*)([^\s()]+)")
            condition_str = pattern_neq.sub(quote_match, condition_str)
        return condition_str
    def export_rules_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for rule in self.rules:
                f.write(f"IF {rule['condition']} THEN prediction = {rule['prediction']} "
                        f"(support={rule['support']}, neg_count={rule['negative_count']}, neg_supp={rule['negative_support']:.4f}, "
                        f"pos_count={rule['positive_count']}, pos_supp={rule['positive_support']:.4f})\n")
    def export_rules_csv(self, filepath):
        keys = ['condition', 'prediction', 'support', 'negative_count', 'negative_support', 'positive_count', 'positive_support']
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rule in self.rules:
                writer.writerow(rule)
    def export_rules_json(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, indent=4, ensure_ascii=False)
    def safe_eval_condition(self, condition, row_dict):
        cond = condition.replace("AND", "and").replace("OR", "or")

        # Asegura que valores categóricos estén entre comillas, pero no los numéricos
        def quote_value(match):
            operator = match.group(1)
            value = match.group(2).strip()

            # Detectar si el valor es numérico
            try:
                float(value)
                is_number = True
            except ValueError:
                is_number = False

            if is_number or value in ['True', 'False']:
                return f"{operator} {value}"

            # Si ya tiene comillas, lo dejamos
            if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                return f"{operator} {value}"

            return f"{operator} '{value}'"

        import re
        pattern = re.compile(r"(==|!=|>=|<=|>|<)\s*([^()\s]+)")
        cond = pattern.sub(quote_value, cond)

        try:
            return eval(cond, {}, row_dict)
        except Exception as e:
            print(f"Error evaluando condición: {cond}\nCon fila: {row_dict}\nError: {e}")
            return False
    def predict_from_rules(self, X, rules_path, min_match_conditions=3):
        """
            Esto es para predecir todo el conjunto de test con las reglas
        """
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        y_pred = []
        for _, row in X.iterrows():
            pred = None
            for rule in rules:
                if self.safe_eval_condition(rule['condition'], row.to_dict()):
                    pred = rule['prediction']
                    break
            # Si no hay coincidencia exacta, buscar coincidencias parciales
            if pred is None and min_match_conditions is not None:
                best_match = None
                max_matched = 0
                for rule in rules:
                    conditions = [c.strip() for c in re.split(r"\s+AND\s+|\s+and\s+", rule['condition'])]

                    match_count = sum([
                        self.safe_eval_condition(cond, row.to_dict())
                        for cond in conditions
                    ])

                    if match_count >= min_match_conditions and match_count > max_matched:
                        best_match = rule
                        max_matched = match_count

                if best_match:
                    pred = best_match['prediction']

            y_pred.append(pred)
        return y_pred
    def rules_covering_test_instance(self, X_test, instance_idx, rules_path):
        """
        Devuelve las reglas que cubren la instancia X_test.iloc[instance_idx].
        """
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        instance = X_test.iloc[instance_idx]
        covering_rules = []
        #print("Instancia a predecir:        " , instance)
        for rule in rules:
            """
            print(" Regla:   ", rule['condition'])
            print(" Condición:  ", instance.to_dict())
            print(" Evluación:  ", self.safe_eval_condition(rule['condition'], instance.to_dict()),"\n\n")
            """
            if self.safe_eval_condition(rule['condition'], instance.to_dict()):
                covering_rules.append(rule)

        return covering_rules
    def export_rules_for_instance(self, rules, instance_idx, path_csv=None, path_json=None, path_txt=None):
        """
        Exporta las reglas para una instancia específica a archivos CSV, JSON y TXT.
        """
        if path_csv:
            with open(path_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = rules[0].keys() if rules else []
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rule in rules:
                    writer.writerow(rule)

        if path_json:
            with open(path_json, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=4, ensure_ascii=False)

        if path_txt:
            with open(path_txt, 'w', encoding='utf-8') as f:
                for i, rule in enumerate(rules):
                    f.write(f"Regla {i+1} para instancia {instance_idx}:\n")
                    f.write(f"IF {rule['condition']} THEN prediction = {rule['prediction']} (support={rule.get('support', 'N/A')})\n")
                    f.write("\n")
