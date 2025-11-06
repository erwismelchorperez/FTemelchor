import pandas as pd
import numpy as np
import os
import re
from sklearn.utils.validation import check_is_fitted

class FunctionalRuleExtractor:
    """
    Extractor general con eliminación de patrones redundantes:
    - Toma un modelo FT4CIPAdvanced (u otro que implemente extract_rules)
    - Evalúa soporte real de las reglas
    - Elimina reglas equivalentes o contenidas (opcional)
    - Exporta a Excel
    """

    def __init__(self, model, X=None, y=None, output_path=None):
        self.model = model
        self.X = pd.DataFrame(X) if X is not None else None
        self.y = np.array(y) if y is not None else None
        self.output_path = output_path or "."
        os.makedirs(self.output_path, exist_ok=True)
        self.rules_df = None

    def extract_rules(self, export=True, remove_redundant=True):
        try:
            check_is_fitted(self.model, "tree_")
        except Exception as e:
            raise RuntimeError("El modelo no está entrenado o no tiene tree_.") from e

        if hasattr(self.model, "extract_rules"):
            df_rules = self.model.extract_rules(X=self.X, y=self.y, export_path=None)
        elif hasattr(self.model, "get_rules"):
            df_rules = self.model.get_rules()
        else:
            raise RuntimeError("El modelo no expone extract_rules ni get_rules.")

        if isinstance(df_rules, list):
            df_rules = pd.DataFrame(df_rules)
        elif not isinstance(df_rules, pd.DataFrame):
            df_rules = pd.DataFrame(df_rules)

        if "pattern" not in df_rules.columns:
            if "conditions" in df_rules.columns:
                df_rules["pattern"] = df_rules["conditions"].apply(
                    lambda c: " AND ".join(c) if isinstance(c, (list, tuple)) else str(c)
                )
            else:
                df_rules["pattern"] = df_rules.astype(str).agg(" | ".join, axis=1)

        # columnas mínimas
        defaults = {
            "countNegative": 0, "negativeSupport": 0.0,
            "countPositive": 0, "positiveSupport": 0.0,
            "prob": 0.0, "dominant_class": ""
        }
        for col, val in defaults.items():
            if col not in df_rules.columns:
                df_rules[col] = val

        self.rules_df = df_rules[
            ["pattern", "countNegative", "negativeSupport",
             "countPositive", "positiveSupport", "prob", "dominant_class"]
        ].copy()

        # 🔹 eliminar patrones redundantes si se pide
        if remove_redundant:
            before = len(self.rules_df)
            self.rules_df = self.remove_redundant_patterns(self.rules_df)
            after = len(self.rules_df)
            print(f"🧹 Eliminados {before - after} patrones redundantes (de {before} a {after}).")

        if export:
            out = os.path.join(self.output_path, "patterns_all_nodes.xlsx")
            self.rules_df.to_excel(out, index=False)
            print(f"📊 Archivo '{out}' generado correctamente.")

        return self.rules_df

    def _parse_value(self, s):
        try:
            return float(s)
        except Exception:
            return str(s).strip().strip("'").strip('"')

    def _evaluate_rule(self, pattern):
        if self.X is None:
            raise RuntimeError("No hay X para validar reglas (pasa X al constructor).")

        mask = pd.Series(True, index=self.X.index)
        parts = [p.strip() for p in re.split(r"\s+AND\s+", pattern)]

        for cond in parts:
            m = re.match(r"^(.+?)\s+in\s+\{(.*)\}$", cond)
            if m:
                feat = m.group(1).strip()
                items = [it.strip().strip("'").strip('"') for it in m.group(2).split(",")]
                mask &= self.X[feat].astype(str).isin(items)
                continue
            m = re.match(r"^(.+?)\s*<=\s*(.+)$", cond)
            if m:
                feat, val = m.group(1).strip(), self._parse_value(m.group(2))
                mask &= pd.to_numeric(self.X[feat], errors='coerce') <= float(val)
                continue
            m = re.match(r"^(.+?)\s*>\s*(.+)$", cond)
            if m:
                feat, val = m.group(1).strip(), self._parse_value(m.group(2))
                mask &= pd.to_numeric(self.X[feat], errors='coerce') > float(val)
                continue
            m = re.match(r"^(.+?)\s*=\s*(.+)$", cond)
            if m:
                feat, val = m.group(1).strip(), m.group(2).strip().strip("'").strip('"')
                mask &= self.X[feat].astype(str) == val
        return mask

    def validate_rule_support_real(self, rules_df=None, export=True):
        if self.X is None or self.y is None:
            raise RuntimeError("Necesitas pasar X y y al constructor para validar soporte real.")
        if rules_df is None:
            rules_df = self.rules_df

        validated_rows = []
        n_total = len(self.y)
        for _, row in rules_df.iterrows():
            pattern = row.get("pattern", "")
            if not pattern:
                validated_rows.append({
                    "pattern": pattern, "countNegative_real": 0, "countPositive_real": 0,
                    "negativeSupport_real": 0.0, "positiveSupport_real": 0.0
                })
                continue
            mask = self._evaluate_rule(pattern)
            neg = int(np.sum((self.y == 0) & mask))
            pos = int(np.sum((self.y == 1) & mask))
            validated_rows.append({
                "pattern": pattern,
                "countNegative_real": neg,
                "countPositive_real": pos,
                "negativeSupport_real": round(neg / n_total * 100, 6),
                "positiveSupport_real": round(pos / n_total * 100, 6)
            })

        validated_df = pd.DataFrame(validated_rows)
        merged = pd.merge(rules_df.reset_index(drop=True), validated_df, on="pattern", how="left")

        if export:
            out = os.path.join(self.output_path, "rules_validated.xlsx")
            merged.to_excel(out, index=False)
            print(f"📂 Archivo de validación generado: {out}")

        return merged

    # ============================================================
    # 🔹 NUEVO: eliminación de patrones redundantes o contenidos
    # ============================================================
    def remove_redundant_patterns(self, df_rules):
        """
        Elimina patrones:
        - Duplicados (mismas condiciones, distinto orden)
        - Contenidos (patrón A ⊂ B)
        Conserva solo los más generales.
        """
        # normalizar condiciones
        df_rules = df_rules.copy()
        df_rules["conditions_set"] = df_rules["pattern"].apply(
            lambda p: frozenset([c.strip() for c in re.split(r"\s+AND\s+", str(p)) if c.strip() != ""])
        )

        # eliminar duplicados exactos
        df_rules = df_rules.drop_duplicates(subset=["conditions_set"])

        # eliminar patrones contenidos
        to_drop = set()
        for i, cond_i in enumerate(df_rules["conditions_set"]):
            for j, cond_j in enumerate(df_rules["conditions_set"]):
                if i != j and cond_i.issubset(cond_j):
                    # i está contenido dentro de j → eliminar j (más específico)
                    to_drop.add(j)
        df_rules = df_rules.drop(df_rules.index[list(to_drop)]).reset_index(drop=True)
        df_rules = df_rules.drop(columns=["conditions_set"])
        return df_rules



    def export_to_excel(self, path="rules_extracted.xlsx"):
        if self.rules_df is None or self.rules_df.empty:
            print("⚠️ No hay reglas para exportar.")
            return
        self.rules_df.to_excel(path, index=False)
        print(f"✅ Reglas exportadas correctamente a: {path}")

    def export_to_csv(self, path="rules_extracted.csv"):
        if self.rules_df is None or self.rules_df.empty:
            print("⚠️ No hay reglas para exportar.")
            return
        self.rules_df.to_csv(path, index=False)
        print(f"✅ Reglas exportadas correctamente a: {path}")
