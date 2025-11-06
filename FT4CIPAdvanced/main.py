import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTENC
from modelos.FT4CIPAdvanced import FT4CIPAdvanced
from modelos.FunctionalRuleExtractor import FunctionalRuleExtractor

# ================================
# 🔹 1. Cargar datos
# ================================
df = pd.read_csv("./dataset/australian.csv")  # Ajusta el path si es necesario

# Verifica la existencia de la columna target
if "target" not in df.columns:
    raise ValueError("❌ La columna 'target' no existe en el dataset. Verifica el nombre real de tu variable objetivo.")

X = df.drop("target", axis=1)
y = df["target"]

# Normalizar etiquetas
y = y.replace({0: "negative", 1: "positive", "0": "negative", "1": "positive"})

# ================================
# 🔹 2. Separar entrenamiento y prueba
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ================================
# 🔹 3. Balancear conjunto de entrenamiento (SMOTENC)
# ================================
print("\n⚖️ Aplicando SMOTENC para balancear clases...")

# Detectar columnas categóricas (por tipo o pocos valores únicos)
categorical_features = [
    i for i, col in enumerate(X_train.columns)
    if X_train[col].dtype == "object" or X_train[col].nunique() <= 10
]

# Convertir categóricas a tipo string para SMOTENC
X_train = X_train.copy()
for col in X_train.columns:
    if col in X_train.columns[categorical_features]:
        X_train[col] = X_train[col].astype(str)

# Aplicar SMOTENC
smote = SMOTENC(categorical_features=categorical_features, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print(f"✅ Clases balanceadas: {dict(pd.Series(y_res).value_counts())}")

# ================================
# 🔹 4. Entrenar modelo
# ================================
print("\n🚀 Entrenando modelo FT4CIPAdvanced...")
model = FT4CIPAdvanced(max_depth=20, alpha=0.5, beta=0.3)
model.fit(X_res, y_res)

# ================================
# 🔹 5. Evaluar modelo
# ================================
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

# Detectar todas las clases presentes
all_labels = sorted(list(set(y_test) | set(y_pred)))
print("\n📈 Resultados del modelo:")
print(classification_report(y_test, y_pred, labels=all_labels, zero_division=0))

# ================================
# 🔹 6. Extraer reglas
# ================================
print("\n🧩 Extrayendo reglas...")
extractor = FunctionalRuleExtractor(model, X_res, y_res)
rules_df = extractor.extract_rules(remove_redundant=False)  # genera patterns_all_nodes.xlsx automáticamente

print(f"\n✅ Total de reglas extraídas: {len(rules_df)}")

# ================================
# 🔹 7. Validar soporte real
# ================================
def validate_rule_support_real(X, y, rules_df):
    """Valida el soporte real de cada regla considerando etiquetas 'positive' y 'negative'."""
    print("\n🔍 Validando soporte real de las reglas...")
    supports = []
    total = len(y)

    for _, row in rules_df.iterrows():
        conditions = row["pattern"]
        mask = pd.Series(True, index=X.index)

        # Evaluar condiciones como "A7 > 1.384 AND A8 = f"
        for cond in conditions.split(" AND "):
            cond = cond.strip()
            if ">" in cond:
                attr, val = cond.split(">")
                mask &= X[attr.strip()].astype(float) > float(val.strip())
            elif "<=" in cond:
                attr, val = cond.split("<=")
                mask &= X[attr.strip()].astype(float) <= float(val.strip())
            elif "=" in cond:
                attr, val = cond.split("=")
                mask &= X[attr.strip()].astype(str) == val.strip().strip("'")

        matched = y[mask]
        count_pos = (matched == "positive").sum()
        count_neg = (matched == "negative").sum()

        supports.append({
            "pattern": conditions,
            "countPositive_real": count_pos,
            "positiveSupport_real": f"{round(count_pos / total * 100, 3)}%",
            "countNegative_real": count_neg,
            "negativeSupport_real": f"{round(count_neg / total * 100, 3)}%"
        })

    df_val = pd.DataFrame(supports)
    df_val.to_excel("patterns_real_support.xlsx", index=False)
    df_val.to_csv("patterns_real_support.csv", index=False)
    print("✅ Validación completada. Soporte real calculado.")
    print("📊 Archivo 'patterns_real_support.xlsx/patterns_real_support.csv' generado correctamente.")
    return df_val


validated = validate_rule_support_real(X_res, y_res, rules_df)

print("\n✅ Todo finalizado correctamente.")
