# main.py
import pandas as pd
from ft4cip import FT4CIP
from sklearn.metrics import roc_auc_score

# Cargar datos desde CSV
df = pd.read_csv("dataset.csv")
X = df.drop(columns=['target']).values
y = df['target'].values

# Entrenar modelo
model = FT4CIP(max_depth=4, min_samples_split=5)
model.fit(X, y)

# Predicción
y_pred = model.predict(X)

# Evaluar AUC
auc = roc_auc_score(y, y_pred)
print(f"AUC: {auc:.4f}")
