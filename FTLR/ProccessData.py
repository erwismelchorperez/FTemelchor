import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC, SMOTE
class ProccessData:
    def __init__(self, namedataset):
        self.namedataset = namedataset
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.categorical_features = []
    def ReadDataset(self):
        self.dataset = pd.read_csv("./../../datasets/" + self.namedataset + ".csv")
        self.PreProccessDataset()
    def PreProccessDataset(self):
        self.dataset = self.dataset.dropna(axis=0)
    def Xy(self):
        self.X = self.dataset.drop(columns="target")
        self.y = self.dataset["target"]
    def SplitDataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42,stratify=self.y)
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
    def BalancedDataset(self):
        # Inicializar SMOTENC indicando las columnas categóricas
        if len(self.categorical_features) == 0:
            # Aplicar SMOTE para balancear
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        else:
            smote_nc = SMOTENC(
                categorical_features=self.categorical_features,
                sampling_strategy='auto',  # Equilibrar las clases
                random_state=42)
            # Aplicar el remuestreo solo al conjunto de entrenamiento
            self.X_train, self.y_train = smote_nc.fit_resample(self.X_train, self.y_train)

    def IdentificarCategoricalFeatures(self):
        # Columnas categóricas (incluye strings y variables booleanas)
        self.categorical_features = self.getXtrain().select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        print("Columnas categóricas:", self.categorical_features)
    def getXtrain(self):
        return self.X_train
    def getytrain(self):
        return self.y_train
    def getXtest(self):
        return self.X_test
    def getytest(self):
        return self.y_test
    def getDataset(self):
        return self.dataset