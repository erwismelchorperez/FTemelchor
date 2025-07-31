import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix,matthews_corrcoef
import os
import sys
sys.path.append(os.path.abspath("./"))
from HybridFunctionalTreeClassifier import HybridFunctionalTreeClassifier
from FunctionalRuleExtractor import FunctionalRuleExtractor
from ProccessData import ProccessData
from typing import List, Dict

import sys
import os
sys.path.append(os.path.abspath("./../"))
medidaSplit = ['gini', 'entropy']
banderahybrid = [False, True]
if __name__ == "__main__":
    rowspatterns = []
    rowsft = []
    PathDir = "./../../datasets/"
    contenido = os.listdir(PathDir)
    df_concat = pd.DataFrame()
    print(contenido)
    contenido = sorted(contenido)
    print(contenido)
    balancear = True
    flag = True
    for cont in contenido:
        print("         " + cont)
        name = cont
        #name = "german"# aer, australian, crx, data-eiz-categorica, default+of+credit+card+clients, german, gmsc, heloc, loan_data_set, newhmeq, ppdaiData
        proccessdata = ProccessData(name)
        proccessdata.ReadDataset()
        proccessdata.Xy()
        proccessdata.SplitDataset()
        proccessdata.IdentificarCategoricalFeatures()
        if balancear:
            """
                si es True se balancea el conjunto de datos
            """
            proccessdata.BalancedDataset()
        for split in medidaSplit:
            splittwoing = True
        #for flag in banderahybrid:
            X_train = proccessdata.getXtrain()
            X_test = proccessdata.getXtest()
            y_train = proccessdata.getytrain()
            y_test = proccessdata.getytest()
            print(split, "          ", flag)
            if split == 'twoing' and flag:
                break
            #clf = HybridFunctionalTreeClassifier(max_depth=5, criterion='gini', alpha=0.5, beta=0.5, hybrid = True)
            clf = HybridFunctionalTreeClassifier(max_depth=5, criterion=split, alpha=0.5, beta=0.5, hybrid = flag)
            
            clf.fit(X_train, y_train)
            print(clf.evaluate(X_test, y_test))
            rules = clf.extract_rules()
            #for r in rules:
            #    print(r)
            extractor = FunctionalRuleExtractor(clf)

            rules = extractor.extract_all_rules(X_train, y_train)

            extractor.export_rules_txt("./patrones/patrones_con_estadisticas.txt")
            extractor.export_rules_csv("./patrones/patrones_con_estadisticas.csv")
            extractor.export_rules_json("./patrones/patrones_con_estadisticas.json")

            predictions = extractor.predict_from_rules(X_test, "./patrones/patrones_con_estadisticas.json")

            print(classification_report(y_test, predictions))
            
            cm = confusion_matrix(y_test, predictions)
            mcc = matthews_corrcoef(y_test, predictions)
            tn, fp, fn, tp = cm.ravel()
            # Cálculo de errores tipo I y II
            error_tipo_I = fp / (fp + tn)  # Falsos positivos entre negativos reales
            error_tipo_II = fn / (fn + tp)  # Falsos negativos entre positivos reales

            print(f"\nVerdaderos Negativos (TN): {tn}")
            print(f"Falsos Positivos (FP): {fp}")
            print(f"Falsos Negativos (FN): {fn}")
            print(f"Verdaderos Positivos (TP): {tp}")

            # Calcular métricas manualmente
            accuracy = (tp+tn)/(tp+tn+fp+fn)
            specificity = tn/(tn+fp) if (tn+fp)>0 else 0
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print(f"\nExactitud (Accuracy): {accuracy:.2f}")
            print(f"Precisión: {precision:.2f}")
            print(f"Sensibilidad (Recall): {recall:.2f}")
            print(f"Puntuación F1: {f1_score:.2f}")
            print(f"Matthews correlation coefficient: {mcc:.2f}")

            print("Error Tipo I (FP rate):", error_tipo_I)
            print("Error Tipo II (FN rate):", error_tipo_II)

            result = extractor.rules_covering_test_instance(X_test, instance_idx=4, rules_path="./patrones/patrones_con_estadisticas.json")
            extractor.export_rules_for_instance(result, instance_idx=4,
                                                path_csv="./patrones/patrones_instancia4.csv",
                                                path_json="./patrones/patrones_instancia4.json",
                                                path_txt="./patrones/patrones_instancia4.txt")
            # Para guardar las filas
            row = {
                "DataSet": name,
                "split":split,
                "hybrid":flag,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1_score, 4),
                "AUC": round((recall+specificity)/2, 4),
                "MCC": round(mcc, 4),
                "ErrorTipoI": round(error_tipo_I, 4),
                "ErrorTipoII": round(error_tipo_II, 4),
                "MatrixConfusion": str(cm.tolist())
            }

            rowspatterns.append(row)

            ### predicción sin patrones
            predictions = clf.predict(X_test)

            print(classification_report(y_test, predictions))
            
            cm = confusion_matrix(y_test, predictions)
            mcc = matthews_corrcoef(y_test, predictions)
            tn, fp, fn, tp = cm.ravel()
            # Cálculo de errores tipo I y II
            error_tipo_I = fp / (fp + tn)  # Falsos positivos entre negativos reales
            error_tipo_II = fn / (fn + tp)  # Falsos negativos entre positivos reales

            print(f"\nVerdaderos Negativos (TN): {tn}")
            print(f"Falsos Positivos (FP): {fp}")
            print(f"Falsos Negativos (FN): {fn}")
            print(f"Verdaderos Positivos (TP): {tp}")

            # Calcular métricas manualmente
            accuracy = (tp+tn)/(tp+tn+fp+fn)
            specificity = tn/(tn+fp) if (tn+fp)>0 else 0
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print(f"\nExactitud (Accuracy): {accuracy:.2f}")
            print(f"Precisión: {precision:.2f}")
            print(f"Sensibilidad (Recall): {recall:.2f}")
            print(f"Puntuación F1: {f1_score:.2f}")
            print(f"Matthews correlation coefficient: {mcc:.2f}")

            print("Error Tipo I (FP rate):", error_tipo_I)
            print("Error Tipo II (FN rate):", error_tipo_II)

            # Para guardar las filas
            row = {
                "DataSet": name,
                "split":split,
                "hybrid":flag,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1_score, 4),
                "AUC": round((recall+specificity)/2, 4),
                "MCC": round(mcc, 4),
                "ErrorTipoI": round(error_tipo_I, 4),
                "ErrorTipoII": round(error_tipo_II, 4),
                "MatrixConfusion": str(cm.tolist())
            }
            rowsft.append(row)
            #break
        #break
    # Exportar a CSV
    cadena = 'SinBalancear'
    if balancear:
        cadena = 'Balanceado'
    df = pd.DataFrame(rowspatterns)
    df.to_csv("./metricas/resumen_metricasPaterns_"+cadena+".csv", index=False)
    df = pd.DataFrame(rowsft)
    df.to_csv("./metricas/resumen_metricasFT_"+cadena+".csv", index=False)
