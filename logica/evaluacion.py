from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
def evaluacion(y_test, y_pred, y_proba ,list_nombres_clases):
    """
        Args:
            - y_test: Conjunto de datos de prueba.
            - y_pred: Predicciones sobre los datos de prueba.
            - y_proba: Probabilidad de pertenecer a la primera clase de cada predicción. 
            - list_nombres_clases: Nombre de las posibles clase de la predicción. 
        Return:
            - acc: La accuracy_score.
            - prec: El precision_score.
            - rec: El recall_score.
            - f1: El f1_score.
            - auc: El roc_auc_score.
    """
    """
        Creamos y mostramos una matriz de confusión.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_nombres_clases)
    disp.plot()
    plt.show()
    """
        Calculamos las diferentes métricas y las mostramos por pantalla.
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precisión: {prec:.2%}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1-Score:  {f1:.2f}")
    print(f"AUC:       {auc:.2f}\n")
    """
        Creamos y mostramos la curva de ROC.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()
    return acc, prec, rec, f1, auc