from sklearn.model_selection import train_test_split 
import pandas as pd
def procesamiento_datos():
    """
        Leemos el conjunto de datos y lo almacenamos en una variable. 
    """
    PATH = "data/dataset_practica_final.csv"
    data = pd.read_csv(PATH)
    """

    """
    """
        Extraemos los valores posibles de la columna `is_canceled`
    """
    list_nombres_clases=data['is_canceled'].unique()
    data = data.dropna()
    """
        Eliminamos las columnas `is_canceled`, `reservation_status` y `reservation_satus_date` del conjunto de datos y los almacenamos en la variable `X`. 
        Convertimos las variables categóricas en variables numéricas mediante one-hot encoding de la variable `X`. Almacenamos en la variable `y` la columna ìs_canceled`. 
    """
    X = data.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"])
    X = pd.get_dummies(X, drop_first=True)
    y = data['is_canceled']
    """
        Dividimos el conjunto de datos en entrenamiento y prueba. 
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, list_nombres_clases