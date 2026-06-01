from sklearn.model_selection import train_test_split 
import pandas as pd
def procesamiento_datos():
    PATH = "data/dataset_practica_final.csv"
    data = pd.read_csv(PATH)
    list_nombres_clases=data['is_canceled'].unique()
    data = data.dropna()
    X = data.drop(columns=["is_canceled", "reservation_status", "reservation_status_date"])
    X = pd.get_dummies(X, drop_first=True)
    y = data['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, list_nombres_clases