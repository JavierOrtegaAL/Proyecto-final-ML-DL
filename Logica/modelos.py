"""
    Importamos las librerias correspondientes a los 5 modelos:
        - LogisticRegression
        - DecisionTreeClassifier
        - RandomForestClassifier
        - XGBClassifier
        - tensorflow: Para la red neuronal multicapa
    Tambien importamos StandarScaler para escalar los datos y matplotlib.pyplot para mostrar graficos.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
def arbol_decision( X_train, y_train, X_test):
    """
        Args:
            - X_train: Conjunto de datos de entrenamiento.
            - y_train: Conjunto de predicciones de datos de entrenamiento.
            - X_test: Conjunto de datos de prueba. 
        Return:
            - y_pred: Predicciones hechas sobre los datos de prueba.
            - y_proba : Probabilidad de pertenecia a la primera clase. 
    """
    """
        Creamos el modelo de DecisionTreeClassifier, con una profundidad maxima de 3 nodos, 
        un numero minimo de muestras que debe tener el nodo para dividirse de 20,
        un número mínimo de muestras que debe tener cada hoja de 10 y fijamos una semilla aleatoria ern 42.
    """
    modelo_dt = DecisionTreeClassifier(max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42)
    """
        Entrenamos el modelo con los datos de entrenamiento.
    """
    modelo_dt.fit(X_train, y_train)
    """
        Hallamos las predicciones de los datos de entrenamiento y la probabilidad de pertenencia a la clase primera.
    """
    y_pred = modelo_dt.predict(X_test)
    y_proba = modelo_dt.predict_proba(X_test)[:, 1]
    """
        Creamos un gráfico de la composición del arbol de decisión.  
    """
    plt.figure(figsize=(12,6))
    plot_tree(modelo_dt, feature_names=X_train.columns.to_list(), class_names=['0', '1'], filled=True)
    plt.show()
    return y_pred, y_proba

def clasificacion_binaria( X_train, y_train, X_test):
    """
        Args:
            - X_train: Conjunto de datos de entrenamiento.
            - y_train: Conjunto de predicciones de datos de entrenamiento.
            - X_test: Conjunto de datos de prueba. 
        Return:
            - y_pred: Predicciones hechas sobre los datos de prueba.
            - y_proba : Probabilidad de pertenecia a la primera clase. 
    """
    """
        Escalamos los datos de entrenamiento, con fit_transform para tranformarlos y aprender de ellos, 
        y de prueba con transform para solo transformarlos. 
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """
        Creamos el modelo de LogisticRegression, 
        un numero maximo de iteraciones de 200 
        y fijamos una semilla aleatoria ern 42.
    """
    modelo_rl = LogisticRegression(max_iter=200, random_state=42)
    """
        Entrenamos el modelo con los datos de entrenamiento.
    """
    modelo_rl.fit(X_train,y_train)
    """
        Calculamos la predicción y la probabilidad de pertencia a la primera clase. 
        Con ravel convertimos un array de varias dimensiones en un vector de una sola dimensión. 
        En y_pred comparando la probabilidad de pertenencia a una clase y le asiganamos un True o False en función del resultado, y lo convertimos a 0 o 1. 
    """
    y_proba = modelo_rl.predict(X_test).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba

def random_forest( X_train, y_train, X_test):
    """
        Args:
            - X_train: Conjunto de datos de entrenamiento.
            - y_train: Conjunto de predicciones de datos de entrenamiento.
            - X_test: Conjunto de datos de prueba. 
        Return:
            - y_pred: Predicciones hechas sobre los datos de prueba.
            - y_proba : Probabilidad de pertenecia a la primera clase. 
    """
    """
        Creamos el modelo de RandomForestClassifier, con un numero de 200 arboles, 
        una profundidad maxima de cada arbol de 5,
        un número mínimo de muestras que debe contener una hoja de 5,  
        y fijamos una semilla aleatoria ern 42.
    """
    modelo_rfc = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    """
        Entrenamos el modelo con los datos de entrenamiento.
    """
    modelo_rfc.fit(X_train, y_train)
    """
        Hallamos las predicciones de los datos de entrenamiento y la probabilidad de pertenencia a la clase primera.
    """
    y_pred = modelo_rfc.predict(X_test)
    y_proba = modelo_rfc.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def xgbc( X_train, y_train, X_test):
    """
        Args:
            - X_train: Conjunto de datos de entrenamiento.
            - y_train: Conjunto de predicciones de datos de entrenamiento.
            - X_test: Conjunto de datos de prueba. 
        Return:
            - y_pred: Predicciones hechas sobre los datos de prueba.
            - y_proba : Probabilidad de pertenecia a la primera clase. 
    """
    """
        Creamos el modelo de XGBClassifier, con un numero de 200 arboles, 
        una profundidad maxima de cada arbol de 5,
        un learning rate de 0.1,  
        logloss como la métrica que el modelo usa para evaluar su rendimiento durante el entrenamiento
        y fijamos una semilla aleatoria ern 42.
    """
    modelo_xgbc = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
    """
        Entrenamos el modelo con los datos de entrenamiento.
    """
    modelo_xgbc.fit(X_train, y_train)
    """
        Hallamos las predicciones de los datos de entrenamiento y la probabilidad de pertenencia a la clase primera.
    """
    y_pred = modelo_xgbc.predict(X_test)
    y_proba = modelo_xgbc.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def	red_neuronal_multicapa( X_train, y_train, X_test):
    """
        Args:
            - X_train: Conjunto de datos de entrenamiento.
            - y_train: Conjunto de predicciones de datos de entrenamiento.
            - X_test: Conjunto de datos de prueba. 
        Return:
            - y_pred: Predicciones hechas sobre los datos de prueba.
            - y_proba : Probabilidad de pertenecia a la primera clase. 
    """
    """
        Escalamos los datos de entrenamiento, con fit_transform para tranformarlos y aprender de ellos, 
        y de prueba con transform para solo transformarlos. 
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """
        Creamos el modelo con Sequential, que se compondra de 5 capas, la de entrada, 3 ocultas con l activacion relu y la de salida con la activacion sigmoid
    """
    model_rnm = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1], ), name='i1'),
            tf.keras.layers.Dense(128, activation='relu', name='h1'),
            tf.keras.layers.Dense(64, activation='relu', name='h2'),
            tf.keras.layers.Dense(32, activation='relu', name='h3'),
            tf.keras.layers.Dense(1, activation='sigmoid',name='o1'),
    ])
    """
        Creamos un optimizador para el modelo de tipo Adam con un learning rate de 0.001.
    """
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    """
        Compilamos el modelo con el optimizador, una perdida de tipo binary_crossentropy y con accuracy como metrica.
    """
    model_rnm.compile(optimizer=optimizer_adam, loss='binary_crossentropy', metrics=['accuracy'])
    """
        Mostramos un resumen del modelo.
    """
    model_rnm.summary()
    """
        Creamos un EarlyStopping para para el modelo en caso de que la perdida comience a aumentar, monitoreara el val_loss, 
        un patience de 10 para limitar a 10 el numero epochs que se puede hacer sin que mejore el modelo,
        true en restore_best_weights para que se restauren los pesos de la época donde el modelo tuvo mejor rendimiento en validación
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    """
        Entrenamos el modelo con los datos de entrenamiento, limitamos a 100 el numero de epochs, un 20% de los datos se usaran para validación,
        y un verbose con valor 1 para que muestre el progreso por pantalla, tambien hacemos un callback al EarlyStopping para que pare cuando deje de mejorar el modelo.
    """
    model_rnm.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    """
        Calculamos la predicción y la probabilidad de pertencia a la primera clase. 
        Con ravel convertimos un array de varias dimensiones en un vector de una sola dimensión. 
        En y_pred comparando la probabilidad de pertenencia a una clase y le asiganamos un True o False en función del resultado, y lo convertimos a 0 o 1. 
    """
    y_proba = model_rnm.predict(X_test).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba