"""
        Importamos las diferentes modulos con las funciones que vamos a utilizar, evaluacion para evaluar el desempleo de cada modelo,
        modelos donde entrenaremos y ejecutaremos cada modelo, procesamiento_datos para procesar los datos y comparacion que devuelve el 
        mejor modelo en función de sus métricas. 
"""
from logica import evaluacion, modelos, procesamiento_datos,comparacion

"""
    Inicializamos una lista donde alamacenaremos los diccionarios en los que alamcenamos los nombres de cada modelo y sus métricas.
"""
resultados=[]

"""
    Llamamos primero a la función procesamiento_datos, despues a cada modelo para entrenarlo a partir de los resultados de la función anterior,
    despues usamos evaluacion con los resultados de la anterior función y con list_nombres_clases de la funcion procesamiento_datos,
    creamos un diccionario en el que almacenamos el nombre del modelo y las métricas de este obtenidas en la función evaluacon,
    agregamos el diccionario a la lista resultados.
"""

X_train, X_test, y_train, y_test, list_nombres_clases = procesamiento_datos.procesamiento_datos()
y_pred, y_proba= modelos.arbol_decision(X_train, y_train, X_test)
acc, prec, rec, f1, auc=evaluacion.evaluacion(y_test,y_pred,y_proba,list_nombres_clases)
valores={"Nombre":"Arbol Decision", "accuracy":acc, "precision": prec, "recall":rec, "f1-score":f1, "auc":auc}
resultados.append(valores)

y_pred, y_proba= modelos.clasificacion_binaria(X_train, y_train, X_test)
acc, prec, rec, f1, auc=evaluacion.evaluacion(y_test,y_pred,y_proba,list_nombres_clases)
valores={"Nombre":"Clasificacion Binaria", "accuracy":acc, "precision": prec, "recall":rec, "f1-score":f1, "auc":auc}
resultados.append(valores)

y_pred, y_proba= modelos.random_forest(X_train, y_train, X_test)
acc, prec, rec, f1, auc=evaluacion.evaluacion(y_test,y_pred,y_proba,list_nombres_clases)
valores={"Nombre":"Random Forest", "accuracy":acc, "precision": prec, "recall":rec, "f1-score":f1, "auc":auc}
resultados.append(valores)

y_pred, y_proba= modelos.xgbc(X_train, y_train, X_test)
acc, prec, rec, f1, auc=evaluacion.evaluacion(y_test,y_pred,y_proba,list_nombres_clases)
valores={"Nombre":"XGBC", "accuracy":acc, "precision": prec, "recall":rec, "f1-score":f1, "auc":auc}
resultados.append(valores)

y_pred, y_proba= modelos.red_neuronal_multicapa(X_train, y_train, X_test)
acc, prec, rec, f1, auc=evaluacion.evaluacion(y_test,y_pred,y_proba,list_nombres_clases)
valores={"Nombre":"Red Neuronal Multicapa", "accuracy":acc, "precision": prec, "recall":rec, "f1-score":f1, "auc":auc}
resultados.append(valores)

"""
    Pasamos la lista resultados a la función comparacion para que nos diga que modelo tiene las mejores métricas. 
"""
comparacion.comparacion(resultados)
