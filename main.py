from Logica import evaluacion, modelos, procesamiento_datos,comparacion

resultados=[]

X_train, X_test, y_train, y_test, list_nombres_clases = procesamiento_datos.procesamiento_datos()
y_pred, y_proba= modelos.Arbol_decision(X_train, y_train, X_test)
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

comparacion.comparacion(resultados)
