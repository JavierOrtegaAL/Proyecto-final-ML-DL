def comparacion(resultados=[]): 
    """
        Args:
            - resultados: Una lista con los diccionarios de cada uno de los modelos.
    """
    """
        Inicializamos cada metrica en 0 y el nombre del modelo, 'elegido' en un string vacío. 
    """
    elegido=""
    acc=0
    prec=0
    rec=0
    f1=0
    auc=0
    """
        Recorremos la lista y en caso que todas las metricas sean mayores a las que tenemos almacenada, seteamos el nombre del modelo "elegido".
    """
    for elemento in resultados:
        if elemento["accuracy"]>acc and elemento["precision"]>prec and elemento["recall"]>rec and elemento["f1-score"]>f1 and elemento["auc"]>auc:
            elegido=elemento["Nombre"]
            acc=elemento["accuracy"]
            prec=elemento["precision"]
            rec=elemento["recall"]
            f1=elemento["f1-score"]
            auc=elemento["auc"]
    """
        Mostramos por pantalla el modelo elegido. 
    """
    print(f"El mejor algoritmo es: {elegido}")