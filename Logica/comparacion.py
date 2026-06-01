def comparacion(resultados=[]): 
    elegido=""
    acc=0
    prec=0
    rec=0
    f1=0
    auc=0
    for elemento in resultados:
        if elemento["accuracy"]>acc and elemento["precision"]>prec and elemento["recall"]>rec and elemento["f1-score"]>f1 and elemento["auc"]>auc:
            elegido=elemento["Nombre"]
            acc=elemento["accuracy"]
            prec=elemento["precision"]
            rec=elemento["recall"]
            f1=elemento["f1-score"]
            auc=elemento["auc"]
    print(f"El mejor algoritmo es: {elegido}")