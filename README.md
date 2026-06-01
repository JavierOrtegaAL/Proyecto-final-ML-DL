# Evaluación Final - Dataset

- **Autor:** Javier Ortega Álvarez de Lara
- **Contacto:** jalvarezdelara23@gmail.com
- **Última actualización:** 01/06/2025

--

Este conjunto de datos contiene información sobre reservas de hoteles hechas a lo largo del tiempo, incluyendo detalles sobre los clientes, el comportamiento de reserva y la probabilidad de cancelación.

Es ideal para aplicar modelos de clasificación binaria, donde el objetivo puede ser predecir si una reserva será cancelada (`is_canceled = 1`) o no (`is_canceled = 0`).

## Variables

| Nombre Variable                  | Descripción                                              |
| -------------------------------- | -------------------------------------------------------- |
| `hotel`                          | Tipo de hotel: City Hotel o Resort Hotel                 |
| `is_canceled`                    | Variable objetivo: 1 si fue cancelado, 0 si no           |
| `lead_time`                      | Días entre la reserva y la fecha de llegada              |
| `arrival_date_year`              | Año de llegada                                           |
| `arrival_date_month`             | Mes de llegada                                           |
| `arrival_date_week_number`       | Número de la semana del año                              |
| `arrival_date_day_of_month`      | Día del mes de llegada                                   |
| `stays_in_weekend_nights`        | Noches de fin de semana reservadas                       |
| `stays_in_week_nights`           | Noches entre semana reservadas                           |
| `adults`                         | Número de adultos                                        |
| `children`                       | Número de niños                                          |
| `babies`                         | Número de bebés                                          |
| `meal`                           | Tipo de comida reservada                                 |
| `country`                        | País de origen del cliente                               |
| `market_segment`                 | Canal de marketing (online, offline, grupos...)          |
| `distribution_channel`           | Canal de distribución (directo, TA/TO...)                |
| `is_repeated_guest`              | 1 si el cliente ha estado anteriormente                  |
| `previous_cancellations`         | Nº de cancelaciones anteriores                           |
| `previous_bookings_not_canceled` | Nº de reservas previas no canceladas                     |
| `reserved_room_type`             | Tipo de habitación reservada                             |
| `assigned_room_type`             | Tipo de habitación asignada                              |
| `booking_changes`                | Nº de cambios en la reserva                              |
| `deposit_type`                   | Tipo de depósito: No Deposit, Refundable, etc.           |
| `agent`                          | ID del agente (puede ser nulo)                           |
| `company`                        | ID de la empresa (puede ser nulo)                        |
| `days_in_waiting_list`           | Días en lista de espera                                  |
| `customer_type`                  | Tipo de cliente: Transient, Group, etc.                  |
| `adr`                            | Average Daily Rate (precio promedio por noche)           |
| `required_car_parking_spaces`    | Plazas de parking solicitadas                            |
| `total_of_special_requests`      | Nº de peticiones especiales                              |
| `reservation_status`             | Estado final de la reserva: Check-Out, Canceled, No-Show |
| `reservation_status_date`        | Fecha en que se actualizó el estado                      |

## Roles
Trabajo individual por falta de pareja.

## Justificación  del Problema
Dado los datos de arriba se pide averiguar la variable 'is_canceled', a partir del resto de datos.

## Análisis exploratorio de datos
Nos encontramos con unos datos estructurados en 31 columnas, de las cuales, la segunda, 'is_canceled' es la que hay que predecir a partir del resto. Nos encontramos con columnas de tipo string, int y float. Debido a la presencia de columnas con formato string debemos usar el metodo get_dumies de la libreria Pandas. Separaremos los datos en 4 conjuntos dos de prueba y dos de entrenamiento. Dentro de ambos se dividiran en la columna 'is_canceled' como y, y el resto de columnas como X. 

## Diseño del sistema
Previamente debemos instalar un entorno virtual con 'python -m venv nombre_entorno', activarlo usando elcomando 'nombre_entorno\Scripts\activate' y una vez hecho eso debemos instalar las librerias especificadas en el requirements.txt con 'pip install -r requirements.txt'. Es importante trabajar con una version de Python inferior o igual a la 3.11.9.
He dividido el trabajo en una carpeta Logica por un lado, en la que se enecuentran los modulos, con sus respectivas funciones del mismo nombre:
+ procesamiento_datos que se encarga de procesar los datos y dividirlos con train_test_split y retorna X_train, X_test, y_train, y_test.
+ modelos que se encarga de llamar a los 5 modelos:
    + Arbol de decisiones
    + Clasificacion binaria
    + Random forest
    + XGBC
    + Red neuronal multicapa
+ evaluacion, encargada de calcular las metricas y mostrar una matriz de confusion y la curva de roc, devuelve las 5 metricas:
    + accuracy
    + precision
    + recall
    + f1-score
    + auc
+ comparacion, que tras agregarse en formato diccionario enl main a una lista cada uno de las metricas de los modelos, compara estas y devuelve el nombre de la que tiene valores mas altos.
+ main se encarga de llamar a todos los modulos y sus respectivas funciones, es el que se ejecuta.

## Resultados y eleccion final
La que mejores resultados me ha dado es la red neuronal multicapa, por tanto esa seria mi elección.

## Reflexión crítica sobre limitaciones y mejoras
+ **DecisionTreeClassifier**: Es un modelo muy interpretable y fácil de visualizar, pero tiene como principal limitación su alta tendencia al sobreajuste, especialmente cuando el árbol crece sin restricciones. Además, es inestable ante pequeñas variaciones en los datos, lo que puede cambiar significativamente su estructura y predicciones. Para mejorar su rendimiento, es recomendable aplicar técnicas de poda mediante parámetros como la profundidad máxima o el número mínimo de muestras por nodo, así como utilizar validación cruzada para controlar el overfitting o integrarlo dentro de métodos ensemble como Random Forest.

+ **LogisticRegression**: Es un modelo lineal sencillo, eficiente y muy interpretable, pero su principal limitación es que solo captura relaciones lineales entre variables, lo que reduce su rendimiento en problemas complejos y no lineales. También es sensible a la escala de los datos y a la multicolinealidad entre variables. Para mejorar su comportamiento, es fundamental estandarizar las variables, aplicar regularización (L1, L2 o Elastic Net) y realizar una adecuada ingeniería de características que permita capturar relaciones no lineales.

+ **RandomForestClassifier**: Es un modelo robusto y generalmente muy competitivo, ya que reduce el sobreajuste del árbol individual mediante la combinación de múltiples árboles de decisión. Sin embargo, pierde interpretabilidad y puede volverse costoso en memoria y tiempo cuando el número de árboles es elevado. Además, no extrapola bien fuera del rango de los datos entrenados. Sus mejoras pasan por ajustar hiperparámetros como el número de estimadores o la profundidad de los árboles, así como optimizar el conjunto de variables mediante selección de características o reducción de dimensionalidad.

+ **XGBClassifier (XGBoost)**: Es uno de los modelos más potentes para datos tabulares debido a su capacidad de optimización mediante boosting, pero requiere un ajuste cuidadoso de hiperparámetros y es propenso al sobreajuste si no se regula correctamente. También presenta un coste computacional elevado y menor interpretabilidad. Para mejorarlo, es clave utilizar técnicas como early stopping, regularización L1/L2, ajuste fino de parámetros como learning rate y profundidad, además de muestreo de filas y columnas; su interpretación puede reforzarse mediante herramientas como SHAP.

+ **Red Neuronal Multicapa (MLP)**: Es un modelo muy flexible y capaz de capturar relaciones complejas no lineales, pero su rendimiento depende fuertemente de la arquitectura y del ajuste de hiperparámetros, lo que lo hace sensible y difícil de optimizar. Además, requiere un buen preprocesamiento (especialmente escalado) y grandes cantidades de datos para generalizar correctamente, siendo también poco interpretable. Sus mejoras incluyen el uso de dropout, regularización L2, batch normalization, early stopping y una optimización cuidadosa de la estructura de capas y la tasa de aprendizaje.