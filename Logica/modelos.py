from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
def Arbol_decision( X_train, y_train, X_test):
    modelo_dt = DecisionTreeClassifier(max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42)
    modelo_dt.fit(X_train, y_train)
    y_pred = modelo_dt.predict(X_test)
    y_proba = modelo_dt.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(12,6))
    plot_tree(modelo_dt, feature_names=X_train.columns.to_list(), class_names=['0', '1'], filled=True)
    plt.show()
    return y_pred, y_proba

def clasificacion_binaria( X_train, y_train, X_test):
    modelo_rl = LogisticRegression(max_iter=200, random_state=42)
    modelo_rl.fit(X_train,y_train)
    y_pred = modelo_rl.predict(X_test)
    y_proba = modelo_rl.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def random_forest( X_train, y_train, X_test):
    modelo_rfc = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    modelo_rfc.fit(X_train, y_train)
    y_pred = modelo_rfc.predict(X_test)
    y_proba = modelo_rfc.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def xgbc( X_train, y_train, X_test):
    modelo_xgbc = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
    modelo_xgbc.fit(X_train, y_train)
    y_pred = modelo_xgbc.predict(X_test)
    y_proba = modelo_xgbc.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def	red_neuronal_multicapa( X_train, y_train, X_test):
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	model_rnm = tf.keras.models.Sequential([
			tf.keras.layers.Input(shape=(X_train.shape[1], ), name='i1'),
			tf.keras.layers.Dense(128, activation='relu', name='h1'),
			tf.keras.layers.Dense(64, activation='relu', name='h2'),
			tf.keras.layers.Dense(32, activation='relu', name='h3'),
			tf.keras.layers.Dense(1, activation='sigmoid',name='o1'),
	])
	optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
	model_rnm.compile(optimizer=optimizer_adam, loss='binary_crossentropy', metrics=['accuracy'])
	model_rnm.summary()
	early_stopping = tf.keras.callbacks.EarlyStopping(
    	monitor='val_loss',
    	patience=10,
    	restore_best_weights=True
	)
	model_rnm.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])
	y_proba = model_rnm.predict(X_test).ravel()
	y_pred = (y_proba > 0.5).astype(int)
	return y_pred, y_proba