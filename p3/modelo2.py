# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Cargar el conjunto de datos (por ejemplo, "otro_archivo.csv")
data = pd.read_csv("otro_archivo.csv")

# Limpieza de datos
# (Realizar las operaciones de limpieza aquí)
# Por ejemplo, imputar valores faltantes o eliminar filas/columnas

# Imputar valores faltantes (si es necesario)
data.fillna(method="ffill", inplace=True)

# Eliminar filas duplicadas
data.drop_duplicates(inplace=True)

# Transformación de variables categóricas (si es necesario)
# Por ejemplo, codificación one-hot para variables categóricas

# Análisis exploratorio de datos (EDA)
# Visualización de la distribución de ventas
plt.figure(figsize=(8, 6))
sns.histplot(data["ventas"], bins=20, kde=True)
plt.xlabel("Ventas")
plt.ylabel("Frecuencia")
plt.title("Distribución de ventas")
plt.show()

# Relación entre el precio y la cantidad de publicidad
plt.figure(figsize=(8, 6))
sns.scatterplot(x="publicidad", y="ventas", data=data)
plt.xlabel("Publicidad")
plt.ylabel("Ventas")
plt.title("Relación entre el precio y la cantidad de publicidad")
plt.show()

# Preparación de datos
X = data.drop("ventas", axis=1)
y = data["ventas"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelado
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R^2: {r2:.2f}")

# Optimización del modelo
# Ajustar hiperparámetros utilizando GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")

# Despliegue
# Guardar el modelo entrenado en un archivo (por ejemplo, "otro_modelo.pkl")
joblib.dump(best_model, "otro_modelo.pkl")
print("Modelo guardado exitosamente como otro_modelo.pkl")

# Crear una función o script que cargue el modelo y haga predicciones en nuevos datos
def cargar_y_predecir(nuevos_datos):
    modelo_cargado = joblib.load("otro_modelo.pkl")
    predicciones = modelo_cargado.predict(nuevos_datos)
    return predicciones


