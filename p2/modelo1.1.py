# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # Importar GridSearchCV para la optimización de hiperparámetros
import joblib  # Importar joblib para guardar y cargar modelos
import polars as pl

# Cargar el conjunto de datos (por ejemplo, "Boston_houses.csv")
data = pd.read_csv("Boston_houses.csv")

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
# Visualización de la distribución de precios de las casas
plt.figure(figsize=(8, 6))
sns.histplot(data["price"], bins=20, kde=True)
plt.xlabel("Precio de la casa")
plt.ylabel("Frecuencia")
plt.title("Distribución de precios de las casas")
plt.show()

# Relación entre el número de habitaciones y el precio
plt.figure(figsize=(8, 6))
sns.scatterplot(x="rooms", y="price", data=data)
plt.xlabel("Número de habitaciones")
plt.ylabel("Precio de la casa")
plt.title("Relación entre el número de habitaciones y el precio")
plt.show()

# Preparación de datos
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=65)

# Modelado
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R^2: {r2:.2f}")

# Optimización del modelo
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='euclidean')

# Ajustar hiperparámetros utilizando GridSearchCV
param_grid = {
    'normalize': [True, False],
    'fit_intercept': [True, False]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")

# Despliegue
# Guardar el modelo entrenado en un archivo (por ejemplo, "house_price_model.pkl")
joblib.dump(best_model, "house_price_model.pkl")
print("Modelo guardado exitosamente como house_price_model.pkl")

# Crear una función o script que cargue el modelo y haga predicciones en nuevos datos
def load_and_predict(new_data):
    loaded_model = joblib.load("house_price_model.pkl")
    predictions = loaded_model.predict(new_data)
    return predictions


