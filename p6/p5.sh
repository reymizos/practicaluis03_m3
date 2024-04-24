#!/bin/bash

# Creo la carpeta principal "Proyecto"
mkdir -p Proyecto

#Creo todas los directorios que van dentro de "Proyecto"
touch Proyecto/app.py
mkdir -p Proyecto/utils Proyecto/utils/aplication_logger Proyecto/utils/models Proyecto/utils/test Proyecto/notebooks Proyecto/data Proyecto/data/raw_data Proyecto/data/processed_data Proyecto/data/external_data Proyecto/results Proyecto/results/graficos Proyecto/results/tablas Proyecto/results/archivos_salida Proyecto/docs Proyecto/config Proyecto/assets Proyecto/assets/imagenes Proyecto/assets/fuentes Proyecto/assets/iconos Proyecto/models Proyecto/scripts Proyecto/.git Proyecto/.git/config Proyecto/.git/.gitignore

#Creo todos los archivos que irán dentro de los subdirectorios.

touch Proyecto/utils/aplication_logger/app_logger.py Proyecto/utils/models/data_processing.py Proyecto/utils/test/test_app.py Proyecto/notebooks/01-primer-cuaderno.ipynb Proyecto/notebooks/02-segundo-cuaderno.ipynb Proyecto/notebooks/prototipo-cuaderno.ipynb Proyecto/docs/informes.md Proyecto/docs/descripciones_metodos.md Proyecto/config/configuracion.yaml Proyecto/config/configuracion.json Proyecto/models/modelo_entrenado.pkl Proyecto/models/modelo_entrenado.h5 Proyecto/scripts/limpieza_datos.py Proyecto/scripts/transformaciones.py


echo "Estructura del proyecto creada exitosamente en la carpeta ´Proyecto'."


