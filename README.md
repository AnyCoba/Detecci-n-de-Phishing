# Detector de Phishing
Por Ana Fernández de la Coba.

## Análisis y Predicción de Phishing de una URL a través de Machine Learning
Este repositorio contiene un modelo de Machine Learning diseñado para determinar si una URL es Phishing o si, por el contrario, es Segura.

### Datos
Fuente: Kaggle
ruta: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning?select=Phishing_Legitimate_full.csv

### Objetivo
Proteger a los usuarios de ataques cibernéticos en los cuales se les roba o daña sus datos personales, pudiendo tener
graves consecuencias como la sustracción de dinero, la suplantación de identidad, la venta de información personal, extorsión, etc.

### Estructura del repositorio
El repositorio está organizado en las siguientes carpetas:

1. app: En su interior se encuentra una aplicación de Streamlit que informa de la problemática y muestra, como solución, un "Detector de Phishing", que realiza una predicción sobre unos modelos reservados.

2. data: Aquí encontramos los datos utilizados en el proyecto en diferentes formas; en crudo, procesados y subdivididos en conjuntos de entrenamiento y prueba para poder llevar a cabo las evoluciones.

3. docs: Contiene una presentación que trata el problema de manera resumida, destacando lo esencial.

4. models: Contiene el modelo seleccionado para el proyecto. Además, se incluye un archivo .yaml que define los parámetros utilizados en el mismo.

5. notebooks: Contiene los archivos jupyter notebook utilizados para el análisis exploratorio y el entrenamiento y evaluación de modelos.

6. src: "contiene los archivos fuente de Python que implementan las funcionalidades clave del proyecto"

## Conclusiones:
Con el modelo creado podremos:
- Conservar nuestra información confidencial.
- Evitar robos o suplantación de identidad.
- Evitar la extorsión de un estafador


