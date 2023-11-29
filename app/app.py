import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from funct.function import eliminar_columnas, entrenar_modelo, mostrar_score, generar_df, dashboard

path_dataframe = r"..\data\raw\Phishing_Legitimate_full.csv"
df = pd.read_csv(path_dataframe, index_col=0)
df_predecir = pd.read_csv(r"..\data\test\test.csv", index_col=0)

# Título de la página
st.set_page_config(page_title = "Detector de Phishing", page_icon= ":anzuelo:")
st.title("¿Es fiable esta URL?:")
st.header("Una Clasificación con Machine Learning")
st.image("https://assets.retarus.com/blog/es/wp-content/uploads/sites/26/2022/05/shutterstock_1086090842.png",
         caption="Ana Fernández")
st.divider()

# Sidebar
st.sidebar.title("Contenido")
st.sidebar.divider()

# Contenido
if st.sidebar.button('¿Me fío o no me fío?'):
    st.title("¡Todos podemos ser víctimas de Phishing!")
    st.markdown("Es por eso que nuestra empresa busca proteger al usuario de ataques cibernéticos con una herramienta altamente eficaz")
    st.markdown("La idea principal es crear un clasificador de URL en la que podamos comprobar si es 'Phishing' o 'Segura', y así no comprometer nuestros datos.")
    st.markdown("Para ello, se utilizan múltiples variables: longitud de la URL, número de puntos, niveles de ruta, número de guiones, longitud del hostname, número de palabras sensibles, porcentaje de hipervínculos, etc.")

st.write("")
st.divider()
st.write("")

# st.write(dashboard(df))
df = eliminar_columnas(df)

# Evaluar y entrenar el modelo
X = df.drop("CLASS_LABEL", axis=1)
y = df["CLASS_LABEL"]


model, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)

print(model.best_estimator_)
print(model.best_score_)
print(model.best_params_)

final_model = model.best_estimator_
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred = final_model.predict(X_test)

# Mostrar el recall_score
st.write("")
st.write("")

score = mostrar_score(y_test, y_pred)
# st.write(f"Recall Score: {score}")

# Mostrar las predicciones
st.write("Predicciones:")
st.write("")
st.write(f"Recall Score: {score.round(2)}")
st.write("")
st.write("")

# Crear y mostrar el mapa de calor de la matriz de confusión

# st.write("")
# st.write(f"{confusion_matrix(y_test, y_pred)}")
# st.write("")
# st.write("")


# Probamos con datos reales
st.write("Probamos con datos reales:")
st.write("")
st.write("")
df_predecir = df_predecir.head()
# st.write(df_predecir)

prediccion = final_model.predict(df_predecir)
st.write("")
st.write("")
st.write(generar_df(df_predecir, prediccion))
st.write("")
st.divider()
st.write(f"Recuerda que el detector tiene una fiabilidad del {score.round(2)}")