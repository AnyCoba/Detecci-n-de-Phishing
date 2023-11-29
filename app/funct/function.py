import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def eliminar_columnas(df):
    # 
    correlaciones=df.corr()
    # df = df.drop(columns=["id", "HttpsInHostname"]) 
    filtro = correlaciones[(correlaciones['CLASS_LABEL']>0.15) | (correlaciones['CLASS_LABEL']<-0.15)]
    mantener = filtro["CLASS_LABEL"].index
    columnas = [col for col in df.columns if col not in mantener]
    df = df.drop(columns=columnas)

    return df

def entrenar_modelo(X, y):
    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.20,random_state=seed)


    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("selectkbest", SelectKBest()),
        ('classifier', RandomForestClassifier())
    ])

    log_params = {
        "scaler" : [StandardScaler(), None],
        "selectkbest__k":np.arange(5,15),
        "classifier": [LogisticRegression()],
        "classifier__C": [0.1,1,10]                
    }
    rf_params = {
        "scaler" : [StandardScaler(), None],
        "selectkbest__k":np.arange(5,15),
        "classifier": [RandomForestClassifier()],
        "classifier__max_depth": [3,5,7]
    }
    gb_params = {
        "scaler" : [StandardScaler(), None],
        "selectkbest__k":np.arange(5,15),
        "classifier": [GradientBoostingClassifier()],
        "classifier__max_depth": [3,5,7]
    }
    knn_params = {
        "scaler" : [StandardScaler(), None],
        "selectkbest__k":np.arange(5,15),
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": np.arange(5,15)
    }
    svm_params = {
        "scaler" : [StandardScaler(), None],
        "selectkbest__k":np.arange(5,15),
        "classifier": [SVC()],
        "classifier__C": [0.1,1,10]
    }
    pipe
    search_space = [
        log_params,
        rf_params,
        gb_params,
        knn_params,
        svm_params]

    clf_rs = RandomizedSearchCV(estimator=pipe, param_distributions=search_space, cv=5, scoring="recall", verbose=2, n_jobs=-1)
    clf_rs.fit(X_train, y_train)
    return clf_rs, X_train, X_test, y_train, y_test

def mostrar_score(y_test, y_pred):

    recall = recall_score(y_test, y_pred)

    return recall

def confusion_matrix(y_test, y_pred):
    c_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

def generar_df(df_predecir, prediccion):

    df_final = pd.DataFrame()
    df_final['ID'] = df_predecir.index
    df_final['Predicción'] = prediccion
    df_final['Predicción'] = df_final['Predicción'].map({0: "Segura", 1: "Phishing"})

    return df_final

def dashboard(df):
    fig, axes = plt.subplots(4, 4, figsize=(15, 20))  

    No_Phishing = df[df["CLASS_LABEL"] == 0]
    Phishing = df[df["CLASS_LABEL"] == 1]

    # Itero sobre los ejes y las columnas del DF
    for i, ax in enumerate(axes.flatten()):
        if i < len(df.columns):
            col_name = df.columns[i]
            ax.hist(No_Phishing[col_name], bins=5, color='r', alpha=0.5, label='No_Phishing')
            ax.hist(Phishing[col_name], bins=5, color='g', alpha=0.5, label='Phishing')
            ax.set_title(col_name)
            ax.legend()

    plt.tight_layout()
    plt.show()