from distutils.command.upload import upload
from logging import PlaceHolder
from nbformat import write
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

global df
        

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Proyecto 2 202003894")

st.sidebar.subheader("Opciones")

archivo = st.sidebar.file_uploader(label="Escoja un archivo", type=['csv','xlsx','xls','json'])

if archivo != None:
    try:
        if archivo.type == 'application/json':
            df = pd.read_json(archivo)
        elif archivo.type == 'text/csv':
            separador = st.sidebar.text_input('Separador',",",placeholder='Ejemplo: ; o ,')
            df = pd.read_csv(archivo, sep=separador)
        else:
            df = pd.read_excel(archivo)

        st.write(df)

        algoritmo = st.sidebar.selectbox(label="Seleccione el algoritmo", key="algoritmo", options=['-----','Lineal','Polinomial','Clas. Gauss','Clas. Arboles', 'Redes Neuronales'])

        if algoritmo == "Lineal":
            columnas = list(df.columns)
            columnX = st.sidebar.selectbox(label="Seleccione X", options=columnas)
            columnY = st.sidebar.selectbox(label="Seleccione Y", options=columnas)

            if columnX != columnY:
                st.subheader("Grafico de Regresion Lineal")
                fig  = px.scatter(df, x = columnX, y=columnY, trendline="ols", trendline_color_override="green")
                st.plotly_chart(fig)

                operaciones = st.multiselect("Escoja las operaciones que desea ver", options=["Prediccion Y","Coeficiente de regresion","Intercepto","R^2","Error Cuadratico"])
                
                x = np.asarray(df[columnX]).reshape(-1, 1)
                y = df[columnY]
                regresion = LinearRegression()
                regresion.fit(x,y)
                prediccion_y = regresion.predict(x)
                r2 = r2_score(y, prediccion_y)

                if "Prediccion Y" in operaciones:
                    st.write("Predicciones de Y")
                    st.write(prediccion_y) 
                
                if "Coeficiente de regresion" in operaciones:
                    st.write("Coeficiente de regresion")
                    st.write(regresion.coef_) 
                
                if "Intercepto" in operaciones:
                    st.write("Intercepto")
                    st.write(regresion.intercept_) 
                
                if "R^2" in operaciones:
                    st.write("R^2")
                    st.write(r2) 
                
                if "Error Cuadratico" in operaciones:
                    st.write("Error Cuadratico")
                    st.write(mean_squared_error(y, prediccion_y)) 
                
                predecir = st.sidebar.text_input("Valor a predecir",placeholder="Ingrese un numero")
                
                if predecir != "":
                    st.write("Prediccion para: "+predecir)
                    st.write(regresion.predict([[float(predecir)]])[0])

        elif algoritmo == "Polinomial":
            columnas = list(df.columns)
            columnX = st.sidebar.selectbox(label="Seleccione X", options=columnas)
            columnY = st.sidebar.selectbox(label="Seleccione Y", options=columnas)
            grado = st.sidebar.text_input("Grado de la regresion",placeholder="Ingrese un numero")

            if columnX != columnY and grado != "":
                st.subheader("Grafico de Regresion Polinomial")

                x = np.asarray(df[columnX]).reshape(-1, 1)
                Xrango = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                y = df[columnY]

                polinomio = PolynomialFeatures(degree=int(grado))
                polinomio.fit(x)
                Xpolinomio = polinomio.transform(x)
                Xrangopoli = polinomio.transform(Xrango)

                regresion = LinearRegression(fit_intercept=False)
                regresion.fit(Xpolinomio, y)
                y_pred = regresion.predict(Xrangopoli)
                
                fig = px.scatter(df, x = columnX, y= columnY, opacity=0.65)
                fig.add_traces(go.Scatter(x=Xrango.squeeze(), y=y_pred,name="grado: "+grado))
                st.plotly_chart(fig)

                operaciones = st.multiselect("Escoja las operaciones que desea ver", options=["R^2","RMSE"])
                
                y_pred = regresion.predict(Xpolinomio)
                if "R^2" in operaciones:
                    r2 = r2_score(y, y_pred)
                    st.write("R^2")
                    st.write(r2)

                
                if "RMSE" in operaciones:
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    st.write("Error Cuadratico")
                    st.write(rmse)
                
                predecir = st.sidebar.text_input("Valor a predecir",placeholder="Ingrese un numero")
                
                if predecir != "":
                    Xmin = float(predecir)
                    Xmax = float(predecir)
                    Xnuevo = np.linspace(Xmin, Xmax, 1)
                    Xnuevo = Xnuevo[:, np.newaxis]
                    Xtrans = polinomio.fit_transform(Xnuevo)
                    st.write("Prediccion para: "+predecir)
                    st.write(regresion.predict(Xtrans)[0])
        
    except:
       st.write("Error al leer el archivo")


