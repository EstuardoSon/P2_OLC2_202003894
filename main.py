from sklearn import preprocessing, tree
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import streamlit as st
import graphviz

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
        
        elif algoritmo == 'Clas. Gauss':
            columnas = list(df.columns)
            columnY = st.sidebar.selectbox(label="Seleccione la columna de salida", options=columnas)
            columnsX = st.multiselect("Escoja la columnas de entrada", options=columnas)
            if len(columnsX) != 0 and not(columnY in columnsX):
                x = df[columnsX]
                y = df[columnY]

                x_trans = x.copy()
                encriptados = []
                for columna in columnsX:
                    leX = preprocessing.LabelEncoder()
                    x_trans[columna] = leX.fit_transform(x_trans[columna])
                    encriptados.append(leX)

                leY = preprocessing.LabelEncoder()
                y = leY.fit_transform(y)
                gauss = GaussianNB()
                gauss.fit( x_trans , y)
                valorPred = st.sidebar.text_input(label="Ingrese valores:", placeholder="eje: 0,1,3,4")

                if valorPred != "":
                    valoresPred = np.array(valorPred.split(","))
                    valoresPred = np.array(valorPred.split(","))
                    floatarray = []
                    for i in range(len(encriptados)):
                        floatarray.append(encriptados[i].transform([str(valoresPred[i])])[0])
                    st.write("Valor de prediccion")
                    st.write(leY.inverse_transform(gauss.predict([floatarray]))[0])

        elif algoritmo == 'Clas. Arboles':
            columnas = list(df.columns)
            columnY = st.sidebar.selectbox(label="Seleccione la columna de salida", options=columnas)
            columnsX = st.multiselect("Escoja la columnas de entrada", options=columnas)
            if len(columnsX) != 0 and not(columnY in columnsX):
                x = df[columnsX]
                y = df[columnY]
                arbol=DecisionTreeClassifier()

                x_trans = x.copy()
                encriptados = []
                for columna in columnsX:
                    leX = preprocessing.LabelEncoder()
                    x_trans[columna] = leX.fit_transform(x_trans[columna])
                    encriptados.append(leX)

                leY = preprocessing.LabelEncoder()
                y = leY.fit_transform(y)
                arbol.fit(x_trans,y)

                dot_data = tree.export_graphviz(arbol, out_file=None)   
                st.graphviz_chart(dot_data)

                valorPred = st.sidebar.text_input(label="Ingrese valores:", placeholder="eje: 0,1,3,4")

                if valorPred != "":
                    valoresPred = np.array(valorPred.split(","))
                    floatarray = []
                    for i in range(len(encriptados)):
                        floatarray.append(encriptados[i].transform([str(valoresPred[i])])[0])

                    st.write("Valor de prediccion")
                    st.write(leY.inverse_transform(arbol.predict([floatarray]))[0])

        elif algoritmo == 'Redes Neuronales':
            columnas = list(df.columns)
            columnY = st.sidebar.selectbox(label="Seleccione la columna de salida", options=columnas)
            columnsX = st.multiselect("Escoja la columnas de entrada", options=columnas)
            
            if len(columnsX) != 0 and not(columnY in columnsX):
                x = df[columnsX]
                y = df[columnY]

                x_trans = x.copy()
                encriptados = []
                for columna in columnsX:
                    leX = preprocessing.LabelEncoder()
                    x_trans[columna] = leX.fit_transform(x_trans[columna])
                    encriptados.append(leX)

                leY = preprocessing.LabelEncoder()
                y = leY.fit_transform(y)
                neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
                neural.fit(x_trans, y)
                
                valorPred = st.sidebar.text_input(label="Ingrese valores:", placeholder="eje: 0,1,3,4|5,6,7,8")

                if valorPred != "":
                    arreglo = valorPred.split("|")
                    matriz = []
                    for i in range(len(arreglo)):
                        fila = arreglo[i].split(",")
                        filacod = []
                        for j in range(len(fila)):
                            filacod.append(encriptados[i].transform([fila[i]])[0])
                        matriz.append(filacod)
                    st.write("Valor de prediccion")
                    st.write(leY.inverse_transform(neural.predict(matriz))[0])


    except Exception as e:
       st.write("Error al leer el archivo")
       st.write(e)


