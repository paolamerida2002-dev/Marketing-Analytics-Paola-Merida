"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")
# Revisamos si la aplicación comienza a ejecutarse.
st.write("Prueba: Streamlit está ejecutando el código.")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"


@st.cache_data
def cargar():
    # Buscamos los datos según dónde esté guardado el archivo.
    carpeta = Path(__file__).resolve().parent
    opciones = [carpeta / "data", carpeta.parent / "data", carpeta.parent.parent / "data"]
    datos = next((ruta for ruta in opciones if (ruta / "catalogo.csv").is_file()), None)

    if datos is None:
        st.error("No encontramos los archivos CSV. Revisa que la carpeta data esté en Tarea, Semana_4 o en la raíz del repositorio.")
        st.stop()

    catalogo = pd.read_csv(datos / "catalogo.csv")
    hist = pd.read_csv(datos / "ordenes_historial.csv")
    hold = pd.read_csv(datos / "ordenes_holdout.csv")
    exp = pd.read_csv(datos / "experimento.csv")
    clientes = pd.read_csv(datos / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
# Revisamos si los datos se cargaron correctamente.
st.write("Prueba 2: Los datos se cargaron correctamente.")
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
# Comparamos las columnas de la matriz X, ya que cada columna representa un producto y sus compras realizadas por los clientes.
# La similitud de coseno permite identificar productos que suelen ser comprados por clientes con comportamientos similares.
S_it = cosine_similarity(X.T)

# Eliminamos la similitud de cada producto consigo mismo, asi evitamos que influya en sus propios puntajes.
np.fill_diagonal(S_it, 0)
# MODELO SVD (8 FACTORES)
svd = TruncatedSVD(n_components=8, random_state=42)
# Calculamos los puntajes de los productos para cada cliente.
Z = svd.fit_transform(X)
S_libre = Z @ svd.components_
# Revisamos si terminaron los cálculos de los modelos.
st.write("Prueba 3: Los modelos se calcularon correctamente.")

# TODO 2 y 3: Calcular los puntajes de los productos
# según el método de recomendación seleccionado.

# Calculamos los puntajes según el método seleccionado.

def puntajes(metodo, x_vec, K, fila=None):

    # Popularidad: usamos los productos más comprados.
    if metodo == "Popularidad":
        return popularidad.astype(float)
    # Filtrado por ítem: buscamos productos parecidos a los que el cliente ya compró.
    if metodo == "Filtrado por ítem":
        return x_vec @ S_it
    # Filtrado por cliente: buscamos otros clientes que tengan compras similares.
    if metodo == "Filtrado por cliente":
        sims = cosine_similarity(x_vec[None, :], X)[0]
        # No incluimos al mismo cliente entre sus vecinos.
        if fila is not None:
            sims[fila] = -1

        vecinos = np.argsort(sims)[-K:]
        return sims[vecinos] @ X[vecinos]
    # SVD: utilizamos los puntajes que ya calculamos.
    if metodo == "SVD (8 factores)":
        return S_libre[fila]
    # Modelo final: usamos ítem si el cliente tiene compras
    # y popularidad si todavía no tiene historial.
