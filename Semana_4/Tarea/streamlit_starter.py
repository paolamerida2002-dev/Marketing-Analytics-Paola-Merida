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
    if metodo == "Final (ítem + respaldo)":

        if x_vec.sum() > 0:
            return x_vec @ S_it
        else:
            return popularidad.astype(float)


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption("Esta aplicación ayuda a recomendar productos según las compras de cada cliente y a comparar los diferentes métodos.")


# TODO 4: Controles para elegir cómo queremos las recomendaciones.
with st.sidebar:
    st.header("Opciones de recomendación")

    # Elegimos el método de recomendación.
    metodo = st.selectbox(
        "Método de recomendación",
        [
            "Popularidad",
            "Filtrado por ítem",
            "Filtrado por cliente",
            "SVD (8 factores)",
            "Final (ítem + respaldo)"
        ]
    )

    # Elegimos cuántos clientes similares vamos a tomar en cuenta.
    K = st.slider(
        "Número de vecinos (K)",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        disabled=(metodo != "Filtrado por cliente")
    )

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])


# TODO 5: Mostrar las recomendaciones del cliente.
with tab1:

    # Elegimos el cliente que queremos consultar.
    cid = st.selectbox("Selecciona un cliente", ids)

    # Buscamos las compras que ha realizado ese cliente.
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    comprados = [productos[j] for j in np.where(x_vec > 0)[0]]

    st.subheader(f"Recomendaciones para el cliente {cid}")

    # Mostramos los productos que ya ha comprado.
    if len(comprados) > 0:
        st.write("Productos que ya compró:", ", ".join(comprados))
    else:
        st.info("Este cliente todavía no tiene compras registradas.")

    # Calculamos los puntajes según el método elegido.
    if len(comprados) == 0:
        scores = popularidad.astype(float)
        st.info("Como no tiene historial, recomendamos productos populares.")
    else:
        scores = puntajes(metodo, x_vec, K, fila)

    # Elegimos los tres productos con mayor puntaje.
    recomendaciones = top3(scores, x_vec)
    st.subheader("Productos recomendados")

    # Mostramos cada producto y explicamos por qué lo recomendamos.
    for i, j in enumerate(recomendaciones):

        producto = productos[j]
        puntaje = scores[j]


        if len(comprados) == 0:
            porque = "Es uno de los productos más comprados."

        elif metodo == "Popularidad":
            porque = "Es uno de los productos más populares."

        elif metodo == "Filtrado por ítem":
            similares = np.where(x_vec > 0)[0]
            mejor = similares[np.argmax(S_it[similares, j])]

            porque = (
                f"Porque ya compró {productos[mejor]}, "
                f"que tiene una similitud de {S_it[mejor, j]:.2f}."
            )

        elif metodo == "Filtrado por cliente":
            sims = cosine_similarity(x_vec[None, :], X)[0]
            sims[fila] = -1
            vecinos = np.argsort(sims)[-K:]
            cantidad = int(X[vecinos, j].sum())

            porque = (
                f"{cantidad} de los {K} clientes más similares "
                f"compraron este producto."
            )

        elif metodo == "SVD (8 factores)":
            porque = (
                "Lo recomendamos porque el modelo encontró "
                "patrones de compra relacionados con este producto."
            )

        elif metodo == "Final (ítem + respaldo)":
            porque = (
                "Lo recomendamos tomando en cuenta "
                "los productos que ya compró."
            )

        # Mostramos cada recomendación con su puntaje y explicación.
        st.markdown(f"### {i + 1}. {producto}")
        st.write(f"**Puntaje:** {puntaje:.2f}")
        st.write(f"**¿Por qué?** {porque}")
        st.divider()


# TODO 6: Comparar los métodos de recomendación.

with tab2:
    st.subheader("Comparación de recomendadores")
    st.write(
        "Comparamos los métodos para ver cuáles recomiendan "
        "mejor los productos y cuáles ofrecen más variedad."
    )

    # Usamos solo los clientes de control que probaron
    # al menos un producto nuevo.
    elegibles = es_control & (NUEVO.sum(axis=1) > 0)
    filas_control = np.where(elegibles)[0]

    # Elegimos cuántos clientes queremos evaluar.
    if len(filas_control) == 0:
        st.warning("No hay clientes elegibles del grupo de control para comparar.")
        st.stop()
    cantidad = st.slider(
        "Cantidad de clientes a evaluar",
        min_value=1,
        max_value=min(500, len(filas_control)),
        value=min(200, len(filas_control)),
        step=1
    )

    # Seleccionamos los primeros clientes del grupo de control.
    muestra = filas_control[:cantidad]

    # Guardamos los resultados de cada método.
    resultados = []

    # Métodos que vamos a comparar.
    metodos = [
        "Popularidad",
        "Filtrado por ítem",
        "Filtrado por cliente",
        "SVD (8 factores)",
        "Final (ítem + respaldo)"
    ]


    for nombre in metodos:

        aciertos = 0
        total_aciertos = 0
        productos_recomendados = set()

        for fila in muestra:

            # Obtenemos las compras del cliente.
            x_vec = X[fila]
            # Calculamos las recomendaciones para este cliente.
            scores = puntajes(nombre, x_vec, K, fila)

            # Revisamos cuáles de los tres productos compró después.
            recomendaciones = top3(scores, x_vec)
            n_aciertos = int(NUEVO[fila, recomendaciones].sum())

            # Contamos los clientes y productos en los que acertamos.
            if n_aciertos > 0:
                aciertos += 1
            total_aciertos += n_aciertos
            productos_recomendados.update(recomendaciones.tolist())
        
        # Calculamos las métricas.
        hit_rate = aciertos / len(muestra)
        precision = total_aciertos / (len(muestra) * 3)
        cobertura = len(productos_recomendados)

        resultados.append({
            "Método": nombre,
            "Hit rate@3": hit_rate,
            "Precisión@3": precision,
            "Cobertura": cobertura
        })

    # Creamos la tabla con los resultados.
    tabla = pd.DataFrame(resultados).set_index("Método")
    st.subheader("Resultados de la comparación")
    st.dataframe(
        tabla.style.format({
            "Hit rate@3": "{:.2%}",
            "Precisión@3": "{:.2%}",
            "Cobertura": "{:.0f}"
        }),
        use_container_width=True
    )
    # Mostramos una gráfica del hit rate.
    st.subheader("Comparación del hit rate@3")
    st.bar_chart(tabla["Hit rate@3"])
    st.caption(
        "Los resultados se calculan con los clientes "
        "seleccionados del grupo de control."
    )
