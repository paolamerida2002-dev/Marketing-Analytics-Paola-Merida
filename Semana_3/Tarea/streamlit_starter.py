"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    # Buscamos los datos desde la ubicación del archivo de la app.
    ruta = Path(__file__).resolve().parent / "data" / "uplift_campaign.csv"
    return pd.read_csv(ruta)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    # Usamos los mismos parámetros del notebook.
    parametros = { "n_estimators": 100,"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 50,"random_state": seed}

    # Modelo de respuesta con cupón.
    modelo_tratado = GradientBoostingClassifier(**parametros)
    modelo_tratado.fit( train_t[FEATURES], train_t["responded_60d"])

    # Modelo de respuesta sin cupón.
    modelo_control = GradientBoostingClassifier(**parametros)
    modelo_control.fit(train_c[FEATURES],train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption(
    "Explore qué porcentaje de clientes contactar para "
    "obtener mayor beneficio neto con un cupón de $5."
)

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
        "Porcentaje de clientes a contactar",
        min_value=10,
        max_value=100,
        value=20,
        step=10
    )

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---
# Ordenamos a los clientes por uplift.
orden = test.sort_values(
    "uplift_estimado", ascending=False
).reset_index(drop=True)

es_tratado = (orden["treatment_group"] == "treatment").values
respondio = orden["responded_60d"].values

# Acumulamos respuestas y cantidades de cada grupo.
respuestas_t = np.cumsum(np.where(es_tratado, respondio, 0))
respuestas_c = np.cumsum(np.where(~es_tratado, respondio, 0))
cantidades_t = np.cumsum(es_tratado)
cantidades_c = np.cumsum(~es_tratado)

n = len(orden)
xs, ys = [0.0], [0.0]

for i in np.linspace(1, n, 40).astype(int):
    nt = cantidades_t[i - 1]
    nc = cantidades_c[i - 1]

    tasa_t = respuestas_t[i - 1] / nt if nt > 0 else 0
    tasa_c = respuestas_c[i - 1] / nc if nc > 0 else 0

    xs.append(i / n)
    ys.append((tasa_t - tasa_c) * n)

xs = np.array(xs)
ys = np.array(ys)

# Calculamos Qini con la fórmula del notebook.
area_modelo = np.trapezoid(ys, xs)
area_azar = np.trapezoid(
    np.linspace(0, ys[-1], len(xs)), xs
)
qini = (area_modelo - area_azar) / n

st.subheader("Curva Qini")
st.metric("Qini de Gradient Boosting", f"{qini:+.3f}")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    xs * 100, ys,
    color=GREEN,
    linewidth=3,
    label="T-learner"
)
ax.plot(
    [0, 100], [0, ys[-1]],
    color=GRAY,
    linestyle="--",
    label="Al azar"
)
ax.axvline(
    pct_contactar,
    color=ORANGE,
    linestyle=":",
    label=f"Selección del {pct_contactar}%"
)

ax.set_xlabel("Porcentaje de clientes seleccionados")
ax.set_ylabel("Ganancia de respuesta según el notebook")
ax.legend()

st.pyplot(fig)
plt.close(fig)

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---

# Comparamos las políticas del 10 % al 100 %.
filas = []

for pct in range(10, 101, 10):
    cantidad = int(len(test) * pct / 100)

    seleccionados = test.sort_values(
        "uplift_estimado", ascending=False
    ).head(cantidad).index

    valor_cliente, cantidad = valor_de_la_politica(
        test, seleccionados
    )

    filas.append({
        "Porcentaje": pct,
        "Clientes seleccionados": cantidad,
        "Valor neto por cliente": valor_cliente,
        "Valor neto total": valor_cliente * cantidad
    })

tabla_politicas = pd.DataFrame(filas)

# Mostramos los resultados del porcentaje elegido en el slider.
elegida = tabla_politicas[
    tabla_politicas["Porcentaje"] == pct_contactar
].iloc[0]

st.subheader(f"Resultados al seleccionar el {pct_contactar}%")
st.caption(
    "Los resultados corresponden a los 12,000 clientes de prueba, "
    "no a toda la campaña."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Clientes seleccionados",
    f"{int(elegida['Clientes seleccionados']):,}"
)
col2.metric(
    "Valor neto por cliente",
    f"${elegida['Valor neto por cliente']:,.2f}"
)
col3.metric(
    "Valor neto total",
    f"${elegida['Valor neto total']:,.2f}"
)

st.subheader("Tabla de políticas")
st.dataframe(
    tabla_politicas.style.format({
        "Porcentaje": "{:.0f}%",
        "Clientes seleccionados": "{:,.0f}",
        "Valor neto por cliente": "${:,.2f}",
        "Valor neto total": "${:,.2f}"
    }),
    hide_index=True,
    use_container_width=True
)

st.info(
    "Compare el valor neto total de las políticas antes de "
    "decidir qué porcentaje contactar."
)