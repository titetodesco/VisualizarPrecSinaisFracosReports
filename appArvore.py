import pandas as pd
import streamlit as st
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components

# Carregar diretamente do GitHub (ajuste o link se precisar)
#df = pd.read_csv("https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main/MapaTriplo_tratado.csv")
# URLs dos arquivos no GitHub
df_path = "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main/MapaTriplo_tratado.xlsx"
try:
    df = pd.read_excel(df_path)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# Esperado: colunas [HTO, Precursor, WeakSignal, Report, Text]

st.title("🌳 Visualização Tríplice: HTO → Precursores → Weak Signals")

# ====== Filtro por Report ======
reports = st.multiselect("Filtrar por Report", df["Report"].unique().tolist())
if reports:
    df = df[df["Report"].isin(reports)]

# ====== Treemap Hierárquico ======
st.subheader("Treemap Hierárquico")
fig = px.treemap(
    df,
    path=["HTO", "Precursor", "WeakSignal"],
    values=[1]*len(df),  # cada ocorrência conta como 1
    hover_data=["Report", "Text"],
)
st.plotly_chart(fig, use_container_width=True)

# ====== Grafo Interativo ======
st.subheader("🕸️ Grafo Interativo")

G = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#222222")

# Paleta para HTO
colors = {"Humano": "#2ecc71", "Técnico": "#3498db", "Organizacional": "#f1c40f"}

# Criar nós
for _, row in df.iterrows():
    hto, prec, ws = row["HTO"], row["Precursor"], row["WeakSignal"]
    rep, txt = row["Report"], row["Text"]

    hto_id = f"HTO::{hto}"
    prec_id = f"P::{prec}"
    ws_id = f"WS::{ws}"

    if hto_id not in G.node_ids:
        G.add_node(hto_id, label=hto, color=colors.get(hto, "#9b59b6"), shape="ellipse")

    if prec_id not in G.node_ids:
        G.add_node(prec_id, label=prec, color="#95a5a6", shape="dot")

    if ws_id not in G.node_ids:
        G.add_node(ws_id, label=ws, color="#e74c3c", shape="dot",
                   title=f"<b>Report:</b> {rep}<br><b>Text:</b> {txt}")

    G.add_edge(hto_id, prec_id, color="#7f8c8d")
    G.add_edge(prec_id, ws_id, color="#bdc3c7")

# Renderizar grafo
html_path = "graph_triplo.html"
G.save_graph(html_path)
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()
components.html(html, height=700, scrolling=True)
