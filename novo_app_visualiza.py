# analisa_novo_evento.py
# ------------------------------------------------
import re, io, json, requests
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Para PDF e DOCX
import fitz  # PyMuPDF
from docx import Document

st.set_page_config(page_title="Análise ESO — WS • Precursores • Taxonomia • Relatórios", layout="wide")
st.title("🔎 Análise Integrada de Eventos (WS • Precursores • Taxonomia • Relatórios)")

# ====== URLs RAW dos artefatos (ajuste para o seu repositório!)
BASE = "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPO/main/artifacts"
URL_EMB_TAXO = f"{BASE}/emb_taxonomia.parquet"
URL_EMB_PREC = f"{BASE}/emb_precursores.parquet"
URL_EMB_WS   = f"{BASE}/emb_weaksignals.parquet"
URL_EMB_MAPA = f"{BASE}/emb_mapatriplo.parquet"

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=True, ttl=3600)
def read_parquet_from_github(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))

model = load_model()

# ====== Carrega embeddings fixos
try:
    emb_taxo = read_parquet_from_github(URL_EMB_TAXO)
    emb_prec = read_parquet_from_github(URL_EMB_PREC)
    emb_ws   = read_parquet_from_github(URL_EMB_WS)
    emb_mapa = read_parquet_from_github(URL_EMB_MAPA)
except Exception as e:
    st.error(f"Erro ao baixar artefatos (.parquet). Verifique as URLs RAW.\n\n{e}")
    st.stop()

# Separa matrizes numéricas
def split_matrix(df: pd.DataFrame):
    # assume que as colunas de embeddings são todas numéricas após as colunas de metadados
    meta_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype == "O"]
    num_cols = [c for c in df.columns if c not in meta_cols]
    X = df[num_cols].to_numpy(dtype=np.float32)
    return df, X, meta_cols, num_cols

taxo_df, X_taxo, _, _ = split_matrix(emb_taxo)
prec_df, X_prec, _, _ = split_matrix(emb_prec)
ws_df,   X_ws,   _, _ = split_matrix(emb_ws)
mapa_df, X_mapa, _, _ = split_matrix(emb_mapa)

# ====== Utilidades
def clean_ws(name: str) -> str:
    s = str(name).strip()
    return re.sub(r"\s*\(\s*0?\.\d+\s*\)\s*$", "", s)

def safe_sent_tokenize(text: str) -> list[str]:
    text = re.sub(r"\r\n", "\n", text)
    # split por parágrafo + fallback por sentenças
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs if paragraphs else re.split(r'(?<=[.!?])\s+', text)

def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        txt = "\n".join([page.get_text() for page in doc])
        return txt
    elif name.endswith(".docx"):
        tmp = io.BytesIO(file.read())
        doc = Document(tmp)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    else:
        return ""

def topk_sim(query_embs: np.ndarray, base_embs: np.ndarray, k=5):
    sim = cosine_similarity(query_embs, base_embs)  # (n_query, n_base)
    idx = np.argsort(-sim, axis=1)[:, :k]
    val = np.take_along_axis(sim, idx, axis=1)
    return idx, val

st.sidebar.header("Parâmetros")
topk = st.sidebar.slider("Top-K por parágrafo", 3, 20, 7, 1)
thr_ws   = st.sidebar.slider("Limite de similaridade (WS)",   0.0, 1.0, 0.55, 0.01)
thr_prec = st.sidebar.slider("Limite de similaridade (Precursor)", 0.0, 1.0, 0.55, 0.01)
thr_taxo = st.sidebar.slider("Limite de similaridade (Taxonomia)", 0.0, 1.0, 0.55, 0.01)
thr_mapa = st.sidebar.slider("Limite de similaridade (Relatórios)", 0.0, 1.0, 0.60, 0.01)

st.write("Faça upload de um ou mais arquivos **PDF/DOCX/TXT** do evento para análise.")
files = st.file_uploader("Arquivos do evento", type=["pdf","docx","txt"], accept_multiple_files=True)

if not files:
    st.info("Aguardando upload…")
    st.stop()

# ====== Embeddings do documento novo
all_paragraphs = []
for f in files:
    txt = extract_text(f)
    paras = safe_sent_tokenize(txt)
    all_paragraphs += [p for p in paras if p]

if not all_paragraphs:
    st.warning("Não encontrei texto nos arquivos enviados.")
    st.stop()

with st.spinner("Gerando embeddings do(s) documento(s)…"):
    X_doc = model.encode(all_paragraphs, batch_size=64, normalize_embeddings=True)

# ====== MATCHING WS
idx_ws, val_ws = topk_sim(X_doc, X_ws, k=topk)
rows_ws = []
for i, p in enumerate(all_paragraphs):
    for j, (col_idx) in enumerate(idx_ws[i]):
        sim = val_ws[i, j]
        if sim >= thr_ws:
            ws_name = ws_df.iloc[col_idx]["WS"]
            rows_ws.append({"ParagraphID": i, "WeakSignal": ws_name, "Similarity": float(sim), "Snippet": all_paragraphs[i][:300]})
df_ws_hits = pd.DataFrame(rows_ws)

# ====== MATCHING PRECURSORES
idx_prec, val_prec = topk_sim(X_doc, X_prec, k=topk)
rows_prec = []
for i, p in enumerate(all_paragraphs):
    for j, col_idx in enumerate(idx_prec[i]):
        sim = val_prec[i, j]
        if sim >= thr_prec:
            hto = prec_df.iloc[col_idx]["HTO"]
            label = prec_df.iloc[col_idx]["label"]
            rows_prec.append({"ParagraphID": i, "HTO": hto, "Precursor": label, "Similarity": float(sim), "Snippet": all_paragraphs[i][:300]})
df_prec_hits = pd.DataFrame(rows_prec)

# ====== MATCHING TAXONOMIA
idx_taxo, val_taxo = topk_sim(X_doc, X_taxo, k=topk)
rows_taxo = []
for i, p in enumerate(all_paragraphs):
    for j, col_idx in enumerate(idx_taxo[i]):
        sim = val_taxo[i, j]
        if sim >= thr_taxo:
            taxo_text = taxo_df.iloc[col_idx]["text"]
            rows_taxo.append({"ParagraphID": i, "Taxo": taxo_text, "Similarity": float(sim), "Snippet": all_paragraphs[i][:300]})
df_taxo_hits = pd.DataFrame(rows_taxo)

# ====== MATCHING RELATÓRIOS PASSADOS (MapaTriplo)
idx_map, val_map = topk_sim(X_doc, X_mapa, k=topk)
rows_map = []
for i, p in enumerate(all_paragraphs):
    for j, col_idx in enumerate(idx_map[i]):
        sim = val_map[i, j]
        if sim >= thr_mapa:
            r = mapa_df.iloc[col_idx]
            rows_map.append({
                "ParagraphID": i, "HTO": r["HTO"], "Precursor": r["Precursor"],
                "WeakSignal": r["WeakSignal"], "Report": r["Report"],
                "Similarity": float(sim), "TextSimilar": r["Text"][:300],
                "SnippetNew": all_paragraphs[i][:300]
            })
df_mapa_hits = pd.DataFrame(rows_map)

# ====== EXIBIÇÃO
st.subheader("✅ Weak Signals encontrados")
st.dataframe(df_ws_hits.sort_values("Similarity", ascending=False), use_container_width=True, height=300)

st.subheader("✅ Precursores (HTO) encontrados")
st.dataframe(df_prec_hits.sort_values("Similarity", ascending=False), use_container_width=True, height=300)

st.subheader("✅ Itens da TaxonomiaCP relacionados")
st.dataframe(df_taxo_hits.sort_values("Similarity", ascending=False), use_container_width=True, height=300)

st.subheader("🗂️ Relatórios pregressos mais similares (MapaTriplo)")
st.dataframe(df_mapa_hits.sort_values("Similarity", ascending=False), use_container_width=True, height=350)

# Pequeno resumo agregado
st.subheader("📊 Resumo por Precursor e Weak Signal")
if not df_ws_hits.empty and not df_prec_hits.empty:
    # junta por parágrafo (para ver coocorrência simples)
    ws_by_p = df_ws_hits.groupby("ParagraphID")["WeakSignal"].apply(list)
    prec_by_p = df_prec_hits.groupby("ParagraphID")[["HTO","Precursor"]].apply(lambda df: list(df.itertuples(index=False, name=None)))
    pairs = []
    for pid, ws_list in ws_by_p.items():
        if pid in prec_by_p:
            for ws in ws_list:
                for (hto, prec) in prec_by_p[pid]:
                    pairs.append((hto, prec, ws))
    if pairs:
        freq = (pd.DataFrame(pairs, columns=["HTO","Precursor","WeakSignal"])
                .value_counts().reset_index(name="Freq")
                .sort_values("Freq", ascending=False))
        st.dataframe(freq, use_container_width=True, height=260)
    else:
        st.info("Nenhuma coocorrência simples (WS ↔ Precursor) no mesmo parágrafo acima dos limites selecionados.")
else:
    st.info("Sem hits suficientes para montar o resumo.")
