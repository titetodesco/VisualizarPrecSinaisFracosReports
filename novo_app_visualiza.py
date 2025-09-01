# analisa_novo_evento.py
import io, json, re, time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ==== PDF / DOCX readers (tenta PyMuPDF; cai para pdfminer se faltar) ====
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    HAVE_PDFMINER = True
except Exception:
    HAVE_PDFMINER = False

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Análise de Eventos: WS • Precursores • TaxonomiaCP", layout="wide")

# Caminho FIXO dos artefatos no GitHub (RAW)
REMOTE_BASE = "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main"

# Modelo de embeddings (o mesmo usado no preparo)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Limiar padrão
DEFAULT_THR = 0.50

# -----------------------------------------------------------------------------
# HELPERS (normalização, leitura, embeddings, utilitários)
# -----------------------------------------------------------------------------
WS_PAREN_RE = re.compile(r"\s*\((?:0\.\d+|1\.0+)\)\s*$")
def clean_ws_name(s: str) -> str:
    """Remove ' (0.53)' do final do WeakSignal, mantendo apenas o texto."""
    if not isinstance(s, str): return ""
    return WS_PAREN_RE.sub("", s).strip()

def _canon(s: str) -> str:
    """Normaliza string para comparação: minúsculas, sem espaços/hífens/underscores."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[\s_\-]+", "", s.strip().lower())

def normalize_taxonomia_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aceita variações como 'Dimensão', 'Sub-fator', 'terms', 'termos', etc.
    Garante colunas-padrão: Dimensao, Fator, Subfator, _termos, _text
    """
    if df is None or df.empty:
        return df

    colmap = {}
    canon = {c: _canon(c) for c in df.columns}

    # Dimensao
    alvo = None
    for c, k in canon.items():
        if k.startswith("dimens"):  # "dimensao", "dimensão"
            alvo = c; break
    if alvo: colmap[alvo] = "Dimensao"

    # Fator
    alvo = None
    for c, k in canon.items():
        if k == "fator" or k == "factor":
            alvo = c; break
    if alvo: colmap[alvo] = "Fator"

    # Subfator
    alvo = None
    for c, k in canon.items():
        if "sub" in k and ("fator" in k or "factor" in k):
            alvo = c; break
    if alvo: colmap[alvo] = "Subfator"

    # _termos (pode vir como "termos", "termo", "terms", "term")
    alvo = None
    for c, k in canon.items():
        if k in {"_termos","termos","termo","terms","term"}:
            alvo = c; break
    if alvo: colmap[alvo] = "_termos"

    df = df.rename(columns=colmap)

    # Completa hierarquia vazia (evita quebrar visualizações)
    for needed in ["Dimensao", "Fator", "Subfator"]:
        if needed not in df.columns:
            df[needed] = ""

    # Garante _termos e _text
    if "_termos" not in df.columns:
        df["_termos"] = (df.get("Dimensao","").astype(str) + " " +
                         df.get("Fator","").astype(str) + " " +
                         df.get("Subfator","").astype(str)).str.strip()
    if "_text" not in df.columns:
        df["_text"] = df["_termos"].astype(str)

    return df

def has_embedding_cols(df: pd.DataFrame) -> bool:
    """Retorna True se existir pelo menos uma coluna 'e_0' (ou prefixo e_)."""
    return any(c.startswith("e_") for c in df.columns)

def load_artifact_df(name: str) -> pd.DataFrame:
    """Lê parquet remoto (GitHub RAW) com pyarrow."""
    url = f"{REMOTE_BASE}/{name}"
    return pd.read_parquet(url, engine="pyarrow")

def load_meta() -> dict:
    url = f"{REMOTE_BASE}/meta.json"
    # lê como texto -> json
    meta_txt = pd.read_json(url, typ="series").to_json()
    return json.loads(meta_txt)

def read_pdf_bytes(file_bytes: bytes) -> str:
    """Leitura robusta de PDF (PyMuPDF se disponível; senão pdfminer)."""
    if HAVE_PYMUPDF:
        try:
            parts = []
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    parts.append(page.get_text("text"))
            return "\n".join(parts)
        except Exception:
            pass
    if HAVE_PDFMINER:
        try:
            return pdfminer_extract(io.BytesIO(file_bytes))
        except Exception:
            pass
    st.error("Nenhum leitor de PDF disponível. Instale `PyMuPDF` (preferível) ou `pdfminer.six`.")
    return ""

def read_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    paras = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            paras.append(t)
    return "\n".join(paras)

def to_paragraphs(raw_text: str, min_len=25) -> List[tuple[int,str]]:
    """Quebra o texto em blocos (parágrafos) minimamente longos."""
    chunks, buf = [], []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            if buf:
                block = " ".join(buf).strip()
                if len(block) >= min_len:
                    chunks.append(block)
                buf = []
        else:
            buf.append(line)
    if buf:
        block = " ".join(buf).strip()
        if len(block) >= min_len:
            chunks.append(block)
    return [(i+1, ch) for i, ch in enumerate(chunks)]

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

def embed_texts(model, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM-L6-v2 -> 384 dims
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("e_")]
    M = df[cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return (M / norms).astype(np.float32)

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        for sheet, d in dfs.items():
            d.to_excel(w, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return bio.read()

def stack_matches(sims: np.ndarray, cand_df: pd.DataFrame, label_cols: list[str], thr: float) -> pd.DataFrame:
    """
    sims: (n_paragraphs x n_candidates)
    cand_df: DataFrame com colunas label_cols (ex.: ["WeakSignal"])
    thr: limiar
    Retorna: DataFrame com colunas ["idx_par","Similarity"] + label_cols (mesmo se vazio)
    """
    hits = np.where(sims >= thr)

    # Sem hits? Retorne DF vazio com colunas esperadas (evita KeyError no sort_values)
    if hits[0].size == 0:
        cols = ["idx_par", "Similarity"] + label_cols
        return pd.DataFrame(columns=cols)

    rows = []
    for i, j in zip(*hits):
        r = {"idx_par": int(i), "Similarity": float(sims[i, j])}
        for c in label_cols:
            r[c] = cand_df.iloc[j][c]
        rows.append(r)

    df = pd.DataFrame(rows)
    return df.sort_values("Similarity", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# CARREGA ARTEFATOS (remoto)
# -----------------------------------------------------------------------------
st.title("🧭 Análise de Eventos: Weak Signals • Precursores (HTO) • TaxonomiaCP")

with st.spinner("Carregando artefatos (embeddings de dicionários e mapa)…"):
    emb_ws   = load_artifact_df("emb_weaksignals.parquet")
    emb_prec = load_artifact_df("emb_precursores.parquet")
    emb_tax  = load_artifact_df("emb_taxonomia.parquet")
    emb_map  = load_artifact_df("emb_mapatriplo.parquet")
    meta     = load_meta()

# Normalizações leves nos artefatos
if "_text" not in emb_ws.columns:
    # tenta achar uma coluna de texto principal
    cand = [c for c in emb_ws.columns if c.lower() in {"_text","weaksignal","term","termo","termos"}]
    if cand:
        emb_ws = emb_ws.rename(columns={cand[0]: "_text"})
    else:
        st.error("Artefato 'weaksignals' sem coluna de texto principal. Refaça o preparo.")
        st.stop()

if "_text" not in emb_prec.columns:
    if "Precursor" in emb_prec.columns:
        emb_prec["_text"] = emb_prec["Precursor"].astype(str)
    else:
        st.error("Artefato 'precursores' sem coluna 'Precursor'. Refaça o preparo.")
        st.stop()
if "HTO" not in emb_prec.columns:
    emb_prec["HTO"] = ""  # evita quebrar

def _norm_tax_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cabeçalhos vindos do parquet/Excel para: Dimensao, Fator, Subfator, _termos."""
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"dimensão", "dimensao"}:            rename_map[c] = "Dimensao"
        elif cl in {"fatores", "fator"}:              rename_map[c] = "Fator"
        elif cl in {"subfator", "subfator 1"}:        rename_map[c] = "Subfator"
        elif cl in {"_termos","termos","bag de termos","bag of terms"}:
            rename_map[c] = "_termos"
        elif c == "_text":                            rename_map[c] = "_text"
    df = df.rename(columns=rename_map)
    for col in ["Dimensao","Fator","Subfator","_termos"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# emb_tax = normalize_taxonomia_cols(emb_tax)
emb_tax = _norm_tax_headers(emb_tax)

if "_text" not in emb_map.columns and "Text" in emb_map.columns:
    emb_map["_text"] = emb_map["Text"].astype(str)

# Garante que há colunas de embeddings
probs = []
for name, df in [("weaksignals", emb_ws), ("precursores", emb_prec), ("taxonomia", emb_tax), ("mapatriplo", emb_map)]:
    if not has_embedding_cols(df):
        probs.append(name)
if probs:
    st.error("Artefatos sem colunas de embeddings (e_0, e_1, ...): " + ", ".join(probs))
    st.stop()

# Monta matrizes
M_ws   = emb_matrix(emb_ws)
M_prec = emb_matrix(emb_prec)
M_tax  = emb_matrix(emb_tax)
M_map  = emb_matrix(emb_map)

# -----------------------------------------------------------------------------
# SIDEBAR – parâmetros
# -----------------------------------------------------------------------------
st.sidebar.header("Parâmetros")
thr_ws   = st.sidebar.slider("Limiar (Weak Signals)", 0.0, 0.95, DEFAULT_THR, 0.01)
thr_prec = st.sidebar.slider("Limiar (Precursores)", 0.0, 0.95, DEFAULT_THR, 0.01)
thr_tax  = st.sidebar.slider("Limiar (TaxonomiaCP)", 0.0, 0.95, 0.55, 0.01)
topk_sim_reports = st.sidebar.slider("Top-N relatórios similares", 3, 20, 8, 1)

# -----------------------------------------------------------------------------
# UPLOAD DE ARQUIVOS
# -----------------------------------------------------------------------------
st.subheader("📎 Faça upload do(s) documento(s) do evento (PDF ou DOCX)")
files = st.file_uploader("Arraste e solte aqui…", type=["pdf","docx"], accept_multiple_files=True)

if not files:
    st.info("Carregue pelo menos um arquivo para iniciar a análise.")
    st.stop()

# -----------------------------------------------------------------------------
# PROCESSA ARQUIVOS (parágrafos + embeddings)
# -----------------------------------------------------------------------------
model = load_model(MODEL_NAME)

all_rows = []
with st.spinner("Lendo e extraindo texto…"):
    for f in files:
        name = f.name
        data = f.read()
        if name.lower().endswith(".pdf"):
            raw = read_pdf_bytes(data)
        else:
            raw = read_docx(data)
        paras = to_paragraphs(raw, min_len=25)
        for par_id, text in paras:
            all_rows.append({"File": name, "Paragraph": par_id, "Text": text})

df_paras = pd.DataFrame(all_rows)
if df_paras.empty:
    st.warning("Não foram encontrados parágrafos válidos nos arquivos.")
    st.stop()

with st.spinner("Gerando embeddings dos parágrafos…"):
    E_doc = embed_texts(model, df_paras["Text"].astype(str).tolist())

# -----------------------------------------------------------------------------
# MATCHING (WS, Precursores, Taxonomia, Relatórios similares)
# -----------------------------------------------------------------------------
with st.spinner("Calculando similaridades…"):
    # Weak Signals
    S_ws = cosine_similarity(E_doc, M_ws)  # (P x W)
    ws_hits = stack_matches(S_ws, emb_ws.rename(columns={"_text":"WeakSignal"}), ["WeakSignal"], thr_ws)
    ws_hits["WeakSignal_clean"] = ws_hits["WeakSignal"].map(clean_ws_name)

    # Precursores
    S_prec = cosine_similarity(E_doc, M_prec)  # (P x Pprec)
    prec_hits = stack_matches(S_prec, emb_prec, ["Precursor","HTO"], thr_prec)

    # Taxonomia
    S_tax = cosine_similarity(E_doc, M_tax)  # (P x T)
    tax_hits = stack_matches(S_tax, emb_tax, ["Dimensao","Fator","Subfator","_termos"], thr_tax)

    # Relatórios similares (usa texto dos parágrafos vs MapaTriplo.Text)
    S_map = cosine_similarity(E_doc, M_map)   # (P x MapRows)
    sim_map_max = S_map.max(axis=0)
    emb_map_reports = emb_map[["Report"]].copy()
    emb_map_reports["max_sim"] = sim_map_max
    sim_reports = (emb_map_reports.groupby("Report", as_index=False)
                   .agg(MaxSim=("max_sim","max"), MeanSim=("max_sim","mean"))
                   .sort_values(["MaxSim","MeanSim"], ascending=False)
                   .head(topk_sim_reports))

def attach_context(df_hits: pd.DataFrame, df_pars: pd.DataFrame) -> pd.DataFrame:
    if df_hits.empty:
        return df_hits
    out = df_hits.merge(
        df_pars.reset_index(drop=True).reset_index().rename(columns={"index":"idx_par"}),
        on="idx_par", how="left"
    )
    return out.rename(columns={"Text":"Trecho"})

ws_hits   = attach_context(ws_hits, df_paras)
prec_hits = attach_context(prec_hits, df_paras)
tax_hits  = attach_context(tax_hits, df_paras)

# ------------------------------------------------------------------
# VISUALIZAÇÃO (Única) — WS, Precursores, Taxonomia (com saneamento)
# ------------------------------------------------------------------
st.success(f"Documentos processados: **{df_paras['File'].nunique()}** | Parágrafos: **{len(df_paras)}**")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Weak Signals (hits)", len(ws_hits))
with c2: st.metric("Precursores (hits)", len(prec_hits))
with c3: st.metric("TaxonomiaCP (hits)", len(tax_hits))

# ----------------- WS -----------------
st.subheader("🔎 Weak Signals encontrados")
if ws_hits.empty:
    st.info("Nenhum Weak Signal acima do limiar.")
else:
    ws_freq = (ws_hits.groupby("WeakSignal_clean", as_index=False)
               .agg(Frequencia=("idx_par","count"))
               .sort_values("Frequencia", ascending=False))
    st.dataframe(ws_freq, use_container_width=True)
    st.dataframe(ws_hits[["WeakSignal","Similarity","File","Paragraph","Trecho"]]
                 .sort_values("Similarity", ascending=False)
                 .head(200), use_container_width=True)

# -------------- PRECURSORES --------------
st.subheader("🧩 Precursores (HTO) encontrados")
if prec_hits.empty:
    st.info("Nenhum Precursor acima do limiar.")
else:
    prec_freq = (prec_hits.groupby(["HTO","Precursor"], as_index=False)
                 .agg(Frequencia=("idx_par","count"))
                 .sort_values(["HTO","Frequencia"], ascending=[True,False]))
    st.dataframe(prec_freq, use_container_width=True)
    st.dataframe(prec_hits[["HTO","Precursor","Similarity","File","Paragraph","Trecho"]]
                 .sort_values("Similarity", ascending=False)
                 .head(200), use_container_width=True)

# ---------------- TAXONOMIA ----------------
st.markdown("## 🧩 Visualizações — TaxonomiaCP (Dimensão → Fator → Subfator)")

if tax_hits.empty:
    st.info("Nenhum fator da Taxonomia acima do limiar.")
else:
    # 1) normaliza headers e valores
    tax_hits = _norm_tax_headers(tax_hits)
    for col in ["Dimensao","Fator","Subfator","_termos"]:
        tax_hits[col] = (tax_hits[col].astype(str)
                         .str.strip()
                         .replace({"": np.nan, "None": np.nan, "nan": np.nan}))

    # 2) reconstrói Fator ausente a partir do parquet (Subfator → Fator)
    sub2fac = (emb_tax[["Subfator","Fator"]]
               .dropna()
               .drop_duplicates())
    sub2fac_map = dict(zip(sub2fac["Subfator"], sub2fac["Fator"]))

    tax_hits["Fator"] = tax_hits["Fator"].fillna(tax_hits["Subfator"].map(sub2fac_map))
    tax_hits["Dimensao"] = tax_hits["Dimensao"].fillna("—")
    tax_hits["Fator"]    = tax_hits["Fator"].fillna("—")
    tax_hits["Subfator"] = tax_hits["Subfator"].fillna("—")

    # 3) Tabela ÚNICA de frequência (Dimensão/Fator/Subfator)
    tax_freq = (tax_hits.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("idx_par","count"))
                .sort_values(["Dimensao","Fator","Frequencia"], ascending=[True,True,False]))

    
    st.subheader("📚 TaxonomiaCP (Dimensão/Fator/Subfator) encontrados")

if tax_hits.empty:
    st.info("Nenhum fator da Taxonomia acima do limiar.")
else:
    # Normaliza cabeçalhos e valores
    tax_hits = _norm_tax_headers(tax_hits).copy()
    for col in ["Dimensao","Fator","Subfator","_termos"]:
        tax_hits[col] = (tax_hits[col].astype(str)
                         .str.strip()
                         .replace({"": np.nan, "None": np.nan, "nan": np.nan}))

    # --- mapeia Subfator -> Fator e Subfator -> Dimensao usando o parquet (fonte da verdade)
    sub2fac_map = (emb_tax[["Subfator","Fator"]]
                   .dropna(subset=["Subfator"])
                   .drop_duplicates()
                   .set_index("Subfator")["Fator"].to_dict())

    sub2dim_map = (emb_tax[["Subfator","Dimensao"]]
                   .dropna(subset=["Subfator"])
                   .drop_duplicates()
                   .set_index("Subfator")["Dimensao"].to_dict())

    # --- Reconstrói Fator/Dimensao quando vierem nulos nos hits (ocorre em match por termo)
    tax_hits["Fator"]    = tax_hits["Fator"].fillna(tax_hits["Subfator"].map(sub2fac_map))
    tax_hits["Dimensao"] = tax_hits["Dimensao"].fillna(tax_hits["Subfator"].map(sub2dim_map))

    # Preenche qualquer resto que tenha ficado vazio (depois do mapeamento)
    tax_hits["Fator"]    = tax_hits["Fator"].fillna("—")
    tax_hits["Dimensao"] = tax_hits["Dimensao"].fillna("—")
    tax_hits["Subfator"] = tax_hits["Subfator"].fillna("—")

    # Base saneada para tudo
    tax_hits_norm = tax_hits.copy()

    # Frequência ÚNICA (sempre no trio Dimensao/Fator/Subfator)
    tax_freq = (tax_hits_norm.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("idx_par","count"))
                .sort_values(["Dimensao","Fator","Frequencia"], ascending=[True,True,False]))

    st.dataframe(tax_freq, use_container_width=True)

    # Amostra dos matches
    st.dataframe(
        tax_hits_norm[["Dimensao","Fator","Subfator","_termos","Similarity","File","Paragraph","Trecho"]]
        .sort_values("Similarity", ascending=False)
        .head(200),
        use_container_width=True
    )

    # A partir daqui, seus gráficos podem usar tax_hits_norm
    tax_plot = tax_hits_norm.copy()
    tax_plot["value"] = 1

    # Proteção: se todo mundo vira "—", evita erro do Plotly
    if tax_plot[["Dimensao","Fator","Subfator"]].nunique().sum() <= 3:
        st.info("Taxonomia com muitos campos vazios. Ajuste os limiares ou verifique os dados.")
    else:
        st.subheader("🌳 Treemap (Dimensão → Fator → Subfator)")
        fig_tax_tree = px.treemap(
            tax_plot,
            path=["Dimensao","Fator","Subfator"],
            values="value",
            hover_data=["_termos","Similarity","File"],
            title="Treemap da TaxonomiaCP encontrada"
        )
        st.plotly_chart(fig_tax_tree, use_container_width=True)
        

        st.subheader("🌞 Sunburst (Dimensão → Fator → Subfator)")
        fig_tax_sun = px.sunburst(
            tax_plot, path=["Dimensao","Fator","Subfator"],
            values="value",
            hover_data=["_termos","Similarity","File"],
            title="Sunburst da TaxonomiaCP encontrada"
        )
        st.plotly_chart(fig_tax_sun, use_container_width=True)

        st.subheader("🏷️ Top Subfatores por frequência")
        sub_rank = (tax_plot.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                    .agg(Frequencia=("value","sum"))
                    .sort_values("Frequencia", ascending=False)
                    .head(20))
        fig_sub_bar = px.bar(
            sub_rank, x="Frequencia", y="Subfator",
            color="Dimensao", orientation="h",
            hover_data=["Fator"],
            title="Top Subfatores (doc atual)"
        )
        st.plotly_chart(fig_sub_bar, use_container_width=True)

        st.subheader("🔥 Heatmap — Dimensão × Fator")
        df_hm = (tax_plot.groupby(["Dimensao","Fator"], as_index=False)
                 .agg(Qtd=("Subfator","nunique")))
        mat_tax = (df_hm
                   .pivot(index="Fator", columns="Dimensao", values="Qtd")
                   .fillna(0).astype(int))
        if not mat_tax.empty:
            fig_tax_hm = px.imshow(
                mat_tax.values,
                labels=dict(x="Dimensão", y="Fator", color="Qtd Subfatores"),
                x=mat_tax.columns.tolist(),
                y=mat_tax.index.tolist(),
                title="Qtd de Subfatores por Dimensão × Fator"
            )
            st.plotly_chart(fig_tax_hm, use_container_width=True)

# ---------------- RELATÓRIOS SIMILARES ----------------
st.subheader("🗂️ Relatórios pregressos mais similares")
if sim_reports.empty:
    st.info("Sem similares acima de 0.")
else:
    st.dataframe(sim_reports, use_container_width=True)


# -----------------------------------------------------------------------------
# DOWNLOAD EXCEL
# -----------------------------------------------------------------------------
st.subheader("⬇️ Download (Excel consolidado)")
dfs_out = {
    "WS_hits": ws_hits,
    "Precursores_hits": prec_hits,
    "Taxonomia_hits": tax_hits,
    "Relatorios_similares": sim_reports
}
st.download_button(
    "Baixar resultados (.xlsx)",
    data=to_excel_bytes(dfs_out),
    file_name=f"analise_evento_{int(time.time())}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Use os limiares na barra lateral para equilibrar cobertura vs. precisão. Os artefatos são carregados do repositório público informado.")
