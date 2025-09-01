# analisa_novo_evento.py
import io, json, re, time, numpy as np, pandas as pd, streamlit as st
from pathlib import Path

import plotly.express as px

# substitua `import fitz` por:
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

# fallback opcional com pdfminer (se quiser)
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

def read_pdf_bytes(file_bytes: bytes) -> str:
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
    st.error("Nenhum leitor de PDF disponível. Instale `PyMuPDF` ou `pdfminer.six`.")
    return ""

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Análise de Eventos: WS • Precursores • TaxonomiaCP", layout="wide")

# Onde estão os artefatos (.parquet/.json)?
# 1) Local: ./artifacts
# 2) ou hospedados (RAW) no GitHub
USE_REMOTE = st.sidebar.checkbox("Carregar artefatos via URL (GitHub RAW)?", value=False)

REMOTE_BASE = st.sidebar.text_input(
    "Base URL (se usar remoto)", 
    "https://raw.githubusercontent.com/titetodesco/main"
)

ART = Path("main")
ART.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Limiar padrão
DEFAULT_THR = 0.50

# -----------------------------
# HELPERS
# -----------------------------
WS_PAREN_RE = re.compile(r"\s*\((?:0\.\d+|1\.0+)\)\s*$")
def clean_ws_name(s: str) -> str:
    if not isinstance(s, str): return ""
    return WS_PAREN_RE.sub("", s).strip()

def load_parquet(name: str) -> pd.DataFrame:
    if USE_REMOTE:
        url = f"{REMOTE_BASE}/{name}"
        return pd.read_parquet(url, engine="pyarrow")
    else:
        return pd.read_parquet(ART / name)

def load_meta() -> dict:
    if USE_REMOTE:
        url = f"{REMOTE_BASE}/meta.json"
        return json.loads(pd.read_json(url, typ="series").to_json())
    else:
        return json.loads((ART / "meta.json").read_text(encoding="utf-8"))

# leitura segura de PDF/DOCX
def read_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            try:
                text_parts.append(page.get_text("text"))
            except Exception:
                continue
    return "\n".join(text_parts)

def read_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    paras = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            paras.append(t)
    return "\n".join(paras)

# quebra em parágrafos “seguros”
def to_paragraphs(raw_text: str, min_len=25) -> list[tuple[int,str]]:
    # quebra em linhas e agrupa blocos
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
    # indexa
    return [(i+1, ch) for i, ch in enumerate(chunks)]

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

def embed_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM-L6-v2 = 384 dims
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, b)

def df_has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        for sheet, d in dfs.items():
            d.to_excel(w, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return bio.read()

# -----------------------------
# CARREGA ARTEFATOS
# -----------------------------
st.title("🧭 Análise de Eventos: Weak Signals • Precursores (HTO) • TaxonomiaCP")

with st.spinner("Carregando artefatos (embeddings de dicionários e mapa)…"):
    emb_ws = load_parquet("emb_weaksignals.parquet")
    emb_prec = load_parquet("emb_precursores.parquet")
    emb_tax = load_parquet("emb_taxonomia.parquet")
    emb_map = load_parquet("emb_mapatriplo.parquet")
    meta = load_meta()

# checa colunas
for name, df, must in [
    ("emb_weaksignals.parquet", emb_ws, ["_text", "e_0"]),
    ("emb_precursores.parquet", emb_prec, ["_text", "HTO", "Precursor", "e_0"]),
    ("emb_taxonomia.parquet", emb_tax, ["Dimensao","Fator","Subfator","_termos","_text","e_0"]),
    ("emb_mapatriplo.parquet", emb_map, ["Report","Text","e_0"]),
]:
    if not df_has_cols(df, must):
        st.error(f"Arquivo {name} não possui colunas esperadas: {must}")
        st.stop()

# monta matrizes
def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("e_")]
    M = df[cols].to_numpy(dtype=np.float32)
    # garante normalizados (caso artefatos antigos)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return (M / norms).astype(np.float32)

M_ws   = emb_matrix(emb_ws)
M_prec = emb_matrix(emb_prec)
M_tax  = emb_matrix(emb_tax)
M_map  = emb_matrix(emb_map)

# -----------------------------
# SIDEBAR – parâmetros
# -----------------------------
st.sidebar.header("Parâmetros")
thr_ws   = st.sidebar.slider("Limiar (Weak Signals)", 0.0, 0.95, DEFAULT_THR, 0.01)
thr_prec = st.sidebar.slider("Limiar (Precursores)", 0.0, 0.95, DEFAULT_THR, 0.01)
thr_tax  = st.sidebar.slider("Limiar (TaxonomiaCP)", 0.0, 0.95, 0.55, 0.01)
topk_sim_reports = st.sidebar.slider("Top-N relatórios similares", 3, 20, 8, 1)

# -----------------------------
# UPLOAD
# -----------------------------
st.subheader("📎 Faça upload do(s) documento(s) do evento (PDF ou DOCX)")
files = st.file_uploader("Arraste e solte aqui…", type=["pdf","docx"], accept_multiple_files=True)

if not files:
    st.info("Carregue pelo menos um arquivo para iniciar a análise.")
    st.stop()

# -----------------------------
# PROCESSA ARQUIVOS
# -----------------------------
model = load_model(MODEL_NAME)

all_rows = []  # para compor DataFrame de parágrafos
with st.spinner("Lendo e extraindo texto…"):
    for f in files:
        name = f.name
        data = f.read()
        if name.lower().endswith(".pdf"):
            raw = read_pdf(data)
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

# -----------------------------
# MATCHING (WS, Precursores, Taxonomia, Relatórios similares)
# -----------------------------
def stack_matches(sims: np.ndarray, cand_df: pd.DataFrame, label_cols: list[str], thr: float) -> pd.DataFrame:
    """
    sims: (n_paragraphs x n_candidates)
    cand_df: DataFrame com colunas label_cols (e.g., ["WeakSignal"])
    thr: limiar
    retorna: DataFrame com linhas acima do limiar
    """
    hits = np.where(sims >= thr)
    rows = []
    for i, j in zip(*hits):
        r = { "idx_par": int(i), "Similarity": float(sims[i, j]) }
        for c in label_cols:
            r[c] = cand_df.iloc[j][c]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("Similarity", ascending=False).reset_index(drop=True)

with st.spinner("Calculando similaridades…"):
    # WS
    S_ws = cosine_similarity(E_doc, M_ws)  # (P x W)
    ws_hits = stack_matches(S_ws, emb_ws.rename(columns={"_text":"WeakSignal"}), ["WeakSignal"], thr_ws)
    ws_hits["WeakSignal_clean"] = ws_hits["WeakSignal"].map(clean_ws_name)
    # Precursores
    S_prec = cosine_similarity(E_doc, M_prec)  # (P x Pprec)
    prec_hits = stack_matches(S_prec, emb_prec, ["Precursor","HTO"], thr_prec)
    # Taxonomia
    S_tax = cosine_similarity(E_doc, M_tax)  # (P x T)
    tax_hits = stack_matches(S_tax, emb_tax, ["Dimensao","Fator","Subfator","_termos"], thr_tax)

    # Relatórios similares (usa o texto do parágrafo vs MapaTriplo.Text)
    S_map = cosine_similarity(E_doc, M_map)   # (P x MapRows)
    sim_map_max = S_map.max(axis=0)           # quão “coberto” cada parágrafo do Mapa fica por este documento
    # alternativa: usar agregação por Report
    emb_map_reports = emb_map[["Report"]].copy()
    emb_map_reports["max_sim"] = sim_map_max
    sim_reports = (emb_map_reports.groupby("Report", as_index=False)
                   .agg(MaxSim=("max_sim","max"), MeanSim=("max_sim","mean"))
                   .sort_values(["MaxSim","MeanSim"], ascending=False)
                   .head(topk_sim_reports))

# anexa Snippet ao ws/prec/tax
def attach_context(df_hits: pd.DataFrame, df_pars: pd.DataFrame) -> pd.DataFrame:
    if df_hits.empty:
        return df_hits
    out = df_hits.merge(
        df_pars.reset_index(drop=True).reset_index().rename(columns={"index":"idx_par"}),
        on="idx_par", how="left"
    )
    return out.rename(columns={"Text":"Snippet"})

ws_hits = attach_context(ws_hits, df_paras)
prec_hits = attach_context(prec_hits, df_paras)
tax_hits = attach_context(tax_hits, df_paras)

# -----------------------------
# VISUALIZAÇÃO
# -----------------------------
st.success(f"Documentos processados: **{df_paras['File'].nunique()}** | Parágrafos: **{len(df_paras)}**")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Weak Signals (hits)", len(ws_hits))
with c2: st.metric("Precursores (hits)", len(prec_hits))
with c3: st.metric("TaxonomiaCP (hits)", len(tax_hits))

st.subheader("🔎 Weak Signals encontrados")
if ws_hits.empty:
    st.info("Nenhum Weak Signal acima do limiar.")
else:
    # agrupar para visão rápida
    ws_freq = (ws_hits.groupby("WeakSignal_clean", as_index=False)
               .agg(Frequencia=("idx_par","count")))
    ws_freq = ws_freq.sort_values("Frequencia", ascending=False)
    st.dataframe(ws_freq, use_container_width=True)

    st.dataframe(ws_hits[["WeakSignal","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

st.subheader("🧩 Precursores (HTO) encontrados")
if prec_hits.empty:
    st.info("Nenhum Precursor acima do limiar.")
else:
    prec_freq = (prec_hits.groupby(["HTO","Precursor"], as_index=False)
                 .agg(Frequencia=("idx_par","count")))
    prec_freq = prec_freq.sort_values(["HTO","Frequencia"], ascending=[True,False])
    st.dataframe(prec_freq, use_container_width=True)
    st.dataframe(prec_hits[["HTO","Precursor","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

st.subheader("📚 TaxonomiaCP (Dimensão/Fator/Subfator) encontrados")
if tax_hits.empty:
    st.info("Nenhum fator da Taxonomia acima do limiar.")
else:
    tax_freq = (tax_hits.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("idx_par","count")))
    tax_freq = tax_freq.sort_values("Frequencia", ascending=False)
    st.dataframe(tax_freq, use_container_width=True)
    st.dataframe(tax_hits[["Dimensao","Fator","Subfator","_termos","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

st.subheader("🗂️ Relatórios pregressos mais similares")
if sim_reports.empty:
    st.info("Sem similares acima de 0.")
else:
    st.dataframe(sim_reports, use_container_width=True)

# Treemap rápido (opcional)
if not prec_hits.empty and not ws_hits.empty:
    st.subheader("🌳 Treemap (HTO → Precursor → WeakSignal)")
    # junta para contar tríades por parágrafo
    join_ws = ws_hits[["idx_par","WeakSignal_clean"]].drop_duplicates()
    join_prec = prec_hits[["idx_par","HTO","Precursor"]].drop_duplicates()
    tri = join_prec.merge(join_ws, on="idx_par", how="inner")
    if not tri.empty:
        tri["value"] = 1
        fig = px.treemap(
            tri,
            path=["HTO","Precursor","WeakSignal_clean"],
            values="value"
        )
        st.plotly_chart(fig, use_container_width=True)

# Árvore simples (texto) só para navegação rápida
with st.expander("Árvore colapsável (texto resumido)"):
    if not tri.empty:
        # constrói listagem em texto (compacta)
        for hto in sorted(tri["HTO"].unique()):
            st.markdown(f"**{hto}**")
            tri_h = tri[tri["HTO"]==hto]
            for prec in sorted(tri_h["Precursor"].unique()):
                st.markdown(f"- {prec}")
                ws_list = sorted(tri_h[tri_h["Precursor"]==prec]["WeakSignal_clean"].unique().tolist())
                if ws_list:
                    st.markdown("  - " + "; ".join(ws_list[:30]))

# -----------------------------
# DOWNLOAD EXCEL
# -----------------------------
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

st.caption("Dica: ajuste os limiares na barra lateral para controlar ruído vs. cobertura. Os embeddings de dicionários e mapa vêm dos artefatos gerados no “bloco 1”.")
