# analisa_novo_evento.py
import io, json, re, time, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
import plotly.express as px

# ==== PDF/DOCX readers (opcionais) ====
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

try:
    from docx import Document
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# ==== Embeddings ====
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Análise de Eventos: WS • Precursores • TaxonomiaCP", layout="wide")
st.title("🧭 Análise de Eventos: Weak Signals • Precursores (HTO) • TaxonomiaCP")

# -----------------------------
# Sidebar – de onde vêm os artefatos?
# -----------------------------
st.sidebar.header("Fonte dos artefatos (.parquet / meta.json)")

USE_REMOTE = st.sidebar.checkbox("Ler artefatos via URL (GitHub RAW)", value=True)

REMOTE_BASE = st.sidebar.text_input(
    "Base URL (RAW GitHub) – sem barra no final",
    value=st.secrets.get("ARTIFACTS_BASE", "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main").strip(),
    help="Formato: https://raw.githubusercontent.com/<user>/<repo>/<branch>"
).strip()

REMOTE_PREFIX = st.sidebar.text_input(
    "Subpasta remota (opcional)", value="",
    help="Deixe vazio se os .parquet estão na raiz do repositório. Ex.: 'artifacts'"
).strip().strip("/")

LOCAL_PREFIX = st.sidebar.text_input(
    "Pasta local (opcional)", value="artifacts",
    help="Se você comitou os .parquet no repositório da app, informe a pasta. Ex.: 'artifacts'"
).strip().strip("/")

# st.sidebar.caption("Se preferir, envie os arquivos .parquet / meta.json abaixo:")
# uploads = st.sidebar.file_uploader("Upload (parquet/json)", type=["parquet","json"], accept_multiple_files=True)
# _upload_map = {f.name: f for f in (uploads or [])}

# -----------------------------
# Helpers
# -----------------------------
WS_PAREN_RE = re.compile(r"\s*\((?:0\.\d+|1\.0+)\)\s*$")
def clean_ws_name(s: str) -> str:
    if not isinstance(s, str): return ""
    return WS_PAREN_RE.sub("", s).strip()

def _try_local(name: str) -> pd.DataFrame | None:
    if not LOCAL_PREFIX:
        return None
    p = Path(LOCAL_PREFIX) / name
    try:
        if p.exists():
            return pd.read_parquet(p, engine="pyarrow")
    except Exception as e:
        st.warning(f"[local] falhou ler {p}: {e}")
    return None

def _try_remote(name: str) -> pd.DataFrame | None:
    if not USE_REMOTE or not REMOTE_BASE:
        return None
    base = REMOTE_BASE.rstrip("/")
    url = f"{base}/{name}" if not REMOTE_PREFIX else f"{base}/{REMOTE_PREFIX}/{name}"
    try:
        return pd.read_parquet(url, engine="pyarrow")
    except Exception as e:
        st.warning(f"[remoto] falhou ler {url}: {e}")
    return None

def load_artifact_df(name: str) -> pd.DataFrame:
    # tenta local
    if LOCAL_PREFIX:
        p = Path(LOCAL_PREFIX) / name
        if p.exists():
            return pd.read_parquet(p, engine="pyarrow")
    # tenta remoto
    if USE_REMOTE and REMOTE_BASE:
        base = REMOTE_BASE.rstrip("/")
        url = f"{base}/{name}" if not REMOTE_PREFIX else f"{base}/{REMOTE_PREFIX}/{name}"
        return pd.read_parquet(url, engine="pyarrow")
    raise FileNotFoundError(f"Não encontrei '{name}'. Verifique URL base/pasta local.")

def load_meta() -> dict:
    # local
    if LOCAL_PREFIX:
        p = Path(LOCAL_PREFIX) / "meta.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    # remoto
    if USE_REMOTE and REMOTE_BASE:
        base = REMOTE_BASE.rstrip("/")
        url = f"{base}/meta.json" if not REMOTE_PREFIX else f"{base}/{REMOTE_PREFIX}/meta.json"
        try:
            return json.loads(pd.read_json(url, typ="series").to_json())
        except Exception:
            pass
    # se não existir, segue sem meta
    return {}

def df_has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

def embed_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("e_")]
    M = df[cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return (M / norms).astype(np.float32)

def to_paragraphs(raw_text: str, min_len=25) -> list[tuple[int,str]]:
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

def read_docx_bytes(file_bytes: bytes) -> str:
    if not HAVE_DOCX:
        st.error("Pacote python-docx não instalado.")
        return ""
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)

def stack_matches(sims: np.ndarray, cand_df: pd.DataFrame, label_cols: list[str], thr: float) -> pd.DataFrame:
    hits = np.where(sims >= thr)
    rows = []
    for i, j in zip(*hits):
        r = {"idx_par": int(i), "Similarity": float(sims[i, j])}
        for c in label_cols:
            r[c] = cand_df.iloc[j][c]
        rows.append(r)
    return (pd.DataFrame(rows)
            .sort_values("Similarity", ascending=False)
            .reset_index(drop=True))

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        for sheet, d in dfs.items():
            d.to_excel(w, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return bio.read()

# -----------------------------
# CARREGAR ARTEFATOS
# -----------------------------
with st.spinner("Carregando artefatos…"):
    emb_ws   = load_artifact_df("emb_weaksignals.parquet")
    emb_prec = load_artifact_df("emb_precursores.parquet")
    emb_tax  = load_artifact_df("emb_taxonomia.parquet")
    emb_map  = load_artifact_df("emb_mapatriplo.parquet")
    meta     = load_meta()
    
def normalize_taxonomia_cols(df: pd.DataFrame) -> pd.DataFrame:
    # mapeia variações → padrão
    colmap = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # Dimensão
    for key in ["dimensao", "dimensão", "dimension"]:
        if key in cols_lower:
            colmap[cols_lower[key]] = "Dimensao"
            break

    # Fator
    for key in ["fator", "factor"]:
        if key in cols_lower:
            colmap[cols_lower[key]] = "Fator"
            break

    # Subfator
    for key in ["subfator", "sub-fator", "subfactor", "sub-factor"]:
        if key in cols_lower:
            colmap[cols_lower[key]] = "Subfator"
            break

    # Termos (lista de termos usados no embedding)
    for key in ["_termos", "termos", "termo", "terms", "term"]:
        if key in cols_lower:
            colmap[cols_lower[key]] = "_termos"
            break

    df = df.rename(columns=colmap)

    # garante _text (se faltar, duplica de _termos)
    if "_text" not in df.columns and "_termos" in df.columns:
        df["_text"] = df["_termos"].astype(str)

    return df

# após carregar:
# emb_tax = load_artifact_df("emb_taxonomia.parquet")
emb_tax = normalize_taxonomia_cols(emb_tax)


# sanity check
for name, df, must in [
    ("weaksignals", emb_ws, ["_text","e_0"]),
    ("precursores", emb_prec, ["HTO","Precursor","_text","e_0"]),
    ("taxonomia",  emb_tax, ["Dimensao","Fator","Subfator","_termos","_text","e_0"]),
    ("mapa",       emb_map, ["Report","Text","e_0"]),
]:
    if not df_has_cols(df, must):
        st.error(f"Artefato '{name}' sem colunas esperadas: {must}")
        st.stop()

M_ws   = emb_matrix(emb_ws)
M_prec = emb_matrix(emb_prec)
M_tax  = emb_matrix(emb_tax)
M_map  = emb_matrix(emb_map)

# -----------------------------
# PARÂMETROS
# -----------------------------
st.sidebar.header("Parâmetros de matching")
thr_ws   = st.sidebar.slider("Limiar (Weak Signals)", 0.0, 0.95, 0.50, 0.01)
thr_prec = st.sidebar.slider("Limiar (Precursores)", 0.0, 0.95, 0.50, 0.01)
thr_tax  = st.sidebar.slider("Limiar (TaxonomiaCP)", 0.0, 0.95, 0.55, 0.01)
topk_sim_reports = st.sidebar.slider("Top-N relatórios similares", 3, 20, 8, 1)

# -----------------------------
# UPLOAD DE EVENTOS
# -----------------------------
st.subheader("📎 Faça upload do(s) documento(s) (PDF ou DOCX)")
files = st.file_uploader("Arraste e solte aqui…", type=["pdf","docx"], accept_multiple_files=True)

if not files:
    st.info("Carregue pelo menos um arquivo para iniciar a análise.")
    st.stop()

# -----------------------------
# PROCESSAR ARQUIVOS
# -----------------------------
all_rows = []
with st.spinner("Lendo e extraindo texto…"):
    for f in files:
        name = f.name
        data = f.read()
        if name.lower().endswith(".pdf"):
            raw = read_pdf_bytes(data)
        else:
            raw = read_docx_bytes(data)
        paras = to_paragraphs(raw, min_len=25)
        for par_id, text in paras:
            all_rows.append({"File": name, "Paragraph": par_id, "Text": text})

df_paras = pd.DataFrame(all_rows)
if df_paras.empty:
    st.warning("Não foram encontrados parágrafos válidos.")
    st.stop()

with st.spinner("Gerando embeddings dos parágrafos…"):
    model = load_model("sentence-transformers/all-MiniLM-L6-v2")
    E_doc = embed_texts(model, df_paras["Text"].astype(str).tolist())

# -----------------------------
# MATCHING
# -----------------------------
with st.spinner("Calculando similaridades…"):
    # WS
    S_ws = cosine_similarity(E_doc, M_ws)
    ws_hits = stack_matches(S_ws, emb_ws.rename(columns={"_text":"WeakSignal"}), ["WeakSignal"], thr_ws)
    ws_hits["WeakSignal_clean"] = ws_hits["WeakSignal"].map(clean_ws_name)

    # Precursores
    S_prec = cosine_similarity(E_doc, M_prec)
    prec_hits = stack_matches(S_prec, emb_prec, ["Precursor","HTO"], thr_prec)

    # Taxonomia
    S_tax = cosine_similarity(E_doc, M_tax)
    tax_hits = stack_matches(S_tax, emb_tax, ["Dimensao","Fator","Subfator","_termos"], thr_tax)

    # Relatórios similares (por parágrafo do MapaTriplo)
    S_map = cosine_similarity(E_doc, M_map)
    sim_map_max = S_map.max(axis=0)
    tmp = emb_map[["Report"]].copy()
    tmp["max_sim"] = sim_map_max
    sim_reports = (tmp.groupby("Report", as_index=False)
                     .agg(MaxSim=("max_sim","max"), MeanSim=("max_sim","mean"))
                     .sort_values(["MaxSim","MeanSim"], ascending=False)
                     .head(topk_sim_reports))

def attach_context(df_hits: pd.DataFrame, df_pars: pd.DataFrame) -> pd.DataFrame:
    if df_hits.empty: return df_hits
    out = df_hits.merge(
        df_pars.reset_index(drop=True).reset_index().rename(columns={"index":"idx_par"}),
        on="idx_par", how="left"
    )
    return out.rename(columns={"Text":"Snippet"})

ws_hits   = attach_context(ws_hits, df_paras)
prec_hits = attach_context(prec_hits, df_paras)
tax_hits  = attach_context(tax_hits, df_paras)

# -----------------------------
# VISUALIZAÇÃO
# -----------------------------
st.success(f"Documentos: **{df_paras['File'].nunique()}** | Parágrafos: **{len(df_paras)}**")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Weak Signals (hits)", len(ws_hits))
with c2: st.metric("Precursores (hits)", len(prec_hits))
with c3: st.metric("TaxonomiaCP (hits)", len(tax_hits))

st.subheader("🔎 Weak Signals encontrados")
if ws_hits.empty:
    st.info("Nenhum Weak Signal acima do limiar.")
else:
    ws_freq = (ws_hits.groupby("WeakSignal_clean", as_index=False)
               .agg(Frequencia=("idx_par","count"))
               .sort_values("Frequencia", ascending=False))
    st.dataframe(ws_freq, use_container_width=True)
    st.dataframe(ws_hits[["WeakSignal","Similarity","File","Paragraph","Snippet"]].head(200), use_container_width=True)

st.subheader("🧩 Precursores (HTO) encontrados")
if prec_hits.empty:
    st.info("Nenhum Precursor acima do limiar.")
else:
    prec_freq = (prec_hits.groupby(["HTO","Precursor"], as_index=False)
                 .agg(Frequencia=("idx_par","count"))
                 .sort_values(["HTO","Frequencia"], ascending=[True,False]))
    st.dataframe(prec_freq, use_container_width=True)
    st.dataframe(prec_hits[["HTO","Precursor","Similarity","File","Paragraph","Snippet"]].head(200), use_container_width=True)

st.subheader("📚 TaxonomiaCP (Dimensão/Fator/Subfator) encontrados")
if tax_hits.empty:
    st.info("Nenhum fator da Taxonomia acima do limiar.")
else:
    tax_freq = (tax_hits.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("idx_par","count"))
                .sort_values("Frequencia", ascending=False))
    st.dataframe(tax_freq, use_container_width=True)
    st.dataframe(tax_hits[["Dimensao","Fator","Subfator","_termos","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

st.subheader("🗂️ Relatórios pregressos mais similares")
if sim_reports.empty:
    st.info("Sem similares acima de 0.")
else:
    st.dataframe(sim_reports, use_container_width=True)

# Treemap rápido
if not ws_hits.empty and not prec_hits.empty:
    st.subheader("🌳 Treemap (HTO → Precursor → WeakSignal)")
    tri = (prec_hits[["idx_par","HTO","Precursor"]].drop_duplicates()
           .merge(ws_hits[["idx_par","WeakSignal_clean"]].drop_duplicates(), on="idx_par", how="inner"))
    if not tri.empty:
        tri["value"] = 1
        fig = px.treemap(tri, path=["HTO","Precursor","WeakSignal_clean"], values="value")
        st.plotly_chart(fig, use_container_width=True)

# Download Excel
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

st.caption("Dica: informe corretamente a URL RAW (usuário/repos/branch) ou comite os artefatos em uma pasta local (ex.: artifacts/).")
