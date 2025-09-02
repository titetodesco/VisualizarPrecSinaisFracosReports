# novo_app_visualiza.py
# App Streamlit: análise de eventos com embeddings + árvore interativa HTO → Precursor → Weak Signal

import io, json, re, time, requests
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- libs opcionais p/ leitura de PDF/DOCX ----------
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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_echarts import st_echarts


# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Análise de Eventos • WS • Precursores • TaxonomiaCP", layout="wide")

# Base dos artefatos NO GITHUB (já faça upload dos .parquet e meta.json nesta pasta)
REMOTE_BASE = "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main"

# Modelo de embeddings (o mesmo usado para gerar os artefatos)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Limiar padrão
DEFAULT_THR = 0.50


# ==============================
# HELPERS
# ==============================
WS_PAREN_RE = re.compile(r"\s*\((?:0\.\d+|1\.0+)\)\s*$")

def clean_ws_tail(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return WS_PAREN_RE.sub("", s).strip()

def http_bytes(url: str) -> io.BytesIO:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return io.BytesIO(r.content)

def load_parquet_remote(name: str) -> pd.DataFrame:
    # usa pyarrow/fsspec internamente via pandas
    url = f"{REMOTE_BASE}/{name}"
    try:
        return pd.read_parquet(url, engine="pyarrow")
    except Exception:
        # fallback por bytes
        bio = http_bytes(url)
        return pd.read_parquet(bio, engine="pyarrow")

def load_meta_remote() -> dict:
    url = f"{REMOTE_BASE}/meta.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return json.loads(r.text)

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
    # tenta PyMuPDF, depois pdfminer
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
    return ""

def read_docx_bytes(file_bytes: bytes) -> str:
    if not HAVE_DOCX:
        return ""
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    paras = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            paras.append(t)
    return "\n".join(paras)

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

def embed_texts(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM-L6-v2
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

def emb_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("e_")]
    M = df[cols].to_numpy(dtype=np.float32)
    # normaliza por garantia
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return (M / norms).astype(np.float32)

def df_has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

def stack_matches(sims: np.ndarray, cand_df: pd.DataFrame, label_cols: list[str], thr: float) -> pd.DataFrame:
    hits = np.where(sims >= thr)
    rows = []
    for i, j in zip(*hits):
        r = {"idx_par": int(i), "Similarity": float(sims[i, j])}
        for c in label_cols:
            r[c] = cand_df.iloc[j][c]
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["idx_par","Similarity"] + label_cols)
    return pd.DataFrame(rows).sort_values("Similarity", ascending=False).reset_index(drop=True)

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        for sheet, d in dfs.items():
            d.to_excel(w, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return bio.read()

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in {"HTO","Precursor","WS"} else None
    return df


# ==============================
# CARREGA ARTEFATOS
# ==============================
st.title("🧭 Análise de Eventos: Weak Signals • Precursores (HTO) • TaxonomiaCP")

with st.spinner("Carregando artefatos (embeddings de dicionários e mapa)…"):
    emb_ws   = load_parquet_remote("emb_weaksignals.parquet")
    emb_prec = load_parquet_remote("emb_precursores.parquet")
    emb_tax  = load_parquet_remote("emb_taxonomia.parquet")
    emb_map  = load_parquet_remote("emb_mapatriplo.parquet")
    meta     = load_meta_remote()

# --- Normalização de colunas do emb_taxonomia ---
def normalize_emb_tax_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    # mapa de variantes -> destino
    # (usamos lower() para casar robusto)
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in lower_map:
                return lower_map[c]
        return None

    col_dim  = pick("dimensao", "dimensão", "dimension")
    col_fat  = pick("fator", "fatores", "factor", "factors")
    col_sub  = pick("subfator", "subfator 1", "sub-fator", "subfactor")
    col_bag  = pick("_termos", "bag de termos", "bag of terms", "termos", "terms")

    rename_map = {}
    if col_dim and col_dim != "Dimensao":
        rename_map[col_dim] = "Dimensao"
    if col_fat and col_fat != "Fator":
        rename_map[col_fat] = "Fator"
    if col_sub and col_sub != "Subfator":
        rename_map[col_sub] = "Subfator"
    if col_bag and col_bag != "_termos":
        rename_map[col_bag] = "_termos"

    if rename_map:
        df = df.rename(columns=rename_map)

    # garante presença mínima
    for need in ["Dimensao", "Fator", "Subfator", "_termos"]:
        if need not in df.columns:
            df[need] = ""

    # _text: se não existir, usa _termos
    if "_text" not in df.columns:
        df["_text"] = df["_termos"].astype(str)

    # saneia strings
    for c in ["Dimensao", "Fator", "Subfator", "_termos", "_text"]:
        df[c] = df[c].astype(str).str.strip()

    # verifica se há colunas de embedding e_*
    e_cols = [c for c in df.columns if c.startswith("e_")]
    if not e_cols:
        st.error("O arquivo 'emb_taxonomia.parquet' não contém colunas de embedding (e_0, e_1, ...). "
                 "Regenere os artefatos com o script de preparação.")
        st.stop()

    return df

emb_tax = normalize_emb_tax_columns(emb_tax)


# checagens mínimas
for name, df, must in [
    ("emb_weaksignals.parquet", emb_ws,   ["_text", "e_0"]),
    ("emb_precursores.parquet", emb_prec, ["HTO","Precursor","_text","e_0"]),
    ("emb_taxonomia.parquet",   emb_tax,  ["Dimensao","Fator","Subfator","_termos","_text","e_0"]),
    ("emb_mapatriplo.parquet",  emb_map,  ["Report","Text","e_0"]),
]:
    if not df_has_cols(df, must):
        st.error(f"Artefato '{name}' sem colunas esperadas: {must}")
        st.stop()

# matrizes
M_ws   = emb_matrix(emb_ws)
M_prec = emb_matrix(emb_prec)
M_tax  = emb_matrix(emb_tax)
M_map  = emb_matrix(emb_map)


# ==============================
# SIDEBAR – parâmetros
# ==============================
st.sidebar.header("Parâmetros")
thr_ws   = st.sidebar.slider("Limiar (Weak Signals)", 0.0, 0.95, DEFAULT_THR, 0.01)
thr_prec = st.sidebar.slider("Limiar (Precursores)",   0.0, 0.95, DEFAULT_THR, 0.01)
thr_tax  = st.sidebar.slider("Limiar (TaxonomiaCP)",   0.0, 0.95, 0.55, 0.01)
topk_sim_reports = st.sidebar.slider("Top-N relatórios similares", 3, 20, 8, 1)


# ==============================
# UPLOAD DE DOCUMENTOS
# ==============================
st.subheader("📎 Faça upload do(s) documento(s) do evento (PDF ou DOCX)")
files = st.file_uploader("Arraste e solte aqui…", type=["pdf","docx"], accept_multiple_files=True)

if not files:
    st.info("Carregue pelo menos um arquivo para iniciar a análise.")
    st.stop()

# lê e quebra em parágrafos
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
    st.warning("Não foram encontrados parágrafos válidos nos arquivos.")
    st.stop()

# embeddings dos parágrafos
with st.spinner("Gerando embeddings dos parágrafos…"):
    model = load_model(MODEL_NAME)
    E_doc = embed_texts(model, df_paras["Text"].astype(str).tolist())


# ==============================
# MATCHING (WS / PRECURSORES / TAXONOMIA / RELATÓRIOS)
# ==============================
with st.spinner("Calculando similaridades…"):
    # Weak Signals
    S_ws = cosine_similarity(E_doc, M_ws)
    ws_hits = stack_matches(S_ws, emb_ws.rename(columns={"_text":"WS"}), ["WS"], thr_ws)

    # Precursores
    S_prec = cosine_similarity(E_doc, M_prec)
    prec_hits = stack_matches(S_prec, emb_prec, ["HTO","Precursor"], thr_prec)

    # Taxonomia
    S_tax = cosine_similarity(E_doc, M_tax)
    tax_hits = stack_matches(S_tax, emb_tax, ["Dimensao","Fator","Subfator","_termos"], thr_tax)

    # Relatórios similares (agregação por report)
    S_map = cosine_similarity(E_doc, M_map)
    sim_map_max = S_map.max(axis=0)
    emb_map_reports = emb_map[["Report"]].copy()
    emb_map_reports["max_sim"] = sim_map_max
    sim_reports = (emb_map_reports.groupby("Report", as_index=False)
                   .agg(MaxSim=("max_sim","max"), MeanSim=("max_sim","mean"))
                   .sort_values(["MaxSim","MeanSim"], ascending=False)
                   .head(topk_sim_reports))

# anexa contexto (File/Paragraph/Text) aos hits
def attach_context(df_hits: pd.DataFrame) -> pd.DataFrame:
    if df_hits is None or df_hits.empty:
        return df_hits
    out = df_hits.merge(
        df_paras.reset_index(drop=True).reset_index().rename(columns={"index":"idx_par"}),
        on="idx_par", how="left"
    )
    return out

ws_hits   = attach_context(ws_hits)
prec_hits = attach_context(prec_hits)
tax_hits  = attach_context(tax_hits)


# ==============================
# VISUALIZAÇÃO — SUMÁRIO
# ==============================
st.success(f"Documentos processados: **{df_paras['File'].nunique()}** | Parágrafos: **{len(df_paras)}**")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Weak Signals (hits)", len(ws_hits))
with c2: st.metric("Precursores (hits)", len(prec_hits))
with c3: st.metric("TaxonomiaCP (hits)", len(tax_hits))


# ==============================
# WEAK SIGNALS — (NORMALIZADO)
# ==============================
st.subheader("🔎 Weak Signals encontrados")

if ws_hits is None or ws_hits.empty:
    st.info("Nenhum Weak Signal acima do limiar.")
else:
    # coluna canônica WS + limpeza do sufixo "(0.53)"
    if "WS" not in ws_hits.columns and "WeakSignal" in ws_hits.columns:
        ws_hits["WS"] = ws_hits["WeakSignal"]

    ws_hits["WS"] = ws_hits["WS"].astype(str).map(clean_ws_tail)

    # garante File / Paragraph / Text (se necessário)
    for col in ["File","Paragraph","Text"]:
        if col not in ws_hits.columns:
            ws_hits = ws_hits.merge(
                df_paras.reset_index(drop=True).reset_index().rename(columns={"index":"idx_par"}),
                on="idx_par", how="left"
            )
            break

    ws_hits = ws_hits[ws_hits["WS"].str.len() > 0].drop_duplicates()

    ws_freq = (ws_hits.groupby("WS", as_index=False)
               .agg(Frequencia=("idx_par","count"))
               .sort_values("Frequencia", ascending=False))
    st.dataframe(ws_freq, use_container_width=True)

    cols_show = [c for c in ["WS","Similarity","File","Paragraph","Text"] if c in ws_hits.columns]
    st.dataframe(ws_hits[cols_show].head(200), use_container_width=True)


# ==============================
# PRECURSORES — HTO
# ==============================
st.subheader("🧩 Precursores (HTO) encontrados")
if prec_hits is None or prec_hits.empty:
    st.info("Nenhum Precursor acima do limiar.")
else:
    prec_hits = ensure_cols(prec_hits, ["HTO","Precursor","File","Paragraph","Text"])
    prec_freq = (prec_hits.groupby(["HTO","Precursor"], as_index=False)
                 .agg(Frequencia=("idx_par","count"))
                 .sort_values(["HTO","Frequencia"], ascending=[True,False]))
    st.dataframe(prec_freq, use_container_width=True)

    cols_show = [c for c in ["HTO","Precursor","Similarity","File","Paragraph","Text"] if c in prec_hits.columns]
    st.dataframe(prec_hits[cols_show].head(200), use_container_width=True)


# ==============================
# TAXONOMIA — VISUAL
# ==============================
st.subheader("📚 TaxonomiaCP (Dimensão/Fator/Subfator) encontrados")
if tax_hits is None or tax_hits.empty:
    st.info("Nenhum fator da Taxonomia acima do limiar.")
else:
    # saneamento leve + preenchimento
    def norm_tax(df):
        df = df.copy()
        for col in ["Dimensao","Fator","Subfator","_termos"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan":"", "None":""})
        # se Fator vazio, tenta mapear a partir do parquet emb_tax
        sub2fac = (emb_tax[["Subfator","Fator"]]
                   .dropna()
                   .astype(str)
                   .drop_duplicates())
        sub_map = dict(zip(sub2fac["Subfator"], sub2fac["Fator"]))
        df["Fator"] = np.where(df["Fator"].str.len()>0, df["Fator"], df["Subfator"].map(sub_map).fillna(""))
        df["Dimensao"] = df["Dimensao"].replace({"": "—"})
        df["Fator"]    = df["Fator"].replace({"": "—"})
        df["Subfator"] = df["Subfator"].replace({"": "—"})
        return df

    tax_hits = norm_tax(tax_hits)

    tax_freq = (tax_hits.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("idx_par","count"))
                .sort_values(["Dimensao","Fator","Frequencia"], ascending=[True,True,False]))
    st.dataframe(tax_freq, use_container_width=True)

    cols_show = [c for c in ["Dimensao","Fator","Subfator","_termos","Similarity","File","Paragraph","Text"] if c in tax_hits.columns]
    st.dataframe(tax_hits[cols_show].head(200), use_container_width=True)

    # Treemap
    st.subheader("🌳 Treemap (Dimensão → Fator → Subfator)")
    tax_plot = tax_hits.copy()
    tax_plot["value"] = 1
    if tax_plot[["Dimensao","Fator","Subfator"]].nunique().sum() <= 3:
        st.info("Taxonomia com muitos campos vazios. Ajuste os limiares ou verifique os dados.")
    else:
        fig_tax_tree = px.treemap(
            tax_plot,
            path=["Dimensao","Fator","Subfator"],
            values="value",
            hover_data=["_termos","Similarity","File"],
            title="Treemap da TaxonomiaCP encontrada"
        )
        st.plotly_chart(fig_tax_tree, use_container_width=True)

        # Sunburst
        st.subheader("🌞 Sunburst (Dimensão → Fator → Subfator)")
        fig_tax_sun = px.sunburst(
            tax_plot,
            path=["Dimensao","Fator","Subfator"],
            values="value",
            hover_data=["_termos","Similarity","File"],
            title="Sunburst da TaxonomiaCP encontrada"
        )
        st.plotly_chart(fig_tax_sun, use_container_width=True)


# ==============================
# RELATÓRIOS SIMILARES
# ==============================
st.subheader("🗂️ Relatórios pregressos mais similares")
if sim_reports is None or sim_reports.empty:
    st.info("Sem similares acima de 0.")
else:
    st.dataframe(sim_reports, use_container_width=True)


# ================================
# 🌳 Treemap / Sunburst — HTO → Precursor → WeakSignal (doc atual)
# (Bloco robusto com normalização de colunas)
# ================================
import plotly.express as px

def pick_col(df, candidates):
    """Retorna a primeira coluna existente na lista candidates (ou None)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

if prec_hits is None or prec_hits.empty or ws_hits is None or ws_hits.empty:
    st.info("Sem interseção suficiente entre Precursores e Weak Signals para treemap/sunburst.")
else:
    # ---- 1) Padroniza colunas de WS ----
    ws_col = pick_col(ws_hits, ["WeakSignal_clean", "WeakSignal", "WS"])
    if ws_col is None:
        st.warning("Não encontrei coluna de Weak Signal (WeakSignal/WeakSignal_clean/WS).")
        ws_col = "WeakSignal"  # evita KeyError adiante
        ws_hits = ws_hits.assign(WeakSignal="")
    # renomeia para 'WS'
    if ws_col != "WS":
        ws_hits = ws_hits.rename(columns={ws_col: "WS"})

    # padroniza texto do parágrafo para 'Text'
    text_ws_col = pick_col(ws_hits, ["Text", "Snippet"])
    if text_ws_col and text_ws_col != "Text":
        ws_hits = ws_hits.rename(columns={text_ws_col: "Text"})

    # ---- 2) Padroniza colunas de Precursor ----
    # garante que haja 'HTO' e 'Precursor'
    for need in ["HTO", "Precursor"]:
        if need not in prec_hits.columns:
            prec_hits[need] = ""

    text_prec_col = pick_col(prec_hits, ["Text", "Snippet"])
    if text_prec_col and text_prec_col != "Text":
        prec_hits = prec_hits.rename(columns={text_prec_col: "Text"})

    # ---- 3) Seleciona e une por parágrafo (idx_par + File + Paragraph) ----
    # OBS: usamos File/Paragraph apenas para estabilizar, se existirem.
    join_ws_cols   = ["idx_par", "WS"]
    join_prec_cols = ["idx_par", "HTO", "Precursor"]

    if "File" in ws_hits.columns:      join_ws_cols.append("File")
    if "Paragraph" in ws_hits.columns: join_ws_cols.append("Paragraph")
    if "Text" in ws_hits.columns:      join_ws_cols.append("Text")

    if "File" in prec_hits.columns:      join_prec_cols.append("File")
    if "Paragraph" in prec_hits.columns: join_prec_cols.append("Paragraph")
    if "Text" in prec_hits.columns:      join_prec_cols.append("Text")

    join_ws   = ws_hits[join_ws_cols].drop_duplicates()
    join_prec = prec_hits[join_prec_cols].drop_duplicates()

    # Faz o merge pela chave mínima garantida (idx_par), e pela interseção das demais
    merge_keys = ["idx_par"]
    if "File" in join_ws.columns and "File" in join_prec.columns:
        merge_keys.append("File")
    if "Paragraph" in join_ws.columns and "Paragraph" in join_prec.columns:
        merge_keys.append("Paragraph")

    tri = join_prec.merge(join_ws, on=merge_keys, how="inner")

    # ---- 4) Saneia strings e remove vazios ----
    for c in ["HTO", "Precursor", "WS"]:
        if c not in tri.columns:
            tri[c] = ""
        tri[c] = tri[c].astype(str).str.strip()

    tri = tri[(tri["HTO"] != "") & (tri["Precursor"] != "") & (tri["WS"] != "")]
    if tri.empty:
        st.warning("Após saneamento, não há combinações HTO/Precursor/WeakSignal válidas para o treemap.")
    else:
        # ---- 5) Treemap ----
        tri["value"] = 1  # cada ocorrência conta 1
        hover_cols = [c for c in ["File", "Paragraph", "Text"] if c in tri.columns]

        st.subheader("🌳 Treemap (HTO → Precursor → WeakSignal)")
        fig_tree = px.treemap(
            tri,
            path=["HTO", "Precursor", "WS"],
            values="value",
            hover_data=hover_cols,
            title="Treemap hierárquico (WeakSignals por Precursor/HTO)"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

        # ---- 6) Sunburst ----
        st.subheader("🌞 Sunburst (HTO → Precursor → WeakSignal)")
        fig_sun = px.sunburst(
            tri,
            path=["HTO", "Precursor", "WS"],
            values="value",
            hover_data=hover_cols,
            title="Sunburst (WeakSignals por Precursor/HTO)"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

        # ---- 7) (Opcional) visão textual compacta ----
        with st.expander("📂 Árvore (texto resumido)"):
            for hto in sorted(tri["HTO"].unique()):
                st.markdown(f"**{hto}**")
                tri_h = tri[tri["HTO"] == hto]
                for prec in sorted(tri_h["Precursor"].unique()):
                    st.markdown(f"- {prec}")
                    ws_list = sorted(tri_h[tri_h["Precursor"] == prec]["WS"].unique().tolist())
                    if ws_list:
                        st.markdown("  - " + "; ".join(ws_list[:30]))


# ================================
# 🌳 ÁRVORE INTERATIVA (ECharts) — HTO → Precursor → Weak Signal
# Usa MapaTriplo_tratado.xlsx para filtrar WS por Precursor e marcar origem
# ================================
import io, re, requests
from streamlit_echarts import st_echarts

MAPTRIP_URL = "https://raw.githubusercontent.com/titetodesco/VisualizarPrecSinaisFracosReports/main/MapaTriplo_tratado.xlsx"

def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

WS_LABEL_RE = re.compile(r"^\s*WeakSignal\s+", re.I)

def clean_ws(s: str) -> str:
    s = "" if s is None else str(s)
    s = WS_LABEL_RE.sub("", s)
    return s.strip()

@st.cache_data(ttl=300, show_spinner=False)
def load_mapatriplo(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    df = pd.read_excel(bio)
    # Normaliza nomes de colunas esperadas
    col_hto  = _pick(df, ["HTO","Hto","hto"])
    col_prec = _pick(df, ["Precursor","precursor","Precursores"])
    col_ws   = _pick(df, ["WeakSignal","WS","Weak Signal","WeakSignal_clean"])
    if not col_prec or not col_ws:
        raise RuntimeError("MapaTriplo precisa ter colunas 'Precursor' e 'WeakSignal'.")
    out = df.copy()
    out = out.rename(columns={col_hto:"HTO", col_prec:"Precursor", col_ws:"WS"})
    out["HTO"] = out.get("HTO", pd.Series(index=out.index, dtype=str)).astype(str).str.strip()
    out["Precursor"] = out["Precursor"].astype(str).str.strip()
    out["WS"] = out["WS"].astype(str).map(clean_ws)
    out = out.replace({"HTO":{"nan":""}})
    return out[["HTO","Precursor","WS"]].dropna(subset=["Precursor","WS"]).drop_duplicates()

# ---- Proteções básicas
if prec_hits is None or prec_hits.empty or ws_hits is None or ws_hits.empty:
    st.info("Sem interseção suficiente entre Precursores e Weak Signals para montar a árvore.")
else:
    # ===== 1) Padroniza/limpa colunas de entrada
    # WS dos hits do documento
    ws_col = _pick(ws_hits, ["WS","WeakSignal_clean","WeakSignal"])
    if ws_col != "WS":
        ws_hits = ws_hits.rename(columns={ws_col:"WS"})
    ws_hits["WS"] = ws_hits["WS"].astype(str).map(clean_ws)

    # Texto do parágrafo (opcional para drilldown)
    txt_col = _pick(ws_hits, ["Text","Snippet"])
    if txt_col and txt_col != "Text":
        ws_hits = ws_hits.rename(columns={txt_col:"Text"})

    # Precursores
    for need in ["HTO","Precursor"]:
        if need not in prec_hits.columns:
            prec_hits[need] = ""
    txtp_col = _pick(prec_hits, ["Text","Snippet"])
    if txtp_col and txtp_col != "Text":
        prec_hits = prec_hits.rename(columns={txtp_col:"Text"})

    # ===== 2) Merge direto por parágrafo (pares que realmente co-ocorrem)
    # chave mínima: idx_par (+ File/Paragraph se existirem)
    ws_on = ["idx_par"]
    prec_on = ["idx_par"]
    for k in ["File","Paragraph"]:
        if k in ws_hits.columns and k in prec_hits.columns:
            ws_on.append(k); prec_on.append(k)

    ws_doc = ws_hits[ws_on + ["WS"]].drop_duplicates()
    prec_doc = prec_hits[prec_on + ["HTO","Precursor"]].drop_duplicates()
    tri_direct = prec_doc.merge(ws_doc, left_on=prec_on, right_on=ws_on, how="inner")[["HTO","Precursor","WS"]]

    # ===== 3) Carrega MapaTriplo e filtra WS presentes no documento
    maptrip = load_mapatriplo(MAPTRIP_URL)
    ws_doc_unique = set(ws_doc["WS"].unique().tolist())
    maptrip_doc = maptrip[maptrip["WS"].isin(ws_doc_unique)].copy()

    # ===== 4) Combina e marca origem (direto / indireto / ambos)
    # - left: pares vindos do documento (co-ocorrência real)
    # - right: pares do MapaTriplo que têm WS presente no documento
    comb = tri_direct.merge(
        maptrip_doc, on=["Precursor","WS"], how="outer", suffixes=("_dir","_map"), indicator=True
    )
    # define HTO preferindo o do documento; se faltar, usa o do mapa
    comb["HTO"] = comb["HTO_dir"].where(comb["HTO_dir"].notna() & (comb["HTO_dir"].astype(str)!=""), comb["HTO_map"])
    comb["origem"] = comb["_merge"].map({"left_only":"direto","right_only":"indireto","both":"ambos"})
    comb = comb[["HTO","Precursor","WS","origem"]].dropna(subset=["Precursor","WS"]).drop_duplicates()

    # ===== 5) Frequência por (Precursor, WS)
    # conta ocorrências reais (no doc) — pares “indireto” recebem 1
    if not tri_direct.empty:
        freq_dir = (tri_direct.groupby(["Precursor","WS"], as_index=False)
                             .size().rename(columns={"size":"freq_dir"}))
    else:
        freq_dir = pd.DataFrame(columns=["Precursor","WS","freq_dir"])

    comb = comb.merge(freq_dir, on=["Precursor","WS"], how="left")
    comb["freq"] = comb["freq_dir"].fillna(1).astype(int)

    # ===== 6) Índice para drilldown (opcional)
    # mapeia chaves para as linhas do documento (quando for 'direto' ou 'ambos')
    idx_map = {}
    if not tri_direct.empty:
        # precisamos saber quais parágrafos formam cada (Precursor, WS)
        if {"idx_par","Precursor"}.issubset(prec_hits.columns) and {"idx_par","WS"}.issubset(ws_hits.columns):
            pw = prec_hits[["idx_par","Precursor"]].drop_duplicates().merge(
                ws_hits[["idx_par","WS"]].drop_duplicates(), on="idx_par", how="inner"
            )
            pw = pw.merge(comb[["Precursor","WS"]], on=["Precursor","WS"], how="inner")
            for (prec, ws), g in pw.groupby(["Precursor","WS"]):
                idx_map[f"{prec}::{ws}"] = set(g["idx_par"].tolist())

    # ===== 7) Monta estrutura de árvore para ECharts (HTO → Precursor → WS)
    def make_node(name, children=None, value=None, extra=None):
        node = {"name": name}
        if value is not None: node["value"] = int(value)
        if extra is not None: node["extra"] = extra
        if children: node["children"] = children
        return node

    tree_children = []
    for hto in sorted(comb["HTO"].fillna("HTO").unique()):
        comb_h = comb[comb["HTO"].fillna("HTO") == hto]
        prec_children = []
        for prec in sorted(comb_h["Precursor"].unique()):
            comb_p = comb_h[comb_h["Precursor"] == prec]
            ws_children = []
            for _, r in comb_p.sort_values(["origem","WS"]).iterrows():
                key = f"{prec}::{r['WS']}"
                ws_children.append(make_node(
                    r["WS"],
                    value=int(r["freq"]),
                    extra={"type":"ws", "key": key, "origem": r["origem"]}
                ))
            prec_children.append(make_node(
                prec,
                children=ws_children,
                value=int(comb_p["freq"].sum()),
                extra={"type":"prec","key": f"PREC::{hto}::{prec}"}
            ))
        tree_children.append(make_node(
            hto if hto else "HTO",
            children=prec_children,
            value=int(comb_h["freq"].sum()),
            extra={"type":"hto","key": f"HTO::{hto or 'HTO'}"}
        ))

    root = make_node("HTO", children=tree_children)

    # ===== 8) Desenha a árvore
    st.subheader("🌿 Árvore Interativa — HTO → Precursores → Weak Signals")
    options = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
            # JS de tooltip: mostra nome e frequência
            "formatter": """function(p){
                var v = (p.value !== undefined) ? ("<br/>Frequência: " + p.value) : "";
                return "<b>" + p.name + "</b>" + v;
            }"""
        },
        "series": [{
            "type": "tree",
            "data": [root],
            "left": "2%", "right": "20%", "top": "2%", "bottom": "2%",
            "symbol": "circle", "symbolSize": 10,
            "expandAndCollapse": True,
            "initialTreeDepth": 2,
            "animationDuration": 300,
            "animationDurationUpdate": 300,
            "label": {"position": "left", "verticalAlign": "middle", "align": "right", "fontSize": 12},
            "leaves": {"label": {"position": "right", "align": "left"}},
            "emphasis": {"focus": "descendant"},
            "roam": True
        }]
    }
    event = st_echarts(options=options, height="720px", events={"click": "function (p) { return p; }"})

    # ===== 9) Drilldown ao clicar num WS
    st.subheader("🔎 Detalhes do nó selecionado")
    if event and "data" in event and event["data"] and "extra" in event["data"]:
        extra = event["data"]["extra"] or {}
        if extra.get("type") == "ws":
            key = extra.get("key")
            idxs = sorted(idx_map.get(key, []))
            if idxs:
                # mostra os parágrafos onde esse par (Precursor, WS) ocorreu (direto/ambos)
                cols = [c for c in ["File","Paragraph","Text"] if c in ws_hits.columns]
                df_detail = (df_paras.loc[idxs, ["File","Paragraph","Text"]] if cols==["File","Paragraph","Text"]
                             else df_paras.loc[idxs, cols]) if "df_paras" in globals() else None
                st.write(f"**WS:** `{event['data']['name']}` — **Origem:** `{extra.get('origem')}` — **Ocorrências (diretas):** {len(idxs)}")
                if df_detail is not None:
                    st.dataframe(df_detail, use_container_width=True)
            else:
                st.info(f"**WS:** `{event['data']['name']}` — **Origem:** `{extra.get('origem')}` (sem parágrafos diretos; vínculo do MapaTriplo).")
        else:
            st.caption("Dica: clique no nó do Weak Signal para ver os detalhes.")



# ==============================
# DOWNLOAD EXCEL — RESULTADOS
# ==============================
st.subheader("⬇️ Download (Excel consolidado)")
dfs_out = {
    "WS_hits":        ws_hits if ws_hits is not None else pd.DataFrame(),
    "Precursores_hits": prec_hits if prec_hits is not None else pd.DataFrame(),
    "Taxonomia_hits":   tax_hits if tax_hits is not None else pd.DataFrame(),
    "Relatorios_similares": sim_reports if sim_reports is not None else pd.DataFrame(),
}
st.download_button(
    "Baixar resultados (.xlsx)",
    data=to_excel_bytes(dfs_out),
    file_name=f"analise_evento_{int(time.time())}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Dica: ajuste os limiares na barra lateral para controlar ruído vs. cobertura. Os embeddings de dicionários e mapa vêm dos artefatos gerados previamente.")
