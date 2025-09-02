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


# ==============================
# TRI (HTO → Precursor → WS) — TREEMAP & SUNBURST
# ==============================
if (prec_hits is not None and not prec_hits.empty) and (ws_hits is not None and not ws_hits.empty):
    st.subheader("🌳 Treemap / Sunburst — HTO → Precursor → WeakSignal (doc atual)")
    # vincula por parágrafo
    join_ws   = ws_hits[["idx_par","WS","File","Paragraph","Text"]].drop_duplicates()
    join_prec = prec_hits[["idx_par","HTO","Precursor","File","Paragraph","Text"]].drop_duplicates()
    tri = join_prec.merge(join_ws, on=["idx_par","File","Paragraph"], how="inner").drop_duplicates()

    if tri.empty:
        st.info("Sem interseção WS ↔ Precursor no mesmo parágrafo para a visualização hierárquica.")
    else        :
        tri["value"] = 1
        fig_tree = px.treemap(
            tri,
            path=["HTO","Precursor","WS"],
            values="value",
            hover_data=["File","Paragraph","Text"],
            title="Treemap hierárquico (WeakSignals por Precursor/HTO)"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

        fig_sun = px.sunburst(
            tri,
            path=["HTO","Precursor","WS"],
            values="value",
            hover_data=["File","Paragraph","Text"],
            title="Sunburst (WeakSignals por Precursor/HTO)"
        )
        st.plotly_chart(fig_sun, use_container_width=True)


# ==============================
# 🌳 ÁRVORE INTERATIVA — HTO → Precursor → WS
# ==============================
st.markdown("## 🌳 Árvore: HTO → Precursores → Weak Signals (documento)")

if prec_hits is None or prec_hits.empty or ws_hits is None or ws_hits.empty:
    st.info("Nenhum match simultâneo de Precursores e Weak Signals para construir a árvore.")
else:
    # monta base canônica p/ árvore
    join_ws   = ws_hits[["idx_par","WS","File","Paragraph","Text"]].drop_duplicates()
    join_prec = prec_hits[["idx_par","HTO","Precursor","File","Paragraph","Text"]].drop_duplicates()
    tree_df = join_prec.merge(join_ws, on=["idx_par","File","Paragraph"], how="inner").drop_duplicates()

    if tree_df.empty:
        st.warning("Não há interseção entre Precursores e WeakSignals nos mesmos parágrafos.")
    else:
        node_index: dict[str, set[int]] = {}

        def add_index(key: str, rows_idx):
            node_index.setdefault(key, set()).update(rows_idx)

        def make_node(name, children=None, value=None, extra=None):
            node = {"name": str(name)}
            if value is not None:
                node["value"] = int(value)
            if extra is not None:
                node["extra"] = extra
            if children:
                node["children"] = children
            return node

        def build_tree(df: pd.DataFrame):
            root_children = []
            for hto in sorted(df["HTO"].dropna().astype(str).unique()):
                df_h = df[df["HTO"] == hto]
                prec_children = []
                for prec in sorted(df_h["Precursor"].dropna().astype(str).unique()):
                    df_p = df_h[df_h["Precursor"] == prec]
                    ws_children = []
                    for ws in sorted(df_p["WS"].dropna().astype(str).unique()):
                        df_w = df_p[df_p["WS"] == ws]
                        key_w = f"WS::{hto}::{prec}::{ws}"
                        add_index(key_w, df_w.index.tolist())
                        ws_children.append(make_node(
                            ws,
                            value=len(df_w),
                            extra={"type": "ws", "key": key_w}
                        ))
                    key_p = f"PREC::{hto}::{prec}"
                    add_index(key_p, df_p.index.tolist())
                    prec_children.append(make_node(
                        prec,
                        children=ws_children,
                        value=len(df_p),
                        extra={"type": "prec", "key": key_p}
                    ))
                key_h = f"HTO::{hto}"
                add_index(key_h, df_h.index.tolist())
                root_children.append(make_node(
                    hto,
                    children=prec_children,
                    value=len(df_h),
                    extra={"type": "hto", "key": key_h}
                ))
            return make_node("ROOT", children=root_children, value=len(df))

        root_data = build_tree(tree_df)

        st.subheader("🌿 Árvore Interativa (colapsável)")
        options = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
                "formatter": """function(p){
                    var v = (p.value!==undefined)?("<br/>Freq: "+p.value):"";
                    return "<b>"+p.name+"</b>"+v;
                }"""
            },
            "series": [{
                "type": "tree",
                "data": [root_data],
                "left": "2%", "right": "20%", "top": "2%", "bottom": "2%",
                "symbol": "circle",
                "symbolSize": 10,
                "expandAndCollapse": True,
                "initialTreeDepth": 2,
                "animationDuration": 300,
                "animationDurationUpdate": 300,
                "label": {"position":"left","verticalAlign":"middle","align":"right","fontSize":12},
                "leaves": {"label":{"position":"right","align":"left"}},
                "emphasis": {"focus":"descendant"},
                "roam": True
            }]
        }

        event = st_echarts(
            options=options,
            height="700px",
            events={"click": "function(p){return p;}"}
        )

        st.subheader("🔎 Detalhes do nó selecionado")
        if event and "name" in event:
            data = event.get("data", {}) or {}
            extra = data.get("extra", {})
            key = extra.get("key")

            if key and key in node_index:
                idxs = sorted(node_index[key])
                cols_show = [c for c in ["HTO","Precursor","WS","File","Paragraph","Text"] if c in tree_df.columns]
                detail = tree_df.loc[idxs, cols_show].copy()
                st.write(f"**Nó:** `{event['name']}` — **linhas:** {len(detail)}")
                st.dataframe(detail, use_container_width=True)
                st.download_button(
                    "📥 Baixar CSV deste nó",
                    data=detail.to_csv(index=False).encode("utf-8"),
                    file_name="detalhes_no.csv",
                    mime="text/csv"
                )
            else:
                st.info("Clique em um nó de **HTO**, **Precursor** ou **Weak Signal** para ver os detalhes.")


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
