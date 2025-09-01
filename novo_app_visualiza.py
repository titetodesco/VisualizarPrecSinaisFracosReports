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

emb_tax = normalize_taxonomia_cols(emb_tax)

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
    return out.rename(columns={"Text":"Snippet"})

ws_hits   = attach_context(ws_hits, df_paras)
prec_hits = attach_context(prec_hits, df_paras)
tax_hits  = attach_context(tax_hits, df_paras)

# -----------------------------------------------------------------------------
# VISUALIZAÇÃO
# -----------------------------------------------------------------------------
st.success(f"Documentos processados: **{df_paras['File'].nunique()}** | Parágrafos: **{len(df_paras)}**")
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

    st.dataframe(ws_hits[["WeakSignal","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

st.subheader("🧩 Precursores (HTO) encontrados")
if prec_hits.empty:
    st.info("Nenhum Precursor acima do limiar.")
else:
    prec_freq = (prec_hits.groupby(["HTO","Precursor"], as_index=False)
                 .agg(Frequencia=("idx_par","count"))
                 .sort_values(["HTO","Frequencia"], ascending=[True,False]))
    st.dataframe(prec_freq, use_container_width=True)

    st.dataframe(prec_hits[["HTO","Precursor","Similarity","File","Paragraph","Snippet"]]
                 .head(200), use_container_width=True)

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

# Treemap HTO → Precursor → WeakSignal (limpo)
tri = pd.DataFrame()  # <- garanta que tri exista mesmo se não for preenchido
if not prec_hits.empty and not ws_hits.empty:
    st.subheader("🌳 Treemap (HTO → Precursor → WeakSignal)")
    join_ws   = ws_hits[["idx_par","WeakSignal_clean"]].drop_duplicates()
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

# Árvore textual compacta (só se houver tri)
with st.expander("Árvore colapsável (texto)"):
    if not tri.empty:
        for hto in sorted(tri["HTO"].dropna().unique()):
            st.markdown(f"**{hto}**")
            for prec in sorted(tri[tri["HTO"]==hto]["Precursor"].dropna().unique()):
                ws_list = sorted(tri[(tri["HTO"]==hto) & (tri["Precursor"]==prec)]["WeakSignal_clean"].unique().tolist())
                if ws_list:
                    st.markdown(f"- {prec}: " + "; ".join(ws_list[:30]))


# Árvore textual compacta (opcional)
with st.expander("Árvore colapsável (texto)"):
    if not prec_hits.empty and not ws_hits.empty:
        for hto in sorted(prec_hits["HTO"].dropna().unique()):
            st.markdown(f"**{hto}**")
            precs = sorted(prec_hits[prec_hits["HTO"]==hto]["Precursor"].dropna().unique())
            for prec in precs[:100]:
                ws_list = sorted(tri[(tri["HTO"]==hto) & (tri["Precursor"]==prec)]["WeakSignal_clean"].unique().tolist())
                if ws_list:
                    st.markdown(f"- {prec}: " + "; ".join(ws_list[:30]))

# ================================
# GRÁFICOS — MAPA TRÍPLICE (doc atual)
# ================================
st.markdown("## 📊 Visualizações — HTO → Precursor → Weak Signal")

if tri is None or tri.empty:
    st.info("Nenhuma tríade (HTO–Precursor–WeakSignal) acima dos limiares para o(s) documento(s) enviado(s).")
else:
    # 1) Treemap (HTO → Precursor → WS)
    st.subheader("🌳 Treemap (HTO → Precursor → WeakSignal)")
    tri_plot = tri.copy()
    tri_plot["value"] = 1
    fig_tri = px.treemap(
        tri_plot,
        path=["HTO", "Precursor", "WeakSignal_clean"],
        values="value",
        title="Treemap das tríades encontradas no(s) documento(s)"
    )
    st.plotly_chart(fig_tri, use_container_width=True)

    # 2) Heatmap (frequência) — Precursor × WeakSignal
    st.subheader("🔥 Heatmap — Frequência de Weak Signals por Precursor")
    freq_pw = (tri_plot
               .groupby(["Precursor", "WeakSignal_clean"], as_index=False)
               .agg(Frequencia=("value", "sum")))
    # opcional: limitar a top-N weak signals (mais frequentes) para reduzir poluição visual
    top_ws = (freq_pw.groupby("WeakSignal_clean", as_index=False)
              .agg(Total=("Frequencia","sum"))
              .sort_values("Total", ascending=False)
              .head(25)["WeakSignal_clean"].tolist())
    freq_pw_top = freq_pw[freq_pw["WeakSignal_clean"].isin(top_ws)]

    if freq_pw_top.empty:
        st.info("Sem dados suficientes para o heatmap com top 25 WS.")
    else:
        mat = (freq_pw_top
               .pivot(index="Precursor", columns="WeakSignal_clean", values="Frequencia")
               .fillna(0)
               .astype(int))
        fig_hm = px.imshow(
            mat.values,
            labels=dict(x="Weak Signal", y="Precursor", color="Frequência"),
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            title="Frequência de Weak Signals (Top-25) por Precursor"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # 3) Sunburst (HTO → Precursor → WS)
    st.subheader("🌞 Sunburst (HTO → Precursor → WeakSignal)")
    fig_sun = px.sunburst(
        tri_plot.assign(value=1),
        path=["HTO", "Precursor", "WeakSignal_clean"],
        values="value",
        title="Sunburst das tríades encontradas"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    # 4) (Opcional) Barra: top precursores por nº de WS distintos
    st.subheader("🏷️ Top Precursores por nº de Weak Signals distintos")
    prec_ws_count = (tri_plot.groupby(["HTO","Precursor"], as_index=False)
                     .agg(WS_distintos=("WeakSignal_clean", lambda s: s.nunique())))
    prec_ws_count = prec_ws_count.sort_values(["WS_distintos","HTO"], ascending=[False, True]).head(20)
    fig_bar_prec = px.bar(
        prec_ws_count,
        x="WS_distintos", y="Precursor", color="HTO",
        orientation="h",
        title="Top precursores por variedade de Weak Signals (doc atual)"
    )
    st.plotly_chart(fig_bar_prec, use_container_width=True)

    # 5) (Opcional) Relatórios históricos mais similares – bar (já existe tabela; só visual rápido)
    if not sim_reports.empty:
        st.subheader("📚 Top relatórios pregressos mais similares (visual)")
        fig_reports = px.bar(
            sim_reports.sort_values("MaxSim", ascending=True),
            x="MaxSim", y="Report",
            orientation="h",
            hover_data=["MeanSim"],
            title="Relatórios mais similares (por similaridade máxima)"
        )
        st.plotly_chart(fig_reports, use_container_width=True)
# ================================
# GRÁFICOS — TAXONOMIACP
# ================================
st.markdown("## 🧩 Visualizações — TaxonomiaCP (Dimensão → Fator → Subfator)")

if tax_hits is None or tax_hits.empty:
    st.info("Nenhum fator da TaxonomiaCP acima do limiar para o(s) documento(s).")
else:
    tax_plot = tax_hits.copy()
    tax_plot["value"] = 1

    # 1) Treemap (Dimensão → Fator → Subfator)
    st.subheader("🌳 Treemap (Dimensão → Fator → Subfator)")
    fig_tax_tree = px.treemap(
        tax_plot,
        path=["Dimensao", "Fator", "Subfator"],
        values="value",
        hover_data=["_termos"],
        title="Treemap da TaxonomiaCP encontrada"
    )
    st.plotly_chart(fig_tax_tree, use_container_width=True)

    # 2) Sunburst (Dimensão → Fator → Subfator)
    st.subheader("🌞 Sunburst (Dimensão → Fator → Subfator)")
    fig_tax_sun = px.sunburst(
        tax_plot,
        path=["Dimensao", "Fator", "Subfator"],
        values="value",
        hover_data=["_termos"],
        title="Sunburst da TaxonomiaCP encontrada"
    )
    st.plotly_chart(fig_tax_sun, use_container_width=True)

    # 3) Ranking de Subfatores (top-N)
    st.subheader("🏷️ Top Subfatores por frequência")
    sub_rank = (tax_plot.groupby(["Dimensao","Fator","Subfator"], as_index=False)
                .agg(Frequencia=("value","sum"))
                .sort_values("Frequencia", ascending=False)
                .head(20))
    fig_sub_bar = px.bar(
        sub_rank,
        x="Frequencia", y="Subfator",
        color="Dimensao",
        orientation="h",
        hover_data=["Fator"],
        title="Top Subfatores (doc atual)"
    )
    st.plotly_chart(fig_sub_bar, use_container_width=True)

    # 4) Heatmap Dimensão × Fator (contagem de Subfatores marcados)
    st.subheader("🔥 Heatmap — Dimensão × Fator")
    df_hm = (tax_plot.groupby(["Dimensao","Fator"], as_index=False)
             .agg(Qtd=("Subfator","nunique")))
    mat_tax = (df_hm
               .pivot(index="Fator", columns="Dimensao", values="Qtd")
               .fillna(0)
               .astype(int))
    if not mat_tax.empty:
        fig_tax_hm = px.imshow(
            mat_tax.values,
            labels=dict(x="Dimensão", y="Fator", color="Qtd Subfatores"),
            x=mat_tax.columns.tolist(),
            y=mat_tax.index.tolist(),
            title="Qtd de Subfatores por Dimensão × Fator"
        )
        st.plotly_chart(fig_tax_hm, use_container_width=True)


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
