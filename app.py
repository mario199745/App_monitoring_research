import io
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_plotly_events import plotly_events


# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Analizador de publicaciones",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SHEET = "BBDD"

TARGET_COL = "General_ Tipo de Publicación"
FILTER_COLUMNS = [
    "Nombre de Base de datos",
    "General_ Repositorio",
    "General_ Tipo de Publicación",
    "General_ Tipo de tesis Pre/Posgrado",
    "General_ Especialidad",
    "General_ Institución, revista o fuente que pública",
    "General_ Institución/Universidad",
]

POSITIVE_PATTERNS = {
    r"\binvestigaci[oó]n\b": 32,
    r"\bcient[ií]fic": 24,
    r"\btesis\b": 20,
    r"\btrabajo de investigaci[oó]n\b": 22,
    r"\bproyecto de investigaci[oó]n\b": 20,
    r"\binforme de investigaci[oó]n\b": 20,
    r"\bcuaderno de investigaci[oó]n\b": 18,
    r"\brevisi[oó]n cient[ií]fica\b": 18,
    r"\bseminario de investigaci[oó]n\b": 14,
    r"\bart[ií]culo\b": 12,
    r"\bconferencia\b": 6,
    r"\bdocumento de trabajo\b": 8,
    r"\bmonograf[ií]a\b": 8,
}
NEGATIVE_PATTERNS = {
    r"\bnota de prensa\b": -30,
    r"\bweb\b": -24,
    r"\bhoja divulgativa\b": -22,
    r"\binfograf[ií]a\b": -22,
    r"\bp[oó]ster\b": -14,
    r"\bposter\b": -14,
    r"\bbolet[ií]n\b": -12,
    r"\bfolleto": -12,
    r"\bgu[ií]a\b": -10,
    r"\bmanual\b": -10,
    r"\bcomentario\b": -10,
    r"\bcorrespondencia\b": -12,
    r"\botros\b": -10,
}

POSITIVE_PROTOTYPES = [
    "articulo cientifico",
    "articulo de investigacion",
    "informe de investigacion",
    "tesis de pregrado",
    "tesis de posgrado",
    "trabajo de investigacion",
    "proyecto de investigacion",
    "revision cientifica",
    "cuaderno de investigacion",
    "seminario de investigacion",
]
NEGATIVE_PROTOTYPES = [
    "nota de prensa",
    "web",
    "infografia",
    "boletin",
    "hoja divulgativa",
    "manual",
    "guia tecnica",
    "otros",
    "poster",
    "comentario",
]


# =========================
# UTILIDADES
# =========================
def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


@st.cache_data(show_spinner=False)
def get_excel_sheets(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def load_excel_sheet(file_bytes: bytes, sheet_name: str):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def compute_keyword_score(text: str) -> int:
    score = 0
    for pattern, weight in POSITIVE_PATTERNS.items():
        if re.search(pattern, text):
            score += weight
    for pattern, weight in NEGATIVE_PATTERNS.items():
        if re.search(pattern, text):
            score += weight
    return score


def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(max_value, value))


def label_priority(score: float) -> str:
    if score >= 70:
        return "Alta"
    if score >= 45:
        return "Media"
    return "Baja"


@st.cache_data(show_spinner=False)
def build_research_priority_table(df: pd.DataFrame, target_col: str):
    working = df.copy()
    working[target_col] = working[target_col].fillna("Sin dato").astype(str).str.strip()

    summary = (
        working.groupby(target_col, dropna=False)
        .size()
        .reset_index(name="Frecuencia")
        .sort_values("Frecuencia", ascending=False)
        .reset_index(drop=True)
    )

    if summary.empty:
        return pd.DataFrame(
            columns=[
                target_col,
                "Frecuencia",
                "Score_investigacion",
                "Prioridad_investigacion",
                "Keyword_score",
                "Similitud_positiva",
                "Similitud_negativa",
            ]
        )

    summary["Texto_normalizado"] = summary[target_col].apply(normalize_text)
    summary["Keyword_score"] = summary["Texto_normalizado"].apply(compute_keyword_score)

    corpus = (
        summary["Texto_normalizado"].tolist()
        + [normalize_text(x) for x in POSITIVE_PROTOTYPES]
        + [normalize_text(x) for x in NEGATIVE_PROTOTYPES]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf = vectorizer.fit_transform(corpus)

    n_classes = len(summary)
    class_vecs = tfidf[:n_classes]
    pos_vecs = tfidf[n_classes : n_classes + len(POSITIVE_PROTOTYPES)]
    neg_vecs = tfidf[n_classes + len(POSITIVE_PROTOTYPES) :]

    pos_similarity = (class_vecs @ pos_vecs.T).toarray().mean(axis=1)
    neg_similarity = (class_vecs @ neg_vecs.T).toarray().mean(axis=1)

    summary["Similitud_positiva"] = pos_similarity
    summary["Similitud_negativa"] = neg_similarity
    summary["Score_semantico"] = ((pos_similarity - neg_similarity + 1) / 2) * 100

    max_freq = max(summary["Frecuencia"].max(), 1)
    summary["Score_frecuencia"] = (summary["Frecuencia"] / max_freq) * 15

    summary["Score_investigacion"] = (
        0.55 * summary["Score_semantico"]
        + 0.35 * (summary["Keyword_score"] + 50)
        + 0.10 * summary["Score_frecuencia"]
    ).apply(clamp)

    summary["Prioridad_investigacion"] = summary["Score_investigacion"].apply(label_priority)
    summary = summary.sort_values(
        ["Score_investigacion", "Frecuencia"], ascending=[False, False]
    ).reset_index(drop=True)

    return summary[
        [
            target_col,
            "Frecuencia",
            "Score_investigacion",
            "Prioridad_investigacion",
            "Keyword_score",
            "Similitud_positiva",
            "Similitud_negativa",
        ]
    ]


def apply_filters(df: pd.DataFrame, visible_filter_cols):
    filtered = df.copy()

    st.sidebar.markdown("## Filtros")
    for col in visible_filter_cols:
        if col not in filtered.columns:
            continue

        series = filtered[col].dropna().astype(str).str.strip()
        options = sorted([x for x in series.unique().tolist() if x != ""])
        if not options:
            continue

        selected = st.sidebar.multiselect(
            label=col,
            options=options,
            default=[],
            key=f"filter_{col}",
        )
        if selected:
            filtered = filtered[
                filtered[col].fillna("").astype(str).str.strip().isin(selected)
            ]

    return filtered


@st.cache_data(show_spinner=False)
def to_excel_bytes(df_main: pd.DataFrame, df_summary: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_main.to_excel(writer, index=False, sheet_name="Datos_filtrados")
        df_summary.to_excel(writer, index=False, sheet_name="Resumen_tipos")
    output.seek(0)
    return output.getvalue()


def human_int(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


# =========================
# INTERFAZ
# =========================
st.title("📚 Analizador interactivo de publicaciones")
st.caption(
    "Explora la base de datos, prioriza los tipos de publicación más relacionados con investigación y exporta los resultados."
)

with st.sidebar:
    st.markdown("## Carga de archivo")
    uploaded_file = st.file_uploader(
        "Sube el archivo Excel",
        type=["xlsx", "xls"],
        help="La app funciona a partir del archivo cargado por el usuario.",
    )

if uploaded_file is None:
    st.info("Sube un archivo Excel para comenzar.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    sheet_names = get_excel_sheets(file_bytes)
except Exception as e:
    st.error(f"No se pudo leer el archivo Excel: {e}")
    st.stop()

if not sheet_names:
    st.error("El archivo Excel no contiene hojas disponibles.")
    st.stop()

selected_sheet = st.sidebar.selectbox(
    "Hoja a analizar",
    options=sheet_names,
    index=sheet_names.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheet_names else 0,
)

try:
    with st.spinner("Procesando archivo..."):
        df = load_excel_sheet(file_bytes, selected_sheet)
except Exception as e:
    st.error(f"No se pudo cargar la hoja seleccionada: {e}")
    st.stop()

if df.empty:
    st.warning("La hoja seleccionada no contiene registros.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

required_columns = [TARGET_COL]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(
        "El archivo no contiene las columnas obligatorias: " + ", ".join(missing_columns)
    )
    st.stop()

available_filter_cols = [c for c in FILTER_COLUMNS if c in df.columns]

with st.sidebar:
    st.markdown("## Parámetros analíticos")
    visible_levels = st.multiselect(
        "Prioridad a mostrar",
        options=["Alta", "Media", "Baja"],
        default=["Alta", "Media", "Baja"],
    )
    score_threshold = st.slider(
        "Umbral mínimo de score de investigación",
        min_value=0,
        max_value=100,
        value=45,
        help="Permite quedarte con categorías más cercanas al ámbito de investigación.",
    )
    max_categories_chart = st.slider(
        "Máximo de categorías en gráficos",
        min_value=5,
        max_value=25,
        value=12,
    )

summary_types = build_research_priority_table(df, TARGET_COL)
priority_map = dict(
    zip(summary_types[TARGET_COL], summary_types["Prioridad_investigacion"])
)
score_map = dict(
    zip(summary_types[TARGET_COL], summary_types["Score_investigacion"])
)

df["Prioridad_investigacion"] = (
    df[TARGET_COL].fillna("Sin dato").astype(str).str.strip().map(priority_map)
)
df["Score_investigacion"] = (
    df[TARGET_COL].fillna("Sin dato").astype(str).str.strip().map(score_map)
)

if "clicked_publication_type" not in st.session_state:
    st.session_state.clicked_publication_type = None

selected_from_click = st.session_state.clicked_publication_type
if selected_from_click:
    st.info(f"Filtro activado desde gráfico: **{selected_from_click}**")
    if st.button("Quitar filtro del gráfico"):
        st.session_state.clicked_publication_type = None
        st.rerun()

df_filtered = df[
    (df["Score_investigacion"] >= score_threshold)
    & (df["Prioridad_investigacion"].isin(visible_levels))
].copy()

if st.session_state.clicked_publication_type:
    df_filtered = df_filtered[
        df_filtered[TARGET_COL].fillna("Sin dato").astype(str).str.strip()
        == st.session_state.clicked_publication_type
    ]

df_filtered = apply_filters(df_filtered, available_filter_cols)
summary_filtered = build_research_priority_table(df_filtered, TARGET_COL)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros visibles", human_int(len(df_filtered)))
c2.metric("Tipos de publicación visibles", human_int(df_filtered[TARGET_COL].nunique(dropna=True)))
c3.metric(
    "Score promedio",
    f"{df_filtered['Score_investigacion'].mean():.1f}" if len(df_filtered) else "0.0",
)
high_share = (
    (df_filtered["Prioridad_investigacion"] == "Alta").mean() * 100 if len(df_filtered) else 0
)
c4.metric("% prioridad alta", f"{high_share:.1f}%")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Ranking de clases según cercanía a investigación")
    st.dataframe(
        summary_types.style.format(
            {
                "Score_investigacion": "{:.1f}",
                "Similitud_positiva": "{:.3f}",
                "Similitud_negativa": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=420,
    )

with right:
    st.subheader("Top categorías priorizadas")
    chart_df = summary_types.head(max_categories_chart).copy()

    if not chart_df.empty:
        fig_bar = px.bar(
            chart_df,
            x="Score_investigacion",
            y=TARGET_COL,
            color="Prioridad_investigacion",
            orientation="h",
            hover_data={"Frecuencia": True, "Score_investigacion": ":.1f"},
        )
        fig_bar.update_layout(
            height=430,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Score de investigación",
            yaxis_title="Tipo de publicación",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        selected_points = plotly_events(
            fig_bar,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=430,
            key="bar_click",
        )
        if selected_points:
            point_index = selected_points[0]["pointIndex"]
            selected_category = chart_df.iloc[point_index][TARGET_COL]
            st.session_state.clicked_publication_type = selected_category
            st.rerun()
    else:
        st.info("No hay categorías para mostrar.")

st.divider()

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Distribución de registros por tipo de publicación")
    freq_chart = (
        summary_filtered.sort_values("Frecuencia", ascending=False)
        .head(max_categories_chart)
        .copy()
    )
    if len(freq_chart):
        fig_freq = px.bar(
            freq_chart,
            x=TARGET_COL,
            y="Frecuencia",
            color="Prioridad_investigacion",
            hover_data={"Score_investigacion": ":.1f"},
        )
        fig_freq.update_layout(
            height=430,
            xaxis_title="Tipo de publicación",
            yaxis_title="Número de registros",
            xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")

with col_b:
    st.subheader("Participación porcentual")
    if len(freq_chart):
        fig_pie = px.pie(
            freq_chart,
            names=TARGET_COL,
            values="Frecuencia",
            hole=0.35,
        )
        fig_pie.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")

if "General_ Año" in df_filtered.columns:
    temp_df = df_filtered.copy()
    temp_df["General_ Año"] = pd.to_numeric(temp_df["General_ Año"], errors="coerce")
    temp_df = temp_df.dropna(subset=["General_ Año"])
    if len(temp_df):
        st.subheader("Evolución temporal")
        time_summary = (
            temp_df.groupby(["General_ Año", TARGET_COL], dropna=False)
            .size()
            .reset_index(name="Frecuencia")
        )
        top_types = (
            summary_filtered.sort_values("Frecuencia", ascending=False)[TARGET_COL]
            .head(min(6, max_categories_chart))
            .tolist()
        )
        time_summary = time_summary[time_summary[TARGET_COL].isin(top_types)]
        if len(time_summary):
            fig_line = px.line(
                time_summary.sort_values("General_ Año"),
                x="General_ Año",
                y="Frecuencia",
                color=TARGET_COL,
                markers=True,
            )
            fig_line.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)

st.divider()

st.subheader("Base filtrada")
st.dataframe(df_filtered, use_container_width=True, height=500)

excel_bytes = to_excel_bytes(df_filtered, summary_filtered)
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        label="⬇️ Descargar Excel filtrado",
        data=excel_bytes,
        file_name="BD_filtrada_investigacion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with d2:
    st.download_button(
        label="⬇️ Descargar CSV filtrado",
        data=csv_bytes,
        file_name="BD_filtrada_investigacion.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Metodología aplicada para priorizar clases"):
    st.markdown(
        """
        **Cómo se calcula el score de investigación**

        La app combina tres componentes:
        1. **Coincidencia por palabras clave**: términos como *investigación*, *científico*, *tesis*, *artículo*, *proyecto de investigación*.
        2. **Similitud semántica TF-IDF**: compara cada categoría de `General_ Tipo de Publicación` contra prototipos positivos y negativos.
        3. **Frecuencia relativa**: favorece categorías que además tienen presencia real en la base.

        El resultado es un **score de 0 a 100**:
        - **Alta**: 70 a 100
        - **Media**: 45 a 69
        - **Baja**: 0 a 44

        Este criterio es **heurístico y ajustable**. Puedes cambiar el umbral desde la barra lateral.
        """
    )
