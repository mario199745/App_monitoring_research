import io
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# CONFIGURACION GENERAL
# =========================
st.set_page_config(
    page_title="Monitor bibliografico CONCYTEC/SERFOR",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent / "data"
GEOJSON_PATH = DATA_DIR / "GEO" / "DEP_PERU.geojson"

APP_FILE_PATTERN = "BD_APP_FINAL_*.xlsx"
APP_SHEET = "BD_APP"
TERRITORIAL_FILE_PATTERN = "BD_APP_TERRITORIAL_*.xlsx"
TERRITORIAL_MAP_SHEET = "MAPA_DEPARTAMENTOS"
TERRITORIAL_EXPANDED_SHEET = "REGIONES_EXPANDIDAS"

TYPE_COL = "TIPO_PUBLICACION_NORM"
CATEGORY_COL = "Categoria_Tesis_Articulo"
YEAR_COL = "General_ Año"
REGION_COL = "REGION_NORM_SUGERIDA"
UNIQUE_COL = "USAR_PARA_CONTEO_UNICO"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"
RECORD_ID_COL = "ID_REGISTRO_ANALISIS"

FILTER_COLUMNS = [
    "Nombre de Base de datos",
    "General_ Repositorio",
    CATEGORY_COL,
    TYPE_COL,
    "General_ Tipo de tesis Pre/Posgrado",
    "General_ Institución/Universidad",
    "General_ Idioma",
    "General_ Publicación Nacional/Extranjera",
    "General_ Tipo de contenido (TD=texto disponible, TN=Texto no disponible)",
    REGION_COL,
    "Ubicación_Provincia",
    "Ubicación_Distrito",
    "ANIFFS: Eje Temático",
    "ANIFFS: Área Temática",
    "ANIFFS: Linea de investigación",
]


# =========================
# UTILIDADES
# =========================
def normalize_key(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def latest_file(pattern: str) -> Path | None:
    if not DATA_DIR.exists():
        return None
    files = [p for p in DATA_DIR.glob(pattern) if not p.name.startswith("~$")]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_excel(path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, dtype=object)


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    for feature in geojson.get("features", []):
        props = feature.setdefault("properties", {})
        props["DEP_KEY"] = normalize_key(props.get("DEPARTAMEN"))

    return geojson


def human_int(value) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except Exception:
        return str(value)


def non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def chart_note(df_scope: pd.DataFrame, column: str, label: str) -> None:
    total = len(df_scope)
    if column not in df_scope.columns:
        st.caption(f"No se encontro la columna {label}.")
        return

    valid = int(non_empty_mask(df_scope[column]).sum())
    missing = total - valid
    coverage = valid / total * 100 if total else 0
    st.caption(
        f"Se muestran {human_int(valid)} registros con {label}. "
        f"No se muestran {human_int(missing)} sin informacion para esta variable. "
        f"Cobertura: {coverage:.1f}%."
    )


def value_counts_table(df_scope: pd.DataFrame, column: str, value_name: str = "Registros") -> pd.DataFrame:
    if column not in df_scope.columns:
        return pd.DataFrame(columns=[column, value_name])

    work = df_scope.loc[non_empty_mask(df_scope[column]), column].astype(str).str.strip()
    return work.value_counts().rename_axis(column).reset_index(name=value_name)


def apply_filters(df_scope: pd.DataFrame, filter_columns: list[str]) -> pd.DataFrame:
    filtered = df_scope.copy()
    st.sidebar.markdown("## Filtros")

    for col in filter_columns:
        if col not in filtered.columns:
            continue

        values = filtered.loc[non_empty_mask(filtered[col]), col].astype(str).str.strip()
        options = sorted(values.unique().tolist())
        if not options:
            continue

        selected = st.sidebar.multiselect(
            col,
            options=options,
            default=[],
            key=f"filter_{col}",
        )
        if selected:
            filtered = filtered[filtered[col].fillna("").astype(str).str.strip().isin(selected)]

    return filtered


def filtered_expanded_regions(df_scope: pd.DataFrame, expanded_regions: pd.DataFrame) -> pd.DataFrame:
    if RECORD_ID_COL not in df_scope.columns or RECORD_ID_COL not in expanded_regions.columns:
        return expanded_regions.iloc[0:0].copy()

    visible_ids = set(df_scope[RECORD_ID_COL].dropna().astype(str))
    expanded = expanded_regions.copy()
    expanded[RECORD_ID_COL] = expanded[RECORD_ID_COL].astype(str)
    return expanded[expanded[RECORD_ID_COL].isin(visible_ids)].copy()


def build_department_summary(df_scope: pd.DataFrame, expanded_regions: pd.DataFrame, map_base: pd.DataFrame) -> pd.DataFrame:
    expanded = filtered_expanded_regions(df_scope, expanded_regions)
    expanded = expanded[expanded.get("DEP_EN_GEOJSON", False).astype(bool)].copy()

    if expanded.empty:
        counts = pd.DataFrame(columns=["DEP_KEY"])
    else:
        counts = (
            expanded.groupby("DEP_KEY", dropna=False)
            .agg(
                registros_territoriales=(RECORD_ID_COL, "count"),
                publicaciones_bibliograficas=(MASTER_KEY_COL, "nunique"),
            )
            .reset_index()
        )

    summary = map_base[["IDDPTO", "DEPARTAMEN_GEO", "DEP_KEY", "CAPITAL"]].merge(
        counts,
        on="DEP_KEY",
        how="left",
    )
    for col in ["registros_territoriales", "publicaciones_bibliograficas"]:
        summary[col] = summary[col].fillna(0).astype(int)

    return summary


def territorial_note(df_scope: pd.DataFrame, expanded_regions: pd.DataFrame) -> None:
    expanded = filtered_expanded_regions(df_scope, expanded_regions)
    total = len(df_scope)
    with_region = expanded.loc[expanded.get("DEP_EN_GEOJSON", False).astype(bool), RECORD_ID_COL].nunique()
    without_region = total - int(with_region)
    coverage = with_region / total * 100 if total else 0
    st.caption(
        f"El mapa muestra {human_int(with_region)} registros con departamento normalizado. "
        f"No se muestran {human_int(without_region)} registros sin region cruzada al GeoJSON. "
        f"Cobertura territorial: {coverage:.1f}%."
    )


@st.cache_data(show_spinner=False)
def to_excel_bytes(df_main: pd.DataFrame, summary_type: pd.DataFrame, summary_region: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_main.to_excel(writer, index=False, sheet_name="Datos_filtrados")
        summary_type.to_excel(writer, index=False, sheet_name="Resumen_tipo")
        summary_region.to_excel(writer, index=False, sheet_name="Resumen_region")
    output.seek(0)
    return output.getvalue()


# =========================
# CARGA DE DATOS
# =========================
app_file = latest_file(APP_FILE_PATTERN)
territorial_file = latest_file(TERRITORIAL_FILE_PATTERN)

if app_file is None:
    st.error(f"No se encontro un archivo {APP_FILE_PATTERN} en la carpeta data.")
    st.stop()
if territorial_file is None:
    st.error(f"No se encontro un archivo {TERRITORIAL_FILE_PATTERN} en la carpeta data.")
    st.stop()
if not GEOJSON_PATH.exists():
    st.error(f"No se encontro el GeoJSON requerido: {GEOJSON_PATH}")
    st.stop()

try:
    df = load_excel(str(app_file), APP_SHEET)
    map_base = load_excel(str(territorial_file), TERRITORIAL_MAP_SHEET)
    expanded_regions = load_excel(str(territorial_file), TERRITORIAL_EXPANDED_SHEET)
    peru_geojson = load_geojson(str(GEOJSON_PATH))
except Exception as exc:
    st.error(f"No se pudo cargar la informacion: {exc}")
    st.stop()

df.columns = [str(col).strip() for col in df.columns]

required = [TYPE_COL, CATEGORY_COL, YEAR_COL, UNIQUE_COL, MASTER_KEY_COL, RECORD_ID_COL]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("La base final no contiene columnas requeridas: " + ", ".join(missing))
    st.stop()


# =========================
# INTERFAZ
# =========================
st.title("Monitor bibliografico CONCYTEC/SERFOR")
st.caption(
    "Explora publicaciones cientificas, tesis y articulos asociados a recursos forestales, "
    "biodiversidad y temas afines. La base puede leerse como publicaciones unicas para evitar "
    "sobreconteos, o como registros completos para conservar relaciones con regiones, temas, "
    "instituciones, especies y repositorios."
)

with st.sidebar:
    st.markdown("## Modo de analisis")
    analysis_mode = st.radio(
        "Unidad de conteo",
        options=["Registros completos", "Publicaciones unicas"],
        help=(
            "Publicaciones unicas usa USAR_PARA_CONTEO_UNICO = SI. "
            "Registros completos conserva relaciones territoriales, tematicas y de fuente."
        ),
    )
    max_categories_chart = st.slider("Maximo de categorias en graficos", 5, 25, 12)

df_mode = df.copy()
if analysis_mode == "Publicaciones unicas":
    df_mode = df_mode[df_mode[UNIQUE_COL].fillna("").astype(str).str.upper().eq("SI")].copy()

available_filter_cols = [col for col in FILTER_COLUMNS if col in df_mode.columns]
df_filtered = apply_filters(df_mode, available_filter_cols)

type_summary = value_counts_table(df_filtered, TYPE_COL)
category_summary = value_counts_table(df_filtered, CATEGORY_COL)
region_summary = build_department_summary(df_filtered, expanded_regions, map_base)

unique_publications = df_filtered[MASTER_KEY_COL].nunique(dropna=True)
additional_relations = max(len(df_filtered) - unique_publications, 0)
regions_with_data = int((region_summary["registros_territoriales"] > 0).sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros visibles", human_int(len(df_filtered)))
k2.metric("Publicaciones unicas", human_int(unique_publications))
k3.metric("Relaciones adicionales", human_int(additional_relations))
k4.metric("Departamentos con datos", human_int(regions_with_data))
k5.metric("Tipos normalizados", human_int(df_filtered[TYPE_COL].nunique(dropna=True)))

st.info(
    "Una misma publicacion puede estar asociada a varias regiones, especies, temas o fuentes. "
    "Use publicaciones unicas para conteos bibliograficos y registros completos para analisis territorial o tematico."
)

st.divider()

st.subheader("Distribucion territorial")
territorial_note(df_filtered, expanded_regions)

map_metric = (
    "publicaciones_bibliograficas"
    if analysis_mode == "Publicaciones unicas"
    else "registros_territoriales"
)

left_map, right_map = st.columns([1.35, 1])
with left_map:
    fig_map = px.choropleth(
        region_summary,
        geojson=peru_geojson,
        locations="DEP_KEY",
        featureidkey="properties.DEP_KEY",
        color=map_metric,
        hover_name="DEPARTAMEN_GEO",
        hover_data={
            "registros_territoriales": True,
            "publicaciones_bibliograficas": True,
            "DEP_KEY": False,
            "IDDPTO": False,
        },
        color_continuous_scale="YlGnBu",
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with right_map:
    st.markdown("#### Departamentos principales")
    top_regions = (
        region_summary.sort_values(map_metric, ascending=False)
        .head(max_categories_chart)
        .rename(columns={"DEPARTAMEN_GEO": "Departamento", map_metric: "Registros"})
    )
    fig_regions = px.bar(
        top_regions,
        x="Registros",
        y="Departamento",
        orientation="h",
        text="Registros",
    )
    fig_regions.update_layout(
        height=520,
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Registros" if analysis_mode == "Registros completos" else "Publicaciones",
        yaxis_title="Departamento",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_regions, use_container_width=True)

st.divider()

st.subheader("Resumen bibliografico")
col_a, col_b, col_c = st.columns([1, 1, 1.2])

with col_a:
    st.markdown("#### Tipo normalizado")
    chart_note(df_filtered, TYPE_COL, "tipo de publicacion")
    if len(type_summary):
        fig_type = px.bar(
            type_summary.head(max_categories_chart),
            x=TYPE_COL,
            y="Registros",
            color=TYPE_COL,
            text="Registros",
        )
        fig_type.update_layout(
            height=380,
            xaxis_title="Tipo de publicacion",
            yaxis_title="Registros",
            xaxis_tickangle=-25,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("No hay datos para mostrar.")

with col_b:
    st.markdown("#### Categoria tesis/articulo")
    chart_note(df_filtered, CATEGORY_COL, "categoria")
    if len(category_summary):
        fig_category = px.pie(
            category_summary,
            names=CATEGORY_COL,
            values="Registros",
            hole=0.4,
        )
        fig_category.update_traces(textposition="inside", textinfo="percent+label")
        fig_category.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.info("No hay datos para mostrar.")

with col_c:
    st.markdown("#### Repositorios principales")
    repo_col = "General_ Repositorio"
    repo_summary = value_counts_table(df_filtered, repo_col)
    chart_note(df_filtered, repo_col, "repositorio")
    if len(repo_summary):
        fig_repo = px.bar(
            repo_summary.head(max_categories_chart),
            x="Registros",
            y=repo_col,
            orientation="h",
            text="Registros",
        )
        fig_repo.update_layout(
            height=380,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Registros",
            yaxis_title="Repositorio",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_repo, use_container_width=True)
    else:
        st.info("No hay datos para mostrar.")

st.divider()

if YEAR_COL in df_filtered.columns:
    st.subheader("Evolucion temporal")
    year_df = df_filtered.copy()
    year_df[YEAR_COL] = pd.to_numeric(year_df[YEAR_COL], errors="coerce")
    year_valid = year_df.dropna(subset=[YEAR_COL]).copy()
    chart_note(df_filtered, YEAR_COL, "anio de publicacion")

    if len(year_valid):
        year_valid[YEAR_COL] = year_valid[YEAR_COL].astype(int)
        temporal = (
            year_valid.groupby([YEAR_COL, TYPE_COL], dropna=False)
            .size()
            .reset_index(name="Registros")
            .sort_values(YEAR_COL)
        )
        fig_time = px.line(
            temporal,
            x=YEAR_COL,
            y="Registros",
            color=TYPE_COL,
            markers=True,
        )
        fig_time.update_layout(
            height=430,
            xaxis_title="Anio",
            yaxis_title="Registros",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No hay registros con anio valido.")

st.divider()

topic_left, topic_right = st.columns(2)
with topic_left:
    topic_col = "ANIFFS: Área Temática"
    st.subheader("Areas tematicas")
    chart_note(df_filtered, topic_col, "area tematica")
    topic_summary = value_counts_table(df_filtered, topic_col)
    if len(topic_summary):
        fig_topic = px.bar(
            topic_summary.head(max_categories_chart),
            x="Registros",
            y=topic_col,
            orientation="h",
            text="Registros",
        )
        fig_topic.update_layout(
            height=430,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Registros",
            yaxis_title="Area tematica",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_topic, use_container_width=True)
    else:
        st.info("No hay datos para mostrar.")

with topic_right:
    line_col = "ANIFFS: Linea de investigación"
    st.subheader("Lineas de investigacion")
    chart_note(df_filtered, line_col, "linea de investigacion")
    line_summary = value_counts_table(df_filtered, line_col)
    if len(line_summary):
        fig_line = px.bar(
            line_summary.head(max_categories_chart),
            x="Registros",
            y=line_col,
            orientation="h",
            text="Registros",
        )
        fig_line.update_layout(
            height=430,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Registros",
            yaxis_title="Linea de investigacion",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No hay datos para mostrar.")

st.divider()

st.subheader("Tabla exploratoria")
default_columns = [
    "General_ Título",
    "General_ Autor(es)",
    YEAR_COL,
    TYPE_COL,
    REGION_COL,
    "General_ Institución/Universidad",
    "General_ Repositorio",
    "URL_PRINCIPAL_DETECTADA",
]
visible_columns = [col for col in default_columns if col in df_filtered.columns]
st.dataframe(df_filtered[visible_columns].head(500), use_container_width=True, height=380)
st.caption("La tabla muestra hasta 500 registros para mantener una navegacion fluida.")

excel_bytes = to_excel_bytes(df_filtered, type_summary, region_summary)
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Descargar Excel filtrado",
        data=excel_bytes,
        file_name="BD_app_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "Descargar CSV filtrado",
        data=csv_bytes,
        file_name="BD_app_filtrada.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Como interpretar la base"):
    st.markdown(
        """
        La base combina informacion bibliografica y relaciones de analisis. Una misma publicacion puede
        aparecer asociada a varias regiones, especies, temas o fuentes.

        - **Publicaciones unicas**: evita sobreconteos bibliograficos usando `USAR_PARA_CONTEO_UNICO = SI`.
        - **Registros completos**: conserva las relaciones territoriales, tematicas e institucionales.
        - **Mapa departamental**: usa solo registros con region normalizada y cruzada al GeoJSON del Peru.
        """
    )
