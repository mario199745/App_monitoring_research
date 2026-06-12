import io
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Revisión bibliográfica DEI",
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
YEAR_COL = "General_ Año"
REGION_COL = "REGION_NORM_SUGERIDA"
UNIQUE_COL = "USAR_PARA_CONTEO_UNICO"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"
RECORD_ID_COL = "ID_REGISTRO_ANALISIS"

SIMPLE_FILTERS = [
    ("Nombre de Base de datos", "Base de datos"),
    (TYPE_COL, "Tipo de publicación"),
    ("General_ Tipo de tesis Pre/Posgrado", "Nivel de tesis"),
    ("General_ Idioma", "Idioma"),
    ("General_ Publicación Nacional/Extranjera", "Ámbito de publicación"),
    (
        "General_ Tipo de contenido (TD=texto disponible, TN=Texto no disponible)",
        "Disponibilidad",
    ),
]

RELATION_CONFIG = {
    "repositorio": {
        "sheet": "DIM_REPOSITORIOS",
        "source": "General_ Repositorio",
        "label": "Repositorio",
    },
    "area": {
        "sheet": "DIM_AREAS_TEMATICAS",
        "source": "ANIFFS: Área Temática",
        "label": "Área temática",
    },
    "eje": {
        "sheet": "DIM_EJES_TEMATICOS",
        "source": "ANIFFS: Eje Temático",
        "label": "Eje temático",
    },
    "linea": {
        "sheet": "DIM_LINEAS_INVESTIGACION",
        "source": "ANIFFS: Linea de investigación",
        "label": "Línea de investigación",
    },
    "region": {
        "sheet": "DIM_REGIONES_NORMALIZADAS",
        "source": REGION_COL,
        "label": "Región normalizada",
    },
    "institucion": {
        "sheet": "DIM_INSTITUCIONES",
        "source": "General_ Institución/Universidad",
        "label": "Institución / universidad",
    },
}

REQUIRED_COLUMNS = [
    TYPE_COL,
    YEAR_COL,
    UNIQUE_COL,
    MASTER_KEY_COL,
    RECORD_ID_COL,
]


def normalize_key(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def latest_file(pattern: str) -> Path | None:
    files = [
        path
        for path in DATA_DIR.glob(pattern)
        if path.is_file() and not path.name.startswith("~$")
    ]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


@st.cache_data(show_spinner=False)
def load_excel(path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, dtype=object)


@st.cache_data(show_spinner=False)
def workbook_sheets(path: str) -> list[str]:
    return pd.ExcelFile(path, engine="openpyxl").sheet_names


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        geojson = json.load(file)
    for feature in geojson.get("features", []):
        properties = feature.setdefault("properties", {})
        properties["DEP_KEY"] = normalize_key(properties.get("DEPARTAMEN"))
    return geojson


def non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def human_int(value) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except Exception:
        return str(value)


def clean_relation(
    relation: pd.DataFrame,
    include_others: bool = True,
) -> pd.DataFrame:
    if relation.empty:
        return relation
    result = relation.copy()
    result[RECORD_ID_COL] = result[RECORD_ID_COL].astype(str)
    result["categoria"] = result["categoria"].astype(str).str.strip()
    result = result[result["categoria"].ne("")]
    if not include_others:
        result = result[result["categoria"].str.casefold().ne("otros")]
    return result


def visible_relation(
    relation: pd.DataFrame,
    df_scope: pd.DataFrame,
    include_others: bool = True,
) -> pd.DataFrame:
    visible_ids = set(df_scope[RECORD_ID_COL].dropna().astype(str))
    result = clean_relation(relation, include_others)
    return result[result[RECORD_ID_COL].isin(visible_ids)].copy()


def relation_summary(
    relation: pd.DataFrame,
    df_scope: pd.DataFrame,
    include_others: bool,
) -> pd.DataFrame:
    visible = visible_relation(relation, df_scope, include_others)
    if visible.empty:
        return pd.DataFrame(columns=["categoria", "Publicaciones"])
    return (
        visible.groupby("categoria")[RECORD_ID_COL]
        .nunique()
        .sort_values(ascending=False)
        .rename("Publicaciones")
        .reset_index()
    )


def simple_summary(df_scope: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df_scope.columns:
        return pd.DataFrame(columns=[column, "Publicaciones"])
    work = df_scope.loc[non_empty_mask(df_scope[column]), [RECORD_ID_COL, column]].copy()
    work[column] = work[column].astype(str).str.strip()
    return (
        work.groupby(column)[RECORD_ID_COL]
        .nunique()
        .sort_values(ascending=False)
        .rename("Publicaciones")
        .reset_index()
    )


def apply_simple_filters(df_scope: pd.DataFrame) -> pd.DataFrame:
    filtered = df_scope.copy()
    for column, label in SIMPLE_FILTERS:
        if column not in filtered.columns:
            continue
        values = filtered.loc[non_empty_mask(filtered[column]), column].astype(str).str.strip()
        options = sorted(values.unique().tolist())
        selected = st.sidebar.multiselect(label, options, key=f"filter_{column}")
        if selected:
            filtered = filtered[
                filtered[column].fillna("").astype(str).str.strip().isin(selected)
            ]
    return filtered


def apply_relation_filters(
    df_scope: pd.DataFrame,
    relations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    filtered = df_scope.copy()
    for key in ["repositorio", "region", "institucion", "eje", "area"]:
        config = RELATION_CONFIG[key]
        relation = relations.get(key, pd.DataFrame())
        if relation.empty:
            continue
        available = visible_relation(relation, filtered, include_others=True)
        options = sorted(available["categoria"].dropna().unique().tolist())
        selected = st.sidebar.multiselect(
            config["label"],
            options,
            key=f"relation_filter_{key}",
        )
        if selected:
            selected_ids = set(
                relation.loc[
                    relation["categoria"].astype(str).str.strip().isin(selected),
                    RECORD_ID_COL,
                ].astype(str)
            )
            filtered = filtered[
                filtered[RECORD_ID_COL].astype(str).isin(selected_ids)
            ]
    return filtered


def filtered_expanded_regions(
    df_scope: pd.DataFrame,
    expanded_regions: pd.DataFrame,
) -> pd.DataFrame:
    visible_ids = set(df_scope[RECORD_ID_COL].dropna().astype(str))
    expanded = expanded_regions.copy()
    expanded[RECORD_ID_COL] = expanded[RECORD_ID_COL].astype(str)
    return expanded[expanded[RECORD_ID_COL].isin(visible_ids)].copy()


def department_summary(
    df_scope: pd.DataFrame,
    expanded_regions: pd.DataFrame,
    map_base: pd.DataFrame,
) -> pd.DataFrame:
    expanded = filtered_expanded_regions(df_scope, expanded_regions)
    expanded = expanded[expanded.get("DEP_EN_GEOJSON", False).astype(bool)].copy()
    if expanded.empty:
        counts = pd.DataFrame(columns=["DEP_KEY", "Publicaciones"])
    else:
        counts = (
            expanded.groupby("DEP_KEY")[RECORD_ID_COL]
            .nunique()
            .rename("Publicaciones")
            .reset_index()
        )
    summary = map_base[["IDDPTO", "DEPARTAMEN_GEO", "DEP_KEY", "CAPITAL"]].merge(
        counts,
        on="DEP_KEY",
        how="left",
    )
    summary["Publicaciones"] = summary["Publicaciones"].fillna(0).astype(int)
    return summary


@st.cache_data(show_spinner=False)
def to_excel_bytes(
    df_main: pd.DataFrame,
    summary_type: pd.DataFrame,
    summary_region: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_main.to_excel(writer, index=False, sheet_name="Datos_filtrados")
        summary_type.to_excel(writer, index=False, sheet_name="Resumen_tipo")
        summary_region.to_excel(writer, index=False, sheet_name="Resumen_region")
    output.seek(0)
    return output.getvalue()


def horizontal_bar(data: pd.DataFrame, category: str, title: str, limit: int):
    plot_data = data.head(limit).sort_values("Publicaciones")
    figure = px.bar(
        plot_data,
        x="Publicaciones",
        y=category,
        orientation="h",
        text="Publicaciones",
        title=title,
        color_discrete_sequence=["#256d5b"],
    )
    figure.update_layout(
        height=430,
        showlegend=False,
        xaxis_title="Publicaciones",
        yaxis_title="",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return figure


app_file = latest_file(APP_FILE_PATTERN)
territorial_file = latest_file(TERRITORIAL_FILE_PATTERN)

if app_file is None or territorial_file is None:
    st.error(
        "No se encontró el par de bases principal y territorial en la carpeta data."
    )
    st.stop()
if not GEOJSON_PATH.exists():
    st.error(f"No se encontró el GeoJSON requerido: {GEOJSON_PATH}")
    st.stop()

try:
    df = load_excel(str(app_file), APP_SHEET)
    app_sheets = workbook_sheets(str(app_file))
    map_base = load_excel(str(territorial_file), TERRITORIAL_MAP_SHEET)
    expanded_regions = load_excel(
        str(territorial_file),
        TERRITORIAL_EXPANDED_SHEET,
    )
    peru_geojson = load_geojson(str(GEOJSON_PATH))
    relations = {}
    for key, config in RELATION_CONFIG.items():
        if config["sheet"] in app_sheets:
            relations[key] = clean_relation(
                load_excel(str(app_file), config["sheet"]),
                include_others=True,
            )
        else:
            relations[key] = pd.DataFrame()
except Exception as exc:
    st.error(f"No se pudo cargar la información: {exc}")
    st.stop()

df.columns = [str(column).strip() for column in df.columns]
missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing:
    st.error("La base final no contiene columnas requeridas: " + ", ".join(missing))
    st.stop()

df_mode = df.copy()
df_mode[RECORD_ID_COL] = df_mode[RECORD_ID_COL].astype(str)
if UNIQUE_COL in df_mode.columns:
    df_mode = df_mode[
        df_mode[UNIQUE_COL].fillna("").astype(str).str.upper().eq("SI")
    ].copy()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f5f8f7;
        border: 1px solid #dbe7e3;
        padding: 0.8rem;
        border-radius: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Revisión bibliográfica DEI")
st.caption(
    "Explora publicaciones sobre recursos forestales, biodiversidad y "
    "fauna silvestre. Las categorías múltiples se contabilizan por separado."
)

with st.sidebar:
    st.markdown("## Filtros")
    max_categories_chart = st.slider("Categorías en gráficos", 5, 25, 12)
    include_others = st.toggle("Mostrar 'Otros' en gráficos", value=True)
    st.caption(f"Base: {app_file.name}")

df_filtered = apply_simple_filters(df_mode)
df_filtered = apply_relation_filters(df_filtered, relations)

type_summary = simple_summary(df_filtered, TYPE_COL)
database_summary = simple_summary(df_filtered, "Nombre de Base de datos")
region_summary = department_summary(df_filtered, expanded_regions, map_base)

repo_summary = relation_summary(
    relations.get("repositorio", pd.DataFrame()),
    df_filtered,
    include_others,
)
area_summary = relation_summary(
    relations.get("area", pd.DataFrame()),
    df_filtered,
    include_others,
)
eje_summary = relation_summary(
    relations.get("eje", pd.DataFrame()),
    df_filtered,
    include_others,
)

unique_publications = df_filtered[RECORD_ID_COL].nunique()
regions_with_data = int(region_summary["Publicaciones"].gt(0).sum())
years = pd.to_numeric(df_filtered[YEAR_COL], errors="coerce")
period = (
    f"{int(years.min())}-{int(years.max())}"
    if years.notna().any()
    else "Sin año"
)
area_count = (
    int(
        area_summary.loc[
            area_summary["categoria"].astype(str).str.casefold().ne("otros"),
            "categoria",
        ].nunique()
    )
    if not area_summary.empty
    else 0
)
database_count = (
    int(df_filtered["Nombre de Base de datos"].dropna().astype(str).nunique())
    if "Nombre de Base de datos" in df_filtered.columns
    else 0
)

metrics = st.columns(5)
metrics[0].metric("Publicaciones", human_int(unique_publications))
metrics[1].metric("Departamentos", human_int(regions_with_data))
metrics[2].metric("Periodo de publicación/aprobación", period)
metrics[3].metric("Áreas temáticas", human_int(area_count))
metrics[4].metric("Bases documentales", human_int(database_count))

st.caption(
    "Una publicación puede pertenecer a varias áreas, ejes, líneas, regiones "
    "o repositorios; por ello, las categorías no son excluyentes."
)

tabs = st.tabs(["General", "Territorio", "Tiempo", "Temas", "Datos"])

with tabs[0]:
    st.subheader("Resumen bibliográfico")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if not type_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    type_summary,
                    TYPE_COL,
                    "Tipo de publicación",
                    max_categories_chart,
                ),
                width="stretch",
            )
        else:
            st.info("No hay datos para mostrar.")
    with col_b:
        if not database_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    database_summary,
                    "Nombre de Base de datos",
                    "Publicaciones por base documental",
                    max_categories_chart,
                ),
                width="stretch",
            )
        else:
            st.info("No hay datos para mostrar.")
    with col_c:
        if not repo_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    repo_summary,
                    "categoria",
                    "Repositorios",
                    max_categories_chart,
                ),
                width="stretch",
            )
        else:
            st.info("No hay relaciones de repositorio disponibles.")

with tabs[1]:
    st.subheader("Distribución territorial")
    territorial_rows = filtered_expanded_regions(df_filtered, expanded_regions)
    publications_with_department = territorial_rows.loc[
        territorial_rows.get("DEP_EN_GEOJSON", False).astype(bool),
        RECORD_ID_COL,
    ].nunique()
    st.caption(
        f"El mapa representa {human_int(publications_with_department)} "
        "publicaciones con departamento identificado."
    )
    map_figure = px.choropleth(
        region_summary,
        geojson=peru_geojson,
        locations="DEP_KEY",
        featureidkey="properties.DEP_KEY",
        color="Publicaciones",
        hover_name="DEPARTAMEN_GEO",
        hover_data={"DEP_KEY": False, "IDDPTO": False},
        color_continuous_scale="YlGnBu",
    )
    map_figure.update_geos(fitbounds="locations", visible=False)
    map_figure.update_layout(height=680, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(map_figure, width="stretch")

    top_regions = region_summary.sort_values(
        "Publicaciones", ascending=False
    ).head(max_categories_chart)
    if not top_regions.empty:
        st.plotly_chart(
            horizontal_bar(
                top_regions.rename(columns={"DEPARTAMEN_GEO": "Departamento"}),
                "Departamento",
                "Departamentos con más publicaciones",
                max_categories_chart,
            ),
            width="stretch",
        )

with tabs[2]:
    st.subheader("Evolución temporal")
    temporal = df_filtered[[RECORD_ID_COL, YEAR_COL, TYPE_COL]].copy()
    temporal[YEAR_COL] = pd.to_numeric(temporal[YEAR_COL], errors="coerce")
    temporal = temporal.dropna(subset=[YEAR_COL])
    if temporal.empty:
        st.info("No hay publicaciones con año válido.")
    else:
        temporal[YEAR_COL] = temporal[YEAR_COL].astype(int)
        temporal_summary = (
            temporal.groupby([YEAR_COL, TYPE_COL])[RECORD_ID_COL]
            .nunique()
            .rename("Publicaciones")
            .reset_index()
        )
        time_figure = px.line(
            temporal_summary,
            x=YEAR_COL,
            y="Publicaciones",
            color=TYPE_COL,
            markers=True,
        )
        time_figure.update_layout(
            height=450,
            xaxis_title="Año de publicación o aprobación",
            yaxis_title="Publicaciones",
        )
        st.plotly_chart(time_figure, width="stretch")

with tabs[3]:
    st.subheader("Temas e instituciones")
    topic_left, topic_right = st.columns(2)
    with topic_left:
        if not area_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    area_summary,
                    "categoria",
                    "Publicaciones vinculadas por área temática",
                    max_categories_chart,
                ),
                width="stretch",
            )
        else:
            st.info("No hay áreas temáticas para mostrar.")
    with topic_right:
        institution_summary = relation_summary(
            relations.get("institucion", pd.DataFrame()),
            df_filtered,
            include_others,
        )
        if not institution_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    institution_summary,
                    "categoria",
                    "Instituciones y universidades",
                    max_categories_chart,
                ),
                width="stretch",
            )
        else:
            st.info("No hay instituciones para mostrar.")

    with st.expander("Ejes temáticos"):
        if not eje_summary.empty:
            st.plotly_chart(
                horizontal_bar(
                    eje_summary,
                    "categoria",
                    "Publicaciones vinculadas por eje temático",
                    max_categories_chart,
                ),
                width="stretch",
            )

with tabs[4]:
    st.subheader("Tabla exploratoria")
    visible_columns = [
        "General_ Título",
        "General_ Autor(es)",
        YEAR_COL,
        TYPE_COL,
        REGION_COL,
        "General_ Institución/Universidad",
        "General_ Repositorio",
        "DOI_NORM",
        "General_ Enlace",
    ]
    visible_columns = [column for column in visible_columns if column in df_filtered]
    st.dataframe(
        df_filtered[visible_columns].head(500),
        width="stretch",
        height=400,
        hide_index=True,
    )
    st.caption(
        "La tabla muestra hasta 500 registros. Las descargas contienen todos "
        "los registros resultantes de los filtros."
    )

    excel_bytes = to_excel_bytes(df_filtered, type_summary, region_summary)
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Descargar Excel filtrado",
        data=excel_bytes,
        file_name="BD_app_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    download_b.download_button(
        "Descargar CSV filtrado",
        data=csv_bytes,
        file_name="BD_app_filtrada.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
st.caption(
    f"Fuente principal: {app_file.name} · Fuente territorial: "
    f"{territorial_file.name}"
)
