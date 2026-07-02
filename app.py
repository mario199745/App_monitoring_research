import io
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

INSTITUTION_CLASS_COL = "CLASE_INSTITUCION"
IS_UNIVERSITY_COL = "ES_UNIVERSIDAD"
PUBLIC_INSTITUTION_CLASS_COL = "CLASE_INSTITUCION_PUBLICA"
PUBLIC_UNIVERSITY_SUBCLASS_COL = "SUBCLASE_UNIVERSIDAD_PUBLICA"
REPOSITORY_CLASS_COL = "CLASE_REPOSITORIO"
UNIVERSITY_REPOSITORY_COL = "ES_REPOSITORIO_UNIVERSITARIO"
PUBLIC_REPOSITORY_CLASS_COL = "CLASE_REPOSITORIO_PUBLICA"


st.set_page_config(
    page_title="MONITOREO DE INVESTIGACION",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent / "data"
GEOJSON_PATH = DATA_DIR / "GEO" / "DEP_PERU.geojson"

APP_FILE_PATTERN = "BD_APP_FINAL_*.xlsx"
APP_SHEET = "BD_APP"
TERRITORIAL_MAP_SHEET = "MAPA_DEPARTAMENTOS"
TERRITORIAL_EXPANDED_SHEET = "REGIONES_EXPANDIDAS"

SOURCE_TYPE_COL = "TIPO_PUBLICACION_NORM"
TYPE_COL = "TIPO_PUBLICACION_PUBLICO"
SUBTYPE_COL = "SUBTIPO_PUBLICACION_PUBLICO"
POSTGRAD_DETAIL_COL = "DETALLE_TESIS_POSGRADO_PUBLICO"
YEAR_COL = "General_ Año"
REGION_COL = "REGION_NORM_SUGERIDA"
ACADEMIC_GRADE_COL = "GRADO_ACADEMICO_PUBLICO"
ACADEMIC_LEVEL_COL = "NIVEL_ACADEMICO_PUBLICO"
UNIQUE_COL = "USAR_PARA_CONTEO_UNICO"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"
RECORD_ID_COL = "ID_PUBLICACION_PROPUESTA"
REGION_FILTER_KEY = "relation_filter_region"
MAP_PENDING_REGIONS_KEY = "_map_pending_regions"
MAP_LAST_SELECTION_KEY = "_map_last_selection"
MAP_WIDGET_VERSION_KEY = "_map_widget_version"
PUBLICATION_CHART_LEVEL_KEY = "_publication_chart_level"
PUBLICATION_CHART_TYPE_KEY = "_publication_chart_type"
PUBLICATION_CHART_SUBTYPE_KEY = "_publication_chart_subtype"
PUBLICATION_DETAIL_FILTER_KEY = "_publication_detail_filter"
PUBLICATION_CHART_PENDING_KEY = "_publication_chart_pending"
PUBLICATION_CHART_VERSION_KEY = "_publication_chart_version"
REPOSITORY_CHART_LEVEL_KEY = "_repository_chart_level"
REPOSITORY_CHART_CLASS_KEY = "_repository_chart_class"
REPOSITORY_CHART_PENDING_KEY = "_repository_chart_pending"
REPOSITORY_CHART_VERSION_KEY = "_repository_chart_version"
DATABASE_CHART_PENDING_KEY = "_database_chart_pending"
DATABASE_CHART_VERSION_KEY = "_database_chart_version"
DATABASE_CHART_LAST_SELECTION_KEY = "_database_chart_last_selection"

SIMPLE_FILTERS = [
    (TYPE_COL, "Tipo de publicación"),
    ("General_ Idioma", "Idioma"),
    ("General_ Publicación Nacional/Extranjera", "Ámbito de publicación"),
]

RELATION_CONFIG = {
    "base": {
        "sheet": "DIM_BASES_DOCUMENTALES",
        "source": "Nombre de Base de datos",
        "label": "Base documental",
    },
    "repositorio": {
        "sheet": "DIM_REPOSITORIOS",
        "source": "General_ Repositorio",
        "label": "Repositorio",
    },
    "repositorio_clase": {
        "sheet": "DIM_REPOSITORIOS",
        "source": PUBLIC_REPOSITORY_CLASS_COL,
        "label": "Clase de repositorio",
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
    "institucion_clase": {
        "sheet": "DIM_INSTITUCIONES",
        "source": PUBLIC_INSTITUTION_CLASS_COL,
        "label": "Clase de institución",
    },
}

REQUIRED_COLUMNS = [
    TYPE_COL,
    YEAR_COL,
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


def normalize_publication_hierarchy(data: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que los eventos sean subcategorías de Artículo al cargar."""
    result = data.copy()
    if TYPE_COL not in result.columns or SUBTYPE_COL not in result.columns:
        return result
    event_subtypes = {
        "Artículo de conferencia",
        "Ponencia o memoria de evento",
        "Publicación de evento científico",
    }
    event_mask = (
        result[TYPE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("Publicación de evento científico")
        | result[SUBTYPE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(event_subtypes)
    )
    result.loc[event_mask, TYPE_COL] = "Artículo"
    result.loc[event_mask, SUBTYPE_COL] = "Publicación de evento científico"
    if POSTGRAD_DETAIL_COL not in result.columns:
        result[POSTGRAD_DETAIL_COL] = pd.NA
    master_mask = result[SUBTYPE_COL].eq("Tesis de maestría")
    doctoral_mask = result[SUBTYPE_COL].eq("Tesis doctoral")
    result.loc[master_mask, POSTGRAD_DETAIL_COL] = "Tesis de maestría"
    result.loc[doctoral_mask, POSTGRAD_DETAIL_COL] = "Tesis doctoral"
    result.loc[master_mask | doctoral_mask, SUBTYPE_COL] = "Tesis de posgrado"
    postgraduate_mask = result[SUBTYPE_COL].isin(
        ["Tesis de posgrado", "Tesis de posgrado no especificada"]
    )
    result.loc[postgraduate_mask, SUBTYPE_COL] = "Tesis de posgrado"
    missing_detail = postgraduate_mask & result[POSTGRAD_DETAIL_COL].isna()
    result.loc[
        missing_detail & result[ACADEMIC_LEVEL_COL].eq("Maestría"),
        POSTGRAD_DETAIL_COL,
    ] = "Tesis de maestría"
    result.loc[
        missing_detail & result[ACADEMIC_LEVEL_COL].eq("Doctorado"),
        POSTGRAD_DETAIL_COL,
    ] = "Tesis doctoral"
    result.loc[
        postgraduate_mask & result[POSTGRAD_DETAIL_COL].isna(),
        POSTGRAD_DETAIL_COL,
    ] = "No identificados"
    return result


def human_int(value) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
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


def enrich_repository_relation(relation: pd.DataFrame) -> pd.DataFrame:
    if relation.empty:
        return relation
    result = relation.copy()
    if REPOSITORY_CLASS_COL not in result.columns:
        result[REPOSITORY_CLASS_COL] = "Otro / no clasificado"
        result[UNIVERSITY_REPOSITORY_COL] = "Indeterminado"
    if PUBLIC_REPOSITORY_CLASS_COL not in result.columns:
        result[PUBLIC_REPOSITORY_CLASS_COL] = "Otros"
    return result


def enrich_institution_relation(relation: pd.DataFrame) -> pd.DataFrame:
    if relation.empty:
        return relation
    result = relation.copy()
    if INSTITUTION_CLASS_COL not in result.columns:
        result[INSTITUTION_CLASS_COL] = "Otro / no clasificado"
        result[IS_UNIVERSITY_COL] = "Indeterminado"
    if PUBLIC_INSTITUTION_CLASS_COL not in result.columns:
        public_mapping = {
            "Universidad publica nacional": ("Universidad nacional", "Pública"),
            "Universidad privada nacional": ("Universidad nacional", "Privada"),
            "Universidad extranjera": ("Universidad extranjera", "Extranjera"),
            "Instituto publico / entidad estatal": (
                "Instituto público / entidad estatal",
                "No aplica",
            ),
            "Centro de investigacion / cooperacion": (
                "Centro de investigación / cooperación",
                "No aplica",
            ),
            "Sociedad cientifica / asociacion": (
                "Centro de investigación / cooperación",
                "No aplica",
            ),
            "Revista / boletin mal ubicado": (
                "Revista / boletín mal ubicado",
                "No aplica",
            ),
            "Otro / no clasificado": ("Otros", "No aplica"),
        }
        public_values = result[INSTITUTION_CLASS_COL].map(public_mapping)
        result[PUBLIC_INSTITUTION_CLASS_COL] = public_values.map(
            lambda item: item[0]
        )
        result[PUBLIC_UNIVERSITY_SUBCLASS_COL] = public_values.map(
            lambda item: item[1]
        )
    return result


def relation_from_column(
    relation: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if relation.empty or column not in relation.columns:
        return pd.DataFrame(columns=[RECORD_ID_COL, "categoria"])
    result = relation[[RECORD_ID_COL, column]].dropna().copy()
    result["categoria"] = result[column].astype(str).str.strip()
    result = result[result["categoria"].ne("")]
    return result[[RECORD_ID_COL, "categoria"]].drop_duplicates()


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


def repository_summary_by_class(
    relation: pd.DataFrame,
    df_scope: pd.DataFrame,
    repository_class: str,
    include_others: bool,
) -> pd.DataFrame:
    if relation.empty or PUBLIC_REPOSITORY_CLASS_COL not in relation.columns:
        return pd.DataFrame(columns=["categoria", "Publicaciones"])
    filtered_relation = relation[
        relation[PUBLIC_REPOSITORY_CLASS_COL]
        .astype(str)
        .str.strip()
        .eq(repository_class)
    ].copy()
    return relation_summary(filtered_relation, df_scope, include_others)


def institution_summary_by_class(
    relation: pd.DataFrame,
    df_scope: pd.DataFrame,
    institution_class: str,
    include_others: bool,
    class_column: str = INSTITUTION_CLASS_COL,
) -> pd.DataFrame:
    if relation.empty or class_column not in relation.columns:
        return pd.DataFrame(columns=["categoria", "Publicaciones"])
    filtered_relation = relation[
        relation[class_column].astype(str).str.strip().eq(institution_class)
    ].copy()
    return relation_summary(filtered_relation, df_scope, include_others)


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


def publication_drilldown_summary(
    filtered_scope: pd.DataFrame,
    full_scope: pd.DataFrame,
    publication_type: str | None,
    category_column: str,
    publication_subtype: str | None = None,
) -> pd.DataFrame:
    """Resume un nivel y evita que filtros residuales oculten la jerarquía."""
    summary = simple_summary(filtered_scope, category_column)
    if not summary.empty or not publication_type:
        return summary

    fallback_scope = full_scope[
        full_scope[TYPE_COL].fillna("").astype(str).str.strip().eq(publication_type)
    ]
    if publication_subtype:
        fallback_scope = fallback_scope[
            fallback_scope[SUBTYPE_COL]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(publication_subtype)
        ]
    return simple_summary(fallback_scope, category_column)


def apply_simple_filters(df_scope: pd.DataFrame) -> pd.DataFrame:
    filtered = df_scope.copy()
    selected_types = []
    for column, label in SIMPLE_FILTERS:
        if column not in filtered.columns:
            continue
        values = filtered.loc[non_empty_mask(filtered[column]), column].astype(str).str.strip()
        options = sorted(values.unique().tolist())
        selected = st.sidebar.multiselect(label, options, key=f"filter_{column}")
        if column == TYPE_COL:
            selected_types = selected
        if selected:
            filtered = filtered[
                filtered[column].fillna("").astype(str).str.strip().isin(selected)
            ]
    if "Tesis" in selected_types:
        for column, label in [
            (ACADEMIC_GRADE_COL, "Grado académico"),
            (ACADEMIC_LEVEL_COL, "Nivel académico"),
        ]:
            values = (
                filtered.loc[non_empty_mask(filtered[column]), column]
                .astype(str)
                .str.strip()
            )
            options = sorted(values.unique().tolist())
            selected = st.sidebar.multiselect(
                label,
                options,
                key=f"filter_{column}",
            )
            if selected:
                filtered = filtered[
                    filtered[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .isin(selected)
                ]
    if selected_types:
        values = (
            filtered.loc[non_empty_mask(filtered[SUBTYPE_COL]), SUBTYPE_COL]
            .astype(str)
            .str.strip()
        )
        selected = st.sidebar.multiselect(
            "Subtipo de publicación",
            sorted(values.unique().tolist()),
            key=f"filter_{SUBTYPE_COL}",
        )
        if selected:
            filtered = filtered[
                filtered[SUBTYPE_COL]
                .fillna("")
                .astype(str)
                .str.strip()
                .isin(selected)
            ]
    return filtered


def apply_relation_filters(
    df_scope: pd.DataFrame,
    relations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    filtered = df_scope.copy()
    for key in [
        "base",
        "repositorio_clase",
        "repositorio",
        "region",
        "institucion_clase",
        "institucion",
        "eje",
        "area",
    ]:
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


def selected_map_region_keys(event) -> tuple[str, ...]:
    if event is None:
        return ()
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points", [])

    selected = []
    for point in points or []:
        location = point.get("location") if isinstance(point, dict) else None
        if not location and isinstance(point, dict):
            customdata = point.get("customdata")
            if isinstance(customdata, (list, tuple)) and customdata:
                location = customdata[0]
        key = normalize_key(location)
        if key:
            selected.append(key)
    return tuple(sorted(set(selected)))


def selected_bar_labels(event) -> tuple[str, ...]:
    """Extrae las categorías seleccionadas de una barra horizontal Plotly."""
    if event is None:
        return ()
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points", [])
    labels = [
        str(point.get("y")).strip()
        for point in points or []
        if isinstance(point, dict) and point.get("y") is not None
    ]
    return tuple(dict.fromkeys(label for label in labels if label))


def map_regions_to_filter_values(
    region_keys: tuple[str, ...],
    region_relation: pd.DataFrame,
) -> list[str]:
    if not region_keys or region_relation.empty:
        return []
    categories = region_relation["categoria"].dropna().astype(str).str.strip()
    lookup = {}
    for category in categories:
        lookup.setdefault(normalize_key(category), category)
    return [lookup[key] for key in region_keys if key in lookup]


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
    territorial_detail: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_main.to_excel(writer, index=False, sheet_name="Datos_filtrados")
        summary_type.to_excel(writer, index=False, sheet_name="Resumen_tipo")
        summary_region.to_excel(writer, index=False, sheet_name="Resumen_region")
        territorial_detail.to_excel(
            writer,
            index=False,
            sheet_name="Detalle_territorial",
        )
    output.seek(0)
    return output.getvalue()


def horizontal_bar(data: pd.DataFrame, category: str, title: str, limit: int):
    plot_data = data.head(limit).sort_values("Publicaciones").copy()
    plot_data["Publicaciones_formato"] = plot_data["Publicaciones"].map(
        human_int
    )
    figure = px.bar(
        plot_data,
        x="Publicaciones",
        y=category,
        orientation="h",
        text="Publicaciones_formato",
        title=title,
        color_discrete_sequence=["#256d5b"],
        labels={"Publicaciones": "N° de Publicaciones"},
    )
    figure.update_layout(
        height=430,
        showlegend=False,
        xaxis_title="N° de Publicaciones",
        yaxis_title="",
        margin=dict(l=10, r=10, t=45, b=10),
        separators=", ",
        clickmode="event+select",
    )
    figure.update_traces(
        texttemplate="%{text}",
        hovertemplate=(
            f"{category}: %{{y}}<br>N° de Publicaciones: %{{x:,.0f}}<extra></extra>"
        ),
    )
    return figure


app_file = latest_file(APP_FILE_PATTERN)
territorial_file = app_file

if app_file is None:
    st.error("No se encontró la base maestra integrada en la carpeta data.")
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
    relations["repositorio"] = enrich_repository_relation(
        relations.get("repositorio", pd.DataFrame())
    )
    relations["repositorio_clase"] = relation_from_column(
        relations["repositorio"],
        PUBLIC_REPOSITORY_CLASS_COL,
    )
    relations["institucion"] = enrich_institution_relation(
        relations.get("institucion", pd.DataFrame())
    )
    relations["institucion_clase"] = relation_from_column(
        relations["institucion"],
        PUBLIC_INSTITUTION_CLASS_COL,
    )
except Exception as exc:
    st.error(f"No se pudo cargar la información: {exc}")
    st.stop()

df.columns = [str(column).strip() for column in df.columns]
df = normalize_publication_hierarchy(df)
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

if MAP_PENDING_REGIONS_KEY in st.session_state:
    st.session_state[REGION_FILTER_KEY] = st.session_state.pop(
        MAP_PENDING_REGIONS_KEY
    )

if PUBLICATION_CHART_PENDING_KEY in st.session_state:
    pending_publication = st.session_state.pop(PUBLICATION_CHART_PENDING_KEY)
    action = pending_publication.get("action")
    if action == "type":
        selected_type = pending_publication["value"]
        st.session_state[f"filter_{TYPE_COL}"] = [selected_type]
        st.session_state[f"filter_{SUBTYPE_COL}"] = []
        st.session_state[PUBLICATION_CHART_TYPE_KEY] = selected_type
        st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "subtypes"
    elif action == "subtype":
        st.session_state[f"filter_{SUBTYPE_COL}"] = [
            pending_publication["value"]
        ]
        if pending_publication["value"] == "Tesis de posgrado":
            st.session_state[PUBLICATION_CHART_SUBTYPE_KEY] = (
                "Tesis de posgrado"
            )
            st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "postgraduate"
            st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None
    elif action == "detail":
        st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = pending_publication[
            "value"
        ]
    elif action == "clear_subtype":
        st.session_state[f"filter_{SUBTYPE_COL}"] = []
    elif action == "clear_detail":
        st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None
    elif action == "back_to_subtypes":
        st.session_state[f"filter_{SUBTYPE_COL}"] = []
        st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None
        st.session_state.pop(PUBLICATION_CHART_SUBTYPE_KEY, None)
        st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "subtypes"
    elif action == "back":
        st.session_state[f"filter_{TYPE_COL}"] = []
        st.session_state[f"filter_{SUBTYPE_COL}"] = []
        st.session_state.pop(PUBLICATION_CHART_TYPE_KEY, None)
        st.session_state.pop(PUBLICATION_CHART_SUBTYPE_KEY, None)
        st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None
        st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "types"
    st.session_state[PUBLICATION_CHART_VERSION_KEY] = (
        st.session_state.get(PUBLICATION_CHART_VERSION_KEY, 0) + 1
    )

if REPOSITORY_CHART_PENDING_KEY in st.session_state:
    pending_repository = st.session_state.pop(REPOSITORY_CHART_PENDING_KEY)
    action = pending_repository.get("action")
    if action == "class":
        selected_class = pending_repository["value"]
        st.session_state["relation_filter_repositorio_clase"] = [
            selected_class
        ]
        st.session_state["relation_filter_repositorio"] = []
        st.session_state[REPOSITORY_CHART_CLASS_KEY] = selected_class
        st.session_state[REPOSITORY_CHART_LEVEL_KEY] = "repositories"
    elif action == "repository":
        st.session_state["relation_filter_repositorio"] = [
            pending_repository["value"]
        ]
    elif action == "clear_repository":
        st.session_state["relation_filter_repositorio"] = []
    elif action == "back":
        st.session_state["relation_filter_repositorio_clase"] = []
        st.session_state["relation_filter_repositorio"] = []
        st.session_state.pop(REPOSITORY_CHART_CLASS_KEY, None)
        st.session_state[REPOSITORY_CHART_LEVEL_KEY] = "classes"
    st.session_state[REPOSITORY_CHART_VERSION_KEY] = (
        st.session_state.get(REPOSITORY_CHART_VERSION_KEY, 0) + 1
    )

if DATABASE_CHART_PENDING_KEY in st.session_state:
    pending_database = st.session_state.pop(DATABASE_CHART_PENDING_KEY)
    action = pending_database.get("action")
    if action == "select":
        selected_database = pending_database["value"]
        st.session_state["relation_filter_base"] = [selected_database]
        st.session_state[DATABASE_CHART_LAST_SELECTION_KEY] = selected_database
    elif action == "clear":
        st.session_state["relation_filter_base"] = []
        st.session_state.pop(DATABASE_CHART_LAST_SELECTION_KEY, None)
    st.session_state[DATABASE_CHART_VERSION_KEY] = (
        st.session_state.get(DATABASE_CHART_VERSION_KEY, 0) + 1
    )

st.markdown(
    """
    <style>
      :root { --forest:#176B55; --sand:#F3EFE6; --ink:#17352D; }
      .stApp { background:linear-gradient(180deg,#F8FAF7 0,#FFFFFF 22rem); }
      .block-container { padding-top:1.4rem; padding-bottom:2rem; }
      [data-testid="stSidebar"] { background:#123E34; }
      [data-testid="stSidebar"] * { color:#F7F2E8; }
      div[data-testid="stMetric"] {
        background:white; border:1px solid #DDE7E1; border-radius:14px;
        padding:1rem 1.1rem; box-shadow:0 4px 18px rgba(23,53,45,.06);
      }
      .hero { padding:1.5rem 1.7rem; border-radius:20px; color:white;
        background:linear-gradient(120deg,#123E34,#24765F); margin-bottom:1.2rem; }
      .hero h1 { margin:0; font-size:2rem; }
      .hero p { margin:.45rem 0 0; color:#E4F0EA; }
      .eyebrow { color:#DDB46A; font-size:.78rem; letter-spacing:.12em;
        text-transform:uppercase; font-weight:700; }
      div[data-testid="stPlotlyChart"] { background:white; border-radius:14px; }
      #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">

      <h1>MONITOREO DE INVESTIGACION</h1>
      <p>Explora publicaciones consolidadas sobre recursos forestales,
      biodiversidad y fauna silvestre. Las categorías múltiples se
      contabilizan por separado.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Filtros")
    max_categories_chart = st.slider("Categorías en gráficos", 5, 25, 12)
    include_others = st.toggle("Mostrar 'Otros' en gráficos", value=True)
    st.caption(f"Base: {app_file.name}")

df_filtered = apply_simple_filters(df_mode)
selected_postgraduate_detail = st.session_state.get(
    PUBLICATION_DETAIL_FILTER_KEY
)
if selected_postgraduate_detail:
    df_filtered = df_filtered[
        df_filtered[POSTGRAD_DETAIL_COL].eq(selected_postgraduate_detail)
    ]
df_filtered = apply_relation_filters(df_filtered, relations)

chart_type = st.session_state.get(PUBLICATION_CHART_TYPE_KEY)
sidebar_types = st.session_state.get(f"filter_{TYPE_COL}", [])
if chart_type and sidebar_types != [chart_type]:
    st.session_state.pop(PUBLICATION_CHART_TYPE_KEY, None)
    st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "types"
    st.session_state.pop(PUBLICATION_CHART_SUBTYPE_KEY, None)
    st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None

chart_subtype = st.session_state.get(PUBLICATION_CHART_SUBTYPE_KEY)
sidebar_subtypes = st.session_state.get(f"filter_{SUBTYPE_COL}", [])
if chart_subtype and sidebar_subtypes != [chart_subtype]:
    st.session_state.pop(PUBLICATION_CHART_SUBTYPE_KEY, None)
    st.session_state[PUBLICATION_DETAIL_FILTER_KEY] = None
    st.session_state[PUBLICATION_CHART_LEVEL_KEY] = "subtypes"

chart_repository_class = st.session_state.get(REPOSITORY_CHART_CLASS_KEY)
sidebar_repository_classes = st.session_state.get(
    "relation_filter_repositorio_clase", []
)
if (
    chart_repository_class
    and sidebar_repository_classes != [chart_repository_class]
):
    st.session_state.pop(REPOSITORY_CHART_CLASS_KEY, None)
    st.session_state[REPOSITORY_CHART_LEVEL_KEY] = "classes"

sidebar_databases = st.session_state.get("relation_filter_base", [])
last_database_selection = st.session_state.get(
    DATABASE_CHART_LAST_SELECTION_KEY
)
if not sidebar_databases:
    st.session_state.pop(DATABASE_CHART_LAST_SELECTION_KEY, None)
elif sidebar_databases != [last_database_selection]:
    st.session_state[DATABASE_CHART_LAST_SELECTION_KEY] = (
        sidebar_databases[0] if len(sidebar_databases) == 1 else None
    )

type_summary = simple_summary(df_filtered, TYPE_COL)
subtype_summary = publication_drilldown_summary(
    df_filtered,
    df_mode,
    chart_type,
    SUBTYPE_COL,
)
postgraduate_detail_summary = publication_drilldown_summary(
    df_filtered,
    df_mode,
    chart_type,
    POSTGRAD_DETAIL_COL,
    publication_subtype="Tesis de posgrado",
)
database_summary = relation_summary(
    relations.get("base", pd.DataFrame()),
    df_filtered,
    include_others,
)
region_summary = department_summary(df_filtered, expanded_regions, map_base)

repo_class_summary = relation_summary(
    relations.get("repositorio_clase", pd.DataFrame()),
    df_filtered,
    include_others,
)
selected_repository_class = st.session_state.get(REPOSITORY_CHART_CLASS_KEY)
repository_drilldown_summary = (
    repository_summary_by_class(
        relations.get("repositorio", pd.DataFrame()),
        df_filtered,
        selected_repository_class,
        include_others,
    )
    if selected_repository_class
    else pd.DataFrame(columns=["categoria", "Publicaciones"])
)
institution_class_summary = relation_summary(
    relations.get("institucion_clase", pd.DataFrame()),
    df_filtered,
    include_others,
)
public_university_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Universidad publica nacional",
    include_others,
)
private_university_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Universidad privada nacional",
    include_others,
)
foreign_university_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Universidad extranjera",
    include_others,
)
public_entity_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Instituto publico / entidad estatal",
    include_others,
)
research_center_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Centro de investigación / cooperación",
    include_others,
    PUBLIC_INSTITUTION_CLASS_COL,
)
misplaced_journal_institution_summary = institution_summary_by_class(
    relations.get("institucion", pd.DataFrame()),
    df_filtered,
    "Revista / boletin mal ubicado",
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
repo_class_count = (
    int(repo_class_summary["categoria"].nunique())
    if not repo_class_summary.empty
    else 0
)

metrics = st.columns(4)
metrics[0].metric("N° de Publicaciones", human_int(unique_publications))
metrics[1].metric("N° de Departamentos", human_int(regions_with_data))
metrics[2].metric("N° de Áreas temáticas", human_int(area_count))
metrics[3].metric("N° de Clases de repositorio", human_int(repo_class_count))

st.caption(
    "Una publicación puede pertenecer a varias áreas, ejes, líneas, regiones "
    "o repositorios; por ello, las categorías no son excluyentes."
)

tabs = st.tabs(
    ["General", "Territorio", "Tiempo", "Temas", "Instituciones", "Datos"]
)

with tabs[0]:
    st.subheader("Resumen bibliográfico")
    col_a, col_b = st.columns(2)
    with col_a:
        publication_level = st.session_state.get(
            PUBLICATION_CHART_LEVEL_KEY, "types"
        )
        selected_publication_type = st.session_state.get(
            PUBLICATION_CHART_TYPE_KEY
        )
        if publication_level == "postgraduate" and selected_publication_type:
            st.caption("Jerarquía seleccionada: Tesis › Tesis de posgrado")
            button_columns = st.columns(2)
            if button_columns[0].button(
                "← Volver a subcategorías",
                key="publication_chart_back_to_subtypes",
            ):
                st.session_state[PUBLICATION_CHART_PENDING_KEY] = {
                    "action": "back_to_subtypes"
                }
                st.rerun()
            if (
                st.session_state.get(PUBLICATION_DETAIL_FILTER_KEY)
                and button_columns[1].button(
                    "Limpiar detalle", key="publication_chart_clear_detail"
                )
            ):
                st.session_state[PUBLICATION_CHART_PENDING_KEY] = {
                    "action": "clear_detail"
                }
                st.rerun()
            chart_data = postgraduate_detail_summary.copy()
            chart_category = "Jerarquía de publicación"
            chart_data[chart_category] = (
                "Tesis › Tesis de posgrado › "
                + chart_data[POSTGRAD_DETAIL_COL].astype(str)
            )
            chart_title = "Tesis de posgrado — detalle"
        elif publication_level == "subtypes" and selected_publication_type:
            st.caption(f"Tipo seleccionado: {selected_publication_type}")
            button_columns = st.columns(2)
            if button_columns[0].button(
                "← Volver a tipos", key="publication_chart_back"
            ):
                st.session_state[PUBLICATION_CHART_PENDING_KEY] = {
                    "action": "back"
                }
                st.rerun()
            if st.session_state.get(f"filter_{SUBTYPE_COL}") and button_columns[1].button(
                "Limpiar subcategoría", key="publication_chart_clear_subtype"
            ):
                st.session_state[PUBLICATION_CHART_PENDING_KEY] = {
                    "action": "clear_subtype"
                }
                st.rerun()
            chart_data = subtype_summary.copy()
            chart_category = "Jerarquía de publicación"
            chart_data[chart_category] = (
                selected_publication_type
                + " › "
                + chart_data[SUBTYPE_COL].astype(str)
            )
            chart_title = f"{selected_publication_type} — subcategorías"
        else:
            chart_data = type_summary
            chart_category = TYPE_COL
            chart_title = "Tipo de publicación"

        if not chart_data.empty:
            publication_event = st.plotly_chart(
                horizontal_bar(
                    chart_data,
                    chart_category,
                    chart_title,
                    max_categories_chart,
                ),
                width="stretch",
                key=(
                    "publication_type_chart_"
                    f"{st.session_state.get(PUBLICATION_CHART_VERSION_KEY, 0)}"
                ),
                on_select="rerun",
                selection_mode="points",
            )
            selected_publication_labels = selected_bar_labels(
                publication_event
            )
            if selected_publication_labels:
                selected_label = selected_publication_labels[0]
                if publication_level in {"subtypes", "postgraduate"} and " › " in selected_label:
                    selected_label = selected_label.rsplit(" › ", 1)[1]
                st.session_state[PUBLICATION_CHART_PENDING_KEY] = {
                    "action": (
                        "type"
                        if publication_level == "types"
                        else "detail"
                        if publication_level == "postgraduate"
                        else "subtype"
                    ),
                    "value": selected_label,
                }
                st.rerun()
        else:
            st.info("No hay datos para mostrar.")
    with col_b:
        repository_level = st.session_state.get(
            REPOSITORY_CHART_LEVEL_KEY, "classes"
        )
        selected_repository_class = st.session_state.get(
            REPOSITORY_CHART_CLASS_KEY
        )
        if repository_level == "repositories" and selected_repository_class:
            st.caption(f"Clase seleccionada: {selected_repository_class}")
            repository_buttons = st.columns(2)
            if repository_buttons[0].button(
                "← Volver a clases", key="repository_chart_back"
            ):
                st.session_state[REPOSITORY_CHART_PENDING_KEY] = {
                    "action": "back"
                }
                st.rerun()
            if (
                st.session_state.get("relation_filter_repositorio")
                and repository_buttons[1].button(
                    "Limpiar repositorio", key="repository_chart_clear"
                )
            ):
                st.session_state[REPOSITORY_CHART_PENDING_KEY] = {
                    "action": "clear_repository"
                }
                st.rerun()
            repository_chart_data = repository_drilldown_summary.copy()
            repository_category = "Jerarquía de repositorio"
            repository_chart_data[repository_category] = (
                selected_repository_class
                + " › "
                + repository_chart_data["categoria"].astype(str)
            )
            repository_title = f"{selected_repository_class} — repositorios"
        else:
            repository_chart_data = repo_class_summary
            repository_category = "categoria"
            repository_title = "Publicaciones por clase de repositorio"

        if not repository_chart_data.empty:
            repository_event = st.plotly_chart(
                horizontal_bar(
                    repository_chart_data,
                    repository_category,
                    repository_title,
                    max_categories_chart,
                ),
                width="stretch",
                key=(
                    "repository_class_chart_"
                    f"{st.session_state.get(REPOSITORY_CHART_VERSION_KEY, 0)}"
                ),
                on_select="rerun",
                selection_mode="points",
            )
            selected_repository_labels = selected_bar_labels(
                repository_event
            )
            if selected_repository_labels:
                selected_repository_label = selected_repository_labels[0]
                if (
                    repository_level == "repositories"
                    and " › " in selected_repository_label
                ):
                    selected_repository_label = selected_repository_label.split(
                        " › ", 1
                    )[1]
                st.session_state[REPOSITORY_CHART_PENDING_KEY] = {
                    "action": (
                        "class"
                        if repository_level == "classes"
                        else "repository"
                    ),
                    "value": selected_repository_label,
                }
                st.rerun()
        else:
            st.info("No hay clases de repositorio disponibles.")

    if not database_summary.empty:
        selected_database_filter = st.session_state.get(
            "relation_filter_base", []
        )
        if selected_database_filter:
            database_header = st.columns([3, 1])
            database_header[0].caption(
                "Base documental seleccionada: "
                + ", ".join(selected_database_filter)
            )
            if database_header[1].button(
                "Limpiar base", key="database_chart_clear"
            ):
                st.session_state[DATABASE_CHART_PENDING_KEY] = {
                    "action": "clear"
                }
                st.rerun()
        database_event = st.plotly_chart(
            horizontal_bar(
                database_summary,
                "categoria",
                "Publicaciones por base documental",
                max_categories_chart,
            ),
            width="stretch",
            key=(
                "database_chart_"
                f"{st.session_state.get(DATABASE_CHART_VERSION_KEY, 0)}"
            ),
            on_select="rerun",
            selection_mode="points",
        )
        selected_database_labels = selected_bar_labels(database_event)
        if selected_database_labels:
            selected_database = selected_database_labels[0]
            if selected_database != st.session_state.get(
                DATABASE_CHART_LAST_SELECTION_KEY
            ):
                st.session_state[DATABASE_CHART_PENDING_KEY] = {
                    "action": "select",
                    "value": selected_database,
                }
                st.rerun()
    else:
        st.info("No hay datos para mostrar.")

with tabs[1]:
    st.subheader("Ámbito de intervención de las investigaciones")
    selected_sidebar_regions = st.session_state.get(REGION_FILTER_KEY, [])
    if selected_sidebar_regions:
        selected_text = ", ".join(selected_sidebar_regions)
        st.info(f"Filtro territorial activo: {selected_text}")
        if st.button(
            "Limpiar selección territorial",
            key="clear_territorial_selection",
            width="stretch",
        ):
            st.session_state[MAP_PENDING_REGIONS_KEY] = []
            st.session_state[MAP_LAST_SELECTION_KEY] = ()
            st.session_state[MAP_WIDGET_VERSION_KEY] = (
                st.session_state.get(MAP_WIDGET_VERSION_KEY, 0) + 1
            )
            st.rerun()
    else:
        st.caption(
            "Selecciona uno o varios departamentos en el mapa para filtrar "
            "todos los indicadores, gráficos, datos y descargas."
        )
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
        custom_data=["DEP_KEY"],
        color_continuous_scale="YlGnBu",
        labels={"Publicaciones": "N° de Publicaciones"},
    )
    map_figure.update_geos(fitbounds="locations", visible=False)
    map_figure.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
        separators=", ",
    )
    map_event = st.plotly_chart(
        map_figure,
        width="stretch",
        key=(
            "territorial_map_"
            f"{st.session_state.get(MAP_WIDGET_VERSION_KEY, 0)}"
        ),
        on_select="rerun",
        selection_mode="points",
    )
    selected_map_keys = selected_map_region_keys(map_event)
    last_map_selection = tuple(
        st.session_state.get(MAP_LAST_SELECTION_KEY, ())
    )
    if selected_map_keys and selected_map_keys != last_map_selection:
        selected_filter_values = map_regions_to_filter_values(
            selected_map_keys,
            relations.get("region", pd.DataFrame()),
        )
        if selected_filter_values:
            st.session_state[MAP_LAST_SELECTION_KEY] = selected_map_keys
            st.session_state[MAP_PENDING_REGIONS_KEY] = selected_filter_values
            st.rerun()

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
            yaxis_title="N° de Publicaciones",
            separators=", ",
        )
        st.plotly_chart(time_figure, width="stretch")

with tabs[3]:
    st.subheader("Vinculación ANIFF 2026 - 2030")
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
    else:
        st.info("No hay ejes temáticos para mostrar.")

with tabs[4]:
    st.subheader("Instituciones y universidades")
    if not institution_class_summary.empty:
        st.plotly_chart(
            horizontal_bar(
                institution_class_summary,
                "categoria",
                "Publicaciones por clase de institución",
                max_categories_chart,
            ),
            width="stretch",
        )
    else:
        st.info("No hay clases de institución para mostrar.")

    if not public_university_summary.empty or not private_university_summary.empty:
        with st.expander("Universidad nacional"):
            national_university_type = st.selectbox(
                "Tipo de universidad nacional",
                ["Pública", "Privada"],
                key="national_university_subclass",
            )
            national_summary = (
                public_university_summary
                if national_university_type == "Pública"
                else private_university_summary
            )
            if not national_summary.empty:
                st.plotly_chart(
                    horizontal_bar(
                        national_summary,
                        "categoria",
                        f"Universidad nacional - {national_university_type}",
                        max_categories_chart,
                    ),
                    width="stretch",
                )
            else:
                st.info("No hay universidades para la subcategoría seleccionada.")

    institution_rankings = [
        ("Universidad extranjera", foreign_university_summary),
        ("Instituto público / entidad estatal", public_entity_summary),
        ("Centro de investigación / cooperación", research_center_summary),
        ("Revista / boletín mal ubicado", misplaced_journal_institution_summary),
    ]
    for title, summary in institution_rankings:
        if summary.empty:
            continue
        with st.expander(title):
            st.plotly_chart(
                horizontal_bar(
                    summary,
                    "categoria",
                    title,
                    max_categories_chart,
                ),
                width="stretch",
            )

with tabs[5]:
    st.subheader("Tabla exploratoria")
    visible_columns = [
        "General_ Título",
        "General_ Autor(es)",
        RECORD_ID_COL,
        YEAR_COL,
        TYPE_COL,
        SUBTYPE_COL,
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

    export_data = df_filtered.copy()
    territorial_export = filtered_expanded_regions(
        df_filtered,
        expanded_regions,
    )
    consolidated_regions = (
        territorial_export.loc[
            territorial_export.get("DEP_EN_GEOJSON", False).astype(bool)
        ]
        .groupby(RECORD_ID_COL)["DEPARTAMEN_GEO"]
        .agg(lambda values: "; ".join(sorted(set(values.dropna().astype(str)))))
        .rename("REGIONES_ESTUDIO_CONSOLIDADAS")
    )
    export_data = export_data.merge(
        consolidated_regions,
        left_on=RECORD_ID_COL,
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    excel_bytes = to_excel_bytes(
        export_data,
        type_summary,
        region_summary,
        territorial_export,
    )
    csv_bytes = export_data.to_csv(index=False).encode("utf-8-sig")
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
    f"Fuente maestra integrada: {app_file.name}"
)


