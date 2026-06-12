from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd


APP_SHEET = "BD_APP"
RECORD_ID_COL = "ID_REGISTRO_ANALISIS"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"
UNIQUE_COL = "USAR_PARA_CONTEO_UNICO"

DIMENSION_RULES = {
    "DIM_REPOSITORIOS": ("General_ Repositorio", r"\s*[,;]\s*"),
    "DIM_REGIONES_NORMALIZADAS": (
        "REGION_NORM_SUGERIDA",
        r"\s*[;,]\s*",
    ),
}

OFFICIAL_AXES = [
    "Manejo, conservación y uso sostenible del recurso forestal",
    "Manejo, conservación y uso sostenible del recurso fauna silvestre",
    "Plantaciones forestales y sistemas agroforestales",
    "Industria y productos forestales maderables y no maderables",
    "Servicios ecosistémicos",
    "Cambio Climático",
    "Institucionalidad y Gobernanza",
    "Interculturalidad",
]

OFFICIAL_AREAS = [
    "Gestión y conservación de ecosistemas",
    "Plantaciones forestales y sistemas agroforestales",
    "Bosques naturales",
    "Manejo de fauna silvestre in situ",
    "Transformación de productos maderables y no maderables",
    "Productos forestales maderables y no maderables",
    "Manejo forestal y de fauna silvestre en comunidades",
    "Funcionalidad de los ecosistemas",
    "Conservación de fauna silvestre",
    "Cambio Climático",
    "Genética y biotecnología forestal",
    "Restauración de áreas degradadas",
    "Manejo de fauna silvestre ex situ",
    "Institucionalidad y Gobernanza",
    "Conflictos con fauna silvestre",
    "Ecología, evolución e historia natural",
    "No bosques naturales",
    "Forestería urbana",
    "Salud ex situ",
]

INSTITUTION_PATTERN = re.compile(
    r"(?i)\b("
    r"universidad|university|universidade|universitat|instituto|institute|"
    r"institution|ministerio|servicio nacional|serfor|consejo|centro de|"
    r"museum|college|academy|academia|sociedad|society|fundación|foundation|"
    r"organization|organisation|association|asociación|corporation|agencia|"
    r"agency|laboratory|laboratorio|herbario|jardín botánico|botanical garden|"
    r"research center|research centre|council|facultad|escuela|cirad|inia"
    r")\b"
)

REQUIRED_APP_COLUMNS = [
    "TIPO_PUBLICACION_NORM",
    "Categoria_Tesis_Articulo",
    "General_ Año",
    UNIQUE_COL,
    MASTER_KEY_COL,
    RECORD_ID_COL,
]


def latest_directory(path: Path, pattern: str) -> Path:
    directories = [item for item in path.glob(pattern) if item.is_dir()]
    if not directories:
        raise FileNotFoundError(f"No se encontraron carpetas {pattern} en {path}")
    return max(directories, key=lambda item: item.stat().st_mtime)


def latest_file(path: Path, pattern: str) -> Path:
    files = [
        item
        for item in path.glob(pattern)
        if item.is_file() and not item.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos {pattern} en {path}")
    return max(files, key=lambda item: item.stat().st_mtime)


def split_values(value, pattern: str) -> list[str]:
    if pd.isna(value):
        return []
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return []
    values = [part.strip() for part in re.split(pattern, text)]
    return list(dict.fromkeys(value for value in values if value))


def normalized_key(value: str) -> str:
    text = str(value).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def canonicalize_by_frequency(dimension: pd.DataFrame) -> pd.DataFrame:
    result = dimension.copy()
    result["_key"] = result["categoria"].map(normalized_key)
    frequencies = (
        result.groupby(["_key", "categoria"])
        .size()
        .rename("frecuencia")
        .reset_index()
        .sort_values(["_key", "frecuencia", "categoria"], ascending=[True, False, True])
    )
    canonical = frequencies.drop_duplicates("_key").set_index("_key")["categoria"]
    result["categoria"] = result["_key"].map(canonical)
    result = result.drop(columns="_key")
    return result.drop_duplicates([RECORD_ID_COL, "categoria"]).reset_index(drop=True)


def axis_values(value) -> list[str]:
    if pd.isna(value):
        return []
    text = normalized_key(value)
    if text == "otros":
        return ["Otros"]
    matches = [
        axis
        for axis in OFFICIAL_AXES
        if normalized_key(axis) in text
    ]
    return list(dict.fromkeys(matches))


def official_values(value, catalog: list[str]) -> list[str]:
    if pd.isna(value):
        return []
    text = normalized_key(value)
    if text == "otros":
        return ["Otros"]
    text = text.replace(
        "funcionalidad de ecosistemas",
        "funcionalidad de los ecosistemas",
    )
    matches = []
    remaining = text
    for label in sorted(catalog, key=lambda item: len(normalized_key(item)), reverse=True):
        key = normalized_key(label)
        if key in remaining:
            matches.append(label)
            remaining = remaining.replace(key, " ")
    order = {label: index for index, label in enumerate(catalog)}
    return sorted(set(matches), key=lambda label: order[label])


def build_dimension_from_lists(
    df: pd.DataFrame,
    source_column: str,
    values,
) -> pd.DataFrame:
    dimension = df[[RECORD_ID_COL, MASTER_KEY_COL, source_column]].copy()
    dimension["categoria"] = dimension[source_column].map(values)
    dimension = dimension.explode("categoria").dropna(subset=["categoria"])
    dimension["categoria"] = dimension["categoria"].astype(str).str.strip()
    dimension = dimension[dimension["categoria"].ne("")]
    dimension = dimension.drop_duplicates([RECORD_ID_COL, "categoria"])
    dimension["orden_categoria"] = dimension.groupby(RECORD_ID_COL).cumcount() + 1
    return dimension[
        [
            RECORD_ID_COL,
            MASTER_KEY_COL,
            source_column,
            "categoria",
            "orden_categoria",
        ]
    ].reset_index(drop=True)


def build_line_dimension(df: pd.DataFrame) -> pd.DataFrame:
    column = "ANIFFS: Linea de investigación"
    description_to_code = {}
    for value in df[column].dropna().astype(str):
        for code, description in re.findall(r"(\d{1,3})\.\s*([^,;/]+)", value):
            description_to_code[normalized_key(description)] = str(int(code))
        full_description = re.match(r"^\s*(\d{1,3})\.\s*(.+?)\s*$", value)
        if full_description:
            description_to_code[
                normalized_key(full_description.group(2))
            ] = str(int(full_description.group(1)))

    def line_values(value):
        if pd.isna(value):
            return []
        text = str(value).strip()
        if normalized_key(text) == "otros":
            return ["Otros"]
        codes = [str(int(code)) for code in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text)]
        if codes:
            return list(dict.fromkeys(codes))
        mapped = description_to_code.get(normalized_key(text))
        return [mapped] if mapped else ["Otros"]

    return build_dimension_from_lists(df, column, line_values)


def build_institution_dimension(df: pd.DataFrame) -> pd.DataFrame:
    column = "General_ Institución/Universidad"

    def institution_values(value):
        if pd.isna(value):
            return []
        text = re.sub(r"\s+", " ", str(value)).strip()
        if normalized_key(text) == "otros":
            return ["Otros"]
        return [text] if INSTITUTION_PATTERN.search(text) else []

    return canonicalize_by_frequency(
        build_dimension_from_lists(df, column, institution_values)
    )


def build_dimension(df: pd.DataFrame, source_column: str, pattern: str) -> pd.DataFrame:
    dimension = df[[RECORD_ID_COL, MASTER_KEY_COL, source_column]].copy()
    dimension["categoria"] = dimension[source_column].map(
        lambda value: split_values(value, pattern)
    )
    dimension = dimension.explode("categoria").dropna(subset=["categoria"])
    dimension["categoria"] = dimension["categoria"].astype(str).str.strip()
    dimension = dimension[dimension["categoria"].ne("")]
    dimension = dimension.drop_duplicates([RECORD_ID_COL, "categoria"])
    dimension["orden_categoria"] = dimension.groupby(RECORD_ID_COL).cumcount() + 1
    return dimension[
        [
            RECORD_ID_COL,
            MASTER_KEY_COL,
            source_column,
            "categoria",
            "orden_categoria",
        ]
    ].reset_index(drop=True)


def main() -> None:
    app_root = Path(__file__).resolve().parents[1]
    project_root = app_root.parent
    data_dir = app_root / "data"
    notebook_outputs = project_root / "NOTEBOOK" / "salidas_limpieza"
    source_data = project_root / "DATOS"
    data_dir.mkdir(parents=True, exist_ok=True)

    latest_output = latest_directory(notebook_outputs, "ejecucion_*")
    homologated_source = latest_output / "BASE_HOMOLOGADA_42_CAMPOS.xlsx"
    territorial_source = latest_file(
        source_data,
        "BD_APP_TERRITORIAL_*_PUBLICA_UNICA.xlsx",
    )

    df = pd.read_excel(
        homologated_source,
        sheet_name="DATOS_HOMOLOGADOS",
        dtype="string",
        engine="openpyxl",
    )
    df.columns = [str(column).strip() for column in df.columns]

    missing = [column for column in REQUIRED_APP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))
    if df[RECORD_ID_COL].isna().any() or not df[RECORD_ID_COL].is_unique:
        raise ValueError(f"{RECORD_ID_COL} debe ser completo y único.")

    dimensions = {
        sheet: canonicalize_by_frequency(build_dimension(df, column, pattern))
        for sheet, (column, pattern) in DIMENSION_RULES.items()
    }
    dimensions["DIM_EJES_TEMATICOS"] = build_dimension_from_lists(
        df,
        "ANIFFS: Eje Temático",
        axis_values,
    )
    dimensions["DIM_AREAS_TEMATICAS"] = build_dimension_from_lists(
        df,
        "ANIFFS: Área Temática",
        lambda value: official_values(value, OFFICIAL_AREAS),
    )
    dimensions["DIM_LINEAS_INVESTIGACION"] = build_line_dimension(df)
    dimensions["DIM_INSTITUCIONES"] = build_institution_dimension(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_output = data_dir / f"BD_APP_FINAL_{timestamp}_HOMOLOGADA.xlsx"
    territorial_output = (
        data_dir / f"BD_APP_TERRITORIAL_{timestamp}_HOMOLOGADA.xlsx"
    )

    summary_rows = [
        {"indicador": "fecha_adaptacion", "valor": datetime.now().isoformat(timespec="seconds")},
        {"indicador": "archivo_fuente_homologado", "valor": str(homologated_source)},
        {"indicador": "archivo_fuente_territorial", "valor": str(territorial_source)},
        {"indicador": "registros", "valor": len(df)},
        {"indicador": "campos_base_principal", "valor": len(df.columns)},
        {"indicador": "identificadores_unicos", "valor": df[RECORD_ID_COL].nunique()},
        {"indicador": "publicaciones_bibliograficas", "valor": df[MASTER_KEY_COL].nunique()},
    ]
    for sheet, dimension in dimensions.items():
        summary_rows.append(
            {"indicador": f"relaciones_{sheet.lower()}", "valor": len(dimension)}
        )
        summary_rows.append(
            {
                "indicador": f"categorias_{sheet.lower()}",
                "valor": dimension["categoria"].nunique(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    contract = pd.DataFrame(
        [
            {
                "hoja": APP_SHEET,
                "proposito": "Una fila por publicación; contrato principal del aplicativo",
                "clave": RECORD_ID_COL,
            },
            *[
                {
                    "hoja": sheet,
                    "proposito": "Relación publicación-categoría normalizada",
                    "clave": f"{RECORD_ID_COL} + categoria",
                }
                for sheet in dimensions
            ],
            {
                "hoja": "RESUMEN_ADAPTACION",
                "proposito": "Origen, cobertura y conteos de la adaptación",
                "clave": "indicador",
            },
            {
                "hoja": "CONTRATO_DATOS",
                "proposito": "Descripción de las hojas consumidas por el aplicativo",
                "clave": "hoja",
            },
        ]
    )

    with pd.ExcelWriter(app_output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=APP_SHEET, index=False)
        for sheet, dimension in dimensions.items():
            dimension.to_excel(writer, sheet_name=sheet, index=False)
        summary.to_excel(writer, sheet_name="RESUMEN_ADAPTACION", index=False)
        contract.to_excel(writer, sheet_name="CONTRATO_DATOS", index=False)

    territorial_xl = pd.ExcelFile(territorial_source, engine="openpyxl")
    with pd.ExcelWriter(territorial_output, engine="xlsxwriter") as writer:
        for sheet in territorial_xl.sheet_names:
            sheet_df = pd.read_excel(
                territorial_source,
                sheet_name=sheet,
                dtype=object,
                engine="openpyxl",
            )
            sheet_df.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame(
            [
                {
                    "indicador": "fecha_adaptacion",
                    "valor": datetime.now().isoformat(timespec="seconds"),
                },
                {
                    "indicador": "archivo_fuente",
                    "valor": str(territorial_source),
                },
                {
                    "indicador": "base_principal_asociada",
                    "valor": app_output.name,
                },
            ]
        ).to_excel(writer, sheet_name="TRAZABILIDAD_ADAPTACION", index=False)

    trace_file = data_dir / "TRAZABILIDAD_BASE_ADAPTADA.md"
    trace_file.write_text(
        f"""# Trazabilidad de la base adaptada

## Ejecución vigente

- **Fecha:** {datetime.now().isoformat(timespec="seconds")}
- **Base homologada de origen:** `{homologated_source}`
- **Base territorial de origen:** `{territorial_source}`
- **Base principal generada:** `{app_output.name}`
- **Base territorial generada:** `{territorial_output.name}`
- **Registros:** {len(df):,}
- **Campos principales:** {len(df.columns)}
- **Identificadores únicos:** {df[RECORD_ID_COL].nunique():,}

## Dimensiones relacionales

"""
        + "\n".join(
            f"- `{sheet}`: {len(dimension):,} relaciones y "
            f"{dimension['categoria'].nunique():,} categorías."
            for sheet, dimension in dimensions.items()
        )
        + """

## Criterios

- La hoja `BD_APP` conserva el contrato de 42 campos requerido por el aplicativo.
- Las dimensiones expandidas evitan contar combinaciones como categorías únicas.
- `Otros` representa información ausente o no especificada.
- `No aplica` se conserva como estado distinto.
- Los archivos fuente no fueron modificados.
""",
        encoding="utf-8",
    )

    print(f"APP_OUTPUT={app_output}")
    print(f"TERRITORIAL_OUTPUT={territorial_output}")
    print(f"TRACE={trace_file}")
    print(f"ROWS={len(df)}")
    print(f"COLUMNS={len(df.columns)}")


if __name__ == "__main__":
    main()
