from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from institution_classification import (
    INSTITUTION_CLASS_COL,
    IS_UNIVERSITY_COL,
    classify_institution,
)
from repository_classification import (
    PUBLIC_REPOSITORY_CLASS_COL,
    PUBLIC_REPOSITORY_REVIEW_COL,
    PUBLIC_REPOSITORY_RULE_COL,
    REPOSITORY_CLASS_COL,
    UNIVERSITY_REPOSITORY_COL,
    classify_public_repository,
    classify_repository,
)


RECORD_ID = "ID_REGISTRO_ANALISIS"
PUBLICATION_ID = "ID_PUBLICACION_PROPUESTA"
REPRESENTATIVE_ID = "ID_REGISTRO_REPRESENTATIVO"
MASTER_KEY = "CLAVE_BIBLIOGRAFICA_MASTER"
ACADEMIC_SOURCE = "General_ Tipo de tesis Pre/Posgrado"
ACADEMIC_GRADE = "GRADO_ACADEMICO_PUBLICO"
ACADEMIC_LEVEL = "NIVEL_ACADEMICO_PUBLICO"
PUBLIC_TYPE = "TIPO_PUBLICACION_PUBLICO"
PUBLIC_SUBTYPE = "SUBTIPO_PUBLICACION_PUBLICO"
INSTITUTION_SOURCE = "General_ Institución/Universidad"
JOURNAL_SOURCE = "General_ Nombre de revista"

MISPLACED_JOURNAL_ENTITIES = {
    "Anales del Jardín Botánico de Madrid": "Jardín Botánico de Madrid",
    "Annals of the American Association of Geographers": (
        "American Association of Geographers"
    ),
    "Annals of the Missouri Botanical Garden": "Missouri Botanical Garden",
    "Biological Journal of the Linnean Society": "Linnean Society",
    "Boletín Sociedad Entomológica Aragonesa": "Sociedad Entomológica Aragonesa",
    "Boletín de la Sociedad Argentina de Botánica": (
        "Sociedad Argentina de Botánica"
    ),
    "Boletín de la Sociedad Geográfica de Lima": "Sociedad Geográfica de Lima",
    "Boletín del Instituto del Mar del Perú": "Instituto del Mar del Perú",
    "Botanical Journal of the Linnean Society": "Linnean Society",
    "PROCEEDINGS-ENTOMOLOGICAL SOCIETY OF WASHINGTON": (
        "Entomological Society of Washington"
    ),
    "Proceedings of the Biological Society of Washington": (
        "Biological Society of Washington"
    ),
    "Proceedings of the Entomological Society of Washington": (
        "Entomological Society of Washington"
    ),
    "Proceedings of the Royal Society B: Biological Sciences": "Royal Society",
    "Proceedings of the Zoological Institute of the Russian Academy of Sciences": (
        "Zoological Institute of the Russian Academy of Sciences"
    ),
    "The Society of Wetland Scientists Bulletin": "Society of Wetland Scientists",
    "Zoologische Abhandlungen Staatliches Museum für Tierkunde Dresden": (
        "Staatliches Museum für Tierkunde Dresden"
    ),
}

DIMENSION_SHEETS = [
    "DIM_REPOSITORIOS",
    "DIM_AREAS_TEMATICAS",
    "DIM_EJES_TEMATICOS",
    "DIM_LINEAS_INVESTIGACION",
    "DIM_REGIONES_NORMALIZADAS",
    "DIM_INSTITUCIONES",
]


def latest_file(path: Path, pattern: str) -> Path:
    files = [
        item
        for item in path.glob(pattern)
        if item.is_file() and not item.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"No se encontró {pattern} en {path}")
    return max(files, key=lambda item: item.stat().st_mtime)


def latest_directory(path: Path, pattern: str) -> Path:
    directories = [item for item in path.glob(pattern) if item.is_dir()]
    if not directories:
        raise FileNotFoundError(f"No se encontró {pattern} en {path}")
    return max(directories, key=lambda item: item.stat().st_mtime)


def normalize_database_name(value):
    if pd.isna(value):
        return value
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"^\d+\.\s*", "", text)
    replacements = {
        "Base estado del arte algarrobo": "Base estado del arte Algarrobo",
        "Base producción cientifica algarrobo": (
            "Base producción científica Algarrobo"
        ),
        "Base producción cientifica Amazonas": (
            "Base producción científica Amazonas"
        ),
        "Condor": "Cóndor",
        "Diagnostico Junin": "Diagnóstico Junín",
        "Diagnostico Madre de Dios": "Diagnóstico Madre de Dios",
        "Diagnostico Moquegua": "Diagnóstico Moquegua",
        "Diagnostico Puno": "Diagnóstico Puno",
        "Producción Cientifica AN,HU,PA,SM,CA": (
            "Producción científica AN, HU, PA, SM, CA"
        ),
        "Producción científica shihuahuaco": (
            "Producción científica Shihuahuaco"
        ),
    }
    return replacements.get(text, text)


def join_unique(values) -> str:
    return "; ".join(
        dict.fromkeys(
            value
            for value in values.dropna().astype(str).str.strip()
            if value
        )
    )


def build_publications(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    source = source.copy()
    mapping = mapping.copy()
    source[RECORD_ID] = source[RECORD_ID].astype(str)
    mapping[RECORD_ID] = mapping[RECORD_ID].astype(str)

    if mapping[RECORD_ID].duplicated().any():
        raise ValueError("La relación registro-publicación contiene duplicados.")
    if set(source[RECORD_ID]) != set(mapping[RECORD_ID]):
        raise ValueError("La relación registro-publicación no cubre la fuente.")

    representative_map = mapping[
        [
            PUBLICATION_ID,
            REPRESENTATIVE_ID,
            "REGISTROS_EN_GRUPO",
            "REGLAS_AGRUPACION",
            "ESTADO",
        ]
    ].drop_duplicates(PUBLICATION_ID)
    publications = representative_map.merge(
        source,
        left_on=REPRESENTATIVE_ID,
        right_on=RECORD_ID,
        how="left",
        validate="one_to_one",
    )

    source_with_publication = source.merge(
        mapping[[RECORD_ID, PUBLICATION_ID]],
        on=RECORD_ID,
        how="left",
        validate="one_to_one",
    )
    databases = (
        source_with_publication.groupby(PUBLICATION_ID)[
            "Nombre de Base de datos"
        ]
        .agg(join_unique)
        .rename("Nombre de Base de datos")
    )
    publications = publications.drop(
        columns=["Nombre de Base de datos"]
    ).merge(databases, on=PUBLICATION_ID, how="left", validate="one_to_one")
    publications["USAR_PARA_CONTEO_UNICO"] = "SI"
    publications[PUBLIC_TYPE] = publications["TIPO_PUBLICACION_NORM"].replace(
        {"Artículo de conferencia": "Artículo"}
    )
    publications[PUBLIC_SUBTYPE] = publications["TIPO_PUBLICACION_NORM"].map(
        {
            "Artículo": "Artículo científico",
            "Artículo de conferencia": "Artículo de conferencia",
        }
    )

    front = [
        PUBLICATION_ID,
        REPRESENTATIVE_ID,
        "REGISTROS_EN_GRUPO",
        "REGLAS_AGRUPACION",
        "ESTADO",
    ]
    remaining = [column for column in publications if column not in front]
    return publications[front + remaining].sort_values(
        PUBLICATION_ID
    ).reset_index(drop=True)


def derive_academic_fields(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = source[
        [RECORD_ID, "TIPO_PUBLICACION_NORM", ACADEMIC_SOURCE]
    ].merge(
        mapping[[RECORD_ID, PUBLICATION_ID]],
        on=RECORD_ID,
        how="inner",
        validate="one_to_one",
    )
    thesis = work[work["TIPO_PUBLICACION_NORM"].eq("Tesis")].copy()
    rows = []
    audit = []

    for publication_id, group in thesis.groupby(PUBLICATION_ID):
        values = sorted(
            {
                str(value).strip()
                for value in group[ACADEMIC_SOURCE].dropna()
                if str(value).strip() and str(value).strip() != "No aplica"
            }
        )
        value_set = set(values)
        if "Doctorado" in value_set:
            grade, level = "Posgrado", "Doctorado"
        elif "Maestría" in value_set:
            grade, level = "Posgrado", "Maestría"
        elif "Suficiencia profesional" in value_set:
            grade, level = "Pregrado", "Suficiencia profesional"
        elif "Pregrado" in value_set:
            grade, level = "Pregrado", "Pregrado"
        elif "Posgrado no especificado" in value_set:
            grade, level = "Posgrado", "Otros"
        else:
            grade, level = "Otros", "Otros"

        rows.append(
            {
                PUBLICATION_ID: publication_id,
                ACADEMIC_GRADE: grade,
                ACADEMIC_LEVEL: level,
            }
        )
        if len(values) > 1:
            broad_grades = {
                "Pregrado"
                if value in {"Pregrado", "Suficiencia profesional"}
                else "Posgrado"
                if value in {
                    "Maestría",
                    "Doctorado",
                    "Posgrado no especificado",
                }
                else "Otros"
                for value in values
            }
            audit.append(
                {
                    PUBLICATION_ID: publication_id,
                    "VALORES_ORIGEN": " | ".join(values),
                    ACADEMIC_GRADE: grade,
                    ACADEMIC_LEVEL: level,
                    "TIPO_CONFLICTO": (
                        "grados_contradictorios"
                        if {"Pregrado", "Posgrado"}.issubset(broad_grades)
                        else "niveles_multiples"
                    ),
                    "REGLA": "Priorizar el nivel académico más específico.",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(audit)


def is_empty_category(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in {"", "otros", "no aplica"}


def migrate_misplaced_journals(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = source.copy()
    audit_rows = []
    if INSTITUTION_SOURCE not in result.columns or JOURNAL_SOURCE not in result.columns:
        return result, pd.DataFrame(audit_rows)

    for index, row in result.iterrows():
        institution_value = row.get(INSTITUTION_SOURCE)
        institution_text = (
            "" if pd.isna(institution_value) else str(institution_value).strip()
        )
        institution_class, _ = classify_institution(institution_text)
        if institution_class != "Revista / boletin mal ubicado":
            continue

        previous_journal = row.get(JOURNAL_SOURCE)
        final_journal = previous_journal
        journal_action = "conservado"
        if is_empty_category(previous_journal):
            final_journal = institution_text
            result.at[index, JOURNAL_SOURCE] = final_journal
            journal_action = "migrado_a_nombre_revista"

        final_institution = MISPLACED_JOURNAL_ENTITIES.get(
            institution_text,
            "Otros",
        )
        result.at[index, INSTITUTION_SOURCE] = final_institution
        audit_rows.append(
            {
                RECORD_ID: row.get(RECORD_ID),
                MASTER_KEY: row.get(MASTER_KEY),
                "INSTITUCION_ORIGINAL": institution_text,
                "REVISTA_ORIGINAL": previous_journal,
                "REVISTA_FINAL": final_journal,
                "INSTITUCION_FINAL": final_institution,
                "ACCION_REVISTA": journal_action,
                "REGLA": (
                    "Valor con apariencia de revista/boletín en institución; "
                    "se migra a nombre de revista si el campo estaba vacío u Otros."
                ),
            }
        )

    return result, pd.DataFrame(audit_rows)


def remap_dimension(
    dimension: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    result = dimension.copy()
    result[RECORD_ID] = result[RECORD_ID].astype(str)
    record_map = mapping[[RECORD_ID, PUBLICATION_ID]].copy()
    record_map[RECORD_ID] = record_map[RECORD_ID].astype(str)
    result = result.merge(
        record_map,
        on=RECORD_ID,
        how="left",
        validate="many_to_one",
    )
    if result[PUBLICATION_ID].isna().any():
        raise ValueError("Una dimensión contiene registros sin publicación.")
    result = result.drop_duplicates([PUBLICATION_ID, "categoria"]).copy()
    result["orden_categoria"] = (
        result.groupby(PUBLICATION_ID).cumcount() + 1
    )
    front = [PUBLICATION_ID, RECORD_ID]
    remaining = [column for column in result if column not in front]
    return result[front + remaining].reset_index(drop=True)


def classify_repository_dimension(dimension: pd.DataFrame) -> pd.DataFrame:
    result = dimension.copy()
    classes = result["categoria"].map(classify_repository)
    result[REPOSITORY_CLASS_COL] = classes.map(lambda item: item[0])
    result[UNIVERSITY_REPOSITORY_COL] = classes.map(lambda item: item[1])
    public_classes = result["categoria"].map(classify_public_repository)
    result[PUBLIC_REPOSITORY_CLASS_COL] = public_classes.map(lambda item: item[0])
    result[PUBLIC_REPOSITORY_RULE_COL] = public_classes.map(lambda item: item[1])
    result[PUBLIC_REPOSITORY_REVIEW_COL] = public_classes.map(lambda item: item[2])
    return result


def classify_institution_dimension(dimension: pd.DataFrame) -> pd.DataFrame:
    result = dimension.copy()
    classes = result["categoria"].map(classify_institution)
    result[INSTITUTION_CLASS_COL] = classes.map(lambda item: item[0])
    result[IS_UNIVERSITY_COL] = classes.map(lambda item: item[1])
    return result


def database_dimension(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    dimension = source[
        [RECORD_ID, MASTER_KEY, "Nombre de Base de datos"]
    ].copy()
    dimension["categoria"] = dimension["Nombre de Base de datos"]
    dimension["orden_categoria"] = 1
    return remap_dimension(dimension, mapping)


def institution_dimension(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    dimension = source[[RECORD_ID, MASTER_KEY, INSTITUTION_SOURCE]].copy()
    dimension["categoria"] = dimension[INSTITUTION_SOURCE]
    dimension["orden_categoria"] = 1
    return remap_dimension(dimension, mapping)


def consolidate_territorial(
    source_file: Path,
    mapping: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(source_file, engine="openpyxl")
    sheets = {
        sheet: pd.read_excel(
            source_file,
            sheet_name=sheet,
            dtype=object,
            engine="openpyxl",
        )
        for sheet in excel.sheet_names
    }
    expanded = sheets["REGIONES_EXPANDIDAS"].copy()
    expanded[RECORD_ID] = expanded[RECORD_ID].astype(str)
    record_map = mapping[[RECORD_ID, PUBLICATION_ID]].copy()
    record_map[RECORD_ID] = record_map[RECORD_ID].astype(str)
    expanded = expanded.merge(
        record_map,
        on=RECORD_ID,
        how="inner",
        validate="many_to_one",
    )
    expanded = expanded.drop_duplicates(
        [PUBLICATION_ID, "DEP_KEY"]
    ).reset_index(drop=True)

    geo_rows = expanded[
        expanded.get("DEP_EN_GEOJSON", False).astype(bool)
    ]
    counts = (
        geo_rows.groupby("DEP_KEY")
        .agg(
            registros_territoriales=(RECORD_ID, "count"),
            publicaciones_unicas=(PUBLICATION_ID, "nunique"),
            publicaciones_bibliograficas=(PUBLICATION_ID, "nunique"),
        )
        .reset_index()
    )
    map_base = sheets["MAPA_DEPARTAMENTOS"][
        ["IDDPTO", "DEPARTAMEN_GEO", "DEP_KEY", "CAPITAL"]
    ].copy()
    map_data = map_base.merge(counts, on="DEP_KEY", how="left")
    count_columns = [
        "registros_territoriales",
        "publicaciones_unicas",
        "publicaciones_bibliograficas",
    ]
    map_data[count_columns] = map_data[count_columns].fillna(0).astype(int)
    sheets["MAPA_DEPARTAMENTOS"] = map_data
    sheets["REGIONES_EXPANDIDAS"] = expanded
    sheets["COBERTURA_REGION"] = pd.DataFrame(
        [
            {
                "indicador": "publicaciones_consolidadas",
                "valor": mapping[PUBLICATION_ID].nunique(),
            },
            {"indicador": "relaciones_territoriales", "valor": len(expanded)},
            {
                "indicador": "publicaciones_con_departamento",
                "valor": geo_rows[PUBLICATION_ID].nunique(),
            },
            {
                "indicador": "departamentos_con_datos",
                "valor": int(map_data["publicaciones_unicas"].gt(0).sum()),
            },
        ]
    )
    return sheets


def main() -> None:
    app_root = Path(__file__).resolve().parents[1]
    project_root = app_root.parent
    data_dir = app_root / "data"
    docs_dir = project_root / "NOTEBOOK" / "docs"

    cleaning_dir = latest_directory(
        project_root / "NOTEBOOK" / "salidas_limpieza",
        "ejecucion_*",
    )
    dedup_dir = latest_directory(
        project_root / "NOTEBOOK" / "salidas_deduplicacion",
        "ejecucion_*",
    )
    homologated_source = cleaning_dir / "BASE_HOMOLOGADA_42_CAMPOS.xlsx"
    dedup_source = dedup_dir / "DIAGNOSTICO_DEDUPLICACION_PUBLICACIONES.xlsx"
    territorial_source = latest_file(
        project_root / "DATOS",
        "BD_APP_TERRITORIAL_*_PUBLICA_UNICA.xlsx",
    )

    source = pd.read_excel(
        homologated_source,
        sheet_name="DATOS_HOMOLOGADOS",
        dtype="string",
        engine="openpyxl",
    )
    source.columns = [str(column).strip() for column in source.columns]
    source["Nombre de Base de datos"] = source[
        "Nombre de Base de datos"
    ].map(normalize_database_name)
    source, institution_audit = migrate_misplaced_journals(source)
    mapping = pd.read_excel(
        dedup_source,
        sheet_name="REGISTRO_PUBLICACION",
        dtype="string",
        engine="openpyxl",
    )
    decisions = pd.read_excel(
        dedup_source,
        sheet_name="DECISIONES_REVISADAS",
        dtype="string",
        engine="openpyxl",
    )
    proposed_ids = pd.read_excel(
        dedup_source,
        sheet_name="PUBLICACIONES_PROPUESTAS",
        dtype="string",
        engine="openpyxl",
    )[
        [
            PUBLICATION_ID,
            "HUELLA_PUBLICACION_PERSISTENTE",
            "METODO_HUELLA",
        ]
    ]
    master_registry = pd.read_excel(
        dedup_source,
        sheet_name="REGISTRO_MAESTRO_IDS",
        dtype="string",
        engine="openpyxl",
    )
    id_merges = pd.read_excel(
        dedup_source,
        sheet_name="FUSIONES_IDS",
        dtype="string",
        engine="openpyxl",
    )

    technical_sources = data_dir / "_fuentes_tecnicas"
    previous_app = latest_file(
        technical_sources
        if technical_sources.exists()
        else data_dir,
        "BD_APP_FINAL_*_HOMOLOGADA.xlsx",
    )
    previous_excel = pd.ExcelFile(previous_app, engine="openpyxl")
    dimensions = {
        sheet: remap_dimension(
            pd.read_excel(
                previous_app,
                sheet_name=sheet,
                dtype="string",
                engine="openpyxl",
            ),
            mapping,
        )
        for sheet in DIMENSION_SHEETS
        if sheet in previous_excel.sheet_names
    }
    if "DIM_REPOSITORIOS" in dimensions:
        dimensions["DIM_REPOSITORIOS"] = classify_repository_dimension(
            dimensions["DIM_REPOSITORIOS"]
        )
    dimensions["DIM_INSTITUCIONES"] = classify_institution_dimension(
        institution_dimension(source, mapping)
    )
    dimensions["DIM_BASES_DOCUMENTALES"] = database_dimension(source, mapping)

    publications = build_publications(source, mapping)
    publications = publications.merge(
        proposed_ids,
        on=PUBLICATION_ID,
        how="left",
        validate="one_to_one",
    )
    academic_fields, academic_audit = derive_academic_fields(source, mapping)
    publications = publications.merge(
        academic_fields,
        on=PUBLICATION_ID,
        how="left",
        validate="one_to_one",
    )
    territorial_sheets = consolidate_territorial(territorial_source, mapping)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    app_output = data_dir / f"BD_APP_FINAL_{timestamp}_DEDUPLICADA.xlsx"
    territorial_output = (
        data_dir / f"BD_APP_TERRITORIAL_{timestamp}_DEDUPLICADA.xlsx"
    )
    summary = pd.DataFrame(
        [
            {"indicador": "fecha_adaptacion", "valor": datetime.now().isoformat(timespec="seconds")},
            {"indicador": "base_homologada", "valor": str(homologated_source)},
            {"indicador": "diagnostico_deduplicacion", "valor": str(dedup_source)},
            {"indicador": "base_territorial", "valor": str(territorial_source)},
            {"indicador": "registros_origen", "valor": len(source)},
            {"indicador": "publicaciones_consolidadas", "valor": len(publications)},
            {"indicador": "registros_redundantes", "valor": len(source) - len(publications)},
            {"indicador": "decisiones_revisadas", "valor": len(decisions)},
            {"indicador": "tesis_con_campos_academicos", "valor": len(academic_fields)},
            {"indicador": "conflictos_academicos_auditados", "valor": len(academic_audit)},
            {"indicador": "migraciones_institucion_revista", "valor": len(institution_audit)},
            {"indicador": "claves_registro_maestro", "valor": len(master_registry)},
            {"indicador": "fusiones_ids_historicos", "valor": len(id_merges)},
            {
                "indicador": "clases_repositorio",
                "valor": int(
                    dimensions["DIM_REPOSITORIOS"][PUBLIC_REPOSITORY_CLASS_COL].nunique()
                )
                if "DIM_REPOSITORIOS" in dimensions
                else 0,
            },
            {
                "indicador": "clases_institucion",
                "valor": int(
                    dimensions["DIM_INSTITUCIONES"][INSTITUTION_CLASS_COL].nunique()
                )
                if "DIM_INSTITUCIONES" in dimensions
                else 0,
            },
            {
                "indicador": "articulos_de_conferencia",
                "valor": int(
                    publications[PUBLIC_SUBTYPE]
                    .eq("Artículo de conferencia")
                    .sum()
                ),
            },
        ]
    )
    contract = pd.DataFrame(
        [
            {"hoja": "BD_APP", "proposito": "Una fila por publicación consolidada", "clave": PUBLICATION_ID},
            {"hoja": "REGISTROS_ORIGEN", "proposito": "Registros homologados sin eliminación", "clave": RECORD_ID},
            {"hoja": "REGISTRO_PUBLICACION", "proposito": "Relación auditable registro-publicación", "clave": RECORD_ID},
            {"hoja": "DECISIONES_REVISADAS", "proposito": "Adjudicación de los 72 pares", "clave": "id_registro_a + id_registro_b"},
            {"hoja": "AUDITORIA_ACADEMICA", "proposito": "Combinaciones académicas resueltas", "clave": PUBLICATION_ID},
            {"hoja": "AUDITORIA_INSTITUCIONES", "proposito": "Migración de revistas o boletines desde institución hacia nombre de revista", "clave": RECORD_ID},
            {"hoja": "REGISTRO_MAESTRO_IDS", "proposito": "Claves de identidad persistentes", "clave": "CLAVE_IDENTIDAD"},
            {"hoja": "FUSIONES_IDS", "proposito": "Alias de identificadores históricos", "clave": "ID_PUBLICACION_ALIAS"},
            *[
                {
                    "hoja": sheet,
                    "proposito": "Relación publicación-categoría",
                    "clave": f"{PUBLICATION_ID} + categoria",
                }
                for sheet in dimensions
            ],
        ]
    )

    with pd.ExcelWriter(app_output, engine="xlsxwriter") as writer:
        publications.to_excel(writer, sheet_name="BD_APP", index=False)
        source.to_excel(writer, sheet_name="REGISTROS_ORIGEN", index=False)
        mapping.to_excel(writer, sheet_name="REGISTRO_PUBLICACION", index=False)
        decisions.to_excel(writer, sheet_name="DECISIONES_REVISADAS", index=False)
        academic_audit.to_excel(
            writer, sheet_name="AUDITORIA_ACADEMICA", index=False
        )
        institution_audit.to_excel(
            writer, sheet_name="AUDITORIA_INSTITUCIONES", index=False
        )
        master_registry.to_excel(
            writer, sheet_name="REGISTRO_MAESTRO_IDS", index=False
        )
        id_merges.to_excel(writer, sheet_name="FUSIONES_IDS", index=False)
        for sheet, dimension in dimensions.items():
            dimension.to_excel(writer, sheet_name=sheet, index=False)
        summary.to_excel(writer, sheet_name="RESUMEN_ADAPTACION", index=False)
        contract.to_excel(writer, sheet_name="CONTRATO_DATOS", index=False)

    with pd.ExcelWriter(territorial_output, engine="xlsxwriter") as writer:
        for sheet, frame in territorial_sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame(
            [
                {"indicador": "fecha_adaptacion", "valor": datetime.now().isoformat(timespec="seconds")},
                {"indicador": "base_principal_asociada", "valor": app_output.name},
                {"indicador": "publicaciones_consolidadas", "valor": len(publications)},
            ]
        ).to_excel(writer, sheet_name="TRAZABILIDAD_ADAPTACION", index=False)

    trace = docs_dir / "TRAZABILIDAD_BASE_ADAPTADA.md"
    trace.write_text(
        f"""# Trazabilidad de la base adaptada

## Ejecución vigente

- **Fecha:** {datetime.now().isoformat(timespec="seconds")}
- **Base homologada:** `{homologated_source}`
- **Diagnóstico de deduplicación:** `{dedup_source}`
- **Base territorial de origen:** `{territorial_source}`
- **Base principal:** `{app_output.name}`
- **Base territorial:** `{territorial_output.name}`
- **Registros de origen:** {len(source):,}
- **Publicaciones consolidadas:** {len(publications):,}
- **Registros redundantes consolidados:** {len(source) - len(publications):,}
- **Decisiones revisadas:** {len(decisions):,}
- **Tesis con clasificación académica pública:** {len(academic_fields):,}
- **Combinaciones académicas auditadas:** {len(academic_audit):,}
- **Migraciones institución-revista auditadas:** {len(institution_audit):,}
- **Claves persistentes registradas:** {len(master_registry):,}
- **Fusiones de identificadores históricos:** {len(id_merges):,}

## Contrato

- `BD_APP` contiene una fila por `ID_PUBLICACION_PROPUESTA`.
- `REGISTROS_ORIGEN` conserva las 6,574 filas homologadas.
- `REGISTRO_PUBLICACION` permite reconstruir cada agrupación.
- Las dimensiones y la base territorial cuentan publicaciones consolidadas.
- Las bases documentales se conservan como una relación multivaluada.
- `DIM_REPOSITORIOS` incorpora `{REPOSITORY_CLASS_COL}` y
  `{UNIVERSITY_REPOSITORY_COL}` para diferenciar repositorios universitarios,
  buscadores académicos, redes académicas, agregadores, indexadores,
  editoriales, revistas, bibliotecas y casos no clasificados.
- `{PUBLIC_REPOSITORY_CLASS_COL}` presenta cuatro categorías públicas:
  `Buscadores académicos`, `Repositorios institucionales`,
  `Repositorios universitarios` y `Revistas`. Las columnas
  `{PUBLIC_REPOSITORY_RULE_COL}` y `{PUBLIC_REPOSITORY_REVIEW_COL}` conservan
  la regla aplicada y señalan los valores ambiguos que requieren revisión.
- `DIM_INSTITUCIONES` incorpora `{INSTITUTION_CLASS_COL}` y
  `{IS_UNIVERSITY_COL}` para diferenciar universidades públicas nacionales,
  universidades privadas nacionales, universidades extranjeras, entidades
  estatales, centros de investigación, sociedades científicas, revistas o
  boletines mal ubicados y casos no clasificados.
- `GRADO_ACADEMICO_PUBLICO` muestra `Pregrado`, `Posgrado` u `Otros`.
- `NIVEL_ACADEMICO_PUBLICO` muestra `Pregrado`, `Maestría`, `Doctorado`,
  `Suficiencia profesional` u `Otros`.
- `No aplica` no se expone en los campos académicos públicos.
- `AUDITORIA_ACADEMICA` conserva las combinaciones de origen resueltas.
- `AUDITORIA_INSTITUCIONES` conserva los valores de revista o boletín
  detectados en `General_ Institución/Universidad`, su migración hacia
  `General_ Nombre de revista` y la institución final asignada.
- `HUELLA_PUBLICACION_PERSISTENTE` identifica de forma estable cada código.
- `REGISTRO_MAESTRO_IDS` relaciona DOI, URL, datos bibliográficos y registros
  con su identificador persistente.
- `FUSIONES_IDS` conserva alias cuando nuevas evidencias fusionen códigos.
- `TIPO_PUBLICACION_PUBLICO` presenta únicamente `Artículo` y `Tesis`.
- `SUBTIPO_PUBLICACION_PUBLICO` distingue `Artículo científico` y
  `Artículo de conferencia`; no se aplica a tesis.
- Los archivos fuente no fueron modificados.
""",
        encoding="utf-8",
    )
    print(f"APP_OUTPUT={app_output}")
    print(f"TERRITORIAL_OUTPUT={territorial_output}")
    print(f"TRACE={trace}")
    print(f"SOURCE_ROWS={len(source)}")
    print(f"PUBLICATIONS={len(publications)}")


if __name__ == "__main__":
    main()
