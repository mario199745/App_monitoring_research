from pathlib import Path

import pandas as pd

from institution_classification import (
    INSTITUTION_CLASS_COL,
    INSTITUTION_CLASSES,
    IS_UNIVERSITY_COL,
)
from repository_classification import (
    PUBLIC_REPOSITORY_CLASS_COL,
    PUBLIC_REPOSITORY_CLASSES,
    PUBLIC_REPOSITORY_REVIEW_COL,
    PUBLIC_REPOSITORY_RULE_COL,
    REPOSITORY_CLASS_COL,
    REPOSITORY_CLASSES,
    UNIVERSITY_REPOSITORY_COL,
)


SOURCE_RECORD_ID_COL = "ID_REGISTRO_ANALISIS"
RECORD_ID_COL = "ID_PUBLICACION_PROPUESTA"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"

REQUIRED_MAIN_COLUMNS = [
    "TIPO_PUBLICACION_NORM",
    "TIPO_PUBLICACION_PUBLICO",
    "SUBTIPO_PUBLICACION_PUBLICO",
    "General_ Año",
    "USAR_PARA_CONTEO_UNICO",
    "GRADO_ACADEMICO_PUBLICO",
    "NIVEL_ACADEMICO_PUBLICO",
    "HUELLA_PUBLICACION_PERSISTENTE",
    RECORD_ID_COL,
]

REQUIRED_DIMENSIONS = [
    "DIM_REPOSITORIOS",
    "DIM_AREAS_TEMATICAS",
    "DIM_EJES_TEMATICOS",
    "DIM_LINEAS_INVESTIGACION",
    "DIM_REGIONES_NORMALIZADAS",
    "DIM_INSTITUCIONES",
    "DIM_BASES_DOCUMENTALES",
]

REQUIRED_TERRITORIAL_SHEETS = [
    "MAPA_DEPARTAMENTOS",
    "REGIONES_EXPANDIDAS",
]


def latest_file(data_dir: Path, pattern: str) -> Path:
    files = [
        path
        for path in data_dir.glob(pattern)
        if path.is_file() and not path.name.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(pattern)
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    app_file = latest_file(data_dir, "BD_APP_FINAL_*.xlsx")
    territorial_file = latest_file(data_dir, "BD_APP_TERRITORIAL_*.xlsx")

    app_xl = pd.ExcelFile(app_file, engine="openpyxl")
    territorial_xl = pd.ExcelFile(territorial_file, engine="openpyxl")

    missing_sheets = [
        sheet
        for sheet in [
            "BD_APP",
            "REGISTROS_ORIGEN",
            "REGISTRO_PUBLICACION",
            "DECISIONES_REVISADAS",
            "AUDITORIA_ACADEMICA",
            "AUDITORIA_INSTITUCIONES",
            "AUDITORIA_REPOSITORIOS",
            "REGISTRO_MAESTRO_IDS",
            "FUSIONES_IDS",
            *REQUIRED_DIMENSIONS,
        ]
        if sheet not in app_xl.sheet_names
    ]
    if missing_sheets:
        raise AssertionError(f"Hojas faltantes en base principal: {missing_sheets}")

    missing_territorial = [
        sheet
        for sheet in REQUIRED_TERRITORIAL_SHEETS
        if sheet not in territorial_xl.sheet_names
    ]
    if missing_territorial:
        raise AssertionError(
            f"Hojas faltantes en base territorial: {missing_territorial}"
        )

    df = pd.read_excel(
        app_file,
        sheet_name="BD_APP",
        dtype="string",
        engine="openpyxl",
    )
    missing_columns = [
        column for column in REQUIRED_MAIN_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise AssertionError(f"Columnas faltantes: {missing_columns}")
    if df[RECORD_ID_COL].isna().any() or not df[RECORD_ID_COL].is_unique:
        raise AssertionError(f"{RECORD_ID_COL} no es una clave primaria válida.")
    if len(df) != 6217:
        raise AssertionError(
            f"Se esperaban 6,217 publicaciones consolidadas y se obtuvieron {len(df)}."
        )
    if (
        df["HUELLA_PUBLICACION_PERSISTENTE"].isna().any()
        or not df["HUELLA_PUBLICACION_PERSISTENTE"].is_unique
    ):
        raise AssertionError("Las huellas persistentes no son completas y únicas.")
    if set(df["TIPO_PUBLICACION_PUBLICO"].dropna()) != {"Artículo", "Tesis"}:
        raise AssertionError("TIPO_PUBLICACION_PUBLICO no cumple la jerarquía.")
    allowed_subtypes = {"Artículo científico", "Artículo de conferencia"}
    article_rows = df["TIPO_PUBLICACION_PUBLICO"].eq("Artículo")
    if not set(
        df.loc[article_rows, "SUBTIPO_PUBLICACION_PUBLICO"].dropna()
    ).issubset(allowed_subtypes):
        raise AssertionError("SUBTIPO_PUBLICACION_PUBLICO contiene valores inválidos.")
    if df.loc[
        ~article_rows, "SUBTIPO_PUBLICACION_PUBLICO"
    ].notna().any():
        raise AssertionError("Las tesis contienen subtipo de artículo.")
    conference_count = int(
        df["SUBTIPO_PUBLICACION_PUBLICO"].eq("Artículo de conferencia").sum()
    )
    if conference_count != 5:
        raise AssertionError(
            f"Se esperaban 5 artículos de conferencia y se obtuvieron {conference_count}."
        )
    non_thesis = df["TIPO_PUBLICACION_NORM"].ne("Tesis")
    if df.loc[
        non_thesis,
        ["GRADO_ACADEMICO_PUBLICO", "NIVEL_ACADEMICO_PUBLICO"],
    ].notna().any().any():
        raise AssertionError("Las publicaciones no tesis tienen nivel académico público.")
    allowed_grade = {"Pregrado", "Posgrado", "Otros"}
    allowed_level = {
        "Pregrado",
        "Maestría",
        "Doctorado",
        "Suficiencia profesional",
        "Otros",
    }
    thesis = df[df["TIPO_PUBLICACION_NORM"].eq("Tesis")]
    if not set(thesis["GRADO_ACADEMICO_PUBLICO"].dropna()).issubset(allowed_grade):
        raise AssertionError("GRADO_ACADEMICO_PUBLICO contiene valores inválidos.")
    if not set(thesis["NIVEL_ACADEMICO_PUBLICO"].dropna()).issubset(allowed_level):
        raise AssertionError("NIVEL_ACADEMICO_PUBLICO contiene valores inválidos.")
    database_names = (
        df["Nombre de Base de datos"].dropna().astype(str).str.strip()
    )
    numbered_database_names = database_names[
        database_names.str.match(r"^\d+\.\s*")
    ].unique()
    if len(numbered_database_names):
        raise AssertionError(
            "Persisten prefijos numéricos en Nombre de Base de datos: "
            + ", ".join(sorted(numbered_database_names))
        )

    ids = set(df[RECORD_ID_COL].astype(str))
    records = pd.read_excel(
        app_file,
        sheet_name="REGISTROS_ORIGEN",
        dtype="string",
        engine="openpyxl",
    )
    relation = pd.read_excel(
        app_file,
        sheet_name="REGISTRO_PUBLICACION",
        dtype="string",
        engine="openpyxl",
    )
    master_registry = pd.read_excel(
        app_file,
        sheet_name="REGISTRO_MAESTRO_IDS",
        dtype="string",
        engine="openpyxl",
    )
    if master_registry["CLAVE_IDENTIDAD"].duplicated().any():
        raise AssertionError("REGISTRO_MAESTRO_IDS contiene claves duplicadas.")
    if not ids.issubset(
        set(master_registry[RECORD_ID_COL].dropna().astype(str))
    ):
        raise AssertionError("El registro maestro no cubre todas las publicaciones.")
    if len(records) != 6574 or records[SOURCE_RECORD_ID_COL].duplicated().any():
        raise AssertionError("REGISTROS_ORIGEN no conserva los 6,574 registros.")
    if (
        len(relation) != 6574
        or relation[SOURCE_RECORD_ID_COL].duplicated().any()
        or set(relation[RECORD_ID_COL].astype(str)) != ids
    ):
        raise AssertionError("REGISTRO_PUBLICACION no cumple el contrato.")
    for sheet in REQUIRED_DIMENSIONS:
        dimension = pd.read_excel(
            app_file,
            sheet_name=sheet,
            dtype="string",
            engine="openpyxl",
        )
        required = {RECORD_ID_COL, "categoria"}
        if not required.issubset(dimension.columns):
            raise AssertionError(f"{sheet} no cumple el contrato mínimo.")
        if dimension.duplicated([RECORD_ID_COL, "categoria"]).any():
            raise AssertionError(f"{sheet} contiene relaciones duplicadas.")
        orphan_ids = set(dimension[RECORD_ID_COL].dropna().astype(str)) - ids
        if orphan_ids:
            raise AssertionError(f"{sheet} contiene identificadores huérfanos.")
        if sheet == "DIM_REPOSITORIOS":
            required_repository_columns = {
                PUBLIC_REPOSITORY_CLASS_COL,
                PUBLIC_REPOSITORY_RULE_COL,
                PUBLIC_REPOSITORY_REVIEW_COL,
                REPOSITORY_CLASS_COL,
                UNIVERSITY_REPOSITORY_COL,
            }
            if not required_repository_columns.issubset(dimension.columns):
                raise AssertionError(
                    "DIM_REPOSITORIOS no contiene la clasificación requerida."
                )
            repository_classes = set(
                dimension[REPOSITORY_CLASS_COL].dropna().astype(str)
            )
            if not repository_classes.issubset(REPOSITORY_CLASSES):
                raise AssertionError(
                    "DIM_REPOSITORIOS contiene clases no permitidas: "
                    + ", ".join(sorted(repository_classes - REPOSITORY_CLASSES))
                )
            university_flags = set(
                dimension[UNIVERSITY_REPOSITORY_COL].dropna().astype(str)
            )
            if not university_flags.issubset({"Si", "No", "Indeterminado"}):
                raise AssertionError(
                    "ES_REPOSITORIO_UNIVERSITARIO contiene valores inválidos."
                )
            public_classes = set(
                dimension[PUBLIC_REPOSITORY_CLASS_COL].dropna().astype(str)
            )
            if public_classes != PUBLIC_REPOSITORY_CLASSES:
                raise AssertionError(
                    "La clasificación pública de repositorios no contiene "
                    "exactamente las categorías públicas requeridas."
                )
            review_flags = set(
                dimension[PUBLIC_REPOSITORY_REVIEW_COL].dropna().astype(str)
            )
            if not review_flags.issubset({"Si", "No"}):
                raise AssertionError(
                    "REQUIERE_REVISION_REPOSITORIO contiene valores inválidos."
                )
            pending = dimension[PUBLIC_REPOSITORY_REVIEW_COL].eq("Si")
            if not dimension.loc[
                pending, PUBLIC_REPOSITORY_CLASS_COL
            ].eq("Otros").all():
                raise AssertionError(
                    "Los repositorios pendientes deben permanecer en Otros."
                )
        if sheet == "DIM_INSTITUCIONES":
            required_institution_columns = {
                INSTITUTION_CLASS_COL,
                IS_UNIVERSITY_COL,
            }
            if not required_institution_columns.issubset(dimension.columns):
                raise AssertionError(
                    "DIM_INSTITUCIONES no contiene la clasificación requerida."
                )
            institution_classes = set(
                dimension[INSTITUTION_CLASS_COL].dropna().astype(str)
            )
            if not institution_classes.issubset(INSTITUTION_CLASSES):
                raise AssertionError(
                    "DIM_INSTITUCIONES contiene clases no permitidas: "
                    + ", ".join(sorted(institution_classes - INSTITUTION_CLASSES))
                )
            university_flags = set(
                dimension[IS_UNIVERSITY_COL].dropna().astype(str)
            )
            if not university_flags.issubset({"Si", "No", "Indeterminado"}):
                raise AssertionError("ES_UNIVERSIDAD contiene valores inválidos.")
            if (
                dimension[INSTITUTION_CLASS_COL]
                .astype(str)
                .eq("Revista / boletin mal ubicado")
                .any()
            ):
                raise AssertionError(
                    "DIM_INSTITUCIONES conserva revistas o boletines mal ubicados."
                )

    institution_audit = pd.read_excel(
        app_file,
        sheet_name="AUDITORIA_INSTITUCIONES",
        dtype="string",
        engine="openpyxl",
    )
    required_audit_columns = {
        SOURCE_RECORD_ID_COL,
        "INSTITUCION_ORIGINAL",
        "REVISTA_FINAL",
        "INSTITUCION_FINAL",
    }
    if not required_audit_columns.issubset(institution_audit.columns):
        raise AssertionError("AUDITORIA_INSTITUCIONES no cumple el contrato mínimo.")

    repository_audit = pd.read_excel(
        app_file,
        sheet_name="AUDITORIA_REPOSITORIOS",
        dtype="string",
        engine="openpyxl",
    )
    required_repository_audit = {
        RECORD_ID_COL,
        "Repositorio_original",
        "Repositorio_actualizado",
        "Categoria_nueva",
        "CLASE_PUBLICA_ANTERIOR",
        "CLASE_PUBLICA_FINAL",
        "REGLA_APLICADA",
    }
    if not required_repository_audit.issubset(repository_audit.columns):
        raise AssertionError("AUDITORIA_REPOSITORIOS no cumple el contrato mínimo.")
    if len(repository_audit) != 135 or repository_audit[RECORD_ID_COL].duplicated().any():
        raise AssertionError(
            "AUDITORIA_REPOSITORIOS no conserva las 135 decisiones aprobadas."
        )
    if repository_audit["Repositorio_original"].str.strip().str.casefold().eq(
        "otros"
    ).any():
        raise AssertionError("La actualización modificó el valor exacto Otros.")
    allowed_repository_rules = {
        "REP_SOLICITADA_20260630",
        "REP_EVALUADA_20260630",
    }
    if not set(repository_audit["REGLA_APLICADA"].dropna()).issubset(
        allowed_repository_rules
    ):
        raise AssertionError("AUDITORIA_REPOSITORIOS contiene reglas inesperadas.")
    repository_dimension = pd.read_excel(
        app_file,
        sheet_name="DIM_REPOSITORIOS",
        dtype="string",
        engine="openpyxl",
    )
    verified_updates = repository_audit.merge(
        repository_dimension[
            [RECORD_ID_COL, "categoria", PUBLIC_REPOSITORY_CLASS_COL]
        ],
        left_on=[RECORD_ID_COL, "Repositorio_actualizado"],
        right_on=[RECORD_ID_COL, "categoria"],
        how="left",
        validate="one_to_one",
    )
    if verified_updates[PUBLIC_REPOSITORY_CLASS_COL].isna().any() or not (
        verified_updates[PUBLIC_REPOSITORY_CLASS_COL]
        == verified_updates["Categoria_nueva"]
    ).all():
        raise AssertionError(
            "Los nombres o clases solicitados no coinciden con DIM_REPOSITORIOS."
        )
    remaining_other_names = set(
        repository_dimension.loc[
            repository_dimension[PUBLIC_REPOSITORY_CLASS_COL].eq("Otros"),
            "categoria",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )
    if remaining_other_names != {"Otros"}:
        raise AssertionError(
            "Persisten fuentes identificables en la categoría pública Otros: "
            + ", ".join(sorted(remaining_other_names - {"Otros"}))
        )

    territorial = pd.read_excel(
        territorial_file,
        sheet_name="REGIONES_EXPANDIDAS",
        dtype="string",
        engine="openpyxl",
    )
    territorial_orphans = (
        set(territorial[RECORD_ID_COL].dropna().astype(str)) - ids
    )
    if territorial_orphans:
        raise AssertionError("La base territorial contiene identificadores huérfanos.")

    print(f"APP_FILE={app_file.name}")
    print(f"TERRITORIAL_FILE={territorial_file.name}")
    print(f"ROWS={len(df)}")
    print(f"COLUMNS={len(df.columns)}")
    print(f"UNIQUE_IDS={df[RECORD_ID_COL].nunique()}")
    print(f"SOURCE_RECORDS={len(records)}")
    print(f"MASTER_KEYS={len(master_registry)}")
    print("VALIDATION=OK")


if __name__ == "__main__":
    main()
