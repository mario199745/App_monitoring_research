from pathlib import Path

import pandas as pd


SOURCE_RECORD_ID_COL = "ID_REGISTRO_ANALISIS"
RECORD_ID_COL = "ID_PUBLICACION_PROPUESTA"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"

REQUIRED_MAIN_COLUMNS = [
    "TIPO_PUBLICACION_NORM",
    "General_ Año",
    "USAR_PARA_CONTEO_UNICO",
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
    print("VALIDATION=OK")


if __name__ == "__main__":
    main()
