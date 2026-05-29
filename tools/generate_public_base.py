from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


APP_PATTERN = "BD_APP_FINAL_*.xlsx"
TERRITORIAL_PATTERN = "BD_APP_TERRITORIAL_*.xlsx"
APP_SHEET = "BD_APP"
MASTER_KEY_COL = "CLAVE_BIBLIOGRAFICA_MASTER"
RECORD_ID_COL = "ID_REGISTRO_ANALISIS"
UNIQUE_COL = "USAR_PARA_CONTEO_UNICO"

PUBLIC_EMPTY_LABEL = "Otros"

CATEGORICAL_PUBLIC_COLUMNS = [
    "Nombre de Base de datos",
    "General_ Repositorio",
    "General_ Tipo de Publicación",
    "General_ Tipo de tesis Pre/Posgrado",
    "General_ Institución/Universidad",
    "General_ SIGLAS UNIVERSIDAD/INSTITUCIÓN",
    "General_ Nombre de revista",
    "General_ Idioma",
    "General_ Lugar de Publicación",
    "General_ Publicación Nacional/Extranjera",
    "General_ Tipo de contenido (TD=texto disponible, TN=Texto no disponible)",
    "Ubicación_Ambito de estudio/Tipo de ecosistema (acuático, terrestre)",
    "Ubicación_Región de estudio",
    "Ubicación_Localidad (comunidad campesina u otros)",
    "Ubicación_Distrito",
    "Ubicación_Provincia",
    "Especie_Nombre científico",
    "ANIFFS: Eje Temático",
    "ANIFFS: Área Temática",
    "ANIFFS: Linea de investigación",
    "Categoria_Tesis_Articulo",
    "TIPO_PUBLICACION_NORM",
    "REGION_NORM_SUGERIDA",
]


def latest_source_file(data_dir: Path, pattern: str) -> Path:
    files = [
        path
        for path in data_dir.glob(pattern)
        if not path.name.startswith("~$") and "PUBLICA_UNICA" not in path.name.upper()
    ]
    if not files:
        raise FileNotFoundError(f"No source files found for pattern {pattern}")
    return max(files, key=lambda path: path.stat().st_mtime)


def non_empty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def recode_empty_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    rows: list[dict[str, object]] = []

    for column in CATEGORICAL_PUBLIC_COLUMNS:
        if column not in result.columns:
            continue

        before_empty = int((~non_empty_mask(result[column])).sum())
        result[column] = result[column].where(non_empty_mask(result[column]), PUBLIC_EMPTY_LABEL)
        result[column] = result[column].astype(str).str.strip().replace("", PUBLIC_EMPTY_LABEL)
        rows.append(
            {
                "campo": column,
                "vacios_recodificados_como_otros": before_empty,
            }
        )

    return result, pd.DataFrame(rows)


def build_unique_public_base(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_orden_original"] = range(len(work))

    if UNIQUE_COL in work.columns:
        work = work[work[UNIQUE_COL].fillna("").astype(str).str.upper().eq("SI")].copy()

    work = work.sort_values("_orden_original")
    if MASTER_KEY_COL in work.columns:
        work = work.drop_duplicates(subset=[MASTER_KEY_COL], keep="first")

    work = work.drop(columns=["_orden_original"])
    if UNIQUE_COL in work.columns:
        work[UNIQUE_COL] = "SI"
    return work.reset_index(drop=True)


def build_public_territorial(
    unique_df: pd.DataFrame,
    territorial_sheets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    expanded = territorial_sheets["REGIONES_EXPANDIDAS"].copy()
    selected = unique_df[[MASTER_KEY_COL, RECORD_ID_COL]].dropna().copy()
    selected[MASTER_KEY_COL] = selected[MASTER_KEY_COL].astype(str)
    selected[RECORD_ID_COL] = selected[RECORD_ID_COL].astype(str)

    master_to_public_record = dict(zip(selected[MASTER_KEY_COL], selected[RECORD_ID_COL]))
    expanded[MASTER_KEY_COL] = expanded[MASTER_KEY_COL].astype(str)
    public_expanded = expanded[expanded[MASTER_KEY_COL].isin(master_to_public_record)].copy()
    public_expanded[RECORD_ID_COL] = public_expanded[MASTER_KEY_COL].map(master_to_public_record)
    if UNIQUE_COL in public_expanded.columns:
        public_expanded[UNIQUE_COL] = "SI"

    dedup_columns = [RECORD_ID_COL, "DEP_KEY"]
    public_expanded = public_expanded.drop_duplicates(subset=dedup_columns, keep="first")

    geo_rows = public_expanded[public_expanded.get("DEP_EN_GEOJSON", False).astype(bool)].copy()
    counts = (
        geo_rows.groupby("DEP_KEY", dropna=False)
        .agg(
            registros_territoriales=(RECORD_ID_COL, "count"),
            publicaciones_unicas=(MASTER_KEY_COL, "nunique"),
            publicaciones_bibliograficas=(MASTER_KEY_COL, "nunique"),
        )
        .reset_index()
    )

    map_base = territorial_sheets["MAPA_DEPARTAMENTOS"][
        ["IDDPTO", "DEPARTAMEN_GEO", "DEP_KEY", "CAPITAL"]
    ].copy()
    public_map = map_base.merge(counts, on="DEP_KEY", how="left")
    for column in ["registros_territoriales", "publicaciones_unicas", "publicaciones_bibliograficas"]:
        public_map[column] = public_map[column].fillna(0).astype(int)

    no_crossed = public_expanded[~public_expanded.get("DEP_EN_GEOJSON", False).astype(bool)].copy()
    no_crossed = no_crossed[[c for c in ["DEPARTAMENTO_BASE", "DEP_KEY"] if c in no_crossed.columns]]
    no_crossed = no_crossed.drop_duplicates().reset_index(drop=True)

    coverage = pd.DataFrame(
        [
            {"indicador": "publicaciones_unicas", "valor": len(unique_df)},
            {"indicador": "relaciones_territoriales_publicas", "valor": len(public_expanded)},
            {"indicador": "relaciones_territoriales_en_geojson", "valor": len(geo_rows)},
            {
                "indicador": "departamentos_con_datos",
                "valor": int((public_map["publicaciones_bibliograficas"] > 0).sum()),
            },
            {"indicador": "departamentos_sin_cruce_geojson", "valor": len(no_crossed)},
        ]
    )

    return {
        "MAPA_DEPARTAMENTOS": public_map,
        "COBERTURA_REGION": coverage,
        "REGIONES_EXPANDIDAS": public_expanded.reset_index(drop=True),
        "REGIONES_NO_CRUZADAS": no_crossed,
        "DEPARTAMENTOS_GEOJSON": territorial_sheets["DEPARTAMENTOS_GEOJSON"].copy(),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    app_source = latest_source_file(data_dir, APP_PATTERN)
    territorial_source = latest_source_file(data_dir, TERRITORIAL_PATTERN)

    df = pd.read_excel(app_source, sheet_name=APP_SHEET, dtype=object)
    df.columns = [str(column).strip() for column in df.columns]

    unique_df = build_unique_public_base(df)
    public_df, recode_summary = recode_empty_categories(unique_df)

    territorial_xl = pd.ExcelFile(territorial_source)
    territorial_sheets = {
        sheet: pd.read_excel(territorial_source, sheet_name=sheet, dtype=object)
        for sheet in territorial_xl.sheet_names
    }
    public_territorial = build_public_territorial(public_df, territorial_sheets)

    app_output = data_dir / f"BD_APP_FINAL_{timestamp}_PUBLICA_UNICA.xlsx"
    territorial_output = data_dir / f"BD_APP_TERRITORIAL_{timestamp}_PUBLICA_UNICA.xlsx"

    base_summary = pd.DataFrame(
        [
            {"indicador": "archivo_fuente_app", "valor": app_source.name},
            {"indicador": "archivo_fuente_territorial", "valor": territorial_source.name},
            {"indicador": "filas_base_original", "valor": len(df)},
            {"indicador": "filas_publicas_unicas", "valor": len(public_df)},
            {
                "indicador": "publicaciones_unicas",
                "valor": public_df[MASTER_KEY_COL].nunique(dropna=True),
            },
            {
                "indicador": "campos_categoricos_vacios_recodificados",
                "valor": int(recode_summary["vacios_recodificados_como_otros"].sum()),
            },
        ]
    )

    with pd.ExcelWriter(app_output, engine="xlsxwriter") as writer:
        public_df.to_excel(writer, sheet_name=APP_SHEET, index=False)
        base_summary.to_excel(writer, sheet_name="RESUMEN_BASE_PUBLICA", index=False)
        recode_summary.to_excel(writer, sheet_name="CAMPOS_OTROS", index=False)

    with pd.ExcelWriter(territorial_output, engine="xlsxwriter") as writer:
        for sheet, sheet_df in public_territorial.items():
            sheet_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"APP_OUTPUT={app_output}")
    print(f"TERRITORIAL_OUTPUT={territorial_output}")
    print(f"ROWS_ORIGINAL={len(df)}")
    print(f"ROWS_PUBLIC_UNIQUE={len(public_df)}")
    print(f"EMPTY_CATEGORICAL_TO_OTROS={int(recode_summary['vacios_recodificados_como_otros'].sum())}")


if __name__ == "__main__":
    main()
