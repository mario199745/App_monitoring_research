from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


ID = "ID_PUBLICACION_PROPUESTA"
RAW_TYPE = "General_ Tipo de Publicación"
RAW_THESIS_LEVEL = "General_ Tipo de tesis Pre/Posgrado"
TITLE = "General_ Título"
SUMMARY = "General_ Resumen"
JOURNAL = "General_ Nombre de revista"
REPOSITORY = "General_ Repositorio"
DOI = "General_ DOI"
VOLUME = "General_ Volume"
ISSUE = "General_ Issue"
URL = "General_ Enlace"
CURRENT_TYPE = "TIPO_PUBLICACION_PUBLICO"
CURRENT_SUBTYPE = "SUBTIPO_PUBLICACION_PUBLICO"


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def present(value: object) -> bool:
    return bool(norm(value)) and norm(value) not in {"otros", "no aplica", "nan"}


PATTERNS = {
    "Informe técnico": r"\b(informe tecnico|technical report|research report|rapport technique)\b",
    "Documento de trabajo": r"\b(documento de trabajo|working paper|discussion paper|document de travail)\b",
    "Boletín técnico": r"\b(boletin tecnico|technical bulletin|research bulletin)\b",
    "Manual o guía": r"\b(manual|guia (?:tecnica|practica|metodologica)|handbook|field guide|guide to)\b",
    "Protocolo": r"\b(protocolo|protocol for|protocol manual)\b",
    "Plan o estrategia": r"\b(plan (?:de|nacional|regional)|estrategia (?:de|nacional|regional)|action plan|management plan)\b",
    "Diagnóstico o evaluación": r"\b(diagnostico|diagnostic report|evaluacion (?:tecnica|ambiental)|assessment report)\b",
    "Capítulo de libro": r"\b(capitulo (?:de|del) libro|book chapter|chapter \d+|chapter in)\b",
    "Libro o monografía": r"\b(monografia|monograph|edited book|libro)\b",
    "Artículo de revisión": r"\b(revision (?:sistematica|bibliografica|de literatura)|systematic review|literature review|review article|meta-analysis|metaanalisis)\b",
    "Nota científica": r"\b(nota cientifica|scientific note|short communication|research note|nota breve)\b",
    "Publicación de evento": r"\b(congreso|conference|symposium|simposio|proceedings|conference paper|ponencia|poster)\b",
    "Conjunto de datos": r"\b(dataset|data set|conjunto de datos|data paper)\b",
}

TECHNICAL_SUBTYPES = {
    "Informe técnico",
    "Documento de trabajo",
    "Boletín técnico",
    "Manual o guía",
    "Protocolo",
    "Plan o estrategia",
    "Diagnóstico o evaluación",
}


def first_match(text: str, allowed: set[str]) -> tuple[str, str]:
    for subtype, pattern in PATTERNS.items():
        if subtype in allowed:
            match = re.search(pattern, text)
            if match:
                return subtype, match.group(0)
    return "", ""


def classify(row: pd.Series) -> dict[str, object]:
    raw_type = norm(row.get(RAW_TYPE))
    title = norm(row.get(TITLE))
    context = " | ".join(
        norm(row.get(column))
        for column in [TITLE, REPOSITORY, JOURNAL, URL]
        if present(row.get(column))
    )
    scholarly_signals = [
        name
        for name, column in [("revista", JOURNAL), ("doi", DOI), ("volumen", VOLUME), ("numero", ISSUE)]
        if present(row.get(column))
    ]
    url_text = norm(row.get(URL))
    editorial_domains = {
        "sciencedirect": "portal editorial",
        "redalyc": "portal editorial",
        "ncbi.nlm.nih.gov/pmc/articles": "artículo en PMC",
        "bioone.org/journals": "portal editorial",
        "copernicus.org/articles": "portal editorial",
        "scielo": "portal editorial",
        "springer.com/article": "portal editorial",
        "link.springer.com/article": "portal editorial",
        "tandfonline.com": "portal editorial",
        "wiley.com/doi": "portal editorial",
    }
    for fragment, label in editorial_domains.items():
        if fragment in url_text and label not in scholarly_signals:
            scholarly_signals.append(label)

    result = {
        "TIPO_PROPUESTO": "Artículo",
        "SUBTIPO_PROPUESTO": "Artículo científico",
        "CONFIANZA_PROPUESTA": "Media" if scholarly_signals else "Baja",
        "REQUIERE_REVISION_MANUAL": "SI" if not scholarly_signals else "NO",
        "EVIDENCIA_CLASIFICACION": ", ".join(scholarly_signals) or "Sin señales editoriales estructuradas",
        "REGLA_APLICADA": "ARTICULO_POR_DEFECTO",
    }

    if raw_type == "tesis":
        level = norm(row.get(RAW_THESIS_LEVEL))
        subtype_map = {
            "doctorado": "Tesis doctoral",
            "maestria": "Tesis de maestría",
            "pregrado": "Tesis de pregrado",
            "suficiencia profesional": "Trabajo de suficiencia profesional",
            "posgrado no especificado": "Tesis de posgrado no especificada",
            "nivel no especificado": "Tesis de nivel no especificado",
            "otros": "Tesis de nivel no especificado",
        }
        subtype = subtype_map.get(level, "Tesis de nivel no especificado")
        result.update(
            TIPO_PROPUESTO="Tesis",
            SUBTIPO_PROPUESTO=subtype,
            CONFIANZA_PROPUESTA="Alta" if level in subtype_map and level not in {"otros", "nivel no especificado"} else "Baja",
            REQUIERE_REVISION_MANUAL="NO" if level in subtype_map and level not in {"otros", "nivel no especificado"} else "SI",
            EVIDENCIA_CLASIFICACION=f"Tipo original=Tesis; nivel original={row.get(RAW_THESIS_LEVEL, '')}",
            REGLA_APLICADA="TESIS_NIVEL_ORIGINAL",
        )
        return result

    if raw_type == "articulo de conferencia":
        result.update(
            TIPO_PROPUESTO="Publicación de evento científico",
            SUBTIPO_PROPUESTO="Artículo de conferencia",
            CONFIANZA_PROPUESTA="Alta",
            REQUIERE_REVISION_MANUAL="NO",
            EVIDENCIA_CLASIFICACION="Tipo original=Artículo de conferencia",
            REGLA_APLICADA="EVENTO_TIPO_ORIGINAL",
        )
        return result

    event_subtype, event_match = first_match(title, {"Publicación de evento"})
    if event_subtype and not scholarly_signals:
        result.update(
            TIPO_PROPUESTO="Publicación de evento científico",
            SUBTIPO_PROPUESTO="Ponencia o memoria de evento",
            CONFIANZA_PROPUESTA="Media",
            REQUIERE_REVISION_MANUAL="SI",
            EVIDENCIA_CLASIFICACION=f"Indicador en título: {event_match}",
            REGLA_APLICADA="EVENTO_POR_TITULO",
        )
        return result

    technical_subtype, technical_match = first_match(title, TECHNICAL_SUBTYPES)
    if technical_subtype:
        conflict = bool(scholarly_signals)
        result.update(
            TIPO_PROPUESTO="Documento técnico" if not conflict else "Artículo",
            SUBTIPO_PROPUESTO=technical_subtype if not conflict else "Artículo científico",
            CONFIANZA_PROPUESTA="Media" if not conflict else "Baja",
            REQUIERE_REVISION_MANUAL="SI",
            EVIDENCIA_CLASIFICACION=(
                f"Indicador en título: {technical_match}; "
                + (f"conflicto con {', '.join(scholarly_signals)}" if conflict else "sin señales editoriales")
            ),
            REGLA_APLICADA="TECNICO_POR_TITULO" if not conflict else "TECNICO_CON_CONFLICTO_EDITORIAL",
        )
        return result

    book_subtype, book_match = first_match(title, {"Capítulo de libro", "Libro o monografía"})
    if book_subtype:
        if scholarly_signals:
            result.update(
                CONFIANZA_PROPUESTA="Media",
                REQUIERE_REVISION_MANUAL="SI",
                EVIDENCIA_CLASIFICACION=f"Indicador bibliográfico en título: {book_match}; conflicto con {', '.join(scholarly_signals)}",
                REGLA_APLICADA="LIBRO_CON_CONFLICTO_EDITORIAL",
            )
            return result
        result.update(
            TIPO_PROPUESTO="Libro",
            SUBTIPO_PROPUESTO=book_subtype,
            CONFIANZA_PROPUESTA="Media",
            REQUIERE_REVISION_MANUAL="SI",
            EVIDENCIA_CLASIFICACION=f"Indicador en título: {book_match}",
            REGLA_APLICADA="LIBRO_POR_TITULO",
        )
        return result

    special_subtype, special_match = first_match(title, {"Artículo de revisión", "Nota científica", "Conjunto de datos"})
    if special_subtype:
        if special_subtype == "Conjunto de datos":
            # "Dataset" en el título también es frecuente en artículos que
            # describen un corpus; una señal editorial impide convertirlo en
            # conjunto de datos sin inspección humana.
            if scholarly_signals:
                result.update(
                    CONFIANZA_PROPUESTA="Media",
                    REQUIERE_REVISION_MANUAL="SI",
                    EVIDENCIA_CLASIFICACION=f"Indicador en título: {special_match}; conflicto con {', '.join(scholarly_signals)}",
                    REGLA_APLICADA="DATOS_CON_CONFLICTO_EDITORIAL",
                )
                return result
            proposed_type = "Datos de investigación"
        else:
            proposed_type = "Artículo"
        result.update(
            TIPO_PROPUESTO=proposed_type,
            SUBTIPO_PROPUESTO=special_subtype,
            CONFIANZA_PROPUESTA="Media",
            REQUIERE_REVISION_MANUAL="SI",
            EVIDENCIA_CLASIFICACION=f"Indicador en título: {special_match}; señales: {', '.join(scholarly_signals) or 'ninguna'}",
            REGLA_APLICADA="SUBTIPO_POR_TITULO",
        )
        return result

    # Context-only keywords are useful as warnings, never as automatic changes.
    context_subtype, context_match = first_match(context, TECHNICAL_SUBTYPES | {"Publicación de evento", "Capítulo de libro", "Libro o monografía"})
    if context_subtype:
        result.update(
            REQUIERE_REVISION_MANUAL="SI",
            CONFIANZA_PROPUESTA="Baja",
            EVIDENCIA_CLASIFICACION=f"Indicador solo en metadatos contextuales: {context_match}; señales: {', '.join(scholarly_signals) or 'ninguna'}",
            REGLA_APLICADA="ALERTA_CONTEXTO_SIN_RECLASIFICAR",
        )
    return result


def autosize(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
    for index, column in enumerate(frame.columns):
        sample = frame[column].fillna("").astype(str).head(500)
        width = min(max(len(str(column)), sample.map(len).max() if not sample.empty else 0) + 2, 55)
        worksheet.set_column(index, index, width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita y propone tipos documentales sin modificar el maestro.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("auditoria"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(args.input, sheet_name="BD_APP", dtype=object, engine="openpyxl")
    required = {ID, RAW_TYPE, RAW_THESIS_LEVEL, TITLE, CURRENT_TYPE, CURRENT_SUBTYPE}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    proposals = pd.DataFrame([classify(row) for _, row in data.iterrows()], index=data.index)
    audit_columns = [
        ID, RAW_TYPE, RAW_THESIS_LEVEL, CURRENT_TYPE, CURRENT_SUBTYPE, TITLE,
        JOURNAL, REPOSITORY, DOI, VOLUME, ISSUE, URL,
    ]
    audit = pd.concat([data[audit_columns], proposals], axis=1)
    audit["CAMBIO_TIPO"] = audit[CURRENT_TYPE].fillna("").ne(audit["TIPO_PROPUESTO"])
    audit["CAMBIO_SUBTIPO"] = audit[CURRENT_SUBTYPE].fillna("").ne(audit["SUBTIPO_PROPUESTO"])
    audit["TITULO_VACIO"] = ~data[TITLE].map(present)

    summary_type = (
        audit.groupby(["TIPO_PROPUESTO", "SUBTIPO_PROPUESTO", "CONFIANZA_PROPUESTA"], dropna=False)
        .size().rename("PUBLICACIONES").reset_index().sort_values("PUBLICACIONES", ascending=False)
    )
    summary_rules = (
        audit.groupby(["REGLA_APLICADA", "REQUIERE_REVISION_MANUAL"], dropna=False)
        .size().rename("PUBLICACIONES").reset_index().sort_values("PUBLICACIONES", ascending=False)
    )
    current = (
        audit.groupby([CURRENT_TYPE, CURRENT_SUBTYPE], dropna=False)
        .size().rename("PUBLICACIONES").reset_index().sort_values("PUBLICACIONES", ascending=False)
    )
    quality = pd.DataFrame(
        [
            ("Publicaciones auditadas", len(audit)),
            ("Tipos originales distintos", data[RAW_TYPE].nunique(dropna=True)),
            ("Títulos vacíos", int(audit["TITULO_VACIO"].sum())),
            ("Cambios de tipo candidatos", int(audit["CAMBIO_TIPO"].sum())),
            ("Cambios de subtipo candidatos", int(audit["CAMBIO_SUBTIPO"].sum())),
            ("Revisión manual requerida", int(audit["REQUIERE_REVISION_MANUAL"].eq("SI").sum())),
            ("Confianza alta", int(audit["CONFIANZA_PROPUESTA"].eq("Alta").sum())),
            ("Confianza media", int(audit["CONFIANZA_PROPUESTA"].eq("Media").sum())),
            ("Confianza baja", int(audit["CONFIANZA_PROPUESTA"].eq("Baja").sum())),
        ], columns=["INDICADOR", "VALOR"]
    )
    rules = pd.DataFrame(
        [
            ("Principio", "No modificar automáticamente el maestro; toda inferencia queda trazable."),
            ("Tesis", "Usar el tipo y nivel originales; los niveles no especificados requieren revisión."),
            ("Evento", "Aceptar el tipo original; por palabras del título, proponer con revisión."),
            ("Documento técnico", "Proponer solo con indicador explícito en título y sin DOI/revista/volumen/número."),
            ("Conflicto editorial", "Si un título parece técnico pero tiene señales de revista, conservar Artículo y revisar."),
            ("Libro", "Proponer por indicador explícito; requiere revisión porque no hay ISBN ni tipo editorial estructurado."),
            ("Artículo", "Conservar como categoría residual; la ausencia de señales editoriales reduce la confianza."),
            ("Revista", "Es medio/fuente de publicación, no tipo documental."),
        ], columns=["REGLA", "DESCRIPCION"]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"AUDITORIA_TIPOS_PUBLICACION_{stamp}.xlsx"
    review = audit[audit["REQUIERE_REVISION_MANUAL"].eq("SI")].copy()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        frames = {
            "INDICADORES": quality,
            "CLASIFICACION_ACTUAL": current,
            "CLASIFICACION_PROPUESTA": summary_type,
            "RESUMEN_REGLAS": summary_rules,
            "REGLAS": rules,
            "REVISION_MANUAL": review,
            "AUDITORIA_COMPLETA": audit,
        }
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            autosize(writer, sheet, frame)

    print(f"OUTPUT={output.resolve()}")
    print(quality.to_string(index=False))
    print("\nCLASIFICACION PROPUESTA")
    print(summary_type.to_string(index=False))


if __name__ == "__main__":
    main()
