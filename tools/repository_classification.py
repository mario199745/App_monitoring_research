from __future__ import annotations

import re
import unicodedata


REPOSITORY_CLASS_COL = "CLASE_REPOSITORIO"
UNIVERSITY_REPOSITORY_COL = "ES_REPOSITORIO_UNIVERSITARIO"
PUBLIC_REPOSITORY_CLASS_COL = "CLASE_REPOSITORIO_PUBLICA"
PUBLIC_REPOSITORY_RULE_COL = "REGLA_CLASIFICACION_REPOSITORIO_PUBLICA"
PUBLIC_REPOSITORY_REVIEW_COL = "REQUIERE_REVISION_REPOSITORIO"

PUBLIC_REPOSITORY_CLASSES = {
    "Buscadores académicos",
    "Repositorios institucionales",
    "Repositorios universitarios",
    "Revistas",
}

REPOSITORY_CLASSES = {
    "Repositorio universitario",
    "Repositorio institucional publico",
    "Repositorio nacional/regional",
    "Buscador academico",
    "Red academica / perfil de autor",
    "Base bibliografica / indexador",
    "Editorial / plataforma de revistas",
    "Revista o portal especifico",
    "Biblioteca / archivo digital",
    "Otro / no clasificado",
}

SEARCH_ENGINES = {
    "GOOGLE",
    "GOOGLE SCHOLAR",
    "GOOHLE SCHOLAR",
    "SEMANTIC SCHOLAR",
}

ACADEMIC_NETWORKS = {
    "ACADEMIA.EDU",
    "AUTHOREA",
    "RESEARCHGATE",
    "RESEARCHGATE.",
}

NATIONAL_AGGREGATORS = {
    "ALICIA",
    "LICIA",
    "RENATI",
    "LA REFERENCIA",
    "CORE",
    "CORE.UK",
    "DIALNET",
    "REDALYC",
    "SCIELO",
}

BIBLIOGRAPHIC_INDEXES = {
    "AGRIS",
    "AGRITOP",
    "CABDIRECT",
    "CITESEERX",
    "INGENTACONNECT",
    "JSTOR",
    "NCBI",
    "PROQUEST",
    "PUBMED",
    "PUDMED",
    "SCOPUS",
}

PUBLISHER_PLATFORMS = {
    "AGUPUBS",
    "ANNUARL REVIEWS",
    "ANTHROSOURCE",
    "BESJOURNALS",
    "BETHAMOPEN",
    "BIOONE",
    "BIOTAXA",
    "CAMBRIDGE CORE",
    "CAMBRIGE UNIVERISTY",
    "CANADIAN SCIENCE PUBLISHING",
    "COPERNICUS PUBLICATIONS",
    "DUKE UNIVERSITY",
    "EGU",
    "ESA JOURNALS",
    "EUROPEAN GEOSCIENCE UNION",
    "FRONTIERSIN",
    "HINDAWI",
    "IOPSCIENCE",
    "LOPSCIENCE",
    "MDPI",
    "NATURE",
    "NEW PHYTOLOGIST",
    "OXFORD ACADEMIC",
    "PEERJ",
    "PENSOFT",
    "PLOS ONE",
    "PNAS",
    "ROYAL SOCIETY PUBLISHING",
    "ROYALTY SOCIETY",
    "SAGEPUB",
    "SCIENCE DIRECT",
    "SCIENCEADVANCE",
    "SPRINGER",
    "SPRINGER LINK",
    "TADFONELINE",
    "TAYLOR & FRANCIS",
    "WILEY ONLINE LIBRARY",
    "ZSLPUBLICATIONS",
}

LIBRARY_ARCHIVES = {
    "BIODIVERSITY HERITAGE LIBRARY BHL",
    "CEU ETD COLLECTION",
    "DIGITAL COLLECTIONS",
    "DIGITALCOOMONS",
    "DIGILTAL LIBRAY WASHIGTON U.",
    "DOCUMENTAL GREDOS",
    "DOMINIO PUBLICO",
    "LIBRARY",
    "SIT DIGITAL COLLECTIONS",
}

PUBLIC_INSTITUTIONS = {
    "AEMNP",
    "BAN",
    "CATIE",
    "CEPROSIMAD",
    "CICADFOR",
    "CIFOR",
    "CINCIA",
    "CONCYTEC",
    "CORBIDI",
    "DBCA",
    "ECOSUR",
    "EQUATORIAN INITIATIVE",
    "FOREST SERVICE -U.S. DEPARTMENT OF AGRICULTURE",
    "ICRAF",
    "IGP",
    "IIAP",
    "IMARPE",
    "INAIGEM",
    "INIA",
    "INS",
    "INSTITUO NACIONAL DE INNOVACION AGRARIA",
    "MINAM",
    "REPOSITORIO INIA",
    "REPOSITORIO SERNAMP",
    "REPOSITORIO SERNANP",
    "SERFOR",
    "SERNANP",
    "USDA",
}

UNIVERSITY_NAMES = {
    "ALAS PERUANAS",
    "CONTINENTAL",
    "COOMMOS CLARK UNIVERSITY",
    "CUNY ACADEMICS WORKS",
    "DEEPBLUE",
    "DUKESPACE",
    "ESPOCH",
    "ESCUELA SUPERIOR POLITECNICA DE CHIMBORAZO",
    "ESCOLARSHIP ORG.",
    "ETH ZURICH'S RESEARCH COLLECTION",
    "FIAT LUX",
    "LA MOLINA",
    "OAKTRUST",
    "PLURIVERSIDAD RICARDO PALMA",
    "PUCP",
    "RE.PUBLIC",
    "REPEBIS UPCH",
    "REPOSITORIO INSTITUCIONAL - CONTINENTAL",
    "REPOSITORIO PIRHUA",
    "REPOSITORIO UNALM",
    "REPOSITORIO UNC",
    "SCHOLARWORKS",
    "SIIS UNMAS",
    "TCU",
    "TESIS PUCP",
    "UDISTRITAL",
    "UMANIZALES",
    "UNIVERSIDAD DE CHILE",
    "UNIVERSIDAD INCA GARCILAZO DE LA VEGA",
    "UNIVERSIDAD NACIONAL DEL ALTIPLANO",
    "UNIVERSIDAD PERUANA LOS ANDES",
    "UNIVERSIDAD POLITECNICA",
    "UNIVERSIDADES",
    "VIRGINA TECH WORKS",
}

UNIVERSITY_ACRONYMS = {
    "UAC",
    "UANDINA",
    "UAP",
    "UARM",
    "UC",
    "UCALDAS",
    "UCSS",
    "UCSUR",
    "UCUNDINAMARCA",
    "UCV",
    "UDEP",
    "UDH",
    "UDL",
    "UFMG",
    "UIGV",
    "UISEK",
    "UMSA",
    "UNA",
    "UNACH",
    "UNAJ",
    "UNALM",
    "UNAMAD",
    "UNAMBA",
    "UNAP",
    "UNAS",
    "UNASAM",
    "UNC",
    "UNCHILE",
    "UNCP",
    "UNDAC",
    "UNESUM",
    "UNFV",
    "UNH",
    "UNHEVAL",
    "UNICA",
    "UNICAM",
    "UNITRU",
    "UNJ",
    "UNJBG",
    "UNLP",
    "UNMBA",
    "UNMSAM",
    "UNMSM",
    "UNPRG",
    "UNS",
    "UNSA",
    "UNSAAC",
    "UNSCH",
    "UNSM",
    "UNT",
    "UNTRM",
    "UNTUMBES",
    "UNU",
    "UPC",
    "UPCH",
    "UPLA",
    "UPM",
    "UPTC",
    "UPV",
    "UQROO",
    "URP",
    "USC",
    "USIL",
    "USMP",
    "USP",
    "USTA",
    "UTA",
    "UTP",
    "WFU",
}


def normalize_repository_key(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().upper())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def classify_repository(value) -> tuple[str, str]:
    key = normalize_repository_key(value)
    if not key or key in {"OTROS", "REPOSITORIO", "PUBLICACIONES"}:
        return "Otro / no clasificado", "Indeterminado"
    if key in SEARCH_ENGINES:
        return "Buscador academico", "No"
    if key in ACADEMIC_NETWORKS:
        return "Red academica / perfil de autor", "No"
    if key in NATIONAL_AGGREGATORS:
        return "Repositorio nacional/regional", "No"
    if key in BIBLIOGRAPHIC_INDEXES:
        return "Base bibliografica / indexador", "No"
    if key.startswith("REVISTA ") or key in {"CHECKLIST", "PHYTOKEYS"}:
        return "Revista o portal especifico", "No"
    if key in PUBLISHER_PLATFORMS:
        return "Editorial / plataforma de revistas", "No"
    if key in LIBRARY_ARCHIVES or "LIBRARY" in key or "COLLECTION" in key:
        return "Biblioteca / archivo digital", "No"
    if key in PUBLIC_INSTITUTIONS:
        return "Repositorio institucional publico", "No"
    if (
        key in UNIVERSITY_NAMES
        or key in UNIVERSITY_ACRONYMS
        or key.startswith("REPOSITORIO U")
        or key.startswith("UNIVERSIDAD ")
    ):
        return "Repositorio universitario", "Si"
    return "Otro / no clasificado", "Indeterminado"


def classify_public_repository(value) -> tuple[str | None, str, str]:
    """Reduce la taxonomía técnica a cuatro clases públicas auditables."""
    technical_class, _ = classify_repository(value)
    public_mapping = {
        "Buscador academico": "Buscadores académicos",
        "Red academica / perfil de autor": "Buscadores académicos",
        "Base bibliografica / indexador": "Buscadores académicos",
        "Repositorio nacional/regional": "Buscadores académicos",
        "Repositorio institucional publico": "Repositorios institucionales",
        "Biblioteca / archivo digital": "Repositorios institucionales",
        "Repositorio universitario": "Repositorios universitarios",
        "Editorial / plataforma de revistas": "Revistas",
        "Revista o portal especifico": "Revistas",
    }
    public_class = public_mapping.get(technical_class)
    if public_class is None:
        return None, "REP_PUBLICA_005_PENDIENTE_REVISION", "Si"
    rule_number = {
        "Buscadores académicos": "001",
        "Repositorios institucionales": "002",
        "Repositorios universitarios": "003",
        "Revistas": "004",
    }[public_class]
    return public_class, f"REP_PUBLICA_{rule_number}", "No"
