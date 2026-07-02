# Diccionario de datos de BD_APP

Grano: una fila por publicación consolidada. Clave primaria: `ID_PUBLICACION_PROPUESTA`.

Clasificación de procedencia:

- **Identidad:** controla identificación, deduplicación y trazabilidad.
- **Representativo:** procede del registro elegido como representante; puede no contener todos los valores del grupo.
- **Consolidado:** combina o deriva información de todos los registros agrupados.
- **Público/derivado:** clasificación preparada para la interfaz.

| N.º | Campo | Procedencia | Definición y regla de uso |
|---:|---|---|---|
| 1 | `ID_PUBLICACION_PROPUESTA` | Identidad | Identificador único de la publicación consolidada (`PUB-*`). Clave para unir dimensiones y filtros. |
| 2 | `ID_REGISTRO_REPRESENTATIVO` | Identidad | Registro fuente seleccionado para aportar los campos representativos de `BD_APP`. No equivale a todos los registros del grupo. |
| 3 | `REGISTROS_EN_GRUPO` | Consolidado | Número de registros originales agrupados en la publicación. Debe coincidir con `REGISTRO_PUBLICACION`. |
| 4 | `REGLAS_AGRUPACION` | Consolidado | Reglas de coincidencia usadas para deduplicar, por ejemplo DOI, título o clave bibliográfica. |
| 5 | `ESTADO` | Consolidado | Estado de la decisión de agrupación o revisión. |
| 6 | `ID_GLOBAL` | Representativo | Identificador global heredado del registro fuente, si existía. No es la clave maestra actual. |
| 7 | `COD BD SERFOR` | Representativo | Código del registro en la base SERFOR de procedencia. |
| 8 | `General_ Repositorio` | Representativo | Repositorio informado en la fila representante. Para el conjunto completo usar `DIM_REPOSITORIOS`. |
| 9 | `General_ Tipo de Publicación` | Representativo | Tipo documental original, previo a la normalización pública. |
| 10 | `General_ Tipo de tesis Pre/Posgrado` | Representativo | Nivel de tesis indicado originalmente. Para análisis usar los campos académicos públicos. |
| 11 | `General_ Institución/Universidad` | Representativo | Institución o universidad asociada al documento. No representa el lugar donde se realizó el estudio. Para todos los valores usar `DIM_INSTITUCIONES`. |
| 12 | `General_ SIGLAS UNIVERSIDAD/INSTITUCIÓN` | Representativo | Siglas de la institución asociada. |
| 13 | `General_ Nombre de revista` | Representativo | Revista o publicación seriada donde apareció el documento. |
| 14 | `General_ Volume` | Representativo | Volumen bibliográfico. |
| 15 | `General_ Issue` | Representativo | Número o fascículo bibliográfico. |
| 16 | `General_ Page start` | Representativo | Página inicial. |
| 17 | `General_ Page end` | Representativo | Página final. |
| 18 | `General_ Número de páginas` | Representativo | Extensión total declarada del documento. |
| 19 | `General_ Año` | Representativo | Año de publicación. Debe validarse como año plausible. |
| 20 | `General_ Título` | Representativo | Título de la publicación usado para presentación y deduplicación. |
| 21 | `General_ Autor(es)` | Representativo | Autoría textual del documento. |
| 22 | `General_ Resumen` | Representativo | Resumen o abstract. |
| 23 | `General_ Palabras clave del documento` | Representativo | Palabras clave declaradas en la fuente. |
| 24 | `General_ Idioma` | Representativo | Idioma principal del documento. |
| 25 | `General_ Lugar de Publicación` | Representativo | Lugar editorial o de publicación; no debe confundirse con la región de estudio. |
| 26 | `General_ Publicación Nacional/Extranjera` | Representativo | Clasificación del ámbito editorial nacional o extranjero. |
| 27 | `General_ Tipo de contenido (TD=texto disponible, TN=Texto no disponible)` | Representativo | Indicador de disponibilidad del texto, no de acceso abierto. |
| 28 | `General_ DOI` | Representativo | DOI tal como fue registrado en la fuente. |
| 29 | `General_ Enlace` | Representativo | URL original asociada al registro. |
| 30 | `Ubicación_Ambito de estudio/Tipo de ecosistema (acuático, terrestre)` | Representativo | Ambiente o ecosistema donde se desarrolla el estudio. |
| 31 | `Ubicación_Región de estudio` | Representativo | Región de realización o intervención indicada por la fila representante. Para cobertura completa usar la dimensión territorial. |
| 32 | `Ubicación_Localidad (comunidad campesina u otros)` | Representativo | Localidad específica del estudio. |
| 33 | `Ubicación_Distrito` | Representativo | Distrito donde se realizó el estudio. |
| 34 | `Ubicación_Provincia` | Representativo | Provincia donde se realizó el estudio. |
| 35 | `Especie_Nombre científico` | Representativo | Nombre científico de la especie estudiada. Puede requerir una dimensión si existen varias especies por publicación. |
| 36 | `ANIFFS: Eje Temático` | Representativo | Eje temático de la fila representante. Para todos los ejes usar `DIM_EJES_TEMATICOS`. |
| 37 | `ANIFFS: Área Temática` | Representativo | Área temática de la fila representante. Para todas las áreas usar `DIM_AREAS_TEMATICAS`. |
| 38 | `ANIFFS: Linea de investigación` | Representativo | Línea de investigación de la fila representante. Para todas las líneas usar `DIM_LINEAS_INVESTIGACION`. |
| 39 | `Categoria_Tesis_Articulo` | Derivado técnico | Clasificación técnica histórica entre tesis y artículo. Se conserva para auditoría. |
| 40 | `ID_REGISTRO_ANALISIS` | Identidad | Identificador del registro fuente materializado en la fila principal; normalmente corresponde al representante. |
| 41 | `CLAVE_BIBLIOGRAFICA_MASTER` | Identidad | Clave normalizada empleada para agrupar o identificar bibliográficamente registros. |
| 42 | `USAR_PARA_CONTEO_UNICO` | Derivado técnico | Bandera para conteo único. En la base consolidada debe ser `SI`. |
| 43 | `TIPO_PUBLICACION_NORM` | Derivado técnico | Tipo documental normalizado antes de la presentación pública. |
| 44 | `DOI_NORM` | Derivado técnico | DOI limpiado y normalizado para comparación y deduplicación. |
| 45 | `REGION_NORM_SUGERIDA` | Representativo normalizado | Región normalizada de la fila representativa. No es el conjunto completo de regiones de una publicación multirregional. |
| 46 | `URL_PRINCIPAL_DETECTADA` | Derivado técnico | URL priorizada o detectada como enlace principal. |
| 47 | `Nombre de Base de datos` | Consolidado | Bases documentales de procedencia reunidas para la publicación. La relación estructurada está en `DIM_BASES_DOCUMENTALES`. |
| 48 | `TIPO_PUBLICACION_PUBLICO` | Público/derivado | Categoría pública superior utilizada por filtros y gráficos: principalmente Artículo o Tesis. |
| 49 | `SUBTIPO_PUBLICACION_PUBLICO` | Público/derivado | Subcategoría documental pública dependiente del tipo. |
| 50 | `HUELLA_PUBLICACION_PERSISTENTE` | Identidad | Huella estable para conservar identidad entre regeneraciones de la base. |
| 51 | `METODO_HUELLA` | Identidad | Método usado para construir la huella persistente, por ejemplo DOI o atributos bibliográficos. |
| 52 | `GRADO_ACADEMICO_PUBLICO` | Público/consolidado | Grado general de tesis derivado de los registros del grupo, por ejemplo pregrado o posgrado. |
| 53 | `NIVEL_ACADEMICO_PUBLICO` | Público/consolidado | Nivel académico específico derivado, por ejemplo maestría o doctorado. |
| 54 | `DETALLE_TESIS_POSGRADO_PUBLICO` | Público/consolidado | Tercer nivel de la jerarquía de tesis de posgrado: maestría, doctorado o no identificado. |

## Dimensiones relacionadas

| Hoja | Significado | Clave |
|---|---|---|
| `REGISTROS_ORIGEN` | Todos los registros homologados antes de deduplicar | `ID_REGISTRO_ANALISIS` |
| `REGISTRO_PUBLICACION` | Correspondencia entre registros fuente y publicaciones | `ID_REGISTRO_ANALISIS` |
| `DIM_REPOSITORIOS` | Todos los repositorios asociados | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_AREAS_TEMATICAS` | Todas las áreas temáticas | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_EJES_TEMATICOS` | Todos los ejes temáticos | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_LINEAS_INVESTIGACION` | Todas las líneas de investigación | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_REGIONES_NORMALIZADAS` | Todas las regiones de estudio normalizadas | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_INSTITUCIONES` | Todas las instituciones asociadas | `ID_PUBLICACION_PROPUESTA + categoria` |
| `DIM_BASES_DOCUMENTALES` | Todas las bases documentales de procedencia | `ID_PUBLICACION_PROPUESTA + categoria` |

## Regla de interpretación

Cuando `REGISTROS_EN_GRUPO > 1`, todo campo marcado como **Representativo** debe leerse como “valor mostrado por la fila representante”, no como “único valor confirmado para la publicación”. Para una descarga exhaustiva deben adjuntarse las dimensiones correspondientes.
