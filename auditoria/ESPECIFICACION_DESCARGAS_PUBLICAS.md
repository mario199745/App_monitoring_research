# Especificación de descargas públicas

Fecha: 2026-07-02

## Objetivo

Las descargas del dashboard presentan información útil para consulta y análisis. No exponen campos creados exclusivamente para deduplicación, validación, coincidencia, auditoría o administración interna.

El filtro activo determina las publicaciones incluidas. Tanto el CSV como la hoja `Datos_filtrados` del Excel contienen una fila por `ID_PUBLICACION_PROPUESTA`.

## Campos incluidos

### Identificación y bibliografía

- `ID_PUBLICACION_PROPUESTA`
- `General_ Título`
- `General_ Autor(es)`
- `General_ Año`
- `General_ Resumen`
- `General_ Palabras clave del documento`
- `General_ Idioma`
- `General_ DOI`
- `General_ Enlace`

### Clasificación documental

- `TIPO_PUBLICACION_PUBLICO`
- `SUBTIPO_PUBLICACION_PUBLICO`
- `GRADO_ACADEMICO_PUBLICO`
- `NIVEL_ACADEMICO_PUBLICO`
- `DETALLE_TESIS_POSGRADO_PUBLICO`
- `General_ Publicación Nacional/Extranjera`
- `General_ Tipo de contenido (TD=texto disponible, TN=Texto no disponible)`

### Información editorial

- `General_ Nombre de revista`, volumen, número, páginas y lugar de publicación.

### Procedencia e instituciones

- `BASES_DOCUMENTALES_CONSOLIDADAS`
- `REPOSITORIOS_CONSOLIDADOS`
- `INSTITUCIONES_CONSOLIDADAS`
- `General_ SIGLAS UNIVERSIDAD/INSTITUCIÓN`

### Ámbito y temática

- `REGIONES_ESTUDIO_CONSOLIDADAS`
- Ámbito o ecosistema, localidad, distrito y provincia.
- `Especie_Nombre científico`
- `AREAS_TEMATICAS_CONSOLIDADAS`
- `EJES_TEMATICOS_CONSOLIDADOS`
- `LINEAS_INVESTIGACION_CONSOLIDADAS`

## Campos excluidos

Se excluyen `ID_REGISTRO_REPRESENTATIVO`, `REGISTROS_EN_GRUPO`, `REGLAS_AGRUPACION`, `ESTADO`, `ID_GLOBAL`, `COD BD SERFOR`, `ID_REGISTRO_ANALISIS`, `CLAVE_BIBLIOGRAFICA_MASTER`, `USAR_PARA_CONTEO_UNICO`, `Categoria_Tesis_Articulo`, `TIPO_PUBLICACION_NORM`, `DOI_NORM`, `REGION_NORM_SUGERIDA`, `URL_PRINCIPAL_DETECTADA`, `HUELLA_PUBLICACION_PERSISTENTE` y `METODO_HUELLA`.

También se excluye cualquier campo futuro cuyo propósito sea indicar coincidencia, confianza, revisión, regla aplicada, control interno o trazabilidad técnica.

## Valores múltiples

Las bases, repositorios, instituciones, regiones, áreas, ejes y líneas se obtienen desde sus relaciones completas. Los valores únicos se ordenan y concatenan con `;` sin duplicar la publicación.

## Contenido del Excel

- `Datos_filtrados`: vista pública de publicaciones.
- `Resumen_tipo`: conteo por tipo.
- `Resumen_region`: conteo territorial.
- `Detalle_territorial`: solo `ID_PUBLICACION_PROPUESTA` y `REGION_ESTUDIO`.

## Controles obligatorios

1. No incluir columnas técnicas excluidas.
2. Mantener único `ID_PUBLICACION_PROPUESTA` en la tabla principal.
3. Derivar las columnas consolidadas de las dimensiones filtradas.
4. No duplicar relaciones publicación–región.
5. Aplicar los filtros antes de construir la exportación.
