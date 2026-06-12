# Trazabilidad del aplicativo

## Propósito

Documentar la adaptación de la base homologada, los cambios funcionales del
aplicativo y los mecanismos para reproducir y validar el resultado.

## Fuentes

- Base homologada: última `BASE_HOMOLOGADA_42_CAMPOS.xlsx` generada por el
  Notebook del proyecto.
- Base territorial: última
  `BD_APP_TERRITORIAL_*_PUBLICA_UNICA.xlsx` disponible en `DATOS`.
- Geometría: `data/GEO/DEP_PERU.geojson`.

## Base adaptada

El script `tools/adapt_homologated_database.py` genera:

- `data/BD_APP_FINAL_<fecha>_HOMOLOGADA.xlsx`
- `data/BD_APP_TERRITORIAL_<fecha>_HOMOLOGADA.xlsx`
- `data/TRAZABILIDAD_BASE_ADAPTADA.md`

La base principal conserva la hoja `BD_APP` con 42 campos para mantener
compatibilidad y trazabilidad. También incorpora:

- `DIM_REPOSITORIOS`
- `DIM_AREAS_TEMATICAS`
- `DIM_EJES_TEMATICOS`
- `DIM_LINEAS_INVESTIGACION`
- `DIM_REGIONES_NORMALIZADAS`
- `DIM_INSTITUCIONES`
- `RESUMEN_ADAPTACION`
- `CONTRATO_DATOS`

Cada dimensión contiene una relación única entre publicación y categoría. Esto
evita representar una combinación como `RENATI, UNALM` o
`Cambio Climático, Manejo...` como una categoría independiente.

## Cambios funcionales

Se conservaron las secciones:

1. `General`
2. `Territorio`
3. `Tiempo`
4. `Temas`
5. `Datos`

Las principales optimizaciones son:

- conteos mediante publicaciones únicas;
- filtros por categorías expandidas;
- áreas, ejes, líneas y repositorios no excluyentes;
- mapa relacionado con los identificadores filtrados;
- una sola versión preferida de DOI, URL, región y tipo;
- catálogo conservador de instituciones que excluye valores con apariencia de revista;
- ejes reducidos a ocho categorías oficiales y líneas normalizadas por código;
- exclusión opcional de `Otros` en gráficos;
- descargas completas de los registros filtrados;
- visualización de la fuente de datos utilizada.

## Validación

El script `tools/validate_app_data.py` comprueba:

- hojas obligatorias;
- columnas principales;
- unicidad de `ID_REGISTRO_ANALISIS`;
- ausencia de relaciones duplicadas;
- integridad referencial de dimensiones y base territorial.

## Procedimiento de actualización

```powershell
& C:\Users\USUARIO\miniconda3\envs\sigexpert\python.exe `
  .\tools\adapt_homologated_database.py

& C:\Users\USUARIO\miniconda3\envs\sigexpert\python.exe `
  .\tools\validate_app_data.py

& C:\Users\USUARIO\miniconda3\Scripts\conda.exe run -n sigexpert `
  streamlit run app.py
```

## Historial

| Fecha | Cambio | Resultado |
| --- | --- | --- |
| 2026-06-12 | Adaptación de la base homologada | Base principal de 6,574 registros y 42 campos, más seis dimensiones relacionales. |
| 2026-06-12 | Optimización de `app.py` | Se conservaron las cinco secciones y se corrigieron filtros y conteos multivaluados. |
| 2026-06-12 | Incorporación de validación | Se añadió control de contrato e integridad referencial. |
| 2026-06-12 | Catálogos analíticos | Se consolidaron ocho ejes oficiales, 19 áreas oficiales, 96 códigos de línea y 209 instituciones conservadoras; `Otros` permanece como categoría adicional. |
| 2026-06-12 | Prueba funcional | La carga inicial mostró 6,574 publicaciones y el filtro `Cambio Climático` redujo correctamente el universo a 375 publicaciones. |
