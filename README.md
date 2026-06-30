# Revisión bibliográfica DEI

Aplicación Streamlit para explorar publicaciones consolidadas relacionadas con
recursos forestales, biodiversidad y fauna silvestre.

## Datos

El aplicativo utiliza el par más reciente encontrado en `data`:

- `BD_APP_FINAL_*.xlsx`
- `BD_APP_TERRITORIAL_*.xlsx`

La base principal contiene una fila por `ID_PUBLICACION_PROPUESTA` e incluye
dimensiones expandidas para bases documentales, repositorios, áreas, ejes,
líneas, regiones e instituciones.

`DIM_REPOSITORIOS` conserva la clasificación técnica en `CLASE_REPOSITORIO` y
presenta una clasificación simplificada en `CLASE_REPOSITORIO_PUBLICA`. La
interfaz utiliza cinco grupos: `Buscadores académicos`, `Repositorios
institucionales`, `Repositorios universitarios`, `Revistas` y `Otros`. La vista
presenta la distribución agregada por clase, sin rankings desplegables. Los
casos no identificados permanecen visibles en `Otros` y se marcan en
`REQUIERE_REVISION_REPOSITORIO`.

`DIM_INSTITUCIONES` clasifica cada entidad con `CLASE_INSTITUCION` y
`ES_UNIVERSIDAD`. La pestaña `Instituciones` muestra primero la distribución
por clase y luego rankings desplegables para universidades, entidades públicas,
centros de investigación, sociedades científicas y revistas o boletines mal
ubicados.

Los valores con apariencia de revista o boletín detectados en
`General_ Institución/Universidad` se migran a `General_ Nombre de revista`.
La hoja `AUDITORIA_INSTITUCIONES` conserva el valor original, la revista final
y la institución final asignada.

La interfaz presenta `Otros` por defecto, utiliza nombres documentales sin
prefijos numéricos y cuenta **6,217 publicaciones consolidadas**. Los 6,574
registros bibliográficos de origen permanecen disponibles para auditoría.

Al seleccionar `Tesis`, se habilitan filtros públicos separados para
`Grado académico` y `Nivel académico`. La categoría técnica `No aplica` no se
muestra en estos filtros.

El tipo público se organiza en `Artículo` y `Tesis`. Al seleccionar
`Artículo`, se habilita el subtipo `Artículo científico` o
`Artículo de conferencia`.

Los códigos `ID_PUBLICACION_PROPUESTA` son persistentes. El registro maestro
ubicado en `../NOTEBOOK/registro_maestro_publicaciones.csv` permite conservar
los identificadores existentes y asignar códigos nuevos de forma incremental.

La disponibilidad del contenido se conserva en la base para auditoría, pero no
se presenta como filtro público porque no equivale necesariamente a acceso
abierto.

Los indicadores y ejes cuantitativos se presentan como `N° de ...`. El mapa
territorial también funciona como filtro: permite seleccionar uno o varios
departamentos, sincroniza la selección con `Región normalizada` y actualiza
indicadores, gráficos, tabla y descargas.

Los millares visibles se separan mediante espacio, por ejemplo `6 217`. La
cabecera superior no muestra la cantidad de bases documentales; esa
distribución permanece disponible en la vista general.

La navegación incluye una pestaña específica de `Instituciones`, separada de
la sección temática, para consultar publicaciones por institución o
universidad.

## Preparación

```powershell
pip install -r requirements.txt
python tools/adapt_homologated_database.py
python tools/validate_app_data.py
```

## Ejecución

```powershell
streamlit run app.py
```

La trazabilidad técnica se encuentra en:

- `../NOTEBOOK/docs/TRAZABILIDAD_APLICATIVO.md`
- `../NOTEBOOK/docs/TRAZABILIDAD_BASE_ADAPTADA.md`
