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
