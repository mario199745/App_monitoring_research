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
