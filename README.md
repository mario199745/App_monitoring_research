# Revisión bibliográfica DEI

Aplicación Streamlit para explorar publicaciones únicas relacionadas con
recursos forestales, biodiversidad y fauna silvestre.

## Datos

El aplicativo utiliza el par más reciente encontrado en `data`:

- `BD_APP_FINAL_*.xlsx`
- `BD_APP_TERRITORIAL_*.xlsx`

La base principal incluye dimensiones expandidas para repositorios, áreas,
ejes, líneas, regiones e instituciones.

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

La trazabilidad técnica se encuentra en
`docs/TRAZABILIDAD_APLICATIVO.md`.
