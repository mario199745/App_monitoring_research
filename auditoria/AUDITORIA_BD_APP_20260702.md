# Auditoría de la base maestra BD_APP

Fecha de corte: 2026-07-02  
Archivo auditado: `BD_APP_FINAL_20260630_232930_DEDUPLICADA.xlsx`  
Hoja principal: `BD_APP`

## Alcance y modelo de datos

`BD_APP` contiene 6 217 publicaciones consolidadas y 54 campos. Su grano es una fila por `ID_PUBLICACION_PROPUESTA`. No contiene todos los valores fuente: conserva como fila base el `ID_REGISTRO_REPRESENTATIVO` de cada grupo deduplicado.

Los 6 574 registros originales permanecen en `REGISTROS_ORIGEN`; su correspondencia con las publicaciones consolidadas está en `REGISTRO_PUBLICACION`. Las variables que admiten varios valores se conservan en hojas de dimensión con clave `ID_PUBLICACION_PROPUESTA + categoria`.

Por ello, los campos escalares de `BD_APP` no deben interpretarse automáticamente como la totalidad de la evidencia disponible. En particular, `REGION_NORM_SUGERIDA` es el valor del registro representativo; la cobertura territorial completa está en `DIM_REGIONES_NORMALIZADAS` y `REGIONES_EXPANDIDAS`.

## Resultados principales

- La clave `ID_PUBLICACION_PROPUESTA` es completa y única: 6 217 de 6 217.
- `REGISTROS_EN_GRUPO` coincide con el número de filas de `REGISTRO_PUBLICACION` para todas las publicaciones.
- La asignación del registro representativo es internamente consistente.
- Existen 281 publicaciones asociadas con más de una región, hasta un máximo de 7.
- En 55 publicaciones, `REGION_NORM_SUGERIDA` no aparece entre las categorías de `DIM_REGIONES_NORMALIZADAS`. Esto impide tratarla como fuente territorial maestra sin una revisión específica.
- Exportar únicamente `BD_APP` ocasionaba pérdida aparente de información en las 281 publicaciones multirregionales.

Caso comprobado: `PUB-000423` consolida `REG_000425` (Ucayali) y `REG_006948` (Ica). La fila principal representa a `REG_006948`, pero la relación territorial conserva ambas regiones de estudio.

## Cardinalidad de dimensiones

| Dimensión | Relaciones | Publicaciones multivalor | Máximo por publicación |
|---|---:|---:|---:|
| Repositorios | 7 300 | 944 | 4 |
| Áreas temáticas | 7 771 | 1 339 | 4 |
| Ejes temáticos | 7 236 | 947 | 3 |
| Líneas de investigación | 8 980 | 2 006 | 8 |
| Regiones normalizadas | 6 567 | 281 | 7 |
| Instituciones | 6 394 | 174 | 3 |
| Bases documentales | 6 546 | 314 | 4 |

## Riesgos identificados

1. **Confusión entre representante y consolidado.** Los campos originales de `BD_APP` proceden de una sola fila representativa, aunque la publicación agrupe varios registros.
2. **Pérdida en descargas planas.** Un CSV con una fila por publicación necesita columnas agregadas para cada dimensión multivalor o archivos relacionales complementarios.
3. **Ambigüedad territorial.** `REGION_NORM_SUGERIDA` y la dimensión territorial no siempre coinciden. El primero no debe denominarse “ámbito completo”.
4. **Recuento inflado.** Las dimensiones deben contarse con `nunique(ID_PUBLICACION_PROPUESTA)`, nunca sumando filas relacionales.
5. **Uso indebido de campos institucionales.** `General_ Institución/Universidad` identifica afiliación o institución asociada; no debe emplearse para inferir el lugar de realización del estudio.

## Correcciones aplicadas a la aplicación

- El mapa territorial vuelve a utilizar `REGIONES_EXPANDIDAS`, que conserva todas las regiones de estudio después de la deduplicación.
- La descarga plana incorpora `REGIONES_ESTUDIO_CONSOLIDADAS`.
- La descarga Excel incorpora la hoja `Detalle_territorial`, con la trazabilidad de cada relación publicación–región.
- `REGION_NORM_SUGERIDA` se conserva por trazabilidad, pero no sustituye el ámbito territorial consolidado.

## Reglas obligatorias para futuras cargas

1. Declarar el grano de cada hoja y su clave primaria en `CONTRATO_DATOS`.
2. Etiquetar los campos de `BD_APP` como `representativo`, `derivado consolidado` o `identificador`.
3. Para toda variable multivalor, mantener una dimensión y generar una columna consolidada legible en las descargas planas.
4. Validar que toda categoría territorial procede de `Ubicación_Región de estudio`; queda prohibido derivarla de la institución o universidad.
5. Reportar como error cualquier región representativa ausente de la dimensión y revisar los 55 casos detectados.
6. Probar antes de publicar un caso de una sola categoría y otro multivalor por cada dimensión.
7. Incluir en las descargas Excel las dimensiones filtradas, no solamente `BD_APP`.
8. Conservar `REGISTROS_ORIGEN` y `REGISTRO_PUBLICACION` para reproducir cada consolidación.

## Archivos complementarios

- `DICCIONARIO_DATOS_BD_APP.md`: significado y uso de los 54 campos.
- `auditoria_bd_app_estadisticas.csv`: completitud, cardinalidad y valores frecuentes por campo.
- `auditoria_bd_app_controles.csv`: controles y resultados.
- `auditoria_bd_app_dimensiones.csv`: cardinalidad de relaciones multivalor.
- `auditoria_bd_app_multirregion.csv`: muestra de publicaciones multirregionales.
