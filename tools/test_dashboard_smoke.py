from streamlit.testing.v1 import AppTest


def assert_clean(app: AppTest, stage: str) -> None:
    if app.exception:
        messages = [str(item.value) for item in app.exception]
        raise AssertionError(f"{stage}: {messages}")


app = AppTest.from_file("app.py", default_timeout=180).run()
assert_clean(app, "carga inicial")

assert len(app.metric) == 4, "Se esperaban cuatro indicadores principales."
assert any("MONITOREO DE INVESTIGACION" in item.value for item in app.markdown)

type_filter = next(
    widget
    for widget in app.sidebar.multiselect
    if widget.label == "Tipo de publicación"
)
type_filter.set_value(["Tesis"])
app.run()
assert_clean(app, "filtro Tesis")

app.session_state["_publication_chart_pending"] = {
    "action": "type",
    "value": "Tesis",
}
app.run()
assert_clean(app, "navegación a subcategorías de Tesis")
assert app.session_state["_publication_chart_level"] == "subtypes"
assert any("Tipo seleccionado: Tesis" in item.value for item in app.caption)

region_filter = next(
    widget
    for widget in app.sidebar.multiselect
    if widget.label == "Región normalizada"
)
assert "Ucayali" in region_filter.options
region_filter.set_value(["Ucayali"])
app.run()
assert_clean(app, "filtro territorial Ucayali")
assert app.metric[0].value not in {"0", "0.0"}

app.session_state["relation_filter_region"] = []
app.session_state["filter_TIPO_PUBLICACION_PUBLICO"] = []
app.session_state["filter_SUBTIPO_PUBLICACION_PUBLICO"] = []
app.session_state["_publication_chart_pending"] = {"action": "back"}
app.run()
assert_clean(app, "limpieza de filtros")

print("DASHBOARD_SMOKE=OK")
print(f"METRICS={[(item.label, item.value) for item in app.metric]}")
print(f"SIDEBAR_FILTERS={[item.label for item in app.sidebar.multiselect]}")
