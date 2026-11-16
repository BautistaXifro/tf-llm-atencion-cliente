import json
from pathlib import Path


PROYECTO = Path(__file__).resolve().parents[1]
NOTEBOOK = PROYECTO / "notebooks" / "04_evaluacion_completa.ipynb"


def _fuente_completa() -> str:
    contenido = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(celda.get("source", [])) for celda in contenido["cells"])


def test_notebook_integral_existe_y_es_valido():
    contenido = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert contenido["nbformat"] == 4
    assert len(contenido["cells"]) >= 20
    assert contenido["metadata"]["kernelspec"]["name"] == "python3"


def test_notebook_abre_en_modo_seguro():
    fuente = _fuente_completa()
    assert 'MODO = "resultados"' in fuente
    assert "RECONSTRUIR_MUESTRA = False" in fuente
    assert "TRADUCIR_MUESTRA = False" in fuente
    assert "RECALCULAR_METRICAS = False" in fuente
    assert "REGENERAR_FIGURAS = False" in fuente
    assert 'CONFIRMAR_CORRIDA_COMPLETA = ""' in fuente


def test_notebook_cubre_el_flujo_final():
    fuente = _fuente_completa()
    for texto in (
        "Auditoría de la muestra",
        "Recuperación RAG",
        "Integridad de la corrida",
        "Métricas automáticas",
        "Contraste de hipótesis",
        "Gráficos del experimento",
        "Guía de selección",
        "Evaluación humana",
    ):
        assert texto in fuente


def test_notebook_espera_los_archivos_generados_por_el_analizador_humano():
    fuente = _fuente_completa()
    assert "ARCHIVO_RESUMEN" in fuente
    assert "ARCHIVO_ACUERDO" in fuente
