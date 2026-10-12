import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import muestra


def _datos_fake(estratos=3, filas_por_estrato=5):
    filas = []
    for estrato in range(estratos):
        for fila in range(filas_por_estrato):
            filas.append(
                {
                    "intent": f"intent_{estrato}",
                    "instruction": f"pregunta {estrato}-{fila}",
                    "response": f"respuesta {estrato}-{fila}",
                }
            )
    return pd.DataFrame(filas)


def test_muestra_balanceada_cubre_todos_los_estratos():
    salida, auditoria = muestra.construir_muestra_balanceada(
        _datos_fake(), 3, semilla=42
    )
    assert len(salida) == 9
    assert salida["intent"].value_counts().eq(3).all()
    assert auditoria["estratos"] == 3
    assert auditoria["filas_seleccionadas"] == 9


def test_muestra_balanceada_es_reproducible():
    datos = _datos_fake()
    a, _ = muestra.construir_muestra_balanceada(datos, 2, semilla=42)
    b, _ = muestra.construir_muestra_balanceada(datos, 2, semilla=42)
    assert a["source_row_id"].tolist() == b["source_row_id"].tolist()
    assert a["consulta_id"].tolist() == b["consulta_id"].tolist()


def test_muestra_elimina_duplicados_normalizados_y_vacios():
    datos = _datos_fake(estratos=2, filas_por_estrato=3)
    extras = pd.DataFrame(
        [
            {"intent": "intent_0", "instruction": "Pregunta 0-0!", "response": "x"},
            {"intent": "intent_0", "instruction": "   ", "response": "x"},
        ]
    )
    salida, auditoria = muestra.construir_muestra_balanceada(
        pd.concat([datos, extras], ignore_index=True), 2, semilla=42
    )
    assert len(salida) == 4
    assert auditoria["duplicados_eliminados"] == 1
    assert auditoria["filas_vacias_eliminadas"] == 1


def test_muestra_falla_si_un_estrato_no_alcanza():
    datos = _datos_fake(estratos=2, filas_por_estrato=2)
    try:
        muestra.construir_muestra_balanceada(datos, 3)
        assert False, "Debía fallar por falta de filas"
    except ValueError as exc:
        assert "No hay suficientes" in str(exc)


def test_submuestra_humana_preserva_ids_y_balance():
    ampliada, _ = muestra.construir_muestra_balanceada(
        _datos_fake(estratos=3, filas_por_estrato=5), 4, semilla=42
    )
    humana = muestra.seleccionar_submuestra_humana(
        ampliada, 2, semilla=43
    )
    assert len(humana) == 6
    assert humana["intent"].value_counts().eq(2).all()
    assert set(humana["consulta_id"]).issubset(set(ampliada["consulta_id"]))
