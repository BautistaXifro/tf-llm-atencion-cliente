import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import experimento


def _evaluacion(n=4):
    return pd.DataFrame(
        [
            {
                "consulta_id": f"q-{i}",
                "instruction_es": f"consulta {i}",
                "response_es": f"respuesta {i}",
            }
            for i in range(n)
        ]
    )


def test_plan_es_reproducible_pareado_y_balanceado():
    a = experimento.construir_plan(_evaluacion(), ["modelo-a", "modelo-b"])
    b = experimento.construir_plan(_evaluacion(), ["modelo-a", "modelo-b"])

    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 16
    assert not a.duplicated(["modelo", "consulta_id", "condicion"]).any()
    for _, bloque in a.groupby("modelo"):
        primeras = bloque.loc[bloque["orden_condicion"] == 1, "condicion"]
        assert primeras.value_counts().to_dict() == {"sin_rag": 2, "rag": 2}
        assert (bloque.groupby("consulta_id").size() == 2).all()


def test_limitar_plan_conserva_ambas_condiciones_por_modelo():
    plan = experimento.construir_plan(_evaluacion(), ["a", "b"])
    limitado = experimento.limitar_plan(plan, 1)

    assert len(limitado) == 4
    assert (limitado.groupby(["modelo", "consulta_id"]).size() == 2).all()


def test_checkpoint_reintenta_error_y_reemplaza_la_misma_clave():
    error = {
        "modelo": "a",
        "consulta_id": "q-1",
        "condicion": "rag",
        "respuesta": None,
        "error": "timeout",
    }
    assert experimento.claves_completadas([error]) == set()

    correcto = {**error, "respuesta": "respuesta", "error": None}
    resultados = experimento.reemplazar_resultado([error], correcto)

    assert resultados == [correcto]
    assert experimento.claves_completadas(resultados) == {("a", "q-1", "rag")}
