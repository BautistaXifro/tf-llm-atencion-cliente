import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluacion_humana import construir_insumos_ciegos


def test_insumos_ciegos_son_completos_y_reproducibles():
    consultas = [f"q{i:02d}" for i in range(54)]
    muestra = pd.DataFrame({"consulta_id": consultas})
    modelos = ["gpt-4o-mini", "llama-3.1-8b", "mistral"]
    filas = []
    for consulta_id in consultas:
        for modelo in modelos:
            for condicion in ("sin_rag", "rag"):
                filas.append(
                    {
                        "consulta_id": consulta_id,
                        "modelo": modelo,
                        "condicion": condicion,
                        "tipo": "propietario" if modelo == "gpt-4o-mini" else "open-source",
                        "category": "TEST",
                        "intent": "test_intent",
                        "consulta": f"Consulta {consulta_id}",
                        "referencia": "Referencia",
                        "respuesta": f"Respuesta {modelo} {condicion}",
                        "documentos_recuperados": '["d1"]' if condicion == "rag" else "[]",
                    }
                )
    resultados = pd.DataFrame(filas)
    corpus = pd.DataFrame({"document_id": ["d1"], "texto_rag": ["Contexto"]})

    formularios_a, clave_a = construir_insumos_ciegos(resultados, muestra, corpus)
    formularios_b, clave_b = construir_insumos_ciegos(resultados, muestra, corpus)

    assert all(len(formulario) == 324 for formulario in formularios_a.values())
    assert formularios_a["evaluador_1"]["respuesta_id"].is_unique
    assert not formularios_a["evaluador_1"]["respuesta_id"].equals(
        formularios_a["evaluador_2"]["respuesta_id"]
    )
    pd.testing.assert_frame_equal(
        formularios_a["evaluador_1"], formularios_b["evaluador_1"]
    )
    pd.testing.assert_frame_equal(clave_a, clave_b)
