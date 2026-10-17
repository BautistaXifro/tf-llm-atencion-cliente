import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import experimento, pipeline_usuario


def _consultas(con_referencia=True, con_intencion=True):
    return pd.DataFrame(
        [
            {
                "consulta_id": "C-1",
                "texto": "¿Dónde está mi pedido?",
                "respuesta_referencia": "Revisá el seguimiento." if con_referencia else None,
                "idioma": "es",
                "categoria": "PEDIDOS",
                "intencion": "seguir_pedido" if con_intencion else None,
                "metadata": {},
            }
        ]
    )


def _documentos(con_intencion=True):
    return pd.DataFrame(
        [
            {
                "documento_id": "D-1",
                "contenido": "Use el número de seguimiento.",
                "idioma": "es",
                "categoria": "PEDIDOS",
                "intencion": "seguir_pedido" if con_intencion else None,
                "fuente": "manual interno",
                "metadata": {},
            }
        ]
    )


def test_preparar_entradas_conecta_esquema_con_alias_del_pipeline():
    evaluacion, documentos = pipeline_usuario.preparar_entradas(
        _consultas(), _documentos()
    )

    assert evaluacion.loc[0, "instruction_es"] == "¿Dónde está mi pedido?"
    assert evaluacion.loc[0, "response_es"] == "Revisá el seguimiento."
    assert documentos.loc[0, "document_id"] == "D-1"
    assert documentos.loc[0, "texto_rag"] == "Use el número de seguimiento."


def test_configuracion_rag_requiere_conocimiento():
    with pytest.raises(ValueError, match="requiere un archivo de conocimiento"):
        pipeline_usuario.validar_configuracion(["rag"], None, top_k=3)


def test_preflight_degrada_metricas_si_faltan_campos_opcionales():
    evaluacion, _ = pipeline_usuario.preparar_entradas(
        _consultas(con_referencia=False, con_intencion=False)
    )
    resumen = pipeline_usuario.resumir_preflight(
        evaluacion, None, ["mistral"], ["sin_rag"]
    )

    assert resumen["respuestas_planificadas"] == 1
    assert resumen["metricas_calidad_automaticas"] is False
    assert resumen["hit_por_intencion"] is False


def test_metricas_recuperacion_son_explicitas_con_y_sin_intencion():
    docs = [
        {"document_id": "D-1", "intent": "seguir_pedido"},
        {"document_id": "D-2", "intent": "cancelar_pedido"},
    ]
    assert pipeline_usuario.metricas_recuperacion(docs, "seguir_pedido") == (
        True,
        True,
    )
    assert pipeline_usuario.metricas_recuperacion(docs, None) == (None, None)


def test_plan_acepta_datos_sin_referencia_y_una_sola_condicion():
    evaluacion, _ = pipeline_usuario.preparar_entradas(
        _consultas(con_referencia=False)
    )
    plan = experimento.construir_plan(
        evaluacion,
        ["mistral"],
        condiciones=["sin_rag"],
    )

    assert len(plan) == 1
    assert plan.loc[0, "condicion"] == "sin_rag"


def test_checkpoint_rechaza_misma_clave_con_otra_consulta():
    evaluacion, _ = pipeline_usuario.preparar_entradas(_consultas())
    plan = experimento.construir_plan(
        evaluacion,
        ["mistral"],
        condiciones=["sin_rag"],
    )
    filas = [
        {
            "modelo": "mistral",
            "consulta_id": "C-1",
            "condicion": "sin_rag",
            "consulta": "Contenido anterior",
        }
    ]

    with pytest.raises(RuntimeError, match="no corresponde al contenido actual"):
        pipeline_usuario.validar_checkpoint(filas, plan, evaluacion)
