import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import esquema_datos


def test_carga_consultas_con_mapeo_y_preserva_columnas_extra(tmp_path):
    entrada = tmp_path / "tickets.csv"
    pd.DataFrame(
        [
            {
                "id": "T-1",
                "mensaje": " Necesito ayuda ",
                "idioma_origen": "es",
                "canal": "chat",
            }
        ]
    ).to_csv(entrada, index=False)

    salida = esquema_datos.cargar_consultas(
        entrada,
        {"consulta_id": "id", "texto": "mensaje", "idioma": "idioma_origen"},
    )

    assert salida.loc[0, "texto"] == "Necesito ayuda"
    assert salida.loc[0, "metadata"] == {"canal": "chat"}
    assert salida.loc[0, "respuesta_referencia"] is None


def test_carga_json_con_clave_conocimiento(tmp_path):
    entrada = tmp_path / "base.json"
    entrada.write_text(
        json.dumps(
            {
                "conocimiento": [
                    {"documento_id": "D-1", "contenido": "Política de cambios"}
                ]
            }
        ),
        encoding="utf-8",
    )

    salida = esquema_datos.cargar_conocimiento(entrada)

    assert salida.loc[0, "idioma"] == "es"
    assert salida.loc[0, "contenido"] == "Política de cambios"


def test_informa_fila_y_campo_requerido_ausente(tmp_path):
    entrada = tmp_path / "consultas.json"
    entrada.write_text('[{"consulta_id": "C-1"}]', encoding="utf-8")

    with pytest.raises(esquema_datos.ErrorValidacionDatos) as exc:
        esquema_datos.cargar_consultas(entrada)

    assert "fila 1" in str(exc.value)
    assert "texto" in str(exc.value)


def test_rechaza_identificadores_duplicados():
    registros = [
        {"consulta_id": "C-1", "texto": "Primera"},
        {"consulta_id": "C-1", "texto": "Segunda"},
    ]

    with pytest.raises(esquema_datos.ErrorValidacionDatos, match="duplicados"):
        esquema_datos.validar_consultas(registros)


def test_metadata_csv_debe_ser_json_valido(tmp_path):
    entrada = tmp_path / "consultas.csv"
    pd.DataFrame(
        [{"consulta_id": "C-1", "texto": "Consulta", "metadata": "no-json"}]
    ).to_csv(entrada, index=False)

    with pytest.raises(ValueError, match="JSON válido"):
        esquema_datos.cargar_consultas(entrada)


def test_adaptador_bitext_prefiere_texto_en_espanol():
    datos = pd.DataFrame(
        [
            {
                "consulta_id": "consulta-001",
                "instruction": "Where is my order?",
                "instruction_es": "¿Dónde está mi pedido?",
                "response": "Track it online.",
                "response_es": "Seguí el envío en línea.",
                "category": "ORDER",
                "intent": "track_order",
            }
        ]
    )

    salida = esquema_datos.adaptar_consultas_bitext(datos)

    assert salida.loc[0, "texto"] == "¿Dónde está mi pedido?"
    assert salida.loc[0, "respuesta_referencia"] == "Seguí el envío en línea."
    assert salida.loc[0, "intencion"] == "track_order"


def test_conversiones_mantienen_compatibilidad_con_pipeline():
    consultas = pd.DataFrame(
        [{"consulta_id": "C-1", "texto": "Consulta", "idioma": "es"}]
    )
    documentos = pd.DataFrame(
        [{"documento_id": "D-1", "contenido": "Respuesta", "idioma": "es"}]
    )

    evaluacion = esquema_datos.consultas_a_formato_pipeline(consultas)
    conocimiento = esquema_datos.conocimiento_a_formato_rag(documentos)

    assert evaluacion.loc[0, "instruction_es"] == "Consulta"
    assert pd.isna(evaluacion.loc[0, "response_es"])
    assert conocimiento.loc[0, "document_id"] == "D-1"
    assert conocimiento.loc[0, "texto_rag"] == "Respuesta"
