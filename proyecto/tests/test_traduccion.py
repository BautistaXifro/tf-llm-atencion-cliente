import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import traduccion


class _CompletionsFake:
    def __init__(self):
        self.llamadas = 0

    def create(self, **kwargs):
        self.llamadas += 1
        original = json.loads(kwargs["messages"][1]["content"])
        contenido = json.dumps(
            {
                "instruction_es": original["instruction"].replace(
                    "Track", "Seguí"
                ).replace("Order Number", "Número de Pedido"),
                "response_es": original["response"].replace(
                    "Open", "Abrí"
                ).replace("Order Number", "Número de Pedido"),
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=contenido))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30),
        )


class _ClienteFake:
    def __init__(self):
        self.completions = _CompletionsFake()
        self.chat = SimpleNamespace(completions=self.completions)


class _CompletionsFallbackFake:
    def __init__(self):
        self.llamadas = 0

    def create(self, **kwargs):
        self.llamadas += 1
        original = json.loads(kwargs["messages"][1]["content"])
        if "instruction" in original:
            contenido = json.dumps(
                {"instruction_es": "Seguí", "response_es": "Abrí"}
            )
        else:
            contenido = json.dumps(
                {
                    "instruction_segments_es": [
                        segmento.replace("Track", "Seguí")
                        for segmento in original["instruction_segments"]
                    ],
                    "response_segments_es": [
                        segmento.replace("Open", "Abrí")
                        for segmento in original["response_segments"]
                    ],
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=contenido))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30),
        )


class _ClienteFallbackFake:
    def __init__(self):
        self.completions = _CompletionsFallbackFake()
        self.chat = SimpleNamespace(completions=self.completions)


def _muestra():
    return pd.DataFrame(
        [
            {
                "consulta_id": "q-1",
                "instruction": "Track {{Order Number}}",
                "response": "Open {{Order Number}}",
            },
            {
                "consulta_id": "q-2",
                "instruction": "Already translated",
                "response": "Existing answer",
                "instruction_es": "Ya traducida",
                "response_es": "Respuesta existente",
                "tokens_traduccion_in": 4,
                "tokens_traduccion_out": 5,
            },
        ]
    )


def test_traducir_pendientes_reanuda_y_guarda_cada_fila():
    cliente = _ClienteFake()
    checkpoints = []
    salida = traduccion.traducir_pendientes(
        _muestra(),
        cliente,
        lambda df: checkpoints.append(df.copy()),
        "consulta_id",
        pausa_s=0,
        dormir=lambda _: None,
        informar=lambda _: None,
    )

    assert cliente.completions.llamadas == 1
    assert len(checkpoints) == 2
    assert salida.loc[0, "instruction_es"] == "Seguí {{Order Number}}"
    assert salida.loc[0, "response_es"] == "Abrí {{Order Number}}"
    assert salida.loc[1, "instruction_es"] == "Ya traducida"
    assert salida["tokens_traduccion_in"].tolist() == [20, 4]
    assert salida["intentos_traduccion"].tolist() == [1, 1]


def test_enmascarar_marcadores_distingue_repeticiones_y_las_restaura():
    original = "Usá {{Cuenta}} y luego cerrá {{Cuenta}}"
    enmascarado, equivalencias = traduccion.enmascarar_marcadores(original, "R")

    assert "{{Cuenta}}" not in enmascarado
    assert len(equivalencias) == 2
    assert len(set(equivalencias)) == 2
    assert traduccion.restaurar_tokens_marcadores(
        enmascarado, equivalencias
    ) == original


def test_separar_y_reconstruir_conserva_marcadores_repetidos():
    original = "Antes {{Cuenta}} durante {{Cuenta}} después"
    segmentos, marcadores = traduccion.separar_por_marcadores(original)

    assert segmentos == ["Antes ", " durante ", " después"]
    assert marcadores == ["{{Cuenta}}", "{{Cuenta}}"]
    assert traduccion.reconstruir_con_marcadores(
        ["Previo ", " mientras ", " luego"], marcadores
    ) == "Previo {{Cuenta}} mientras {{Cuenta}} luego"


def test_traduccion_reintenta_por_segmentos_si_el_modelo_omite_marcador():
    cliente = _ClienteFallbackFake()
    muestra = _muestra().iloc[[0]].copy()
    salida = traduccion.traducir_pendientes(
        muestra,
        cliente,
        lambda _: None,
        "consulta_id",
        pausa_s=0,
        max_reintentos=2,
        dormir=lambda _: None,
        informar=lambda _: None,
    )

    assert cliente.completions.llamadas == 2
    assert salida.loc[0, "instruction_es"] == "Seguí {{Order Number}}"
    assert salida.loc[0, "response_es"] == "Abrí {{Order Number}}"
    assert salida.loc[0, "intentos_traduccion"] == 2
    assert salida.loc[0, "salidas_traduccion_rechazadas"] == 1


def test_validar_checkpoint_rechaza_otra_muestra():
    entrada = traduccion.preparar_dataframe(_muestra(), "consulta_id")
    checkpoint = entrada.iloc[::-1].reset_index(drop=True)

    try:
        traduccion.validar_checkpoint(entrada, checkpoint, "consulta_id")
        assert False, "Debía rechazar identificadores en otro orden"
    except ValueError as exc:
        assert "no corresponde" in str(exc)


def test_validar_checkpoint_rechaza_texto_modificado_con_mismos_ids():
    entrada = traduccion.preparar_dataframe(_muestra(), "consulta_id")
    checkpoint = entrada.copy()
    checkpoint.loc[0, "instruction"] = "Texto modificado"

    try:
        traduccion.validar_checkpoint(entrada, checkpoint, "consulta_id")
        assert False, "Debía rechazar un texto original modificado"
    except ValueError as exc:
        assert "instruction" in str(exc)


def test_construir_auditoria_informa_completitud_y_costo():
    corpus = traduccion.preparar_dataframe(_muestra(), "consulta_id")
    auditoria = traduccion.construir_auditoria(corpus, "consulta_id")

    assert auditoria["filas_totales"] == 2
    assert auditoria["filas_traducidas"] == 1
    assert auditoria["filas_pendientes"] == 1
    assert auditoria["tokens_in"] == 4
    assert auditoria["tokens_out"] == 5
    assert auditoria["costo_total_usd"] == 0.000004
