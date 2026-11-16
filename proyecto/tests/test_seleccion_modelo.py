import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.seleccion_modelo import construir_guia_seleccion


def _resumen() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "modelo": "gpt-4o-mini",
                "condicion": "rag",
                "bertscore_f1": 0.75,
                "latencia_media_s": 1.50,
                "costo_oficial_medio_usd": 0.00016,
            },
            {
                "modelo": "llama-3.1-8b",
                "condicion": "rag",
                "bertscore_f1": 0.73,
                "latencia_media_s": 1.10,
                "costo_oficial_medio_usd": 0.00021,
            },
            {
                "modelo": "mistral",
                "condicion": "rag",
                "bertscore_f1": 0.74,
                "latencia_media_s": 1.58,
                "costo_oficial_medio_usd": 0.00030,
            },
        ]
    )


def test_guia_asigna_un_modelo_por_prioridad():
    guia = construir_guia_seleccion(_resumen()).set_index("prioridad")

    assert guia.loc["Mayor calidad semántica", "modelo_recomendado"] == "GPT-4o mini"
    assert guia.loc["Menor costo estimado", "modelo_recomendado"] == "GPT-4o mini"
    assert guia.loc["Menor latencia media", "modelo_recomendado"] == "LLaMA 3.1 8B"
    assert guia.loc[
        "Mayor calidad entre modelos abiertos", "modelo_recomendado"
    ] == "Mistral"


def test_guia_rechaza_metricas_faltantes():
    with pytest.raises(ValueError, match="Faltan columnas"):
        construir_guia_seleccion(_resumen().drop(columns="bertscore_f1"))


def test_guia_requiere_una_fila_rag_por_modelo():
    duplicado = pd.concat([_resumen(), _resumen().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="una fila RAG"):
        construir_guia_seleccion(duplicado)
