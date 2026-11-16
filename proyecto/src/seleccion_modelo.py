"""Guía descriptiva de selección basada en los resultados del experimento."""

from __future__ import annotations

import pandas as pd


ETIQUETAS_MODELO = {
    "gpt-4o-mini": "GPT-4o mini",
    "llama-3.1-8b": "LLaMA 3.1 8B",
    "mistral": "Mistral",
}

COLUMNAS_REQUERIDAS = {
    "modelo",
    "condicion",
    "bertscore_f1",
    "latencia_media_s",
    "costo_oficial_medio_usd",
}


def construir_guia_seleccion(resumen: pd.DataFrame) -> pd.DataFrame:
    """Construye recomendaciones trazables para cuatro prioridades operativas.

    La guía no calcula un ganador universal. Selecciona la mejor observación de
    la condición RAG para cada prioridad y conserva la métrica que fundamenta
    esa selección.
    """
    faltantes = COLUMNAS_REQUERIDAS - set(resumen.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para construir la guía: {sorted(faltantes)}")

    rag = resumen.loc[resumen["condicion"].eq("rag")].copy()
    modelos_esperados = set(ETIQUETAS_MODELO)
    if set(rag["modelo"]) != modelos_esperados or len(rag) != len(modelos_esperados):
        raise ValueError("La guía requiere una fila RAG por cada modelo evaluado.")
    if rag[list(COLUMNAS_REQUERIDAS - {"modelo", "condicion"})].isna().any().any():
        raise ValueError("Las métricas utilizadas por la guía no pueden ser nulas.")

    rag["costo_1000_usd"] = rag["costo_oficial_medio_usd"] * 1000
    abiertos = rag.loc[rag["modelo"].ne("gpt-4o-mini")]

    mejor_calidad = rag.loc[rag["bertscore_f1"].idxmax()]
    menor_costo = rag.loc[rag["costo_1000_usd"].idxmin()]
    menor_latencia = rag.loc[rag["latencia_media_s"].idxmin()]
    mejor_abierto = abiertos.loc[abiertos["bertscore_f1"].idxmax()]

    filas = [
        {
            "prioridad": "Mayor calidad semántica",
            "modelo_recomendado": ETIQUETAS_MODELO[mejor_calidad["modelo"]],
            "evidencia": f"BERTScore F1 con RAG: {mejor_calidad['bertscore_f1']:.4f}",
            "alcance": "Resultado medio sobre las 324 consultas del experimento.",
        },
        {
            "prioridad": "Menor costo estimado",
            "modelo_recomendado": ETIQUETAS_MODELO[menor_costo["modelo"]],
            "evidencia": f"Costo por 1.000 consultas: USD {menor_costo['costo_1000_usd']:.4f}",
            "alcance": "Compara la API con el escenario cloud definido en la metodología.",
        },
        {
            "prioridad": "Menor latencia media",
            "modelo_recomendado": ETIQUETAS_MODELO[menor_latencia["modelo"]],
            "evidencia": f"Latencia media con RAG: {menor_latencia['latencia_media_s']:.3f} s",
            "alcance": "Corresponde al equipo, la red y la sesión experimental documentados.",
        },
        {
            "prioridad": "Mayor calidad entre modelos abiertos",
            "modelo_recomendado": ETIQUETAS_MODELO[mejor_abierto["modelo"]],
            "evidencia": f"BERTScore F1 con RAG: {mejor_abierto['bertscore_f1']:.4f}",
            "alcance": "La decisión entre modelos abiertos también debe considerar latencia y despliegue.",
        },
    ]
    return pd.DataFrame(filas)
