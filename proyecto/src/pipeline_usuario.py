"""Integración del esquema v1 con el pipeline experimental."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .esquema_datos import (
    conocimiento_a_formato_rag,
    consultas_a_formato_pipeline,
)
from .experimento import CONDICIONES


def preparar_entradas(
    consultas: pd.DataFrame,
    conocimiento: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convierte estructuras canónicas a los alias internos del pipeline."""
    evaluacion = consultas_a_formato_pipeline(consultas)
    documentos = (
        conocimiento_a_formato_rag(conocimiento)
        if conocimiento is not None
        else None
    )
    return evaluacion, documentos


def validar_configuracion(
    condiciones: Iterable[str],
    documentos: pd.DataFrame | None,
    top_k: int,
) -> tuple[str, ...]:
    """Valida una corrida y devuelve las condiciones materializadas."""
    condiciones = tuple(condiciones)
    if not condiciones or len(condiciones) != len(set(condiciones)):
        raise ValueError("Debe indicarse una lista no vacía de condiciones únicas.")
    desconocidas = set(condiciones) - set(CONDICIONES)
    if desconocidas:
        raise ValueError(f"Condiciones desconocidas: {sorted(desconocidas)}")
    if "rag" in condiciones and (documentos is None or documentos.empty):
        raise ValueError(
            "La condición RAG requiere un archivo de conocimiento con documentos."
        )
    if top_k < 1:
        raise ValueError("top_k debe ser al menos 1.")
    return condiciones


def capacidades_evaluacion(evaluacion: pd.DataFrame) -> dict[str, bool]:
    """Informa qué métricas admite el conjunto sin impedir su ejecución."""
    referencias = evaluacion["response_es"].notna() & evaluacion[
        "response_es"
    ].astype(str).str.strip().ne("")
    intenciones = evaluacion["intent"].notna() & evaluacion["intent"].astype(
        str
    ).str.strip().ne("")
    return {
        "metricas_calidad_automaticas": bool(referencias.all()),
        "hit_por_intencion": bool(intenciones.all()),
    }


def resumir_preflight(
    evaluacion: pd.DataFrame,
    documentos: pd.DataFrame | None,
    modelos: Iterable[str],
    condiciones: Iterable[str],
) -> dict[str, Any]:
    """Construye un resumen serializable antes de invocar modelos."""
    modelos = tuple(modelos)
    condiciones = tuple(condiciones)
    capacidades = capacidades_evaluacion(evaluacion)
    return {
        "consultas": len(evaluacion),
        "documentos_conocimiento": 0 if documentos is None else len(documentos),
        "modelos": list(modelos),
        "condiciones": list(condiciones),
        "respuestas_planificadas": len(evaluacion) * len(modelos) * len(condiciones),
        **capacidades,
    }


def metricas_recuperacion(
    documentos: list[dict[str, Any]],
    intencion_esperada: Any,
) -> tuple[bool | None, bool | None]:
    """Calcula Hit@1/Hit@k o devuelve ausencia explícita de métrica."""
    if pd.isna(intencion_esperada) or not str(intencion_esperada).strip():
        return None, None
    intenciones = [
        documento.get("intent")
        for documento in documentos
        if documento.get("intent") is not None
        and not pd.isna(documento.get("intent"))
    ]
    if not intenciones:
        return None, None
    esperada = str(intencion_esperada)
    return str(intenciones[0]) == esperada, esperada in map(str, intenciones)


def validar_checkpoint(
    filas: list[dict[str, Any]],
    plan: pd.DataFrame,
    evaluacion: pd.DataFrame,
) -> None:
    """Impide reanudar resultados que pertenecen a otra entrada o plan."""
    if not filas:
        return
    checkpoint = pd.DataFrame(filas)
    requeridas = {"modelo", "consulta_id", "condicion", "consulta"}
    faltantes = requeridas - set(checkpoint.columns)
    if faltantes:
        raise RuntimeError(
            f"El checkpoint no contiene las columnas requeridas: {sorted(faltantes)}"
        )
    claves = ["modelo", "consulta_id", "condicion"]
    if checkpoint.duplicated(claves).any():
        raise RuntimeError("El checkpoint contiene claves experimentales duplicadas.")

    claves_plan = {
        tuple(map(str, fila))
        for fila in plan[claves].itertuples(index=False, name=None)
    }
    claves_checkpoint = {
        tuple(map(str, fila))
        for fila in checkpoint[claves].itertuples(index=False, name=None)
    }
    ajenas = claves_checkpoint - claves_plan
    if ajenas:
        raise RuntimeError(
            "El checkpoint contiene unidades que no pertenecen al plan actual: "
            f"{sorted(ajenas)[:3]}"
        )

    textos = evaluacion.set_index("consulta_id")["texto"].astype(str).to_dict()
    for fila in checkpoint.itertuples(index=False):
        consulta_id = str(fila.consulta_id)
        if consulta_id not in textos or str(fila.consulta) != textos[consulta_id]:
            raise RuntimeError(
                "El checkpoint no corresponde al contenido actual de la consulta "
                f"{consulta_id}. Use otra ruta de salida."
            )
