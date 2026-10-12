"""Preparación reproducible de la evaluación humana ciega."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


SEMILLAS_EVALUADORES = {
    "evaluador_1": 20261021,
    "evaluador_2": 20261022,
}

COLUMNAS_FORMULARIO = [
    "orden_presentacion",
    "respuesta_id",
    "consulta_id",
    "consulta",
    "respuesta_referencia",
    "contexto_evaluacion",
    "respuesta_a_evaluar",
    "correccion_1_5",
    "utilidad_1_5",
    "claridad_1_5",
    "fidelidad_contexto_1_5",
    "alucinacion_0_1",
    "observaciones",
]


def _respuesta_id(fila: pd.Series) -> str:
    clave = f"{fila['consulta_id']}|{fila['modelo']}|{fila['condicion']}"
    digest = hashlib.sha256(clave.encode("utf-8")).hexdigest()[:12].upper()
    return f"EH-{digest}"


def _contextos_por_consulta(
    resultados: pd.DataFrame,
    corpus: pd.DataFrame,
) -> dict[str, str]:
    documentos = corpus.set_index("document_id")["texto_rag"].to_dict()
    rag = resultados.loc[resultados["condicion"].eq("rag")]
    contextos = {}
    for consulta_id, grupo in rag.groupby("consulta_id", sort=False):
        listas = grupo["documentos_recuperados"].drop_duplicates()
        if len(listas) != 1:
            raise RuntimeError(
                f"{consulta_id} no tiene una única recuperación compartida."
            )
        ids = json.loads(listas.iloc[0])
        faltantes = [documento_id for documento_id in ids if documento_id not in documentos]
        if faltantes:
            raise RuntimeError(f"Documentos ausentes para {consulta_id}: {faltantes}")
        contextos[consulta_id] = "\n\n---\n\n".join(
            documentos[documento_id] for documento_id in ids
        )
    return contextos


def construir_insumos_ciegos(
    resultados: pd.DataFrame,
    muestra_humana: pd.DataFrame,
    corpus: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Devuelve dos formularios aleatorizados y la clave de desenmascaramiento."""
    ids = muestra_humana["consulta_id"].tolist()
    if len(ids) != 54 or len(set(ids)) != 54:
        raise ValueError("La submuestra humana debe contener 54 consultas únicas.")

    seleccion = resultados.loc[resultados["consulta_id"].isin(ids)].copy()
    if len(seleccion) != 324:
        raise RuntimeError("Se esperaban 324 respuestas para evaluación humana.")
    if seleccion.duplicated(["modelo", "consulta_id", "condicion"]).any():
        raise RuntimeError("Hay respuestas experimentales duplicadas.")
    if set(seleccion["modelo"]) != {"gpt-4o-mini", "llama-3.1-8b", "mistral"}:
        raise RuntimeError("La selección no contiene los tres modelos previstos.")
    if set(seleccion["condicion"]) != {"rag", "sin_rag"}:
        raise RuntimeError("La selección no contiene ambas condiciones previstas.")

    contextos = _contextos_por_consulta(seleccion, corpus)
    seleccion["respuesta_id"] = seleccion.apply(_respuesta_id, axis=1)
    if seleccion["respuesta_id"].duplicated().any():
        raise RuntimeError("Se generaron identificadores ciegos duplicados.")
    seleccion["contexto_evaluacion"] = seleccion["consulta_id"].map(contextos)

    base = seleccion.rename(
        columns={
            "referencia": "respuesta_referencia",
            "respuesta": "respuesta_a_evaluar",
        }
    )
    formularios = {}
    ordenes = {}
    for nombre, semilla in SEMILLAS_EVALUADORES.items():
        formulario = base.sample(frac=1, random_state=semilla).reset_index(drop=True)
        formulario.insert(0, "orden_presentacion", range(1, len(formulario) + 1))
        for columna in COLUMNAS_FORMULARIO[7:]:
            formulario[columna] = ""
        formulario = formulario[COLUMNAS_FORMULARIO]
        formularios[nombre] = formulario
        ordenes[nombre] = formulario.set_index("respuesta_id")[
            "orden_presentacion"
        ].to_dict()

    clave = seleccion[
        [
            "respuesta_id",
            "consulta_id",
            "category",
            "intent",
            "modelo",
            "condicion",
            "tipo",
        ]
    ].copy()
    clave["orden_evaluador_1"] = clave["respuesta_id"].map(
        ordenes["evaluador_1"]
    )
    clave["orden_evaluador_2"] = clave["respuesta_id"].map(
        ordenes["evaluador_2"]
    )
    clave = clave.sort_values("respuesta_id").reset_index(drop=True)
    return formularios, clave
