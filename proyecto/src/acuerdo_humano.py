"""Validación y acuerdo interevaluador para las planillas ciegas."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRICAS_LIKERT = [
    "correccion_1_5",
    "utilidad_1_5",
    "claridad_1_5",
    "fidelidad_contexto_1_5",
]
METRICA_BINARIA = "alucinacion_0_1"
METRICAS = [*METRICAS_LIKERT, METRICA_BINARIA]


def kappa_cohen(
    a,
    b,
    *,
    categorias: list[int],
    ponderacion: str | None = None,
) -> float:
    """Calcula kappa de Cohen; admite ponderación lineal para escalas ordinales."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b) or not len(a):
        raise ValueError("Las dos series deben tener igual longitud y no estar vacías.")
    indices = {valor: posicion for posicion, valor in enumerate(categorias)}
    if any(valor not in indices for valor in np.concatenate([a, b])):
        raise ValueError("Se encontraron valores fuera de las categorías previstas.")

    matriz = np.zeros((len(categorias), len(categorias)), dtype=float)
    for valor_a, valor_b in zip(a, b):
        matriz[indices[valor_a], indices[valor_b]] += 1
    observada = matriz / matriz.sum()
    esperada = np.outer(observada.sum(axis=1), observada.sum(axis=0))

    if ponderacion is None:
        pesos = np.ones_like(matriz) - np.eye(len(categorias))
    elif ponderacion == "lineal":
        posiciones = np.arange(len(categorias))
        pesos = np.abs(posiciones[:, None] - posiciones[None, :]) / (
            len(categorias) - 1
        )
    else:
        raise ValueError(f"Ponderación desconocida: {ponderacion}")
    denominador = float((pesos * esperada).sum())
    if denominador == 0:
        return np.nan
    return 1 - float((pesos * observada).sum()) / denominador


def validar_formulario(formulario: pd.DataFrame, nombre: str) -> pd.DataFrame:
    requeridas = {"respuesta_id", *METRICAS}
    faltantes = requeridas - set(formulario.columns)
    if faltantes:
        raise ValueError(f"{nombre}: faltan columnas {sorted(faltantes)}")
    if len(formulario) != 324 or formulario["respuesta_id"].nunique() != 324:
        raise ValueError(f"{nombre}: se esperaban 324 respuestas_id únicas.")

    validado = formulario[["respuesta_id", *METRICAS]].copy()
    for metrica in METRICAS:
        validado[metrica] = pd.to_numeric(validado[metrica], errors="coerce")
    if validado[METRICAS].isna().any().any():
        pendientes = int(validado[METRICAS].isna().any(axis=1).sum())
        raise ValueError(f"{nombre}: quedan {pendientes} filas sin completar.")
    for metrica in METRICAS_LIKERT:
        if not validado[metrica].isin(range(1, 6)).all():
            raise ValueError(f"{nombre}: {metrica} debe contener enteros de 1 a 5.")
    if not validado[METRICA_BINARIA].isin([0, 1]).all():
        raise ValueError(f"{nombre}: {METRICA_BINARIA} debe contener 0 o 1.")
    return validado


def calcular_acuerdo(
    evaluador_1: pd.DataFrame,
    evaluador_2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    uno = validar_formulario(evaluador_1, "evaluador_1")
    dos = validar_formulario(evaluador_2, "evaluador_2")
    pares = uno.merge(
        dos,
        on="respuesta_id",
        suffixes=("_evaluador_1", "_evaluador_2"),
        validate="one_to_one",
    )
    if len(pares) != 324:
        raise ValueError("Los evaluadores no calificaron el mismo conjunto de respuestas.")

    filas = []
    for metrica in METRICAS:
        a = pares[f"{metrica}_evaluador_1"].to_numpy()
        b = pares[f"{metrica}_evaluador_2"].to_numpy()
        es_likert = metrica in METRICAS_LIKERT
        filas.append(
            {
                "metrica": metrica,
                "n": len(pares),
                "acuerdo_exacto_pct": float((a == b).mean() * 100),
                "diferencia_absoluta_media": float(np.abs(a - b).mean()),
                "kappa_cohen": kappa_cohen(
                    a,
                    b,
                    categorias=list(range(1, 6)) if es_likert else [0, 1],
                    ponderacion="lineal" if es_likert else None,
                ),
                "tipo_kappa": "ponderado_lineal" if es_likert else "sin_ponderar",
            }
        )
        pares[f"{metrica}_promedio"] = (a + b) / 2
    return pares, pd.DataFrame(filas)
