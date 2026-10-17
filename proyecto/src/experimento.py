"""Plan reproducible y checkpoint del experimento ampliado."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SEMILLA_PLAN = 20261013
CONDICIONES = ("sin_rag", "rag")


def construir_plan(
    evaluacion: pd.DataFrame,
    modelos: Iterable[str],
    semilla: int = SEMILLA_PLAN,
    condiciones: Iterable[str] = CONDICIONES,
) -> pd.DataFrame:
    """Crea un plan pareado, aleatorio y balanceado por modelo.

    Las dos condiciones de una consulta quedan consecutivas. El orden AB/BA se
    alterna, por lo que una muestra par asigna exactamente la mitad a cada uno.
    """
    requeridas = {"consulta_id"}
    faltantes = requeridas - set(evaluacion.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en la evaluación: {sorted(faltantes)}")
    if evaluacion["consulta_id"].isna().any():
        raise ValueError("consulta_id no puede contener valores nulos.")
    if evaluacion["consulta_id"].duplicated().any():
        raise ValueError("consulta_id debe ser único.")
    modelos = list(modelos)
    if not modelos or len(modelos) != len(set(modelos)):
        raise ValueError("Debe indicarse una lista no vacía de modelos únicos.")
    condiciones = tuple(condiciones)
    if not condiciones or len(condiciones) != len(set(condiciones)):
        raise ValueError("Debe indicarse una lista no vacía de condiciones únicas.")
    desconocidas = set(condiciones) - set(CONDICIONES)
    if desconocidas:
        raise ValueError(f"Condiciones desconocidas: {sorted(desconocidas)}")

    filas = []
    orden_global = 0
    for orden_modelo, modelo in enumerate(modelos, start=1):
        consultas = evaluacion.sample(
            frac=1,
            random_state=semilla + orden_modelo - 1,
        )
        for posicion, (_, consulta) in enumerate(consultas.iterrows(), start=1):
            orden_condiciones = condiciones if posicion % 2 else condiciones[::-1]
            for orden_condicion, condicion in enumerate(orden_condiciones, start=1):
                orden_global += 1
                filas.append(
                    {
                        "orden_global": orden_global,
                        "orden_modelo": orden_modelo,
                        "posicion_consulta_modelo": posicion,
                        "orden_condicion": orden_condicion,
                        "modelo": modelo,
                        "consulta_id": consulta["consulta_id"],
                        "condicion": condicion,
                    }
                )
    return pd.DataFrame(filas)


def limitar_plan(plan: pd.DataFrame, limite_consultas: int | None) -> pd.DataFrame:
    """Limita consultas por modelo sin separar sus dos condiciones."""
    if limite_consultas is None:
        return plan.copy()
    if limite_consultas < 1:
        raise ValueError("limite_consultas debe ser al menos 1.")
    return plan.loc[
        plan["posicion_consulta_modelo"] <= limite_consultas
    ].copy()


def clave_resultado(fila) -> tuple[str, str, str]:
    return (str(fila["modelo"]), str(fila["consulta_id"]), str(fila["condicion"]))


def claves_completadas(resultados: list[dict]) -> set[tuple[str, str, str]]:
    """Devuelve claves con respuesta válida; los errores quedan reanudables."""
    completadas = set()
    for fila in resultados:
        if pd.isna(fila.get("error")) and not pd.isna(fila.get("respuesta")):
            completadas.add(clave_resultado(fila))
    return completadas


def reemplazar_resultado(resultados: list[dict], nuevo: dict) -> list[dict]:
    """Reemplaza un intento anterior de la misma unidad experimental."""
    clave_nueva = clave_resultado(nuevo)
    return [
        fila for fila in resultados if clave_resultado(fila) != clave_nueva
    ] + [nuevo]
