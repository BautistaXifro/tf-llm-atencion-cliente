"""Estadística reproducible para la evaluación pareada del experimento."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SEMILLA_BOOTSTRAP = 20261014


def normalizar(serie: pd.Series, invertir: bool = False) -> pd.Series:
    """Normaliza a [0, 1]; usa 0,5 si todos los valores son iguales."""
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(0.5, index=serie.index, dtype=float)
    normalizada = (serie - serie.min()) / rango
    return 1 - normalizada if invertir else normalizada


def normalizar_por_grupo(
    datos: pd.DataFrame,
    columna: str,
    grupo: str,
    *,
    invertir: bool = False,
) -> pd.Series:
    """Normaliza una columna por separado dentro de cada grupo."""
    faltantes = {columna, grupo} - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    resultado = pd.Series(index=datos.index, dtype=float)
    for _, indices in datos.groupby(grupo, sort=False).groups.items():
        resultado.loc[indices] = normalizar(
            datos.loc[indices, columna],
            invertir=invertir,
        )
    return resultado


def intervalo_bootstrap_media(
    valores,
    estratos=None,
    *,
    repeticiones: int = 5000,
    semilla: int = SEMILLA_BOOTSTRAP,
) -> tuple[float, float]:
    """IC percentil pareado, opcionalmente re-muestreado dentro de estratos."""
    valores = np.asarray(valores, dtype=float)
    if valores.ndim != 1 or not len(valores):
        raise ValueError("valores debe ser un vector no vacío.")
    if repeticiones < 100:
        raise ValueError("repeticiones debe ser al menos 100.")
    rng = np.random.default_rng(semilla)

    if estratos is None:
        indices = rng.integers(0, len(valores), size=(repeticiones, len(valores)))
        medias = valores[indices].mean(axis=1)
    else:
        estratos = np.asarray(estratos)
        if len(estratos) != len(valores):
            raise ValueError("estratos y valores deben tener la misma longitud.")
        grupos = [np.flatnonzero(estratos == e) for e in np.unique(estratos)]
        medias = np.empty(repeticiones, dtype=float)
        for repeticion in range(repeticiones):
            muestra = np.concatenate(
                [rng.choice(grupo, size=len(grupo), replace=True) for grupo in grupos]
            )
            medias[repeticion] = valores[muestra].mean()
    inferior, superior = np.quantile(medias, [0.025, 0.975])
    return float(inferior), float(superior)


def rango_biserial_pareado(deltas) -> float:
    """Tamaño de efecto de rangos; positivo favorece la condición RAG."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[deltas != 0]
    if not len(deltas):
        return 0.0
    rangos = rankdata(np.abs(deltas), method="average")
    positivos = float(rangos[deltas > 0].sum())
    negativos = float(rangos[deltas < 0].sum())
    return (positivos - negativos) / (positivos + negativos)


def prueba_wilcoxon(deltas) -> tuple[float, float]:
    """Devuelve estadístico y p bilateral; tolera una serie toda en cero."""
    deltas = np.asarray(deltas, dtype=float)
    if np.all(deltas == 0):
        return 0.0, 1.0
    resultado = wilcoxon(
        deltas,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(resultado.statistic), float(resultado.pvalue)


def ajuste_holm(p_valores) -> np.ndarray:
    """Corrige una familia de p-valores con el método secuencial de Holm."""
    p = np.asarray(p_valores, dtype=float)
    if p.ndim != 1 or not len(p):
        raise ValueError("p_valores debe ser un vector no vacío.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Los p-valores deben estar entre 0 y 1.")
    orden = np.argsort(p)
    ajustados_ordenados = np.empty(len(p), dtype=float)
    acumulado = 0.0
    for posicion, indice in enumerate(orden):
        candidato = (len(p) - posicion) * p[indice]
        acumulado = max(acumulado, candidato)
        ajustados_ordenados[posicion] = min(acumulado, 1.0)
    ajustados = np.empty(len(p), dtype=float)
    ajustados[orden] = ajustados_ordenados
    return ajustados
