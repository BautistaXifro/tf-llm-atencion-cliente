"""Construcción reproducible de muestras experimentales balanceadas.

El módulo trabaja sobre un ``DataFrame`` y permite indicar los nombres de las
columnas. La descarga y el conocimiento de Bitext quedan fuera de este archivo,
en el adaptador de línea de comandos correspondiente.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


SEMILLA_MUESTRA_AMPLIADA = 20261011
SEMILLA_EVALUACION_HUMANA = 20261012


def _normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", str(texto)).casefold()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return " ".join(texto.split())


def construir_muestra_balanceada(
    datos: pd.DataFrame,
    filas_por_estrato: int,
    *,
    columna_estrato: str = "intent",
    columna_texto: str = "instruction",
    semilla: int = SEMILLA_MUESTRA_AMPLIADA,
    prefijo_id: str = "eval",
) -> tuple[pd.DataFrame, dict]:
    """Selecciona la misma cantidad de filas por estrato.

    Primero descarta textos vacíos y duplicados normalizados. Conserva el índice
    original en ``source_row_id`` y genera un ``consulta_id`` estable para la
    muestra resultante. Falla de forma explícita cuando un estrato no dispone de
    suficientes filas, evitando completar la muestra de manera desbalanceada.
    """
    if filas_por_estrato < 1:
        raise ValueError("filas_por_estrato debe ser al menos 1.")

    requeridas = {columna_estrato, columna_texto}
    faltantes = requeridas - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    trabajo = datos.copy()
    trabajo["source_row_id"] = trabajo.index
    filas_iniciales = len(trabajo)

    texto_no_vacio = trabajo[columna_texto].notna() & (
        trabajo[columna_texto].astype(str).str.strip() != ""
    )
    estrato_no_vacio = trabajo[columna_estrato].notna() & (
        trabajo[columna_estrato].astype(str).str.strip() != ""
    )
    trabajo = trabajo.loc[texto_no_vacio & estrato_no_vacio].copy()
    filas_vacias_eliminadas = filas_iniciales - len(trabajo)

    trabajo["texto_normalizado"] = trabajo[columna_texto].map(_normalizar_texto)
    antes_duplicados = len(trabajo)
    trabajo = trabajo.drop_duplicates("texto_normalizado", keep="first").copy()
    duplicados_eliminados = antes_duplicados - len(trabajo)

    conteos = trabajo.groupby(columna_estrato, sort=True).size()
    insuficientes = conteos[conteos < filas_por_estrato]
    if not insuficientes.empty:
        detalle = ", ".join(
            f"{estrato}={cantidad}"
            for estrato, cantidad in insuficientes.items()
        )
        raise ValueError(
            "No hay suficientes filas únicas para todos los estratos: " + detalle
        )

    bloques = []
    for _, grupo in trabajo.groupby(columna_estrato, sort=True):
        bloques.append(
            grupo.sample(n=filas_por_estrato, random_state=semilla)
        )

    muestra = pd.concat(bloques, ignore_index=True)
    muestra = muestra.sort_values(
        [columna_estrato, "source_row_id"], kind="stable"
    ).reset_index(drop=True)
    ancho = max(4, len(str(len(muestra))))
    muestra.insert(
        0,
        "consulta_id",
        [f"{prefijo_id}-{i:0{ancho}d}" for i in range(1, len(muestra) + 1)],
    )

    auditoria = {
        "filas_dataset": filas_iniciales,
        "filas_vacias_eliminadas": filas_vacias_eliminadas,
        "duplicados_eliminados": duplicados_eliminados,
        "columna_estrato": columna_estrato,
        "columna_texto": columna_texto,
        "estratos": int(muestra[columna_estrato].nunique()),
        "filas_por_estrato": filas_por_estrato,
        "filas_seleccionadas": len(muestra),
        "semilla": semilla,
    }
    return muestra, auditoria


def seleccionar_submuestra_humana(
    muestra: pd.DataFrame,
    filas_por_estrato: int = 2,
    *,
    columna_estrato: str = "intent",
    semilla: int = SEMILLA_EVALUACION_HUMANA,
) -> pd.DataFrame:
    """Selecciona consultas para evaluación humana preservando sus IDs."""
    requeridas = {"consulta_id", columna_estrato}
    faltantes = requeridas - set(muestra.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if filas_por_estrato < 1:
        raise ValueError("filas_por_estrato debe ser al menos 1.")

    conteos = muestra.groupby(columna_estrato, sort=True).size()
    insuficientes = conteos[conteos < filas_por_estrato]
    if not insuficientes.empty:
        detalle = ", ".join(
            f"{estrato}={cantidad}"
            for estrato, cantidad in insuficientes.items()
        )
        raise ValueError(
            "No hay suficientes consultas para la submuestra humana: " + detalle
        )

    bloques = []
    for _, grupo in muestra.groupby(columna_estrato, sort=True):
        bloques.append(
            grupo.sample(n=filas_por_estrato, random_state=semilla)
        )
    return pd.concat(bloques, ignore_index=True).sort_values(
        [columna_estrato, "consulta_id"], kind="stable"
    ).reset_index(drop=True)
