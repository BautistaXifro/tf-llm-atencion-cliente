"""Utilidades reproducibles para traducir corpus con checkpoint.

El módulo no crea clientes ni lee credenciales. Recibe un cliente compatible
con OpenAI para que la lógica de preparación, reanudación y auditoría pueda
probarse sin realizar llamadas reales.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pandas as pd

from src.rag import PATRON_MARCADOR, restaurar_marcadores


MODELO_TRADUCCION = "gpt-4o-mini-2024-07-18"
PRECIO_ENTRADA_MILLON = 0.15
PRECIO_SALIDA_MILLON = 0.60
PROMPT_TRADUCCION = (
    "Traducí al español rioplatense neutro la pregunta y la respuesta de "
    "soporte. Conservá sin cambios y en la misma posición todos los tokens "
    "que comienzan con __TF_ y terminan con __. Conservá el sentido, los pasos "
    "y el tono. Respondé solamente un objeto JSON con las claves "
    "instruction_es y response_es."
)
PROMPT_SEGMENTOS = (
    "Traducí al español rioplatense neutro cada elemento de los arreglos de "
    "texto. Conservá exactamente la cantidad y el orden de los elementos, "
    "incluidos los elementos vacíos. No unas ni elimines fragmentos. Respondé "
    "solamente un objeto JSON con las claves instruction_segments_es y "
    "response_segments_es."
)

COLUMNAS_TRADUCCION = (
    "instruction_es",
    "response_es",
    "tokens_traduccion_in",
    "tokens_traduccion_out",
    "intentos_traduccion",
    "salidas_traduccion_rechazadas",
)


def costo_traduccion(tokens_in: int, tokens_out: int) -> float:
    """Calcula el costo según los precios fijados para el experimento."""
    return (
        tokens_in / 1_000_000 * PRECIO_ENTRADA_MILLON
        + tokens_out / 1_000_000 * PRECIO_SALIDA_MILLON
    )


def preparar_dataframe(df: pd.DataFrame, columna_id: str) -> pd.DataFrame:
    """Valida el esquema y agrega las columnas necesarias para el checkpoint."""
    requeridas = {columna_id, "instruction", "response"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para traducir: {sorted(faltantes)}")
    if df[columna_id].isna().any() or df[columna_id].duplicated().any():
        raise ValueError(f"La columna {columna_id!r} debe ser única y no nula.")

    preparado = df.copy()
    for columna in ("instruction_es", "response_es"):
        if columna not in preparado:
            preparado[columna] = pd.NA
    for columna in ("tokens_traduccion_in", "tokens_traduccion_out"):
        if columna not in preparado:
            preparado[columna] = 0
        preparado[columna] = pd.to_numeric(
            preparado[columna], errors="coerce"
        ).fillna(0).astype(int)
    if "intentos_traduccion" not in preparado:
        preparado["intentos_traduccion"] = (
            preparado["tokens_traduccion_in"]
            + preparado["tokens_traduccion_out"]
            > 0
        ).astype(int)
    if "salidas_traduccion_rechazadas" not in preparado:
        preparado["salidas_traduccion_rechazadas"] = 0
    for columna in (
        "intentos_traduccion",
        "salidas_traduccion_rechazadas",
    ):
        preparado[columna] = pd.to_numeric(
            preparado[columna], errors="coerce"
        ).fillna(0).astype(int)
    return preparado


def validar_checkpoint(
    entrada: pd.DataFrame,
    checkpoint: pd.DataFrame,
    columna_id: str,
) -> None:
    """Impide reanudar con una salida perteneciente a otra muestra."""
    ids_entrada = entrada[columna_id].astype(str).tolist()
    ids_checkpoint = checkpoint[columna_id].astype(str).tolist()
    if ids_entrada != ids_checkpoint:
        raise ValueError(
            "El checkpoint no corresponde a la entrada: los identificadores "
            "o su orden son diferentes."
        )
    for columna in ("instruction", "response"):
        original = entrada[columna].fillna("").astype(str)
        guardado = checkpoint[columna].fillna("").astype(str)
        if not original.equals(guardado):
            raise ValueError(
                "El checkpoint no corresponde a la entrada: cambió la columna "
                f"{columna!r}."
            )


def _esta_pendiente(valor) -> bool:
    return pd.isna(valor) or not str(valor).strip()


def mascara_pendientes(df: pd.DataFrame) -> pd.Series:
    """Marca filas a las que les falta al menos uno de los dos textos."""
    return df["instruction_es"].map(_esta_pendiente) | df["response_es"].map(
        _esta_pendiente
    )


def _extraer_json(respuesta) -> dict:
    contenido = respuesta.choices[0].message.content
    datos = json.loads(contenido)
    requeridas = {"instruction_es", "response_es"}
    faltantes = requeridas - set(datos)
    if faltantes:
        raise ValueError(
            f"La respuesta de traducción no contiene: {sorted(faltantes)}"
        )
    if not all(isinstance(datos[columna], str) for columna in requeridas):
        raise ValueError("Las traducciones recibidas deben ser cadenas de texto.")
    return datos


def enmascarar_marcadores(texto: str, prefijo: str) -> tuple[str, dict[str, str]]:
    """Sustituye cada marcador por un token único antes de traducir."""
    equivalencias: dict[str, str] = {}

    def reemplazar(coincidencia):
        numero = len(equivalencias) + 1
        token = f"__TF_{prefijo}_MARCADOR_{numero:03d}__"
        equivalencias[token] = coincidencia.group(0)
        return token

    return PATRON_MARCADOR.sub(reemplazar, str(texto)), equivalencias


def restaurar_tokens_marcadores(
    traduccion: str,
    equivalencias: dict[str, str],
) -> str:
    """Restaura tokens únicos y rechaza pérdidas o duplicaciones."""
    restaurada = str(traduccion)
    for token, marcador in equivalencias.items():
        apariciones = restaurada.count(token)
        if apariciones != 1:
            raise ValueError(
                f"La traducción contiene {apariciones} apariciones de {token}; "
                "se esperaba exactamente una."
            )
        restaurada = restaurada.replace(token, marcador)
    return restaurada


def separar_por_marcadores(texto: str) -> tuple[list[str], list[str]]:
    """Separa texto y marcadores para una traducción de fallback sin pérdidas."""
    texto = str(texto)
    return PATRON_MARCADOR.split(texto), PATRON_MARCADOR.findall(texto)


def reconstruir_con_marcadores(
    segmentos: list[str],
    marcadores: list[str],
) -> str:
    """Intercala segmentos traducidos con los marcadores originales."""
    if len(segmentos) != len(marcadores) + 1:
        raise ValueError(
            "La cantidad de segmentos traducidos no permite reconstruir los "
            f"marcadores: {len(segmentos)} segmentos y {len(marcadores)} marcadores."
        )
    partes: list[str] = []
    for posicion, segmento in enumerate(segmentos):
        if not isinstance(segmento, str):
            raise ValueError("Todos los segmentos traducidos deben ser texto.")
        partes.append(segmento)
        if posicion < len(marcadores):
            partes.append(marcadores[posicion])
    return "".join(partes)


def _extraer_segmentos(
    respuesta,
    cantidad_instruction: int,
    cantidad_response: int,
) -> tuple[list[str], list[str]]:
    contenido = respuesta.choices[0].message.content
    datos = json.loads(contenido)
    instruction = datos.get("instruction_segments_es")
    response = datos.get("response_segments_es")
    if not isinstance(instruction, list) or not isinstance(response, list):
        raise ValueError("La respuesta segmentada no contiene los dos arreglos.")
    if len(instruction) != cantidad_instruction or len(response) != cantidad_response:
        raise ValueError(
            "La traducción cambió la cantidad de segmentos: "
            f"instruction {cantidad_instruction}->{len(instruction)}, "
            f"response {cantidad_response}->{len(response)}."
        )
    return instruction, response


def traducir_pendientes(
    df: pd.DataFrame,
    cliente,
    guardar_checkpoint: Callable[[pd.DataFrame], None],
    columna_id: str,
    *,
    modelo: str = MODELO_TRADUCCION,
    pausa_s: float = 0.1,
    max_reintentos: int = 3,
    dormir: Callable[[float], None] = time.sleep,
    informar: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Traduce solo filas pendientes y persiste cada resultado exitoso."""
    if max_reintentos < 1:
        raise ValueError("max_reintentos debe ser al menos 1.")
    corpus = preparar_dataframe(df, columna_id)
    indices = corpus.index[mascara_pendientes(corpus)].tolist()
    total = len(indices)

    for posicion, indice in enumerate(indices, start=1):
        fila = corpus.loc[indice]
        instruction_api, marcadores_instruction = enmascarar_marcadores(
            fila["instruction"], "I"
        )
        response_api, marcadores_response = enmascarar_marcadores(
            fila["response"], "R"
        )
        segmentos_instruction, originales_instruction = separar_por_marcadores(
            fila["instruction"]
        )
        segmentos_response, originales_response = separar_por_marcadores(
            fila["response"]
        )
        usar_segmentos = bool(
            corpus.at[indice, "salidas_traduccion_rechazadas"]
        )

        for intento in range(1, max_reintentos + 1):
            respuesta = None
            try:
                if usar_segmentos:
                    contenido = json.dumps(
                        {
                            "instruction_segments": segmentos_instruction,
                            "response_segments": segmentos_response,
                        },
                        ensure_ascii=False,
                    )
                    prompt = PROMPT_SEGMENTOS
                else:
                    contenido = json.dumps(
                        {
                            "instruction": instruction_api,
                            "response": response_api,
                        },
                        ensure_ascii=False,
                    )
                    prompt = PROMPT_TRADUCCION
                respuesta = cliente.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": contenido},
                    ],
                    temperature=0.0,
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                )
                corpus.at[indice, "tokens_traduccion_in"] += int(
                    respuesta.usage.prompt_tokens
                )
                corpus.at[indice, "tokens_traduccion_out"] += int(
                    respuesta.usage.completion_tokens
                )
                corpus.at[indice, "intentos_traduccion"] += 1
                guardar_checkpoint(corpus)
                if usar_segmentos:
                    instruction_es_segmentos, response_es_segmentos = (
                        _extraer_segmentos(
                            respuesta,
                            len(segmentos_instruction),
                            len(segmentos_response),
                        )
                    )
                    instruction_es = reconstruir_con_marcadores(
                        instruction_es_segmentos, originales_instruction
                    )
                    response_es = reconstruir_con_marcadores(
                        response_es_segmentos, originales_response
                    )
                else:
                    traduccion = _extraer_json(respuesta)
                    instruction_es = restaurar_tokens_marcadores(
                        traduccion["instruction_es"], marcadores_instruction
                    )
                    response_es = restaurar_tokens_marcadores(
                        traduccion["response_es"], marcadores_response
                    )
                instruction_es = restaurar_marcadores(
                    fila["instruction"], instruction_es
                )
                response_es = restaurar_marcadores(fila["response"], response_es)
                break
            except Exception as exc:
                if respuesta is not None:
                    corpus.at[indice, "salidas_traduccion_rechazadas"] += 1
                    guardar_checkpoint(corpus)
                    usar_segmentos = True
                if intento == max_reintentos:
                    raise RuntimeError(
                        f"No se pudo traducir {fila[columna_id]!r} después de "
                        f"{max_reintentos} intentos."
                    ) from exc
                espera = 2 ** (intento - 1)
                informar(
                    f"Reintento {intento}/{max_reintentos - 1} para "
                    f"{fila[columna_id]} en {espera} s: {exc}"
                )
                dormir(espera)

        corpus.at[indice, "instruction_es"] = instruction_es
        corpus.at[indice, "response_es"] = response_es
        guardar_checkpoint(corpus)
        informar(
            f"Traducido {posicion}/{total}: {fila[columna_id]}"
        )
        if pausa_s:
            dormir(pausa_s)

    return corpus


def construir_auditoria(df: pd.DataFrame, columna_id: str) -> dict:
    """Resume completitud, consumo y costo del checkpoint actual."""
    corpus = preparar_dataframe(df, columna_id)
    pendientes = int(mascara_pendientes(corpus).sum())
    tokens_in = int(corpus["tokens_traduccion_in"].sum())
    tokens_out = int(corpus["tokens_traduccion_out"].sum())
    return {
        "modelo": MODELO_TRADUCCION,
        "filas_totales": len(corpus),
        "filas_traducidas": len(corpus) - pendientes,
        "filas_pendientes": pendientes,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "intentos_api_registrados": int(corpus["intentos_traduccion"].sum()),
        "salidas_rechazadas_registradas": int(
            corpus["salidas_traduccion_rechazadas"].sum()
        ),
        "precio_entrada_usd_por_millon": PRECIO_ENTRADA_MILLON,
        "precio_salida_usd_por_millon": PRECIO_SALIDA_MILLON,
        "costo_total_usd": round(costo_traduccion(tokens_in, tokens_out), 6),
    }
