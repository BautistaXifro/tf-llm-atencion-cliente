"""Traduce al español la muestra ampliada del Subproyecto 3.

Este script realiza llamadas pagas a OpenAI. Guarda un checkpoint después de
cada fila y, al reiniciarse, procesa únicamente las traducciones pendientes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.traduccion import (  # noqa: E402
    construir_auditoria,
    mascara_pendientes,
    preparar_dataframe,
    traducir_pendientes,
    validar_checkpoint,
)


COLUMNA_ID = "consulta_id"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entrada",
        type=Path,
        default=PROYECTO / "data" / "muestra_ampliada_en.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "data" / "muestra_ampliada_es.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entrada = preparar_dataframe(pd.read_csv(args.entrada), COLUMNA_ID)

    if args.salida.exists():
        corpus = preparar_dataframe(pd.read_csv(args.salida), COLUMNA_ID)
        validar_checkpoint(entrada, corpus, COLUMNA_ID)
        print(f"Checkpoint encontrado: {args.salida}", flush=True)
    else:
        corpus = entrada

    def guardar(df: pd.DataFrame):
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.salida, index=False)

    pendientes = int(mascara_pendientes(corpus).sum())
    print(
        f"Muestra: {len(corpus)} filas; pendientes: {pendientes}.",
        flush=True,
    )
    if pendientes:
        load_dotenv(PROYECTO / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en proyecto/.env.")
        cliente = OpenAI(api_key=api_key)
        corpus = traducir_pendientes(
            corpus,
            cliente,
            guardar,
            COLUMNA_ID,
            informar=lambda mensaje: print(mensaje, flush=True),
        )
    else:
        print("No hay traducciones pendientes; no se llamó a la API.", flush=True)

    guardar(corpus)
    auditoria = construir_auditoria(corpus, COLUMNA_ID)
    ruta_auditoria = args.salida.with_name(
        f"{args.salida.stem}_auditoria.json"
    )
    ruta_auditoria.write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(auditoria, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
