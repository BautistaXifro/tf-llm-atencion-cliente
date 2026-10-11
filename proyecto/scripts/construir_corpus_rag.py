"""Construye el corpus piloto de conocimiento para el Subproyecto 2.

Uso desde ``proyecto/``:

    .\venv\Scripts\python.exe scripts\construir_corpus_rag.py

Descarga Bitext únicamente si no está presente en la caché de Hugging Face.
No realiza llamadas a modelos generativos ni genera costos de API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from datasets import load_dataset  # noqa: E402
from src.rag import construir_corpus_conocimiento  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluacion",
        type=Path,
        default=PROYECTO / "data" / "corpus_es_muestra.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "data" / "conocimiento_rag.csv",
    )
    parser.add_argument("--documentos-por-intencion", type=int, default=3)
    parser.add_argument("--umbral-similitud", type=float, default=0.90)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluacion = pd.read_csv(args.evaluacion)
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    corpus, auditoria = construir_corpus_conocimiento(
        dataset.to_pandas(),
        evaluacion,
        documentos_por_intencion=args.documentos_por_intencion,
        umbral_similitud=args.umbral_similitud,
    )

    columnas = [
        "document_id",
        "source_row_id",
        "category",
        "intent",
        "instruction",
        "response",
        "similitud_max_evaluacion",
        "texto_rag",
    ]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    corpus[columnas].to_csv(args.salida, index=False)

    ruta_auditoria = args.salida.with_name(f"{args.salida.stem}_auditoria.json")
    ruta_auditoria.write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(auditoria, ensure_ascii=False, indent=2))
    print(f"Corpus: {args.salida}")
    print(f"Auditoría: {ruta_auditoria}")


if __name__ == "__main__":
    main()
