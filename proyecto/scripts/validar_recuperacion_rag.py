"""Valida recuperación RAG sin llamar a ningún modelo generativo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from src.rag import IndiceRAG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROYECTO / "data" / "conocimiento_rag.csv",
    )
    parser.add_argument(
        "--evaluacion",
        type=Path,
        default=PROYECTO / "data" / "corpus_es_muestra.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "resultados_recuperacion_rag_en.csv",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    corpus = pd.read_csv(args.corpus)
    evaluacion = pd.read_csv(args.evaluacion)
    indice = IndiceRAG(corpus).construir()

    filas = []
    for _, consulta in evaluacion.iterrows():
        recuperacion = indice.recuperar(consulta["instruction_es"], top_k=args.top_k)
        documentos = recuperacion["documentos"]
        intenciones = [doc["intent"] for doc in documentos]
        filas.append(
            {
                "consulta": consulta["instruction_es"],
                "intent_esperado": consulta["intent"],
                "documento_top1": documentos[0]["document_id"],
                "intent_top1": intenciones[0],
                "score_top1": documentos[0]["score"],
                "hit_top1": intenciones[0] == consulta["intent"],
                "hit_top_k": consulta["intent"] in intenciones,
                "documentos_recuperados": "|".join(
                    doc["document_id"] for doc in documentos
                ),
                "latencia_recuperacion_s": recuperacion[
                    "latencia_recuperacion_s"
                ],
            }
        )

    resultados = pd.DataFrame(filas)
    resultados.to_csv(args.salida, index=False)
    print(f"Consultas: {len(resultados)}")
    print(f"Hit@1 por intención: {resultados['hit_top1'].mean():.1%}")
    print(f"Hit@{args.top_k} por intención: {resultados['hit_top_k'].mean():.1%}")
    print(
        "Latencia media de recuperación: "
        f"{resultados['latencia_recuperacion_s'].mean() * 1000:.2f} ms"
    )
    print(f"Resultado: {args.salida}")


if __name__ == "__main__":
    main()
