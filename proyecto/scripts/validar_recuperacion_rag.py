"""Valida recuperación RAG sin llamar a ningún modelo generativo."""

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
    for posicion, consulta in evaluacion.iterrows():
        recuperacion = indice.recuperar(consulta["instruction_es"], top_k=args.top_k)
        documentos = recuperacion["documentos"]
        intenciones = [doc["intent"] for doc in documentos]
        filas.append(
            {
                "consulta_id": consulta.get(
                    "consulta_id", f"consulta-{posicion + 1:04d}"
                ),
                "category": consulta.get("category", pd.NA),
                "consulta": consulta["instruction_es"],
                "intent_esperado": consulta["intent"],
                "documento_top1": documentos[0]["document_id"],
                "intent_top1": intenciones[0],
                "score_top1": documentos[0]["score"],
                "hit_top1": intenciones[0] == consulta["intent"],
                "hit_top_k": consulta["intent"] in intenciones,
                "documentos_recuperados": json.dumps(
                    [doc["document_id"] for doc in documentos]
                ),
                "intents_recuperados": json.dumps(intenciones),
                "scores_recuperacion": json.dumps(
                    [round(doc["score"], 6) for doc in documentos]
                ),
                "latencia_recuperacion_s": recuperacion[
                    "latencia_recuperacion_s"
                ],
            }
        )

    resultados = pd.DataFrame(filas)
    resultados.to_csv(args.salida, index=False)
    resumen_intenciones = (
        resultados.groupby("intent_esperado", as_index=False)
        .agg(
            consultas=("consulta_id", "size"),
            hit_top1=("hit_top1", "mean"),
            hit_top_k=("hit_top_k", "mean"),
            latencia_media_s=("latencia_recuperacion_s", "mean"),
        )
        .sort_values(["hit_top1", "hit_top_k", "intent_esperado"])
    )
    ruta_resumen = args.salida.with_name(
        f"{args.salida.stem}_por_intencion.csv"
    )
    resumen_intenciones.to_csv(ruta_resumen, index=False)
    print(f"Consultas: {len(resultados)}")
    print(f"Hit@1 por intención: {resultados['hit_top1'].mean():.1%}")
    print(f"Hit@{args.top_k} por intención: {resultados['hit_top_k'].mean():.1%}")
    print(
        "Latencia media de recuperación: "
        f"{resultados['latencia_recuperacion_s'].mean() * 1000:.2f} ms"
    )
    print(f"Resultado: {args.salida}")
    print(f"Resumen por intención: {ruta_resumen}")


if __name__ == "__main__":
    main()
