"""Ejecuta 324 consultas × tres modelos × dos condiciones con checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv


PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402
from src.experimento import (  # noqa: E402
    clave_resultado,
    claves_completadas,
    construir_plan,
    limitar_plan,
    reemplazar_resultado,
)
from src.rag import IndiceRAG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluacion",
        type=Path,
        default=PROYECTO / "data" / "muestra_ampliada_es.csv",
    )
    parser.add_argument(
        "--corpus-rag",
        type=Path,
        default=PROYECTO / "data" / "conocimiento_rag_ampliado.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "resultados_experimento_ampliado.csv",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        choices=list(ftf.MODELOS),
        default=list(ftf.MODELOS),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limite-consultas", type=int)
    parser.add_argument("--max-errores-consecutivos", type=int, default=3)
    parser.add_argument("--confirmar-corrida-completa", action="store_true")
    return parser.parse_args()


def _guardar(ruta: Path, filas: list[dict]):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).sort_values("orden_global").to_csv(ruta, index=False)


def main():
    args = parse_args()
    if args.limite_consultas is None and not args.confirmar_corrida_completa:
        raise RuntimeError(
            "La corrida completa requiere --confirmar-corrida-completa. Para "
            "una prueba use --limite-consultas N."
        )
    if args.max_errores_consecutivos < 1:
        raise ValueError("max-errores-consecutivos debe ser al menos 1.")

    load_dotenv(PROYECTO / ".env")
    evaluacion = pd.read_csv(args.evaluacion)
    conocimiento = pd.read_csv(args.corpus_rag)
    plan = limitar_plan(
        construir_plan(evaluacion, args.modelos), args.limite_consultas
    )
    consultas_plan = set(plan["consulta_id"])
    evaluacion_indice = evaluacion.set_index("consulta_id", drop=False)

    if any(ftf.MODELOS[modelo]["proveedor"] == "openai" for modelo in args.modelos):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Falta OPENAI_API_KEY en proyecto/.env.")
    if any(ftf.MODELOS[modelo]["proveedor"] == "ollama" for modelo in args.modelos):
        ftf.verificar_ollama()

    indice = IndiceRAG(conocimiento).construir()
    recuperaciones = {
        consulta_id: indice.recuperar(
            evaluacion_indice.loc[consulta_id, "instruction_es"],
            top_k=args.top_k,
        )
        for consulta_id in consultas_plan
    }

    if args.salida.exists():
        filas = pd.read_csv(args.salida).to_dict("records")
        print(f"Checkpoint encontrado: {len(filas)} filas.", flush=True)
    else:
        filas = []
    completadas = claves_completadas(filas)

    print(
        f"Plan activo: {len(plan)} respuestas; "
        f"completadas: {len(completadas)}.",
        flush=True,
    )
    modelo_anterior = None
    errores_consecutivos = 0
    for _, unidad in plan.iterrows():
        clave = clave_resultado(unidad)
        if clave in completadas:
            continue

        modelo = unidad["modelo"]
        config = ftf.MODELOS[modelo]
        if modelo != modelo_anterior and config["proveedor"] == "ollama":
            calentamiento = ftf.generar(modelo, "Respondé solamente: listo")
            if calentamiento["error"]:
                raise RuntimeError(
                    f"Falló el precalentamiento de {modelo}: "
                    f"{calentamiento['error']}"
                )
            print(f"{modelo}: precalentado.", flush=True)
        modelo_anterior = modelo

        consulta = evaluacion_indice.loc[unidad["consulta_id"]]
        recuperacion = recuperaciones[unidad["consulta_id"]]
        es_rag = unidad["condicion"] == "rag"
        contexto = recuperacion["contexto"] if es_rag else None
        salida = ftf.generar(modelo, consulta["instruction_es"], contexto=contexto)
        documentos = recuperacion["documentos"] if es_rag else []
        latencia_recuperacion = (
            recuperacion["latencia_recuperacion_s"] if es_rag else 0.0
        )
        nuevo = {
            **unidad.to_dict(),
            "tipo": config["tipo"],
            "category": consulta["category"],
            "intent": consulta["intent"],
            "consulta": consulta["instruction_es"],
            "referencia": consulta["response_es"],
            "respuesta": salida["respuesta"],
            "documentos_recuperados": json.dumps(
                [doc["document_id"] for doc in documentos]
            ),
            "intents_recuperados": json.dumps(
                [doc["intent"] for doc in documentos]
            ),
            "scores_recuperacion": json.dumps(
                [round(doc["score"], 6) for doc in documentos]
            ),
            "hit_top1": (
                documentos[0]["intent"] == consulta["intent"] if es_rag else pd.NA
            ),
            "hit_top_k": (
                consulta["intent"] in [doc["intent"] for doc in documentos]
                if es_rag
                else pd.NA
            ),
            "latencia_recuperacion_s": latencia_recuperacion,
            "latencia_generacion_s": salida["latencia_s"],
            "latencia_total_s": round(
                latencia_recuperacion + salida["latencia_s"], 6
            ),
            "latencia_s": salida["latencia_s"],
            "tokens_in": salida["tokens_in"],
            "tokens_out": salida["tokens_out"],
            "error": salida["error"],
        }
        filas = reemplazar_resultado(filas, nuevo)
        _guardar(args.salida, filas)

        if salida["error"]:
            errores_consecutivos += 1
        else:
            errores_consecutivos = 0
            completadas.add(clave)
        estado = salida["error"] or f"{salida['latencia_s']:.3f} s"
        print(
            f"{len(completadas):04d}/{len(plan)} — {modelo} — "
            f"{unidad['consulta_id']} — {unidad['condicion']} — "
            f"{estado}",
            flush=True,
        )
        if errores_consecutivos >= args.max_errores_consecutivos:
            raise RuntimeError(
                f"Se alcanzaron {errores_consecutivos} errores consecutivos; "
                "la corrida se detuvo y puede reanudarse."
            )
        if config["proveedor"] == "openai":
            time.sleep(0.5)

    print(
        f"Finalizado: {len(completadas)}/{len(plan)} respuestas completas. "
        f"Resultado: {args.salida}",
        flush=True,
    )


if __name__ == "__main__":
    main()
