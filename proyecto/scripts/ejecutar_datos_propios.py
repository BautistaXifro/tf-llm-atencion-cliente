"""Ejecuta el framework sobre datos externos normalizados con el esquema v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402
from src.esquema_datos import (  # noqa: E402
    cargar_conocimiento,
    cargar_consultas,
    cargar_mapeo,
)
from src.experimento import (  # noqa: E402
    clave_resultado,
    claves_completadas,
    construir_plan,
    limitar_plan,
    reemplazar_resultado,
)
from src.pipeline_usuario import (  # noqa: E402
    metricas_recuperacion,
    preparar_entradas,
    resumir_preflight,
    validar_checkpoint,
    validar_configuracion,
)
from src.rag import IndiceRAG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida datos propios, muestra el plan y, con confirmación explícita, "
            "genera respuestas con el pipeline del Trabajo Final."
        )
    )
    parser.add_argument("--consultas", type=Path, required=True)
    parser.add_argument("--conocimiento", type=Path)
    parser.add_argument("--mapeo-consultas", type=Path)
    parser.add_argument("--mapeo-conocimiento", type=Path)
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "resultados_datos_propios.csv",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        choices=list(ftf.MODELOS),
        default=list(ftf.MODELOS),
    )
    parser.add_argument(
        "--condiciones",
        nargs="+",
        choices=("sin_rag", "rag"),
        help=(
            "Por defecto usa sin_rag y, si se proporciona conocimiento, también rag."
        ),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limite-consultas", type=int)
    parser.add_argument("--max-errores-consecutivos", type=int, default=3)
    parser.add_argument("--confirmar-ejecucion", action="store_true")
    return parser.parse_args()


def _guardar(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).sort_values("orden_global").to_csv(
        ruta, index=False, encoding="utf-8"
    )


def _opcional(valor):
    return None if pd.isna(valor) else valor


def main() -> None:
    args = parse_args()
    if args.max_errores_consecutivos < 1:
        raise ValueError("max-errores-consecutivos debe ser al menos 1.")

    consultas = cargar_consultas(
        args.consultas,
        cargar_mapeo(args.mapeo_consultas),
    )
    conocimiento = None
    if args.conocimiento is not None:
        conocimiento = cargar_conocimiento(
            args.conocimiento,
            cargar_mapeo(args.mapeo_conocimiento),
        )
    evaluacion, documentos = preparar_entradas(consultas, conocimiento)
    condiciones = args.condiciones or (
        ["sin_rag", "rag"] if documentos is not None else ["sin_rag"]
    )
    condiciones = validar_configuracion(condiciones, documentos, args.top_k)
    plan = limitar_plan(
        construir_plan(
            evaluacion,
            args.modelos,
            condiciones=condiciones,
        ),
        args.limite_consultas,
    )
    consultas_activas = evaluacion.loc[
        evaluacion["consulta_id"].isin(plan["consulta_id"].unique())
    ]
    resumen = resumir_preflight(
        consultas_activas,
        documentos,
        args.modelos,
        condiciones,
    )
    resumen["respuestas_planificadas"] = len(plan)
    print(json.dumps(resumen, ensure_ascii=False, indent=2), flush=True)

    if not args.confirmar_ejecucion:
        print(
            "Previsualización finalizada sin invocar modelos. "
            "Agregue --confirmar-ejecucion para iniciar la generación.",
            flush=True,
        )
        return

    load_dotenv(PROYECTO / ".env")
    if any(ftf.MODELOS[m]["proveedor"] == "openai" for m in args.modelos):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Falta OPENAI_API_KEY en proyecto/.env.")
    if any(ftf.MODELOS[m]["proveedor"] == "ollama" for m in args.modelos):
        ftf.verificar_ollama()

    evaluacion_indice = evaluacion.set_index("consulta_id", drop=False)
    recuperaciones = {}
    if "rag" in condiciones:
        indice = IndiceRAG(documentos).construir()
        recuperaciones = {
            consulta_id: indice.recuperar(
                evaluacion_indice.loc[consulta_id, "instruction_es"],
                top_k=args.top_k,
            )
            for consulta_id in set(plan["consulta_id"])
        }

    if args.salida.exists():
        filas = pd.read_csv(args.salida).to_dict("records")
        validar_checkpoint(filas, plan, evaluacion)
        print(f"Checkpoint encontrado: {len(filas)} filas.", flush=True)
    else:
        filas = []
    completadas = claves_completadas(filas)
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
        modelo_anterior = modelo

        consulta = evaluacion_indice.loc[unidad["consulta_id"]]
        es_rag = unidad["condicion"] == "rag"
        recuperacion = recuperaciones.get(unidad["consulta_id"])
        contexto = recuperacion["contexto"] if es_rag else None
        salida = ftf.generar(modelo, consulta["instruction_es"], contexto=contexto)
        docs = recuperacion["documentos"] if es_rag else []
        latencia_recuperacion = (
            recuperacion["latencia_recuperacion_s"] if es_rag else 0.0
        )
        hit_top1, hit_top_k = metricas_recuperacion(docs, consulta["intent"])
        nuevo = {
            **unidad.to_dict(),
            "tipo": config["tipo"],
            "categoria": _opcional(consulta["categoria"]),
            "intencion": _opcional(consulta["intencion"]),
            "consulta": consulta["texto"],
            "referencia": _opcional(consulta["respuesta_referencia"]),
            "respuesta": salida["respuesta"],
            "documentos_recuperados": json.dumps(
                [doc["document_id"] for doc in docs], ensure_ascii=False
            ),
            "scores_recuperacion": json.dumps(
                [round(doc["score"], 6) for doc in docs]
            ),
            "hit_top1": hit_top1,
            "hit_top_k": hit_top_k,
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
        costos = ftf.costo_consulta(nuevo)
        nuevo.update(costos)
        nuevo["costo_oficial_usd"] = ftf.costo_oficial(nuevo)
        filas = reemplazar_resultado(filas, nuevo)
        _guardar(args.salida, filas)

        if salida["error"]:
            errores_consecutivos += 1
        else:
            errores_consecutivos = 0
            completadas.add(clave)
        print(
            f"{len(completadas):04d}/{len(plan)} — {modelo} — "
            f"{unidad['consulta_id']} — {unidad['condicion']}",
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
