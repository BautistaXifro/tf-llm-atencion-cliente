"""Ejecuta el piloto RAG de 20 consultas × 3 modelos con checkpoint."""

from __future__ import annotations

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
from src.rag import IndiceRAG  # noqa: E402


TOP_K = 3
SALIDA = PROYECTO / "resultados_rag_piloto.csv"


def _guardar(filas: list[dict]):
    pd.DataFrame(filas).to_csv(SALIDA, index=False)


def main():
    load_dotenv(PROYECTO / ".env")
    ftf.verificar_ollama()
    evaluacion = pd.read_csv(PROYECTO / "data" / "corpus_es_muestra.csv")
    conocimiento = pd.read_csv(PROYECTO / "data" / "conocimiento_rag.csv")
    indice = IndiceRAG(conocimiento).construir()

    recuperaciones = {
        idx: indice.recuperar(fila["instruction_es"], top_k=TOP_K)
        for idx, fila in evaluacion.iterrows()
    }
    if SALIDA.exists():
        filas = pd.read_csv(SALIDA).to_dict("records")
        print(f"Checkpoint encontrado: {len(filas)} filas.", flush=True)
    else:
        filas = []

    for modelo, config in ftf.MODELOS.items():
        if config["proveedor"] == "ollama":
            calentamiento = ftf.generar(modelo, "Respondé solamente: listo")
            if calentamiento["error"]:
                raise RuntimeError(
                    f"Falló el precalentamiento de {modelo}: "
                    f"{calentamiento['error']}"
                )
            print(f"{modelo}: precalentado.", flush=True)

        for idx, fila in evaluacion.iterrows():
            completada = any(
                previa.get("modelo") == modelo
                and previa.get("consulta") == fila["instruction_es"]
                and pd.isna(previa.get("error"))
                for previa in filas
            )
            if completada:
                continue

            rec = recuperaciones[idx]
            salida = ftf.generar(
                modelo,
                fila["instruction_es"],
                contexto=rec["contexto"],
            )
            intents = [doc["intent"] for doc in rec["documentos"]]
            nueva = {
                "modelo": modelo,
                "tipo": config["tipo"],
                "condicion": "rag",
                "category": fila["category"],
                "intent": fila["intent"],
                "consulta": fila["instruction_es"],
                "referencia": fila["response_es"],
                "respuesta": salida["respuesta"],
                "documentos_recuperados": json.dumps(
                    [doc["document_id"] for doc in rec["documentos"]]
                ),
                "intents_recuperados": json.dumps(intents),
                "scores_recuperacion": json.dumps(
                    [round(doc["score"], 6) for doc in rec["documentos"]]
                ),
                "hit_top1": intents[0] == fila["intent"],
                "hit_top_k": fila["intent"] in intents,
                "latencia_recuperacion_s": rec["latencia_recuperacion_s"],
                "latencia_generacion_s": salida["latencia_s"],
                "latencia_total_s": round(
                    rec["latencia_recuperacion_s"] + salida["latencia_s"], 6
                ),
                "latencia_s": salida["latencia_s"],
                "tokens_in": salida["tokens_in"],
                "tokens_out": salida["tokens_out"],
                "error": salida["error"],
            }

            filas = [
                previa
                for previa in filas
                if not (
                    previa.get("modelo") == modelo
                    and previa.get("consulta") == fila["instruction_es"]
                )
            ]
            filas.append(nueva)
            _guardar(filas)
            estado = "OK" if salida["error"] is None else salida["error"]
            print(
                f"{modelo} {idx + 1:02d}/{len(evaluacion)} — {estado} — "
                f"{salida['latencia_s']:.3f} s",
                flush=True,
            )
            if config["proveedor"] == "openai":
                time.sleep(0.5)

    resultados = pd.DataFrame(filas)
    errores = int(resultados["error"].notna().sum())
    print(
        f"Finalizado: {len(resultados) - errores}/{len(resultados)} sin error. "
        f"Resultado: {SALIDA}",
        flush=True,
    )
    if errores:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
