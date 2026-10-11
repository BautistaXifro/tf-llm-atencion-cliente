"""Repite el baseline sin RAG con warm-up y checkpoint comparables al piloto."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402


SALIDA = PROYECTO / "resultados_baseline_controlado.csv"


def _guardar(filas: list[dict]):
    pd.DataFrame(filas).to_csv(SALIDA, index=False)


def main():
    load_dotenv(PROYECTO / ".env")
    ftf.verificar_ollama()
    evaluacion = pd.read_csv(PROYECTO / "data" / "corpus_es_muestra.csv")

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

            salida = ftf.generar(modelo, fila["instruction_es"])
            nueva = {
                "modelo": modelo,
                "tipo": config["tipo"],
                "condicion": "sin_rag_controlado",
                "category": fila["category"],
                "intent": fila["intent"],
                "consulta": fila["instruction_es"],
                "referencia": fila["response_es"],
                "respuesta": salida["respuesta"],
                "latencia_recuperacion_s": 0.0,
                "latencia_generacion_s": salida["latencia_s"],
                "latencia_total_s": salida["latencia_s"],
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
