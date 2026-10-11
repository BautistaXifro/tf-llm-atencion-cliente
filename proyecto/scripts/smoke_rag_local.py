"""Prueba mínima RAG con dos consultas y los modelos locales de Ollama."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402
from src.rag import IndiceRAG  # noqa: E402


MODELOS_LOCALES = [
    clave for clave, cfg in ftf.MODELOS.items() if cfg["proveedor"] == "ollama"
]


def main():
    ftf.verificar_ollama()
    conocimiento = pd.read_csv(PROYECTO / "data" / "conocimiento_rag.csv")
    evaluacion = pd.read_csv(PROYECTO / "data" / "corpus_es_muestra.csv").head(2)
    indice = IndiceRAG(conocimiento).construir()
    recuperaciones = {
        idx: indice.recuperar(fila["instruction_es"], top_k=3)
        for idx, fila in evaluacion.iterrows()
    }

    resultados = []
    for modelo in MODELOS_LOCALES:
        calentamiento = ftf.generar(modelo, "Respondé solamente: listo")
        if calentamiento["error"]:
            raise RuntimeError(
                f"Falló el precalentamiento de {modelo}: {calentamiento['error']}"
            )
        for idx, fila in evaluacion.iterrows():
            recuperacion = recuperaciones[idx]
            salida = ftf.generar(
                modelo,
                fila["instruction_es"],
                contexto=recuperacion["contexto"],
            )
            resultados.append(
                {
                    "modelo": modelo,
                    "consulta": fila["instruction_es"],
                    "documentos": [
                        doc["document_id"] for doc in recuperacion["documentos"]
                    ],
                    "latencia_recuperacion_s": recuperacion[
                        "latencia_recuperacion_s"
                    ],
                    "latencia_generacion_s": salida["latencia_s"],
                    "error": salida["error"],
                    "respuesta": salida["respuesta"],
                }
            )

    contenido = json.dumps(resultados, ensure_ascii=False, indent=2)
    salida = PROYECTO / "resultados_smoke_rag_local.json"
    salida.write_text(contenido + "\n", encoding="utf-8")
    print(contenido)
    print(f"Resultado: {salida}")
    if any(fila["error"] for fila in resultados):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
