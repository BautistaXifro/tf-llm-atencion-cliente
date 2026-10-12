"""Construye la muestra ampliada de Bitext sin llamadas generativas.

Uso desde ``proyecto/``:

    .\venv\Scripts\python.exe scripts\construir_muestra_ampliada.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROYECTO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROYECTO / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(PROYECTO))

from datasets import load_dataset  # noqa: E402
from src.muestra import (  # noqa: E402
    SEMILLA_EVALUACION_HUMANA,
    SEMILLA_MUESTRA_AMPLIADA,
    construir_muestra_balanceada,
    seleccionar_submuestra_humana,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--por-intencion", type=int, default=12)
    parser.add_argument("--por-intencion-humana", type=int, default=2)
    parser.add_argument("--semilla", type=int, default=SEMILLA_MUESTRA_AMPLIADA)
    parser.add_argument(
        "--semilla-humana", type=int, default=SEMILLA_EVALUACION_HUMANA
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "data" / "muestra_ampliada_en.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    ).to_pandas()

    muestra, auditoria = construir_muestra_balanceada(
        dataset,
        args.por_intencion,
        columna_estrato="intent",
        columna_texto="instruction",
        semilla=args.semilla,
        prefijo_id="bitext-eval",
    )
    humana = seleccionar_submuestra_humana(
        muestra,
        args.por_intencion_humana,
        columna_estrato="intent",
        semilla=args.semilla_humana,
    )

    columnas = [
        "consulta_id",
        "source_row_id",
        "category",
        "intent",
        "instruction",
        "response",
    ]
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    muestra[columnas].to_csv(args.salida, index=False)

    salida_humana = args.salida.with_name(
        f"{args.salida.stem}_evaluacion_humana.csv"
    )
    humana[["consulta_id", "source_row_id", "category", "intent"]].to_csv(
        salida_humana, index=False
    )

    auditoria.update(
        {
            "filas_evaluacion_humana": len(humana),
            "filas_por_estrato_evaluacion_humana": args.por_intencion_humana,
            "semilla_evaluacion_humana": args.semilla_humana,
        }
    )
    ruta_auditoria = args.salida.with_name(f"{args.salida.stem}_auditoria.json")
    ruta_auditoria.write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(auditoria, ensure_ascii=False, indent=2))
    print(f"Muestra: {args.salida}")
    print(f"Submuestra humana: {salida_humana}")
    print(f"Auditoría: {ruta_auditoria}")


if __name__ == "__main__":
    main()
