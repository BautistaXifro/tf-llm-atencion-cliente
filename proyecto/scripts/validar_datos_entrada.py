"""Valida y normaliza datos externos sin ejecutar modelos ni usar APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.esquema_datos import (  # noqa: E402
    cargar_conocimiento,
    cargar_consultas,
    cargar_mapeo,
    conocimiento_a_formato_rag,
    consultas_a_formato_pipeline,
)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida CSV/JSON y los convierte al esquema de entrada v1."
    )
    parser.add_argument("--tipo", choices=("consultas", "conocimiento"), required=True)
    parser.add_argument("--entrada", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument(
        "--mapeo",
        type=Path,
        help="JSON con pares campo_canónico: columna_de_origen.",
    )
    parser.add_argument(
        "--formato-pipeline",
        action="store_true",
        help="Incluye los alias usados por los scripts experimentales actuales.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    mapeo = cargar_mapeo(args.mapeo)
    if args.tipo == "consultas":
        datos = cargar_consultas(args.entrada, mapeo)
        if args.formato_pipeline:
            datos = consultas_a_formato_pipeline(datos)
    else:
        datos = cargar_conocimiento(args.entrada, mapeo)
        if args.formato_pipeline:
            datos = conocimiento_a_formato_rag(datos)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    if args.salida.suffix.lower() == ".csv":
        salida = datos.copy()
        salida["metadata"] = salida["metadata"].map(
            lambda valor: json.dumps(valor, ensure_ascii=False)
        )
        salida.to_csv(args.salida, index=False, encoding="utf-8")
    elif args.salida.suffix.lower() == ".json":
        args.salida.write_text(
            datos.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        raise ValueError("La salida debe tener extensión .csv o .json.")
    print(f"OK: {len(datos)} filas válidas guardadas en {args.salida}")


if __name__ == "__main__":
    main()
