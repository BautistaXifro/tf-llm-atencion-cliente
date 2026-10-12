"""Genera los CSV reproducibles para dos evaluadores y su clave separada."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.evaluacion_humana import (  # noqa: E402
    SEMILLAS_EVALUADORES,
    construir_insumos_ciegos,
)


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    resultados = pd.read_csv(PROYECTO / "resultados_experimento_ampliado.csv")
    muestra = pd.read_csv(
        PROYECTO / "data" / "muestra_ampliada_en_evaluacion_humana.csv"
    )
    corpus = pd.read_csv(PROYECTO / "data" / "conocimiento_rag_ampliado.csv")
    formularios, clave = construir_insumos_ciegos(resultados, muestra, corpus)

    destino = PROYECTO / "evaluacion_humana"
    destino.mkdir(exist_ok=True)
    rutas = {}
    for nombre, formulario in formularios.items():
        ruta = destino / f"{nombre}.csv"
        formulario.to_csv(ruta, index=False, encoding="utf-8-sig")
        rutas[nombre] = ruta
    ruta_clave = destino / "clave_ciega.csv"
    clave.to_csv(ruta_clave, index=False, encoding="utf-8-sig")
    rutas["clave"] = ruta_clave

    auditoria = {
        "consultas": 54,
        "respuestas_por_evaluador": 324,
        "evaluadores_previstos": 2,
        "semillas": SEMILLAS_EVALUADORES,
        "archivos": {
            nombre: {"ruta": ruta.name, "sha256": _sha256(ruta)}
            for nombre, ruta in rutas.items()
        },
    }
    ruta_auditoria = destino / "auditoria.json"
    ruta_auditoria.write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(auditoria, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
