"""Analiza dos evaluaciones completas y revela modelo/condición mediante la clave."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.acuerdo_humano import METRICAS, calcular_acuerdo  # noqa: E402


def parse_args():
    base = PROYECTO / "evaluacion_humana"
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluador-1", type=Path, default=base / "evaluador_1.xlsx")
    parser.add_argument("--evaluador-2", type=Path, default=base / "evaluador_2.xlsx")
    parser.add_argument("--clave", type=Path, default=base / "clave_ciega.csv")
    parser.add_argument("--salida-dir", type=Path, default=base / "resultados")
    return parser.parse_args()


def _leer_formulario(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta)
    if ruta.suffix.lower() == ".xlsx":
        return pd.read_excel(ruta, sheet_name="Evaluación")
    raise ValueError(f"Formato no admitido: {ruta.suffix}")


def main():
    args = parse_args()
    evaluador_1 = _leer_formulario(args.evaluador_1)
    evaluador_2 = _leer_formulario(args.evaluador_2)
    pares, acuerdo = calcular_acuerdo(evaluador_1, evaluador_2)

    clave = pd.read_csv(args.clave)
    if clave["respuesta_id"].duplicated().any() or len(clave) != 324:
        raise ValueError("La clave ciega debe contener 324 respuesta_id únicas.")
    respuestas = pares.merge(clave, on="respuesta_id", validate="one_to_one")
    columnas_promedio = [f"{metrica}_promedio" for metrica in METRICAS]
    respuestas["puntaje_calidad_promedio"] = respuestas[
        [f"{metrica}_promedio" for metrica in METRICAS[:-1]]
    ].mean(axis=1)
    resumen = respuestas.groupby(["modelo", "condicion"], as_index=False).agg(
        n=("respuesta_id", "size"),
        correccion=("correccion_1_5_promedio", "mean"),
        utilidad=("utilidad_1_5_promedio", "mean"),
        claridad=("claridad_1_5_promedio", "mean"),
        fidelidad=("fidelidad_contexto_1_5_promedio", "mean"),
        tasa_alucinacion=("alucinacion_0_1_promedio", "mean"),
        calidad_promedio=("puntaje_calidad_promedio", "mean"),
    )

    args.salida_dir.mkdir(parents=True, exist_ok=True)
    acuerdo.to_csv(args.salida_dir / "acuerdo_interevaluador.csv", index=False)
    resumen.to_csv(args.salida_dir / "resumen_evaluacion_humana.csv", index=False)
    respuestas[
        ["respuesta_id", "consulta_id", "modelo", "condicion", *columnas_promedio,
         "puntaje_calidad_promedio"]
    ].to_csv(args.salida_dir / "respuestas_evaluacion_humana.csv", index=False)
    print("Acuerdo interevaluador")
    print(acuerdo.to_string(index=False))
    print("\nResumen por modelo y condición")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
