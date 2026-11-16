"""Analiza dos evaluaciones completas y revela modelo/condición mediante la clave."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.acuerdo_humano import (  # noqa: E402
    ARCHIVO_ACUERDO,
    ARCHIVO_EFECTO_RAG,
    ARCHIVO_RESPUESTAS,
    ARCHIVO_RESUMEN,
    ARCHIVO_RESUMEN_POR_EVALUADOR,
    METRICAS,
    calcular_acuerdo,
)


def parse_args():
    base = PROYECTO / "evaluacion_humana"
    evaluador_1 = base / "evaluador_1_completado.xlsx"
    evaluador_2 = base / "evaluador_2_completado.xlsx"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluador-1",
        type=Path,
        default=evaluador_1 if evaluador_1.exists() else base / "evaluador_1.xlsx",
    )
    parser.add_argument(
        "--evaluador-2",
        type=Path,
        default=evaluador_2 if evaluador_2.exists() else base / "evaluador_2.xlsx",
    )
    parser.add_argument("--clave", type=Path, default=base / "clave_ciega.csv")
    parser.add_argument("--salida-dir", type=Path, default=base / "resultados")
    return parser.parse_args()


def _leer_formulario(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() == ".csv":
        return pd.read_csv(ruta)
    if ruta.suffix.lower() == ".xlsx":
        return pd.read_excel(ruta, sheet_name="Evaluación")
    raise ValueError(f"Formato no admitido: {ruta.suffix}")


def _resumir_por_evaluador(respuestas: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for evaluador in ("evaluador_1", "evaluador_2"):
        calidad = [f"{metrica}_{evaluador}" for metrica in METRICAS[:-1]]
        temporal = respuestas.assign(
            evaluador=evaluador,
            puntaje_calidad=respuestas[calidad].mean(axis=1),
        )
        agrupado = temporal.groupby(
            ["evaluador", "modelo", "condicion"], as_index=False
        ).agg(
            n=("respuesta_id", "size"),
            correccion=(f"correccion_1_5_{evaluador}", "mean"),
            utilidad=(f"utilidad_1_5_{evaluador}", "mean"),
            claridad=(f"claridad_1_5_{evaluador}", "mean"),
            fidelidad=(f"fidelidad_contexto_1_5_{evaluador}", "mean"),
            tasa_alucinacion=(f"alucinacion_0_1_{evaluador}", "mean"),
            calidad_promedio=("puntaje_calidad", "mean"),
        )
        filas.append(agrupado)
    return pd.concat(filas, ignore_index=True)


def _calcular_efecto_rag(resumen: pd.DataFrame) -> pd.DataFrame:
    metricas = [
        "correccion",
        "utilidad",
        "claridad",
        "fidelidad",
        "tasa_alucinacion",
        "calidad_promedio",
    ]
    base = resumen.loc[resumen["condicion"] == "sin_rag"].set_index("modelo")
    rag = resumen.loc[resumen["condicion"] == "rag"].set_index("modelo")
    filas = []
    for modelo in sorted(set(base.index) & set(rag.index)):
        fila = {"modelo": modelo}
        for metrica in metricas:
            fila[f"{metrica}_sin_rag"] = float(base.loc[modelo, metrica])
            fila[f"{metrica}_rag"] = float(rag.loc[modelo, metrica])
            fila[f"delta_{metrica}"] = float(
                rag.loc[modelo, metrica] - base.loc[modelo, metrica]
            )
        filas.append(fila)
    return pd.DataFrame(filas)


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
    resumen_por_evaluador = _resumir_por_evaluador(respuestas)
    efecto_rag = _calcular_efecto_rag(resumen)

    args.salida_dir.mkdir(parents=True, exist_ok=True)
    acuerdo.to_csv(args.salida_dir / ARCHIVO_ACUERDO, index=False)
    resumen.to_csv(args.salida_dir / ARCHIVO_RESUMEN, index=False)
    resumen_por_evaluador.to_csv(
        args.salida_dir / ARCHIVO_RESUMEN_POR_EVALUADOR, index=False
    )
    efecto_rag.to_csv(args.salida_dir / ARCHIVO_EFECTO_RAG, index=False)
    respuestas[
        ["respuesta_id", "consulta_id", "modelo", "condicion", *columnas_promedio,
         "puntaje_calidad_promedio"]
    ].to_csv(args.salida_dir / ARCHIVO_RESPUESTAS, index=False)
    print("Acuerdo interevaluador")
    print(acuerdo.to_string(index=False))
    print("\nResumen por modelo y condición")
    print(resumen.to_string(index=False))
    print("\nEfecto descriptivo de RAG")
    print(
        efecto_rag[
            ["modelo", "delta_calidad_promedio", "delta_tasa_alucinacion"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
