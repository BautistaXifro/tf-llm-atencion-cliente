"""Calcula calidad, costos y comparación del piloto RAG."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402


def normalizar(serie: pd.Series, invertir: bool = False) -> pd.Series:
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(0.5, index=serie.index)
    normalizada = (serie - serie.min()) / rango
    return 1 - normalizada if invertir else normalizada


def main():
    ruta_raw = PROYECTO / "resultados_rag_piloto.csv"
    ruta_evaluados = PROYECTO / "resultados_rag_piloto_evaluados.csv"
    ruta_resumen = PROYECTO / "resumen_rag_piloto.csv"
    ruta_comparacion = PROYECTO / "comparacion_baseline_rag.csv"

    resultados = pd.read_csv(ruta_raw)
    ok = resultados[resultados["error"].isna()].copy()
    if len(ok) != 60:
        raise RuntimeError(f"Se esperaban 60 resultados válidos y hay {len(ok)}.")

    _, _, f1 = bertscore(
        cands=ok["respuesta"].tolist(),
        refs=ok["referencia"].tolist(),
        lang="es",
        verbose=True,
    )
    ok["bertscore_f1"] = f1.numpy().round(6)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    ok["rouge_l"] = [
        round(scorer.score(ref, cand)["rougeL"].fmeasure, 6)
        for ref, cand in zip(ok["referencia"], ok["respuesta"])
    ]

    def costos_pipeline(fila):
        fila_costo = fila.copy()
        fila_costo["latencia_s"] = fila["latencia_total_s"]
        return pd.Series(ftf.costo_consulta(fila_costo))

    costos = ok.apply(costos_pipeline, axis=1)
    ok = pd.concat([ok, costos], axis=1)
    ok["costo_oficial_usd"] = ok.apply(ftf.costo_oficial, axis=1)
    ok["n_calidad"] = normalizar(ok["bertscore_f1"])
    ok["n_costo"] = normalizar(ok["costo_oficial_usd"], invertir=True)
    ok["n_latencia"] = normalizar(ok["latencia_total_s"], invertir=True)
    ok["indice"] = (
        (ok["n_calidad"] + ok["n_costo"] + ok["n_latencia"]) / 3
    ).round(6)
    ok.to_csv(ruta_evaluados, index=False)

    resumen = ok.groupby("modelo").agg(
        n=("consulta", "count"),
        bertscore_f1=("bertscore_f1", "mean"),
        rouge_l=("rouge_l", "mean"),
        latencia_recuperacion_s=("latencia_recuperacion_s", "mean"),
        latencia_generacion_s=("latencia_generacion_s", "mean"),
        latencia_total_s=("latencia_total_s", "mean"),
        tokens_in=("tokens_in", "sum"),
        tokens_out=("tokens_out", "sum"),
        costo_api_total_usd=("costo_usd", "sum"),
        costo_instancia_medio_usd=("costo_instancia_usd", "mean"),
        costo_oficial_medio_usd=("costo_oficial_usd", "mean"),
        indice=("indice", "mean"),
    ).reset_index()
    resumen.to_csv(ruta_resumen, index=False)

    baseline = pd.read_csv(PROYECTO / "resultados_fundamentos_20260913_1422.csv")
    base_resumen = baseline.groupby("modelo").agg(
        bertscore_sin_rag=("bertscore_f1", "mean"),
        rouge_l_sin_rag=("rouge_l", "mean"),
        latencia_sin_rag=("latencia_s", "mean"),
    )
    rag_resumen = ok.groupby("modelo").agg(
        bertscore_con_rag=("bertscore_f1", "mean"),
        rouge_l_con_rag=("rouge_l", "mean"),
        latencia_con_rag=("latencia_total_s", "mean"),
    )
    comparacion = base_resumen.join(rag_resumen)
    comparacion["delta_bertscore"] = (
        comparacion["bertscore_con_rag"] - comparacion["bertscore_sin_rag"]
    )
    comparacion["delta_rouge_l"] = (
        comparacion["rouge_l_con_rag"] - comparacion["rouge_l_sin_rag"]
    )
    comparacion["delta_latencia_s"] = (
        comparacion["latencia_con_rag"] - comparacion["latencia_sin_rag"]
    )
    comparacion.reset_index().to_csv(ruta_comparacion, index=False)

    print("Resumen RAG")
    print(resumen.to_string(index=False))
    print("\nComparación con baseline")
    print(comparacion.to_string())
    print(f"\nResultados evaluados: {ruta_evaluados}")


if __name__ == "__main__":
    main()
