"""Evalúa el baseline con warm-up y lo compara contra el piloto RAG."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402


def evaluar_baseline() -> pd.DataFrame:
    raw = pd.read_csv(PROYECTO / "resultados_baseline_controlado.csv")
    ok = raw[raw["error"].isna()].copy()
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
    costos = ok.apply(lambda fila: pd.Series(ftf.costo_consulta(fila)), axis=1)
    ok = pd.concat([ok, costos], axis=1)
    ok["costo_oficial_usd"] = ok.apply(ftf.costo_oficial, axis=1)
    ok.to_csv(PROYECTO / "resultados_baseline_controlado_evaluados.csv", index=False)
    return ok


def main():
    baseline = evaluar_baseline()
    rag = pd.read_csv(PROYECTO / "resultados_rag_piloto_evaluados.csv")

    resumen_base = baseline.groupby("modelo").agg(
        bertscore_sin_rag=("bertscore_f1", "mean"),
        rouge_l_sin_rag=("rouge_l", "mean"),
        latencia_sin_rag=("latencia_total_s", "mean"),
        tokens_in_sin_rag=("tokens_in", "sum"),
        tokens_out_sin_rag=("tokens_out", "sum"),
        costo_api_sin_rag=("costo_usd", "sum"),
        costo_oficial_medio_sin_rag=("costo_oficial_usd", "mean"),
    )
    resumen_rag = rag.groupby("modelo").agg(
        bertscore_con_rag=("bertscore_f1", "mean"),
        rouge_l_con_rag=("rouge_l", "mean"),
        latencia_con_rag=("latencia_total_s", "mean"),
        tokens_in_con_rag=("tokens_in", "sum"),
        tokens_out_con_rag=("tokens_out", "sum"),
        costo_api_con_rag=("costo_usd", "sum"),
        costo_oficial_medio_con_rag=("costo_oficial_usd", "mean"),
    )
    comparacion = resumen_base.join(resumen_rag)
    comparacion["delta_bertscore"] = (
        comparacion["bertscore_con_rag"] - comparacion["bertscore_sin_rag"]
    )
    comparacion["delta_rouge_l"] = (
        comparacion["rouge_l_con_rag"] - comparacion["rouge_l_sin_rag"]
    )
    comparacion["delta_latencia_s"] = (
        comparacion["latencia_con_rag"] - comparacion["latencia_sin_rag"]
    )
    comparacion["factor_latencia_rag"] = (
        comparacion["latencia_con_rag"] / comparacion["latencia_sin_rag"]
    )
    comparacion["delta_costo_oficial_medio_usd"] = (
        comparacion["costo_oficial_medio_con_rag"]
        - comparacion["costo_oficial_medio_sin_rag"]
    )
    comparacion.reset_index().to_csv(
        PROYECTO / "comparacion_controlada_rag.csv", index=False
    )

    pares = rag.merge(
        baseline[["modelo", "consulta", "bertscore_f1", "rouge_l"]],
        on=["modelo", "consulta"],
        suffixes=("_rag", "_base"),
    )
    pares["delta_bertscore"] = (
        pares["bertscore_f1_rag"] - pares["bertscore_f1_base"]
    )
    pares["delta_rouge_l"] = pares["rouge_l_rag"] - pares["rouge_l_base"]
    conteos = pares.groupby("modelo").agg(
        consultas_mejoran_bert=(
            "delta_bertscore",
            lambda serie: int((serie > 0).sum()),
        ),
        consultas_empeoran_bert=(
            "delta_bertscore",
            lambda serie: int((serie < 0).sum()),
        ),
        consultas_mejoran_rouge=(
            "delta_rouge_l",
            lambda serie: int((serie > 0).sum()),
        ),
        consultas_empeoran_rouge=(
            "delta_rouge_l",
            lambda serie: int((serie < 0).sum()),
        ),
    )
    conteos.reset_index().to_csv(
        PROYECTO / "conteos_comparacion_controlada.csv", index=False
    )

    print("Comparación controlada")
    print(comparacion.to_string())
    print("\nCambios por consulta")
    print(conteos.to_string())


if __name__ == "__main__":
    main()
