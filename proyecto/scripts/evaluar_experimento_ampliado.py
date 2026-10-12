"""Calcula métricas, costos y contrastes pareados del Subproyecto 3."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer


PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src import framework_tf as ftf  # noqa: E402
from src.evaluacion import (  # noqa: E402
    ajuste_holm,
    intervalo_bootstrap_media,
    normalizar_por_grupo,
    prueba_wilcoxon,
    rango_biserial_pareado,
)


METRICAS_PAREADAS = (
    "bertscore_f1",
    "rouge_l",
    "latencia_total_s",
    "costo_oficial_usd",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entrada",
        type=Path,
        default=PROYECTO / "resultados_experimento_ampliado.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROYECTO / "resultados_experimento_ampliado_evaluados.csv",
    )
    return parser.parse_args()


def _validar(resultados: pd.DataFrame):
    requeridas = {
        "modelo",
        "consulta_id",
        "condicion",
        "respuesta",
        "referencia",
        "latencia_total_s",
        "tokens_in",
        "tokens_out",
        "error",
        "intent",
    }
    faltantes = requeridas - set(resultados.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if resultados["error"].notna().any() or resultados["respuesta"].isna().any():
        raise RuntimeError("La entrada contiene errores o respuestas vacías.")
    if resultados.duplicated(["modelo", "consulta_id", "condicion"]).any():
        raise RuntimeError("La entrada contiene claves experimentales duplicadas.")
    conteos = resultados.groupby(["modelo", "condicion"]).size()
    if len(conteos) != 6 or not (conteos == 324).all():
        raise RuntimeError(f"Se esperaban seis grupos de 324 filas: {conteos.to_dict()}")


def _calcular_metricas(resultados: pd.DataFrame) -> pd.DataFrame:
    evaluados = resultados.copy()
    _, _, f1 = bertscore(
        cands=evaluados["respuesta"].tolist(),
        refs=evaluados["referencia"].tolist(),
        lang="es",
        batch_size=32,
        verbose=True,
    )
    evaluados["bertscore_f1"] = f1.cpu().numpy().round(6)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    evaluados["rouge_l"] = [
        round(scorer.score(ref, cand)["rougeL"].fmeasure, 6)
        for ref, cand in zip(evaluados["referencia"], evaluados["respuesta"])
    ]

    return evaluados


def _agregar_derivadas(evaluados: pd.DataFrame) -> pd.DataFrame:
    """Recalcula costos e índice dentro de cada condición experimental."""
    evaluados = evaluados.drop(
        columns=[
            "costo_usd",
            "costo_electricidad_usd",
            "costo_instancia_usd",
            "costo_oficial_usd",
            "n_calidad",
            "n_costo",
            "n_latencia",
            "indice",
        ],
        errors="ignore",
    ).copy()

    def costos_pipeline(fila):
        fila_costo = fila.copy()
        fila_costo["latencia_s"] = fila["latencia_total_s"]
        return pd.Series(ftf.costo_consulta(fila_costo))

    costos = evaluados.apply(costos_pipeline, axis=1)
    evaluados = pd.concat([evaluados, costos], axis=1)
    evaluados["costo_oficial_usd"] = evaluados.apply(ftf.costo_oficial, axis=1)
    evaluados["n_calidad"] = normalizar_por_grupo(
        evaluados, "bertscore_f1", "condicion"
    )
    evaluados["n_costo"] = normalizar_por_grupo(
        evaluados, "costo_oficial_usd", "condicion", invertir=True
    )
    evaluados["n_latencia"] = normalizar_por_grupo(
        evaluados, "latencia_total_s", "condicion", invertir=True
    )
    evaluados["indice"] = (
        (evaluados["n_calidad"] + evaluados["n_costo"] + evaluados["n_latencia"])
        / 3
    ).round(6)
    return evaluados


def _resumen(evaluados: pd.DataFrame) -> pd.DataFrame:
    return evaluados.groupby(["modelo", "condicion"], as_index=False).agg(
        n=("consulta_id", "size"),
        bertscore_f1=("bertscore_f1", "mean"),
        rouge_l=("rouge_l", "mean"),
        latencia_media_s=("latencia_total_s", "mean"),
        latencia_mediana_s=("latencia_total_s", "median"),
        latencia_p95_s=("latencia_total_s", lambda x: x.quantile(0.95)),
        tokens_in=("tokens_in", "sum"),
        tokens_out=("tokens_out", "sum"),
        costo_api_total_usd=("costo_usd", lambda x: x.sum(min_count=1)),
        costo_instancia_total_usd=(
            "costo_instancia_usd",
            lambda x: x.sum(min_count=1),
        ),
        costo_oficial_medio_usd=("costo_oficial_usd", "mean"),
        indice=("indice", "mean"),
    )


def _deltas_y_contrastes(evaluados: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = evaluados.loc[evaluados["condicion"].eq("sin_rag")]
    rag = evaluados.loc[evaluados["condicion"].eq("rag")]
    columnas = [
        "modelo",
        "consulta_id",
        "intent",
        *METRICAS_PAREADAS,
    ]
    pares = rag[columnas].merge(
        base[columnas],
        on=["modelo", "consulta_id", "intent"],
        suffixes=("_rag", "_base"),
        validate="one_to_one",
    )
    for metrica in METRICAS_PAREADAS:
        pares[f"delta_{metrica}"] = (
            pares[f"{metrica}_rag"] - pares[f"{metrica}_base"]
        )

    filas = []
    for modelo, grupo in pares.groupby("modelo", sort=True):
        if len(grupo) != 324:
            raise RuntimeError(f"{modelo} no contiene 324 pares.")
        for metrica in METRICAS_PAREADAS:
            delta = grupo[f"delta_{metrica}"].to_numpy()
            inferior, superior = intervalo_bootstrap_media(
                delta,
                grupo["intent"].to_numpy(),
            )
            estadistico, p_valor = prueba_wilcoxon(delta)
            base_media = float(grupo[f"{metrica}_base"].mean())
            filas.append(
                {
                    "modelo": modelo,
                    "metrica": metrica,
                    "n": len(grupo),
                    "media_base": base_media,
                    "media_rag": float(grupo[f"{metrica}_rag"].mean()),
                    "delta_media": float(delta.mean()),
                    "delta_mediana": float(np.median(delta)),
                    "ic95_inferior": inferior,
                    "ic95_superior": superior,
                    "cambio_relativo_pct": (
                        float(delta.mean() / base_media * 100)
                        if base_media != 0
                        else np.nan
                    ),
                    "mejoran": int((delta > 0).sum()),
                    "empeoran": int((delta < 0).sum()),
                    "empatan": int((delta == 0).sum()),
                    "wilcoxon_w": estadistico,
                    "p_bilateral": p_valor,
                    "rango_biserial": rango_biserial_pareado(delta),
                }
            )
    contrastes = pd.DataFrame(filas)
    contrastes["p_holm"] = np.nan
    for _, indices in contrastes.groupby("metrica").groups.items():
        contrastes.loc[indices, "p_holm"] = ajuste_holm(
            contrastes.loc[indices, "p_bilateral"].to_numpy()
        )
    contrastes["significativo_005"] = contrastes["p_holm"] < 0.05
    return pares, contrastes


def _contraste_hipotesis_rag(evaluados: pd.DataFrame) -> pd.DataFrame:
    """Compara cada modelo abierto con GPT usando solo pares con RAG."""
    rag = evaluados.loc[evaluados["condicion"].eq("rag")]
    gpt = rag.loc[
        rag["modelo"].eq("gpt-4o-mini"),
        [
            "consulta_id",
            "intent",
            "bertscore_f1",
            "costo_oficial_usd",
            "indice",
        ],
    ]
    if len(gpt) != 324:
        raise RuntimeError("La condición RAG de GPT no contiene 324 consultas.")

    filas = []
    for modelo in ("llama-3.1-8b", "mistral"):
        abierto = rag.loc[
            rag["modelo"].eq(modelo),
            [
                "consulta_id",
                "intent",
                "bertscore_f1",
                "costo_oficial_usd",
                "indice",
            ],
        ]
        pares = abierto.merge(
            gpt,
            on=["consulta_id", "intent"],
            suffixes=("_abierto", "_gpt"),
            validate="one_to_one",
        )
        if len(pares) != 324:
            raise RuntimeError(f"{modelo} no contiene 324 pares RAG con GPT.")
        delta_bert = (
            pares["bertscore_f1_abierto"] - pares["bertscore_f1_gpt"]
        ).to_numpy()
        inferior, superior = intervalo_bootstrap_media(
            delta_bert,
            pares["intent"].to_numpy(),
        )
        diferencia = float(delta_bert.mean())
        costo_abierto = float(pares["costo_oficial_usd_abierto"].mean())
        costo_gpt = float(pares["costo_oficial_usd_gpt"].mean())
        indice_abierto = float(pares["indice_abierto"].mean())
        indice_gpt = float(pares["indice_gpt"].mean())
        punto_en_margen = abs(diferencia) < 0.02
        ic_en_margen = inferior > -0.02 and superior < 0.02
        costo_menor = costo_abierto < costo_gpt
        indice_superior = indice_abierto > indice_gpt
        filas.append(
            {
                "modelo_abierto": modelo,
                "n_pares": len(pares),
                "bertscore_rag_abierto": float(
                    pares["bertscore_f1_abierto"].mean()
                ),
                "bertscore_rag_gpt": float(pares["bertscore_f1_gpt"].mean()),
                "diferencia_bert_abierto_menos_gpt": diferencia,
                "ic95_diferencia_inferior": inferior,
                "ic95_diferencia_superior": superior,
                "margen_referencia_h2": 0.02,
                "diferencia_puntual_dentro_margen_002": punto_en_margen,
                "ic95_completo_dentro_margen_002": ic_en_margen,
                "costo_rag_1000_abierto_usd": costo_abierto * 1000,
                "costo_rag_1000_gpt_usd": costo_gpt * 1000,
                "ratio_costo_abierto_sobre_gpt": costo_abierto / costo_gpt,
                "costo_abierto_menor_que_gpt": costo_menor,
                "indice_rag_abierto": indice_abierto,
                "indice_rag_gpt": indice_gpt,
                "indice_abierto_supera_gpt": indice_superior,
                "apoyo_h1_con_margen_002": ic_en_margen,
                "h2_apoyada": costo_menor and punto_en_margen,
                "h3_apoyada": indice_superior,
            }
        )
    return pd.DataFrame(filas)


def main():
    args = parse_args()
    ruta_resumen = args.salida.with_name("resumen_experimento_ampliado.csv")
    ruta_pares = args.salida.with_name("deltas_experimento_ampliado.csv")
    ruta_contrastes = args.salida.with_name("comparacion_pareada_ampliada.csv")
    ruta_intenciones = args.salida.with_name(
        "resumen_intenciones_experimento_ampliado.csv"
    )
    ruta_hipotesis = args.salida.with_name("contraste_hipotesis_rag.csv")

    if args.salida.exists():
        evaluados = pd.read_csv(args.salida)
        _validar(evaluados)
        columnas_metricas = {"bertscore_f1", "rouge_l", "costo_oficial_usd"}
        if not columnas_metricas.issubset(evaluados.columns):
            raise RuntimeError("La salida existente no contiene todas las métricas.")
        print(f"Métricas existentes reutilizadas: {args.salida}", flush=True)
    else:
        resultados = pd.read_csv(args.entrada)
        _validar(resultados)
        evaluados = _calcular_metricas(resultados)

    # Las métricas costosas se reutilizan, pero las derivadas se recalculan para
    # que la normalización del índice nunca mezcle RAG con la línea base.
    evaluados = _agregar_derivadas(evaluados)
    evaluados.to_csv(args.salida, index=False)

    resumen = _resumen(evaluados)
    pares, contrastes = _deltas_y_contrastes(evaluados)
    hipotesis = _contraste_hipotesis_rag(evaluados)
    resumen_intenciones = evaluados.groupby(
        ["modelo", "condicion", "intent"], as_index=False
    ).agg(
        n=("consulta_id", "size"),
        bertscore_f1=("bertscore_f1", "mean"),
        rouge_l=("rouge_l", "mean"),
        latencia_media_s=("latencia_total_s", "mean"),
    )

    resumen.to_csv(ruta_resumen, index=False)
    pares.to_csv(ruta_pares, index=False)
    contrastes.to_csv(ruta_contrastes, index=False)
    resumen_intenciones.to_csv(ruta_intenciones, index=False)
    hipotesis.to_csv(ruta_hipotesis, index=False)

    print("\nResumen por modelo y condición")
    print(resumen.to_string(index=False))
    print("\nContrastes pareados")
    print(contrastes.to_string(index=False))
    print("\nContraste de hipótesis con RAG")
    print(hipotesis.to_string(index=False))
    print(f"\nResultados evaluados: {args.salida}")


if __name__ == "__main__":
    main()
