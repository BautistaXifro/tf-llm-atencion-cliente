"""Genera figuras reproducibles a partir de los resultados del experimento ampliado.

Las imágenes producidas por este script se utilizan tanto en el dashboard como en
el Trabajo Final. No se modifican los archivos de resultados originales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figuras"
SUMMARY_PATH = ROOT / "resumen_experimento_ampliado.csv"
RETRIEVAL_PATH = ROOT / "resultados_recuperacion_rag_ampliada_por_intencion.csv"

MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o mini",
    "llama-3.1-8b": "LLaMA 3.1 8B",
    "mistral": "Mistral",
}
CONDITION_LABELS = {"sin_rag": "Sin RAG", "rag": "Con RAG"}
COLORS = {"sin_rag": "#8A8F98", "rag": "#2F6690"}


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    retrieval = pd.read_csv(RETRIEVAL_PATH)
    summary["modelo_etiqueta"] = summary["modelo"].map(MODEL_LABELS)
    summary["condicion_etiqueta"] = summary["condicion"].map(CONDITION_LABELS)
    summary["costo_1000_usd"] = summary["costo_oficial_medio_usd"] * 1000
    return summary, retrieval


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def quality_figure(summary: pd.DataFrame) -> Path:
    models = ["gpt-4o-mini", "llama-3.1-8b", "mistral"]
    conditions = ["sin_rag", "rag"]
    x = np.arange(len(models))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["bertscore_f1", "rouge_l"],
        ["BERTScore F1 medio", "ROUGE-L medio"],
        strict=True,
    ):
        for offset, condition in zip([-width / 2, width / 2], conditions, strict=True):
            values = [
                summary.loc[
                    (summary["modelo"] == model) & (summary["condicion"] == condition),
                    metric,
                ].iloc[0]
                for model in models
            ]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=CONDITION_LABELS[condition],
                color=COLORS[condition],
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
        ax.set_ylim(0, 0.84 if metric == "bertscore_f1" else 0.38)
        _style_axis(ax)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(
        "Calidad automática por modelo y condición (n = 324 consultas)",
        fontsize=13,
        weight="bold",
    )
    return _save(fig, "calidad_modelos_condicion.png")


def efficiency_figure(summary: pd.DataFrame) -> Path:
    rag = summary.loc[summary["condicion"] == "rag"].copy()
    colors = ["#1B4965", "#5FA8D3", "#CA6702"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for (_, row), color in zip(rag.iterrows(), colors, strict=True):
        ax.scatter(
            row["latencia_media_s"],
            row["costo_1000_usd"],
            s=900 * (row["bertscore_f1"] - 0.69),
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{row['modelo_etiqueta']}\nBERTScore {row['bertscore_f1']:.3f}",
            (row["latencia_media_s"], row["costo_1000_usd"]),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Relación entre latencia, costo y calidad con RAG", fontsize=12, weight="bold")
    ax.set_xlabel("Latencia media por respuesta (s)")
    ax.set_ylabel("Costo estimado por 1.000 consultas (USD)")
    ax.set_xlim(max(0, rag["latencia_media_s"].min() - 0.18), rag["latencia_media_s"].max() + 0.28)
    ax.set_ylim(0, rag["costo_1000_usd"].max() + 0.08)
    _style_axis(ax)
    return _save(fig, "eficiencia_modelos_rag.png")


def dashboard_figure(summary: pd.DataFrame, retrieval: pd.DataFrame) -> Path:
    hit1 = retrieval["hit_top1"].mean() * 100
    hit3 = retrieval["hit_top_k"].mean() * 100
    rag = summary.loc[summary["condicion"] == "rag"].set_index("modelo")
    base = summary.loc[summary["condicion"] == "sin_rag"].set_index("modelo")
    models = ["gpt-4o-mini", "llama-3.1-8b", "mistral"]
    labels = [MODEL_LABELS[m] for m in models]

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[0.72, 1.7, 1.7])
    fig.suptitle("Dashboard del experimento ampliado", fontsize=17, weight="bold")

    cards = [
        ("Consultas", "324", "12 por cada una de las 27 intenciones"),
        ("Respuestas", "1.944", "3 modelos × 2 condiciones"),
        ("Hit@1", f"{hit1:.1f} %", "recuperación correcta en primer lugar"),
        ("Hit@3", f"{hit3:.1f} %", "intención correcta entre los tres primeros"),
    ]
    for idx, (label, value, note) in enumerate(cards):
        ax = fig.add_subplot(grid[0, idx])
        ax.axis("off")
        ax.text(0.02, 0.78, label, fontsize=10, color="#555555", transform=ax.transAxes)
        ax.text(0.02, 0.35, value, fontsize=21, weight="bold", color="#1B4965", transform=ax.transAxes)
        ax.text(0.02, 0.05, note, fontsize=7.8, color="#666666", transform=ax.transAxes)

    ax_quality = fig.add_subplot(grid[1, :2])
    x = np.arange(len(models))
    width = 0.34
    b1 = ax_quality.bar(x - width / 2, base.loc[models, "bertscore_f1"], width, color=COLORS["sin_rag"], label="Sin RAG")
    b2 = ax_quality.bar(x + width / 2, rag.loc[models, "bertscore_f1"], width, color=COLORS["rag"], label="Con RAG")
    ax_quality.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax_quality.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax_quality.set_title("BERTScore F1", fontsize=11, weight="bold")
    ax_quality.set_xticks(x, labels)
    ax_quality.set_ylim(0.66, 0.78)
    ax_quality.legend(frameon=False, fontsize=8)
    _style_axis(ax_quality)

    ax_delta = fig.add_subplot(grid[1, 2:])
    delta_bert = (rag.loc[models, "bertscore_f1"] - base.loc[models, "bertscore_f1"]).to_numpy()
    delta_rouge = (rag.loc[models, "rouge_l"] - base.loc[models, "rouge_l"]).to_numpy()
    d1 = ax_delta.bar(x - width / 2, delta_bert, width, color="#2F6690", label="Δ BERTScore")
    d2 = ax_delta.bar(x + width / 2, delta_rouge, width, color="#6C9A8B", label="Δ ROUGE-L")
    ax_delta.bar_label(d1, fmt="%+.3f", fontsize=8, padding=2)
    ax_delta.bar_label(d2, fmt="%+.3f", fontsize=8, padding=2)
    ax_delta.axhline(0, color="#333333", linewidth=0.8)
    ax_delta.set_title("Cambio medio producido por RAG", fontsize=11, weight="bold")
    ax_delta.set_xticks(x, labels)
    ax_delta.legend(frameon=False, fontsize=8)
    _style_axis(ax_delta)

    ax_latency = fig.add_subplot(grid[2, :2])
    l1 = ax_latency.bar(x - width / 2, base.loc[models, "latencia_media_s"], width, color=COLORS["sin_rag"], label="Sin RAG")
    l2 = ax_latency.bar(x + width / 2, rag.loc[models, "latencia_media_s"], width, color=COLORS["rag"], label="Con RAG")
    ax_latency.bar_label(l1, fmt="%.2f", fontsize=8, padding=2)
    ax_latency.bar_label(l2, fmt="%.2f", fontsize=8, padding=2)
    ax_latency.set_title("Latencia media por respuesta (s)", fontsize=11, weight="bold")
    ax_latency.set_xticks(x, labels)
    ax_latency.set_ylim(0, 1.9)
    ax_latency.legend(frameon=False, fontsize=8)
    _style_axis(ax_latency)

    ax_cost = fig.add_subplot(grid[2, 2:])
    costs = rag.loc[models, "costo_1000_usd"].to_numpy()
    c = ax_cost.bar(x, costs, color=["#1B4965", "#5FA8D3", "#CA6702"])
    ax_cost.bar_label(c, labels=[f"USD {v:.3f}" for v in costs], fontsize=8, padding=2)
    ax_cost.set_title("Costo estimado con RAG por 1.000 consultas", fontsize=11, weight="bold")
    ax_cost.set_xticks(x, labels)
    ax_cost.set_ylim(0, 0.36)
    _style_axis(ax_cost)

    return _save(fig, "dashboard_experimento_ampliado.png")


def main() -> None:
    summary, retrieval = _load_data()
    paths = [
        dashboard_figure(summary, retrieval),
        quality_figure(summary),
        efficiency_figure(summary),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
