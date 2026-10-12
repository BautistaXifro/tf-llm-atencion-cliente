"""Dashboard interactivo de los resultados del experimento ampliado."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o mini",
    "llama-3.1-8b": "LLaMA 3.1 8B",
    "mistral": "Mistral",
}
CONDITION_LABELS = {"sin_rag": "Sin RAG", "rag": "Con RAG"}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ROOT / "resumen_experimento_ampliado.csv")
    comparison = pd.read_csv(ROOT / "comparacion_pareada_ampliada.csv")
    retrieval = pd.read_csv(ROOT / "resultados_recuperacion_rag_ampliada_por_intencion.csv")
    summary["Modelo"] = summary["modelo"].map(MODEL_LABELS)
    summary["Condición"] = summary["condicion"].map(CONDITION_LABELS)
    summary["Costo por 1.000 consultas (USD)"] = summary["costo_oficial_medio_usd"] * 1000
    return summary, comparison, retrieval


st.set_page_config(page_title="Evaluación de LLM para atención al cliente", layout="wide")
st.title("Evaluación de modelos de lenguaje para atención al cliente")
st.caption("Experimento ampliado: 324 consultas, 27 intenciones y 3 modelos evaluados con y sin RAG")

summary, comparison, retrieval = load_data()
rag = summary.loc[summary["condicion"] == "rag"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Consultas", "324", "12 por intención")
col2.metric("Respuestas", "1.944", "3 modelos × 2 condiciones")
col3.metric("Hit@1", f"{retrieval['hit_top1'].mean() * 100:.1f} %")
col4.metric("Hit@3", f"{retrieval['hit_top_k'].mean() * 100:.1f} %")

st.subheader("Comparación de calidad")
metric = st.radio(
    "Métrica",
    options=["bertscore_f1", "rouge_l"],
    format_func=lambda value: "BERTScore F1" if value == "bertscore_f1" else "ROUGE-L",
    horizontal=True,
)
quality = summary.pivot(index="Modelo", columns="Condición", values=metric)
st.bar_chart(quality, height=360)

left, right = st.columns(2)
with left:
    st.subheader("Latencia media")
    latency = summary.pivot(index="Modelo", columns="Condición", values="latencia_media_s")
    st.bar_chart(latency, height=320)
with right:
    st.subheader("Costo con RAG")
    st.bar_chart(rag.set_index("Modelo")[["Costo por 1.000 consultas (USD)"]], height=320)

st.subheader("Resultados con RAG")
st.dataframe(
    rag[
        [
            "Modelo",
            "bertscore_f1",
            "rouge_l",
            "latencia_media_s",
            "latencia_p95_s",
            "Costo por 1.000 consultas (USD)",
            "indice",
        ]
    ].rename(
        columns={
            "bertscore_f1": "BERTScore F1",
            "rouge_l": "ROUGE-L",
            "latencia_media_s": "Latencia media (s)",
            "latencia_p95_s": "Latencia p95 (s)",
            "indice": "Índice compuesto",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Recuperación por intención")
retrieval_view = retrieval.set_index("intent_esperado")[["hit_top1", "hit_top_k"]].rename(
    columns={"hit_top1": "Hit@1", "hit_top_k": "Hit@3"}
)
st.bar_chart(retrieval_view, height=480)

with st.expander("Ver cambios medios producidos por RAG"):
    st.dataframe(comparison, hide_index=True, use_container_width=True)
