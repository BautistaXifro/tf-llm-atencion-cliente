"""Dashboard interactivo de los resultados del experimento ampliado."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.acuerdo_humano import ARCHIVO_ACUERDO, ARCHIVO_RESUMEN
from src.seleccion_modelo import construir_guia_seleccion


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

st.subheader("Guía de selección según prioridad")
guia = construir_guia_seleccion(summary)
prioridad = st.selectbox("Prioridad operativa", guia["prioridad"].tolist())
recomendacion = guia.loc[guia["prioridad"].eq(prioridad)].iloc[0]
st.markdown(f"**Modelo sugerido:** {recomendacion['modelo_recomendado']}")
st.write(recomendacion["evidencia"])
st.caption(recomendacion["alcance"])
st.dataframe(guia, hide_index=True, use_container_width=True)
st.caption(
    "La guía resume el experimento y no establece un ganador universal. La "
    "decisión final debe considerar privacidad, infraestructura, volumen y "
    "resultados de la evaluación humana."
)

st.subheader("Recuperación por intención")
retrieval_view = retrieval.set_index("intent_esperado")[["hit_top1", "hit_top_k"]].rename(
    columns={"hit_top1": "Hit@1", "hit_top_k": "Hit@3"}
)
st.bar_chart(retrieval_view, height=480)

with st.expander("Ver cambios medios producidos por RAG"):
    st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader("Evaluación humana")
human_results = ROOT / "evaluacion_humana" / "resultados"
human_summary_path = human_results / ARCHIVO_RESUMEN
human_agreement_path = human_results / ARCHIVO_ACUERDO

if human_summary_path.exists() and human_agreement_path.exists():
    human_summary = pd.read_csv(human_summary_path)
    human_agreement = pd.read_csv(human_agreement_path)
    human_summary["Modelo"] = human_summary["modelo"].map(MODEL_LABELS)
    human_summary["Condición"] = human_summary["condicion"].map(CONDITION_LABELS)
    st.success("Resultados humanos completos e integrados.")
    st.dataframe(human_summary, hide_index=True, use_container_width=True)
    human_left, human_right = st.columns(2)
    with human_left:
        st.caption("Calidad humana promedio (1–5)")
        st.bar_chart(
            human_summary.pivot(
                index="Modelo", columns="Condición", values="calidad_promedio"
            ),
            height=320,
        )
    with human_right:
        st.caption("Tasa de alucinación (0–1)")
        st.bar_chart(
            human_summary.pivot(
                index="Modelo", columns="Condición", values="tasa_alucinacion"
            ),
            height=320,
        )
    with st.expander("Ver acuerdo entre evaluadores"):
        st.dataframe(human_agreement, hide_index=True, use_container_width=True)
else:
    st.info(
        "La evaluación humana continúa pendiente. Cuando se completen y analicen "
        "los dos formularios, sus resultados aparecerán aquí automáticamente."
    )
