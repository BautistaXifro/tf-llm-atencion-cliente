"""Recuperación de contexto para el Subproyecto 2 (RAG).

El módulo separa dos responsabilidades:

1. construir un corpus de conocimiento sin copiar las consultas de evaluación;
2. recuperar documentos con embeddings multilingües y similitud coseno exacta.

La búsqueda se implementa con NumPy porque el piloto contiene pocas decenas de
documentos. Esto evita agregar un índice aproximado que no aporta ventajas a
esta escala y mantiene el procedimiento fácil de auditar.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import time
import unicodedata

import numpy as np
import pandas as pd


MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENTOS_POR_INTENCION = 3
UMBRAL_SIMILITUD_FUGA = 0.90
SEMILLA = 42
PATRON_MARCADOR = re.compile(r"\{\{[^{}]+\}\}")


def normalizar_texto(texto: str) -> str:
    """Normaliza texto para detectar duplicados de forma reproducible."""
    texto = unicodedata.normalize("NFKC", str(texto)).casefold()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return " ".join(texto.split())


def _similitud_superficial(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _limpiar_espacios_lineas(texto: str) -> str:
    """Quita espacios residuales al final de líneas sin alterar el contenido."""
    return "\n".join(linea.rstrip() for linea in str(texto).splitlines()).strip()


def restaurar_marcadores(original: str, traduccion: str) -> str:
    """Restaura en una traducción los marcadores ``{{...}}`` del original.

    Los reemplaza por posición y falla explícitamente si la traducción agregó o
    eliminó marcadores, porque en ese caso no existe una corrección inequívoca.
    """
    marcadores_originales = PATRON_MARCADOR.findall(str(original))
    marcadores_traducidos = PATRON_MARCADOR.findall(str(traduccion))
    if len(marcadores_originales) != len(marcadores_traducidos):
        raise ValueError(
            "La traducción cambió la cantidad de marcadores: "
            f"{len(marcadores_originales)} -> {len(marcadores_traducidos)}."
        )
    pendientes = iter(marcadores_originales)
    return PATRON_MARCADOR.sub(lambda _: next(pendientes), str(traduccion))


def construir_corpus_conocimiento(
    df_full: pd.DataFrame,
    evaluacion: pd.DataFrame,
    documentos_por_intencion: int = DOCUMENTOS_POR_INTENCION,
    umbral_similitud: float = UMBRAL_SIMILITUD_FUGA,
    semilla: int = SEMILLA,
) -> tuple[pd.DataFrame, dict]:
    """Selecciona documentos Bitext y excluye posibles fugas de evaluación.

    Se eliminan instrucciones duplicadas dentro del dataset, coincidencias
    exactas con la evaluación y variantes cuya similitud superficial alcanza
    ``umbral_similitud``. Después se muestrea la misma cantidad por intención.

    Devuelve ``(corpus, auditoria)``. El corpus conserva el índice original en
    ``source_row_id`` para asegurar trazabilidad hasta el dataset de origen.
    """
    requeridas_full = {"instruction", "response", "category", "intent"}
    faltantes = requeridas_full - set(df_full.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en df_full: {sorted(faltantes)}")
    if "instruction" not in evaluacion.columns:
        raise ValueError("La evaluación debe contener la columna 'instruction'.")
    if documentos_por_intencion < 1:
        raise ValueError("documentos_por_intencion debe ser al menos 1.")
    if not 0 <= umbral_similitud <= 1:
        raise ValueError("umbral_similitud debe estar entre 0 y 1.")

    candidatos = df_full.copy()
    candidatos["source_row_id"] = candidatos.index.astype(int)
    candidatos["instruction_norm"] = candidatos["instruction"].map(normalizar_texto)
    filas_iniciales = len(candidatos)

    candidatos = candidatos.drop_duplicates("instruction_norm", keep="first").copy()
    duplicados_internos = filas_iniciales - len(candidatos)

    evaluacion_norm = {
        normalizar_texto(texto) for texto in evaluacion["instruction"].dropna()
    }
    mascara_exacta = candidatos["instruction_norm"].isin(evaluacion_norm)
    coincidencias_exactas = int(mascara_exacta.sum())
    candidatos = candidatos.loc[~mascara_exacta].copy()

    def similitud_maxima(texto: str) -> float:
        if not evaluacion_norm:
            return 0.0
        return max(_similitud_superficial(texto, q) for q in evaluacion_norm)

    candidatos["similitud_max_evaluacion"] = candidatos["instruction_norm"].map(
        similitud_maxima
    )
    mascara_cercana = candidatos["similitud_max_evaluacion"] >= umbral_similitud
    coincidencias_cercanas = int(mascara_cercana.sum())
    candidatos = candidatos.loc[~mascara_cercana].copy()

    bloques = []
    for _, grupo in candidatos.groupby("intent", sort=True):
        n = min(documentos_por_intencion, len(grupo))
        bloques.append(grupo.sample(n=n, random_state=semilla))
    corpus = pd.concat(bloques, ignore_index=True)
    corpus = corpus.sort_values(["intent", "source_row_id"]).reset_index(drop=True)
    for columna in ("instruction", "response"):
        corpus[columna] = corpus[columna].map(_limpiar_espacios_lineas)
    corpus.insert(
        0,
        "document_id",
        corpus["source_row_id"].map(lambda i: f"bitext-{int(i):05d}"),
    )
    corpus["texto_rag"] = corpus.apply(crear_texto_rag, axis=1)

    auditoria = {
        "filas_dataset": filas_iniciales,
        "duplicados_internos_eliminados": duplicados_internos,
        "coincidencias_exactas_excluidas": coincidencias_exactas,
        "coincidencias_cercanas_excluidas": coincidencias_cercanas,
        "umbral_similitud": umbral_similitud,
        "intenciones_disponibles": int(candidatos["intent"].nunique()),
        "documentos_seleccionados": len(corpus),
        "documentos_por_intencion_objetivo": documentos_por_intencion,
    }
    return corpus, auditoria


def crear_texto_rag(fila: pd.Series) -> str:
    """Construye el fragmento indexable, prefiriendo español si está disponible."""
    pregunta_es = fila.get("instruction_es")
    respuesta_es = fila.get("response_es")
    pregunta = pregunta_es if pd.notna(pregunta_es) and pregunta_es else fila["instruction"]
    respuesta = respuesta_es if pd.notna(respuesta_es) and respuesta_es else fila["response"]
    return (
        f"Categoría: {fila['category']}\n"
        f"Intención: {fila['intent']}\n"
        f"Pregunta frecuente: {pregunta}\n"
        f"Procedimiento recomendado: {respuesta}"
    )


def _normalizar_vectores(vectores: np.ndarray) -> np.ndarray:
    vectores = np.asarray(vectores, dtype=np.float32)
    if vectores.ndim == 1:
        vectores = vectores.reshape(1, -1)
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    if np.any(normas == 0):
        raise ValueError("El modelo produjo al menos un embedding nulo.")
    return vectores / normas


class IndiceRAG:
    """Índice exacto por similitud coseno para el corpus piloto."""

    def __init__(
        self,
        documentos: pd.DataFrame,
        embedder=None,
        modelo: str = MODELO_EMBEDDINGS,
    ):
        requeridas = {"document_id", "texto_rag"}
        faltantes = requeridas - set(documentos.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas en documentos: {sorted(faltantes)}")
        if documentos.empty:
            raise ValueError("No se puede construir un índice sin documentos.")
        self.documentos = documentos.reset_index(drop=True).copy()
        self.modelo = modelo
        self._embedder = embedder
        self._vectores = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.modelo)
        return self._embedder

    def construir(self) -> "IndiceRAG":
        textos = self.documentos["texto_rag"].astype(str).tolist()
        vectores = self.embedder.encode(
            textos,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._vectores = _normalizar_vectores(vectores)
        return self

    def buscar(self, consulta: str, top_k: int = 3) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k debe ser al menos 1.")
        if self._vectores is None:
            self.construir()

        vector_consulta = self.embedder.encode(
            [consulta],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector_consulta = _normalizar_vectores(vector_consulta)[0]
        puntajes = self._vectores @ vector_consulta
        k = min(top_k, len(self.documentos))
        posiciones = np.argsort(-puntajes, kind="stable")[:k]

        resultados = []
        for posicion in posiciones:
            fila = self.documentos.iloc[int(posicion)].to_dict()
            fila["score"] = float(puntajes[int(posicion)])
            resultados.append(fila)
        return resultados

    def recuperar(self, consulta: str, top_k: int = 3) -> dict:
        """Recupera documentos y devuelve contexto más latencia de retrieval."""
        inicio = time.perf_counter()
        documentos = self.buscar(consulta, top_k=top_k)
        latencia = time.perf_counter() - inicio
        return {
            "documentos": documentos,
            "contexto": formatear_contexto(documentos),
            "latencia_recuperacion_s": round(latencia, 6),
        }


def formatear_contexto(documentos: list[dict]) -> str:
    """Convierte resultados recuperados en un contexto trazable para el LLM."""
    bloques = []
    for posicion, doc in enumerate(documentos, start=1):
        bloques.append(
            f"[Fuente {posicion}: {doc['document_id']}; "
            f"similitud={doc['score']:.4f}]\n{doc['texto_rag']}"
        )
    return "\n\n".join(bloques)
