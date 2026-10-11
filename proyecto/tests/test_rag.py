import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import rag


def _dataset_fake():
    return pd.DataFrame(
        [
            {"category": "ACCOUNT", "intent": "registro", "instruction": "How do I sign up?", "response": "Use the registration form."},
            {"category": "ACCOUNT", "intent": "registro", "instruction": "How do I sign up!", "response": "Open the registration page."},
            {"category": "ACCOUNT", "intent": "registro", "instruction": "Where can I create an account?", "response": "Create it on the account page."},
            {"category": "ACCOUNT", "intent": "registro", "instruction": "Can I register online?", "response": "Yes, use the online form."},
            {"category": "REFUND", "intent": "reembolso", "instruction": "How can I request a refund?", "response": "Open a refund request."},
            {"category": "REFUND", "intent": "reembolso", "instruction": "Where is the refund form?", "response": "It is in your order."},
            {"category": "REFUND", "intent": "reembolso", "instruction": "I need my money back", "response": "Contact the refund team."},
        ]
    )


def test_normalizar_texto_unifica_mayusculas_puntuacion_y_espacios():
    assert rag.normalizar_texto("  ¿CÓMO   pago? ") == "cómo pago"


def test_construir_corpus_excluye_fugas_y_conserva_trazabilidad():
    evaluacion = pd.DataFrame({"instruction": ["How do I sign up?"]})
    corpus, auditoria = rag.construir_corpus_conocimiento(
        _dataset_fake(),
        evaluacion,
        documentos_por_intencion=2,
        umbral_similitud=0.90,
    )

    assert auditoria["duplicados_internos_eliminados"] == 1
    assert auditoria["coincidencias_exactas_excluidas"] == 1
    assert len(corpus) == 4
    assert corpus["intent"].value_counts().to_dict() == {
        "registro": 2,
        "reembolso": 2,
    }
    assert corpus["document_id"].str.startswith("bitext-").all()
    assert not corpus["instruction_norm"].isin({"how do i sign up"}).any()


def test_construccion_es_reproducible():
    evaluacion = pd.DataFrame({"instruction": ["unrelated evaluation query"]})
    a, _ = rag.construir_corpus_conocimiento(
        _dataset_fake(), evaluacion, documentos_por_intencion=2, semilla=42
    )
    b, _ = rag.construir_corpus_conocimiento(
        _dataset_fake(), evaluacion, documentos_por_intencion=2, semilla=42
    )
    assert a["document_id"].tolist() == b["document_id"].tolist()


def test_construccion_limpia_espacios_finales_multilinea():
    datos = _dataset_fake()
    datos.loc[0, "response"] = "Primera línea.   \nSegunda línea.  "
    corpus, _ = rag.construir_corpus_conocimiento(
        datos,
        pd.DataFrame({"instruction": ["consulta ajena"]}),
        documentos_por_intencion=4,
    )
    respuesta = corpus.loc[corpus.source_row_id == 0, "response"].iloc[0]
    assert respuesta == "Primera línea.\nSegunda línea."


def test_crear_texto_rag_prefiere_espanol_y_tolera_nulos():
    fila_es = pd.Series({
        "category": "ACCOUNT",
        "intent": "registro",
        "instruction": "register",
        "response": "use the form",
        "instruction_es": "registrarme",
        "response_es": "usá el formulario",
    })
    assert "registrarme" in rag.crear_texto_rag(fila_es)
    assert "usá el formulario" in rag.crear_texto_rag(fila_es)

    fila_sin_es = fila_es.copy()
    fila_sin_es["instruction_es"] = pd.NA
    fila_sin_es["response_es"] = pd.NA
    assert "register" in rag.crear_texto_rag(fila_sin_es)


def test_restaurar_marcadores_conserva_tokens_originales():
    original = "Track {{Order Number}} with {{Tracking Number}}"
    traduccion = "Seguí {{Número de Pedido}} con {{Número de Seguimiento}}"
    assert rag.restaurar_marcadores(original, traduccion) == (
        "Seguí {{Order Number}} con {{Tracking Number}}"
    )


def test_restaurar_marcadores_rechaza_cantidad_distinta():
    try:
        rag.restaurar_marcadores("Usá {{PIN}}", "Usá tu PIN")
        assert False, "Debía detectar la eliminación del marcador"
    except ValueError as exc:
        assert "cantidad de marcadores" in str(exc)


class _EmbedderFake:
    def encode(self, textos, **kwargs):
        vectores = []
        for texto in textos:
            texto = texto.lower()
            vectores.append([
                float("refund" in texto or "reembolso" in texto),
                float("register" in texto or "registro" in texto),
            ])
        return np.asarray(vectores, dtype=np.float32)


def test_indice_recupera_documento_relevante_y_formatea_fuente():
    documentos = pd.DataFrame(
        [
            {"document_id": "doc-registro", "texto_rag": "Ayuda para registro"},
            {"document_id": "doc-reembolso", "texto_rag": "Proceso de reembolso"},
        ]
    )
    indice = rag.IndiceRAG(documentos, embedder=_EmbedderFake()).construir()
    salida = indice.recuperar("Necesito un reembolso", top_k=1)

    assert salida["documentos"][0]["document_id"] == "doc-reembolso"
    assert "[Fuente 1: doc-reembolso" in salida["contexto"]
    assert salida["latencia_recuperacion_s"] >= 0


def test_indice_rechaza_top_k_invalido():
    documentos = pd.DataFrame(
        [{"document_id": "doc", "texto_rag": "registro"}]
    )
    indice = rag.IndiceRAG(documentos, embedder=_EmbedderFake())
    try:
        indice.buscar("registro", top_k=0)
        assert False, "Debía rechazar top_k=0"
    except ValueError as exc:
        assert "top_k" in str(exc)
