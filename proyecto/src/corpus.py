"""Muestreo estratificado y traducción del corpus — Sub-proyecto 1 (Fundamentos)."""

import os
import pandas as pd

SEMILLA = 42

PROMPT_TRADUCTOR = (
    'Traducí el siguiente texto de inglés a español rioplatense neutro, '
    'preservando el tono y el registro. Devolvé únicamente la traducción, '
    'sin comentarios adicionales.'
)


def muestreo_estratificado(df_full, n_muestra, semilla=SEMILLA):
    """Al menos 1 registro por categoría, completando al azar hasta `n_muestra`."""
    # Use loop approach to handle pandas 3.0 groupby behavior
    muestra_list = []
    original_indices = []
    for category, group in df_full.groupby('category'):
        sampled = group.sample(1, random_state=semilla)
        muestra_list.append(sampled)
        # Track original df_full indices BEFORE concat resets them
        original_indices.extend(sampled.index.tolist())

    muestra = pd.concat(muestra_list, ignore_index=True)

    if len(muestra) < n_muestra:
        # Use original_indices (not muestra.index) to find untouched rows
        restantes = df_full[~df_full.index.isin(original_indices)]
        extra = restantes.sample(n_muestra - len(muestra), random_state=semilla)
        muestra = pd.concat([muestra, extra], ignore_index=True)

    return muestra.head(n_muestra).reset_index(drop=True)


def _traducir_texto(cliente, texto):
    r = cliente.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': PROMPT_TRADUCTOR},
            {'role': 'user', 'content': texto},
        ],
        temperature=0.0,
    )
    return r.choices[0].message.content.strip()


def traducir_muestra(cliente, muestra):
    """Agrega `instruction_es` y `response_es` sin tocar las columnas originales."""
    muestra = muestra.copy()
    muestra['instruction_es'] = muestra['instruction'].apply(lambda t: _traducir_texto(cliente, t))
    muestra['response_es'] = muestra['response'].apply(lambda t: _traducir_texto(cliente, t))
    return muestra


def cargar_o_generar_corpus_es(path, df_full, n_muestra, cliente, semilla=SEMILLA):
    """Reutiliza el CSV persistido si ya existe; si no, muestrea, traduce y guarda."""
    if os.path.exists(path):
        return pd.read_csv(path)

    muestra = muestreo_estratificado(df_full, n_muestra, semilla)
    muestra_es = traducir_muestra(cliente, muestra)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    muestra_es.to_csv(path, index=False)
    return muestra_es
