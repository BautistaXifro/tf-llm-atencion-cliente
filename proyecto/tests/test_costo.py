import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import framework_tf as ftf


def test_costo_electricidad_formula(monkeypatch):
    monkeypatch.setattr(ftf, 'CONSUMO_W', 415)
    monkeypatch.setattr(ftf, 'PRECIO_KWH_USD', 0.1453)
    costo = ftf.costo_electricidad(latencia_s=3600)  # 1 hora exacta
    esperado = (415 / 1000) * (3600 / 3600) * 0.1453
    assert round(costo, 8) == round(esperado, 8)


def test_costo_instancia_formula(monkeypatch):
    monkeypatch.setattr(ftf, 'PRECIO_INSTANCIA_HORA', 0.69)
    costo = ftf.costo_instancia(latencia_s=1.0)  # 3600 consultas/hora
    esperado = 0.69 / 3600
    assert round(costo, 8) == round(esperado, 8)


def test_costo_consulta_propietario():
    fila = {'modelo': 'gpt-4o-mini', 'tokens_in': 1000, 'tokens_out': 500, 'latencia_s': 1.2}
    r = ftf.costo_consulta(fila)
    esperado = 1000 / 1_000_000 * 0.150 + 500 / 1_000_000 * 0.600
    assert round(r['costo_usd'], 8) == round(esperado, 8)
    assert r['costo_electricidad_usd'] is None
    assert r['costo_instancia_usd'] is None


def test_costo_consulta_open_source():
    fila = {'modelo': 'llama-3.1-8b', 'tokens_in': 50, 'tokens_out': 80, 'latencia_s': 2.0}
    r = ftf.costo_consulta(fila)
    assert r['costo_usd'] is None
    assert r['costo_electricidad_usd'] > 0
    assert r['costo_instancia_usd'] > 0


def test_costo_oficial_selecciona_columna_correcta():
    fila_propietario = {'costo_usd': 0.001, 'costo_instancia_usd': None}
    fila_open_source = {'costo_usd': None, 'costo_instancia_usd': 0.0002}
    assert ftf.costo_oficial(fila_propietario) == 0.001
    assert ftf.costo_oficial(fila_open_source) == 0.0002


def test_costo_oficial_con_nan_de_pandas_apply():
    """Regresión: a diferencia de un dict hecho a mano con `None`,
    `pandas.DataFrame.apply(axis=1)` produce filas (`pandas.Series`) donde una
    columna faltante/no aplicable queda como `numpy.nan` (float), no como
    `None`. `is not None` no detecta ese NaN y devolvería el propio NaN en vez
    de caer a `costo_instancia_usd` (bug real encontrado durante la
    integración del Task 6, corregido con `pd.notna()`). Este test falla si
    `costo_oficial()` vuelve a usar `is not None` en lugar de `pd.notna()`."""
    df = pd.DataFrame([
        {'costo_usd': np.nan, 'costo_instancia_usd': 0.0013},
    ])
    resultado = df.apply(ftf.costo_oficial, axis=1)
    assert resultado.iloc[0] == 0.0013
