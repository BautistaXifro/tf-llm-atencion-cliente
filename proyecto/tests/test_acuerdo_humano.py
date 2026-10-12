import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.acuerdo_humano import METRICAS, calcular_acuerdo, kappa_cohen


def _formulario():
    datos = {"respuesta_id": [f"r{i:03d}" for i in range(324)]}
    for metrica in METRICAS[:-1]:
        datos[metrica] = [(i % 5) + 1 for i in range(324)]
    datos[METRICAS[-1]] = [i % 2 for i in range(324)]
    return pd.DataFrame(datos)


def test_kappa_perfecto_es_uno():
    assert kappa_cohen([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], categorias=[1, 2, 3, 4, 5], ponderacion="lineal") == 1.0


def test_acuerdo_perfecto_genera_promedios():
    formulario = _formulario()
    pares, acuerdo = calcular_acuerdo(formulario, formulario.copy())
    assert len(pares) == 324
    assert np.allclose(acuerdo["acuerdo_exacto_pct"], 100.0)
    assert np.allclose(acuerdo["kappa_cohen"], 1.0)


def test_formulario_incompleto_falla_con_mensaje_claro():
    uno = _formulario()
    dos = _formulario()
    dos.loc[0, "claridad_1_5"] = np.nan
    with pytest.raises(ValueError, match="queda.*1 filas sin completar"):
        calcular_acuerdo(uno, dos)
