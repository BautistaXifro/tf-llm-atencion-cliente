import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import evaluacion


def test_normalizar_e_invertir():
    serie = pd.Series([10.0, 20.0, 30.0])
    assert evaluacion.normalizar(serie).tolist() == [0.0, 0.5, 1.0]
    assert evaluacion.normalizar(serie, invertir=True).tolist() == [1.0, 0.5, 0.0]


def test_normalizar_por_grupo_no_mezcla_condiciones():
    datos = pd.DataFrame(
        {
            "condicion": ["rag", "rag", "sin_rag", "sin_rag"],
            "valor": [10.0, 20.0, 100.0, 200.0],
        }
    )
    resultado = evaluacion.normalizar_por_grupo(datos, "valor", "condicion")
    assert resultado.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_bootstrap_estratificado_es_reproducible_y_contiene_media():
    valores = np.array([1.0, 2.0, 10.0, 11.0])
    estratos = np.array(["a", "a", "b", "b"])
    a = evaluacion.intervalo_bootstrap_media(
        valores, estratos, repeticiones=500, semilla=7
    )
    b = evaluacion.intervalo_bootstrap_media(
        valores, estratos, repeticiones=500, semilla=7
    )
    assert a == b
    assert a[0] <= valores.mean() <= a[1]


def test_rango_biserial_respeta_direccion_y_empate():
    assert evaluacion.rango_biserial_pareado([1, 2, 3]) == 1.0
    assert evaluacion.rango_biserial_pareado([-1, -2, -3]) == -1.0
    assert evaluacion.rango_biserial_pareado([0, 0]) == 0.0


def test_wilcoxon_todo_cero_es_neutro():
    estadistico, p = evaluacion.prueba_wilcoxon([0, 0, 0])
    assert estadistico == 0.0
    assert p == 1.0


def test_ajuste_holm_es_monotono_en_orden_de_significancia():
    p = np.array([0.01, 0.04, 0.03])
    ajustados = evaluacion.ajuste_holm(p)
    assert np.allclose(ajustados, [0.03, 0.06, 0.06])
    assert np.all(ajustados >= p)
