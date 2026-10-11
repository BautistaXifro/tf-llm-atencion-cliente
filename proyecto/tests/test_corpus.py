import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import corpus


def _df_fake(n_categorias=3, filas_por_categoria=5):
    filas = []
    for c in range(n_categorias):
        for i in range(filas_por_categoria):
            filas.append({
                'category': f'cat_{c}',
                'instruction': f'pregunta {c}-{i}',
                'response': f'respuesta {c}-{i}',
            })
    return pd.DataFrame(filas)


def test_muestreo_estratificado_cubre_todas_las_categorias():
    df = _df_fake(n_categorias=4, filas_por_categoria=5)
    muestra = corpus.muestreo_estratificado(df, n_muestra=4, semilla=42)
    assert len(muestra) == 4
    assert muestra['category'].nunique() == 4


def test_muestreo_estratificado_completa_con_extra_si_faltan():
    df = _df_fake(n_categorias=2, filas_por_categoria=5)
    muestra = corpus.muestreo_estratificado(df, n_muestra=5, semilla=42)
    assert len(muestra) == 5


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeClienteTraductor:
    """Traduce anteponiendo '[ES] ' — alcanza para verificar el flujo sin costo real."""
    class chat:
        class completions:
            @staticmethod
            def create(messages, **kwargs):
                original = messages[-1]['content']
                return _FakeResponse(f'[ES] {original}')


def test_traducir_muestra_agrega_columnas_es():
    muestra = pd.DataFrame({
        'category': ['x'], 'instruction': ['hello'], 'response': ['world'],
    })
    out = corpus.traducir_muestra(_FakeClienteTraductor(), muestra)
    assert out.loc[0, 'instruction_es'] == '[ES] hello'
    assert out.loc[0, 'response_es'] == '[ES] world'
    # las columnas originales se preservan intactas
    assert out.loc[0, 'instruction'] == 'hello'


def test_cargar_o_generar_no_retraduce_si_el_archivo_ya_existe(tmp_path):
    path = tmp_path / 'corpus.csv'
    pd.DataFrame({
        'category': ['x'], 'instruction': ['hola'], 'response': ['chau'],
        'instruction_es': ['hola'], 'response_es': ['chau'],
    }).to_csv(path, index=False)

    llamadas = {'n': 0}

    class _ClienteQueNoDeberiaLlamarse:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    llamadas['n'] += 1
                    raise AssertionError('No debería traducir si el archivo ya existe')

    df = corpus.cargar_o_generar_corpus_es(
        str(path), df_full=None, n_muestra=1,
        cliente=_ClienteQueNoDeberiaLlamarse(),
    )
    assert llamadas['n'] == 0
    assert df.iloc[0]['instruction_es'] == 'hola'


def test_cargar_o_generar_crea_el_archivo_si_no_existe(tmp_path):
    path = tmp_path / 'nuevo' / 'corpus.csv'
    df_full = _df_fake(n_categorias=2, filas_por_categoria=3)

    df = corpus.cargar_o_generar_corpus_es(
        str(path), df_full=df_full, n_muestra=2,
        cliente=_FakeClienteTraductor(),
    )
    assert path.exists()
    assert 'instruction_es' in df.columns
    assert len(df) == 2


def test_muestreo_estratificado_sin_duplicados():
    """Verifica que no hay duplicados cuando se rellenan filas extra."""
    # 3 categorías, 4 filas por categoría = 12 total
    # Muestreo: 1 por categoría = 3, luego 4 extra para completar 7
    df = _df_fake(n_categorias=3, filas_por_categoria=4)
    muestra = corpus.muestreo_estratificado(df, n_muestra=7, semilla=42)

    # Verificar que no hay duplicados por contenido (instruction debe ser único)
    assert len(muestra) == 7
    assert muestra['instruction'].nunique() == 7, \
        f"Found duplicates: {muestra['instruction'].value_counts()}"
