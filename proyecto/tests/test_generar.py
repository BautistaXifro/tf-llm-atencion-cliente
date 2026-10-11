import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import framework_tf as ftf


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeUsage:
    def __init__(self, tin, tout):
        self.prompt_tokens = tin
        self.completion_tokens = tout


class _FakeOpenAIResponse:
    def __init__(self, content, tin, tout):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(tin, tout)


class _FakeOpenAIClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return _FakeOpenAIResponse(' Hola, ¿en qué puedo ayudarte? ', 12, 8)


class _FakeOpenAIClientCaptura:
    mensajes = None

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                _FakeOpenAIClientCaptura.mensajes = kwargs["messages"]
                return _FakeOpenAIResponse("Respuesta", 20, 5)


class _FakeOllamaClient:
    def chat(self, **kwargs):
        return {
            'message': {'content': ' Claro, te ayudo. '},
            'total_duration': 2_500_000_000,  # 2.5 s en nanosegundos
            'prompt_eval_count': 30,
            'eval_count': 40,
        }


def test_generar_openai_dispatch(monkeypatch):
    monkeypatch.setitem(ftf._clientes, 'openai', _FakeOpenAIClient())
    out = ftf.generar('gpt-4o-mini', 'hola')
    assert out['error'] is None
    assert out['respuesta'] == 'Hola, ¿en qué puedo ayudarte?'
    assert out['tokens_in'] == 12
    assert out['tokens_out'] == 8


def test_generar_ollama_dispatch(monkeypatch):
    monkeypatch.setitem(ftf._clientes, 'ollama', _FakeOllamaClient())
    out = ftf.generar('llama-3.1-8b', 'hola')
    assert out['error'] is None
    assert out['respuesta'] == 'Claro, te ayudo.'
    assert out['latencia_s'] == 2.5
    assert out['tokens_in'] == 30
    assert out['tokens_out'] == 40


def test_generar_modelo_desconocido():
    try:
        ftf.generar('modelo-inexistente', 'hola')
        assert False, 'Debería haber lanzado KeyError'
    except KeyError:
        pass


def test_generar_captura_error_del_proveedor(monkeypatch):
    class _ClienteQueFalla:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionError('timeout simulado')

    monkeypatch.setitem(ftf._clientes, 'openai', _ClienteQueFalla())
    out = ftf.generar('gpt-4o-mini', 'hola')
    assert out['error'] is not None
    assert 'timeout simulado' in out['error']
    assert out['respuesta'] is None


def test_generar_con_rag_incluye_reglas_contexto_y_consulta(monkeypatch):
    monkeypatch.setitem(ftf._clientes, 'openai', _FakeOpenAIClientCaptura())
    out = ftf.generar('gpt-4o-mini', '¿Cómo pago?', contexto='Aceptamos tarjeta.')

    assert out['error'] is None
    mensaje = _FakeOpenAIClientCaptura.mensajes[-1]['content']
    assert ftf.INSTRUCCION_RAG in mensaje
    assert 'Aceptamos tarjeta.' in mensaje
    assert '¿Cómo pago?' in mensaje


def test_limpiar_respuesta_quita_espacios_finales_por_linea():
    assert ftf._limpiar_respuesta('  primera  \nsegunda   \n') == (
        'primera\nsegunda'
    )
