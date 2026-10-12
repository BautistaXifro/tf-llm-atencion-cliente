# Evaluación de LLMs para soporte al cliente

Trabajo Final de Juan Bautista Xifro (UAI, 2026). Compara GPT-4o mini, LLaMA 3.1 8B y Mistral sobre consultas de atención al cliente, midiendo calidad, latencia y costo en dos condiciones: baseline sin RAG y recuperación aumentada (RAG).

El corpus proviene de [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). En `data/` se conservan la muestra piloto de 20 consultas y la muestra ampliada de 324 consultas traducidas al español.

## Contenido

- `src/`: corpus, clientes de OpenAI/Ollama, costos y recuperación RAG.
- `tests/`: pruebas unitarias sin llamadas reales a APIs.
- `notebooks/02_fundamentos.ipynb`: experimento reproducible del baseline sin RAG.
- `notebooks/03_rag_piloto.ipynb`: piloto RAG; la generación está desactivada por defecto.
- `data/corpus_es_muestra.csv`: muestra traducida para evitar repetir costo y tiempo.
- `data/conocimiento_rag.csv`: 81 documentos Bitext, tres por intención, sin solapamientos directos con la evaluación.
- `data/conocimiento_rag_auditoria.json`: trazabilidad de duplicados y fugas excluidas.
- `data/conocimiento_rag_es.csv`: traducción controlada usada como análisis de sensibilidad.
- `data/conocimiento_rag_es_auditoria.json`: tokens, costo y correcciones de marcadores de la traducción.
- `data/muestra_ampliada_en.csv`: 324 consultas del Subproyecto 3, con 12
  casos para cada una de las 27 intenciones.
- `data/muestra_ampliada_en_auditoria.json`: parámetros y controles de la
  selección reproducible de la muestra ampliada.
- `data/muestra_ampliada_en_evaluacion_humana.csv`: submuestra prevista para
  la evaluación humana ciega.
- `data/muestra_ampliada_es.csv`: consultas y respuestas de referencia
  traducidas, con tokens e intentos auditados por fila.
- `data/muestra_ampliada_es_auditoria.json`: completitud, consumo y costo
  registrado de la traducción ampliada.
- `data/muestra_ampliada_es_incidentes.json`: incidente de traducción que no
  pudo reconstruirse desde el checkpoint inicial.
- `data/conocimiento_rag_ampliado.csv`: corpus RAG reconstruido después de
  separar la muestra ampliada y excluir fugas.
- `resultados_fundamentos_20260913_1422.csv`: resultados de la primera corrida.
- `resultados_recuperacion_rag_en.csv`: validación inicial español→inglés del recuperador.
- `resultados_recuperacion_rag_es.csv`: validación con consultas y conocimiento en español.
- `resultados_smoke_rag_local.json`: cuatro generaciones RAG de control con los modelos locales.
- `resultados_rag_piloto.csv`: 60 respuestas RAG crudas con checkpoint por consulta.
- `resultados_rag_piloto_evaluados.csv`: respuestas RAG con calidad, costos e índice compuesto.
- `resumen_rag_piloto.csv`: resumen por modelo.
- `comparacion_baseline_rag.csv`: diferencias exploratorias frente al baseline original.
- `resultados_baseline_controlado_evaluados.csv`: nueva corrida sin RAG con warm-up.
- `comparacion_controlada_rag.csv`: comparación principal bajo el mismo protocolo.
- `conteos_comparacion_controlada.csv`: cantidad de consultas que mejoran o empeoran.
- `resultados_recuperacion_rag_ampliada.csv`: recuperación sobre las 324
  consultas del Subproyecto 3.
- `resultados_recuperacion_rag_ampliada_por_intencion.csv`: diagnóstico de
  recuperación para cada intención.
- `resultados_smoke_experimento_ampliado.csv`: cuatro generaciones locales que
  validan el ejecutor ampliado y su checkpoint.
- `resultados_experimento_ampliado.csv`: 1.944 respuestas crudas del diseño
  completo, con checkpoint por modelo, consulta y condición.
- `resultados_experimento_ampliado_evaluados.csv`: respuestas con BERTScore,
  ROUGE-L, costos y el índice normalizado dentro de cada condición.
- `resumen_experimento_ampliado.csv`: métricas agregadas por modelo y condición.
- `comparacion_pareada_ampliada.csv`: contrastes de RAG frente a la línea base.
- `contraste_hipotesis_rag.csv`: comparación pareada de LLaMA y Mistral contra
  GPT-4o mini usando exclusivamente la condición con RAG.
- `evaluacion_humana/evaluador_1.xlsx` y `evaluador_2.xlsx`: formularios ciegos
  con 324 respuestas, órdenes distintos, rúbrica y validaciones de puntaje.
- `evaluacion_humana/clave_ciega.xlsx`: correspondencia separada entre el
  identificador ciego, el modelo y la condición. No debe entregarse a los
  evaluadores antes de cerrar ambos formularios.
- `TF_Juan_Bautista_Xifro_2026.docx`: única versión vigente de la tesis.
- `requirements.txt`: dependencias Python reproducibles.
- `requirements-lock.txt`: versiones exactas verificadas en Windows.
- `.python-version`: versión de Python usada por el proyecto.

## Requisitos

- Windows 10/11.
- Python 3.11 (las versiones muy nuevas pueden ser incompatibles con `torch`/`bert-score`).
- Ollama con `llama3.1:8b` y `mistral` descargados.
- Una API key de OpenAI con crédito disponible.
- Aproximadamente 10–12 GB libres para modelos y cachés.

## Instalación en PowerShell

Ejecutar desde esta carpeta (`proyecto/`):

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

ollama pull llama3.1:8b
ollama pull mistral

Copy-Item .env.example .env
```

Si una política de Windows impide instalar Python globalmente, puede utilizarse
[uv](https://docs.astral.sh/uv/). El proyecto fija Python 3.11.17:

```powershell
uv venv --python 3.11.17 venv
uv pip install --python .\venv\Scripts\python.exe -r requirements-lock.txt
```

El archivo `requirements.txt` declara las dependencias directas y permite
actualizarlas. `requirements-lock.txt` conserva las versiones exactas del entorno
validado y es la opción recomendada para reproducir una corrida.

Luego editar `.env` y reemplazar el valor de ejemplo por una clave válida. `.env` y `venv/` son locales y Git los ignora.

Para iniciar Ollama en Windows, tanto con una instalación global como con la
distribución portable local, ejecutar:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\iniciar_ollama.ps1
```

## Verificación

```powershell
.\venv\Scripts\python.exe -m pytest tests -v
.\.ollama-bin\ollama.exe list
```

El resultado esperado es `53 passed`. Antes de ejecutar el experimento generativo, Ollama debe estar activo en `http://localhost:11434`.

## Preparar y validar RAG

Desde `proyecto/`:

```powershell
# Reconstruye los 81 documentos desde Bitext. No usa una API generativa.
.\venv\Scripts\python.exe .\scripts\construir_corpus_rag.py

# Valida Hit@1, Hit@3 y latencia sin generar respuestas.
.\venv\Scripts\python.exe .\scripts\validar_recuperacion_rag.py

# Smoke test sin costo de API: dos consultas en LLaMA y Mistral.
.\venv\Scripts\python.exe .\scripts\smoke_rag_local.py

# Piloto completo con checkpoint y evaluación posterior.
.\venv\Scripts\python.exe .\scripts\ejecutar_rag_piloto.py
.\venv\Scripts\python.exe .\scripts\evaluar_rag_piloto.py

# Baseline con el mismo warm-up y comparación controlada.
.\venv\Scripts\python.exe .\scripts\ejecutar_baseline_controlado.py
.\venv\Scripts\python.exe .\scripts\evaluar_comparacion_controlada.py
```

El recuperador usa [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). La primera validación cruzada español→inglés obtuvo `Hit@1 = 80 %`, `Hit@3 = 90 %` y una latencia media de `14,40 ms`. El modelo se descarga la primera vez y queda en una caché local ignorada por Git.

La traducción controlada del conocimiento se realiza por separado:

```powershell
.\venv\Scripts\python.exe .\scripts\traducir_corpus_rag.py
```

Ese comando usa GPT-4o mini solamente si existen filas pendientes. Guarda cada avance en `data/conocimiento_rag_es.csv`, por lo que puede reanudarse después de una interrupción sin repetir documentos ya traducidos. La corrida completa costó USD 0,010899.

### Decisión de idioma del conocimiento

La traducción no mejoró la recuperación: obtuvo `Hit@1 = 70 %` y `Hit@3 = 85 %`, con `14,56 ms` de latencia media. También se probó exploratoriamente `multilingual-e5-small` con los prefijos de recuperación recomendados por su documentación; su mejor variante alcanzó `70 % / 85 %`. Por eso el piloto generativo fija como condición principal MiniLM con el conocimiento original en inglés. La variante española se conserva como análisis de sensibilidad y esta decisión deberá confirmarse con la muestra ampliada.

## Preparar el Subproyecto 3

La muestra ampliada contiene 324 consultas porque se encuentra dentro del
rango de 300–500 previsto en el TF y permite seleccionar exactamente 12 casos
para cada una de las 27 intenciones. Puede reconstruirse sin usar APIs pagas:

```powershell
.\venv\Scripts\python.exe .\scripts\construir_muestra_ampliada.py
```

La traducción al español se ejecuta por separado y sí realiza llamadas pagas:

```powershell
.\venv\Scripts\python.exe .\scripts\traducir_muestra_ampliada.py
```

El traductor guarda cada fila en `data/muestra_ampliada_es.csv`, valida que un
checkpoint corresponda a la misma entrada y reanuda únicamente las filas
pendientes. Su auditoría registra completitud, tokens y costo real.

La recuperación sobre la muestra ampliada se valida sin generar respuestas ni
usar una API paga:

```powershell
.\venv\Scripts\python.exe .\scripts\validar_recuperacion_rag.py `
  --corpus .\data\conocimiento_rag_ampliado.csv `
  --evaluacion .\data\muestra_ampliada_es.csv `
  --salida .\resultados_recuperacion_rag_ampliada.csv
```

La traducción finalizó con 324/324 filas, 74.418 tokens de entrada, 56.962 de
salida y un costo registrado de USD 0,045340. Tres salidas que omitían un
marcador fueron rechazadas y luego resueltas con traducción segmentada. Hubo
además tres intentos iniciales cuyos tokens no pudieron recuperarse; el archivo
de incidentes conserva esta limitación para conciliarla con el panel de uso.

La recuperación ampliada obtuvo `Hit@1 = 81,8 %`, `Hit@3 = 96,0 %` y una
latencia media de `14,43 ms`. Los resultados crudos se guardan en
`resultados_recuperacion_rag_ampliada.csv` y el diagnóstico por intención en
`resultados_recuperacion_rag_ampliada_por_intencion.csv`.

El ejecutor ampliado construye un plan reproducible de 1.944 respuestas. Para
cada modelo, alterna el orden de las condiciones: 162 consultas comienzan sin
RAG y 162 con RAG. Un smoke test local puede ejecutarse así:

```powershell
.\venv\Scripts\python.exe .\scripts\ejecutar_experimento_ampliado.py `
  --modelos llama-3.1-8b mistral `
  --limite-consultas 1 `
  --salida .\resultados_smoke_experimento_ampliado.csv
```

La corrida completa requiere una confirmación explícita para evitar iniciarla
por accidente:

```powershell
.\venv\Scripts\python.exe .\scripts\ejecutar_experimento_ampliado.py `
  --confirmar-corrida-completa
```

El checkpoint permite reanudar por la combinación única de modelo, consulta y
condición. La corrida ampliada finalizó con 1.944/1.944 respuestas, sin errores
ni campos vacíos. El costo registrado de GPT-4o mini fue USD 0,072590.

Las métricas y los contrastes se regeneran sin invocar modelos ni APIs:

```powershell
.\venv\Scripts\python.exe .\scripts\evaluar_experimento_ampliado.py
```

El script reutiliza BERTScore y ROUGE-L si ya existen, pero recalcula costos,
normalizaciones e índice. La normalización se realiza dentro de cada condición
para que el contraste formal de H1–H3 utilice únicamente las configuraciones
con RAG.

### Resultado ampliado con RAG

| Modelo | BERTScore F1 | ROUGE-L | Latencia media | Costo por 1.000 | Índice compuesto |
|---|---:|---:|---:|---:|---:|
| GPT-4o mini | 0,7491 | 0,3021 | 1,530 s | USD 0,1620 | 0,7680 |
| LLaMA 3.1 8B | 0,7314 | 0,2589 | 1,112 s | USD 0,2130 | 0,7563 |
| Mistral | 0,7372 | 0,2720 | 1,576 s | USD 0,3020 | 0,7177 |

RAG elevó significativamente BERTScore y ROUGE-L en los tres modelos, aunque
también aumentó su latencia y costo. Frente a GPT, Mistral quedó completamente
dentro del margen de ±0,02 BERTScore (IC 95 % de la diferencia: −0,0150 a
−0,0088). LLaMA cumplió el margen en la diferencia puntual (−0,0177), pero su
intervalo lo cruzó levemente (−0,0217 a −0,0137). Por ello H1 recibe apoyo para
Mistral y evidencia parcial para LLaMA bajo ese margen de referencia. H2 y H3
no reciben apoyo: bajo el escenario cloud definido, ambos modelos abiertos
costaron más por 1.000 consultas y ninguno superó el índice de GPT-4o mini.

### Visualizar los resultados

Las figuras incluidas en el TF se regeneran directamente desde los CSV del
experimento, sin datos inventados ni llamadas a modelos:

```powershell
.\venv\Scripts\python.exe .\scripts\generar_figuras_resultados.py
```

El dashboard interactivo permite filtrar por modelo y condición, revisar las
métricas agregadas y comparar calidad, latencia y costo:

```powershell
.\venv\Scripts\streamlit.exe run .\dashboard.py
```

Streamlit mostrará la dirección local en la terminal, normalmente
`http://localhost:8501`. Los gráficos estáticos quedan en `figuras/`.

### Evaluación humana ciega

Los insumos se reconstruyen sin invocar modelos ni APIs:

```powershell
.\venv\Scripts\python.exe .\scripts\preparar_evaluacion_humana.py
```

El comando produce dos CSV con órdenes aleatorios diferentes y una clave
separada. Las versiones Excel incluidas agregan la rúbrica, validaciones de
entrada y una columna de estado. Cada evaluador debe completar únicamente su
archivo, sin recibir `clave_ciega.xlsx` ni `clave_ciega.csv`.

La escala es de 1 a 5 para corrección, utilidad, claridad y fidelidad al
contexto. La alucinación se marca con 0 (ausente) o 1 (presente). El mismo
contexto se muestra en las seis respuestas de cada consulta para conservar el
cegamiento respecto de la condición generativa.

Cuando ambos archivos estén completos, el análisis se ejecuta así:

```powershell
.\venv\Scripts\python.exe .\scripts\analizar_evaluacion_humana.py
```

El script valida las 324 filas por evaluador, calcula kappa de Cohen ponderado
lineal para las escalas ordinales, kappa sin ponderar para alucinación y genera
los promedios por modelo y condición. Si queda una fila incompleta, se detiene
sin revelar resultados parciales.

## Resultado del piloto RAG

Las 60 generaciones finalizaron sin errores. Promedios:

| Modelo | BERTScore F1 | ROUGE-L | Latencia total | Cambio BERTScore vs. baseline |
|---|---:|---:|---:|---:|
| GPT-4o mini | 0,7526 | 0,3054 | 1,591 s | +0,0406 |
| LLaMA 3.1 8B | 0,7213 | 0,2224 | 1,496 s | +0,0059 |
| Mistral | 0,7333 | 0,2770 | 1,867 s | +0,0230 |

GPT-4o mini consumió USD 0,002958 en total. Los costos locales publicados por el código continúan siendo provisionales porque dependen de una tarifa de instancia pendiente de reemplazar por una fuente real.

### Comparación controlada contra el baseline

Se repitió el baseline con el mismo calentamiento y en la misma sesión de trabajo. Esta comparación reemplaza a la histórica para analizar el efecto de RAG:

| Modelo | BERTScore sin RAG | BERTScore con RAG | Δ BERTScore | Δ ROUGE-L | Latencia sin RAG | Latencia con RAG |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4o mini | 0,7121 | 0,7526 | +0,0405 | +0,0690 | 1,723 s | 1,591 s |
| LLaMA 3.1 8B | 0,7151 | 0,7213 | +0,0061 | −0,0255 | 1,213 s | 1,496 s |
| Mistral | 0,7058 | 0,7333 | +0,0275 | +0,0414 | 1,350 s | 1,867 s |

RAG aumentó la latencia local un 23,3 % en LLaMA y un 38,3 % en Mistral. La API de GPT fue un 7,6 % más rápida con RAG en esta corrida, pese a recibir 9,35 veces más tokens de entrada; esto se interpreta como variabilidad del servicio remoto, no como una aceleración causada por RAG.

El costo real de las 20 respuestas de GPT pasó de USD 0,001333 sin RAG a USD 0,002958 con RAG. Para los modelos locales, la comparación principal amortiza una instancia Lambda Cloud con NVIDIA Quadro RTX 6000 de 24 GB publicada a USD 0,69 por GPU-hora. Como sensibilidad se calcula una cota superior eléctrica de 415 W (350 W de la RTX 3080 Ti más 65 W del Ryzen 5 5600X) con un benchmark comercial de USD 0,1453/kWh. Esta última cifra no representa una tarifa argentina ni un consumo medido en pared.

Fuentes de los parámetros, consultadas el 4 de octubre de 2026: [precios de GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), [instancias de Lambda Cloud](https://lambda.ai/instances), [especificaciones de la RTX 3080 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/), [especificaciones del Ryzen 5 5600X](https://www.amd.com/en/products/processors/desktops/ryzen/5000-series/amd-ryzen-5-5600x.html) y [precio comercial de electricidad de la EIA](https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a).

## Ejecutar el notebook

En VS Code:

1. Abrir `notebooks/02_fundamentos.ipynb`.
2. Seleccionar el kernel `Python 3.11 (Trabajo Final)` o `proyecto\venv\Scripts\python.exe`.
3. Ejecutar las celdas en orden.

También se puede ejecutar completo desde PowerShell:

```powershell
Set-Location notebooks
..\venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace 02_fundamentos.ipynb
Set-Location ..
```

La ejecución completa realiza llamadas pagas a OpenAI y usa los modelos locales de Ollama. El corpus ya traducido evita repetir la traducción mientras `data/corpus_es_muestra.csv` exista.

Para el Subproyecto 2, abrir `notebooks/03_rag_piloto.ipynb`. Sus celdas de carga y recuperación son seguras; para habilitar las 60 generaciones (20 consultas × 3 modelos) se debe cambiar explícitamente `EJECUTAR_EXPERIMENTO = False` a `True`.

## Portabilidad

Al clonar en otra computadora se deben recrear `venv/` y `.env`, instalar Ollama y volver a descargar sus modelos. No se deben copiar ni versionar el entorno virtual, las credenciales ni los cachés: todo lo necesario para reconstruirlos está declarado en este repositorio.
