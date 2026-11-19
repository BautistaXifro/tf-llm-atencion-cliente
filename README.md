# Trabajo Final — Juan Bautista Xifro

Repositorio del Trabajo Final de la Licenciatura en Gestión de la Tecnología
Informática (UAI, 2026): evaluación comparativa de modelos de lenguaje con RAG
para atención al cliente.

## Estructura publicada

- `proyecto/src/`: componentes del framework, recuperación, evaluación y esquema
  de datos externos.
- `proyecto/scripts/`: preparación de datos, ejecución de experimentos, análisis
  y generación de figuras.
- `proyecto/notebooks/04_evaluacion_completa.ipynb`: recorrido integral y seguro
  de la evaluación, reutilizando resultados existentes por defecto.
- `proyecto/data/`: muestras, corpus RAG y auditorías de trazabilidad.
- `proyecto/evaluacion_humana/`: protocolo, formularios y resultados de la
  evaluación ciega.
- `proyecto/figuras/` y `proyecto/dashboard.py`: visualización de resultados.
- `proyecto/tests/`: suite automatizada de 73 pruebas.
- `proyecto/requirements.txt` y `requirements-lock.txt`: dependencias directas y
  entorno reproducible.
- `proyecto/TF_Juan_Bautista_Xifro_2026.docx`: única versión vigente del Trabajo Final.

Los materiales internos de gestión y contexto de desarrollo se conservan únicamente de manera local y no forman parte del repositorio publicado.

Las instrucciones de instalación y ejecución están en [`proyecto/README.md`](proyecto/README.md).

Los CSV de resultados se versionan deliberadamente: permiten auditar el piloto,
la corrida ampliada y la evaluación humana sin volver a consumir APIs. Los
archivos locales `proyecto/venv/`, las cachés de modelos y `proyecto/.env` no se
versionan; el entorno se recrea desde las dependencias declaradas y las
credenciales nunca deben subirse.
