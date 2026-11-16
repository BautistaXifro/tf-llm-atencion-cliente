# Instrucciones para la evaluación humana ciega

## Objetivo

Evaluar respuestas de atención al cliente sin conocer qué modelo las generó ni
si utilizaron recuperación aumentada (RAG). Cada persona debe trabajar de forma
independiente y completar únicamente la planilla que recibió.

## Requisitos del evaluador

- Comprender español y tener lectura suficiente de inglés, porque parte del
  contexto recuperado conserva el idioma original del corpus.
- Poder valorar respuestas de atención al cliente siguiendo la rúbrica.
- No haber participado en la generación de las respuestas.
- No consultar ni recibir la clave que identifica modelos y condiciones.

No es necesario tener experiencia técnica en inteligencia artificial.

## Archivos que se entregan

- Evaluador 1: `evaluador_1.xlsx`.
- Evaluador 2: `evaluador_2.xlsx`.

No se deben compartir con los evaluadores `clave_ciega.csv`,
`clave_ciega.xlsx`, los resultados comparativos ni el archivo del otro
evaluador.

## Procedimiento

1. Abrir la hoja `Rúbrica` y leer los criterios antes de puntuar.
2. Volver a la hoja `Evaluación`.
3. Para cada fila, leer la consulta, la respuesta de referencia, el contexto y
   la respuesta que se debe evaluar.
4. Completar las cinco columnas de puntaje:
   - `correccion_1_5`: exactitud y ausencia de errores sustantivos.
   - `utilidad_1_5`: capacidad para ayudar al usuario a avanzar.
   - `claridad_1_5`: comprensión, orden y concisión.
   - `fidelidad_contexto_1_5`: consistencia con la referencia y el contexto.
   - `alucinacion_0_1`: 0 si no hay afirmaciones inventadas; 1 si las hay.
5. Usar `observaciones` cuando exista una duda, un error relevante o una
   alucinación. Esta columna es opcional, pero ayuda a interpretar los casos
   problemáticos.
6. Confirmar que la columna `estado` diga `Completo` en todas las filas.
7. Guardar el archivo como `evaluador_1_completado.xlsx` o
   `evaluador_2_completado.xlsx`, según corresponda, y devolverlo al responsable
   del TF.

Los valores 2 y 4 pueden utilizarse como niveles intermedios entre los anclajes
1, 3 y 5 descritos en la rúbrica.

## Reglas para preservar el experimento

- No intentar identificar el modelo ni buscar fragmentos de las respuestas en
  Internet.
- No conversar sobre los puntajes con el otro evaluador hasta finalizar ambas
  planillas.
- No modificar las columnas de identificación, consulta, referencia, contexto,
  respuesta o estado.
- No insertar, eliminar ni reordenar filas.
- Se puede trabajar en varias sesiones. Guardar el archivo después de cada
  bloque para evitar perder avances.

Cada planilla contiene 324 respuestas. Conviene dividir el trabajo en bloques
breves para mantener un criterio estable y evitar fatiga. Si un criterio genera
dudas repetidas, se debe releer la rúbrica y aplicar la misma interpretación al
resto de las filas.

## Cierre

El responsable del Trabajo Final verificará que ambas planillas estén completas
antes de revelar la clave y calcular el acuerdo entre evaluadores. No se
publicarán resultados parciales identificados por modelo.
