"""Esquema canónico y adaptadores para datos externos del framework."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


CAMPOS_CONSULTA = (
    "consulta_id",
    "texto",
    "respuesta_referencia",
    "idioma",
    "categoria",
    "intencion",
    "metadata",
)
CAMPOS_CONOCIMIENTO = (
    "documento_id",
    "contenido",
    "idioma",
    "categoria",
    "intencion",
    "fuente",
    "metadata",
)


class ErrorValidacionDatos(ValueError):
    """Error legible al validar archivos de entrada."""


def _es_nulo(valor: Any) -> bool:
    if valor is None:
        return True
    try:
        resultado = pd.isna(valor)
    except (TypeError, ValueError):
        return False
    if hasattr(resultado, "__len__"):
        return False
    return bool(resultado)


def _texto_opcional(valor: Any) -> str | None:
    if _es_nulo(valor):
        return None
    texto = str(valor).strip()
    return texto or None


def _metadata(valor: Any) -> dict[str, Any]:
    if _es_nulo(valor) or valor == "":
        return {}
    if isinstance(valor, str):
        try:
            valor = json.loads(valor)
        except json.JSONDecodeError as exc:
            raise ValueError("debe contener un objeto JSON válido") from exc
    if not isinstance(valor, dict):
        raise ValueError("debe ser un objeto")
    return valor


class ConsultaEntrada(BaseModel):
    """Unidad de evaluación recibida por el framework."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    consulta_id: str = Field(min_length=1)
    texto: str = Field(min_length=1)
    respuesta_referencia: str | None = None
    idioma: str = Field(default="es", min_length=1)
    categoria: str | None = None
    intencion: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalizar_opcionales = field_validator(
        "respuesta_referencia", "categoria", "intencion", mode="before"
    )(_texto_opcional)
    _normalizar_metadata = field_validator("metadata", mode="before")(_metadata)


class DocumentoConocimiento(BaseModel):
    """Documento que puede incorporarse al índice RAG."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    documento_id: str = Field(min_length=1)
    contenido: str = Field(min_length=1)
    idioma: str = Field(default="es", min_length=1)
    categoria: str | None = None
    intencion: str | None = None
    fuente: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalizar_opcionales = field_validator(
        "categoria", "intencion", "fuente", mode="before"
    )(_texto_opcional)
    _normalizar_metadata = field_validator("metadata", mode="before")(_metadata)


def _leer_registros(ruta: str | Path, clave_json: str) -> list[dict[str, Any]]:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")

    extension = ruta.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(ruta).to_dict(orient="records")
    if extension in {".jsonl", ".ndjson"}:
        return pd.read_json(ruta, lines=True).to_dict(orient="records")
    if extension != ".json":
        raise ErrorValidacionDatos(
            "Formato no admitido. Use .csv, .json, .jsonl o .ndjson."
        )

    with ruta.open(encoding="utf-8") as archivo:
        contenido = json.load(archivo)
    if isinstance(contenido, dict):
        contenido = contenido.get(clave_json)
    if not isinstance(contenido, list) or not all(
        isinstance(fila, dict) for fila in contenido
    ):
        raise ErrorValidacionDatos(
            f"El JSON debe ser una lista de objetos o contener la clave '{clave_json}'."
        )
    return contenido


def _canonizar_registros(
    registros: Iterable[Mapping[str, Any]],
    campos: tuple[str, ...],
    mapeo: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Convierte columnas de origen al esquema canónico.

    ``mapeo`` usa nombres canónicos como claves y nombres de origen como valores.
    Las columnas no mapeadas se preservan dentro de ``metadata``.
    """
    mapeo = dict(mapeo or {})
    desconocidos = set(mapeo) - set(campos)
    if desconocidos:
        raise ErrorValidacionDatos(
            f"El mapeo contiene campos canónicos desconocidos: {sorted(desconocidos)}"
        )

    fuentes_usadas = {mapeo.get(campo, campo) for campo in campos}
    salida = []
    for registro in registros:
        fila = {}
        for campo in campos:
            if campo == "metadata":
                continue
            nombre_origen = mapeo.get(campo, campo)
            if nombre_origen in registro and not _es_nulo(registro[nombre_origen]):
                fila[campo] = registro[nombre_origen]
        nombre_metadata = mapeo.get("metadata", "metadata")
        metadata = _metadata(registro.get(nombre_metadata))
        for nombre, valor in registro.items():
            if nombre not in fuentes_usadas and not _es_nulo(valor):
                metadata[nombre] = valor
        fila["metadata"] = metadata
        salida.append(fila)
    return salida


def _mensaje_validacion(indice: int, exc: ValidationError) -> str:
    detalles = []
    for error in exc.errors(include_url=False, include_input=False):
        campo = ".".join(str(parte) for parte in error["loc"])
        detalles.append(f"{campo}: {error['msg']}")
    return f"fila {indice}: " + "; ".join(detalles)


def _validar_coleccion(
    registros: Iterable[Mapping[str, Any]],
    modelo: type[BaseModel],
    campo_id: str,
) -> list[BaseModel]:
    validados = []
    errores = []
    for indice, registro in enumerate(registros, start=1):
        try:
            validados.append(modelo.model_validate(registro))
        except ValidationError as exc:
            errores.append(_mensaje_validacion(indice, exc))
    if errores:
        raise ErrorValidacionDatos("Datos de entrada inválidos:\n- " + "\n- ".join(errores))

    ids = [getattr(fila, campo_id) for fila in validados]
    duplicados = sorted({identificador for identificador in ids if ids.count(identificador) > 1})
    if duplicados:
        raise ErrorValidacionDatos(
            f"El campo {campo_id} debe ser único. Valores duplicados: {duplicados}"
        )
    return validados


def validar_consultas(
    registros: Iterable[Mapping[str, Any]],
) -> list[ConsultaEntrada]:
    return _validar_coleccion(registros, ConsultaEntrada, "consulta_id")  # type: ignore[return-value]


def validar_conocimiento(
    registros: Iterable[Mapping[str, Any]],
) -> list[DocumentoConocimiento]:
    return _validar_coleccion(registros, DocumentoConocimiento, "documento_id")  # type: ignore[return-value]


def _a_dataframe(modelos: Iterable[BaseModel], campos: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [modelo.model_dump() for modelo in modelos],
        columns=list(campos),
    )


def cargar_consultas(
    ruta: str | Path,
    mapeo: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    registros = _leer_registros(ruta, "consultas")
    canonicos = _canonizar_registros(registros, CAMPOS_CONSULTA, mapeo)
    return _a_dataframe(validar_consultas(canonicos), CAMPOS_CONSULTA)


def cargar_conocimiento(
    ruta: str | Path,
    mapeo: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    registros = _leer_registros(ruta, "conocimiento")
    canonicos = _canonizar_registros(registros, CAMPOS_CONOCIMIENTO, mapeo)
    return _a_dataframe(validar_conocimiento(canonicos), CAMPOS_CONOCIMIENTO)


def cargar_mapeo(ruta: str | Path | None) -> dict[str, str] | None:
    if ruta is None:
        return None
    with Path(ruta).open(encoding="utf-8") as archivo:
        contenido = json.load(archivo)
    if not isinstance(contenido, dict) or not all(
        isinstance(clave, str) and isinstance(valor, str)
        for clave, valor in contenido.items()
    ):
        raise ErrorValidacionDatos("El mapeo debe ser un objeto JSON de texto a texto.")
    return contenido


def _valor_fila(fila: Mapping[str, Any], *nombres: str) -> Any:
    for nombre in nombres:
        valor = fila.get(nombre)
        if not _es_nulo(valor) and str(valor).strip():
            return valor
    return None


def adaptar_consultas_bitext(
    datos: pd.DataFrame,
    preferir_espanol: bool = True,
) -> pd.DataFrame:
    """Adapta las muestras Bitext existentes al esquema canónico."""
    registros = []
    for posicion, fila_serie in enumerate(datos.to_dict(orient="records"), start=1):
        texto = _valor_fila(
            fila_serie,
            *("instruction_es", "instruction") if preferir_espanol else ("instruction", "instruction_es"),
        )
        referencia = _valor_fila(
            fila_serie,
            *("response_es", "response") if preferir_espanol else ("response", "response_es"),
        )
        identificador = _valor_fila(fila_serie, "consulta_id", "source_row_id")
        registros.append(
            {
                "consulta_id": str(identificador or f"bitext-consulta-{posicion:04d}"),
                "texto": texto,
                "respuesta_referencia": referencia,
                "idioma": "es" if preferir_espanol and fila_serie.get("instruction_es") else "en",
                "categoria": _valor_fila(fila_serie, "categoria", "category"),
                "intencion": _valor_fila(fila_serie, "intencion", "intent"),
                "metadata": {
                    "fuente": "bitext",
                    "source_row_id": _valor_fila(fila_serie, "source_row_id"),
                },
            }
        )
    return _a_dataframe(validar_consultas(registros), CAMPOS_CONSULTA)


def adaptar_conocimiento_bitext(
    datos: pd.DataFrame,
    preferir_espanol: bool = False,
) -> pd.DataFrame:
    """Adapta el corpus RAG Bitext existente al esquema canónico."""
    registros = []
    for posicion, fila in enumerate(datos.to_dict(orient="records"), start=1):
        contenido = _valor_fila(fila, "texto_rag")
        if contenido is None:
            consulta = _valor_fila(
                fila,
                *("instruction_es", "instruction") if preferir_espanol else ("instruction", "instruction_es"),
            )
            respuesta = _valor_fila(
                fila,
                *("response_es", "response") if preferir_espanol else ("response", "response_es"),
            )
            contenido = f"Consulta: {consulta}\nRespuesta: {respuesta}"
        identificador = _valor_fila(fila, "documento_id", "document_id", "source_row_id")
        registros.append(
            {
                "documento_id": str(identificador or f"bitext-documento-{posicion:04d}"),
                "contenido": contenido,
                "idioma": "es" if preferir_espanol else "en",
                "categoria": _valor_fila(fila, "categoria", "category"),
                "intencion": _valor_fila(fila, "intencion", "intent"),
                "fuente": "Bitext Customer Support LLM Chatbot Training Dataset",
                "metadata": {
                    "source_row_id": _valor_fila(fila, "source_row_id"),
                },
            }
        )
    return _a_dataframe(validar_conocimiento(registros), CAMPOS_CONOCIMIENTO)


def consultas_a_formato_pipeline(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los alias que consumen los scripts experimentales actuales."""
    canonicos = _a_dataframe(
        validar_consultas(datos.to_dict(orient="records")), CAMPOS_CONSULTA
    )
    salida = canonicos.copy()
    salida["instruction_es"] = salida["texto"]
    salida["response_es"] = salida["respuesta_referencia"]
    salida["category"] = salida["categoria"]
    salida["intent"] = salida["intencion"]
    return salida


def conocimiento_a_formato_rag(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los alias que consume ``IndiceRAG`` sin perder el esquema v1."""
    canonicos = _a_dataframe(
        validar_conocimiento(datos.to_dict(orient="records")), CAMPOS_CONOCIMIENTO
    )
    salida = canonicos.copy()
    salida["document_id"] = salida["documento_id"]
    salida["texto_rag"] = salida["contenido"]
    salida["category"] = salida["categoria"]
    salida["intent"] = salida["intencion"]
    return salida
