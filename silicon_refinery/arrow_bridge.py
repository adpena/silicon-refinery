"""
Arrow IPC Bridge for SiliconRefinery.

Zero-copy data sharing between processes using Arrow IPC format. This enables
SiliconRefinery to interoperate with Polars, Pandas, DuckDB, and any other
Arrow-compatible tool without serialization overhead.

All pyarrow and polars imports are lazy so the module can be imported without
those packages installed.
"""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger("silicon_refinery")

__all__ = [
    "ArrowStreamWriter",
    "from_arrow_ipc",
    "from_arrow_ipc_buffer",
    "from_polars",
    "to_arrow_ipc",
    "to_arrow_ipc_buffer",
    "to_polars",
]


def _require_pyarrow() -> Any:
    """Lazy-import pyarrow and raise a clear error if it is not installed."""
    try:
        return importlib.import_module("pyarrow")
    except ImportError:
        raise ImportError("Install pyarrow: uv add pyarrow") from None


def _infer_schema(data: list[dict[str, Any]], pa: Any) -> Any:
    """Infer a pyarrow schema from a list of dicts."""
    if not data:
        return pa.schema([])

    def _kind(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        return "other"

    # Gather all keys across all records and track observed non-null types.
    seen_types: dict[str, set[str]] = {}
    for record in data:
        for key, value in record.items():
            seen_types.setdefault(key, set())
            if value is not None:
                seen_types[key].add(_kind(value))

    fields: list[Any] = []
    for key in sorted(seen_types):
        kinds = seen_types[key]
        if not kinds:
            fields.append(pa.field(key, pa.string()))
        elif kinds == {"bool"}:
            fields.append(pa.field(key, pa.bool_()))
        elif kinds == {"int"}:
            fields.append(pa.field(key, pa.int64()))
        elif kinds <= {"int", "float"}:
            fields.append(pa.field(key, pa.float64()))
        elif kinds == {"str"}:
            fields.append(pa.field(key, pa.string()))
        else:
            # Mixed/unknown types fallback to string for robustness.
            fields.append(pa.field(key, pa.string()))

    return pa.schema(fields)


def _dicts_to_table(data: list[dict[str, Any]], pa: Any) -> Any:
    """Convert a list of dicts to a pyarrow Table."""
    if not data:
        return pa.table({})

    schema = _infer_schema(data, pa)
    columns: dict[str, list[Any]] = {field.name: [] for field in schema}

    for record in data:
        for field in schema:
            value = record.get(field.name)
            # Convert non-string values to string for string-typed fallback fields
            if value is not None and field.type == pa.string() and not isinstance(value, str):
                value = str(value)
            columns[field.name].append(value)

    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.table(dict(zip([f.name for f in schema], arrays, strict=False)))


def _table_to_dicts(table: Any) -> list[dict[str, Any]]:
    """Convert a pyarrow Table back to a list of dicts."""
    if table.num_rows == 0:
        return []
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


# ---------------------------------------------------------------------------
# File-based IPC
# ---------------------------------------------------------------------------


def to_arrow_ipc(data: list[dict[str, Any]], path: str | Path) -> None:
    """Serialize a list of extraction result dicts to Arrow IPC file format.

    Args:
        data: List of dictionaries to serialize.
        path: Destination file path.
    """
    pa = _require_pyarrow()
    table = _dicts_to_table(data, pa)
    target_path = Path(path)
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        ) as tmp_file:
            temp_path = tmp_file.name

        writer = pa.ipc.new_file(temp_path, table.schema)
        try:
            writer.write_table(table)
        finally:
            writer.close()

        os.replace(temp_path, str(target_path))
        temp_path = None
    finally:
        if temp_path is not None:
            with suppress(FileNotFoundError):
                os.remove(temp_path)

    logger.debug("Wrote %d records to Arrow IPC file: %s", len(data), path)


def from_arrow_ipc(path: str | Path) -> list[dict[str, Any]]:
    """Deserialize Arrow IPC file back to a list of dicts.

    Args:
        path: Source file path.

    Returns:
        List of dictionaries.
    """
    pa = _require_pyarrow()
    reader = pa.ipc.open_file(str(path))
    try:
        table = reader.read_all()
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    return _table_to_dicts(table)


# ---------------------------------------------------------------------------
# Buffer-based IPC (for streaming between processes)
# ---------------------------------------------------------------------------


def to_arrow_ipc_buffer(data: list[dict[str, Any]]) -> bytes:
    """Serialize a list of extraction result dicts to in-memory IPC bytes.

    Args:
        data: List of dictionaries to serialize.

    Returns:
        Bytes containing the Arrow IPC stream.
    """
    pa = _require_pyarrow()
    table = _dicts_to_table(data, pa)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def from_arrow_ipc_buffer(buf: bytes) -> list[dict[str, Any]]:
    """Deserialize from IPC bytes back to a list of dicts.

    Args:
        buf: Bytes containing an Arrow IPC stream.

    Returns:
        List of dictionaries.
    """
    pa = _require_pyarrow()
    reader = pa.ipc.open_stream(buf)
    try:
        table = reader.read_all()
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    return _table_to_dicts(table)


# ---------------------------------------------------------------------------
# ArrowStreamWriter — async context manager for incremental writes
# ---------------------------------------------------------------------------


class ArrowStreamWriter:
    """Async context manager for streaming Arrow IPC writes.

    Usage::

        async with ArrowStreamWriter("output.arrow") as writer:
            await writer.write({"name": "Alice", "age": 30})
            await writer.write({"name": "Bob", "age": 25})
    """

    def __init__(
        self,
        path: str | Path,
        schema_hint: dict[str, str] | None = None,
    ) -> None:
        self._path = str(path)
        self._schema_hint = schema_hint
        self._pa: Any = None
        self._writer: Any = None
        self._schema: Any = None
        self._schema_field_names: set[str] = set()
        self._buffer: list[dict[str, Any]] = []

    async def __aenter__(self) -> ArrowStreamWriter:
        self._pa = _require_pyarrow()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _resolve_schema(self, record: dict[str, Any]) -> Any:
        """Build the Arrow schema from a schema_hint or the first record."""
        pa = self._pa
        if self._schema_hint is not None:
            _type_map = {
                "str": pa.string(),
                "string": pa.string(),
                "int": pa.int64(),
                "int64": pa.int64(),
                "float": pa.float64(),
                "float64": pa.float64(),
                "bool": pa.bool_(),
            }
            fields = [
                pa.field(name, _type_map.get(typ, pa.string()))
                for name, typ in self._schema_hint.items()
            ]
            return pa.schema(fields)

        # Infer from the first record
        return _infer_schema([record], pa)

    async def write(self, record: dict[str, Any]) -> None:
        """Write a single record dict.

        If the schema has not been determined yet (first call), the record is used
        to infer the schema and the IPC file writer is opened.
        """
        pa = self._pa
        if pa is None:
            raise RuntimeError("ArrowStreamWriter must be used as an async context manager")

        if self._writer is None:
            self._schema = self._resolve_schema(record)
            self._schema_field_names = {field.name for field in self._schema}
            self._writer = pa.ipc.new_file(self._path, self._schema)

        unexpected_keys = sorted(set(record) - self._schema_field_names)
        if unexpected_keys:
            expected_keys = sorted(self._schema_field_names)
            raise ValueError(
                "Record contains unexpected keys not present in Arrow schema: "
                f"{unexpected_keys}. Expected keys: {expected_keys}"
            )

        # Build a single-row batch
        arrays = []
        for field in self._schema:
            value = record.get(field.name)
            if value is not None and field.type == pa.string() and not isinstance(value, str):
                value = str(value)
            arrays.append(pa.array([value], type=field.type))

        batch = pa.record_batch(arrays, schema=self._schema)
        self._writer.write_batch(batch)

    async def close(self) -> None:
        """Flush and close the underlying IPC writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.debug("Closed ArrowStreamWriter for: %s", self._path)


# ---------------------------------------------------------------------------
# Polars convenience functions
# ---------------------------------------------------------------------------


def to_polars(data: list[dict[str, Any]]) -> Any:
    """Convert a list of extraction result dicts to a Polars DataFrame.

    Args:
        data: List of dictionaries.

    Returns:
        polars.DataFrame
    """
    try:
        pl = importlib.import_module("polars")
    except ImportError:
        raise ImportError("Install polars: uv add polars") from None
    return pl.DataFrame(data)


def from_polars(df: Any) -> list[dict[str, Any]]:
    """Convert a Polars DataFrame back to a list of dicts.

    Args:
        df: A polars.DataFrame instance.

    Returns:
        List of dictionaries.
    """
    return df.to_dicts()
