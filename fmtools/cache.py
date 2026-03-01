"""
Content-addressable sqlite3 extraction cache for FMTools.

Memoises LLM extraction results so that repeated calls with identical
(instructions, schema, input_text) triples return instantly from local storage
instead of burning Neural Engine cycles.

Usage:
    from fmtools.cache import ExtractionCache, cache_extract

    cache = ExtractionCache()                       # default ~/.cache path
    result = await cache_extract(cache, session, text, schema)

    # Or use the decorator variant:
    @cached_local_extract(schema=MySchema, cache=cache)
    async def extract_person(text: str):
        \"\"\"Extract name and age.\"\"\"
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar, Union, cast

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import ModelProtocol, create_model, create_session

logger = logging.getLogger("fmtools")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_DB_DIR = Path.home() / ".cache" / "fmtools"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "extraction_cache.db"
_DEFAULT_TTL_SECONDS = 86_400  # 24 hours

# Transient errors worth retrying (mirrors decorators.py)
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)


class _CacheCoercionError(ValueError):
    """Raised when cached payload cannot be coerced to the requested schema."""


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _schema_identifier(schema: type[Any]) -> str:
    """Return a fully qualified identifier for a schema class."""
    return f"{schema.__module__}.{schema.__qualname__}"


def _cache_key(instructions: str, schema_identifier: str, input_text: str) -> str:
    """
    Compute a deterministic SHA-256 hex key from the extraction triple.

    Using a separator unlikely to appear in real data avoids collisions.
    """
    payload = f"{instructions}\x00{schema_identifier}\x00{input_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize(value: Any) -> str:
    """Serialise a value to JSON, falling back to ``str`` for exotic types."""
    if hasattr(value, "__dict__"):
        return json.dumps(vars(value), default=str)
    return json.dumps(value, default=str)


def _coerce_cached_value(schema: type[T], cached: Any) -> T:
    """Convert cached JSON payloads back into ``schema`` instances."""
    if isinstance(cached, schema):
        return cached
    if isinstance(cached, dict):
        try:
            return schema(**cached)
        except Exception as exc:
            raise _CacheCoercionError(
                "Cached payload does not match schema "
                f"{_schema_identifier(schema)}; recomputation required."
            ) from exc
    raise _CacheCoercionError(
        "Cached payload type is incompatible with schema "
        f"{_schema_identifier(schema)}: {type(cached).__name__}."
    )


# ---------------------------------------------------------------------------
# ExtractionCache
# ---------------------------------------------------------------------------


class ExtractionCache:
    """
    A thread-safe, content-addressable sqlite3 cache with lazy TTL pruning.

    Parameters
    ----------
    db_path:
        Path to the sqlite3 database file.  Parent directories are created
        automatically.
    ttl:
        Time-to-live in seconds for cached entries.  ``0`` means no expiry.
    """

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        ttl: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

        # Ensure parent directories exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key       TEXT PRIMARY KEY,
                value     TEXT NOT NULL,
                created   REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- public API ----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """
        Look up *key* in the cache.

        Returns the deserialised value on a hit, or ``None`` on a miss.
        Expired entries are pruned lazily.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                self._misses += 1
                return None

            value_json, created = row

            # TTL check
            if self._ttl > 0 and (time.time() - created) > self._ttl:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                self._misses += 1
                return None

            self._hits += 1
            return json.loads(value_json)

    def put(self, key: str, value: Any) -> None:
        """Insert or replace a cache entry."""
        value_json = _serialize(value)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
                (key, value_json, time.time()),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Remove all entries and reset hit/miss counters."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return current hit/miss statistics."""
        with self._lock:
            return {"hits": self._hits, "misses": self._misses}

    def close(self) -> None:
        """Close the underlying sqlite3 connection."""
        self._conn.close()

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExtractionCache(db_path={self._db_path!r}, ttl={self._ttl}, "
            f"hits={self._hits}, misses={self._misses})"
        )


# ---------------------------------------------------------------------------
# Async cache helpers
# ---------------------------------------------------------------------------


async def cache_extract(
    cache: ExtractionCache,
    session: Any,
    input_text: str,
    schema: type[T],
    instructions: str = "",
) -> T:
    """
    Perform a cached extraction: check the cache first, call the FM session
    on a miss, then store the result.

    Parameters
    ----------
    cache:
        The :class:`ExtractionCache` to consult.
    session:
        A ``LanguageModelSession`` (or anything with an async ``respond``).
    input_text:
        The raw text to extract from.
    schema:
        The ``@fm.generable()`` class for structured generation.
    instructions:
        The system instructions used to create *session* (needed for the
        cache key so different instructions don't collide).

    Returns
    -------
    An instance of *schema* populated by the FM (or the cache).
    """
    schema_id = _schema_identifier(schema)
    key = _cache_key(instructions, schema_id, input_text)
    cached = cache.get(key)
    if cached is not None:
        try:
            return _coerce_cached_value(schema, cached)
        except _CacheCoercionError as exc:
            logger.warning(
                "[FMTools] Cache schema mismatch for key %s: %s. Recomputing.",
                key,
                exc,
            )

    result = await session.respond(input_text, generating=schema)
    cache.put(key, result)
    return cast("T", result)


async def cached_stream_extract(
    source_iterable: Iterable[Any] | AsyncIterable[Any],
    schema: type[T],
    cache: ExtractionCache,
    instructions: str = "Extract data.",
) -> AsyncGenerator[T, None]:
    """
    Async generator that wraps ``stream_extract`` with per-chunk caching.

    Each chunk is looked up in *cache* before calling the FM.  Results are
    cached for future runs, making re-processing of the same data free.
    """
    schema_id = _schema_identifier(schema)
    model = create_model()
    session = create_session(instructions=instructions, model=model)
    is_async = isinstance(source_iterable, AsyncIterable)

    if is_async:
        async_source = cast("AsyncIterable[Any]", source_iterable)
        async for chunk in async_source:
            chunk_str = str(chunk)
            key = _cache_key(instructions, schema_id, chunk_str)
            cached = cache.get(key)
            if cached is not None:
                try:
                    yield _coerce_cached_value(schema, cached)
                    continue
                except _CacheCoercionError as exc:
                    logger.warning(
                        "[FMTools] Cache schema mismatch for key %s: %s. Recomputing.",
                        key,
                        exc,
                    )

            result = await session.respond(chunk_str, generating=schema)
            cache.put(key, result)
            yield result

            # Re-create session per chunk (clear mode) to avoid context buildup
            session = create_session(instructions=instructions, model=model)
        return

    sync_source = cast("Iterable[Any]", source_iterable)
    for chunk in sync_source:
        chunk_str = str(chunk)
        key = _cache_key(instructions, schema_id, chunk_str)
        cached = cache.get(key)
        if cached is not None:
            try:
                yield _coerce_cached_value(schema, cached)
                continue
            except _CacheCoercionError as exc:
                logger.warning(
                    "[FMTools] Cache schema mismatch for key %s: %s. Recomputing.",
                    key,
                    exc,
                )

        result = await session.respond(chunk_str, generating=schema)
        cache.put(key, result)
        yield result

        # Re-create session per chunk (clear mode) to avoid context buildup
        session = create_session(instructions=instructions, model=model)


# ---------------------------------------------------------------------------
# Decorator variant
# ---------------------------------------------------------------------------


def cached_local_extract(
    schema: type[T],
    cache: ExtractionCache,
    retries: int = 3,
    debug_timing: bool = False,
) -> Callable[[F], F]:
    """
    A decorator combining ``@local_extract`` behaviour with extraction caching.

    Identical to ``@local_extract`` but checks the cache before calling the FM.
    On a cache hit the Neural Engine is bypassed entirely.

    Parameters
    ----------
    schema:
        A class decorated with ``@apple_fm_sdk.generable()``.
    cache:
        An :class:`ExtractionCache` instance.
    retries:
        Number of retry attempts on transient errors.
    debug_timing:
        If ``True``, log wall-clock timings.
    """
    if retries < 1:
        raise ValueError("retries must be >= 1")

    def decorator(func: F) -> F:
        _cached_model: ModelProtocol | None = None
        instructions = (func.__doc__ or "Extract the following data.").strip()
        schema_id = _schema_identifier(schema)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal _cached_model

            # Format inputs (same logic as local_extract)
            parts = list(map(str, args))
            input_text = " ".join(parts)
            kw_parts = [f"\n{k}: {v}" for k, v in kwargs.items()]
            input_text += "".join(kw_parts)

            # Check cache first
            key = _cache_key(instructions, schema_id, input_text)
            cached = cache.get(key)
            if cached is not None:
                try:
                    return _coerce_cached_value(schema, cached)
                except _CacheCoercionError as exc:
                    logger.warning(
                        "[FMTools] Cache schema mismatch for key %s: %s. Recomputing.",
                        key,
                        exc,
                    )

            # Lazy model initialisation
            if _cached_model is None:
                _cached_model = create_model()
                try:
                    ensure_model_available(_cached_model, context="cached_local_extract")
                except AppleFMSetupError:
                    _cached_model = None
                    raise

            model = _cached_model

            last_exception: Exception | None = None
            for attempt in range(retries):
                session = create_session(instructions=instructions, model=model)
                try:
                    start_time = time.perf_counter()
                    result = await session.respond(input_text, generating=schema)
                    elapsed = time.perf_counter() - start_time

                    if debug_timing:
                        logger.info(
                            "[FMTools] Extraction completed in %.3fs. Input length: %d chars.",
                            elapsed,
                            len(input_text),
                        )

                    # Store in cache before returning
                    cache.put(key, result)
                    return result

                except _TRANSIENT_ERRORS as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep((2**attempt) * 0.1)
                    continue
                except AppleFMSetupError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to generate structured data: {e}") from e

            raise RuntimeError(
                f"Failed to generate structured data after {retries} attempts: {last_exception}"
            ) from last_exception

        return cast("F", wrapper)

    return decorator
