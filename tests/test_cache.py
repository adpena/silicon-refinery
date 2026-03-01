"""
Tests for fmtools.cache — sqlite3 extraction cache.

Covers:
  - Cache hit / miss behaviour
  - TTL-based expiration
  - clear() and stats()
  - Cache key determinism and collision avoidance
  - cache_extract async helper (miss + hit)
  - cached_stream_extract async generator
  - cached_local_extract decorator
  - JSON serialisation edge cases
  - Automatic directory creation
  - Thread safety under concurrent access
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fmtools.cache import (
    ExtractionCache,
    _cache_key,
    _schema_identifier,
    _serialize,
    cache_extract,
    cached_local_extract,
    cached_stream_extract,
)
from fmtools.exceptions import AppleFMSetupError

from .conftest import MockSchema, make_mock_model, make_mock_session


class StrictSchema:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, StrictSchema):
            return NotImplemented
        return self.name == other.name


# ========================================================================
# Basic get / put / miss / hit
# ========================================================================


class TestCacheBasics:
    def test_cache_miss_returns_none(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        assert cache.get("nonexistent_key") is None

    def test_cache_hit_returns_stored_value(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("key1", {"name": "Alice", "age": 30})
        result = cache.get("key1")
        assert result == {"name": "Alice", "age": 30}

    def test_cache_put_overwrites_existing(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("key1", {"v": 1})
        cache.put("key1", {"v": 2})
        assert cache.get("key1") == {"v": 2}


# ========================================================================
# TTL expiration
# ========================================================================


class TestCacheTTL:
    def test_cache_ttl_expiration(self, tmp_path: Path):
        """Entries older than TTL should be treated as misses and pruned."""
        cache = ExtractionCache(db_path=tmp_path / "test.db", ttl=0.1)
        cache.put("key1", {"data": "old"})

        # Immediately should be a hit
        assert cache.get("key1") is not None

        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_cache_ttl_zero_means_no_expiry(self, tmp_path: Path):
        """TTL=0 should disable expiration entirely."""
        cache = ExtractionCache(db_path=tmp_path / "test.db", ttl=0)
        cache.put("key1", {"data": "forever"})
        time.sleep(0.05)
        assert cache.get("key1") == {"data": "forever"}


# ========================================================================
# clear and stats
# ========================================================================


class TestCacheClear:
    def test_cache_clear(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_cache_clear_resets_stats(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k_miss")  # miss
        cache.clear()
        assert cache.stats() == {"hits": 0, "misses": 0}


class TestCacheStats:
    def test_cache_stats_tracking(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("k1", "v1")

        cache.get("k1")  # hit
        cache.get("k1")  # hit
        cache.get("k_missing")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_cache_stats_initial_zero(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        assert cache.stats() == {"hits": 0, "misses": 0}


# ========================================================================
# Cache key determinism and collision
# ========================================================================


class TestCacheKey:
    def test_cache_key_deterministic(self):
        """Same inputs must always produce the same key."""
        k1 = _cache_key("instr", "MySchema", "hello world")
        k2 = _cache_key("instr", "MySchema", "hello world")
        assert k1 == k2

    def test_cache_key_differs_with_different_inputs(self):
        """Changing any component should change the key."""
        base = _cache_key("instr", "MySchema", "hello")
        diff_instr = _cache_key("other_instr", "MySchema", "hello")
        diff_schema = _cache_key("instr", "OtherSchema", "hello")
        diff_text = _cache_key("instr", "MySchema", "world")

        assert base != diff_instr
        assert base != diff_schema
        assert base != diff_text

    def test_cache_key_is_sha256_hex(self):
        """Keys should be 64-char lowercase hex (SHA-256)."""
        key = _cache_key("a", "b", "c")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_schema_identifier_is_fully_qualified(self):
        class LocalSchema:
            pass

        assert _schema_identifier(LocalSchema) == (
            f"{LocalSchema.__module__}.{LocalSchema.__qualname__}"
        )


# ========================================================================
# cache_extract async function
# ========================================================================


class TestCacheExtract:
    async def test_cache_extract_calls_fm_on_miss(self, tmp_path: Path):
        """On a cache miss, cache_extract should call session.respond()."""
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        expected = MockSchema(name="Alice")

        session = MagicMock()
        session.respond = AsyncMock(return_value=expected)

        result = await cache_extract(
            cache, session, "Alice is 30", MockSchema, instructions="Extract."
        )

        session.respond.assert_called_once()
        # Result should be stored as dict (from vars()) in cache
        assert result == expected

    async def test_cache_extract_returns_cached_on_hit(self, tmp_path: Path):
        """On a cache hit, session.respond() should NOT be called."""
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        expected = MockSchema(name="Bob")

        session = MagicMock()
        session.respond = AsyncMock(return_value=expected)

        # First call populates the cache
        await cache_extract(cache, session, "Bob is 25", MockSchema, instructions="Extract.")
        assert session.respond.call_count == 1

        # Second call should hit cache
        result2 = await cache_extract(
            cache, session, "Bob is 25", MockSchema, instructions="Extract."
        )
        # session.respond should NOT have been called a second time
        assert session.respond.call_count == 1
        assert result2 == MockSchema(name="Bob")

    async def test_cache_extract_uses_fully_qualified_schema_identifier(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")

        def _schema_init(self, name):
            self.name = name

        SchemaA = type("DuplicateSchema", (), {"__init__": _schema_init, "__module__": "pkg.a"})
        SchemaB = type("DuplicateSchema", (), {"__init__": _schema_init, "__module__": "pkg.b"})

        session = MagicMock()
        session.respond = AsyncMock(side_effect=[SchemaA("a"), SchemaB("b")])

        await cache_extract(cache, session, "same input", SchemaA, instructions="Extract.")
        await cache_extract(cache, session, "same input", SchemaB, instructions="Extract.")

        # Same class name but different modules should not collide.
        assert session.respond.call_count == 2

    async def test_cache_extract_recomputes_on_schema_mismatch(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        session = MagicMock()
        session.respond = AsyncMock(return_value=StrictSchema("fresh"))

        key = _cache_key("Extract.", _schema_identifier(StrictSchema), "input")
        cache.put(key, {"unexpected": "shape"})

        result = await cache_extract(cache, session, "input", StrictSchema, instructions="Extract.")

        assert result == StrictSchema("fresh")
        assert session.respond.call_count == 1
        assert cache.get(key) == {"name": "fresh"}


# ========================================================================
# cached_stream_extract async generator
# ========================================================================


class TestCachedStreamExtract:
    async def test_cached_stream_extract_caches_each_chunk(self, tmp_path: Path):
        """Each chunk should be cached; replaying the stream should hit cache."""
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        data = ["Alice is 30", "Bob is 25"]

        mock_model = make_mock_model(available=True)
        results_iter = iter([MockSchema(name="Alice"), MockSchema(name="Bob")])
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=lambda *a, **kw: next(results_iter))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            # First pass: populates cache
            results = []
            async for item in cached_stream_extract(
                data, MockSchema, cache=cache, instructions="Extract."
            ):
                results.append(item)

            assert len(results) == 2
            # respond called twice (one per chunk)
            assert mock_session.respond.call_count == 2

            # Reset respond mock
            mock_session.respond.reset_mock()
            mock_session.respond = AsyncMock(side_effect=AssertionError("should not be called"))

            # Second pass: everything from cache, respond should NOT be called
            results2 = []
            async for item in cached_stream_extract(
                data, MockSchema, cache=cache, instructions="Extract."
            ):
                results2.append(item)

            assert len(results2) == 2

    async def test_cached_stream_extract_accepts_async_source(self, tmp_path: Path):
        """cached_stream_extract should support async iterables as input."""
        cache = ExtractionCache(db_path=tmp_path / "test_async.db")

        async def source():
            yield "Alice is 30"
            yield "Bob is 25"

        mock_model = make_mock_model(available=True)
        results_iter = iter([MockSchema(name="Alice"), MockSchema(name="Bob")])
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=lambda *a, **kw: next(results_iter))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            first_pass = [item async for item in cached_stream_extract(source(), MockSchema, cache)]
            assert [item.name for item in first_pass] == ["Alice", "Bob"]
            assert mock_session.respond.call_count == 2

            mock_session.respond.reset_mock()
            mock_session.respond = AsyncMock(side_effect=AssertionError("should not be called"))

            second_pass = [
                item async for item in cached_stream_extract(source(), MockSchema, cache)
            ]
            assert [item.name for item in second_pass] == ["Alice", "Bob"]
            assert mock_session.respond.call_count == 0

    async def test_cached_stream_extract_recomputes_on_schema_mismatch(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=StrictSchema("fresh"))

        key = _cache_key("Extract data.", _schema_identifier(StrictSchema), "broken")
        cache.put(key, {"wrong": "payload"})

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = [
                item
                async for item in cached_stream_extract(
                    ["broken"], StrictSchema, cache=cache, instructions="Extract data."
                )
            ]

        assert results == [StrictSchema("fresh")]
        assert mock_session.respond.call_count == 1


# ========================================================================
# cached_local_extract decorator
# ========================================================================


class TestCachedLocalExtract:
    async def test_decorator_returns_result(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        expected = MockSchema(name="Alice")
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=expected)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @cached_local_extract(schema=MockSchema, cache=cache, retries=1)
            async def extract_name(text: str):
                """Extract the name."""

            result = await extract_name("Alice is 30")
            # First call returns the MockSchema object from FM
            assert result == expected

    async def test_decorator_uses_cache_on_second_call(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        expected = MockSchema(name="Alice")
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=expected)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @cached_local_extract(schema=MockSchema, cache=cache, retries=1)
            async def extract_name(text: str):
                """Extract the name."""

            await extract_name("Alice is 30")
            await extract_name("Alice is 30")

            # respond only called once — second call served from cache
            assert mock_session.respond.call_count == 1
            assert cache.stats()["hits"] == 1

    async def test_decorator_raises_setup_error_when_model_unavailable(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        mock_model = make_mock_model(available=False, reason="not downloaded")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession"),
        ):

            @cached_local_extract(schema=MockSchema, cache=cache, retries=1)
            async def extract_name(text: str):
                """Extract the name."""

            with pytest.raises(AppleFMSetupError, match="Foundation Model is not available"):
                await extract_name("Alice is 30")

    def test_decorator_retries_validation(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        with pytest.raises(ValueError, match="retries must be >= 1"):
            cached_local_extract(schema=MockSchema, cache=cache, retries=0)

    async def test_decorator_passthrough_setup_error_from_generation(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=AppleFMSetupError("preserve setup diagnostics")
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @cached_local_extract(schema=MockSchema, cache=cache, retries=2)
            async def extract_name(text: str):
                """Extract the name."""

            with pytest.raises(AppleFMSetupError, match="preserve setup diagnostics"):
                await extract_name("Alice is 30")

    async def test_decorator_recomputes_on_cache_schema_mismatch(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=StrictSchema("Alice"))
        instructions = "Extract strict name."
        input_text = "Alice"
        key = _cache_key(instructions, _schema_identifier(StrictSchema), input_text)
        cache.put(key, {"wrong": "shape"})

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @cached_local_extract(schema=StrictSchema, cache=cache, retries=1)
            async def extract_name(text: str):
                """Extract strict name."""

            result = await extract_name(input_text)
            assert result == StrictSchema("Alice")
            assert mock_session.respond.call_count == 1


# ========================================================================
# JSON serialisation edge cases
# ========================================================================


class TestCacheSerialisation:
    def test_cache_handles_json_serialization_edge_cases(self, tmp_path: Path):
        """Values with non-JSON-native types should be serialised via default=str."""
        cache = ExtractionCache(db_path=tmp_path / "test.db")

        # datetime-like string, nested structures
        complex_value = {
            "timestamp": "2025-01-01T00:00:00",
            "items": [1, 2, 3],
            "nested": {"a": "b"},
        }
        cache.put("complex", complex_value)
        assert cache.get("complex") == complex_value

    def test_serialize_object_with_vars(self):
        """_serialize should use vars() for objects with __dict__."""
        obj = MockSchema(name="Alice", age=30)
        result = json.loads(_serialize(obj))
        assert result == {"name": "Alice", "age": 30}

    def test_serialize_plain_dict(self):
        """_serialize should handle plain dicts."""
        result = json.loads(_serialize({"a": 1}))
        assert result == {"a": 1}

    def test_serialize_with_non_json_types(self):
        """_serialize with default=str should handle Path, set, etc."""
        obj = MockSchema(path=Path("/tmp"))
        result = json.loads(_serialize(obj))
        assert result["path"] == "/tmp"

    def test_cache_stores_and_retrieves_list(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("list_key", [1, 2, 3])
        assert cache.get("list_key") == [1, 2, 3]

    def test_cache_stores_and_retrieves_string(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("str_key", "hello")
        assert cache.get("str_key") == "hello"

    def test_cache_stores_and_retrieves_null(self, tmp_path: Path):
        """JSON null is a valid value (not the same as a cache miss)."""
        cache = ExtractionCache(db_path=tmp_path / "test.db")
        cache.put("null_key", None)
        # json.dumps(None) -> "null", json.loads("null") -> None
        # But our get() returns None on miss — we need to distinguish.
        # This test documents the current behavior: None value == miss-like
        # because get returns None for both. This is a known trade-off.
        result = cache.get("null_key")
        assert result is None


# ========================================================================
# Directory creation
# ========================================================================


class TestCacheDirectoryCreation:
    def test_cache_creates_directory_if_missing(self, tmp_path: Path):
        """ExtractionCache should create parent dirs if they don't exist."""
        nested = tmp_path / "deeply" / "nested" / "dir" / "cache.db"
        assert not nested.parent.exists()
        cache = ExtractionCache(db_path=nested)
        assert nested.parent.exists()
        cache.put("k", "v")
        assert cache.get("k") == "v"


# ========================================================================
# Thread safety
# ========================================================================


class TestCacheThreadSafety:
    def test_cache_thread_safety(self, tmp_path: Path):
        """Concurrent get/put from multiple threads should not corrupt the DB."""
        cache = ExtractionCache(db_path=tmp_path / "thread_test.db")
        errors: list[Exception] = []
        num_threads = 8
        ops_per_thread = 50

        def worker(thread_id: int) -> None:
            try:
                for i in range(ops_per_thread):
                    key = f"thread_{thread_id}_item_{i}"
                    cache.put(key, {"thread": thread_id, "item": i})
                    result = cache.get(key)
                    assert result is not None, f"Expected value for {key}"
                    assert result["thread"] == thread_id
                    assert result["item"] == i
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

        # Verify all entries are present
        for tid in range(num_threads):
            for i in range(ops_per_thread):
                key = f"thread_{tid}_item_{i}"
                assert cache.get(key) is not None


# ========================================================================
# repr
# ========================================================================


class TestCacheRepr:
    def test_repr(self, tmp_path: Path):
        cache = ExtractionCache(db_path=tmp_path / "test.db", ttl=3600)
        r = repr(cache)
        assert "ExtractionCache" in r
        assert "ttl=3600" in r
