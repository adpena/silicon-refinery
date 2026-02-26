"""
Comprehensive tests for silicon_refinery.pipeline (Source, Extract, Sink, Pipeline).

Covers:
  - Node >> operator composition
  - Pipeline chain building (3+ nodes)
  - Pipeline.execute() -- works as async generator (updated code) or list return
  - Pipeline.collect() returns a list
  - Source: yields all items as async generator
  - Extract: calls FM session.respond for each item, handles errors via on_error
  - Sink: invokes sync and async callbacks, yields items through
  - Edge cases: empty source, single node pipeline, Extract failure modes
  - Integration: Source >> Extract >> Sink end-to-end
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_refinery.exceptions import AppleFMSetupError
from silicon_refinery.pipeline import Extract, Node, Pipeline, Sink, Source

from .conftest import MockSchema, make_mock_model, make_mock_session

# ========================================================================
# Node base class
# ========================================================================


class TestNode:
    def test_rshift_creates_pipeline(self):
        a = Node()
        b = Node()
        result = a >> b
        assert isinstance(result, Pipeline)
        assert result.nodes == (a, b)

    def test_rshift_chaining(self):
        a, b, c = Node(), Node(), Node()
        result = a >> b >> c
        assert isinstance(result, Pipeline)
        assert result.nodes == (a, b, c)


# ========================================================================
# Pipeline composition
# ========================================================================


class TestPipelineComposition:
    def test_two_node_pipeline(self, mock_fm_available):
        s = Source(["a"])
        e = Extract(schema=MockSchema)
        pipeline = s >> e
        assert len(pipeline.nodes) == 2

    def test_three_node_pipeline(self, mock_fm_available):
        s = Source(["a"])
        e = Extract(schema=MockSchema)
        k = Sink(print)
        pipeline = s >> e >> k
        assert len(pipeline.nodes) == 3
        assert pipeline.nodes[0] is s
        assert pipeline.nodes[1] is e
        assert pipeline.nodes[2] is k

    def test_pipeline_rshift_appends_node(self, mock_fm_available):
        s = Source(["a"])
        e = Extract(schema=MockSchema)
        k = Sink(print)
        p = Pipeline(s, e)
        p2 = p >> k
        assert len(p2.nodes) == 3

    def test_empty_pipeline(self):
        p = Pipeline()
        assert p.nodes == ()


# ========================================================================
# Source node
# ========================================================================


class TestSource:
    async def test_yields_all_items(self):
        src = Source(["a", "b", "c"])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == ["a", "b", "c"]

    async def test_empty_source_yields_nothing(self):
        src = Source([])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == []

    async def test_source_with_various_types(self):
        src = Source([1, 2.5, None, {"key": "val"}])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == [1, 2.5, None, {"key": "val"}]

    async def test_source_ignores_incoming_stream(self):
        """Source.process() receives `incoming_stream` but ignores it."""
        src = Source(["x"])
        results = []
        async for item in src.process("this_is_ignored"):
            results.append(item)
        assert results == ["x"]


# ========================================================================
# Extract node
# ========================================================================


class TestExtract:
    def test_extract_validates_model_availability_on_init(self, mock_fm_unavailable):
        with pytest.raises(AppleFMSetupError, match="Foundation Model is not available"):
            Extract(schema=MockSchema)

    async def test_extract_processes_each_item(self, mock_fm_available):
        expected = MockSchema(name="test_result")

        async def fake_stream():
            for item in ["Alice", "Bob"]:
                yield item

        ext = Extract(schema=MockSchema, instructions="Extract names.")
        results = []
        async for item in ext.process(fake_stream()):
            results.append(item)

        assert len(results) == 2
        assert all(r == expected for r in results)

    async def test_extract_creates_session_per_item(self, mock_fm_available):
        async def fake_stream():
            for item in ["A", "B", "C"]:
                yield item

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        # Session created per item
        assert mock_fm_available["session_cls"].call_count == 3

    async def test_extract_uses_custom_instructions(self, mock_fm_available):
        async def fake_stream():
            yield "data"

        ext = Extract(schema=MockSchema, instructions="My custom instruction.")
        async for _ in ext.process(fake_stream()):
            pass

        call_kwargs = mock_fm_available["session_cls"].call_args
        assert call_kwargs[1]["instructions"] == "My custom instruction."

    async def test_extract_converts_item_to_string(self, mock_fm_available):
        async def fake_stream():
            yield 42

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "42"

    async def test_extract_respond_called_with_schema(self, mock_fm_available):
        async def fake_stream():
            yield "data"

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        call_kwargs = mock_fm_available["session"].respond.call_args
        assert call_kwargs[1]["generating"] == MockSchema

    async def test_extract_default_on_error_skips(self):
        """Default on_error='skip' skips failed items."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("item 2 failed")
            return MockSchema(name=f"item_{call_count}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            async def fake_stream():
                for item in ["A", "B", "C"]:
                    yield item

            ext = Extract(schema=MockSchema)  # default on_error="skip"
            results = []
            async for item in ext.process(fake_stream()):
                results.append(item)

            # Item 2 failed, so we get 2 results
            assert len(results) == 2

    async def test_extract_on_error_raise(self):
        """on_error='raise' propagates the exception."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=RuntimeError("boom"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            async def fake_stream():
                yield "data"

            ext = Extract(schema=MockSchema, on_error="raise")
            with pytest.raises(RuntimeError, match="boom"):
                async for _ in ext.process(fake_stream()):
                    pass

    async def test_extract_on_error_yield_none(self):
        """on_error='yield_none' yields None for failed items."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("item 2 failed")
            return MockSchema(name=f"item_{call_count}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            async def fake_stream():
                for item in ["A", "B", "C"]:
                    yield item

            ext = Extract(schema=MockSchema, on_error="yield_none")
            results = []
            async for item in ext.process(fake_stream()):
                results.append(item)

            assert len(results) == 3
            assert results[0] == MockSchema(name="item_1")
            assert results[1] is None
            assert results[2] == MockSchema(name="item_3")

    async def test_extract_propagates_setup_error_even_when_on_error_skip(self):
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=AppleFMSetupError("setup diagnostics should propagate")
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            async def fake_stream():
                yield "A"

            ext = Extract(schema=MockSchema, on_error="skip")
            with pytest.raises(AppleFMSetupError, match="setup diagnostics should propagate"):
                async for _ in ext.process(fake_stream()):
                    pass


# ========================================================================
# Sink node
# ========================================================================


class TestSink:
    async def test_sync_callback_invoked(self):
        collected = []

        async def fake_stream():
            for item in ["a", "b"]:
                yield item

        sink = Sink(callback=collected.append)
        results = []
        async for item in sink.process(fake_stream()):
            results.append(item)

        assert collected == ["a", "b"]
        assert results == ["a", "b"]  # Sink yields items through

    async def test_async_callback_invoked(self):
        collected = []

        async def async_cb(item):
            collected.append(item)

        async def fake_stream():
            for item in ["x", "y"]:
                yield item

        sink = Sink(callback=async_cb)
        results = []
        async for item in sink.process(fake_stream()):
            results.append(item)

        assert collected == ["x", "y"]
        assert results == ["x", "y"]

    async def test_sink_passes_items_through(self):
        """Sink should yield every item it receives, even after callback."""

        async def fake_stream():
            for item in [1, 2, 3]:
                yield item

        sink = Sink(callback=lambda x: None)
        results = []
        async for item in sink.process(fake_stream()):
            results.append(item)

        assert results == [1, 2, 3]

    async def test_sink_empty_stream(self):
        collected = []

        async def fake_stream():
            return
            yield  # make it an async generator

        sink = Sink(callback=collected.append)
        results = []
        async for item in sink.process(fake_stream()):
            results.append(item)

        assert collected == []
        assert results == []

    def test_sink_caches_iscoroutinefunction(self):
        """Sink should determine if callback is async at init time."""

        def sync_cb(x):
            pass

        async def async_cb(x):
            pass

        sync_sink = Sink(callback=sync_cb)
        async_sink = Sink(callback=async_cb)

        assert sync_sink._is_async is False
        assert async_sink._is_async is True


# ========================================================================
# Pipeline.execute() -- supports both async generator and list return
# ========================================================================


class TestPipelineExecute:
    async def test_empty_pipeline_yields_nothing(self):
        """Empty pipeline execute should produce no results."""
        p = Pipeline()
        results = []
        # execute() is an async generator in the updated code
        result = p.execute()
        if hasattr(result, "__aiter__"):
            async for item in result:
                results.append(item)
        else:
            # Old code returned a list or None
            if result is not None:
                results = result if isinstance(result, list) else [result]
        assert results == [] or result is None

    async def test_source_only_pipeline(self):
        """A pipeline with only a Source should yield its items."""
        p = Pipeline(Source(["a", "b"]))
        results = []
        result = p.execute()
        if hasattr(result, "__aiter__"):
            async for item in result:
                results.append(item)
        else:
            results = result if isinstance(result, list) else []
        assert results == ["a", "b"]

    async def test_full_pipeline_source_extract_sink(self, mock_fm_available):
        collected = []
        s = Source(["Alice", "Bob"])
        e = Extract(schema=MockSchema)
        k = Sink(callback=collected.append)

        p = s >> e >> k

        results = []
        result = p.execute()
        if hasattr(result, "__aiter__"):
            async for item in result:
                results.append(item)
        else:
            results = result if isinstance(result, list) else []

        assert len(results) == 2
        assert len(collected) == 2

    async def test_pipeline_with_extract_error_skips_item(self):
        """If Extract fails on one item, it skips it but processes others."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first item fails")
            return MockSchema(name="success")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            s = Source(["A", "B"])
            e = Extract(schema=MockSchema)
            p = s >> e

            results = []
            result = p.execute()
            if hasattr(result, "__aiter__"):
                async for item in result:
                    results.append(item)
            else:
                results = result if isinstance(result, list) else []

            # First item fails, second succeeds
            assert len(results) == 1


# ========================================================================
# Pipeline.collect()
# ========================================================================


class TestPipelineCollect:
    async def test_collect_returns_list(self, mock_fm_available):
        """Pipeline.collect() should always return a list."""
        s = Source(["data"])
        e = Extract(schema=MockSchema)
        p = s >> e
        result = await p.collect()

        assert isinstance(result, list)
        assert len(result) == 1

    async def test_collect_full_pipeline(self, mock_fm_available):
        collected = []
        s = Source(["Alice", "Bob", "Charlie"])
        e = Extract(schema=MockSchema)
        k = Sink(callback=collected.append)
        p = s >> e >> k

        result = await p.collect()

        assert len(result) == 3
        assert len(collected) == 3

    async def test_collect_empty_source(self, mock_fm_available):
        s = Source([])
        p = Pipeline(s)
        result = await p.collect()
        assert result == []


# ========================================================================
# Edge cases
# ========================================================================


class TestPipelineEdgeCases:
    async def test_source_with_none_items(self, mock_fm_available):
        s = Source([None, "valid"])
        e = Extract(schema=MockSchema)
        p = s >> e
        result = await p.collect()
        # Both items should be processed (None gets str(None)="None")
        assert len(result) == 2

    def test_pipeline_is_not_node_subclass(self):
        """Pipeline itself is not a Node."""
        p = Pipeline()
        assert not isinstance(p, Node)

    def test_extract_stores_model_at_init(self):
        """Extract creates the model in __init__."""
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            ext = Extract(schema=MockSchema)
            assert ext._model.raw is mock_model

    def test_extract_stores_on_error(self, mock_fm_available):
        """Extract stores the on_error parameter."""
        ext = Extract(schema=MockSchema, on_error="raise")
        assert ext.on_error == "raise"

    def test_extract_default_on_error_is_skip(self, mock_fm_available):
        """Extract defaults to on_error='skip'."""
        ext = Extract(schema=MockSchema)
        assert ext.on_error == "skip"

    def test_extract_default_instructions(self, mock_fm_available):
        """Extract defaults to a sensible instruction string."""
        ext = Extract(schema=MockSchema)
        assert "Process" in ext.instructions or "structure" in ext.instructions


# ========================================================================
# Extract on_error validation
# ========================================================================


class TestExtractOnErrorValidation:
    """Test that invalid on_error values are rejected."""

    def test_invalid_on_error_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid on_error='log'"):
            Extract(schema=MockSchema, on_error="log")

    def test_valid_on_error_values_accepted(self, mock_fm_available):
        for value in ("skip", "raise", "yield_none"):
            ext = Extract(schema=MockSchema, on_error=value)
            assert ext.on_error == value


# ========================================================================
# Source with async iterables
# ========================================================================


class TestSourceAsyncIterable:
    """Test Source with async iterables."""

    async def test_source_accepts_async_iterable(self):
        async def async_gen():
            for item in ["a", "b", "c"]:
                yield item

        src = Source(async_gen())
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == ["a", "b", "c"]

    async def test_source_async_iterable_in_pipeline(self, mock_fm_available):
        async def async_gen():
            for item in ["Alice", "Bob"]:
                yield item

        collected = []
        s = Source(async_gen())
        e = Extract(schema=MockSchema)
        k = Sink(callback=collected.append)
        p = s >> e >> k

        results = await p.collect()

        assert len(results) == 2
        assert len(collected) == 2


# ========================================================================
# Sink callback error propagation
# ========================================================================


class TestSinkCallbackError:
    """Test Sink behavior when callback raises."""

    async def test_sync_callback_error_propagates(self):
        def bad_callback(item):
            raise RuntimeError("sync callback boom")

        async def fake_stream():
            yield "data"

        sink = Sink(callback=bad_callback)
        with pytest.raises(RuntimeError, match="sync callback boom"):
            async for _ in sink.process(fake_stream()):
                pass

    async def test_async_callback_error_propagates(self):
        async def bad_async_callback(item):
            raise RuntimeError("async callback boom")

        async def fake_stream():
            yield "data"

        sink = Sink(callback=bad_async_callback)
        with pytest.raises(RuntimeError, match="async callback boom"):
            async for _ in sink.process(fake_stream()):
                pass


# ========================================================================
# Pipeline with non-Source first node
# ========================================================================


class TestPipelineWithNonSourceFirst:
    """Test pipeline starting with non-Source node."""

    async def test_extract_as_first_node_raises_type_error(self, mock_fm_available):
        ext = Extract(schema=MockSchema)
        p = Pipeline(ext)
        with pytest.raises(TypeError):
            async for _ in p.execute():
                pass
