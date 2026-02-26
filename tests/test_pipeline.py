"""
Comprehensive tests for silicon_refinery.pipeline (Source, Extract, Sink, Pipeline).

Covers:
  - Node >> operator composition
  - Pipeline chain building (3+ nodes)
  - Pipeline.execute() end-to-end with mocked FM
  - Source: yields all items as async generator
  - Extract: calls FM session.respond for each item, handles errors gracefully
  - Sink: invokes sync and async callbacks, yields items through
  - Edge cases: empty source, single node pipeline, Extract failure
  - Integration: Source >> Extract >> Sink end-to-end
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from silicon_refinery.pipeline import Source, Extract, Sink, Pipeline, Node
from .conftest import MockSchema, make_mock_model


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

    def test_two_node_pipeline(self):
        s = Source(["a"])
        e = Extract(schema=MockSchema)
        pipeline = s >> e
        assert len(pipeline.nodes) == 2

    def test_three_node_pipeline(self):
        s = Source(["a"])
        e = Extract(schema=MockSchema)
        k = Sink(print)
        pipeline = s >> e >> k
        assert len(pipeline.nodes) == 3
        assert pipeline.nodes[0] is s
        assert pipeline.nodes[1] is e
        assert pipeline.nodes[2] is k

    def test_pipeline_rshift_appends_node(self):
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

    @pytest.mark.asyncio
    async def test_yields_all_items(self):
        src = Source(["a", "b", "c"])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_empty_source_yields_nothing(self):
        src = Source([])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == []

    @pytest.mark.asyncio
    async def test_source_with_various_types(self):
        src = Source([1, 2.5, None, {"key": "val"}])
        results = []
        async for item in src.process(None):
            results.append(item)
        assert results == [1, 2.5, None, {"key": "val"}]

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_extract_creates_session_per_item(self, mock_fm_available):
        async def fake_stream():
            for item in ["A", "B", "C"]:
                yield item

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        # 1 initial model creation; session created per item
        assert mock_fm_available["session_cls"].call_count == 3

    @pytest.mark.asyncio
    async def test_extract_uses_custom_instructions(self, mock_fm_available):
        async def fake_stream():
            yield "data"

        ext = Extract(schema=MockSchema, instructions="My custom instruction.")
        async for _ in ext.process(fake_stream()):
            pass

        call_kwargs = mock_fm_available["session_cls"].call_args
        assert call_kwargs[1]["instructions"] == "My custom instruction."

    @pytest.mark.asyncio
    async def test_extract_converts_item_to_string(self, mock_fm_available):
        async def fake_stream():
            yield 42

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "42"

    @pytest.mark.asyncio
    async def test_extract_skips_failed_items(self, capsys):
        """Extract catches exceptions per-item and prints a warning."""
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

            ext = Extract(schema=MockSchema)
            results = []
            async for item in ext.process(fake_stream()):
                results.append(item)

            # Item 2 failed, so we get 2 results
            assert len(results) == 2
            captured = capsys.readouterr()
            assert "Failed to process item" in captured.out

    @pytest.mark.asyncio
    async def test_extract_respond_called_with_schema(self, mock_fm_available):
        async def fake_stream():
            yield "data"

        ext = Extract(schema=MockSchema)
        async for _ in ext.process(fake_stream()):
            pass

        call_kwargs = mock_fm_available["session"].respond.call_args
        assert call_kwargs[1]["generating"] == MockSchema


# ========================================================================
# Sink node
# ========================================================================

class TestSink:

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


# ========================================================================
# Pipeline.execute() integration
# ========================================================================

class TestPipelineExecute:

    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_none(self):
        p = Pipeline()
        result = await p.execute()
        assert result is None

    @pytest.mark.asyncio
    async def test_source_only_pipeline(self):
        p = Pipeline(Source(["a", "b"]))
        # Source.process(None) should yield items
        # Pipeline.execute collects them
        result = await p.execute()
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_full_pipeline_source_extract_sink(self, mock_fm_available):
        collected = []
        s = Source(["Alice", "Bob"])
        e = Extract(schema=MockSchema)
        k = Sink(callback=collected.append)

        p = s >> e >> k
        result = await p.execute()

        assert len(result) == 2
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_pipeline_execute_returns_list(self, mock_fm_available):
        s = Source(["data"])
        e = Extract(schema=MockSchema)
        p = s >> e
        result = await p.execute()

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
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
            result = await p.execute()

            # First item fails, second succeeds
            assert len(result) == 1


# ========================================================================
# Edge cases
# ========================================================================

class TestPipelineEdgeCases:

    @pytest.mark.asyncio
    async def test_source_with_none_items(self, mock_fm_available):
        s = Source([None, "valid"])
        e = Extract(schema=MockSchema)
        p = s >> e
        result = await p.execute()
        # Both items should be processed (None gets str(None)="None")
        assert len(result) == 2

    def test_pipeline_is_not_node_subclass(self):
        """Pipeline itself is not a Node."""
        p = Pipeline()
        assert not isinstance(p, Node)
