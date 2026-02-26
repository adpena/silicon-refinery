"""
Integration tests for SiliconRefinery.

These tests verify end-to-end workflows across multiple modules,
with all FM SDK calls mocked.

Covers:
  - Full pipeline: Source >> Extract >> Sink with data flowing through
  - stream_extract feeding into pipeline sink
  - local_extract used inside a pipeline-like flow
  - Error propagation across module boundaries
  - Concurrent stream_extract followed by Sink-style collection
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from silicon_refinery.pipeline import Source, Extract, Sink
from silicon_refinery.async_generators import stream_extract
from silicon_refinery.decorators import local_extract
from .conftest import MockSchema, make_mock_model, make_mock_session


# ========================================================================
# End-to-end pipeline: Source >> Extract >> Sink
# ========================================================================

class TestEndToEndPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_collects_all_results(self):
        """Source feeds 3 items, Extract processes each, Sink collects them."""
        mock_model = make_mock_model(available=True)

        idx = 0

        async def progressive_respond(*args, **kwargs):
            nonlocal idx
            idx += 1
            return MockSchema(name=f"Person_{idx}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=progressive_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            collected = []

            s = Source(["Alice is 30", "Bob is 25", "Charlie is 40"])
            e = Extract(schema=MockSchema, instructions="Extract person info.")
            k = Sink(callback=collected.append)

            pipeline = s >> e >> k
            results = await pipeline.execute()

            assert len(results) == 3
            assert len(collected) == 3
            # Verify progressive results
            names = [r.name for r in results]
            assert names == ["Person_1", "Person_2", "Person_3"]

    @pytest.mark.asyncio
    async def test_pipeline_with_partial_failures(self):
        """Extract fails on some items but continues processing."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError(f"Item {call_count} failed")
            return MockSchema(name=f"OK_{call_count}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            s = Source(["A", "B", "C", "D"])
            e = Extract(schema=MockSchema)
            pipeline = s >> e

            results = await pipeline.execute()

            # Items 1 and 3 succeed (odd call_count), items 2 and 4 fail
            assert len(results) == 2


# ========================================================================
# stream_extract into collection
# ========================================================================

class TestStreamExtractIntegration:

    @pytest.mark.asyncio
    async def test_stream_extract_collects_all(self):
        """stream_extract processes a list and yields structured objects."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="streamed"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["line1", "line2", "line3"],
                schema=MockSchema,
                instructions="Extract data.",
                concurrency=1,
            ):
                results.append(item)

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_stream_extract_with_chunking_and_collection(self):
        """stream_extract with chunking reduces call count."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="chunked"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["a", "b", "c", "d"],
                schema=MockSchema,
                lines_per_chunk=2,
                concurrency=1,
            ):
                results.append(item)

            # 4 lines / 2 per chunk = 2 chunks
            assert len(results) == 2


# ========================================================================
# local_extract decorator used in a flow
# ========================================================================

class TestLocalExtractIntegration:

    @pytest.mark.asyncio
    async def test_decorated_function_used_in_loop(self):
        """Using local_extract to process multiple items sequentially."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="extracted"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=2)
            async def extract_person(text: str):
                """Extract person data from text."""
                pass

            items = ["Alice is 30", "Bob is 25"]
            results = []
            for item in items:
                result = await extract_person(item)
                results.append(result)

            assert len(results) == 2
            assert all(r == MockSchema(name="extracted") for r in results)


# ========================================================================
# Cross-module error propagation
# ========================================================================

class TestCrossModuleErrors:

    @pytest.mark.asyncio
    async def test_model_unavailable_blocks_pipeline(self):
        """If the FM is unavailable, Extract node fails for every item."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=RuntimeError("model offline"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            s = Source(["data1", "data2"])
            e = Extract(schema=MockSchema)
            pipeline = s >> e

            results = await pipeline.execute()
            # All items fail, Extract catches and skips
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_model_unavailable_blocks_local_extract(self):
        """local_extract raises RuntimeError when model is unavailable."""
        mock_model = make_mock_model(available=False, reason="not downloaded")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession"),
        ):

            @local_extract(schema=MockSchema, retries=1)
            async def extract(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Foundation Model is not available"):
                await extract("test")
