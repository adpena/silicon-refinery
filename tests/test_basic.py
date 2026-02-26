"""
Basic smoke tests for SiliconRefinery.

These verify that the core API surface is importable and the main
abstractions compose correctly, without requiring a real FM model.
"""

from unittest.mock import MagicMock, patch

from silicon_refinery.async_generators import stream_extract
from silicon_refinery.decorators import local_extract
from silicon_refinery.pipeline import Extract, Sink, Source

from .conftest import MockSchema


async def test_local_extract_decorator():
    """Decorator wraps a function correctly."""

    @local_extract(schema=MockSchema, retries=1)
    async def extract_dummy(text: str) -> MockSchema:
        """Extract a name."""
        pass

    assert extract_dummy.__name__ == "extract_dummy"


async def test_stream_extract():
    """stream_extract returns an async generator."""
    generator = stream_extract(["John Doe", "Jane Doe"], schema=MockSchema)
    assert hasattr(generator, "__aiter__")


def test_pipeline_composition():
    """Pipeline nodes compose with >> operator."""
    source = Source(["Test1"])
    mock_model = MagicMock()
    mock_model.is_available.return_value = (True, None)
    with patch("silicon_refinery.pipeline.create_model", return_value=mock_model):
        extract = Extract(schema=MockSchema)
        sink = Sink(print)

        pipeline = source >> extract >> sink
        assert len(pipeline.nodes) == 3
        assert pipeline.nodes[0] == source
        assert pipeline.nodes[1] == extract
        assert pipeline.nodes[2] == sink
