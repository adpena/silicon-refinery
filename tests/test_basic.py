import pytest
import apple_fm_sdk as fm
from silicon_refinery import local_extract, stream_extract, Source, Extract, Sink


@fm.generable()
class DummySchema:
    name: str


@pytest.mark.asyncio
async def test_local_extract_decorator():
    @local_extract(schema=DummySchema, retries=1)
    async def extract_dummy(text: str) -> DummySchema:
        """Extract a name."""
        pass

    # We won't actually call the model in CI unless we mock fm.SystemLanguageModel
    # For now, we just ensure the decorator wraps successfully
    assert extract_dummy.__name__ == "extract_dummy"


@pytest.mark.asyncio
async def test_stream_extract():
    generator = stream_extract(["John Doe", "Jane Doe"], schema=DummySchema)
    # Just asserting it yields an async generator successfully
    assert hasattr(generator, "__aiter__")


@pytest.mark.asyncio
async def test_pipeline_composition():
    source = Source(["Test1"])
    extract = Extract(schema=DummySchema)
    sink = Sink(print)

    pipeline = source >> extract >> sink
    assert len(pipeline.nodes) == 3
    assert pipeline.nodes[0] == source
    assert pipeline.nodes[1] == extract
    assert pipeline.nodes[2] == sink
