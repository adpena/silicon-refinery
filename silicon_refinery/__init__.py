"""
local_refinery: A Zero-Trust Local Data Refinery Framework built on python-apple-fm-sdk

This framework provides intuitive, idiomatic Python patterns (decorators, pipelines, generators, and polars integration)
for extracting and transforming large datasets completely on-device, with zero cloud dependency and zero data egress.
"""

from .decorators import local_extract
from .pipeline import Source, Extract, Sink
from .async_generators import stream_extract
# Note: Polars and DSPy extensions are imported directly via their namespaces when needed.

__all__ = ["local_extract", "Source", "Extract", "Sink", "stream_extract"]
