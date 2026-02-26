"""
SiliconRefinery: A Zero-Trust Local Data Refinery Framework built on python-apple-fm-sdk

This framework provides intuitive, idiomatic Python patterns (decorators, pipelines, generators, and polars integration)
for extracting and transforming large datasets completely on-device, with zero cloud dependency and zero data egress.
"""

try:
    import apple_fm_sdk
except ImportError:
    raise ImportError(
        "\n\n[SiliconRefinery] Error: 'apple-fm-sdk' is not installed.\n"
        "This framework requires the Apple Foundation Models SDK to be installed manually.\n"
        "Please follow the installation guide: https://github.com/adpena/silicon-refinery#installation\n"
    ) from None

from .decorators import local_extract
from .pipeline import Source, Extract, Sink
from .async_generators import stream_extract
from .debugging import enhanced_debug
# Note: Polars and DSPy extensions are imported directly via their namespaces when needed.

__all__ = ["local_extract", "Source", "Extract", "Sink", "stream_extract", "enhanced_debug"]
