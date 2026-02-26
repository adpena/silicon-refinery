import asyncio
import inspect
import logging
from collections.abc import AsyncIterable

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import create_model, create_session

logger = logging.getLogger("silicon_refinery")


class Node:
    """Base class for pipeline nodes."""

    def __rshift__(self, other):
        return Pipeline(self, other)


class Pipeline:
    def __init__(self, *nodes):
        self.nodes = nodes

    def __rshift__(self, other):
        return Pipeline(*self.nodes, other)

    async def execute(self):
        """Async generator that streams pipeline results without buffering."""
        if not self.nodes:
            return
        stream = self.nodes[0].process(None)
        for node in self.nodes[1:]:
            stream = node.process(stream)
        async for item in stream:
            yield item

    async def collect(self):
        """Convenience method that materializes all results into a list."""
        return [item async for item in self.execute()]


class Source(Node):
    def __init__(self, iterable):
        self.iterable = iterable

    async def process(self, incoming_stream):
        if isinstance(self.iterable, AsyncIterable):
            async for item in self.iterable:
                yield item
        else:
            for i, item in enumerate(self.iterable):
                yield item
                if i > 0 and i % 100 == 0:
                    await asyncio.sleep(0)


class Extract(Node):
    _VALID_ON_ERROR = frozenset({"skip", "raise", "yield_none"})

    def __init__(
        self, schema, instructions: str = "Process and structure this input.", on_error="skip"
    ):
        if on_error not in self._VALID_ON_ERROR:
            raise ValueError(
                f"Invalid on_error={on_error!r}. Must be one of: {', '.join(sorted(self._VALID_ON_ERROR))}"
            )
        self.schema = schema
        self.instructions = instructions
        self.on_error = on_error
        self._model = create_model()
        ensure_model_available(self._model, context="pipeline.Extract")

    async def process(self, incoming_stream):
        async for item in incoming_stream:
            # Recreate session per item to avoid context window explosion
            session = create_session(instructions=self.instructions, model=self._model)
            try:
                payload = str(item)
                result = await session.respond(payload, generating=self.schema)
                yield result
            except AppleFMSetupError:
                # Setup diagnostics must always propagate.
                raise
            except Exception as e:
                logger.warning(f"[Extract Node] Failed to process item. Error: {e}")
                if self.on_error == "raise":
                    raise
                elif self.on_error == "yield_none":
                    yield None
                # else: "skip" - just continue


class Sink(Node):
    def __init__(self, callback):
        self.callback = callback
        self._is_async = inspect.iscoroutinefunction(callback)

    async def process(self, incoming_stream):
        async for item in incoming_stream:
            if self._is_async:
                await self.callback(item)
            else:
                self.callback(item)
            yield item
