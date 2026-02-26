import asyncio
import apple_fm_sdk as fm


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
        if not self.nodes:
            return None
        stream = self.nodes[0].process(None)
        for node in self.nodes[1:]:
            stream = node.process(stream)
        results = []
        async for item in stream:
            results.append(item)
        return results


class Source(Node):
    def __init__(self, iterable):
        self.iterable = iterable

    async def process(self, incoming_stream):
        for item in self.iterable:
            yield item
            await asyncio.sleep(0)


class Extract(Node):
    def __init__(self, schema, instructions: str = "Process and structure this input."):
        self.schema = schema
        self.instructions = instructions

    async def process(self, incoming_stream):
        model = fm.SystemLanguageModel()
        async for item in incoming_stream:
            # Recreate session per item to avoid context window explosion
            session = fm.LanguageModelSession(model=model, instructions=self.instructions)
            try:
                payload = str(item)
                result = await session.respond(payload, generating=self.schema)
                yield result
            except Exception as e:
                print(f"[Extract Node Warning] Failed to process item. Error: {e}")


class Sink(Node):
    def __init__(self, callback):
        self.callback = callback

    async def process(self, incoming_stream):
        async for item in incoming_stream:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(item)
            else:
                self.callback(item)
            yield item
