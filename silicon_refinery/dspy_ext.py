import asyncio
import collections
import concurrent.futures
import contextlib
import weakref

import dspy

from .exceptions import ensure_model_available
from .protocols import create_model, create_session


def _shutdown_executor(executor: concurrent.futures.ThreadPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=True)


class AppleFMLM(dspy.LM):
    """
    A custom DSPy Language Model wrapper that routes inference through the
    local, zero-latency apple_fm_sdk.
    """

    def __init__(self, model_name="system_foundation_model"):
        # We pass model to the parent class to satisfy DSPy requirements
        super().__init__(model=model_name)
        self.fm_model = create_model()
        ensure_model_available(self.fm_model, context="AppleFMLM")
        self.kwargs = {
            "temperature": 0.0,  # Not configurable directly in FM SDK but good to declare
            "max_tokens": 1024,
        }
        self.provider = "apple_fm"
        self.history = collections.deque(maxlen=1000)
        self._closed = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._executor_finalizer = weakref.finalize(self, _shutdown_executor, self._executor)

    def close(self) -> None:
        """Release background executor resources. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._executor_finalizer.alive:
            self._executor_finalizer()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("AppleFMLM is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def basic_request(self, prompt: str, **kwargs):
        """Standard request wrapper required by some DSPy flows."""
        if self._closed:
            raise RuntimeError("AppleFMLM is closed")

        session = create_session(instructions="", model=self.fm_model)

        async def _call():
            return await session.respond(prompt)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            future = self._executor.submit(asyncio.run, _call())
            response = future.result(timeout=60)
        else:
            response = asyncio.run(_call())

        return [str(response)]

    def __call__(self, prompt=None, messages=None, **kwargs):
        """The primary execution method for DSPy modules."""
        # DSPy v2.5+ passes 'messages' which is a list of dicts.
        if messages is not None:
            # Flatten the messages into a single prompt string for Apple FM SDK
            prompt_str = "\n".join(msg.get("content", "") for msg in messages)
        elif prompt is not None:
            prompt_str = prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        response = self.basic_request(prompt_str, **kwargs)
        self.history.append({"prompt": prompt_str, "response": response})
        return response
