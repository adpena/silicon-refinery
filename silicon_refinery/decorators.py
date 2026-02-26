import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import ModelProtocol, create_model, create_session

logger = logging.getLogger("silicon_refinery")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Transient errors that are worth retrying
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)


def local_extract(
    schema: type[T], retries: int = 3, debug_timing: bool = False
) -> Callable[[F], F]:
    """
    A decorator that transforms a Python function into an intelligent, on-device data extractor.

    The docstring of the decorated function serves as the system instruction for the LLM.
    It intercepts the arguments passed to the function, injects them into the local model,
    enforces structured generation according to the provided schema, and returns a fully-validated object.

    Args:
        schema: A class decorated with `@apple_fm_sdk.generable()`.
        retries (int, optional): The number of times to retry generation if an error occurs. Defaults to 3.
        debug_timing (bool, optional): If True, logs the time taken by the Neural Engine. Defaults to False.

    Returns:
        Callable: The wrapped function returning the requested structured schema.
    """
    if retries < 1:
        raise ValueError("retries must be >= 1")

    def decorator(func: F) -> F:
        # Cache model lazily on first call
        _cached_model: ModelProtocol | None = None
        # Pre-compute instructions at decoration time (not per-call)
        instructions = (func.__doc__ or "Extract the following data.").strip()

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal _cached_model

            # Format inputs using list + join pattern
            parts = list(map(str, args))
            input_text = " ".join(parts)
            kw_parts = [f"\n{k}: {v}" for k, v in kwargs.items()]
            input_text += "".join(kw_parts)

            # Cache model at decoration time (lazily on first call)
            if _cached_model is None:
                _cached_model = create_model()
                try:
                    ensure_model_available(_cached_model, context="local_extract")
                except AppleFMSetupError:
                    _cached_model = None
                    raise

            model = _cached_model

            last_exception = None
            for attempt in range(retries):
                # Fresh session per attempt to avoid stale conversation state on retry
                session = create_session(instructions=instructions, model=model)
                try:
                    start_time = time.perf_counter()
                    result = await session.respond(input_text, generating=schema)
                    elapsed = time.perf_counter() - start_time

                    if debug_timing:
                        input_len = len(input_text)
                        logger.info(
                            f"[SiliconRefinery] Extraction completed in {elapsed:.3f}s. Input length: {input_len} chars."
                        )

                    return result
                except _TRANSIENT_ERRORS as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep((2**attempt) * 0.1)
                    continue
                except AppleFMSetupError:
                    # Preserve setup diagnostics.
                    raise
                except Exception as e:
                    # Non-transient errors fail immediately
                    raise RuntimeError(f"Failed to generate structured data: {e}") from e

            raise RuntimeError(
                f"Failed to generate structured data after {retries} attempts: {last_exception}"
            ) from last_exception

        return cast("F", wrapper)

    return decorator
