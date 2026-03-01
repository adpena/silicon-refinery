import asyncio
import atexit
import concurrent.futures
import json
import logging
import os
import threading

import polars as pl

from .protocols import create_model, create_session

logger = logging.getLogger("fmtools")

_loop = None
_thread = None
_lock = threading.Lock()
_FUTURE_RESULT_TIMEOUT_SECONDS = 30.0


def _get_loop():
    global _loop, _thread
    with _lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, daemon=True)
            _thread.start()
    return _loop


def _shutdown_loop():
    """Cleanly stop the background event loop and join its thread."""
    global _loop, _thread
    with _lock:
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
        if _thread is not None:
            _thread.join(timeout=5)
        _loop = None
        _thread = None


atexit.register(_shutdown_loop)


@pl.api.register_expr_namespace("local_llm")
class LocalLLMExpr:
    """
    A Polars extension that allows running local, on-device LLM inference
    directly inside Polars expressions: `df.select(pl.col("text").local_llm.extract(MySchema))`
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def extract(self, schema, instructions="Extract and structure the text.", concurrency=None):
        if concurrency is None:
            concurrency = min(os.cpu_count() or 1, 4)
        elif not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError("concurrency must be an integer >= 1")

        model = create_model()

        # We define a function to process batches of strings
        def process_batch(series: pl.Series) -> pl.Series:
            async def run(s):
                sem = asyncio.Semaphore(concurrency)

                async def process_one(val):
                    if val is None:
                        return None
                    async with sem:
                        try:
                            # Fresh session per row to prevent context window explosion
                            session = create_session(
                                instructions=instructions,
                                model=model,
                            )
                            res = await session.respond(str(val), generating=schema)
                            try:
                                res_dict = vars(res)
                            except TypeError:
                                res_dict = {"_raw": str(res)}
                            return json.dumps(res_dict, default=str)
                        except Exception:
                            logger.exception(
                                "[FMTools Polars] Failed to process row: %s",
                                repr(val)[:200],
                            )
                            return None

                # Process all rows concurrently within the batch
                tasks = [process_one(val) for val in s]
                results = await asyncio.gather(*tasks)
                return list(results)

            # Run the async coroutine via the persistent background thread
            future = asyncio.run_coroutine_threadsafe(run(series), _get_loop())
            try:
                res_list = future.result(timeout=_FUTURE_RESULT_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    "LocalLLMExpr.extract timed out waiting for batch results after "
                    f"{_FUTURE_RESULT_TIMEOUT_SECONDS} seconds"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "LocalLLMExpr.extract failed while waiting for batch results"
                ) from exc
            return pl.Series(res_list)

        # Apply our batch processing function
        return self._expr.map_batches(process_batch, return_dtype=pl.String)
