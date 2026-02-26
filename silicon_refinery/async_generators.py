import time
import logging
import apple_fm_sdk as fm
from typing import AsyncGenerator, Iterable, AsyncIterable, Union, Literal, TypeVar

logger = logging.getLogger("silicon_refinery")

T = TypeVar("T")


def _chunk_lines(source: Iterable[str], chunk_size: int) -> Iterable[str]:
    """Helper to chunk an iterable of lines into blocks of lines."""
    chunk = []
    for line in source:
        chunk.append(str(line))
        if len(chunk) >= chunk_size:
            yield "\\n".join(chunk)
            chunk = []
    if chunk:
        yield "\\n".join(chunk)


async def _achunk_lines(source: AsyncIterable[str], chunk_size: int) -> AsyncGenerator[str, None]:
    """Helper to chunk an async iterable of lines into blocks of lines."""
    chunk = []
    async for line in source:
        chunk.append(str(line))
        if len(chunk) >= chunk_size:
            yield "\\n".join(chunk)
            chunk = []
    if chunk:
        yield "\\n".join(chunk)


async def _compact_history(
    model: fm.SystemLanguageModel, instructions: str, session: fm.LanguageModelSession
) -> fm.LanguageModelSession:
    """
    Experimental: Compresses session history into a single summary to maintain
    context without blowing up the context window limit.
    """
    logger.info(
        "[SiliconRefinery Stream] Compacting session history to prevent context window explosion..."
    )
    try:
        # We ask the model to summarize its own context
        summary = await session.respond(
            "Please summarize the key entities, rules, and facts from our conversation so far concisely."
        )
        # Recreate the session, appending the summary to the system instructions
        new_instructions = f"{instructions}\\n\\nPrior Context Summary:\\n{summary}"
        new_session = fm.LanguageModelSession(model=model, instructions=new_instructions)
        return new_session
    except Exception as e:
        logger.warning(
            f"[SiliconRefinery Stream] History compaction failed: {e}. Falling back to clean session."
        )
        return fm.LanguageModelSession(model=model, instructions=instructions)


async def stream_extract(
    source_iterable: Union[Iterable, AsyncIterable],
    schema: type[T],
    instructions: str = "Extract data.",
    lines_per_chunk: int = 1,
    history_mode: Literal["clear", "keep", "hybrid", "compact"] = "clear",
    concurrency: int | None = None,
    debug_timing: bool = False,
) -> AsyncGenerator[T, None]:
    """
    An asynchronous generator that processes a massive stream of incoming data chunks
    through the local Apple Foundation Model, yielding structured schema objects.

    Args:
        source_iterable (Iterable or AsyncIterable): An iterable producing text or dictionary chunks to process.
        schema: A class decorated with `@apple_fm_sdk.generable()`.
        instructions (str, optional): The system prompt for the Foundation Model.
        lines_per_chunk (int, optional): If > 1, chunks incoming items into larger blocks. Defaults to 1.
        history_mode (str):
            - 'clear': (Default) Recreates the session per chunk. Zero memory accumulation.
            - 'keep': Retains session history. May throw ExceededContextWindowSizeError on large streams.
            - 'hybrid': Retains history, but if ExceededContextWindowSizeError occurs, clears it and retries.
            - 'compact': Retains history, but when reaching limits or periodically, summarizes history.
        concurrency (int | None, optional): Number of parallel extraction tasks to run. Defaults to `os.cpu_count()`.
                                     If > 1, chunks are processed concurrently and yielded out-of-order (like imap_unordered).
                                     Forces history_mode='clear'.
        debug_timing (bool, optional): If True, logs processing time and throughput for each chunk.

    Yields:
        schema: The populated structured object.
    """
    import asyncio
    import os

    model = fm.SystemLanguageModel()

    if concurrency is None:
        concurrency = os.cpu_count() or 1

    if concurrency > 1 and history_mode != "clear":
        logger.warning(
            f"[SiliconRefinery Stream] Concurrency ({concurrency}) > 1 requires isolated sessions. Forcing history_mode='clear'."
        )
        history_mode = "clear"

    # Initialize our first session
    session = fm.LanguageModelSession(model=model, instructions=instructions)

    # Apply chunking
    if lines_per_chunk > 1:
        if hasattr(source_iterable, "__aiter__"):
            iterable = _achunk_lines(source_iterable, lines_per_chunk)
        else:
            iterable = _chunk_lines(source_iterable, lines_per_chunk)
    else:
        iterable = source_iterable

    is_async = hasattr(iterable, "__aiter__")

    if concurrency == 1:
        if is_async:
            async for chunk in iterable:
                session, result = await _process_chunk(
                    model, session, instructions, chunk, schema, history_mode, debug_timing
                )
                if result is not None:
                    yield result
        else:
            for chunk in iterable:
                session, result = await _process_chunk(
                    model, session, instructions, chunk, schema, history_mode, debug_timing
                )
                if result is not None:
                    yield result
    else:
        # Concurrent processing (imap_unordered style)
        pending = set()

        if is_async:
            async_iterator = iterable.__aiter__()
        else:
            sync_iterator = iter(iterable)

        while True:
            # Fill pending tasks up to the concurrency limit
            while len(pending) < concurrency:
                try:
                    if is_async:
                        chunk = await async_iterator.__anext__()
                    else:
                        chunk = next(sync_iterator)
                except (StopAsyncIteration, StopIteration):
                    break

                # Each concurrent task gets a fresh session pointer (if 'clear') or its own isolated flow
                task = asyncio.create_task(
                    _process_chunk(
                        model, session, instructions, chunk, schema, history_mode, debug_timing
                    )
                )
                pending.add(task)

            if not pending:
                break

            # Wait for at least one task to complete
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                _, result = task.result()
                if result is not None:
                    yield result


async def _process_chunk(
    model: fm.SystemLanguageModel,
    session: fm.LanguageModelSession,
    instructions: str,
    chunk: str,
    schema: type[T],
    history_mode: str,
    debug_timing: bool,
) -> tuple[fm.LanguageModelSession, T | None]:

    if history_mode == "clear":
        session = fm.LanguageModelSession(model=model, instructions=instructions)

    payload = str(chunk)

    try:
        start_time = time.perf_counter()
        result = await session.respond(payload, generating=schema)
        elapsed = time.perf_counter() - start_time

        if debug_timing:
            chunk_len = len(payload)
            chars_per_sec = chunk_len / elapsed if elapsed > 0 else 0
            logger.info(
                f"[SiliconRefinery Stream] Chunk processed in {elapsed:.3f}s. "
                f"Size: {chunk_len} chars. Throughput: {chars_per_sec:.0f} chars/sec."
            )

        return session, result

    except Exception as e:
        error_str = str(e)
        # Check if the error is related to context window
        if (
            "Context window size exceeded" in error_str
            or "ExceededContextWindowSizeError" in error_str
        ):
            if history_mode == "hybrid":
                logger.info(
                    "[SiliconRefinery Stream] Context window exceeded in 'hybrid' mode. Clearing history and retrying chunk..."
                )
                session = fm.LanguageModelSession(model=model, instructions=instructions)
                return await _process_chunk(
                    model, session, instructions, chunk, schema, "clear", debug_timing
                )

            elif history_mode == "compact":
                # Try to compact the history, then retry
                session = await _compact_history(model, instructions, session)
                return await _process_chunk(
                    model, session, instructions, chunk, schema, "keep", debug_timing
                )  # retry with keep so we don't loop infinitely if compact fails

        logger.error(f"[SiliconRefinery Stream] Failed to process chunk. Error: {e}")
        raise e
