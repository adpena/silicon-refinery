import asyncio
import concurrent.futures
import functools
import inspect
import logging
import sys
import traceback

import apple_fm_sdk as fm

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import create_model, create_session

logger = logging.getLogger("silicon_refinery.debug")
_SYNC_ANALYSIS_TIMEOUT_SECONDS = 30


@fm.generable()
class DebuggingAnalysis:
    error_summary: str = fm.guide(description="A brief summary of what went wrong.")
    possible_causes: list[str] = fm.guide(
        description="List of likely root causes based on the traceback."
    )
    certainty_level: str = fm.guide(
        anyOf=["LOW", "MEDIUM", "HIGH"], description="Confidence in the diagnosis."
    )
    suggested_fix: str = fm.guide(description="Actionable steps to fix the issue.")


def _run_analysis_sync_once(
    exc: Exception, func_name: str, route_to: str, prompt_file: str | None
) -> None:
    coro = _handle_exception(exc, func_name, route_to, prompt_file)
    try:
        asyncio.run(coro)
    finally:
        coro.close()


def _run_analysis_sync_best_effort(
    exc: Exception, func_name: str, route_to: str, prompt_file: str | None
) -> None:
    try:
        _run_analysis_sync_once(exc, func_name, route_to, prompt_file)
        return
    except RuntimeError:
        # asyncio.run() can fail if the current thread already has a running loop.
        pass
    except Exception:
        logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run_analysis_sync_once, exc, func_name, route_to, prompt_file)
        try:
            future.result(timeout=_SYNC_ANALYSIS_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("[SiliconRefinery Debug] AI analysis timed out.")
        except Exception:
            logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
    finally:
        # Do not block original exception propagation on background analysis shutdown.
        executor.shutdown(wait=False, cancel_futures=True)


def enhanced_debug(route_to: str = "stdout", prompt_file: str | None = None):
    """
    A decorator that catches exceptions, prints the traceback, and invokes the Apple Foundation Model
    to perform a detailed root-cause analysis.

    Args:
        route_to: Where to output the analysis ("stdout" or "log").
        prompt_file: If provided, writes a detailed prompt payload to this file, which can be
                     easily copy-pasted into powerful coding agents like Gemini, Claude, or Codex.
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    try:
                        await _handle_exception(e, func.__name__, route_to, prompt_file)
                    except Exception:
                        logger.warning("[SiliconRefinery Debug] AI analysis failed.", exc_info=True)
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _run_analysis_sync_best_effort(e, func.__name__, route_to, prompt_file)
                    raise

            return sync_wrapper

    return decorator


async def _handle_exception(exc: Exception, func_name: str, route_to: str, prompt_file: str | None):
    """Core logic to analyze the exception with the local FM."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Print original traceback
    print(f"\n--- Exception caught in '{func_name}' ---", file=sys.stderr)
    print(tb_str, file=sys.stderr)
    print("--- End of standard traceback ---\n", file=sys.stderr)

    model = create_model()
    try:
        ensure_model_available(model, context="enhanced_debug")
    except AppleFMSetupError as setup_error:
        logger.error("%s", setup_error)
        return

    session = create_session(
        instructions="You are an expert Principal Software Engineer. Analyze the following Python traceback. Identify the root cause, provide a certainty level, and suggest a fix.",
        model=model,
    )

    print("SiliconRefinery is analyzing the crash locally via Neural Engine...", file=sys.stderr)

    try:
        analysis = await session.respond(tb_str, generating=DebuggingAnalysis)

        output = [
            "\n" + "=" * 50,
            f"SiliconRefinery AI Debug Analysis (Certainty: {analysis.certainty_level})",
            "=" * 50,
            f"Summary: {analysis.error_summary}\n",
            "Possible Causes:",
        ]
        for idx, cause in enumerate(analysis.possible_causes, 1):
            output.append(f"  {idx}. {cause}")
        output.append(f"\nSuggested Fix: {analysis.suggested_fix}")
        output.append("=" * 50 + "\n")

        output_str = "\n".join(output)

        if route_to == "stdout":
            print(output_str)
        else:
            logger.error(output_str)

        if prompt_file:
            prompt_content = f"""I encountered a crash in my Python application. Here is the traceback:
```python
{tb_str}
```

A local AI agent performed a preliminary analysis and concluded:
- Summary: {analysis.error_summary}
- Possible Causes: {", ".join(analysis.possible_causes)}
- Suggested Fix: {analysis.suggested_fix}

Please act as an expert developer (like Jeff Dean or a top-tier engineer). Provide a comprehensive, patched version of the code that resolves this issue, along with any necessary explanations.
"""
            with open(prompt_file, "w") as f:
                f.write(prompt_content)
            print(f"Generated AI Agent Prompt written to: {prompt_file}\n")

    except Exception as e:
        print(f"SiliconRefinery AI analysis failed: {e}", file=sys.stderr)
