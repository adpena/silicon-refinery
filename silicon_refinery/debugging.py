import functools
import traceback
import sys
import apple_fm_sdk as fm
import logging

logger = logging.getLogger("silicon_refinery.debug")


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
        # We handle both sync and async functions gracefully
        import asyncio

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    await _handle_exception(e, func.__name__, route_to, prompt_file)
                    raise e

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # To run async model in sync context, we use asyncio.run
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    if loop.is_running():
                        import nest_asyncio

                        nest_asyncio.apply()
                    asyncio.run(_handle_exception(e, func.__name__, route_to, prompt_file))
                    raise e

            return sync_wrapper

    return decorator


async def _handle_exception(exc: Exception, func_name: str, route_to: str, prompt_file: str | None):
    """Core logic to analyze the exception with the local FM."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Print original traceback
    print(f"\n--- Exception caught in '{func_name}' ---", file=sys.stderr)
    print(tb_str, file=sys.stderr)
    print("--- End of standard traceback ---\n", file=sys.stderr)

    model = fm.SystemLanguageModel()
    if not model.is_available()[0]:
        logger.error("Foundation Model unavailable for debugging analysis.")
        return

    session = fm.LanguageModelSession(
        model=model,
        instructions="You are an expert Principal Software Engineer. Analyze the following Python traceback. Identify the root cause, provide a certainty level, and suggest a fix.",
    )

    print(
        "🔍 SiliconRefinery is analyzing the crash locally via Neural Engine...", file=sys.stderr
    )

    try:
        analysis: DebuggingAnalysis = await session.respond(tb_str, generating=DebuggingAnalysis)

        output = [
            "\n" + "=" * 50,
            f"🧠 SiliconRefinery AI Debug Analysis (Certainty: {analysis.certainty_level})",
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
            print(f"📄 Generated AI Agent Prompt written to: {prompt_file}\n")

    except Exception as e:
        print(f"⚠️ SiliconRefinery AI analysis failed: {e}", file=sys.stderr)
