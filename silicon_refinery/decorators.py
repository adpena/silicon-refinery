import functools
import time
import logging
import apple_fm_sdk as fm
from typing import TypeVar, Callable, Any, cast

logger = logging.getLogger("silicon_refinery")

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def local_extract(schema: type[T], retries: int = 3, debug_timing: bool = False) -> Callable[[F], F]:
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
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            instructions = func.__doc__ or "Extract the following data."
            instructions = instructions.strip()
            
            # Format inputs gracefully
            input_text = " ".join(map(str, args))
            for k, v in kwargs.items():
                input_text += f"\\n{k}: {v}"
                
            model = fm.SystemLanguageModel()
            is_available, reason = model.is_available()
            if not is_available:
                raise RuntimeError(f"Foundation Model is not available: {reason}")
                
            session = fm.LanguageModelSession(
                model=model,
                instructions=instructions
            )
            
            for attempt in range(retries):
                try:
                    start_time = time.perf_counter()
                    result = await session.respond(input_text, generating=schema)
                    elapsed = time.perf_counter() - start_time
                    
                    if debug_timing:
                        input_len = len(input_text)
                        logger.info(f"[SiliconRefinery] Extraction completed in {elapsed:.3f}s. Input length: {input_len} chars.")
                        
                    return cast(T, result)
                except Exception as e:
                    if attempt == retries - 1:
                        raise RuntimeError(f"Failed to generate structured data after {retries} attempts: {e}")
            
            raise RuntimeError("Exhausted retries")
            
        return cast(F, wrapper)
    return decorator
