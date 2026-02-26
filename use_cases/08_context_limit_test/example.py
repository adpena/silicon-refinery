import asyncio
import time
import apple_fm_sdk as fm
from silicon_refinery import local_extract


@fm.generable()
class ContextSummary:
    key_theme: str
    word_count_estimate: int


@local_extract(schema=ContextSummary, debug_timing=False)
async def summarize_context(huge_text: str) -> ContextSummary:
    """Summarize the massive input text and estimate its word count."""
    pass


async def main():
    print("🧠 Testing Apple Foundation Model Context Window Limits...\n")

    base_paragraph = (
        "Apple Silicon Neural Engines provide an incredible leap in local machine learning performance. "
        * 50
    )
    # base_paragraph is ~5000 characters.

    multipliers = [1, 5, 10, 20, 50]  # Testing 5k, 25k, 50k, 100k, 250k characters

    for mult in multipliers:
        test_text = base_paragraph * mult
        char_count = len(test_text)
        print(
            f"Testing Context Size: {char_count:,} characters (approx {char_count // 4:,} tokens)..."
        )

        try:
            start_time = time.perf_counter()
            result = await summarize_context(test_text)
            elapsed = time.perf_counter() - start_time

            print(f"✅ Success! Processed {char_count:,} chars in {elapsed:.2f} seconds.")
            print(f"   -> Theme: {result.key_theme[:50]}...")

        except Exception as e:
            print(f"❌ Failed at {char_count:,} characters.")
            print(f"   Error: {e}")
            break  # Stop testing if we hit the limit

        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
