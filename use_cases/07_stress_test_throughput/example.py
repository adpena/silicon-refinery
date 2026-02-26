import asyncio
import time
import logging
import apple_fm_sdk as fm
from silicon_refinery import stream_extract

# Enable debug logging for SiliconRefinery
logging.basicConfig(level=logging.INFO)


@fm.generable()
class DataRow:
    category: str = fm.guide(anyOf=["Finance", "Technology", "Healthcare", "Other"])
    is_urgent: bool


def generate_gigantic_dataset(rows: int):
    """Generate a massive stream of unstructured text."""
    templates = [
        "The recent quarterly earnings showed a 20% increase in revenue. Please review the attached spreadsheet immediately.",
        "Server CPU utilization has spiked to 99% in the us-east region. All services are currently degraded.",
        "Patient XYZ requires a follow-up appointment next week for their routine blood work.",
        "Just wanted to say hello and see if you wanted to grab lunch today.",
    ]
    for i in range(rows):
        yield templates[i % len(templates)]


async def main():
    TOTAL_ROWS = 1000  # Adjust this to test larger loads (e.g. 100k)
    CHUNK_SIZE = 5  # Process 5 rows at a time

    print(f"🔥 Stress Testing SiliconRefinery: Processing {TOTAL_ROWS} unstructured records...")
    print(f"Chunk Size: {CHUNK_SIZE} lines per chunk.")

    dataset_stream = generate_gigantic_dataset(TOTAL_ROWS)

    start_time = time.perf_counter()
    processed_count = 0

    # Run the streaming extraction with debug_timing=True
    async for structured_record in stream_extract(
        dataset_stream,
        schema=DataRow,
        instructions="Classify the following text and determine if it requires urgent attention.",
        lines_per_chunk=CHUNK_SIZE,
        debug_timing=True,
    ):
        processed_count += CHUNK_SIZE
        if processed_count % 100 == 0:
            print(f"Progress: {processed_count} / {TOTAL_ROWS} records processed locally...")

    total_time = time.perf_counter() - start_time
    rows_per_sec = TOTAL_ROWS / total_time

    print("\n✅ Stress Test Complete.")
    print("-------------------------------------------------")
    print(f"Total Records: {TOTAL_ROWS}")
    print(f"Total Time:    {total_time:.2f} seconds")
    print(f"Throughput:    {rows_per_sec:.2f} records/second")
    print("-------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
