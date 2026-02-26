import asyncio
import csv
from silicon_refinery import stream_extract
import apple_fm_sdk as fm


@fm.generable()
class Feedback:
    sentiment: str = fm.guide(anyOf=["Positive", "Neutral", "Negative"])
    key_feature: str = fm.guide(description="The primary feature or aspect being discussed")


def yield_reviews(filepath):
    """Yield reviews one by one to simulate an infinite stream."""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["review_text"]


async def main():
    print("Streaming processing locally from product_reviews.csv:\n")

    review_stream = yield_reviews("datasets/product_reviews.csv")

    # We use stream_extract to process the infinite generator memory-efficiently
    async for enriched in stream_extract(
        review_stream, schema=Feedback, instructions="Analyze user feedback and extract sentiment."
    ):
        print(f"[{enriched.sentiment:8}] Focus: {enriched.key_feature}")


if __name__ == "__main__":
    asyncio.run(main())
