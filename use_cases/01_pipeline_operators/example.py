import asyncio
import csv
from silicon_refinery import Source, Extract, Sink
import apple_fm_sdk as fm


@fm.generable()
class LogEntry:
    level: str = fm.guide(anyOf=["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"])
    module: str = fm.guide(description="The service or module emitting the log")
    message: str = fm.guide(description="The core description of the event")


def read_logs_from_csv(filepath):
    """A simple generator that reads rows from our dataset."""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["log_message"]


async def main():
    print("Executing Local Pipeline on server_logs.csv >>\n")

    # We pipe the CSV reader directly into our LLM Extract node, and sink it to stdout
    pipeline = (
        Source(read_logs_from_csv("datasets/server_logs.csv"))
        >> Extract(
            schema=LogEntry,
            instructions="Parse the raw server log string into structured JSON data.",
        )
        >> Sink(
            callback=lambda item: print(
                f"Structured Log | Level: {item.level:7} | Module: {item.module:15} | Message: {item.message}"
            )
        )
    )

    await pipeline.execute()
    print("\nPipeline execution complete. Zero cloud cost incurred.")


if __name__ == "__main__":
    asyncio.run(main())
