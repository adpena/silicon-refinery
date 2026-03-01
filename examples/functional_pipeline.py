"""
Functional pipeline example for FMTools.

Demonstrates both pure functional transforms and schema extraction steps.
"""

from __future__ import annotations

import asyncio

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    import apple_fm_sdk as fm
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("functional_pipeline.py", exc)

try:
    from fmtools.functional import collect, extract, filter_fn, map_fn, pipe, source
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("functional_pipeline.py", exc)


@fm.generable()
class Person:
    name: str = fm.guide(description="Person name")
    age: int = fm.guide(description="Estimated age as an integer")


async def run_transform_only_pipeline() -> None:
    result = await pipe(
        source(range(1, 11)),
        map_fn(lambda n: n * 3),
        filter_fn(lambda n: n % 2 == 0),
        collect(),
    )()
    print("Transform-only result:", result)


async def run_extraction_pipeline() -> None:
    require_apple_fm("functional_pipeline.py")

    text_rows = [
        "Alice is 31 and works in product engineering.",
        "Bob is 27 and runs operations.",
    ]

    people = await pipe(
        source(text_rows),
        extract(Person, instructions="Extract person name and integer age."),
        collect(),
    )()

    print("Extraction result:")
    for person in people:
        print(f"  - {person}")


async def main() -> None:
    await run_transform_only_pipeline()
    await run_extraction_pipeline()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
