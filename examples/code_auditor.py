"""
Code auditor example for SiliconRefinery.

Runs `audit_file`, `audit_directory`, and `audit_diff` with a demo backend so
the example is runnable without a live Foundation Model.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery.auditor import (
        audit_diff,
        audit_directory,
        audit_file,
        format_audit_report,
    )
    from silicon_refinery.protocols import get_backend, set_backend
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("code_auditor.py", exc)


class DemoModel:
    def is_available(self) -> tuple[bool, str | None]:
        return (True, None)


class DemoSession:
    async def respond(self, prompt: str, generating=None):
        del prompt, generating
        return json.dumps(
            {
                "issues": [
                    {
                        "line": 1,
                        "severity": "warning",
                        "category": "style",
                        "message": "Use clearer variable names.",
                        "suggestion": "Rename ambiguous identifiers.",
                    }
                ],
                "summary": "Demo audit response",
                "score": 82,
            }
        )


class DemoBackend:
    def create_model(self) -> DemoModel:
        return DemoModel()

    def __call__(self, model: DemoModel, instructions: str) -> DemoSession:
        del model, instructions
        return DemoSession()


async def main() -> None:
    require_apple_fm("code_auditor.py")

    original = get_backend()
    set_backend(DemoBackend())
    try:
        with tempfile.TemporaryDirectory(prefix="sr_audit_") as tmp:
            root = Path(tmp)
            file_a = root / "a.py"
            file_b = root / "b.py"
            file_a.write_text("x=1\n", encoding="utf-8")
            file_b.write_text("y=2\n", encoding="utf-8")

            one = await audit_file(file_a)
            many = await audit_directory(root, max_concurrency=4)
            diff = await audit_diff("+x=1\n-y=2\n")

            print("Single-file score:", one.score)
            print("Directory files audited (max_concurrency=4):", len(many))
            print("Diff score:", diff.score)
            print("\n" + format_audit_report([one, diff]))
    finally:
        set_backend(original)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
