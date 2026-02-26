# Adapted from the python-apple-fm-sdk examples.
# Original examples: Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Modifications: Copyright (c) 2026 Andres D. Pena. MIT License.

"""
Streaming Response Example

This example demonstrates how to stream responses from the model,
receiving chunks of text as they are generated.
"""

import asyncio

from examples._support import AppleFMSetupError, require_apple_fm


async def main():
    """Run a streaming inference session."""
    print("=== Streaming Response Example ===\n")

    fm, _model = require_apple_fm("streaming_example.py")

    # Create a session
    session = fm.LanguageModelSession(instructions="You are a helpful assistant.")

    # Stream a response
    prompt = "Tell me a short story about a cat."
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)

    # Iterate through response chunks as they arrive
    last_text = ""
    async for chunk in session.stream_response(prompt):
        text = str(chunk)
        # Some SDK implementations stream cumulative text; print only deltas so
        # output stays readable across both cumulative and token-delta modes.
        if text.startswith(last_text):
            print(text[len(last_text) :], end="", flush=True)
        else:
            print(text, end="", flush=True)
        last_text = text

    print("\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
