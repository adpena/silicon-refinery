# Adapted from the python-apple-fm-sdk examples.
# Original examples: Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Modifications: Copyright (c) 2026 Andres D. Pena. MIT License.

"""
Simple Inference Example

This example demonstrates the simplest way to use the Foundation Models SDK
to create a session and get responses.
"""

import asyncio

from examples._support import AppleFMSetupError, require_apple_fm


async def main():
    """Run a simple inference session."""
    print("=== Simple Inference Example ===\n")

    fm, _model = require_apple_fm("simple_inference.py")

    # Create a session with instructions
    session = fm.LanguageModelSession(
        instructions="You are a helpful assistant that provides concise answers."
    )

    # Send a prompt and get a response
    prompt = "What is the capital of France?"
    print(f"User: {prompt}")

    response = await session.respond(prompt)
    print(f"Assistant: {response}\n")

    # Continue the session
    follow_up = "What is its population?"
    print(f"User: {follow_up}")

    response = await session.respond(follow_up)
    print(f"Assistant: {response}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
