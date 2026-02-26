# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Streaming Response Example

This example demonstrates how to stream responses from the model,
receiving chunks of text as they are generated.
"""

import asyncio
import apple_fm_sdk as fm


async def main():
    """Run a streaming inference session."""
    print("=== Streaming Response Example ===\n")

    # Check if the model is available
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()

    if not is_available:
        print(f"Model not available: {reason}")
        return

    # Create a session
    session = fm.LanguageModelSession(instructions="You are a helpful assistant.")

    # Stream a response
    prompt = "Tell me a short story about a cat."
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)

    # Iterate through response chunks as they arrive
    async for chunk in session.stream_response(prompt):
        print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
