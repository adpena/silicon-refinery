# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Simple Inference Example

This example demonstrates the simplest way to use the Foundation Models SDK
to create a session and get responses.
"""

import asyncio
import apple_fm_sdk as fm


async def main():
    """Run a simple inference session."""
    print("=== Simple Inference Example ===\n")

    # Check if the model is available
    model = fm.SystemLanguageModel()
    is_available, reason = model.is_available()

    if not is_available:
        print(f"Model not available: {reason}")
        return

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
    asyncio.run(main())
