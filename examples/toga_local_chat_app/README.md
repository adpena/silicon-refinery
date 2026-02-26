# SiliconRefineryChat (Toga + Briefcase)

Standalone repo mirror: [adpena/silicon-refinery-chat](https://github.com/adpena/silicon-refinery-chat)

A demo desktop app for Apple Foundation Models that runs fully local and is designed for realtime UX:

- Realtime streaming responses
- Sidebar vertical chat tabs (Codex-style stacked conversations)
- SQLite-backed multi-chat persistence + resume
- Query-derived unique chat names
- Settings modal (tone/depth/verbosity/citations)
- Automatic steering interjection: interrupt in-flight generation, inject the steering update into history, and rerun
- Codex-style rolling compaction for long conversations
- Familiar slash commands: `/help`, `/new`, `/clear`, `/export`

## Quickstart

```bash
cd examples/toga_local_chat_app
uv venv .venv
source .venv/bin/activate
# Requires CPython 3.13+
uv pip install briefcase toga-cocoa apple-fm-sdk
briefcase dev
```

## Run from repository root

From `/Users/adpena/PycharmProjects/silicon-refinery`:

```bash
silicon-refinery chat

# or the explicit uv/briefcase invocation:
uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app briefcase dev
```

`silicon-refinery chat` now prefers free-threaded CPython (`3.14t` then `3.13t`) automatically and only falls back to standard-GIL when no no-GIL runtime is available.

For maximum demo stability, force standard-GIL directly:

```bash
silicon-refinery chat --standard-gil
```

Or run without Briefcase packaging:

```bash
silicon-refinery chat --python

# or:
uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app python -m silicon_refinery_chat
```

## Local run without Briefcase packaging

```bash
cd examples/toga_local_chat_app
source .venv/bin/activate
PYTHONPATH=src python -m silicon_refinery_chat
```

## Package with Briefcase

```bash
cd examples/toga_local_chat_app
source .venv/bin/activate
briefcase create macOS
briefcase build macOS
# Local-only build on this machine
briefcase package macOS --adhoc-sign

# Redistributable build (Developer ID signing)
briefcase package macOS --identity "Developer ID Application: <YOUR NAME> (<TEAM_ID>)" --no-notarize
APPLE_NOTARY_PROFILE="<YOUR_NOTARY_PROFILE>" \
  ../../scripts/notarize_macos_artifact.sh --artifact "dist/SiliconRefineryChat-<VERSION>.dmg" --app-name "SiliconRefineryChat.app"
```

`--adhoc-sign` artifacts are local-only and will trigger Gatekeeper warnings on other machines. Use Developer ID + notarization for user-facing releases.

## Roadmap

- TODO: Add support for attachments with a nonblocking ingest pipeline and streaming-safe UI integration.
- TODO: Build conversation-query mode so users can start a chat that queries the sqlite database containing all prior conversations.
- TODO: Implement durable memory behavior so relevant context can persist and be reused safely across sessions.
- TODO: Harden context compaction and add targeted tests for guardrail behavior and error handling paths.
- TODO: Test the API on real-world data and harden/flesh out the Python library accordingly.
- TODO: Keep pushing the frontier with modern Python + Apple FM SDK + the broader OSS ecosystem.

## Steering interjection behavior

If steering changes while a response is streaming:

1. Current inference is interrupted.
2. A high-priority `STEERING_INTERJECTION` message is appended to chat history.
3. The response is regenerated from the same query using updated steering + full history context.

This keeps steering responsive without forcing the user to resend manually.

## UI responsiveness + performance notes

- Context compaction work is also offloaded to avoid jank on long histories.
- Streaming transcript updates are throttled (`chars delta` + `time interval`) for smooth rendering.
- Busy state is surfaced with a native `ActivityIndicator` while preserving steering interjections.
