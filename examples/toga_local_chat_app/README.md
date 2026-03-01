# FMChat (Toga + Briefcase)

Standalone repo mirror: [adpena/fmchat](https://github.com/adpena/fmchat)

A desktop app for Apple Foundation Models that runs fully local and is designed for realtime UX.
It is both a production-oriented local chat application and a reference implementation for Python developers who want to understand and ship SDK-powered chat workflows quickly.

- Realtime streaming responses
- Sidebar vertical chat tabs (Codex-style stacked conversations)
- SQLite-backed multi-chat persistence + resume
- Query-derived unique chat names
- Settings modal (tone/depth/verbosity/citations)
- Automatic steering interjection: interrupt in-flight generation, inject the steering update into history, and rerun
- Codex-style rolling compaction for long conversations
- Familiar slash commands: `/help`, `/new`, `/clear`, `/export`
- Hardened send interaction: `Cmd+Enter` sends, `Enter` keeps newline behavior in compose

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

From `/Users/adpena/PycharmProjects/fmtools`:

```bash
fmtools chat

# or the explicit uv/briefcase invocation:
uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app briefcase dev
```

```text
Actual output excerpt from local run (`fmtools chat --help`):

Usage: fmtools chat [OPTIONS] [APP_ARGS]...
Options:
  --python
  --no-run
  -r, --update-requirements
  --standard-gil
  -h, --help
```

`fmtools chat` now prefers free-threaded CPython (`3.14t` then `3.13t`) automatically and only falls back to standard-GIL when no no-GIL runtime is available.

For maximum demo stability, force standard-GIL directly:

```bash
fmtools chat --standard-gil
```

Or run without Briefcase packaging:

```bash
fmtools chat --python

# or:
uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app python -m fmchat
```

## Local run without Briefcase packaging

```bash
cd examples/toga_local_chat_app
source .venv/bin/activate
PYTHONPATH=src python -m fmchat
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
export CHAT_SIGN_IDENTITY="${CHAT_SIGN_IDENTITY:?Set your Developer ID identity string first}"
export APPLE_NOTARY_PROFILE="${APPLE_NOTARY_PROFILE:?Set your stored notary profile name first}"
briefcase package macOS --identity "$CHAT_SIGN_IDENTITY" --no-notarize
APPLE_NOTARY_PROFILE="$APPLE_NOTARY_PROFILE" \
  ../../scripts/notarize_macos_artifact.sh --artifact "$(ls -t dist/FMChat-*.dmg | head -n 1)" --app-name "FMChat.app"
```

`--adhoc-sign` artifacts are local-only and will trigger Gatekeeper warnings on other machines. Use Developer ID + notarization for user-facing releases.
The parent release script (`../../scripts/publish_chat_repo.sh`) enforces trusted release artifacts by default and refuses to upload ad-hoc/non-notarized builds unless explicitly overridden with `--allow-untrusted-release`.

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
