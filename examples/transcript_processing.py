# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Transcript Processing Example

This example demonstrates how to process transcripts exported from Swift apps
using the Foundation Models SDK for Python. This is a key evaluation workflow
for analyzing session data from your Swift application.

The example shows:
- Loading transcripts exported from Swift
- Analyzing session structure and content
- Extracting metrics and statistics
- Comparing multiple transcripts
"""

import json
from typing import Dict, List, Any


def load_transcript(file_path: str) -> Dict[str, Any]:
    """
    Load a transcript exported from a Swift app.

    Swift code to export transcript:
    ```swift
    import FoundationModels

    let transcript = session.transcript
    let jsonData = try JSONEncoder().encode(transcript)
    try jsonData.write(to: URL(fileURLWithPath: "transcript.json"))
    ```

    Args:
        file_path: Path to the transcript JSON file

    Returns:
        Transcript dictionary with structure:
        {
            "version": 1,
            "type": "FoundationModels.Transcript",
            "transcript": {
                "entries": [...]
            }
        }
    """
    with open(file_path, "r") as f:
        return json.load(f)


def extract_text_from_contents(contents: List[Dict[str, Any]]) -> str:
    """
    Extract text from a contents array.

    Args:
        contents: List of content objects

    Returns:
        Combined text from all text-type contents
    """
    text_parts = []
    for content in contents:
        if content.get("type") == "text":
            text_parts.append(content.get("text", ""))
        elif content.get("type") == "structure":
            # For structured content, convert to string representation
            structure = content.get("structure", {})
            text_parts.append(json.dumps(structure.get("content", {})))
    return " ".join(text_parts)


def analyze_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a transcript and extract key metrics.

    Args:
        transcript: Transcript dictionary from Swift app with structure:
            {
                "version": 1,
                "transcript": {"entries": [...]},
                "type": "FoundationModels.Transcript"
            }

    Returns:
        Analysis results dictionary
    """
    entries = transcript.get("transcript", {}).get("entries", [])

    # Count entry types by role
    instructions_entries = [e for e in entries if e.get("role") == "instructions"]
    user_entries = [e for e in entries if e.get("role") == "user"]
    response_entries = [e for e in entries if e.get("role") == "response"]
    tool_entries = [e for e in entries if e.get("role") == "tool"]

    # Calculate content lengths
    total_user_chars = sum(
        len(extract_text_from_contents(e.get("contents", []))) for e in user_entries
    )
    total_response_chars = sum(
        len(extract_text_from_contents(e.get("contents", []))) for e in response_entries
    )

    # Extract tool calls from response entries
    tool_calls = []
    for entry in response_entries:
        if "toolCalls" in entry:
            tool_calls.extend(entry["toolCalls"])

    # Extract available tools from instructions
    available_tools = []
    for entry in instructions_entries:
        if "tools" in entry:
            available_tools.extend(entry["tools"])

    # Check for structured output (responseFormat)
    has_structured_output = any("responseFormat" in e for e in user_entries)

    # Check for assets (model information)
    assets = []
    for entry in response_entries:
        if "assets" in entry:
            assets.extend(entry["assets"])

    analysis = {
        "total_entries": len(entries),
        "instructions_entries": len(instructions_entries),
        "user_entries": len(user_entries),
        "response_entries": len(response_entries),
        "tool_entries": len(tool_entries),
        "total_user_chars": total_user_chars,
        "total_response_chars": total_response_chars,
        "avg_user_entry_length": total_user_chars / len(user_entries) if user_entries else 0,
        "avg_response_entry_length": total_response_chars / len(response_entries)
        if response_entries
        else 0,
        "tool_calls_count": len(tool_calls),
        "has_tools": len(tool_calls) > 0,
        "available_tools_count": len(available_tools),
        "has_structured_output": has_structured_output,
        "unique_assets": len(set(assets)),
    }

    return analysis


def print_transcript_summary(transcript: Dict[str, Any], analysis: Dict[str, Any]):
    """Print a formatted summary of the transcript."""
    print("=" * 60)
    print("TRANSCRIPT SUMMARY")
    print("=" * 60)

    # Transcript metadata
    version = transcript.get("version", "N/A")
    transcript_type = transcript.get("type", "N/A")
    print(f"\nVersion: {version}")
    print(f"Type: {transcript_type}")

    # Entry statistics
    print("\nEntry Statistics:")
    print(f"  Total entries: {analysis['total_entries']}")
    print(f"  Instructions entries: {analysis['instructions_entries']}")
    print(f"  User entries: {analysis['user_entries']}")
    print(f"  Response entries: {analysis['response_entries']}")
    print(f"  Tool entries: {analysis['tool_entries']}")

    # Content statistics
    print("\nContent Statistics:")
    print(f"  Total user characters: {analysis['total_user_chars']}")
    print(f"  Total response characters: {analysis['total_response_chars']}")
    print(f"  Avg user entry length: {analysis['avg_user_entry_length']:.1f} chars")
    print(f"  Avg response entry length: {analysis['avg_response_entry_length']:.1f} chars")

    # Tool usage
    if analysis["has_tools"]:
        print("\nTool Usage:")
        print(f"  Available tools: {analysis['available_tools_count']}")
        print(f"  Tool calls made: {analysis['tool_calls_count']}")

    # Structured output
    if analysis["has_structured_output"]:
        print("\nStructured Output: Yes (JSON Schema)")

    # Model assets
    if analysis["unique_assets"] > 0:
        print(f"\nModel Assets: {analysis['unique_assets']} unique asset(s)")

    print("=" * 60)


def print_transcript_entries(transcript: Dict[str, Any], max_entries: int = 5):
    """Print the first few entries from the transcript."""
    entries = transcript.get("transcript", {}).get("entries", [])

    print(f"\nFirst {min(max_entries, len(entries))} entries:")
    print("-" * 60)

    for i, entry in enumerate(entries[:max_entries], 1):
        role = entry.get("role", "unknown")
        entry_id = entry.get("id", "N/A")

        print(f"\n[{i}] {role.upper()} (ID: {entry_id[:8]}...)")

        # Show contents
        if "contents" in entry:
            contents = entry["contents"]
            text = extract_text_from_contents(contents)
            if len(text) > 100:
                text = text[:100] + "..."
            if text:
                print(f"    Content: {text}")

        # Show tool calls if present
        if "toolCalls" in entry:
            for tool_call in entry["toolCalls"]:
                tool_name = tool_call.get("name", "unknown")
                print(f"    [Tool Call: {tool_name}]")

        # Show tool name if this is a tool response
        if "toolName" in entry:
            print(f"    [Tool Response: {entry['toolName']}]")

        # Show available tools if present
        if "tools" in entry:
            tool_count = len(entry["tools"])
            print(f"    [Available Tools: {tool_count}]")

        # Show response format if present
        if "responseFormat" in entry:
            format_type = entry["responseFormat"].get("type", "unknown")
            print(f"    [Response Format: {format_type}]")

    if len(entries) > max_entries:
        print(f"\n... and {len(entries) - max_entries} more entries")

    print("-" * 60)


def compare_transcripts(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple transcripts and generate comparison metrics.

    Args:
        transcripts: List of transcript dictionaries

    Returns:
        Comparison results
    """
    analyses = [analyze_transcript(t) for t in transcripts]

    comparison = {
        "count": len(transcripts),
        "avg_entries": sum(a["total_entries"] for a in analyses) / len(analyses),
        "avg_user_entries": sum(a["user_entries"] for a in analyses) / len(analyses),
        "avg_response_entries": sum(a["response_entries"] for a in analyses) / len(analyses),
        "avg_user_chars": sum(a["total_user_chars"] for a in analyses) / len(analyses),
        "avg_response_chars": sum(a["total_response_chars"] for a in analyses) / len(analyses),
        "tool_usage_rate": sum(1 for a in analyses if a["has_tools"]) / len(analyses),
        "structured_output_rate": sum(1 for a in analyses if a["has_structured_output"])
        / len(analyses),
    }

    return comparison


def export_analysis_to_jsonl(transcripts: List[Dict[str, Any]], output_file: str):
    """
    Export transcript analyses to JSONL for further processing.

    Args:
        transcripts: List of transcript dictionaries
        output_file: Path to output JSONL file
    """
    with open(output_file, "w") as f:
        for i, transcript in enumerate(transcripts, 1):
            analysis = analyze_transcript(transcript)
            analysis["transcript_id"] = i
            analysis["version"] = transcript.get("version", 1)
            analysis["type"] = transcript.get("type", "unknown")
            f.write(json.dumps(analysis) + "\n")

    print(f"\n✓ Exported {len(transcripts)} transcript analyses to {output_file}")


def main():
    """Main function demonstrating transcript processing."""
    print("Example: Processing Transcripts from Swift Apps\n")
    print("This demonstrates how to analyze session data exported from")
    print("your Swift app using the Foundation Models Framework.\n")

    # Use tester transcript
    transcript_file = "tests/tester_schemas/test_transcript_full.json"

    # Load and analyze transcript
    print("Loading transcript...")
    transcript = load_transcript(transcript_file)

    print("Analyzing transcript...")
    analysis = analyze_transcript(transcript)

    # Print summary
    print_transcript_summary(transcript, analysis)

    # Print entries
    print_transcript_entries(transcript)

    # Example: Compare multiple transcripts
    print("\n" + "=" * 60)
    print("COMPARING MULTIPLE TRANSCRIPTS")
    print("=" * 60)

    # Create a few more example transcripts by varying entry count
    transcripts = [transcript]
    for i in range(2, 4):
        t = transcript.copy()
        t["transcript"] = {"entries": transcript["transcript"]["entries"][: i + 1]}
        transcripts.append(t)

    comparison = compare_transcripts(transcripts)

    print(f"\nCompared {comparison['count']} transcripts:")
    print(f"  Avg entries per session: {comparison['avg_entries']:.1f}")
    print(f"  Avg user entries: {comparison['avg_user_entries']:.1f}")
    print(f"  Avg response entries: {comparison['avg_response_entries']:.1f}")
    print(f"  Avg user chars: {comparison['avg_user_chars']:.1f}")
    print(f"  Avg response chars: {comparison['avg_response_chars']:.1f}")
    print(f"  Tool usage rate: {comparison['tool_usage_rate']:.1%}")
    print(f"  Structured output rate: {comparison['structured_output_rate']:.1%}")

    # Export analyses
    export_analysis_to_jsonl(transcripts, "transcript_analyses.jsonl")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Export transcripts from your Swift app using the code shown above")
    print("2. Load them with load_transcript()")
    print("3. Analyze with analyze_transcript()")
    print("4. Compare multiple sessions with compare_transcripts()")
    print("5. Use insights to improve your Swift app's features")
    print("=" * 60)


if __name__ == "__main__":
    main()
