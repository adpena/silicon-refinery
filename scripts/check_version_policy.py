#!/usr/bin/env python3
"""Validate FMTools version synchronization and release bump policy."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tomllib

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(value: str, *, label: str) -> tuple[int, int, int]:
    match = SEMVER_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"{label} must be <major>.<minor>.<patch>; got '{value}'.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def load_toml(path: pathlib.Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_versions(root_dir: pathlib.Path) -> tuple[str, str, str]:
    root_toml = load_toml(root_dir / "pyproject.toml")
    chat_toml = load_toml(root_dir / "examples" / "toga_local_chat_app" / "pyproject.toml")
    root_version = root_toml["project"]["version"]
    chat_project_version = chat_toml["project"]["version"]
    chat_briefcase_version = chat_toml["tool"]["briefcase"]["version"]
    return root_version, chat_project_version, chat_briefcase_version


def find_previous_tag_version(
    *, root_dir: pathlib.Path, current: tuple[int, int, int]
) -> tuple[int, int, int] | None:
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=root_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    versions: list[tuple[int, int, int]] = []
    for line in result.stdout.splitlines():
        tag = line.strip()
        if not tag.startswith("v"):
            continue
        try:
            parsed = parse_semver(tag[1:], label=f"tag '{tag}'")
        except ValueError:
            continue
        if parsed != current:
            versions.append(parsed)
    if not versions:
        return None
    return max(versions)


def format_version(version: tuple[int, int, int]) -> str:
    return f"{version[0]}.{version[1]}.{version[2]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enforce-thousandth-bump",
        action="store_true",
        help="Require current version to be previous tag +1 at the patch/thousandth segment.",
    )
    parser.add_argument(
        "--release-tag",
        default="",
        help="Optional release tag expected for the current version (e.g. v0.0.210).",
    )
    args = parser.parse_args()

    root_dir = pathlib.Path(__file__).resolve().parents[1]

    try:
        root_version, chat_project_version, chat_briefcase_version = load_versions(root_dir)
        root_semver = parse_semver(root_version, label="root project version")
        chat_project_semver = parse_semver(chat_project_version, label="chat project version")
        chat_briefcase_semver = parse_semver(chat_briefcase_version, label="chat briefcase version")
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"[version-policy] {exc}", file=sys.stderr)
        return 1

    if root_semver != chat_project_semver or root_semver != chat_briefcase_semver:
        print(
            "[version-policy] Version mismatch detected:\n"
            f"  root:            {root_version}\n"
            f"  chat project:    {chat_project_version}\n"
            f"  chat briefcase:  {chat_briefcase_version}",
            file=sys.stderr,
        )
        return 1

    if args.release_tag:
        expected_tag = f"v{format_version(root_semver)}"
        if args.release_tag != expected_tag:
            print(
                f"[version-policy] Release tag mismatch: expected {expected_tag}, got {args.release_tag}",
                file=sys.stderr,
            )
            return 1

    if args.enforce_thousandth_bump:
        try:
            previous = find_previous_tag_version(root_dir=root_dir, current=root_semver)
        except subprocess.CalledProcessError as exc:
            print(f"[version-policy] Unable to inspect git tags: {exc}", file=sys.stderr)
            return 1
        if previous is not None:
            expected = (previous[0], previous[1], previous[2] + 1)
            if root_semver != expected:
                print(
                    "[version-policy] Thousandth-place increment required.\n"
                    f"  previous tag: {format_version(previous)}\n"
                    f"  expected:     {format_version(expected)}\n"
                    f"  found:        {format_version(root_semver)}",
                    file=sys.stderr,
                )
                return 1

    print(f"[version-policy] OK ({format_version(root_semver)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
