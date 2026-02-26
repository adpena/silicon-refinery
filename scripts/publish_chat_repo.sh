#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/examples/toga_local_chat_app"
TEMPLATE_DIR="$ROOT_DIR/standalone/silicon-refinery-chat"

CHAT_REPO="${CHAT_REPO:-adpena/silicon-refinery-chat}"
CHAT_APP_NAME="${CHAT_APP_NAME:-SiliconRefineryChat}"
CHAT_REPO_DESCRIPTION="${CHAT_REPO_DESCRIPTION:-Standalone macOS app for SiliconRefineryChat (Toga + Briefcase).}"
CHAT_REPO_DIR="${CHAT_REPO_DIR:-$HOME/.cache/silicon-refinery-chat-repo}"
CHAT_RELEASE_TAG="${CHAT_RELEASE_TAG:-}"
CHAT_RELEASE_TITLE="${CHAT_RELEASE_TITLE:-}"
CHAT_RELEASE_NOTES="${CHAT_RELEASE_NOTES:-}"
CHAT_SKIP_BUILD="${CHAT_SKIP_BUILD:-0}"
CHAT_SKIP_RELEASE="${CHAT_SKIP_RELEASE:-0}"
CHAT_ALLOW_CREATE="${CHAT_ALLOW_CREATE:-1}"
CHAT_SIGN_MODE="${CHAT_SIGN_MODE:-auto}"
CHAT_SIGN_IDENTITY="${CHAT_SIGN_IDENTITY:-${APPLE_SIGN_IDENTITY:-}}"
CHAT_NOTARIZE_MODE="${CHAT_NOTARIZE_MODE:-auto}"
CHAT_APP_BUNDLE_NAME="${CHAT_APP_BUNDLE_NAME:-${CHAT_APP_NAME}.app}"

usage() {
  cat <<USAGE
Usage: scripts/publish_chat_repo.sh [options]

Options:
  --repo <owner/name>         Target GitHub repo (default: ${CHAT_REPO})
  --repo-dir <path>           Local sync checkout path
  --app-name <name>           Formal app name (default: ${CHAT_APP_NAME})
  --release-tag <tag>         Child repo release tag (default: v<app version>)
  --skip-build                Skip Briefcase create/build/package
  --skip-release              Skip GitHub release upload/update
  --adhoc-sign                Force ad-hoc signing (local-only app; no notarization)
  --sign-identity <identity>  Use Developer ID identity for redistributable signing
  --no-notarize               Disable notarization even when signing with identity
  --notarize-required         Require notarization (fail if creds are unavailable)
  --no-create                 Fail if the target repo does not already exist
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      CHAT_REPO="$2"
      shift 2
      ;;
    --repo-dir)
      CHAT_REPO_DIR="$2"
      shift 2
      ;;
    --app-name)
      CHAT_APP_NAME="$2"
      shift 2
      ;;
    --release-tag)
      CHAT_RELEASE_TAG="$2"
      shift 2
      ;;
    --skip-build)
      CHAT_SKIP_BUILD=1
      shift
      ;;
    --skip-release)
      CHAT_SKIP_RELEASE=1
      shift
      ;;
    --adhoc-sign)
      CHAT_SIGN_MODE="adhoc"
      shift
      ;;
    --sign-identity)
      CHAT_SIGN_MODE="developer-id"
      CHAT_SIGN_IDENTITY="$2"
      shift 2
      ;;
    --no-notarize)
      CHAT_NOTARIZE_MODE="off"
      shift
      ;;
    --notarize-required)
      CHAT_NOTARIZE_MODE="required"
      shift
      ;;
    --no-create)
      CHAT_ALLOW_CREATE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

log() {
  printf '\n[%s] %s\n' "publish-chat-repo" "$*"
}

has_notary_credentials() {
  if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
    return 0
  fi
  if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]]; then
    return 0
  fi
  if [[ -n "${APPLE_NOTARY_KEY_PATH:-}" && -n "${APPLE_NOTARY_KEY_ID:-}" && -n "${APPLE_NOTARY_ISSUER:-}" ]]; then
    return 0
  fi
  return 1
}

validate_developer_id_identity() {
  local identity="$1"
  local listing

  require_cmd security
  listing="$(security find-identity -v -p codesigning 2>/dev/null || true)"
  if [[ -z "$listing" ]]; then
    echo "Error: unable to read codesigning identities from macOS keychain." >&2
    exit 1
  fi

  if [[ "$identity" =~ ^[A-Fa-f0-9]{40}$ ]]; then
    if ! grep -Fq "$identity" <<<"$listing"; then
      echo "Error: signing identity hash '$identity' was not found in keychain." >&2
      exit 1
    fi
    if ! grep -F "$identity" <<<"$listing" | grep -Fq "Developer ID Application"; then
      echo "Error: identity hash '$identity' is not a Developer ID Application certificate." >&2
      exit 1
    fi
    return 0
  fi

  if [[ "$identity" != *"Developer ID Application"* ]]; then
    echo "Error: developer-id mode requires a 'Developer ID Application' identity." >&2
    echo "Provided: $identity" >&2
    exit 1
  fi
  if ! grep -Fq "\"$identity\"" <<<"$listing"; then
    echo "Error: signing identity '$identity' was not found in keychain." >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 2
  fi
}

require_cmd gh
require_cmd git
require_cmd rsync
require_cmd uv

if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh is not authenticated. Run 'gh auth login' or set GH_TOKEN." >&2
  exit 2
fi

if [[ ! -f "$APP_DIR/pyproject.toml" ]]; then
  echo "Error: app project not found at $APP_DIR" >&2
  exit 1
fi

APP_VERSION="$(
APP_DIR="$APP_DIR" python3 - <<'PY'
import os
import pathlib
import tomllib

path = pathlib.Path(os.environ["APP_DIR"]) / "pyproject.toml"
with path.open("rb") as f:
    data = tomllib.load(f)
print(data["project"]["version"])
PY
)"
APP_VERSION="${APP_VERSION//[[:space:]]/}"

ROOT_VERSION="$(
ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
import os
import pathlib
import tomllib

path = pathlib.Path(os.environ["ROOT_DIR"]) / "pyproject.toml"
with path.open("rb") as f:
    data = tomllib.load(f)
print(data["project"]["version"])
PY
)"
ROOT_VERSION="${ROOT_VERSION//[[:space:]]/}"

if [[ "$APP_VERSION" != "$ROOT_VERSION" ]]; then
  echo "Error: chat app version (${APP_VERSION}) does not match root version (${ROOT_VERSION})." >&2
  echo "Set examples/toga_local_chat_app/pyproject.toml versions to ${ROOT_VERSION} before publish." >&2
  exit 1
fi

if [[ ! "$APP_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: expected semantic version format <major>.<minor>.<patch>, got '${APP_VERSION}'." >&2
  exit 1
fi

if [[ -z "$CHAT_RELEASE_TAG" ]]; then
  CHAT_RELEASE_TAG="v${APP_VERSION}"
fi
if [[ -z "$CHAT_RELEASE_TITLE" ]]; then
  CHAT_RELEASE_TITLE="${CHAT_APP_NAME} ${CHAT_RELEASE_TAG}"
fi
if [[ -z "$CHAT_RELEASE_NOTES" ]]; then
  CHAT_RELEASE_NOTES="Automated sync from adpena/silicon-refinery (${CHAT_APP_NAME}, version ${APP_VERSION})."
fi

log "Target repo: ${CHAT_REPO}"
log "App version: ${APP_VERSION}"
log "Sign mode: ${CHAT_SIGN_MODE}"
log "Notarize mode: ${CHAT_NOTARIZE_MODE}"

if gh repo view "$CHAT_REPO" >/dev/null 2>&1; then
  log "Repository exists."
elif [[ "$CHAT_ALLOW_CREATE" == "1" ]]; then
  log "Repository missing. Creating ${CHAT_REPO} as public repo..."
  gh repo create "$CHAT_REPO" \
    --public \
    --description "$CHAT_REPO_DESCRIPTION" \
    --clone=false
else
  echo "Error: repo ${CHAT_REPO} does not exist and --no-create was set." >&2
  exit 1
fi

mkdir -p "$(dirname "$CHAT_REPO_DIR")"
if [[ -d "$CHAT_REPO_DIR/.git" ]]; then
  log "Updating existing checkout at ${CHAT_REPO_DIR}"
  git -C "$CHAT_REPO_DIR" remote set-url origin "https://github.com/${CHAT_REPO}.git"
  git -C "$CHAT_REPO_DIR" fetch origin --prune
  if git -C "$CHAT_REPO_DIR" show-ref --verify --quiet refs/remotes/origin/main; then
    git -C "$CHAT_REPO_DIR" checkout -B main origin/main
  else
    git -C "$CHAT_REPO_DIR" checkout -B main
  fi
else
  rm -rf "$CHAT_REPO_DIR"
  if ! gh repo clone "$CHAT_REPO" "$CHAT_REPO_DIR"; then
    log "Clone fallback for empty/new repo"
    mkdir -p "$CHAT_REPO_DIR"
    git -C "$CHAT_REPO_DIR" init -b main
    git -C "$CHAT_REPO_DIR" remote add origin "https://github.com/${CHAT_REPO}.git"
  fi
fi

SIGNING_PERFORMED=0
if [[ "$CHAT_SKIP_BUILD" != "1" ]]; then
  log "Building macOS app artifact with Briefcase"
  uv sync --project "$APP_DIR" --directory "$APP_DIR"
  BRIEFCASE_CREATE_DIR="$APP_DIR/build/silicon_refinery_chat/macos/app"
  if [[ -d "$BRIEFCASE_CREATE_DIR" ]]; then
    log "Briefcase create already initialized; skipping create step"
  else
    uv run --project "$APP_DIR" --directory "$APP_DIR" briefcase create macOS --no-input
  fi
  uv run --project "$APP_DIR" --directory "$APP_DIR" briefcase build macOS --no-input

  BRIEFCASE_PACKAGE_ARGS=(package macOS --no-input)
  case "$CHAT_SIGN_MODE" in
    developer-id)
      if [[ -z "$CHAT_SIGN_IDENTITY" ]]; then
        echo "Error: --sign-identity or CHAT_SIGN_IDENTITY/APPLE_SIGN_IDENTITY is required for developer-id signing." >&2
        exit 1
      fi
      require_cmd xcrun
      require_cmd codesign
      validate_developer_id_identity "$CHAT_SIGN_IDENTITY"
      BRIEFCASE_PACKAGE_ARGS+=(--identity "$CHAT_SIGN_IDENTITY" --no-notarize)
      SIGNING_PERFORMED=1
      ;;
    adhoc)
      BRIEFCASE_PACKAGE_ARGS+=(--adhoc-sign)
      ;;
    auto)
      if [[ -n "$CHAT_SIGN_IDENTITY" ]]; then
        require_cmd xcrun
        require_cmd codesign
        validate_developer_id_identity "$CHAT_SIGN_IDENTITY"
        BRIEFCASE_PACKAGE_ARGS+=(--identity "$CHAT_SIGN_IDENTITY" --no-notarize)
        SIGNING_PERFORMED=1
      else
        BRIEFCASE_PACKAGE_ARGS+=(--adhoc-sign)
      fi
      ;;
    *)
      echo "Error: unsupported CHAT_SIGN_MODE '$CHAT_SIGN_MODE' (expected auto|developer-id|adhoc)." >&2
      exit 2
      ;;
  esac
  uv run --project "$APP_DIR" --directory "$APP_DIR" briefcase "${BRIEFCASE_PACKAGE_ARGS[@]}"
else
  log "Skipping build by request"
fi

ARTIFACT_SOURCE_DIR="$APP_DIR/dist"
FOUND_ARTIFACTS=()
if [[ -d "$ARTIFACT_SOURCE_DIR" ]]; then
  shopt -s nullglob
  FOUND_ARTIFACTS=(
    "$ARTIFACT_SOURCE_DIR"/*-"$APP_VERSION".dmg
    "$ARTIFACT_SOURCE_DIR"/*-"$APP_VERSION".pkg
    "$ARTIFACT_SOURCE_DIR"/*-"$APP_VERSION".zip
  )
  shopt -u nullglob
fi

if [[ "${#FOUND_ARTIFACTS[@]}" -eq 0 ]]; then
  if [[ "$CHAT_SKIP_BUILD" == "1" ]]; then
    log "No artifacts found (build skipped), continuing without artifact sync"
  else
    echo "Error: no build artifacts found in $ARTIFACT_SOURCE_DIR" >&2
    exit 1
  fi
fi

SHOULD_NOTARIZE=0
case "$CHAT_NOTARIZE_MODE" in
  off)
    SHOULD_NOTARIZE=0
    ;;
  required)
    if ! has_notary_credentials; then
      echo "Error: CHAT_NOTARIZE_MODE=required but no notarization credentials were provided." >&2
      echo "Set APPLE_NOTARY_PROFILE, or APPLE_ID/APPLE_TEAM_ID/APPLE_APP_SPECIFIC_PASSWORD," >&2
      echo "or APPLE_NOTARY_KEY_PATH/APPLE_NOTARY_KEY_ID/APPLE_NOTARY_ISSUER." >&2
      exit 1
    fi
    SHOULD_NOTARIZE=1
    ;;
  auto)
    if [[ "$SIGNING_PERFORMED" == "1" ]] && has_notary_credentials; then
      SHOULD_NOTARIZE=1
    fi
    ;;
  *)
    echo "Error: unsupported CHAT_NOTARIZE_MODE '$CHAT_NOTARIZE_MODE' (expected auto|required|off)." >&2
    exit 2
    ;;
esac

if [[ "$SHOULD_NOTARIZE" == "1" && "${#FOUND_ARTIFACTS[@]}" -gt 0 ]]; then
  NOTARY_SCRIPT="$ROOT_DIR/scripts/notarize_macos_artifact.sh"
  if [[ ! -x "$NOTARY_SCRIPT" ]]; then
    echo "Error: notarization script is missing or not executable: $NOTARY_SCRIPT" >&2
    exit 1
  fi
  for artifact in "${FOUND_ARTIFACTS[@]}"; do
    case "$artifact" in
      *.dmg|*.pkg|*.zip)
        log "Notarizing artifact: $(basename "$artifact")"
        "$NOTARY_SCRIPT" --artifact "$artifact" --app-name "$CHAT_APP_BUNDLE_NAME"
        ;;
      *)
        log "Skipping unsupported notarization artifact type: $artifact"
        ;;
    esac
  done
elif [[ "$SIGNING_PERFORMED" == "1" ]]; then
  log "Developer ID signing used without notarization. Installers may trigger Gatekeeper warnings."
fi

log "Syncing source tree to child repo"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.briefcase/' \
  --exclude 'artifacts/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.egg-info/' \
  --exclude '*.dist-info/' \
  "$APP_DIR/" "$CHAT_REPO_DIR/"

if [[ -d "$TEMPLATE_DIR" ]]; then
  rsync -a "$TEMPLATE_DIR/" "$CHAT_REPO_DIR/"
fi
cp "$ROOT_DIR/LICENSE" "$CHAT_REPO_DIR/LICENSE"

ARTIFACT_DEST_DIR="$CHAT_REPO_DIR/artifacts"
mkdir -p "$ARTIFACT_DEST_DIR"
if [[ "${#FOUND_ARTIFACTS[@]}" -gt 0 ]]; then
  rm -f "$ARTIFACT_DEST_DIR"/*
  for artifact in "${FOUND_ARTIFACTS[@]}"; do
    cp "$artifact" "$ARTIFACT_DEST_DIR/"
  done
fi

log "Ensuring repo metadata"
if gh repo edit "$CHAT_REPO" --description "$CHAT_REPO_DESCRIPTION" >/dev/null 2>&1; then
  log "Repository description updated"
fi

log "Committing and pushing changes (if any)"
if [[ ! -d "$CHAT_REPO_DIR/.git" ]]; then
  echo "Error: expected child checkout at $CHAT_REPO_DIR to contain .git" >&2
  exit 1
fi
if [[ -n "${GH_TOKEN:-}" ]]; then
  git -C "$CHAT_REPO_DIR" remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${CHAT_REPO}.git"
fi
ORIGIN_URL="$(git -C "$CHAT_REPO_DIR" remote get-url origin 2>/dev/null || true)"
if [[ "$ORIGIN_URL" != *"$CHAT_REPO"* ]]; then
  echo "Error: child checkout origin mismatch. Expected repo containing '$CHAT_REPO', got '$ORIGIN_URL'" >&2
  exit 1
fi
if ! git -C "$CHAT_REPO_DIR" config user.email >/dev/null; then
  git -C "$CHAT_REPO_DIR" config user.email "${GIT_COMMITTER_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"
fi
if ! git -C "$CHAT_REPO_DIR" config user.name >/dev/null; then
  git -C "$CHAT_REPO_DIR" config user.name "${GIT_COMMITTER_NAME:-github-actions[bot]}"
fi
git -C "$CHAT_REPO_DIR" add -A
if ! git -C "$CHAT_REPO_DIR" diff --cached --quiet; then
  SHORT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  git -C "$CHAT_REPO_DIR" commit -m "Sync ${CHAT_APP_NAME} from silicon-refinery (${SHORT_SHA})"
  git -C "$CHAT_REPO_DIR" push origin main
else
  log "No repo content changes detected"
fi

if [[ "$CHAT_SKIP_RELEASE" == "1" ]]; then
  log "Skipping release sync by request"
else
  prune_release_assets() {
    local existing_assets=()
    mapfile -t existing_assets < <(gh release view "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" --json assets --jq '.assets[].name')
    for asset_name in "${existing_assets[@]}"; do
      case "$asset_name" in
        *-"$APP_VERSION".dmg|*-"$APP_VERSION".pkg|*-"$APP_VERSION".zip)
          ;;
        *)
          log "Removing stale release asset: ${asset_name}"
          gh release delete-asset "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" "$asset_name" -y
          ;;
      esac
    done
  }

  if [[ "${#FOUND_ARTIFACTS[@]}" -gt 0 ]]; then
    shopt -s nullglob
    RELEASE_ASSETS=("$ARTIFACT_DEST_DIR"/*)
    shopt -u nullglob
    log "Publishing release assets to ${CHAT_REPO} (${CHAT_RELEASE_TAG})"
    if gh release view "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" >/dev/null 2>&1; then
      gh release upload "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" --clobber "${RELEASE_ASSETS[@]}"
      gh release edit "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" --title "$CHAT_RELEASE_TITLE" --notes "$CHAT_RELEASE_NOTES"
    else
      gh release create "$CHAT_RELEASE_TAG" --repo "$CHAT_REPO" \
        --title "$CHAT_RELEASE_TITLE" \
        --notes "$CHAT_RELEASE_NOTES" \
        "${RELEASE_ASSETS[@]}"
    fi
    prune_release_assets
  else
    log "No release assets found; skipping release upload"
  fi
fi

log "Done."
log "Repo: https://github.com/${CHAT_REPO}"
