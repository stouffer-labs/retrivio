#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync tracked files to GitHub via Contents API (no git push).

Usage:
  scripts/publish-gh-api.sh [options]

Options:
  --owner <org_or_user>   GitHub owner (default: $GITHUB_OWNER or "stouffer-labs")
  --repo <name>           Repository name (default: current directory name)
  --branch <name>         Branch to write (default: main)
  --create-repo           Create the GitHub repo if it does not exist
  --public                Visibility when creating repo (default)
  --private               Visibility when creating repo
  --dry-run               Show actions without writing
  -h, --help              Show this help

Notes:
  - Requires: gh CLI authenticated with repo scope.
  - Sync source is a strict built-in allowlist of public files/directories.
  - This avoids accidentally publishing local runtime/debug/private artifacts.
  - Each file becomes a separate commit in GitHub.
  - This command adds/updates tracked files; it does not delete remote-only files.
EOF
}

OWNER="${GITHUB_OWNER:-stouffer-labs}"
REPO="$(basename "$(pwd)")"
BRANCH="main"
DRY_RUN=0
CREATE_REPO=0
VISIBILITY="public"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)
      OWNER="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --create-repo)
      CREATE_REPO=1
      shift
      ;;
    --public)
      VISIBILITY="public"
      shift
      ;;
    --private)
      VISIBILITY="private"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated; run: gh auth login" >&2
  exit 1
fi

if ! gh api "repos/${OWNER}/${REPO}" >/dev/null 2>&1; then
  if [[ "$CREATE_REPO" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: would create repo ${OWNER}/${REPO} (${VISIBILITY})"
    else
      echo "repo not found: creating ${OWNER}/${REPO} (${VISIBILITY})..."
      gh repo create "${OWNER}/${REPO}" "--${VISIBILITY}" >/dev/null
      echo "repo created: ${OWNER}/${REPO}"
    fi
  else
    echo "error: repo '${OWNER}/${REPO}' not found or not accessible (HTTP 404)." >&2
    echo "hint: create it first, or rerun with --create-repo." >&2
    echo "      gh repo create ${OWNER}/${REPO} --public" >&2
    exit 1
  fi
fi

ALLOWLIST=(
  "README.md"
  "Cargo.toml"
  "Cargo.lock"
  "LICENSE"
  ".gitignore"
  "retrivio"
  "crates/retrivio"
  "scripts/publish-gh-api.sh"
  "scripts/retrivio-shell.sh"
  "scripts/retrivio-bedrock-refresh.sh"
)

collect_allowlist_files() {
  local path
  for path in "${ALLOWLIST[@]}"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
    elif [[ -d "$path" ]]; then
      find "$path" -type f ! -name '.DS_Store' -print
    fi
  done | sed 's#^\./##' | sort -u
}

mapfile -t FILES < <(collect_allowlist_files)
SOURCE_MODE="allowlist"

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "nothing to sync (no files found in ${SOURCE_MODE} mode)"
  exit 0
fi

echo "sync target: ${OWNER}/${REPO} (branch=${BRANCH})"
echo "source mode: ${SOURCE_MODE}"
echo "files: ${#FILES[@]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "mode: dry-run"
fi

added=0
updated=0
skipped=0
failed=0

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "skip: $file (not a regular file)"
    skipped=$((skipped + 1))
    continue
  fi

  sha=""
  if probed_sha="$(gh api "repos/${OWNER}/${REPO}/contents/${file}" --jq '.sha' 2>/dev/null)"; then
    if [[ -n "$probed_sha" && "$probed_sha" != "null" ]]; then
      sha="$probed_sha"
    fi
  fi
  content_file="$(mktemp -t retrivio-publish-content.XXXXXX)"
  base64 -i "$file" | tr -d '\n' >"$content_file"
  payload="$(mktemp -t retrivio-publish-payload.XXXXXX)"

  if [[ -n "$sha" ]]; then
    action="update"
    message="Update ${file}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: ${action} ${file}"
      updated=$((updated + 1))
      rm -f "$payload" "$content_file"
      continue
    fi
    jq -n \
      --arg message "$message" \
      --rawfile content "$content_file" \
      --arg sha "$sha" \
      --arg branch "$BRANCH" \
      '{message:$message, content:$content, sha:$sha, branch:$branch}' >"$payload"
    if gh api --method PUT "repos/${OWNER}/${REPO}/contents/${file}" --input "$payload" >/dev/null; then
      echo "updated: $file"
      updated=$((updated + 1))
    else
      echo "failed: $file" >&2
      failed=$((failed + 1))
    fi
  else
    action="add"
    message="Add ${file}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "dry-run: ${action} ${file}"
      added=$((added + 1))
      rm -f "$payload" "$content_file"
      continue
    fi
    jq -n \
      --arg message "$message" \
      --rawfile content "$content_file" \
      --arg branch "$BRANCH" \
      '{message:$message, content:$content, branch:$branch}' >"$payload"
    if gh api --method PUT "repos/${OWNER}/${REPO}/contents/${file}" --input "$payload" >/dev/null; then
      echo "added: $file"
      added=$((added + 1))
    else
      echo "failed: $file" >&2
      failed=$((failed + 1))
    fi
  fi
  rm -f "$payload" "$content_file"
done

echo
echo "summary: added=${added} updated=${updated} skipped=${skipped} failed=${failed}"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
