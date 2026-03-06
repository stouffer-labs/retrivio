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
  --skip-ci               Append "[skip ci]" to commit messages (default)
  --no-skip-ci            Do not append CI skip marker
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
SKIP_CI=1
MAX_RETRIES="${PUBLISH_GH_API_MAX_RETRIES:-5}"
RETRY_BASE_DELAY_SEC="${PUBLISH_GH_API_RETRY_DELAY_SEC:-1}"

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
    --skip-ci)
      SKIP_CI=1
      shift
      ;;
    --no-skip-ci)
      SKIP_CI=0
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

is_retryable_http_code() {
  case "$1" in
    408|409|425|429|500|502|503|504)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

extract_http_code() {
  local text="${1:-}"
  if [[ "$text" =~ HTTP[[:space:]]+([0-9]{3}) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$text" =~ status[[:space:]]code:[[:space:]]*([0-9]{3}) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

gh_content_sha_with_retry() {
  local file="$1"
  local endpoint="repos/${OWNER}/${REPO}/contents/${file}"
  local attempt=1
  local delay="$RETRY_BASE_DELAY_SEC"
  local err_file err_text code sha

  while true; do
    err_file="$(mktemp -t retrivio-publish-sha.XXXXXX)"
    sha="$(gh api "$endpoint" --jq '.sha' 2>"$err_file" || true)"
    err_text="$(cat "$err_file" 2>/dev/null || true)"
    rm -f "$err_file"

    if [[ -n "$sha" && "$sha" != "null" ]]; then
      printf '%s\n' "$sha"
      return 0
    fi

    code="$(extract_http_code "$err_text" || true)"
    if [[ "$code" == "404" ]]; then
      return 1
    fi
    if [[ -n "$code" ]] && ! is_retryable_http_code "$code"; then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 2
    fi
    if (( attempt >= MAX_RETRIES )); then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 2
    fi

    if [[ -n "$code" ]]; then
      echo "warn: sha probe failed for ${file} (HTTP ${code}); retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    else
      echo "warn: sha probe failed for ${file}; retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    fi
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

gh_put_content_with_retry() {
  local endpoint="$1"
  local payload="$2"
  local file="$3"
  local attempt=1
  local delay="$RETRY_BASE_DELAY_SEC"
  local err_file err_text code

  while true; do
    err_file="$(mktemp -t retrivio-publish-put.XXXXXX)"
    if gh api --method PUT "$endpoint" --input "$payload" >/dev/null 2>"$err_file"; then
      rm -f "$err_file"
      return 0
    fi
    err_text="$(cat "$err_file" 2>/dev/null || true)"
    rm -f "$err_file"

    code="$(extract_http_code "$err_text" || true)"
    if [[ -n "$code" ]] && ! is_retryable_http_code "$code"; then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 1
    fi
    if (( attempt >= MAX_RETRIES )); then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 1
    fi

    if [[ -n "$code" ]]; then
      echo "warn: publish failed for ${file} (HTTP ${code}); retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    else
      echo "warn: publish failed for ${file}; retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    fi
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

gh_repo_accessible_with_retry() {
  local endpoint="repos/${OWNER}/${REPO}"
  local attempt=1
  local delay="$RETRY_BASE_DELAY_SEC"
  local err_file err_text code

  while true; do
    err_file="$(mktemp -t retrivio-publish-repo.XXXXXX)"
    if gh api "$endpoint" >/dev/null 2>"$err_file"; then
      rm -f "$err_file"
      return 0
    fi
    err_text="$(cat "$err_file" 2>/dev/null || true)"
    rm -f "$err_file"
    code="$(extract_http_code "$err_text" || true)"

    if [[ "$code" == "404" ]]; then
      return 1
    fi
    if [[ -n "$code" ]] && ! is_retryable_http_code "$code"; then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 1
    fi
    if (( attempt >= MAX_RETRIES )); then
      [[ -n "$err_text" ]] && echo "$err_text" >&2
      return 1
    fi

    if [[ -n "$code" ]]; then
      echo "warn: repo check failed for ${OWNER}/${REPO} (HTTP ${code}); retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    else
      echo "warn: repo check failed for ${OWNER}/${REPO}; retry ${attempt}/${MAX_RETRIES} in ${delay}s..." >&2
    fi
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated; run: gh auth login" >&2
  exit 1
fi

if ! gh_repo_accessible_with_retry; then
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
  "assets/animated"
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
if [[ "$SKIP_CI" -eq 1 ]]; then
  echo "ci: skip marker enabled ([skip ci])"
else
  echo "ci: skip marker disabled"
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
  sha_status=1
  if probed_sha="$(gh_content_sha_with_retry "$file")"; then
    sha="$probed_sha"
  else
    sha_status=$?
  fi
  if [[ "$sha_status" -eq 2 ]]; then
    echo "failed: $file (unable to probe remote sha)" >&2
    failed=$((failed + 1))
    continue
  fi
  content_file="$(mktemp -t retrivio-publish-content.XXXXXX)"
  base64 -i "$file" | tr -d '\n' >"$content_file"
  payload="$(mktemp -t retrivio-publish-payload.XXXXXX)"

  if [[ -n "$sha" ]]; then
    action="update"
    message="Update ${file}"
    if [[ "$SKIP_CI" -eq 1 ]]; then
      message="${message} [skip ci]"
    fi
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
    if gh_put_content_with_retry "repos/${OWNER}/${REPO}/contents/${file}" "$payload" "$file"; then
      echo "updated: $file"
      updated=$((updated + 1))
    else
      echo "failed: $file" >&2
      failed=$((failed + 1))
    fi
  else
    action="add"
    message="Add ${file}"
    if [[ "$SKIP_CI" -eq 1 ]]; then
      message="${message} [skip ci]"
    fi
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
    if gh_put_content_with_retry "repos/${OWNER}/${REPO}/contents/${file}" "$payload" "$file"; then
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
