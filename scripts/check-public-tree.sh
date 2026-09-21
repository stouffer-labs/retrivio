#!/usr/bin/env bash
# Fails when the git-tracked tree contains a path outside the known public set.
# Run before every push; CI runs it too.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

allowed='^(README\.md|Cargo\.toml|Cargo\.lock|LICENSE|\.gitignore|\.github/workflows/[^/]+\.yml|\.github/branding/[^/]+|crates/retrivio/.+|docs/DISTRIBUTION\.md|docs/API_HELP\.md|docs/TESTING\.md|docs/skills/.+|scripts/install\.sh|scripts/retrivio-shell\.sh|scripts/retrivio-bedrock-refresh\.sh|scripts/check-public-tree\.sh|scripts/sign-macos\.sh|scripts/smoke\.sh|scripts/harness-check\.sh|Formula/retrivio\.rb|assets/animated/[^/]+\.gif|retrivio)$'
denied='(^|/)(tmp|\.tmp|internal|sessions|superpowers|\.comparo|\.retrivio[^/]*|target)(/|$)|HANDOFF|\.db$|\.jsonl$|\.DS_Store$'

status=0
while IFS= read -r path; do
  if [[ "$path" =~ $denied ]]; then
    echo "DENIED   $path"; status=1
  elif ! [[ "$path" =~ $allowed ]]; then
    echo "UNKNOWN  $path"; status=1
  fi
done < <(git ls-files)

if [[ $status -ne 0 ]]; then
  echo "check-public-tree: tracked files outside the public set (see above)" >&2
  exit 1
fi
echo "check-public-tree: ok ($(git ls-files | wc -l | tr -d ' ') tracked files)"
