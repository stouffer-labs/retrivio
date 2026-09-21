#!/usr/bin/env bash
set -euo pipefail

# AWS credential refresh helper for Retrivio Bedrock embedding backend.
#
# This script attempts to refresh AWS credentials using available tooling.
# It can be set as:
#   retrivio config set aws_refresh_cmd "scripts/retrivio-bedrock-refresh.sh <account>"
# or via env:
#   RETRIVIO_AWS_REFRESH_CMD="scripts/retrivio-bedrock-refresh.sh <account>"

account="${1:-}"

if [[ -z "${account}" ]]; then
  echo "usage: retrivio-bedrock-refresh.sh <account-or-profile>" >&2
  echo "  Refreshes AWS credentials for the given account using available tooling." >&2
  echo "  Alternatively, set RETRIVIO_AWS_REFRESH_CMD to your own credential refresh command." >&2
  exit 2
fi

# Try known credential refresh tools in order of preference.
for candidate in \
  "$(command -v isengardcli 2>/dev/null || true)" \
  "${HOME}/Scripts/isengardcli/isengardcli"; do
  if [[ -n "${candidate}" ]] && [[ -x "${candidate}" ]]; then
    exec "${candidate}" add-profile "${account}"
  fi
done

echo "No supported credential refresh tool found." >&2
echo "Set RETRIVIO_AWS_REFRESH_CMD to your own credential refresh command." >&2
exit 1
