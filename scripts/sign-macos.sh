#!/usr/bin/env bash
# Sign a locally built retrivio binary on macOS so that the privacy grants the watcher needs
# (System Settings > Privacy & Security > Files and Folders: Documents, Desktop, ...) survive a
# rebuild. macOS identifies a program by its code signature; an unsigned binary gets a new
# code hash on every build and is asked for folder access again, which blocks the launchd
# watcher on a prompt nobody sees.
#
# Usage:
#   scripts/sign-macos.sh [<binary>]          default: target/release/retrivio
#
# Environment:
#   RETRIVIO_CODESIGN_IDENTITY   name of a code-signing identity in your keychain (a self-signed
#                                certificate is enough; see docs/DISTRIBUTION.md). With it, the
#                                grant persists across rebuilds. Without it the binary is signed
#                                ad hoc, which still re-prompts after every build.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "sign-macos: not macOS, nothing to do" >&2
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${1:-${repo_root}/target/release/retrivio}"
if [[ ! -f "$bin" ]]; then
  echo "sign-macos: binary not found: $bin (build it with: cargo build --release -p retrivio)" >&2
  exit 1
fi
if ! command -v codesign >/dev/null 2>&1; then
  echo "sign-macos: codesign not found (install the Xcode Command Line Tools: xcode-select --install)" >&2
  exit 1
fi

identifier="com.stouffer-labs.retrivio"
identity="${RETRIVIO_CODESIGN_IDENTITY:-}"
if [[ -n "$identity" ]]; then
  codesign --force --sign "$identity" --identifier "$identifier" --timestamp=none "$bin"
  echo "sign-macos: signed $bin with identity '$identity' (identifier $identifier)"
else
  codesign --force --sign - --identifier "$identifier" --timestamp=none "$bin"
  echo "sign-macos: ad-hoc signature on $bin (RETRIVIO_CODESIGN_IDENTITY unset); macOS still asks for folder access again after every rebuild"
fi
codesign --verify --strict "$bin"
codesign -dv "$bin"
