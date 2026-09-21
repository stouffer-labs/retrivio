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
# The binary must not be running: the script refuses to sign one that is (launchd would keep
# the old process, and the signature it carries in memory is the old one). The sequence is
#   retrivio service uninstall && cargo build --release -p retrivio \
#     && scripts/sign-macos.sh && retrivio service install
#
# Environment:
#   RETRIVIO_CODESIGN_IDENTITY   name of a code-signing identity in your keychain (a self-signed
#                                certificate is enough; see docs/DISTRIBUTION.md). With it, the
#                                grant persists across rebuilds. Without it the binary is signed
#                                ad hoc, which still re-prompts after every build.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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
# Absolute path: launchd's command line names the binary that way, and codesign's output does too.
bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"
if ! command -v codesign >/dev/null 2>&1; then
  echo "sign-macos: codesign not found (install the Xcode Command Line Tools: xcode-select --install)" >&2
  exit 1
fi

# Refuse to sign a binary that is executing. lsof finds processes with the file open (the
# running executable itself); pgrep -f finds processes whose command line names the path
# (launchd's watcher). This script's own pid is dropped: the path may be on its command line.
candidates="$( { lsof -t -- "$bin" 2>/dev/null || true; pgrep -f -- "$bin" 2>/dev/null || true; } | sort -un || true)"
running=()
for pid in $candidates; do
  [[ "$pid" == "$$" ]] && continue
  running+=("$pid")
done
if [[ ${#running[@]} -gt 0 ]]; then
  echo "sign-macos: refusing to sign $bin: it is running (pid(s): ${running[*]})." >&2
  ps -o pid=,command= -p "$(IFS=,; echo "${running[*]}")" 2>/dev/null | sed 's/^/  /' >&2 || true
  echo "sign-macos: stop it first, then sign, then start it again on the signed build:" >&2
  echo "  retrivio service uninstall" >&2
  echo "  scripts/sign-macos.sh $bin" >&2
  echo "  retrivio service install" >&2
  exit 1
fi

identifier="com.stouffer-labs.retrivio"
identity="${RETRIVIO_CODESIGN_IDENTITY:-}"
# --timestamp=none on purpose: a secure timestamp is for distributed signatures whose
# certificate may expire; a local build signed with a self-signed identity gains nothing from
# the round trip to Apple's timestamp server.
if [[ -n "$identity" ]]; then
  codesign --force --sign "$identity" --identifier "$identifier" --timestamp=none "$bin"
  echo "sign-macos: signed $bin with identity '$identity' (identifier $identifier)"
else
  codesign --force --sign - --identifier "$identifier" --timestamp=none "$bin"
  echo "sign-macos: ad-hoc signature on $bin (RETRIVIO_CODESIGN_IDENTITY unset); macOS still asks for folder access again after every rebuild"
fi
codesign --verify --strict "$bin"
codesign -dv "$bin"
# The designated requirement is what identifies the program to macOS across builds. With an
# identity it reads: identifier "com.stouffer-labs.retrivio" and certificate leaf = H"<sha1>";
# it must read the same after every rebuild for the folder-access grant to carry over. Ad hoc
# it is the code hash (cdhash), which changes with every build.
echo "sign-macos: designated requirement (compare it after your next rebuild; it must not change):"
codesign -d -r- "$bin" 2>&1 | sed 's/^/  /'
