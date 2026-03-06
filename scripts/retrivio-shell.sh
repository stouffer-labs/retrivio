#!/usr/bin/env bash

# Compatibility shim.
# Preferred setup is adding one line to your shell rc file:
#   eval "$(retrivio init bash)"
# or:
#   eval "$(retrivio init zsh)"

_retrivio_shell_src="${BASH_SOURCE[0]:-$0}"
_retrivio_repo_dir="$(cd "$(dirname "$_retrivio_shell_src")/.." >/dev/null 2>&1 && pwd)"

# If this shell previously loaded legacy integration, drop it now so
# the Rust-emitted wrapper takes over deterministically.
if [[ -n "${BASH_VERSION:-}" ]]; then
  for __fn in $(compgen -A function); do
    case "$__fn" in
      _retrivio_*|retrivio|retrivio_*|s|sd|sf)
        unset -f "$__fn" >/dev/null 2>&1 || true
        ;;
    esac
  done
  unset __fn
fi

__retrivio_bin=""
if [[ -x "$_retrivio_repo_dir/retrivio" ]]; then
  __retrivio_bin="$_retrivio_repo_dir/retrivio"
elif command -v retrivio >/dev/null 2>&1; then
  __retrivio_bin="$(command -v retrivio)"
fi

if [[ -z "$__retrivio_bin" ]]; then
  echo "retrivio-shell: retrivio binary not found (expected at $_retrivio_repo_dir/retrivio or in PATH)" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${ZSH_VERSION:-}" ]]; then
  eval "$($__retrivio_bin init zsh)"
else
  eval "$($__retrivio_bin init bash)"
fi

unset _retrivio_shell_src _retrivio_repo_dir __retrivio_bin
