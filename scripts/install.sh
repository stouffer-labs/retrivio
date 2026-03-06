#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install retrivio from GitHub Releases.

Usage:
  scripts/install.sh [version]

Examples:
  scripts/install.sh            # installs latest release
  scripts/install.sh v0.1.2     # installs a specific tag
  scripts/install.sh 0.1.2      # same as above

Environment variables:
  RETRIVIO_GITHUB_OWNER   GitHub owner (default: stouffer-labs)
  RETRIVIO_GITHUB_REPO    Repository name (default: retrivio)
  RETRIVIO_INSTALL_DIR    Install destination (default: ~/.local/bin)
  RETRIVIO_USE_SUDO       Set to 1 to use sudo if install dir is not writable
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
}

normalize_os() {
  local raw
  raw="$(uname -s)"
  case "$raw" in
    Darwin) echo "darwin" ;;
    Linux) echo "linux" ;;
    *)
      echo "error: unsupported OS: $raw" >&2
      exit 1
      ;;
  esac
}

normalize_arch() {
  local raw
  raw="$(uname -m)"
  case "$raw" in
    x86_64|amd64) echo "x86_64" ;;
    arm64|aarch64) echo "arm64" ;;
    *)
      echo "error: unsupported architecture: $raw" >&2
      exit 1
      ;;
  esac
}

latest_tag() {
  local owner="$1"
  local repo="$2"
  local api_url="https://api.github.com/repos/${owner}/${repo}/releases/latest"
  local body
  body="$(curl -fsSL "$api_url")"
  printf '%s\n' "$body" | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1
}

verify_checksum() {
  local archive_path="$1"
  local checksums_path="$2"
  local archive_name
  archive_name="$(basename "$archive_path")"
  local expected_line
  expected_line="$(grep -E "[[:space:]]${archive_name}\$" "$checksums_path" || true)"
  if [[ -z "$expected_line" ]]; then
    echo "error: checksum entry not found for ${archive_name}" >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$archive_path")" && grep -E "[[:space:]]${archive_name}\$" "$checksums_path" | sha256sum -c -)
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    local expected actual
    expected="$(printf '%s\n' "$expected_line" | awk '{print $1}')"
    actual="$(shasum -a 256 "$archive_path" | awk '{print $1}')"
    if [[ "$expected" != "$actual" ]]; then
      echo "error: checksum mismatch for ${archive_name}" >&2
      exit 1
    fi
    return
  fi
  echo "error: no SHA-256 utility found (need sha256sum or shasum)" >&2
  exit 1
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  need_cmd curl
  need_cmd tar
  need_cmd uname

  local owner repo install_dir requested_version tag version os arch asset_name
  owner="${RETRIVIO_GITHUB_OWNER:-stouffer-labs}"
  repo="${RETRIVIO_GITHUB_REPO:-retrivio}"
  install_dir="${RETRIVIO_INSTALL_DIR:-$HOME/.local/bin}"
  requested_version="${1:-latest}"

  if [[ "$requested_version" == "latest" ]]; then
    tag="$(latest_tag "$owner" "$repo")"
    if [[ -z "$tag" ]]; then
      echo "error: unable to determine latest release tag for ${owner}/${repo}" >&2
      exit 1
    fi
  else
    if [[ "$requested_version" == v* ]]; then
      tag="$requested_version"
    else
      tag="v${requested_version}"
    fi
  fi
  version="${tag#v}"

  os="$(normalize_os)"
  arch="$(normalize_arch)"
  case "${os}-${arch}" in
    darwin-x86_64|darwin-arm64|linux-x86_64)
      ;;
    *)
      echo "error: no prebuilt release for ${os}-${arch}" >&2
      echo "hint: build from source with: cargo build --release -p retrivio" >&2
      exit 1
      ;;
  esac

  asset_name="retrivio-${version}-${os}-${arch}.tar.gz"
  local base_url archive_url checksums_url tmp_dir archive_path checksums_path
  base_url="https://github.com/${owner}/${repo}/releases/download/${tag}"
  archive_url="${base_url}/${asset_name}"
  checksums_url="${base_url}/SHA256SUMS.txt"

  tmp_dir="$(mktemp -d -t retrivio-install.XXXXXX)"
  trap "rm -rf -- '${tmp_dir}'" EXIT
  archive_path="${tmp_dir}/${asset_name}"
  checksums_path="${tmp_dir}/SHA256SUMS.txt"

  echo "retrivio installer"
  echo "repo: ${owner}/${repo}"
  echo "version: ${tag}"
  echo "target: ${os}-${arch}"
  echo "download: ${archive_url}"

  curl -fsSL "$archive_url" -o "$archive_path"
  curl -fsSL "$checksums_url" -o "$checksums_path"
  verify_checksum "$archive_path" "$checksums_path"

  tar -xzf "$archive_path" -C "$tmp_dir"
  local extracted_bin
  extracted_bin="$(find "$tmp_dir" -type f -name retrivio | head -n1)"
  if [[ -z "$extracted_bin" ]]; then
    echo "error: retrivio binary not found in archive" >&2
    exit 1
  fi

  if ! mkdir -p "$install_dir" 2>/dev/null; then
    if [[ "${RETRIVIO_USE_SUDO:-0}" == "1" ]]; then
      sudo mkdir -p "$install_dir"
    else
      echo "error: cannot create install dir: $install_dir" >&2
      echo "hint: set RETRIVIO_INSTALL_DIR to a writable path, or set RETRIVIO_USE_SUDO=1" >&2
      exit 1
    fi
  fi

  if ! install -m 0755 "$extracted_bin" "${install_dir}/retrivio" 2>/dev/null; then
    if [[ "${RETRIVIO_USE_SUDO:-0}" == "1" ]]; then
      sudo install -m 0755 "$extracted_bin" "${install_dir}/retrivio"
    else
      echo "error: cannot write ${install_dir}/retrivio" >&2
      echo "hint: rerun with RETRIVIO_USE_SUDO=1 or choose another RETRIVIO_INSTALL_DIR" >&2
      exit 1
    fi
  fi

  echo "installed: ${install_dir}/retrivio"
  if [[ ":$PATH:" != *":${install_dir}:"* ]]; then
    echo "hint: add ${install_dir} to PATH"
  fi
  "${install_dir}/retrivio" --help >/dev/null 2>&1 && echo "health: retrivio executable OK"
}

main "$@"
