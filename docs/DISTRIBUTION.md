# Retrivio Distribution

This document covers end-user installation paths and maintainer release flow.

## End-User Install Options

### 1. One-line installer (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/stouffer-labs/retrivio/main/scripts/install.sh | bash
```

Default install location: `~/.local/bin/retrivio`

Optional environment variables:
- `RETRIVIO_INSTALL_DIR` to override install path
- `RETRIVIO_USE_SUDO=1` to allow privileged install when needed
- `RETRIVIO_GITHUB_OWNER` / `RETRIVIO_GITHUB_REPO` for custom forks

### 2. Homebrew tap

```bash
brew tap --custom-remote stouffer-labs/retrivio https://github.com/stouffer-labs/retrivio
brew install stouffer-labs/retrivio/retrivio
```

The tap formula builds from source (`main`) using Rust. It is a valid install path, but the GitHub Releases installer is the primary end-user path.

### 3. Manual download

From Releases, download the archive for your platform and extract `retrivio`:
- `retrivio-<version>-darwin-arm64.tar.gz`
- `retrivio-<version>-darwin-x86_64.tar.gz`
- `retrivio-<version>-linux-x86_64.tar.gz`

Verify with `SHA256SUMS.txt`.

## Platform Support

- macOS: `arm64`, `x86_64`
- Linux: `x86_64`

Notes:
- `retrivio ui` opens via `open` (macOS) or `xdg-open` (Linux)
- `retrivio watch` uses `fswatch` when present, otherwise polling fallback
- if your configured embedding backend is Ollama, initial indexing requires a running Ollama daemon and the configured embedding model to be available locally

## Maintainer Release Flow

1. Tag a release:

```bash
git tag v0.1.2
git push origin v0.1.2
```

2. GitHub Actions workflow `.github/workflows/release.yml` builds platform binaries and publishes:
- `retrivio-<version>-<os>-<arch>.tar.gz`
- `SHA256SUMS.txt`

3. End users can then install with:

```bash
curl -fsSL https://raw.githubusercontent.com/stouffer-labs/retrivio/main/scripts/install.sh | bash
```

If a user runs `retrivio add <path>` before Ollama is ready, the tracked root is still added. Retrivio skips the initial index and instructs the user to run `retrivio setup` or `retrivio index` after starting Ollama or changing backends.
