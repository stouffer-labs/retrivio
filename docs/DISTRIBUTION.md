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

This project does not use a local `.git` repository. Source is published via GitHub Contents API.

### 1. Bump version

Edit `crates/retrivio/Cargo.toml` and update the `version` field. Also update version examples in `scripts/install.sh`.

### 2. Build locally

```bash
cargo build --release -p retrivio
./target/release/retrivio --version
```

### 3. Publish source to GitHub

```bash
scripts/publish-gh-api.sh
```

This syncs the allowlisted files to `stouffer-labs/Retrivio` on GitHub. Each file is a separate commit. The `[skip ci]` marker is appended by default to avoid triggering CI on every commit. **Do not use `--no-skip-ci`** — it causes 50+ CI runs.

### 4. Create a release tag

```bash
SHA=$(gh api repos/stouffer-labs/Retrivio/git/ref/heads/main --jq '.object.sha')
gh api repos/stouffer-labs/Retrivio/git/refs \
  --method POST -f ref="refs/tags/v0.1.4" -f sha="$SHA"
```

### 5. Release builds automatically

GitHub Actions workflow `.github/workflows/release.yml` triggers on `v*` tags and builds:
- `retrivio-<version>-darwin-arm64.tar.gz`
- `retrivio-<version>-darwin-x86_64.tar.gz`
- `retrivio-<version>-linux-x86_64.tar.gz`
- `SHA256SUMS.txt`

### 6. End users install with

```bash
curl -fsSL https://raw.githubusercontent.com/stouffer-labs/retrivio/main/scripts/install.sh | bash
```

If a user runs `retrivio add <path>` before Ollama is ready, the tracked root is still added. Retrivio skips the initial index and instructs the user to run `retrivio setup` or `retrivio index` after starting Ollama or changing backends.
