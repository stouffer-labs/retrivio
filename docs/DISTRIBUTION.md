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

The repository is a normal git repository. Work happens on a branch, lands on `main` through a pull request, and a version tag triggers the release build.

### 1. Branch and pull request

```bash
git checkout -b <topic>
# ... commit ...
scripts/check-public-tree.sh     # fails if any tracked file is outside the public set
git push -u origin <topic>
gh pr create --fill
```

CI (`.github/workflows/ci-rust.yml`) runs the public-tree check, `cargo check`, `cargo test`, `cargo clippy` and `cargo fmt --check` on every pull request and on every push to `main`. Merge with a squash so `main` carries one commit per change.

### 2. Bump version

Edit `crates/retrivio/Cargo.toml` and update the `version` field. Also update the version examples in `scripts/install.sh`. `cargo build` refreshes `Cargo.lock`.

### 3. Build and test locally

```bash
cargo build --release -p retrivio
cargo test -p retrivio
./target/release/retrivio --version
```

### 4. Tag the release

After the version bump has merged to `main`:

```bash
git checkout main && git pull
git tag v0.2.0
git push origin v0.2.0
```

### 5. Release builds automatically

GitHub Actions workflow `.github/workflows/release.yml` triggers on `v*` tags and builds:
- `retrivio-<version>-darwin-arm64.tar.gz`
- `retrivio-<version>-darwin-x86_64.tar.gz`
- `retrivio-<version>-linux-x86_64.tar.gz`
- `SHA256SUMS.txt`

If the tag push did not start the workflow, dispatch it by hand: `gh workflow run release.yml -f tag=v0.2.0`.

### 6. End users install with

```bash
curl -fsSL https://raw.githubusercontent.com/stouffer-labs/retrivio/main/scripts/install.sh | bash
```

If a user runs `retrivio add <path>` before Ollama is ready, the tracked root is still added. Retrivio skips the initial index and instructs the user to run `retrivio setup` or `retrivio index` after starting Ollama or changing backends.
