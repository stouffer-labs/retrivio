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

## Code signing on macOS (local builds)

macOS keys a folder-access grant (the "retrivio would like to access your Documents folder" prompt; TCC service `kTCCServiceSystemPolicyDocumentsFolder`) to the requesting program's code signature. An unsigned binary has a different code hash after every build, so each rebuild is a new program: the launchd watcher is prompted again and, having no window to answer in, stays blocked until the prompt is clicked. Signing local builds with a stable identity keeps the identity across rebuilds; a self-signed certificate is enough because the grant needs a stable identity, not one Apple trusts.

Create the certificate once (Keychain Access):

1. Keychain Access > Keychain Access menu > Certificate Assistant > Create a Certificate…
2. Name `Retrivio Dev` (any name; it becomes the identity), Identity Type `Self Signed Root`, Certificate Type `Code Signing`. Create.
3. `security find-identity -v -p codesigning` lists it.

Then, after every build, with the watcher stopped. The script refuses to sign a binary that is running (`lsof`/`pgrep` find it): launchd would keep the old process, and re-signing the file under it changes nothing for TCC. Stop the watcher, build, sign, start it again on the signed build:

```bash
export RETRIVIO_CODESIGN_IDENTITY="Retrivio Dev"   # put it in your shell rc
retrivio service uninstall                         # stop the watcher (a no-op when it is not installed)
cargo build --release -p retrivio
scripts/sign-macos.sh                              # default target: target/release/retrivio
retrivio service install                           # launchd starts the signed build
```

The script runs `codesign --force --sign "$RETRIVIO_CODESIGN_IDENTITY" --identifier com.stouffer-labs.retrivio --timestamp=none <binary>`, verifies the result (`codesign --verify --strict`, `codesign -dv`) and prints the designated requirement (`codesign -d -r-`), which for a certificate that is not Apple's reads `identifier "com.stouffer-labs.retrivio" and certificate leaf = H"<sha1 of your certificate>"`. That requirement is what identifies the program to macOS from one build to the next; both of its parts are fixed by the flags (the identifier is given explicitly, the certificate is yours), so the folder-access grant is expected to persist across rebuilds. That is the expectation from the flags, not something demonstrated here yet: confirm it once. Grant access on the first signed run, then rebuild, sign and reinstall twice more; each time compare the designated requirement line with the previous one (it must be identical), check that `~/.retrivio/watch.log` gets its bootstrap tick, and check that `log show --last 30m --predicate 'subsystem == "com.apple.TCC"' | grep -i retrivio` shows no new prompt. `--timestamp=none` stays on: a secure timestamp is for distributed signatures whose certificate may expire, and a local build signed with a self-signed identity gains nothing from the round trip to Apple's timestamp server.

Without `RETRIVIO_CODESIGN_IDENTITY` the script signs ad hoc (`--sign -`): the binary carries a signature, but its designated requirement is the code hash of that build, so macOS still asks again after every rebuild. The release workflow does not sign the published binaries; a downloaded release only changes when you upgrade, so it prompts once per upgrade.

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

Then run the layered plan in [`docs/TESTING.md`](TESTING.md): L0 to L3 (unit tests, offline smoke test, live-store checks, Claude Code and Codex end-to-end checks) before installing a build on your live store, and every layer (L0 to L4 plus the scorecard regression) before a tag.

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
