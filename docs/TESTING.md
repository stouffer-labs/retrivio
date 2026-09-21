# Testing Retrivio

This is the test plan for Retrivio and for the coding-agent harnesses it plugs into (shell/CLI, Claude Code, Codex). It is layered so that the cheap layers run on every change and the expensive ones before a build touches a live index or ships. Every check is a command with an expected piece of evidence; a PASS or FAIL is only ever backed by captured output. Run the layers the same way every time, record the run, and update this file when behaviour changes (last section).

| Layer | What | Needs | Time |
|---|---|---|---|
| L0 | unit tests, formatting, lints, public-tree check | Rust toolchain | 1 to 5 min |
| L1 | offline indexer smoke test on a synthetic corpus (`scripts/smoke.sh`) | a built binary, python3 optional | under 1 min |
| L2 | live-store checks on the developer machine (`scripts/harness-check.sh`) | installed binary, `~/.retrivio` index, embedding credentials for the semantic path | 1 min |
| L3 | end-to-end through the real `claude` and `codex` CLIs (`scripts/harness-check.sh`) | both CLIs installed, hooks and MCP registered | 5 to 15 min |
| L4 | background watcher through launchd | macOS, `service install` | 30 min wall, mostly waiting |
| scorecard | retrieval quality regression against the last baseline (private) | private eval folder, live index | 5 to 10 min |

## When to run which layer

| Event | Layers |
|---|---|
| every commit | L0 |
| every pull request | L0 + L1 (CI runs L0; run L1 locally, it needs no credentials) |
| before installing a build on the live store (`~/.local/bin/retrivio`) | L0, L1, L2, L3 |
| before a version tag | every layer plus the scorecard regression |
| after any change to `recall`, `hook install`, the MCP server, the skill, or the hook/MCP config files | L3 (and L2 first, it is the fast half of the same script) |
| after a change to the indexer, the watcher, prune or LanceDB handling | L1, then L4 |

## Recording a run

Every L2/L3 run writes its own report; keep it. Convention: `tmp/harness-runs/<date>-<version>/report.md` (the `tmp/` folder is gitignored and never public), created by `scripts/harness-check.sh --out tmp/harness-runs/<date>-<version>`. Put the L0 and L1 console output, the smoke logs (`tmp/smoke/logs/`) and the L4 observations next to it when you ran them. A run is "clean" when no check is FAIL; WARN and SKIP are allowed but must be explained in the record (for example "watcher deliberately uninstalled on this machine"). Before a tag, run `scripts/harness-check.sh --strict`, which counts every WARN and SKIP as a FAIL, so a lexical fallback, a stopped watcher or a missing CLI cannot hide in a release run. Reports contain absolute paths, lead excerpts and raw CLI output from the machine: keep them under `tmp/`, never publish them.

## L0: unit tests, formatting, lints, public tree

```bash
cargo test -p retrivio                 # expected: "test result: ok. N passed; 0 failed"
cargo fmt --check                      # expected: no output, exit 0
cargo clippy -p retrivio               # reported, not gating: note the warning count in the run record
scripts/check-public-tree.sh           # expected: "check-public-tree: ok (N tracked files)"
```

`check-public-tree.sh` fails when a tracked file is outside the allowlist; when you add a public file, extend the `allowed` regex in the script in the same commit. CI (`.github/workflows/ci-rust.yml`) runs the same four steps.

If a test fails: fix the code or the test in the same change, never mark it ignored to get green. If `check-public-tree.sh` fails with `UNKNOWN`, either the file does not belong in the public repo (move it to the private parent folder) or the allowlist needs the new path.

## L1: offline smoke test (`scripts/smoke.sh`)

```bash
cargo build --release -p retrivio
RETRIVIO_BIN=target/release/retrivio scripts/smoke.sh tmp/smoke     # default work dir: tmp/smoke; never /tmp
```

The script needs no network and no credentials. It generates a synthetic corpus (a markdown note, a Python file, a YAML file, a nested directory, a root-level file, and a tiny `.docx` and `.pptx` built with python3's `zipfile` when python3 exists, otherwise SKIP), writes a config with `embed_backend = "hash"` (offline feature-hashing vectors, deterministic, not semantic) and an isolated `--data-dir`, then runs the indexer through its life cycle and judges each step on the counters `retrivio index` prints:

| Step | Assertion (from the counters) |
|---|---|
| `index-first` | projects found and files selected match the corpus; `chunks embedded` equals `chunks indexed`; reused 0; deleted 0; `documents extracted: 2 (failed: 0)` when the Office files exist |
| `index-steady` | second run: projects found still equals the corpus (a discovery regression that finds nothing must not pass as "steady"), every project skipped, `files selected: 0`, `chunks embedded: 0` |
| `search-python`, `search-yaml` | before any edit: a Python identifier finds the `.py` file, a YAML value finds the `.yaml` file (proves both were ingested) |
| `refresh-reuses-vectors` | forced `refresh` re-reads every file but embeds 0 and reuses every stored vector |
| `index-edit-one-file`, `search-edited-text` | after editing one file in a three-file project: only that project is indexed, `files selected: 3 (unchanged 2, rechunked 1)`, `chunks embedded: 1`; a term that exists only in the new text finds the file |
| `index-delete-one-file`, `search-deleted-absent` | after deleting one file: `deleted: 1`, `stale chunks pruned: 1 (from 1 files)`; a search for its content no longer returns it |
| `search-*` | `search --view files --json` returns the expected file for a markdown term, a root-level file, a nested file, a docx paragraph and a pptx speaker note; the payload has `query/results/timing_ms/view` and each result has `path, score, content_date, age_days, freshness_tier, date_source, project_path` |
| `recall-dry-run` | `recall --format text --query ...` against the smoke store prints a `<retrivio_leads>` block whose first lead is the expected note and writes a log line |
| `prune-dry-run`, `prune` | after deleting another file: the dry run reports `chunks would prune: 1 (from 1 files)` and writes nothing; the real run prunes 1 and prints the `lancedb compacted:` line |
| `index-after-prune`, `index-final-steady` | the next run may re-collect the pruned project once via the manifest fast path but reads, embeds and deletes nothing (prune does not advance the change-gate signature today; a build that skips it at once also passes); the run after that skips every project, with the project count intact |

Each step prints `PASS`/`FAIL`/`SKIP` with the evidence line; raw command output is in `<work dir>/logs/<step>.log`; exit status 1 on any FAIL. When a step fails, read its log first: the counters and the per-project lines (`[i/n] index <project> files=... embedded=... reused=... deleted=...`) usually say what happened. A FAIL here is a real indexer regression unless the corpus generator itself failed (`corpus-documents` SKIP with a note). The counter assertions are exact on purpose: the corpus is fixed and every file is one chunk, so a different number means the accounting changed. When chunking or selection changes by design, update the expected numbers here and in the script in the same commit. The `.docx`/`.pptx` fixtures are the minimal archives the extractor's own unit fixtures use, not full Office packages; a stricter extractor would need richer fixtures.

## L2: live store on the developer machine

```bash
scripts/harness-check.sh --skip-claude --skip-codex --out tmp/harness-runs/<date>-<version>
```

Uses `~/.local/bin/retrivio` (override with `--bin` or `RETRIVIO_BIN`) and its live index, read-only: recall dry runs only touch the per-session memory under `~/.retrivio/recall/` (a dedicated session id, reset before and after) and append to `~/.retrivio/recall.log`.

| Check | Command | Expected evidence |
|---|---|---|
| `l2-version` | `retrivio version` | `retrivio <semver>` |
| `l2-doctor` | `retrivio doctor` | `database ready: yes` and `embedding migration: ready` |
| `l2-mcp-doctor` | `retrivio mcp doctor` | `status: ready`; `Claude Code registered`; `Codex registered` (each only when that CLI is not skipped) |
| `l2-mcp-registration-claude/-codex` | read `~/.claude.json` `mcpServers.retrivio.command` and `~/.codex/config.toml` `[mcp_servers.retrivio] command` | both equal the binary under test. L3 always exercises the binary the hooks and MCP registrations point at, so when you pass `--bin <candidate>` these checks (and `l2-hook-status-*`, whose "matches current binary" compares against `--bin`) fail until the candidate is installed; that is the signal to install first, then run L3 |
| `l2-skill-installed-claude/-codex` | read `~/.claude/skills/retrivio-recall/SKILL.md` and `~/.agents/skills/retrivio-recall/SKILL.md` | file present, frontmatter `name: retrivio-recall`, the sentence the L3 skill check quotes is in it; WARN `l2-skill-same-content` when the two copies differ |
| `l2-hook-status-claude` | `retrivio hook status` | Claude Code section: `installed: yes (UserPromptSubmit + SessionStart)` and `bin: ... matches current binary` |
| `l2-hook-status-codex` | same | Codex section: same two lines plus `trust: userPromptSubmit trusted` |
| `l2-service-status` | `retrivio service status` | `loaded: yes` is PASS; `loaded: no` is WARN (FAIL with `--require-watcher`) |
| `l2-recall-content` | `retrivio recall --format text --query "<content prompt>" --session <id> --cwd <dir> --verbose` | a `<retrivio_leads>` block and a log line `... semantic|lexical <ms>ms cand=N leads>=1` |
| `l2-recall-semantic` | from the same log line | `semantic` is PASS; `lexical` is WARN: the embedding backend was unavailable or the 10 minute breaker is open (files `embed-breaker`/`embed-slow` under `~/.retrivio/recall/`) |
| `l2-recall-block-wellformed` | from the same block | header line present; 1 to `recall_max_leads` numbered items; no raw `<` or `>` inside the block (excerpts use ‹ ›) |
| `l2-recall-nr` | same with `--query "nr: <content prompt>"` | empty stdout, log `skipped:nr-prefix` |
| `l2-recall-ack` | same with `--query "continue"` | empty stdout, log `skipped:ack` |
| `l2-recall-latency` | the cold content run and three warm repeats (session reset before each) | every sample under 4000 ms wall clock (the hook's hard deadline is 4 s; the hook timeout in the CLIs is 5 s); the evidence lists all four so a load spike is visible |
| `l2-search-json` | `retrivio search --view files --limit 3 --json "<content prompt>"` | at least one result; payload keys and per-result freshness fields as in L1 |

The content prompt defaults to a question about Retrivio's own recall hook, which yields leads on any machine whose index covers the Retrivio repo. On another machine pass `--content-prompt "<question about something the index covers>"` (or `HARNESS_CONTENT_PROMPT`); keep it free of names you would not put in a public report.

### Scorecard regression (private)

The retrieval-quality regression lives in the private parent folder, not in this repository: a 36-entry scorecard (29 positives, 7 negatives) mined from real sessions, a stdlib runner `tmp/eval/run_eval.py`, and the frozen baseline `tmp/eval/baseline-0.1.7.jsonl`. It runs `search --view files --limit 10 --json` and a `recall` dry run per entry against the live index and adds only query-embedding cache rows to the database.

```bash
# from the private parent folder
scripts/regression.sh --binary <path to the candidate binary> --tag <new tag>          # runs the 36 queries (minutes)
# writes tmp/eval/baseline-<tag>.jsonl, baseline-<tag>-summary.md and compare-<tag>-vs-<base>.md
scripts/regression.sh --compare-only --tag <tag> --base 0.1.7                            # re-compare without re-running
```

The compare file lists hit@1/3/10, MRR, recall lead-hit, negatives with leads, median and p90 latency, and every entry whose search rank, recall hit or lead count changed. Gate (exit 1 unless `--no-gate`): **hit@3 must not drop and negatives-with-leads must not rise** against the base. Scores drift by about 0.0001 between runs because recency decay is computed from wall-clock age, so compare ranks and hits, not raw scores. A candidate binary is evaluated against the live index without installing it: pass its path with `--binary`.

## L3: end-to-end through Claude Code and Codex

```bash
scripts/harness-check.sh --out tmp/harness-runs/<date>-<version>          # L2 + L3, both CLIs
scripts/harness-check.sh --skip-codex ...                                  # one CLI only
scripts/harness-check.sh --timeout 600 ...                                 # slow machine or slow remote MCP servers
```

Preconditions: `retrivio hook install` done for both CLIs (Codex hooks trusted: `retrivio hook status` shows `trust: userPromptSubmit trusted`), `retrivio mcp register` done, the `retrivio-recall` skill present at `~/.claude/skills/retrivio-recall/SKILL.md` and `~/.agents/skills/retrivio-recall/SKILL.md`, `RETRIVIO_HOOK` not set to `0`, no `.retrivio/hook-off` between the run directory and `$HOME` (the script warns). A CLI that is not on `PATH` makes its checks SKIP, not FAIL.

Each check is one non-interactive CLI call from a neutral working directory (`<out>/cwd`) with stdin from `/dev/null` (Codex hangs otherwise) and a per-call timeout (`--timeout`, default 300 s). Claude Code calls pass `--session-id <uuid> --no-session-persistence`; Codex calls use `codex exec --skip-git-repo-check --ephemeral -C <cwd> -o <last-message file>` and the session id printed on stderr. The recall log tags every run with the first 8 hex digits of `sha1(session_id)`, so each check reads its own line from `~/.retrivio/recall.log` and is not confused by other sessions running on the machine.

| Check (per CLI) | Prompt | Expected evidence |
|---|---|---|
| `hook-content` | "Reply only with any `<retrivio_leads>` block you received in this conversation, reproduced verbatim, else reply exactly NONE. Do not use any tools. `<content prompt>`" | the reply contains the block; the session's log line reads `semantic` or `lexical` with `leads>=1`; Codex stderr also shows `hook: UserPromptSubmit Completed` |
| `block-wellformed` | (from the reply above) | 1 to `recall_max_leads` numbered items, header present, no raw `<` or `>` inside |
| `leads-match-dryrun` | (from the reply above) | the lead paths in the model's block equal the lead paths of a dry run of the exact same prompt (instruction prefix included) from the same cwd (`raw/l2-recall-hook-prompt.out`); WARN when they differ (ranking drift between the two runs, or a block the model composed itself: compare `raw/l2-recall-content.out` with the reply) |
| `hook-nr` | `nr: ` + the same prompt | no block in the reply (the model says NONE); log line `skipped:nr-prefix` |
| `hook-ack` | `continue` | no block; log line `skipped:ack` (the hook still runs, recall exits silently; Codex still prints `hook: UserPromptSubmit Completed`). Judged on the log line and block absence; a non-zero exit of the Claude CLI itself is recorded separately as WARN `hook-ack-cli` |
| `mcp-search-files` | "nr: Call the Retrivio MCP tool search_files with query ... and limit 3. Then reply with only the absolute path of the first result ... else MCP-UNAVAILABLE." (Claude: `--allowedTools mcp__retrivio__search_files`) | the reply's first line is an absolute path to a regular file that is also in the top 3 of `retrivio search --view files --limit 3 --json` for the same query (the `l2-search-json` run), so a guessed or unrelated path fails; Codex stderr must show `mcp: retrivio/search_files (completed)` |
| `skill` | "nr: Load the retrivio-recall skill and quote, verbatim and on a single line, the sentence ... which begins with the words \"Leads never change\" ... else SKILL-UNAVAILABLE." | the quoted text is found verbatim in the installed SKILL.md (`l2-skill-installed-*` already checked the file). Residual gap: a model could complete the sentence from memory; the check proves the file is reachable and intact, not that the CLI's skill mechanism (rather than a plain file read) delivered it |

The `nr:` prefix on the MCP and skill prompts keeps recall out of those checks so they measure one thing each. Measured on 2026-09-21 with Claude Code 2.1.278 and Codex 0.154.0 on Amazon Bedrock: Claude calls took 7 to 21 s each, Codex calls 14 to 26 s; the full L2 + L3 run took 4 minutes. The `continue` call to Claude passes `--tools ""` so the model cannot wander into tool use on a bare acknowledgement; the hook-content and `nr:` prompts say "Do not use any tools" instead, because they must stay content prompts. Codex startup is slower when an unrelated remote MCP server in the user's config is unreachable; raise `--timeout` rather than skipping.

When a check fails:

- `hook-content` with no log line for the session: the hook did not run. Check `retrivio hook status` (installed, binary path matches), for Codex the trust lines (`retrivio hook trust --codex`), `RETRIVIO_HOOK`, and hook-off markers. Compare `~/.claude/settings.json` / `~/.codex/hooks.json` with what `hook install` writes.
- `hook-content` with a log line but no block in the reply: recall ran but produced nothing (`leads=0`, weak matches for this prompt on this index; change `--content-prompt`) or the model did not reproduce it (read `raw/<check>.out`; the log line is the ground truth for injection, the reply for delivery).
- `hook-nr` / `hook-ack` with a `semantic` log line: the skip rules regressed in `recall.rs` (`nr:` prefix, acknowledgement list). Confirm with the L2 dry runs, which bypass the CLI.
- `mcp-search-files`: `retrivio mcp doctor` must say registered for that CLI and `l2-mcp-registration-*` must point at the binary you mean to test; for Claude the tool needs `--allowedTools mcp__retrivio__search_files` in print mode; a path that exists but is not in the search top 3 means the tool ignored the query or the model answered from elsewhere (read the raw reply and stderr); run `retrivio mcp serve` by hand and send an `initialize` frame if the server itself is suspect.
- `skill`: the skill folder must exist in both locations (same inode or a symlink); Codex shortens skill descriptions when many are installed, which is a warning, not a failure.
- `block-wellformed`: a raw `<` or `>` inside the block means sanitization regressed (`recall.rs` replaces them with ‹ ›); more items than `recall_max_leads` means the cap regressed.
- Timeouts (`rc: 124` in `raw/<check>.meta`): rerun with a larger `--timeout`; if only Codex times out, look at its stderr for a remote MCP server failing to start.

Raw output of every check (`stdout`, `stderr`, Codex last message, return code, wall time) is under `<out>/raw/`, and `<out>/report.md` embeds the first 60 lines of each.

## L4: background watcher

Run this on a machine where the watcher may run against the live index (it indexes; it is not read-only), after L1 to L3 are clean.

```bash
retrivio service install                 # launchd agent com.stouffer-labs.retrivio.watch; watch --quiet --interval 300
retrivio service status                  # expected: loaded: yes, a pid, fswatch resolved on the plist PATH
lance_before=$(ls ~/.retrivio/lance/*.lance/_versions 2>/dev/null | wc -l)
echo "watcher check $(date)" >> <a file in a small tracked project>   # one file, one project
sleep 30; tail -5 ~/.retrivio/watch.log
```

Expected within one interval: a `watch: changes in <that project>` line naming only that project, then one tick `event updated=1 removed=0 ... files=<n>/<n-1>/1 ... chunks_embedded=<small> chunks_reused=0 chunks_deleted=0`, where the embedded count is the edited file's chunk count and no other project appears. Over the next 30 minutes (six sweeps at `--interval 300`) the sweep ticks must show `updated=0` for an idle tree, and the LanceDB version count (`ls ~/.retrivio/lance/*.lance/_versions | wc -l`) must stay bounded: it may grow by a few per write and must drop back after a `sweep lancedb compacted: versions <before> -> <after>` line once it passes `lance_compact_versions` (default 200). Record the before/after version counts and the tick lines. Finish with `retrivio service uninstall` if the machine must stay quiet, otherwise leave it running and note that in the record.

Failure signs: a tick listing several projects for one touched file (event filtering regressed), `updated>0` on idle sweeps (change gate regressed), a version count that only grows (compaction regressed), or `watch: index busy` repeating (another writer holds `index.lock`; stop it before judging the watcher).

## Known gaps (not covered by any check yet)

- The `SessionStart` reset hook (matcher `compact|clear`) is only checked as installed (`hook status`); no check triggers `/compact` or `/clear` in a real session and verifies the session memory file is removed. Manual: note the `~/.retrivio/recall/<sha1(session)>.json` file of a live session, run `/clear`, confirm it is gone.
- Hook failure modes are not exercised end to end: a recall crash, a run over the 4 s deadline, an open embedding breaker, `RETRIVIO_HOOK=0`, `.retrivio/hook-off`, a zero-lead prompt. They are documented as diagnoses; adding them as checks means changing the user's environment during the run.
- L4 is a manual procedure; a watcher regression is only caught when someone runs it.
- The scorecard regression compares two runs against the live index, which changes between runs; the fingerprint lines in the two summaries (config sha256, database size and mtime) must be compared by the reader, and the gate only covers hit@3 and negatives (the rest is reported as warnings).

## Updating this plan

- Behaviour changed on purpose (new counter, new skip rule, new hook file, new CLI flag): change the assertion in the script and the row in this file in the same commit, and say so in the commit message.
- A check flakes: do not loosen it first. Find the uncontrolled variable (another session writing the recall log, an open embedding breaker, a remote MCP server slowing Codex startup) and make the check control for it, the way the session-hash attribution does.
- New surface (a new CLI, a new MCP tool the agents rely on, a new opt-out): add a check to `scripts/harness-check.sh` with the prompt, the evidence and the failure notes here, and add the fixture or step to `scripts/smoke.sh` when it is indexer-side.
- New machine: run L2 first; every SKIP or WARN it prints is a precondition for L3.
- Keep prompts generic. Reports under `tmp/harness-runs/` are private, but this file and the scripts are public: no personal paths, customer names or private queries in them.
