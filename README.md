# retrivio

`retrivio` is a local-first semantic code intelligence and navigation index for project workspaces. It combines AST-aware chunking, multi-stage retrieval, and graph-based context expansion to power fast project search and AI-ready context packaging.

Supports:
- semantic + lexical + frecency + graph ranking with query-adaptive weights
- AST-aware code chunking via tree-sitter (9 languages)
- cross-encoder re-ranking and HyDE for high-precision retrieval
- interactive terminal picker for fast selection
- local HTTP API + MCP server for AI agent integrations

## Quick Start

```bash
cd /path/to/repo

# Initialize Retrivio (embedded LanceDB + SQLite under ~/.retrivio by default)
retrivio install

# Ensure Ollama is up and embedding model is local
brew services start ollama   # or run `ollama serve` in another terminal
ollama pull qwen3-embedding

# Initialize Retrivio config for Ollama
retrivio init --embed-backend ollama

# Track and index your workspace roots
retrivio add ~/c-projects
retrivio add ~/estouff --exclude Downloads --exclude Library --exclude .docker
retrivio index

# Search (non-interactive)
retrivio search storage replication

# Interactive picker (prints a selected path)
retrivio pick --query "storage replication"

# To cd (a subprocess cannot change the parent shell's cwd)
cd "$(retrivio pick --query 'storage replication')"
```

Bedrock option (AWS, no local model required):

```bash
# Uses normal AWS CLI auth chain + Retrivio setup wizard (no manual env var required)
retrivio init --embed-backend bedrock --embed-model amazon.titan-embed-text-v2:0

# Guided setup: choose AWS profile (from ~/.aws/config) and optional credential refresh
retrivio setup
# or auth-only re-selection later
retrivio auth select

# Preflight identity + model access checks
retrivio doctor --fix
```

## How It Works

### Storage Architecture

Retrivio uses **SQLite** as the durable source of truth and **embedded LanceDB** as the local vector index.

- **SQLite** stores all project metadata, chunks, embeddings, edges, symbols, imports, and tracked roots. It is portable and requires zero infrastructure.
- **LanceDB** stores ANN vectors under `.retrivio/lance` and runs in-process (no external service).
- **SQLite FTS5** powers lexical retrieval (`project_fts`, `chunk_fts`, `symbol_fts`) plus graph relationship lookups (`project_edges`, `file_dependency_edges`).

This design targets **100-200GB of files across 10,000+ directories** on a single machine:

- **Index-aware adjacency and FTS5** keep graph/path lookups fast without external graph infrastructure
- **HNSW vector indexes** support approximate nearest-neighbor search without loading all vectors into memory

SQLite remains the source of truth so LanceDB can be rebuilt from scratch at any time (`retrivio reembed` rebuilds vectors; `retrivio refresh` ensures state is current).

### Code Intelligence Pipeline

Retrivio uses **tree-sitter** to parse source files into ASTs, extracting semantic structure instead of splitting on fixed character windows.

**AST-Aware Chunking**: Source files in [9 supported languages](#supported-languages) are parsed into semantic chunks -- functions, classes, structs, methods, import blocks, and preambles. Each chunk maps to a meaningful code unit. Large definitions are split at method/inner-function boundaries, then at blank lines. Maximum chunk size is 1500 characters for code files. Unsupported languages fall back to character-window chunking with ~16% overlap.

**Contextual Headers**: Each chunk gets a contextual header prepended *only during embedding generation* (not stored in chunk text). Headers include file path, parent context, and symbol signature -- so embeddings capture where code lives, not just what it says.

**Symbol Extraction**: The AST walker extracts function, class, method, struct, trait, interface, enum, type alias, constant, and module definitions into a dedicated `symbols` table with FTS5 full-text search. Symbols include qualified names, signatures, doc comments, visibility, and parent relationships.

**Import Graph**: Import statements are extracted and resolved to file paths within each project, building a `file_dependency_edges` graph. Resolution handles language-specific conventions (Python dotted paths, JS/TS relative imports with extension probing, Rust `crate::` paths, Go package paths, Java dotted paths, C/C++ includes).

### Retrieval Pipeline

Search uses a multi-stage ranking pipeline with query-adaptive weights:

**1. Query Classification** -- Queries are classified into four types, each with tuned weight profiles:

| Query Type | Triggers | Emphasis |
|---|---|---|
| Symbol | `camelCase`, `snake_case`, identifiers | Lexical + path keywords |
| NaturalLanguage | "how does...", "explain...", question words | Semantic similarity |
| CodePattern | `fn validate`, `class Auth`, language keywords | Balanced semantic + lexical |
| PathQuery | `src/auth`, `*.rs`, file extensions | Path keyword matching |

**2. Multi-Source Retrieval** -- Four sources queried in parallel from LanceDB + SQLite:

- Vector kNN from embedded LanceDB (cosine similarity)
- Chunk FTS5 retrieval from SQLite
- Path keyword matching
- Symbol FTS5 prefix matching

**3. Weighted RRF Fusion** -- Results from all sources fused via Reciprocal Rank Fusion (k=60) with query-type-adaptive weights:

```
score(item) = SUM(weight_i / (60 + rank_i + 1))
```

**4. Graph-Aware Expansion** -- Top chunks seed a 2-hop BFS through project neighbor graph. Decay factor 0.6 per hop. Same-project chunks get high graph weight (0.76-0.88 depending on semantic similarity); cross-project chunks weighted by edge strength (base 0.20, scale 0.70, cap 0.90).

**5. Cross-Encoder Re-Ranking** -- Top 60 candidates scored by an Ollama LLM (default: `qwen3:0.6b`) for pointwise relevance (0-10 scale). Processed in parallel batches of 8. Final score: `0.70 * reranker + 0.30 * original`. Enabled by default; adds ~200-500ms.

**6. HyDE (Hypothetical Document Embedding)** -- For natural language queries, optionally generates a hypothetical code snippet via Ollama, embeds it, and uses it as an additional vector query. HyDE results merge with existing scores (0.3 blend weight). Opt-in via `hyde_enabled = true`; adds ~800ms.

**7. Tiered Search** -- For codebases with 200+ projects, a project-level pre-filter narrows to the top 30 projects before chunk-level search.

### Context Packing

`pack_context` builds AI-ready context packages from search results:

- **Deduplication**: Chunks with >50% Jaccard token overlap are deduplicated
- **Diversity Limits**: Max 3 chunks per file, max 8 chunks per project
- **Budget Enforcement**: Stops adding chunks when character budget is reached
- **Related Expansion**: Each seed chunk pulls up to 3 related chunks via graph lineage
- **Coherence Ordering**: Same-file chunks grouped together, ordered by chunk index (reading order); different files ordered by best score
- **Optional Full Docs**: Can include reconstructed full documents from chunk sequences

### Indexing & Performance

**File-Level Incremental Detection**: Each file tracked by mtime + size + xxhash64 content hash. Fast path compares mtime/size first; content hash handles clock skew and `touch` without modification.

**Parallel Project Indexing**: AST parsing and corpus collection run on 4 threads (`std::thread::scope`). Embedding and storage run sequentially (require DB connection). Chunk embeddings batched in groups of 24 with 3-retry exponential backoff.

**Inline LanceDB Vector Writes**: During indexing, chunk vectors are persisted to SQLite and upserted into embedded LanceDB. `retrivio reembed` can fully rebuild LanceDB from SQLite vectors when model settings change.

**Query Embedding Cache**: In-memory LRU cache (4096 entries, 1-hour TTL) backed by persistent SQLite `query_embed_cache` table. Cache key is normalized query text + model identifier.

## Publish via GitHub API

If you need API-based sync (instead of `git push`), use:

```bash
# preview
scripts/publish-gh-api.sh --owner stouffer-labs --repo retrivio --dry-run

# sync tracked files
scripts/publish-gh-api.sh --owner stouffer-labs --repo retrivio
```

GitHub branding assets live in `.github/branding/`:
- `retrivio-mark-1024.png` (avatar/icon)
- `retrivio-social-1280x640.png` (social preview)

## Building

```bash
cargo build --release -p retrivio
```

Shell integration (recommended):

```bash
eval "$(retrivio init bash)"   # or: eval "$(retrivio init zsh)"
```

That enables interactive `cd` navigation via `retrivio <query>` in the current shell.
Legacy compatibility script remains available: `source scripts/retrivio-shell.sh`.

## CLI Commands

### Setup & Configuration

| Command | Description |
|---|---|
| `retrivio install [--no-system-install] [--no-download] [--no-shell-hook] [--legacy] [--venv <path>] [--bench]` | Bootstrap runtime deps, ensure embedded state, install shell hooks |
| `retrivio init [--root <path>] [--embed-backend ollama\|bedrock] [--embed-model <id>]` | Initialize Retrivio config |
| `retrivio init bash\|zsh\|fish` | Print shell integration snippet for `eval` |
| `retrivio setup` | Guided backend/auth/profile setup wizard |
| `retrivio auth [select\|status]` | Auth/profile selection and status |
| `retrivio config [edit\|show\|set <key> <value>\|autotune]` | Interactive config editor and helpers |
| `retrivio autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]` | History-driven ranking weight auto-tuning |
| `retrivio doctor [--fix]` | Runtime checks; `--fix` runs AWS preflight remediation |
| `retrivio version` | Print version |

### Tracking & Indexing

| Command | Description |
|---|---|
| `retrivio add <path> [path ...] [--exclude <pattern>] [--refresh\|--no-refresh]` | Start tracking a root directory |
| `retrivio del <path> [path ...] [--refresh\|--no-refresh]` | Stop tracking a root directory |
| `retrivio roots` | List tracked roots and their exclude patterns |
| `retrivio exclude <root> <pattern> ...` | Add exclude patterns to a tracked root |
| `retrivio include <root> <pattern> ...` | Remove exclude patterns from a tracked root |
| `retrivio index` | Run incremental index pass |
| `retrivio refresh [path ...]` | Force-refresh all or specific paths |
| `retrivio reembed` | Full vector rebuild after embedding model change |
| `retrivio watch --interval 30 --debounce-ms 900` | Event-driven watcher (fswatch) with polling fallback |

### Search & Navigation

| Command | Description |
|---|---|
| `retrivio search [--view projects\|files] [--limit <n>] <query...>` | Search across projects/files |
| `retrivio pick --query "<query>" [--view projects\|files] [--limit <n>]` | Interactive picker |
| `retrivio jump [--files\|--dirs] [--limit <n>] [query...]` | Jump to files or directories |

### Graph

| Command | Description |
|---|---|
| `retrivio ui [--host <addr>] [--port <n>]` | Start and open graph web UI in browser (preferred) |
| `retrivio graph neighbors [--path <path>] [--limit <n>] [--threshold <f>]` | Terminal neighbor table |
| `retrivio graph lineage [--path <path>] [--depth <n>] [--threshold <f>] [--limit <n>]` | Terminal lineage table |
| `retrivio graph doctor\|status` | Show embedded LanceDB readiness and path |
| `retrivio graph start\|stop\|provision` | Backward-compatible no-op (no external server needed) |
| `retrivio graph view\|open` | Deprecated aliases for `retrivio ui` |

### Daemon & API

| Command | Description |
|---|---|
| `retrivio daemon [start\|stop\|restart\|status\|logs [n]] [--host <addr>] [--port <n>]` | Daemon lifecycle management |
| `retrivio stop [--wait <seconds>]` | Backward-compatible no-op (kept for older scripts) |
| `retrivio api --port 8765` | Start local JSON API server |
| `retrivio mcp [serve\|doctor\|register\|unregister]` | MCP server over stdio / readiness check / IDE registration |

### Diagnostics

| Command | Description |
|---|---|
| `retrivio self-test [--query "<text>"] [--lifecycle] [--timeout <seconds>]` | Run internal self-tests |
| `retrivio bench [plan\|doctor\|export\|run]` | Backend benchmark harness |

### Picker & Config Keybinds

`retrivio pick` / `retrivio jump` keys:
- `Enter`: print selected path to stdout
- `Tab`: toggle directory/file mode
- `Ctrl-D` / `Ctrl-F`: switch to directory/file mode
- `Ctrl-U`: clear current query text

`retrivio config edit` keys:
- `Up/Down`: move, `Enter`: edit, `Left/Right`: cycle enum values
- `a`: autotune, `s`: save, `q`: discard

### Watch Notes

- Uses event-driven sync when `fswatch` is available, with automatic polling fallback
- `retrivio install` attempts to install `fswatch` via Homebrew when available
- `--debounce-ms` controls event-batch delay before indexing
- `--quiet` suppresses watch progress output

### Autotune Notes

`retrivio autotune` learns from `selection_events` and tunes ranking knobs automatically:
- Objective: maximize weighted MRR/hit-rate over real queries and picks
- Tuned families: chunk weights, project weights, graph expansion weights/limits, lexical/vector candidate sizes
- `--deep` runs additional rounds for higher quality
- `--dry-run` previews recommended values without writing config
- Reports saved to `~/.retrivio/autotune/latest.{json,md}` and timestamped snapshots (or your `--data-dir` path)

## Configuration Reference

All keys are set via `retrivio config set <key> <value>` or by editing `~/.retrivio/config.toml` directly.

### Embedding Backend

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `embed_backend` | string | `"ollama"` | `ollama`, `bedrock` | Embedding provider |
| `embed_model` | string | `"qwen3-embedding"` | -- | Model identifier (Ollama or Bedrock model ID) |
| `local_embed_dim` | integer | `384` | >= 64 | Embedding dimension for local models |

### AWS / Bedrock

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `aws_profile` | string | `""` | -- | AWS CLI profile name |
| `aws_region` | string | `""` | -- | AWS region for Bedrock |
| `aws_refresh_cmd` | string | `""` | -- | Credential refresh command (run once per process) |
| `bedrock_concurrency` | integer | `32` | 1-128 | Concurrent Bedrock API calls |
| `bedrock_max_retries` | integer | `3` | 0-12 | Max retry attempts per embedding call |
| `bedrock_retry_base_ms` | integer | `250` | 50-10000 | Exponential backoff base (ms) |

### Retrieval Backend

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `retrieval_backend` | string | `"lancedb"` | `lancedb` | Retrieval backend (only value currently supported) |

### Retrieval Tuning

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `lexical_candidates` | integer | `120` | 10-5000 | BM25 candidate count per search |
| `vector_candidates` | integer | `120` | 10-5000 | Vector kNN candidate count per search |
| `max_chars_per_project` | integer | `12000` | 1000-500000 | Max characters indexed per project |

### Chunk Ranking Weights

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `rank_chunk_semantic_weight` | float | `0.66` | 0.0-1.0 | Semantic similarity weight in chunk scoring |
| `rank_chunk_lexical_weight` | float | `0.24` | 0.0-1.0 | Lexical (BM25) weight in chunk scoring |
| `rank_chunk_graph_weight` | float | `0.10` | 0.0-1.0 | Graph expansion weight in chunk scoring |
| `rank_quality_mix` | float | `0.70` | 0.0-1.0 | Quality signal mixing factor |

### Relation Quality

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `rank_relation_quality_good_boost` | float | `0.08` | 0.0-1.0 | Score boost for `quality=good` relations |
| `rank_relation_quality_weak_penalty` | float | `0.20` | 0.0-1.0 | Score penalty for `quality=weak` relations |
| `rank_relation_quality_wrong_penalty` | float | `0.65` | 0.0-1.0 | Score penalty for `quality=wrong` relations |

### Project Ranking Weights

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `rank_project_content_weight` | float | `0.58` | 0.0-2.0 | Content-match weight in project scoring |
| `rank_project_semantic_weight` | float | `0.14` | 0.0-2.0 | Semantic similarity weight in project scoring |
| `rank_project_path_weight` | float | `0.10` | 0.0-2.0 | Path keyword weight in project scoring |
| `rank_project_graph_weight` | float | `0.10` | 0.0-2.0 | Graph neighbor weight in project scoring |
| `rank_project_frecency_weight` | float | `0.08` | 0.0-2.0 | Frecency (recent use) weight in project scoring |

### Graph Expansion

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `graph_seed_limit` | integer | `10` | 2-64 | Number of top chunks used as graph seeds |
| `graph_neighbor_limit` | integer | `100` | 8-500 | Max neighbors traversed per expansion |
| `graph_same_project_high` | float | `0.88` | 0.0-1.0 | Same-project weight when semantic >= 0.72 |
| `graph_same_project_low` | float | `0.76` | 0.0-1.0 | Same-project weight when semantic < 0.72 |
| `graph_related_base` | float | `0.20` | 0.0-1.0 | Cross-project base graph weight |
| `graph_related_scale` | float | `0.70` | 0.0-2.0 | Cross-project weight scale factor |
| `graph_related_cap` | float | `0.90` | 0.0-1.0 | Cross-project weight upper cap |

### Cross-Encoder Re-Ranking (Ollama)

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `reranker_enabled` | bool | `true` | -- | Enable LLM-based cross-encoder re-ranking |
| `reranker_model` | string | `"qwen3:0.6b"` | -- | Ollama model for re-ranking |
| `reranker_pool_size` | integer | `60` | 10-200 | Candidate pool size for re-ranking |
| `reranker_batch_size` | integer | `8` | 1-32 | Parallel batch size per re-rank round |
| `reranker_timeout_ms` | integer | `3000` | 500-30000 | Timeout per re-rank request (ms) |

### HyDE

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `hyde_enabled` | bool | `false` | -- | Enable Hypothetical Document Embedding (adds ~800ms) |

> **Note:** `reranker_*` and `hyde_enabled` are advanced keys not shown in `retrivio config show`. Set them by editing `~/.retrivio/config.toml` directly.

## MCP Tools Reference

Run the MCP server: `retrivio mcp`

Readiness check: `retrivio mcp doctor`

### Search Tools

| Tool | Parameters | Description |
|---|---|---|
| `search_projects` | `query` (required), `limit` (default 12) | Semantic search across tracked projects with ranked evidence docs |
| `search_files` | `query` (required), `limit` (default 20) | Semantic search across indexed files with project context |
| `search_chunks` | `query` (required), `limit` (default 30) | Semantic + keyword search across indexed chunks |
| `search_symbols` | `query` (required), `limit` (default 20) | AST-extracted symbol search with FTS5 prefix matching |

### Context & Graph Tools

| Tool | Parameters | Description |
|---|---|---|
| `get_related_chunks` | `chunk_id` (required), `limit` (default 20) | Graph lineage + semantic similarity for related chunks |
| `read_chunk` | `chunk_id` (required), `max_chars` (default 8000) | Full indexed text for a chunk |
| `read_document` | `path` (required), `max_chars` (default 120000) | Reconstructed document from chunk sequence |
| `pack_context` | `query` (required), `budget_chars` (default 12000), `seed_limit` (default 8), `related_per_seed` (default 3), `include_docs` (default false), `doc_max_chars` (default 12000) | AI-ready context package with deduplication and coherence ordering |
| `get_project_neighbors` | `path` (required), `limit` (default 20) | Relationship graph neighbors for a project |

### Curation Tools

| Tool | Parameters | Description |
|---|---|---|
| `list_relation_feedback` | `chunk_id` (required), `decision?`, `quality?`, `limit` (default 120) | List relation curation feedback history |
| `suppress_relation` | `source_chunk_id`, `target_chunk_id`, `relation` (all required), `note?` | Suppress a relation between chunks |
| `restore_relation` | `source_chunk_id`, `target_chunk_id`, `relation` (all required), `note?` | Restore a previously suppressed relation |
| `set_relation_quality` | `source_chunk_id`, `target_chunk_id`, `relation`, `quality_label` (all required), `note?` | Rate relation quality (good/weak/wrong/unspecified) |

### Lifecycle & Index Tools

| Tool | Parameters | Description |
|---|---|---|
| `list_tracked_roots` | (none) | List tracked root directories |
| `add_tracked_root` | `path` (required), `refresh` (default true) | Track new root and optionally index immediately |
| `remove_tracked_root` | `path` (required), `refresh` (default true) | Stop tracking a root directory |
| `run_incremental_index` | (none) | Run incremental index across all tracked roots |
| `run_forced_refresh` | `paths?` (optional array) | Force-refresh all roots or a specific subset |

### MCP Resource

- `retrivio://status` -- system status and health information

### Registration

```bash
codex mcp add retrivio -- "$(pwd)/retrivio" mcp
```

## API Endpoints

Run API server: `retrivio api --host 127.0.0.1 --port 8765`

### Health & Status

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tracked` | List tracked roots |

### Search

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?q=<query>&limit=<n>&view=projects` | Search projects |
| GET | `/search?q=<query>&limit=<n>&view=files` | Search files |

### Chunks

| Method | Endpoint | Description |
|---|---|---|
| GET | `/chunks/search?q=<query>&limit=<n>` | Search chunks |
| GET | `/chunks/related?chunk_id=<id>&limit=<n>` | Related chunks |
| GET | `/chunks/get?chunk_id=<id>&max_chars=<n>` | Read a chunk |
| GET | `/chunks/feedback?chunk_id=<id>&decision=<d>&quality=<q>&limit=<n>` | List relation feedback |
| POST | `/chunks/feedback/suppress` | Suppress relation (`source_chunk_id`, `target_chunk_id`, `relation`, `note?`) |
| POST | `/chunks/feedback/restore` | Restore relation (same body) |
| POST | `/chunks/feedback/quality` | Set quality (`quality_label`: good/weak/wrong/unspecified) |

### Context

| Method | Endpoint | Description |
|---|---|---|
| GET | `/context/pack?q=<query>&budget_chars=<n>&seed_limit=<n>&related_per_seed=<n>&include_docs=0\|1&doc_max_chars=<n>` | Pack context (GET) |
| POST | `/context/pack` | Pack context (JSON body: `query`, `budget_chars`, `seed_limit`, `related_per_seed`, `include_docs`, `doc_max_chars`) |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| GET | `/docs/read?path=<abs_or_tilde_path>&max_chars=<n>` | Read document from chunks |

### Graph

| Method | Endpoint | Description |
|---|---|---|
| GET | `/graph/neighbors?path=<abs_or_tilde_path>&limit=<n>` | Project neighbors |
| GET | `/graph/view/data?focus=<project_path>&limit=<n>` | Graph viewer data |
| GET | `/graph/view/chunks?path=<project_path>&limit=<n>` | Chunk list for project |
| GET | `/graph/view/related?chunk_id=<id>&limit=<n>` | Related chunks for graph viewer |

### Lifecycle

| Method | Endpoint | Description |
|---|---|---|
| POST | `/refresh` | Force refresh (`path` or `paths` array) |
| POST | `/select` | Record selection event (`path`, `query`) |
| POST | `/tracked/add` | Add tracked root (`path` or `paths` array) |
| POST | `/tracked/del` | Remove tracked root (`path` or `paths` array) |

Chunk payloads include a stable `schema` field (`chunk-search-v1`, `chunk-related-v1`, `chunk-get-v1`, `doc-read-v1`) for contract-safe consumers.

## Environment Variables

### Core Paths

By default, Retrivio always uses `~/.retrivio/` for config/state.

Per-run overrides (must come before subcommand/query):
- `--data-dir /path/to/state_dir`
- `--config /path/to/config.toml`

### Retrieval Runtime

Retrivio's retrieval path is fully embedded (`lancedb` + SQLite FTS5) and does not use external service environment variables.

### AWS / Bedrock

| Variable | Default | Description |
|---|---|---|
| `RETRIVIO_AWS_PROFILE` | (falls through to `AWS_PROFILE`) | AWS profile name |
| `RETRIVIO_AWS_REGION` | (falls through to `AWS_REGION` / `AWS_DEFAULT_REGION` / `us-east-1`) | AWS region |
| `RETRIVIO_AWS_REFRESH_CMD` | (none) | Credential refresh command |
| `RETRIVIO_AWS_REFRESH_ALWAYS` | `false` | Re-run refresh every call (vs. once per process) |
| `RETRIVIO_AWS_CLI` | `aws` | AWS CLI binary path |
| `RETRIVIO_BEDROCK_CONCURRENCY` | `32` | Concurrent Bedrock API calls (1-128) |
| `RETRIVIO_BEDROCK_MAX_RETRIES` | `3` | Max retry attempts (0-12) |
| `RETRIVIO_BEDROCK_RETRY_BASE_MS` | `250` | Retry backoff base (50-10000ms) |
| `RETRIVIO_BEDROCK_NORMALIZE` | `true` | Normalize Bedrock embedding vectors |

### Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API host |
| `RETRIVIO_OLLAMA_KEEP_ALIVE` | `24h` | Ollama model keep-alive duration |
| `RETRIVIO_OLLAMA_AUTOSTART` | `true` | Auto-attempt `ollama serve` when embedding calls cannot reach Ollama |
| `RETRIVIO_OLLAMA_AUTOSTART_ALWAYS` | `false` | Retry auto-start on every reachability failure (instead of once per process/host) |
| `RETRIVIO_OLLAMA_AUTOSTART_TIMEOUT_SEC` | `12` | Max seconds to wait for Ollama API after auto-start attempt (2-90) |

### API / Daemon

| Variable | Default | Description |
|---|---|---|
| `RETRIVIO_API_HOST` | `127.0.0.1` | API/daemon bind host |
| `RETRIVIO_API_PORT` | `8765` | API/daemon bind port |
| `RETRIVIO_API_START_TIMEOUT` | `8` | Daemon start timeout (seconds, min 1) |
| `RETRIVIO_API_TRACE` | `false` | Enable API request tracing |

### Query Cache

| Variable | Default | Range | Description |
|---|---|---|---|
| `RETRIVIO_QUERY_EMBED_CACHE_SIZE` | `4096` | 64-100000 | In-memory cache entries |
| `RETRIVIO_QUERY_EMBED_CACHE_TTL_SEC` | `3600` | 15-86400 | Cache TTL (seconds) |

### Legacy Bridge / Misc

| Variable | Default | Description |
|---|---|---|
| `RETRIVIO_PYTHON` | `python3` | Python interpreter for `retrivio legacy` bridge |
| `RETRIVIO_APP_VENV` | (none) | Optional fallback venv path for legacy app discovery |
| `RETRIVIO_PROGRESS_HEARTBEAT_MS` | `5000` | Progress heartbeat interval in milliseconds (clamped 500-60000) |

## Supported Languages

Retrivio uses tree-sitter for AST-aware code intelligence. The following languages have full support:

| Language | Extensions | AST Chunking | Symbol Extraction | Import Graph |
|---|---|---|---|---|
| Python | `.py`, `.pyi`, `.pyw` | Yes | Yes | Yes |
| JavaScript | `.js`, `.mjs`, `.cjs`, `.jsx` | Yes | Yes | Yes |
| TypeScript | `.ts`, `.mts`, `.cts` | Yes | Yes | Yes |
| TSX | `.tsx` | Yes | Yes | Yes |
| Rust | `.rs` | Yes | Yes | Yes |
| Go | `.go` | Yes | Yes | Yes |
| Java | `.java` | Yes | Yes | Yes |
| C | `.c`, `.h` | Yes | Yes | Yes |
| C++ | `.cc`, `.cpp`, `.cxx`, `.hpp`, `.hxx`, `.hh` | Yes | Yes | Yes |

All other text files are indexed using character-window chunking (~16% overlap, max 28 chunks per file). Embedding and search work for all file types; AST features (semantic chunks, symbols, imports) require a supported language.

## Architecture

### Storage: SQLite + LanceDB

See [Storage Architecture](#storage-architecture) above.

### Bedrock Embedding: Native HTTP + SigV4

When `embed_backend=bedrock`, Retrivio uses native HTTP requests with AWS SigV4 request signing in pure Rust:
- Resolves AWS credentials once via `aws configure export-credentials` and caches them
- Signs requests using HMAC-SHA256 (SigV4) without any AWS SDK dependency
- Uses `ureq::Agent` with connection pooling for HTTP/1.1 keep-alive
- Falls back to CLI subprocess automatically if native signing fails
- Supports Cohere batch models (up to 96 texts per request) and Titan single-text models

### LanceDB Vector Index

LanceDB vectors are maintained from SQLite embeddings:
- Indexing writes vectors to SQLite and upserts corresponding chunk vectors into embedded LanceDB
- `retrivio reembed` rebuilds vectors and then reconstructs LanceDB from SQLite
- `retrivio graph doctor` / `retrivio graph status` report LanceDB readiness and data path

### Exclude Patterns

Tracked roots support exclude patterns to skip specific directories during project discovery and indexing:

```bash
# Add a root with excludes
retrivio add ~/estouff --exclude Downloads --exclude Library --exclude .docker

# Add excludes to an existing root
retrivio exclude ~/estouff node_modules .cache

# Remove an exclude (re-include a directory)
retrivio include ~/estouff .cache

# View current excludes
retrivio roots
```

Excludes are relative directory paths from the root. They are checked during directory traversal using O(1) absolute path lookup, so excluded subtrees are never traversed.

## Paths

Default paths (all invocations):
- Config: `~/.retrivio/config.toml`
- Data: `~/.retrivio/`
- DB: `~/.retrivio/retrivio.db`

Per-run override flags (must come before subcommand/query):
- `retrivio --data-dir /path/to/state_dir <command...>`
- `retrivio --config /path/to/config.toml <command...>`

## Graph Viewer

```bash
retrivio ui --host 127.0.0.1 --port 8780
# open http://127.0.0.1:8780/
```

Open graph viewer in one command:

```bash
retrivio ui
```

`retrivio ui` prints the active URL. If port 8780 is occupied/stale, Retrivio can launch on a different free port.

Compatibility notes:
- `retrivio graph view` and `retrivio graph open` still work but are deprecated aliases.
- `retrivio graph start|stop|provision` are no-ops (kept for older scripts).

If the browser says `No graph data yet`:

```bash
retrivio doctor                 # verify db path + tracked roots
retrivio graph neighbors --limit 5
retrivio index                 # only needed if neighbors are empty
retrivio ui
```

Terminal graph summaries (table output):

```bash
# direct outgoing neighbors for a project
retrivio graph neighbors --path ~/c-projects/202601-storage-cost-analysis --limit 12 --threshold 0.60

# lineage around a focus project (incoming + outgoing + relay edges when depth > 1)
retrivio graph lineage --path ~/c-projects/202601-storage-cost-analysis --depth 2 --threshold 0.60 --limit 20
```

Graph viewer highlights:
- project graph canvas with selectable project nodes
- chunk inspector per project with chunk-level drill-down
- related chunk panel with explainability (`Why`) and recursive `Drill` traversal
- relation curation in viewer:
  - `Suppress` / `Restore`
  - quality labels per relation (`Good` / `Weak` / `Wrong` / `Clear`)
  - local feedback history (decision + quality + note + timestamps)

## Embedding Model Changes

When `embed_model` changes, Retrivio marks the index as migration-required. Search/pick/jump/API/MCP calls are blocked until re-embed is complete. Run `retrivio reembed` to rebuild vectors and reconstruct the embedded LanceDB index in one step.

## Self-Test

`retrivio self-test --lifecycle` runs a one-shot native lifecycle probe:
- daemon start/health/stop on a temporary port
- retrieval backend probe (embedded LanceDB readiness)
- post-probe retrieval search verification
