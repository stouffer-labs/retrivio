# retrivio

<p align="center">
  <img src="https://stouffer-labs.github.io/retrivio-logo-v2.svg" alt="retrivio logo" width="120" />
</p>

<p align="center">
  <img src="https://stouffer-labs.github.io/retrivio-social-1280x640.png" alt="retrivio social banner" width="840" />
</p>

`retrivio` is a semantic intelligence and navigation index for project workspaces. It combines AST-aware chunking, multi-stage retrieval, and graph-based context expansion to power fast project search and AI-ready context packaging. Think of it like a scalable memory recall system, providing broad project-ware context to the LLM.

Supports:
- semantic + lexical + frecency + graph ranking with query-adaptive weights
- AST-aware code chunking via tree-sitter (9 languages)
- cross-encoder re-ranking and HyDE for high-precision retrieval
- interactive terminal picker for fast selection
- local HTTP API if you want to integrate retrivio
- MCP server makes it compatible with popular CLI coding tools

## Quick Start

```bash
cd /path/to/repo

# Initialize Retrivio (embedded LanceDB + SQLite under ~/.retrivio by default)
retrivio install

# Install ollama if you want local model capabilities
brew services start ollama

# Configure Retrivio model providers
retrivio setup

# Track and index your workspace roots
retrivio add ~/projects

# Optional: add a root with excludes
retrivio add ~/projects --exclude node_modules --exclude .cache --exclude dist

# Search for a term (non-interactive) "s3vectors" example
retrivio search s3vectors

# Preflight identity + model access checks
retrivio doctor --fix
```

## How It Works

### Storage Architecture

Retrivio uses **SQLite** as the durable source of truth and **embedded LanceDB** as the local vector index.

- **SQLite** stores all project metadata, chunks, embeddings, edges, symbols, imports, and tracked roots. It is portable and requires zero infrastructure.
- **LanceDB** stores ANN vectors under `.retrivio/lance` and runs in-process (no external service).
- **SQLite FTS5** powers lexical retrieval (`project_fts`, `chunk_fts`, `symbol_fts`) plus graph relationship lookups (`project_edges`, `file_dependency_edges`).

This design is purpose built to handle at least **100-200GB of context-worthy project files across 10,000+ directories** on a single Macbook M1:

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


## Building

```bash
cargo build --release -p retrivio
```


## CLI Commands

### Setup & Configuration

| Command | Description |
|---|---|
| `retrivio setup` | Guided backend/auth/profile setup wizard |

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

### Watch Notes

- Uses event-driven sync when `fswatch` is available, with automatic polling fallback
- `retrivio install` attempts to install `fswatch` via Homebrew when available
- `--debounce-ms` controls event-batch delay before indexing
- `--quiet` suppresses watch progress output



## MCP Tools Reference

Run the MCP server: `retrivio mcp`

Readiness check: `retrivio mcp doctor`



## API Reference

- [`docs/API_HELP.md`](docs/API_HELP.md)


## Supported Languages for 'smart chunking' (AST-aware)

Retrivio uses tree-sitter for AST-aware code intelligence. This produces better optimized document chunks for use with vector retrieval systems. The following languages have full support:

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




### LanceDB Vector Index

LanceDB vectors are maintained from SQLite embeddings:
- Indexing writes vectors to SQLite and upserts corresponding chunk vectors into embedded LanceDB
- `retrivio reembed` rebuilds vectors and then reconstructs LanceDB from SQLite
- `retrivio graph doctor` / `retrivio graph status` report LanceDB readiness and data path

### Exclude Patterns

Tracked roots support exclude patterns to skip specific directories during project discovery and indexing:

```bash
# Add a root with excludes
retrivio add ~/projects --exclude node_modules --exclude .cache --exclude dist

# Add excludes to an existing root
retrivio exclude ~/projects node_modules .cache

# Remove an exclude (re-include a directory)
retrivio include ~/projects .cache

# View current excludes
retrivio roots
```

Excludes are relative directory paths from the root. They are checked during directory traversal using O(1) absolute path lookup, so excluded subtrees are never traversed.

## Paths

Default paths (all invocations):
- Config: `~/.retrivio/config.toml`
- Data: `~/.retrivio/`
- DB: `~/.retrivio/retrivio.db`

## Graph Viewer

```bash
retrivio ui --host 127.0.0.1 --port 8780
# open http://127.0.0.1:8780/
```

Open graph viewer in one command:

```bash
retrivio ui
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
