# Retrivio API Help

This guide contains API-specific reference material that was moved out of the main `README.md`.

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
| GET | `/search?q=<query>&limit=<n>&view=files&since_days=<n>` | Search files (`since_days` optional: drop files whose content date is older than `n` days) |
| GET | `/search/pick?q=<query>&timeout=<seconds>&verbose=<0\|1>&mode=<dirs\|files\|projects>` | Interactive picker response payload used by shell integration |

`retrivio search --json [--since <days>]` prints exactly the `/search` payload for the chosen view.

#### Freshness fields (file, chunk and evidence results)

Every file result, chunk result, evidence hit and `/context/pack` `chunks[]` entry carries:

| Field | Type | Meaning |
|---|---|---|
| `doc_mtime` | number | File modification time (unix seconds) recorded at index time (file/chunk results) |
| `content_date` | number | Date used for ranking: the newer of the path date (`YYYYMMDD`/`YYYYMM`/`YYYY-MM-DD` prefix on a path component) and `doc_mtime`, unix seconds |
| `date_source` | string | `path-date` or `mtime` |
| `age_days` | number | `now - content_date` in days, never negative |
| `freshness_tier` | string | `fresh` (< 14 d), `aging` (14–35 d), `stale` (> 35 d); `record` for point-in-time documents matched by `recency_record_patterns` |
| `is_record` | bool | True when the path matches `recency_record_patterns` |

Project results (`view=projects`, `search_projects`) carry `recency`: the relevance-weighted mean recency (0..1) of the project's evidence chunks. Scores already include the recency blend `final = (1 - w) * score + w * 0.5^(age_days / half_life)`; see the README "Freshness" section for the config keys.

### Chunks

| Method | Endpoint | Description |
|---|---|---|
| GET | `/chunks/search?q=<query>&limit=<n>&since_days=<n>` | Search chunks (`since_days` optional, same semantics as `/search`) |
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
| POST | `/refresh` | Force refresh (`path` or `paths` array). No paths: every tracked root. Each path must be a tracked root (discovery runs on it) or a project directory (exactly that project is re-collected; its child directories are never indexed as projects). Any other path returns 400 with a message naming the project or root to use. The MCP tool `run_forced_refresh` applies the same rules and reports the resolved `roots` and `projects` |
| POST | `/select` | Record selection event (`path` required, `query` optional) |
| POST | `/tracked/add` | Add tracked root (`path` or `paths` array) |
| POST | `/tracked/del` | Remove tracked root (`path` or `paths` array) |

Chunk payloads include a stable `schema` field (`chunk-search-v2`, `chunk-related-v1`, `chunk-get-v1`, `doc-read-v1`) for contract-safe consumers. `chunk-search-v2` added the freshness fields (`doc_mtime`, `content_date`, `date_source`, `age_days`, `freshness_tier`, `is_record`) to every result; `chunk-search-v1` consumers only need to ignore the extra keys.

The MCP tools `search_files` and `search_chunks` accept the same optional `since_days` argument, and `search_files` / `search_chunks` / `pack_context` return the same freshness fields as the HTTP endpoints.

## API / Daemon Env Vars

| Variable | Default | Description |
|---|---|---|
| `RETRIVIO_API_HOST` | `127.0.0.1` | API/daemon bind host |
| `RETRIVIO_API_PORT` | `8765` | API/daemon bind port |
| `RETRIVIO_API_START_TIMEOUT` | `8` | Daemon start timeout (seconds, min 1) |
| `RETRIVIO_API_TRACE` | `false` | Enable API request tracing |
