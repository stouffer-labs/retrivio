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
| GET | `/search?q=<query>&limit=<n>&view=files&since_days=<n>&include_superseded=1` | Search files (`since_days` optional: drop files whose content date is older than `n` days; `include_superseded` optional: rank older handoffs of a series at full strength) |
| GET | `/search/pick?q=<query>&timeout=<seconds>&verbose=<0\|1>&mode=<dirs\|files\|projects>` | Interactive picker response payload used by shell integration |

`retrivio search --json [--since <days>] [--include-superseded]` prints exactly the `/search` payload for the chosen view.

#### Freshness, role and score fields (file, chunk and evidence results)

Every file result, chunk result, evidence hit and `/context/pack` `chunks[]` entry carries:

| Field | Type | Meaning |
|---|---|---|
| `doc_mtime` | number | File modification time (unix seconds) recorded at index time (file/chunk results) |
| `content_date` | number | Date used for ranking, unix seconds. State and knowledge: the newer of the path date (`YYYYMMDD`/`YYYYMM`/`YYYY-MM-DD` prefix on a path component) and `doc_mtime`. Records: the path date when there is one (the event), else `doc_mtime` |
| `date_source` | string | `path-date` or `mtime` |
| `age_days` | number | `now - content_date` in days, never negative |
| `freshness_tier` | string | `fresh` (< 14 d), `aging` (14–35 d), then `stale` (knowledge) or `verify` (state) over 35 d; `record` for records at any age |
| `role` | string | `state` (handoffs, status briefs, plans), `knowledge` (specs, notes, code, documents) or `record` (transcripts, call and meeting notes, customer signals), decided from the path relative to the project and, for `.txt`, the text shape; see the README "Roles and supersession" |
| `verify` | bool | True for `state` older than 35 days: it was current once and must be re-checked before its facts are repeated |
| `is_record` | bool | Compatibility alias: `role == "record"` |
| `noise` | bool | True for machine artefacts (chat dumps, `.jsonl`/`.log`, lockfiles, minified code); their `quality` is low (file and chunk results) |
| `raw_similarity` | number or null | Cosine similarity between the query and the result's best chunk, in [-1, 1]; the honest absolute number that `recall_min_abs_score` and `search_min_abs_score` compare against. `null` only in lexical-only retrieval (recall's fallback) |
| `superseded_by` | string or null | File results only. For a `state` file, the path of the newest file of the same series (project, parent directory, normalised stem; newest by the date in the path, else the last edit) when this one is not it; such results are downranked unless `include_superseded` is set or the query asks for history explicitly (see README "Roles and supersession") |

`score` and `semantic` stay relative: `semantic` is min-max normalised over the query's vector hits (the best hit is 1.0) and `score` is the fused, recency-blended value. Compare them between results of one query, never with a floor.

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
| POST | `/refresh` | Force refresh (`path` or `paths` array). No paths: every tracked root. Each path must be a tracked root (discovery runs on it, including the root's own "root files" project) or a project directory (exactly that project is re-collected; its child directories are never indexed as projects). Any other path returns 400 with a message naming the project or root to use. While another writer (`index`, `prune`, `watch`, another refresh) holds the index lock the call returns 409 with `index busy: another retrivio writer is running (pid N)`. The response carries a `stats` object (see below). The MCP tool `run_forced_refresh` applies the same rules and reports the resolved `roots` and `projects` |
| POST | `/select` | Record selection event (`path` required, `query` optional) |
| POST | `/tracked/add` | Add tracked root (`path` or `paths` array). Taken under the index writer lock like every other write; 409 with `index busy: ...` while another writer runs |
| POST | `/tracked/del` | Remove tracked root (`path` or `paths` array). Same lock and 409 as `/tracked/add` |

#### Index statistics (`stats` of `POST /refresh`, MCP `run_incremental_index` and `run_forced_refresh`)

| Field | Meaning |
|---|---|
| `total_projects`, `updated_projects`, `skipped_projects`, `removed_projects` | Projects discovered, re-collected, skipped by the change gate, and removed because their directory is gone |
| `vectorized_projects` | Project summary vectors embedded (a summary is re-embedded only when its text changed) |
| `files_selected`, `files_unchanged`, `files_rechunked` | Files the scans selected; of those, how many the manifest showed unchanged (never read) and how many were read and chunked |
| `files_unreadable`, `projects_incomplete` | Directory entries that could not be read, and projects with at least one; such a project is indexed from what was readable, nothing of it is pruned and its signature does not advance |
| `files_evicted_by_cap`, `files_truncated_by_cap` | Files left out entirely by `max_files_per_project` / `max_chunks_per_project`, and files indexed only in part |
| `documents_extracted`, `documents_failed` | Documents (docx, pptx, odt, odp, xlsx, pdf, html) whose text was extracted this run, and documents that yielded none (over `max_document_bytes` or `max_document_uncompressed_bytes`, corrupt, parser error or panic, PDF child killed at `document_extract_timeout_ms` or 1 GiB). A failed document that was indexed before keeps its old content; one never indexed is absent |
| `projects_failed`, `failures` | Projects whose run failed and `"<project>: <reason>"` for each; every one keeps its previous state (old signature, nothing pruned, nothing published) and is retried next run; the fingerprints do not advance |
| `stopped` | Non-empty when the run stopped before visiting every project (the embedding backend failed after retries); names the reason and how many projects were left |
| `lance_error` | Non-empty when LanceDB could not be opened, repaired or written this run; the dirty marker is set and the next writer run repairs LanceDB from the SQLite vectors, which are complete |
| `chunk_rows`, `chunk_vectors`, `chunks_embedded`, `chunks_reused`, `chunks_deleted` | Chunks the scanned projects hold after the run, vectors written, chunks sent to the embedder, chunks whose stored vector was reused, chunk rows deleted |
| `lance_repaired`, `lance_orphans_removed` | LanceDB rows rebuilt from SQLite vectors and LanceDB rows without a SQLite vector removed by the repair step |
| `graph_edges`, `retrieval_backend`, `retrieval_synced_chunks`, `retrieval_error`, `vector_failures`, `tracked_roots` | Project graph edges rebuilt, backend name, LanceDB row count, last sync error if any, embedding failures, roots covered |

Chunk payloads include a stable `schema` field (`chunk-search-v2`, `chunk-related-v1`, `chunk-get-v1`, `doc-read-v1`) for contract-safe consumers. `chunk-search-v2` added the freshness fields (`doc_mtime`, `content_date`, `date_source`, `age_days`, `freshness_tier`, `is_record`) to every result; version 0.2.0 adds `role`, `verify`, `noise` and `raw_similarity` under the same schema name (additive keys); `chunk-search-v1` consumers only need to ignore the extra keys.

The MCP tools `search_files` and `search_chunks` accept the same optional `since_days` argument, `search_files` also accepts `include_superseded` (boolean), and `search_files` / `search_chunks` / `pack_context` return the same freshness, role and score fields as the HTTP endpoints.

## API / Daemon Env Vars

| Variable | Default | Description |
|---|---|---|
| `RETRIVIO_API_HOST` | `127.0.0.1` | API/daemon bind host |
| `RETRIVIO_API_PORT` | `8765` | API/daemon bind port |
| `RETRIVIO_API_START_TIMEOUT` | `8` | Daemon start timeout (seconds, min 1) |
| `RETRIVIO_API_TRACE` | `false` | Enable API request tracing |
