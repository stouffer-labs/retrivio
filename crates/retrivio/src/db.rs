//! SQLite access: schema and migrations, connection helpers, the writer lock, app_state keys, the reembed requirement, tracked roots and the process-wide LanceDB handle.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use std::{fs, process, thread};

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use crate::config::{data_dir, db_path, ConfigValues};
use crate::embed::model_key_for_cfg;
use crate::index::{embed_input_for, embed_input_hash};
use crate::lance_store;
use crate::util::{bool_env, chrono_like_now, normalize_path, now_ts};

pub(crate) const APP_STATE_ACTIVE_MODEL_KEY: &str = "active_model_key";
pub(crate) const APP_STATE_REEMBED_REQUIRED: &str = "reembed_required";
pub(crate) const APP_STATE_REEMBED_REASON: &str = "reembed_reason";
/// The embedding fingerprint (model key, dimension, normalisation, pipeline version; see
/// [`EmbedIdentity::fingerprint`]) the store was last completely indexed with. A different
/// current fingerprint revisits every project so the per-chunk identity check decides.
pub(crate) const APP_STATE_EMBED_FINGERPRINT: &str = "embedding_fingerprint";
/// The scan caps and document settings of the last complete run (see [`ScanCaps::fingerprint`]).
pub(crate) const APP_STATE_SCAN_CAPS_FINGERPRINT: &str = "scan_caps_fingerprint";
/// "1" from the moment a batch of vectors is committed to sqlite until the matching LanceDB
/// write succeeded; so a failed or interrupted Lance write leaves the marker set and the next
/// writer run repairs LanceDB from sqlite (no embedding). Absent on stores that predate it,
/// which makes the first writer run reconcile once.
pub(crate) const APP_STATE_LANCE_DIRTY: &str = "lance_dirty";

pub(crate) static LANCE_STORE: OnceLock<Mutex<Option<lance_store::LanceStore>>> = OnceLock::new();

pub(crate) fn get_or_open_lance(cwd: &Path, dim: usize) -> Result<(), String> {
    let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(store) = guard.as_ref() {
        if lance_store::dim(store) == dim {
            return Ok(());
        }
        *guard = None;
    }
    let lance_path = data_dir(cwd).join("lance");
    if !lance_path.exists() {
        std::fs::create_dir_all(&lance_path)
            .map_err(|e| format!("failed creating lance directory: {}", e))?;
    }
    let store = lance_store::open(&lance_path, dim)?;
    *guard = Some(store);
    Ok(())
}

pub(crate) fn with_lance_store<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut lance_store::LanceStore) -> Result<R, String>,
{
    let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    match guard.as_mut() {
        Some(store) => f(store),
        None => Err("LanceDB not initialized; run index first".to_string()),
    }
}

/// True once `get_or_open_lance` (or a rebuild) has installed a store for this process.
pub(crate) fn lance_store_is_open() -> bool {
    LANCE_STORE
        .get()
        .map(|lock| lock.lock().unwrap_or_else(|p| p.into_inner()).is_some())
        .unwrap_or(false)
}

pub(crate) fn ensure_retrieval_backend_ready(
    _cfg: &ConfigValues,
    _auto_start: bool,
    _context: &str,
) -> Result<(), String> {
    // LanceDB is embedded — no external server process needed.
    // Just ensure the lance directory exists.
    let lance_path = data_dir(Path::new("")).join("lance");
    if !lance_path.exists() {
        fs::create_dir_all(&lance_path)
            .map_err(|e| format!("failed creating lance directory: {}", e))?;
    }
    Ok(())
}

pub(crate) fn list_project_paths(conn: &Connection) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing project list query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying project paths: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project path row: {}", e))?);
    }
    Ok(out)
}

pub(crate) fn vector_dim_from_sqlite(conn: &Connection, model_key: &str) -> Option<usize> {
    conn.query_row(
        "SELECT dim FROM project_chunk_vectors WHERE model = ?1 LIMIT 1",
        params![model_key],
        |row| row.get::<_, i64>(0),
    )
    .ok()
    .map(|d| d.max(1) as usize)
}

/// Make sure a store exists (creating a brand-new one). Never migrates an existing store.
pub(crate) fn ensure_db_schema(db_path: &Path) -> Result<(), String> {
    let _ = open_db_rw(db_path)?;
    Ok(())
}

pub(crate) fn ensure_tracked_root(
    db_path: &Path,
    path: &Path,
    added_at: f64,
) -> Result<(), String> {
    let conn = open_db_rw(db_path)?;
    ensure_tracked_root_conn(&conn, path, added_at)
}

pub(crate) fn remove_tracked_root(db_path: &Path, path: &Path) -> Result<i64, String> {
    let conn = open_db_rw(db_path)?;
    conn.execute(
        "DELETE FROM tracked_roots WHERE path = ?1",
        params![path.to_string_lossy().to_string()],
    )
    .map(|n| n as i64)
    .map_err(|e| format!("failed to remove tracked root: {}", e))
}

pub(crate) fn list_tracked_roots(db_path: &Path) -> Result<Vec<PathBuf>, String> {
    let conn = open_db_rw(db_path)?;
    list_tracked_roots_conn(&conn)
}

pub(crate) fn get_exclude_patterns_conn(
    conn: &Connection,
    root_path: &Path,
) -> Result<Vec<String>, String> {
    let raw: String = conn
        .query_row(
            "SELECT exclude_patterns FROM tracked_roots WHERE path = ?1 AND enabled = 1",
            params![root_path.to_string_lossy().to_string()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed reading exclude patterns: {}", e))?;
    Ok(raw
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

pub(crate) fn add_exclude_patterns_conn(
    conn: &Connection,
    root_path: &Path,
    patterns: &[String],
) -> Result<(), String> {
    let existing = get_exclude_patterns_conn(conn, root_path).unwrap_or_default();
    let mut merged: Vec<String> = existing;
    for p in patterns {
        let trimmed = p.trim().to_string();
        if !trimmed.is_empty() && !merged.contains(&trimmed) {
            merged.push(trimmed);
        }
    }
    merged.sort();
    let joined = merged.join("\n");
    conn.execute(
        "UPDATE tracked_roots SET exclude_patterns = ?1 WHERE path = ?2",
        params![joined, root_path.to_string_lossy().to_string()],
    )
    .map_err(|e| format!("failed updating exclude patterns: {}", e))?;
    Ok(())
}

pub(crate) fn remove_exclude_patterns_conn(
    conn: &Connection,
    root_path: &Path,
    patterns: &[String],
) -> Result<usize, String> {
    let existing = get_exclude_patterns_conn(conn, root_path).unwrap_or_default();
    let remove_set: HashSet<&str> = patterns.iter().map(|s| s.trim()).collect();
    let filtered: Vec<String> = existing
        .into_iter()
        .filter(|p| !remove_set.contains(p.as_str()))
        .collect();
    let removed = patterns.len().saturating_sub(filtered.len());
    let joined = filtered.join("\n");
    conn.execute(
        "UPDATE tracked_roots SET exclude_patterns = ?1 WHERE path = ?2",
        params![joined, root_path.to_string_lossy().to_string()],
    )
    .map_err(|e| format!("failed updating exclude patterns: {}", e))?;
    Ok(removed)
}

pub(crate) fn app_state_get(conn: &Connection, key: &str) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT value FROM app_state WHERE key = ?1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|e| format!("failed reading app state '{}': {}", key, e))
}

pub(crate) fn app_state_set(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO app_state(key, value, updated_at)
VALUES (?1, ?2, ?3)
ON CONFLICT(key) DO UPDATE SET
    value = excluded.value,
    updated_at = excluded.updated_at
"#,
        params![key, value, now_ts()],
    )
    .map_err(|e| format!("failed writing app state '{}': {}", key, e))?;
    Ok(())
}

pub(crate) fn app_state_bool(conn: &Connection, key: &str) -> Result<bool, String> {
    let raw = app_state_get(conn, key)?.unwrap_or_default();
    let v = raw.trim().to_ascii_lowercase();
    Ok(matches!(v.as_str(), "1" | "true" | "yes" | "on"))
}

pub(crate) fn has_any_indexed_vectors(conn: &Connection) -> Result<bool, String> {
    let chunk_any: Option<i64> = conn
        .query_row("SELECT 1 FROM project_chunk_vectors LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|e| format!("failed checking chunk vectors: {}", e))?;
    if chunk_any.is_some() {
        return Ok(true);
    }
    let project_any: Option<i64> = conn
        .query_row("SELECT 1 FROM project_vectors LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|e| format!("failed checking project vectors: {}", e))?;
    Ok(project_any.is_some())
}

pub(crate) fn model_change_reembed_reason(old_model_key: &str, new_model_key: &str) -> String {
    format!(
        "embedding model changed from '{}' to '{}'; run `retrivio reembed` before searching.",
        old_model_key, new_model_key
    )
}

pub(crate) fn dominant_chunk_vector_model(conn: &Connection) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT model FROM project_chunk_vectors GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1",
        [],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|e| format!("failed reading dominant chunk vector model: {}", e))
}

pub(crate) fn chunk_vector_compatibility_reason(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    let any_chunk_vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM project_chunk_vectors", [], |row| {
            row.get(0)
        })
        .map_err(|e| format!("failed counting chunk vectors: {}", e))?;
    if any_chunk_vectors <= 0 {
        return Ok(None);
    }

    let current_model_key = model_key_for_cfg(cfg);
    let total_chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM project_chunks", [], |row| row.get(0))
        .map_err(|e| format!("failed counting project chunks: {}", e))?;
    let current_model_vectors: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM project_chunk_vectors WHERE model = ?1",
            params![current_model_key.clone()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed counting vectors for current model: {}", e))?;

    if current_model_vectors == 0 {
        let existing_model =
            dominant_chunk_vector_model(conn)?.unwrap_or_else(|| "<unknown>".to_string());
        return Ok(Some(format!(
            "existing chunk embeddings are for '{}' but current config expects '{}'; run `retrivio reembed` before searching.",
            existing_model, current_model_key
        )));
    }

    if total_chunks > 0 && current_model_vectors < total_chunks {
        return Ok(Some(format!(
            "chunk embeddings for '{}' are incomplete ({} of {} chunks); run `retrivio reembed` before searching.",
            current_model_key, current_model_vectors, total_chunks
        )));
    }

    let dim_variants: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT dim) FROM project_chunk_vectors WHERE model = ?1",
            params![current_model_key.clone()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed checking vector dimensions for current model: {}", e))?;
    if dim_variants > 1 {
        return Ok(Some(format!(
            "chunk embeddings for '{}' have inconsistent dimensions ({} variants); run `retrivio reembed` before searching.",
            current_model_key, dim_variants
        )));
    }

    Ok(None)
}

pub(crate) fn sync_reembed_requirement_state(
    cwd: &Path,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    let conn = open_db_rw(&db_path(cwd))?;
    let current_model_key = model_key_for_cfg(cfg);

    if let Some(reason) = chunk_vector_compatibility_reason(&conn, cfg)? {
        if let Some(existing_model) = dominant_chunk_vector_model(&conn)? {
            if !existing_model.trim().is_empty() && existing_model != current_model_key {
                app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &existing_model)?;
            }
        }
        app_state_set(&conn, APP_STATE_REEMBED_REQUIRED, "1")?;
        app_state_set(&conn, APP_STATE_REEMBED_REASON, &reason)?;
        return Ok(Some(reason));
    }

    if has_any_indexed_vectors(&conn)? {
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &current_model_key)?;
    }
    app_state_set(&conn, APP_STATE_REEMBED_REQUIRED, "0")?;
    app_state_set(&conn, APP_STATE_REEMBED_REASON, "")?;
    Ok(None)
}

pub(crate) fn refresh_reembed_requirement_for_config_change(
    cwd: &Path,
    before: &ConfigValues,
    after: &ConfigValues,
) -> Result<Option<String>, String> {
    // Preserve existing behavior that marks migration required on explicit model changes.
    let _ = mark_reembed_required_if_model_changed(cwd, before, after)?;
    // Then run a quick compatibility check so prompts/warnings reflect current DB reality.
    sync_reembed_requirement_state(cwd, after)
}

pub(crate) fn mark_reembed_required_if_model_changed(
    cwd: &Path,
    before: &ConfigValues,
    after: &ConfigValues,
) -> Result<Option<String>, String> {
    let old_model_key = model_key_for_cfg(before);
    let new_model_key = model_key_for_cfg(after);
    if old_model_key == new_model_key {
        return Ok(None);
    }
    let conn = open_db_rw(&db_path(cwd))?;
    let has_vectors = has_any_indexed_vectors(&conn)?;
    let reason = model_change_reembed_reason(&old_model_key, &new_model_key);
    if has_vectors {
        // Keep old model key so reembed knows what to migrate from.
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &old_model_key)?;
    } else {
        // No vectors yet — just record the new model for future embeds.
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &new_model_key)?;
    }
    app_state_set(
        &conn,
        APP_STATE_REEMBED_REQUIRED,
        if has_vectors { "1" } else { "0" },
    )?;
    app_state_set(
        &conn,
        APP_STATE_REEMBED_REASON,
        if has_vectors { &reason } else { "" },
    )?;
    Ok(Some(reason))
}

/// Why a re-embed is needed before searching, or `None`. Pure over `app_state`: the stored
/// flag and reason, plus a model key that differs from the one the store was embedded with,
/// are combined here without writing anything, so read-only connections (search, the MCP
/// status resource, the dossier, `doctor`) can ask. Writers normalise the stored state with
/// [`persist_reembed_requirement`].
pub(crate) fn reembed_requirement_reason(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    Ok(reembed_requirement(conn, cfg)?.map(|(reason, _)| reason))
}

/// [`reembed_requirement_reason`] plus whether the stored state differs from the computed one
/// (a writer should persist it).
pub(crate) fn reembed_requirement(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<(String, bool)>, String> {
    let current_model_key = model_key_for_cfg(cfg);
    let stored_required = app_state_bool(conn, APP_STATE_REEMBED_REQUIRED)?;
    let last_model_key = app_state_get(conn, APP_STATE_ACTIVE_MODEL_KEY)?.unwrap_or_default();
    let last_model_key = last_model_key.trim();
    let stored_reason = app_state_get(conn, APP_STATE_REEMBED_REASON)?
        .unwrap_or_default()
        .trim()
        .to_string();
    let model_changed = !last_model_key.is_empty() && last_model_key != current_model_key;
    if !stored_required && !model_changed {
        return Ok(None);
    }
    let reason = if !stored_reason.is_empty() && stored_required {
        stored_reason.clone()
    } else if model_changed {
        model_change_reembed_reason(last_model_key, &current_model_key)
    } else {
        format!(
            "embedding model migration required for '{}'; run `retrivio reembed` before searching.",
            current_model_key
        )
    };
    let stale = !stored_required || stored_reason != reason;
    Ok(Some((reason, stale)))
}

/// Writer-side companion of [`reembed_requirement_reason`]: stores the flag and the reason
/// when the computed requirement differs from what `app_state` holds (a model change seen for
/// the first time, or a flag without a reason). Only commands holding the writer lock call it.
pub(crate) fn persist_reembed_requirement(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    match reembed_requirement(conn, cfg)? {
        Some((reason, stale)) => {
            if stale {
                app_state_set(conn, APP_STATE_REEMBED_REQUIRED, "1")?;
                app_state_set(conn, APP_STATE_REEMBED_REASON, &reason)?;
            }
            Ok(Some(reason))
        }
        None => Ok(None),
    }
}

pub(crate) fn ensure_reembed_ready(
    conn: &Connection,
    cfg: &ConfigValues,
    context: &str,
) -> Result<(), String> {
    if let Some(reason) = reembed_requirement_reason(conn, cfg)? {
        return Err(format!("{} blocked: {}", context, reason));
    }
    Ok(())
}

pub(crate) fn mark_reembed_completed(conn: &Connection, model_key: &str) -> Result<(), String> {
    app_state_set(conn, APP_STATE_ACTIVE_MODEL_KEY, model_key)?;
    app_state_set(conn, APP_STATE_REEMBED_REQUIRED, "0")?;
    app_state_set(conn, APP_STATE_REEMBED_REASON, "")?;
    Ok(())
}

pub(crate) fn record_selection_event(
    conn: &Connection,
    query: &str,
    path: &str,
    selected_at: f64,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO selection_events(query, path, selected_at)
VALUES (?1, ?2, ?3)
"#,
        params![query, path, selected_at],
    )
    .map_err(|e| format!("failed recording selection event: {}", e))?;
    Ok(())
}

/// Open the database read-write WITHOUT migrating it.
///
/// Every command that is not a writer (search, recall, the MCP read tools, the API read
/// endpoints, `roots`, `config`, ...) opens this way; none of them depends on a column a
/// migration adds. The one exception is a brand-new store (no `projects` table yet), which is
/// created in full here: there is nothing to migrate, and `add --no-refresh`, `roots` and
/// friends must work before the first `index`. Schema migrations of an existing store run
/// only through [`open_db_writer`], under the writer lock.
pub(crate) fn open_db_rw(db_path: &Path) -> Result<Connection, String> {
    let conn = open_db_rw_raw(db_path)?;
    if !db_has_table(&conn, "projects")? {
        init_schema(&conn)?;
    }
    Ok(conn)
}

/// Open the database for a writer that holds the lock and bring the schema up to date. Every
/// CREATE, ALTER and backfill runs inside one `BEGIN IMMEDIATE` transaction with the column
/// checks re-done inside it, so two writers can never both migrate and a reader never sees a
/// half migration. The lock parameter is only proof that the caller holds it.
pub(crate) fn open_db_writer(db_path: &Path, _writer: &WriterLock) -> Result<Connection, String> {
    let conn = open_db_rw_raw(db_path)?;
    init_schema(&conn)?;
    Ok(conn)
}

/// Read-write connection for best-effort side writes from read paths (the query-embedding
/// cache): waits at most 100 ms on a busy database instead of [`DB_BUSY_TIMEOUT`], so a
/// watcher transaction never stalls `search`, the MCP tools or the recall hook. Never
/// creates or migrates a store.
pub(crate) fn open_db_side_writer(db_path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("failed opening database: {}", e))?;
    conn.busy_timeout(Duration::from_millis(100))
        .map_err(|e| format!("failed setting db busy timeout: {}", e))?;
    Ok(conn)
}

pub(crate) fn open_db_rw_raw(db_path: &Path) -> Result<Connection, String> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating db dir: {}", e))?;
    }
    let conn = Connection::open(db_path).map_err(|e| format!("failed opening database: {}", e))?;
    // Wait briefly on SQLITE_BUSY so a concurrent watcher write never fails readers outright.
    conn.busy_timeout(DB_BUSY_TIMEOUT)
        .map_err(|e| format!("failed setting db busy timeout: {}", e))?;
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
"#,
    )
    .map_err(|e| format!("failed setting db pragmas: {}", e))?;
    Ok(conn)
}

pub(crate) fn db_has_table(conn: &Connection, table: &str) -> Result<bool, String> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
        params![table],
        |_| Ok(()),
    )
    .optional()
    .map(|row| row.is_some())
    .map_err(|e| format!("failed inspecting database schema: {}", e))
}

/// Name of the advisory lock file in the data directory that serialises writers.
pub(crate) const WRITER_LOCK_FILE: &str = "index.lock";
/// Prefix of the error every writer entry point returns while another writer holds the lock.
pub(crate) const INDEX_BUSY_PREFIX: &str = "index busy: another retrivio writer is running";

/// The one writer at a time: an OS advisory lock (`flock`) on `<data_dir>/index.lock`, taken
/// by index, refresh, reembed, prune, the watcher and the API/MCP index tools before the
/// database is opened for writing and before any schema migration. The kernel drops the lock
/// when the holding process exits, however it exits, so there is no stale sentinel to clean
/// up. The holder's pid is written into the file for the "busy" message only.
#[derive(Debug)]
pub(crate) struct WriterLock {
    file: fs::File,
}

impl WriterLock {
    /// Try once. `Err` is the busy message (see [`is_index_busy_error`]) when another process
    /// holds the lock, or an I/O error.
    pub(crate) fn try_acquire(data_dir: &Path) -> Result<WriterLock, String> {
        fs::create_dir_all(data_dir)
            .map_err(|e| format!("failed creating {}: {}", data_dir.display(), e))?;
        let path = data_dir.join(WRITER_LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| format!("failed opening writer lock {}: {}", path.display(), e))?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK)
                || err.kind() == std::io::ErrorKind::WouldBlock
            {
                let holder = fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok());
                return Err(match holder {
                    Some(pid) => format!("{} (pid {})", INDEX_BUSY_PREFIX, pid),
                    None => format!("{} (pid unknown)", INDEX_BUSY_PREFIX),
                });
            }
            return Err(format!("failed locking {}: {}", path.display(), err));
        }
        // Our pid, for the busy message of the next contender. Best effort: the lock itself
        // is the kernel's, not this file's content.
        let _ = file.set_len(0);
        let _ = (&file).write_all(format!("{}\n", process::id()).as_bytes());
        let _ = (&file).flush();
        Ok(WriterLock { file })
    }

    /// Wait for the lock, retrying every `poll`, calling `on_wait` once with the busy message.
    /// The watcher uses this: it would rather wait for a manual `index` to finish than fail.
    fn acquire_waiting(
        data_dir: &Path,
        poll: Duration,
        mut on_wait: impl FnMut(&str),
    ) -> Result<WriterLock, String> {
        let mut reported = false;
        loop {
            match Self::try_acquire(data_dir) {
                Ok(lock) => return Ok(lock),
                Err(e) if is_index_busy_error(&e) => {
                    if !reported {
                        on_wait(&e);
                        reported = true;
                    }
                    thread::sleep(poll);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // Closing the descriptor releases the lock as well; the explicit unlock keeps the
        // order obvious. The file stays: deleting it would let a later opener lock a
        // different inode than a concurrent holder.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod writer_lock_tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    use std::{env, fs, process, thread};

    fn lock_dir(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("lock-{}-{}", name, process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("lock dir");
        dir
    }

    #[test]
    fn a_second_writer_is_told_who_holds_the_lock_and_gets_it_after_release() {
        let dir = lock_dir("contend");
        let first = WriterLock::try_acquire(&dir).expect("first lock");
        let err = WriterLock::try_acquire(&dir).expect_err("second must be busy");
        assert!(is_index_busy_error(&err), "{}", err);
        assert_eq!(
            err,
            format!(
                "index busy: another retrivio writer is running (pid {})",
                process::id()
            )
        );
        drop(first);
        WriterLock::try_acquire(&dir).expect("lock is free after drop");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn acquire_waiting_reports_once_and_returns_when_the_holder_releases() {
        let dir = lock_dir("waiting");
        let held = WriterLock::try_acquire(&dir).expect("hold");
        let dir2 = dir.clone();
        let releaser = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            drop(held);
        });
        let mut reports = 0usize;
        let got = WriterLock::acquire_waiting(&dir2, Duration::from_millis(50), |msg| {
            reports += 1;
            assert!(is_index_busy_error(msg), "{}", msg);
        })
        .expect("acquired after release");
        releaser.join().expect("releaser thread");
        assert_eq!(reports, 1);
        drop(got);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Helper body, not a test of its own: when `RETRIVIO_TEST_HOLD_LOCK` names a directory,
    /// take its writer lock, drop a marker file and sleep until killed. The SIGKILL test runs
    /// this very test binary with that variable set.
    #[test]
    fn hold_writer_lock_helper() {
        let Ok(dir) = env::var("RETRIVIO_TEST_HOLD_LOCK") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let _lock = WriterLock::try_acquire(&dir).expect("helper lock");
        fs::write(dir.join("held"), b"1").expect("marker");
        thread::sleep(Duration::from_secs(120));
    }

    #[test]
    fn the_lock_is_released_when_the_holder_dies_from_sigkill() {
        let dir = lock_dir("sigkill");
        let exe = env::current_exe().expect("test exe");
        let mut child = Command::new(&exe)
            .args([
                "db::writer_lock_tests::hold_writer_lock_helper",
                "--exact",
                "--nocapture",
            ])
            .env("RETRIVIO_TEST_HOLD_LOCK", &dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn helper");
        let deadline = Instant::now() + Duration::from_secs(30);
        while !dir.join("held").exists() {
            assert!(Instant::now() < deadline, "helper never took the lock");
            thread::sleep(Duration::from_millis(20));
        }
        let err = WriterLock::try_acquire(&dir).expect_err("busy while the helper lives");
        assert_eq!(
            err,
            format!(
                "index busy: another retrivio writer is running (pid {})",
                child.id()
            )
        );
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGKILL);
        }
        let _ = child.wait();
        // No cleanup ran in the child (SIGKILL), yet the kernel released the flock. The stale
        // pid in the file is overwritten by the new holder.
        let ours = WriterLock::try_acquire(&dir).expect("lock released after SIGKILL");
        let recorded = fs::read_to_string(dir.join(WRITER_LOCK_FILE)).expect("lock file");
        assert_eq!(recorded.trim(), process::id().to_string());
        drop(ours);
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod migration_discipline_tests {
    use super::*;
    use crate::config::ConfigValues;
    use crate::embed::{disk_cache_lookup, disk_cache_store, query_cache_key};
    use crate::index::embed_input_hash;
    use crate::rank::lexical_file_candidates;
    use rusqlite::Connection;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::{fs, process};

    /// Columns 0.2 adds to a 0.1.x store. Dropping them from a freshly created schema yields
    /// the 0.1.x shape, which is what an un-upgraded store looks like to this binary.
    const ADDED_COLUMNS: &[(&str, &str)] = &[
        ("projects", "scan_signature"),
        ("projects", "index_in_progress"),
        ("project_vectors", "normalized"),
        ("project_vectors", "pipeline_version"),
        ("project_chunk_vectors", "embed_input_hash"),
        ("project_chunk_vectors", "normalized"),
        ("project_chunk_vectors", "pipeline_version"),
    ];

    fn old_shape_store(name: &str) -> (PathBuf, PathBuf) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("migrate-{}-{}", name, process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("store dir");
        let db = dir.join("retrivio.db");
        {
            let conn = open_db_rw(&db).expect("create fresh store");
            for (table, column) in ADDED_COLUMNS {
                conn.execute(&format!("ALTER TABLE {} DROP COLUMN {}", table, column), [])
                    .expect("drop added column");
            }
            conn.execute_batch(
                r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/alpha', 'alpha', 'alpha notes', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (10, 1, '/p/alpha/notes.md', 'notes.md', 0, 0, 3, 'h10', 'orion runbook bedrock region', 0);
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector)
VALUES (10, 'bedrock:amazon.titan-embed-text-v2:0', 1, 1.0, x'00000000');
"#,
            )
            .expect("seed 0.1.x rows");
        }
        (dir, db)
    }

    fn has_added_columns(conn: &Connection) -> Vec<bool> {
        ADDED_COLUMNS
            .iter()
            .map(|(t, c)| table_has_column(conn, t, c).expect("table_info"))
            .collect()
    }

    #[test]
    fn read_paths_never_migrate_and_still_answer_then_a_writer_migrates_once() {
        let (dir, db) = old_shape_store("read-paths");
        let none = vec![false; ADDED_COLUMNS.len()];
        let all = vec![true; ADDED_COLUMNS.len()];
        let cfg = ConfigValues::from_map(HashMap::new());

        // The plain read-write open (roots, config, MCP tools, the query cache write) and the
        // read-only open (recall, search) leave the schema exactly as it was ...
        {
            let conn = open_db_rw(&db).expect("open rw");
            assert_eq!(has_added_columns(&conn), none);
            let hits = lexical_file_candidates(&conn, &cfg, &["runbook".to_string()], 5);
            assert_eq!(hits.len(), 1, "search answers on the old shape");
            assert_eq!(hits[0].doc_rel_path, "notes.md");
        }
        {
            let conn = open_db_read_only(&db).expect("open ro");
            assert_eq!(has_added_columns(&conn), none);
            let hits = lexical_file_candidates(&conn, &cfg, &["bedrock".to_string()], 5);
            assert_eq!(hits.len(), 1);
        }
        assert!(ensure_db_schema(&db).is_ok());
        assert_eq!(
            has_added_columns(&open_db_read_only(&db).unwrap()),
            none,
            "ensure_db_schema is not a migration either"
        );

        // ... and the writer open, under the lock, migrates in one go.
        let writer = WriterLock::try_acquire(&dir).expect("lock");
        {
            let conn = open_db_writer(&db, &writer).expect("open writer");
            assert_eq!(has_added_columns(&conn), all);
            assert_eq!(
                conn.query_row(
                    "SELECT embed_input_hash FROM project_chunk_vectors WHERE chunk_id = 10",
                    [],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
                embed_input_hash("orion runbook bedrock region"),
                "the backfill ran with the migration"
            );
        }
        // Idempotent for the next writer, and the read paths see the same rows as before.
        let conn = open_db_writer(&db, &writer).expect("second writer open");
        assert_eq!(has_added_columns(&conn), all);
        let hits = lexical_file_candidates(&conn, &cfg, &["runbook".to_string()], 5);
        assert_eq!(hits.len(), 1);
        drop(conn);
        drop(writer);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The query-embedding cache holds hashes and vectors, never query text; a store with the
    /// pre-0.2.1 table (text key) misses on read, refuses the side write, and is recreated by
    /// the next writer.
    #[test]
    fn query_cache_keys_on_a_hash_and_the_old_table_shape_is_a_miss_until_a_writer_recreates_it() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("query-cache-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let db = dir.join("retrivio.db");
        let rw = open_db_rw(&db).expect("fresh store");
        assert!(table_has_column(&rw, "query_embed_cache", "query_hash").unwrap());
        assert!(!table_has_column(&rw, "query_embed_cache", "query_normalized").unwrap());

        // Put the old shape back, with a row holding query text, as a 0.2.0 store would.
        rw.execute_batch(
            r#"
DROP TABLE query_embed_cache;
CREATE TABLE query_embed_cache (
    query_normalized TEXT NOT NULL,
    model_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    cached_at REAL NOT NULL,
    PRIMARY KEY(query_normalized, model_key)
);
INSERT INTO query_embed_cache VALUES ('what did acme say', 'm', x'0000803f', 1.0);
"#,
        )
        .unwrap();
        let ro = open_db_read_only(&db).expect("read-only");
        // Read paths tolerate the old shape as a miss; the side writer's store fails and is
        // ignored by its caller.
        assert!(disk_cache_lookup(&ro, "what did acme say", "m").is_err());
        let side = open_db_side_writer(&db).expect("side writer");
        assert!(disk_cache_store(&side, "what did acme say", "m", &[1.0]).is_err());
        drop(side);

        // A writer recreates the table in the new shape; the old rows (query text) are gone.
        init_schema(&rw).expect("writer schema pass");
        assert!(table_has_column(&rw, "query_embed_cache", "query_hash").unwrap());
        assert!(!table_has_column(&rw, "query_embed_cache", "query_normalized").unwrap());
        let rows: i64 = rw
            .query_row("SELECT COUNT(*) FROM query_embed_cache", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);

        // Store and look up through the hash: case and whitespace are normalised by the
        // caller, the key is the same for both spellings, and no column carries the text.
        let side = open_db_side_writer(&db).expect("side writer");
        disk_cache_store(&side, "what did acme say", "m", &[1.0, 2.0]).unwrap();
        assert_eq!(
            disk_cache_lookup(&ro, "what did acme say", "m").unwrap(),
            vec![1.0, 2.0]
        );
        assert!(disk_cache_lookup(&ro, "what did acme say", "other-model").is_err());
        assert_eq!(
            query_cache_key("m", "what did acme say"),
            query_cache_key("m", &"  What did ACME say ".trim().to_ascii_lowercase())
        );
        assert_ne!(
            query_cache_key("m", "what did acme say"),
            query_cache_key("m2", "what did acme say")
        );
        let dump: String = {
            let mut stmt = rw
                .prepare("SELECT query_hash, model_key, cached_at FROM query_embed_cache")
                .unwrap();
            let rows = stmt
                .query_map([], |r| {
                    Ok(format!(
                        "{} {} {}",
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?
                    ))
                })
                .unwrap();
            rows.map(|r| r.unwrap()).collect::<Vec<_>>().join("\n")
        };
        assert!(!dump.contains("acme"), "{}", dump);
        assert!(dump.starts_with(&query_cache_key("m", "what did acme say")));
        drop(ro);
        drop(side);
        drop(rw);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Readers may ask whether a re-embed is due without writing: a model key that differs
    /// from the stored one is reported on a read-only connection and the state is unchanged;
    /// a writer normalises the stored flag and reason.
    #[test]
    fn reembed_reason_is_computed_read_only_and_persisted_by_writers() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("reembed-ro-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let db = dir.join("retrivio.db");
        let rw = open_db_rw(&db).expect("fresh store");
        app_state_set(&rw, APP_STATE_ACTIVE_MODEL_KEY, "ollama:old-model").unwrap();
        app_state_set(&rw, APP_STATE_REEMBED_REQUIRED, "0").unwrap();
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("embed_backend".into(), "ollama".into());
        map.insert("embed_model".into(), "new-model".into());
        let cfg = ConfigValues::from_map(map);

        let ro = open_db_read_only(&db).expect("read-only");
        let reason = reembed_requirement_reason(&ro, &cfg)
            .expect("no write on a read-only connection")
            .expect("a model change is a reason");
        assert!(reason.contains("ollama:old-model") && reason.contains("ollama:new-model"));
        let blocked = ensure_reembed_ready(&ro, &cfg, "search").unwrap_err();
        assert!(blocked.starts_with("search blocked:"), "{}", blocked);
        assert!(!blocked.contains("readonly"), "{}", blocked);
        assert_eq!(
            app_state_get(&ro, APP_STATE_REEMBED_REQUIRED)
                .unwrap()
                .as_deref(),
            Some("0"),
            "the reader changed nothing"
        );
        assert!(app_state_get(&ro, APP_STATE_REEMBED_REASON)
            .unwrap()
            .is_none());

        // A flag without a reason is reported with a generic reason, still without writing.
        app_state_set(&rw, APP_STATE_ACTIVE_MODEL_KEY, "ollama:new-model").unwrap();
        app_state_set(&rw, APP_STATE_REEMBED_REQUIRED, "1").unwrap();
        let generic = reembed_requirement_reason(&ro, &cfg).unwrap().unwrap();
        assert!(generic.contains("migration required"), "{}", generic);
        assert!(app_state_get(&ro, APP_STATE_REEMBED_REASON)
            .unwrap()
            .is_none());

        // The writer persists the normalised state; a second call finds nothing stale.
        app_state_set(&rw, APP_STATE_ACTIVE_MODEL_KEY, "ollama:old-model").unwrap();
        app_state_set(&rw, APP_STATE_REEMBED_REQUIRED, "0").unwrap();
        let persisted = persist_reembed_requirement(&rw, &cfg).unwrap().unwrap();
        assert_eq!(persisted, reason);
        assert_eq!(
            app_state_get(&rw, APP_STATE_REEMBED_REQUIRED)
                .unwrap()
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            app_state_get(&rw, APP_STATE_REEMBED_REASON)
                .unwrap()
                .as_deref(),
            Some(reason.as_str())
        );
        assert!(!reembed_requirement(&rw, &cfg).unwrap().unwrap().1);
        // No requirement at all: same key, flag clear.
        app_state_set(&rw, APP_STATE_ACTIVE_MODEL_KEY, "ollama:new-model").unwrap();
        app_state_set(&rw, APP_STATE_REEMBED_REQUIRED, "0").unwrap();
        assert!(reembed_requirement_reason(&ro, &cfg).unwrap().is_none());
        drop(ro);
        drop(rw);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_brand_new_store_is_created_by_whoever_opens_it_first() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("migrate-fresh-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let db = dir.join("retrivio.db");
        let conn = open_db_rw(&db).expect("fresh store through the non-migrating open");
        assert!(db_has_table(&conn, "projects").unwrap());
        assert!(db_has_table(&conn, "query_embed_cache").unwrap());
        assert!(table_has_column(&conn, "project_chunk_vectors", "embed_input_hash").unwrap());
        drop(conn);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// True for the error [`WriterLock::try_acquire`] returns while another writer runs.
pub(crate) fn is_index_busy_error(e: &str) -> bool {
    e.starts_with(INDEX_BUSY_PREFIX)
}

/// The watcher's lock acquisition: wait for a manual `index`/`prune` to finish instead of
/// failing, and say so once (also under `--quiet`; a long wait is worth one line in the log).
pub(crate) fn acquire_writer_lock_for_watch(cwd: &Path) -> Result<WriterLock, String> {
    WriterLock::acquire_waiting(&data_dir(cwd), Duration::from_secs(2), |msg| {
        println!("[{}] watch: {}; waiting", chrono_like_now(), msg);
    })
}

/// How long a connection waits on a locked database before returning SQLITE_BUSY.
pub(crate) const DB_BUSY_TIMEOUT: Duration = Duration::from_millis(2000);

pub(crate) fn open_db_read_only(db_path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("failed opening database readonly: {}", e))?;
    conn.busy_timeout(DB_BUSY_TIMEOUT)
        .map_err(|e| format!("failed setting db busy timeout: {}", e))?;
    Ok(conn)
}

/// Create the schema and migrate an existing one, inside a single `BEGIN IMMEDIATE`
/// transaction: the write lock is taken up front, every `ensure_*` re-checks the column state
/// inside the transaction, and a failure rolls everything back so the next writer retries.
/// Only [`open_db_writer`] (writers holding the lock) and the creation of a brand-new store
/// reach this; see [`open_db_rw`].
pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch("BEGIN IMMEDIATE;")
        .map_err(|e| format!("failed starting schema transaction: {}", e))?;
    match init_schema_in_tx(conn) {
        Ok(()) => conn
            .execute_batch("COMMIT;")
            .map_err(|e| format!("failed committing schema transaction: {}", e)),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK;");
            Err(e)
        }
    }
}

pub(crate) fn init_schema_in_tx(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_mtime REAL NOT NULL,
    last_indexed REAL NOT NULL,
    scan_signature TEXT NOT NULL DEFAULT '',
    index_in_progress INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_vectors (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL,
    normalized INTEGER NOT NULL DEFAULT 0,
    pipeline_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_chunks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_path TEXT NOT NULL,
    doc_rel_path TEXT NOT NULL,
    doc_mtime REAL NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    updated_at REAL NOT NULL,
    chunk_kind TEXT NOT NULL DEFAULT 'text_window',
    symbol_name TEXT NOT NULL DEFAULT '',
    parent_context TEXT NOT NULL DEFAULT '',
    line_start INTEGER NOT NULL DEFAULT 0,
    line_end INTEGER NOT NULL DEFAULT 0,
    context_header TEXT NOT NULL DEFAULT '',
    UNIQUE(project_id, doc_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_project_chunks_project
    ON project_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_project_chunks_doc
    ON project_chunks(doc_path);

CREATE TABLE IF NOT EXISTS project_chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES project_chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL,
    embed_input_hash TEXT NOT NULL DEFAULT '',
    normalized INTEGER NOT NULL DEFAULT 0,
    pipeline_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS selection_events (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    path TEXT NOT NULL,
    selected_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at REAL NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    exclude_patterns TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS lance_pending (
    chunk_id INTEGER PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_tracked_roots_enabled_path
    ON tracked_roots(enabled, path);

CREATE TABLE IF NOT EXISTS project_edges (
    src_project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(src_project_id, dst, kind)
);

CREATE INDEX IF NOT EXISTS idx_project_edges_dst
    ON project_edges(dst);
CREATE INDEX IF NOT EXISTS idx_project_edges_src_kind
    ON project_edges(src_project_id, kind);

CREATE TABLE IF NOT EXISTS chunk_relation_feedback (
    id INTEGER PRIMARY KEY,
    src_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    dst_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('active', 'suppressed')),
    quality_label TEXT NOT NULL DEFAULT 'unspecified' CHECK(quality_label IN ('unspecified', 'good', 'weak', 'wrong')),
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(src_chunk_id, dst_chunk_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_decision_time
    ON chunk_relation_feedback(src_chunk_id, decision, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_dst
    ON chunk_relation_feedback(dst_chunk_id);

CREATE INDEX IF NOT EXISTS idx_selection_events_path_time
    ON selection_events(path, selected_at DESC);
CREATE INDEX IF NOT EXISTS idx_selection_events_query_time
    ON selection_events(query, selected_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS project_fts USING fts5(
    path,
    title,
    summary,
    content='projects',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS projects_ai AFTER INSERT ON projects BEGIN
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_ad AFTER DELETE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_au AFTER UPDATE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    doc_path,
    doc_rel_path,
    text,
    content='project_chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS project_chunks_ai AFTER INSERT ON project_chunks BEGIN
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_ad AFTER DELETE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_au AFTER UPDATE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;
"#,
    )
    .map_err(|e| {
        format!(
            "failed initializing db schema (ensure sqlite build includes FTS5): {}",
            e
        )
    })?;
    // Persistent query embedding cache (survives process restarts). Keyed by the hash of
    // (model key, normalised query) since 0.2.1; the earlier shape stored the query text
    // itself. It is a cache: an old table is dropped and recreated rather than migrated.
    if db_has_table(conn, "query_embed_cache")?
        && table_has_column(conn, "query_embed_cache", "query_normalized")?
    {
        conn.execute_batch("DROP TABLE query_embed_cache;")
            .map_err(|e| format!("failed dropping the old query_embed_cache table: {}", e))?;
    }
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS query_embed_cache (
    query_hash TEXT NOT NULL PRIMARY KEY,
    model_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    cached_at REAL NOT NULL
);
"#,
    )
    .map_err(|e| format!("failed creating query_embed_cache table: {}", e))?;

    // File-level change detection for incremental indexing
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS project_files (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_mtime REAL NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    last_indexed REAL NOT NULL,
    UNIQUE(project_id, rel_path)
);

CREATE INDEX IF NOT EXISTS idx_project_files_project ON project_files(project_id);
CREATE INDEX IF NOT EXISTS idx_project_files_hash ON project_files(content_hash);
"#,
    )
    .map_err(|e| format!("failed creating project_files table: {}", e))?;

    // Code intelligence: symbol index tables
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_path TEXT NOT NULL,
    doc_rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    signature TEXT NOT NULL DEFAULT '',
    doc_comment TEXT NOT NULL DEFAULT '',
    visibility TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_project ON symbols(project_id);
CREATE INDEX IF NOT EXISTS idx_symbols_doc ON symbols(doc_path);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

-- symbol_chunk_map (0.1.x) was created but never written or read; writers drop it, readers
-- never touch it.
DROP TABLE IF EXISTS symbol_chunk_map;

CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(
    name,
    qualified_name,
    signature,
    doc_comment,
    content='symbols',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbol_fts(rowid, name, qualified_name, signature, doc_comment)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbol_fts(symbol_fts, rowid, name, qualified_name, signature, doc_comment)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbol_fts(symbol_fts, rowid, name, qualified_name, signature, doc_comment)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.doc_comment);
    INSERT INTO symbol_fts(rowid, name, qualified_name, signature, doc_comment)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.doc_comment);
END;
"#,
    )
    .map_err(|e| format!("failed creating symbol index tables: {}", e))?;

    // Import graph tables
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS file_imports (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_doc_path TEXT NOT NULL,
    import_kind TEXT NOT NULL,
    raw_specifier TEXT NOT NULL,
    resolved_path TEXT NOT NULL DEFAULT '',
    imported_names TEXT NOT NULL DEFAULT '',
    line_number INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_imports_project ON file_imports(project_id);
CREATE INDEX IF NOT EXISTS idx_file_imports_source ON file_imports(source_doc_path);

CREATE TABLE IF NOT EXISTS file_dependency_edges (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_doc_path TEXT NOT NULL,
    target_doc_path TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    imported_symbol_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    UNIQUE(project_id, source_doc_path, target_doc_path, edge_kind)
);

CREATE INDEX IF NOT EXISTS idx_file_dep_edges_project ON file_dependency_edges(project_id);
CREATE INDEX IF NOT EXISTS idx_file_dep_edges_source ON file_dependency_edges(source_doc_path);
CREATE INDEX IF NOT EXISTS idx_file_dep_edges_target ON file_dependency_edges(target_doc_path);
"#,
    )
    .map_err(|e| format!("failed creating import graph tables: {}", e))?;

    ensure_vector_model_column(conn)?;
    ensure_relation_feedback_quality_column(conn)?;
    ensure_tracked_roots_exclude_column(conn)?;
    ensure_chunk_code_intel_columns(conn)?;
    ensure_projects_scan_columns(conn)?;
    ensure_chunk_vector_identity_columns(conn)?;
    ensure_project_vector_identity_columns(conn)?;
    Ok(())
}

/// True when `table` already has a column named `column`. Both are code constants.
pub(crate) fn table_has_column(
    conn: &Connection,
    table: &str,
    column: &str,
) -> Result<bool, String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({})", table))
        .map_err(|e| format!("failed inspecting {} schema: {}", table, e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading {} schema: {}", table, e))?;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `projects.scan_signature` (digest of the selected files' path, size and mtime, written only
/// after a complete successful scan) and `projects.index_in_progress` (set while a run writes
/// the project, cleared with the signature). Existing rows get '' / 0 and are re-scanned once.
pub(crate) fn ensure_projects_scan_columns(conn: &Connection) -> Result<(), String> {
    if !table_has_column(conn, "projects", "scan_signature")? {
        conn.execute(
            "ALTER TABLE projects ADD COLUMN scan_signature TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating projects.scan_signature: {}", e))?;
    }
    if !table_has_column(conn, "projects", "index_in_progress")? {
        conn.execute(
            "ALTER TABLE projects ADD COLUMN index_in_progress INTEGER NOT NULL DEFAULT 0",
            [],
        )
        .map_err(|e| format!("failed migrating projects.index_in_progress: {}", e))?;
    }
    Ok(())
}

/// Embedding identity on `project_vectors` (`normalized`, `pipeline_version`; `model` and
/// `dim` already exist), backfilled from the current normalisation setting and pipeline
/// version 1, never by re-embedding.
pub(crate) fn ensure_project_vector_identity_columns(conn: &Connection) -> Result<(), String> {
    if table_has_column(conn, "project_vectors", "pipeline_version")? {
        return Ok(());
    }
    for sql in [
        "ALTER TABLE project_vectors ADD COLUMN normalized INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE project_vectors ADD COLUMN pipeline_version INTEGER NOT NULL DEFAULT 0",
    ] {
        conn.execute(sql, [])
            .map_err(|e| format!("failed migrating project_vectors identity: {}", e))?;
    }
    let bedrock_normalized = bool_env("RETRIVIO_BEDROCK_NORMALIZE", true);
    conn.execute(
        r#"
UPDATE project_vectors
SET normalized = CASE WHEN model LIKE 'bedrock:%' AND ?1 THEN 1 ELSE 0 END,
    pipeline_version = 1
WHERE pipeline_version = 0
"#,
        params![bedrock_normalized as i64],
    )
    .map_err(|e| format!("failed backfilling project_vectors identity: {}", e))?;
    Ok(())
}

/// Embedding identity on `project_chunk_vectors` (`embed_input_hash`, `normalized`,
/// `pipeline_version`; `model` and `dim` already exist). Existing rows are backfilled from the
/// stored chunk text and context header with the current normalisation setting and pipeline
/// version 1, so an upgrade never re-embeds. Column adds and backfill share init_schema's
/// transaction: a failure leaves the old schema in place and the next writer retries.
pub(crate) fn ensure_chunk_vector_identity_columns(conn: &Connection) -> Result<(), String> {
    if table_has_column(conn, "project_chunk_vectors", "embed_input_hash")? {
        return Ok(());
    }
    // Runs inside init_schema's transaction: the column adds and the backfill commit together.
    for sql in [
        "ALTER TABLE project_chunk_vectors ADD COLUMN embed_input_hash TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE project_chunk_vectors ADD COLUMN normalized INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE project_chunk_vectors ADD COLUMN pipeline_version INTEGER NOT NULL DEFAULT 0",
    ] {
        conn.execute(sql, [])
            .map_err(|e| format!("failed migrating project_chunk_vectors identity: {}", e))?;
    }
    backfill_chunk_vector_identity(conn)?;
    Ok(())
}

/// Fill empty identity columns from the stored chunk rows. Pages by chunk id so a large store
/// never holds every chunk text in memory at once. Returns the number of rows updated.
pub(crate) fn backfill_chunk_vector_identity(conn: &Connection) -> Result<usize, String> {
    const PAGE: i64 = 2000;
    let bedrock_normalized = bool_env("RETRIVIO_BEDROCK_NORMALIZE", true);
    let mut select = conn
        .prepare(
            r#"
SELECT v.chunk_id, v.model, c.context_header, c.text
FROM project_chunk_vectors v
JOIN project_chunks c ON c.id = v.chunk_id
WHERE v.embed_input_hash = '' AND v.chunk_id > ?1
ORDER BY v.chunk_id
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing vector identity backfill: {}", e))?;
    let mut update = conn
        .prepare(
            "UPDATE project_chunk_vectors SET embed_input_hash = ?1, normalized = ?2, pipeline_version = ?3 WHERE chunk_id = ?4",
        )
        .map_err(|e| format!("failed preparing vector identity update: {}", e))?;
    let mut last_id = i64::MIN;
    let mut updated = 0usize;
    loop {
        let page: Vec<(i64, String, String, String)> = {
            let rows = select
                .query_map(params![last_id, PAGE], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(|e| format!("failed reading vector identity backfill page: {}", e))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| format!("failed reading backfill row: {}", e))?);
            }
            out
        };
        if page.is_empty() {
            break;
        }
        for (chunk_id, model, header, text) in &page {
            let hash = embed_input_hash(&embed_input_for(header, text));
            let normalized = model.starts_with("bedrock:") && bedrock_normalized;
            update
                .execute(params![hash, normalized as i64, 1i64, chunk_id])
                .map_err(|e| format!("failed backfilling vector identity: {}", e))?;
            updated += 1;
            last_id = *chunk_id;
        }
    }
    Ok(updated)
}

pub(crate) fn ensure_vector_model_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(project_vectors)")
        .map_err(|e| format!("failed inspecting project_vectors schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading project_vectors schema: {}", e))?;
    let mut has_model = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "model" {
            has_model = true;
            break;
        }
    }
    if !has_model {
        conn.execute(
            "ALTER TABLE project_vectors ADD COLUMN model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating project_vectors.model: {}", e))?;
        conn.execute("UPDATE project_vectors SET model = '' WHERE model = ''", [])
            .map_err(|e| format!("failed finalizing project_vectors.model migration: {}", e))?;
    }
    Ok(())
}

pub(crate) fn ensure_relation_feedback_quality_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(chunk_relation_feedback)")
        .map_err(|e| format!("failed inspecting chunk_relation_feedback schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading chunk_relation_feedback schema: {}", e))?;
    let mut has_quality = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "quality_label" {
            has_quality = true;
            break;
        }
    }
    if !has_quality {
        conn.execute(
            "ALTER TABLE chunk_relation_feedback ADD COLUMN quality_label TEXT NOT NULL DEFAULT 'unspecified'",
            [],
        )
        .map_err(|e| format!("failed migrating chunk_relation_feedback.quality_label: {}", e))?;
        conn.execute(
            "UPDATE chunk_relation_feedback SET quality_label = 'unspecified' WHERE quality_label = '' OR quality_label IS NULL",
            [],
        )
        .map_err(|e| {
            format!(
                "failed finalizing chunk_relation_feedback.quality_label migration: {}",
                e
            )
        })?;
    }
    conn.execute_batch(
        r#"
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_quality
    ON chunk_relation_feedback(src_chunk_id, quality_label, updated_at DESC);
"#,
    )
    .map_err(|e| {
        format!(
            "failed ensuring chunk_relation_feedback quality index: {}",
            e
        )
    })?;
    Ok(())
}

pub(crate) fn ensure_tracked_roots_exclude_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(tracked_roots)")
        .map_err(|e| format!("failed inspecting tracked_roots schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading tracked_roots schema: {}", e))?;
    let mut has_exclude = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "exclude_patterns" {
            has_exclude = true;
            break;
        }
    }
    if !has_exclude {
        conn.execute(
            "ALTER TABLE tracked_roots ADD COLUMN exclude_patterns TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating tracked_roots.exclude_patterns: {}", e))?;
    }
    Ok(())
}

/// Migrate project_chunks table to include code intelligence columns.
/// Checks for `chunk_kind` as the sentinel — if missing, adds all 6 new columns.
pub(crate) fn ensure_chunk_code_intel_columns(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(project_chunks)")
        .map_err(|e| format!("failed inspecting project_chunks schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading project_chunks schema: {}", e))?;
    let mut has_chunk_kind = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "chunk_kind" {
            has_chunk_kind = true;
            break;
        }
    }
    if !has_chunk_kind {
        let alters = [
            "ALTER TABLE project_chunks ADD COLUMN chunk_kind TEXT NOT NULL DEFAULT 'text_window'",
            "ALTER TABLE project_chunks ADD COLUMN symbol_name TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE project_chunks ADD COLUMN parent_context TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE project_chunks ADD COLUMN line_start INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE project_chunks ADD COLUMN line_end INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE project_chunks ADD COLUMN context_header TEXT NOT NULL DEFAULT ''",
        ];
        for sql in &alters {
            conn.execute(sql, []).map_err(|e| {
                format!("failed migrating project_chunks code-intel columns: {}", e)
            })?;
        }
    }
    Ok(())
}

pub(crate) struct TrackedRoot {
    pub(crate) path: PathBuf,
    pub(crate) exclude_patterns: Vec<String>,
}

impl TrackedRoot {
    /// Returns absolute paths that should be excluded during traversal.
    pub(crate) fn absolute_excludes(&self) -> std::collections::HashSet<PathBuf> {
        self.exclude_patterns
            .iter()
            .map(|p| normalize_path(&self.path.join(p).to_string_lossy()))
            .collect()
    }
}

pub(crate) fn ensure_tracked_root_conn(
    conn: &Connection,
    path: &Path,
    added_at: f64,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO tracked_roots(path, added_at, enabled)
VALUES (?1, ?2, 1)
ON CONFLICT(path) DO UPDATE SET enabled = 1
"#,
        params![path.to_string_lossy().to_string(), added_at],
    )
    .map_err(|e| format!("failed ensuring tracked root: {}", e))?;
    Ok(())
}

pub(crate) fn list_tracked_roots_conn(conn: &Connection) -> Result<Vec<PathBuf>, String> {
    Ok(list_tracked_roots_full_conn(conn)?
        .into_iter()
        .map(|r| r.path)
        .collect())
}

pub(crate) fn list_tracked_roots_full_conn(conn: &Connection) -> Result<Vec<TrackedRoot>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, exclude_patterns
FROM tracked_roots
WHERE enabled = 1
ORDER BY path
"#,
        )
        .map_err(|e| format!("failed preparing tracked roots query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| format!("failed querying tracked roots: {}", e))?;

    let mut out = Vec::new();
    for row in rows {
        let (path_str, excl_str) =
            row.map_err(|e| format!("failed reading tracked root row: {}", e))?;
        let exclude_patterns: Vec<String> = excl_str
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        out.push(TrackedRoot {
            path: PathBuf::from(path_str),
            exclude_patterns,
        });
    }
    Ok(out)
}

pub(crate) fn tracked_roots_count(db_path: &Path) -> Option<i64> {
    if !db_path.exists() {
        return Some(0);
    }
    let conn = open_db_read_only(db_path).ok()?;
    conn.query_row(
        "SELECT COUNT(*) FROM tracked_roots WHERE enabled = 1",
        [],
        |row| row.get::<_, i64>(0),
    )
    .ok()
}

pub(crate) fn database_ready(db_path: &Path) -> bool {
    if !db_path.exists() {
        return false;
    }
    match open_db_read_only(db_path) {
        Ok(conn) => conn
            .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
            .map(|_| true)
            .unwrap_or(false),
        Err(_) => false,
    }
}
