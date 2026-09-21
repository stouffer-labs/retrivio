//! Pruning: the prune command body, stale project and row removal, and the keep-set bookkeeping.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use std::{fs, thread};

use rusqlite::{params, Connection};

use crate::config::{data_dir, db_path, ConfigValues, ScanSettings};
use crate::db::{
    get_or_open_lance, lance_store_is_open, open_db_writer, persist_reembed_requirement,
    vector_dim_from_sqlite, with_lance_store, TrackedRoot, WriterLock,
};
use crate::embed::model_key_for_cfg;
use crate::index::{
    get_project_by_path, lance_delete_marked, lance_mark_dirty, lance_reconcile_preview,
    repair_lance_from_sqlite, ProjectChunk, ProjectCorpus,
};
use crate::lance_store;
use crate::scan::{
    collect_project_corpus, discover_projects_full, project_excludes_for_path, project_scan,
    resolve_roots, FileManifest, ScanCaps,
};
use crate::util::{format_bytes, format_duration_ms, is_under_any};

/// `retrivio prune`: re-collect every in-scope project's corpus with the same code path
/// indexing uses (excludes, skip dirs, indexable suffixes, size limits, caps, chunking, but
/// no embedding) and delete the rows the corpus no longer covers. Then drop project rows
/// whose directory vanished or is no longer discovered, and LanceDB rows with no sqlite
/// vector. Real runs then compact LanceDB and drop its old versions (unless `compact` is
/// off), which is what actually frees disk space. With `dry_run`, only report.
pub(crate) fn run_prune(
    cwd: &Path,
    cfg: &ConfigValues,
    scope: Option<Vec<PathBuf>>,
    dry_run: bool,
    compact: bool,
) -> Result<(), String> {
    let t_start = Instant::now();
    let settings = ScanSettings::from_cfg(cfg);
    let dbp = db_path(cwd);
    if !dbp.exists() {
        return Err("no index database found; run `retrivio index` first".to_string());
    }
    let writer = WriterLock::try_acquire(&data_dir(cwd))?;
    let conn = open_db_writer(&dbp, &writer)?;
    let roots = resolve_roots(&conn, cfg, None)?;
    if roots.is_empty() {
        return Err("No tracked roots configured. Add one with `retrivio add <path>`.".to_string());
    }
    let scope_set: Option<HashSet<PathBuf>> = scope.map(|v| v.into_iter().collect());
    let in_scope = |p: &Path| scope_set.as_ref().is_none_or(|s| is_under_any(p, s));
    let discovery = discover_projects_full(&roots, &settings);
    let incomplete_roots = discovery.incomplete_roots.clone();
    let discovered_set: HashSet<String> = discovery
        .projects
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let projects: Vec<PathBuf> = discovery
        .projects
        .iter()
        .filter(|p| in_scope(p))
        .cloned()
        .collect();
    let discovery = &discovery;

    let model_key = model_key_for_cfg(cfg);
    let lance_ready = match vector_dim_from_sqlite(&conn, &model_key) {
        // A dry run must not create the LanceDB directory or table.
        Some(_) if dry_run && !data_dir(cwd).join("lance").exists() => false,
        Some(dim) => match get_or_open_lance(cwd, dim) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("warning: LanceDB open failed ({}); pruning sqlite only", e);
                false
            }
        },
        None => false,
    };

    let verb = if dry_run { "would prune" } else { "pruned" };
    println!(
        "prune: {} projects in scope{}",
        projects.len(),
        if dry_run { " (dry run)" } else { "" }
    );

    let max_chars = cfg.max_chars_per_project as usize;
    let caps = ScanCaps::from_cfg(cfg);
    const PRUNE_PARALLELISM: usize = 4;
    let mut total = PruneOutcome::default();
    let mut projects_pruned = 0usize;
    let mut projects_skipped = 0usize;
    let mut lance_deleted = 0usize;
    let mut lance_delete_failed = 0usize;
    for work in projects.chunks(PRUNE_PARALLELISM) {
        let results = Mutex::new(Vec::<(usize, PathBuf, Result<ProjectCorpus, String>)>::new());
        thread::scope(|s| {
            let handles: Vec<_> = work
                .iter()
                .enumerate()
                .map(|(i, dir)| {
                    let results = &results;
                    let roots = &roots;
                    let caps = &caps;
                    let settings = &settings;
                    s.spawn(move || {
                        // An unreadable directory anywhere in the project (unmounted volume,
                        // permissions) must not read as a smaller corpus and wipe rows.
                        let excludes = project_excludes_for_path(dir, roots);
                        let scan =
                            project_scan(dir, &excludes, caps, discovery.is_shallow(dir), settings);
                        let result = if !scan.complete() {
                            Err(format!(
                                "{}: {} directory entries unreadable",
                                dir.display(),
                                scan.listing.unreadable
                            ))
                        } else {
                            Ok(collect_project_corpus(
                                dir,
                                &scan,
                                caps,
                                max_chars,
                                &FileManifest::new(),
                                true,
                                settings,
                            ))
                        };
                        if let Ok(mut v) = results.lock() {
                            v.push((i, dir.clone(), result));
                        }
                    })
                })
                .collect();
            for h in handles {
                let _ = h.join();
            }
        });
        let mut corpora = results.into_inner().unwrap_or_default();
        corpora.sort_by_key(|(i, _, _)| *i);
        for (_, dir, corpus_result) in corpora {
            let corpus = match corpus_result {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  skip (nothing pruned): {}", e);
                    projects_skipped += 1;
                    continue;
                }
            };
            let path_str = corpus.doc.path.to_string_lossy().to_string();
            // Never indexed: nothing to prune.
            let Some(row) = get_project_by_path(&conn, &path_str)? else {
                continue;
            };
            let keep = PruneKeepSet::from_corpus(&corpus);
            let outcome = prune_stale_project_rows(&conn, row.id, &keep, dry_run)?;
            if outcome.is_empty() {
                continue;
            }
            projects_pruned += 1;
            let name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("project");
            let mut extras: Vec<String> = Vec::new();
            if outcome.manifest_rows > 0 {
                extras.push(format!("{} manifest", outcome.manifest_rows));
            }
            if outcome.symbol_rows > 0 {
                extras.push(format!("{} symbols", outcome.symbol_rows));
            }
            if outcome.import_rows > 0 {
                extras.push(format!("{} imports", outcome.import_rows));
            }
            if outcome.edge_rows > 0 {
                extras.push(format!("{} edges", outcome.edge_rows));
            }
            println!(
                "  {} {}: {} chunks from {} files{}",
                verb,
                name,
                outcome.chunks,
                outcome.files,
                if extras.is_empty() {
                    String::new()
                } else {
                    format!(" (+ {})", extras.join(", "))
                }
            );
            if !dry_run && lance_ready && !outcome.chunk_ids.is_empty() {
                match lance_delete_marked(&conn, &outcome.chunk_ids) {
                    Ok(n) => lance_deleted += n,
                    Err(e) => {
                        lance_delete_failed += outcome.chunk_ids.len();
                        eprintln!("warning: LanceDB delete failed for {}: {}", name, e);
                    }
                }
            }
            total.absorb(outcome);
        }
    }

    // Projects whose directory vanished, is now excluded, or whose root is no longer tracked.
    // Projects under a tracked root that is not (fully) readable right now are left alone.
    let mut unavailable_roots = unavailable_roots(&roots);
    unavailable_roots.extend(incomplete_roots);
    for root in &unavailable_roots {
        println!(
            "  note: tracked root {} is not readable now; its projects are left alone",
            root.display()
        );
    }
    let mut stale_projects = 0usize;
    let mut stale_project_chunks = 0usize;
    let project_rows: Vec<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, path FROM projects ORDER BY path")
            .map_err(|e| format!("failed preparing project list query: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("failed listing projects: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| format!("failed reading project row: {}", e))?);
        }
        out
    };
    for (id, path) in project_rows {
        if discovered_set.contains(&path)
            || !in_scope(Path::new(&path))
            || is_under_any(Path::new(&path), &unavailable_roots)
        {
            continue;
        }
        let ids = project_chunk_ids(&conn, id)?;
        let reason = if Path::new(&path).is_dir() {
            "excluded or root no longer tracked"
        } else {
            "directory missing"
        };
        println!(
            "  {} project {} ({}; {} chunks)",
            if dry_run { "would remove" } else { "removed" },
            path,
            reason,
            ids.len()
        );
        if !dry_run {
            conn.execute("DELETE FROM projects WHERE id = ?1", params![id])
                .map_err(|e| format!("failed deleting stale project row: {}", e))?;
            if lance_ready && !ids.is_empty() {
                match lance_delete_marked(&conn, &ids) {
                    Ok(n) => lance_deleted += n,
                    Err(e) => {
                        lance_delete_failed += ids.len();
                        eprintln!("warning: LanceDB delete failed for {}: {}", path, e);
                    }
                }
            }
        }
        stale_projects += 1;
        stale_project_chunks += ids.len();
    }

    // Reconcile LanceDB with sqlite: rows with no sqlite vector are removed, rows sqlite has
    // a vector for and LanceDB lacks are rebuilt from the stored blobs (no embedding).
    let mut lance_orphans = 0usize;
    let mut lance_rebuilt = 0usize;
    let mut lance_rows_after: Option<usize> = None;
    // While a re-embed is pending (model changed), sqlite may hold vectors of two models under
    // the same chunk ids; reconciling against the new model would drop the old rows LanceDB
    // still serves. `reembed` rebuilds LanceDB wholesale when it completes.
    let reembed_pending = persist_reembed_requirement(&conn, cfg)?.is_some();
    if reembed_pending {
        println!("  note: a re-embed is pending; LanceDB reconciliation skipped until `retrivio reembed` completes");
    }
    if lance_ready && !reembed_pending {
        if dry_run {
            match lance_reconcile_preview(&conn, &model_key) {
                Ok((missing, orphans)) => {
                    lance_orphans = orphans;
                    lance_rebuilt = missing;
                    if orphans > 0 || missing > 0 {
                        println!(
                            "  would remove {} LanceDB vectors with no sqlite chunk and rebuild {} missing from sqlite",
                            orphans, missing
                        );
                    }
                }
                Err(e) => eprintln!("warning: LanceDB reconcile scan failed: {}", e),
            }
        } else {
            match repair_lance_from_sqlite(&conn, &model_key, true) {
                Ok(report) => {
                    lance_orphans = report.orphans_removed;
                    lance_rebuilt = report.rebuilt;
                    lance_deleted += report.orphans_removed;
                    if report.orphans_removed > 0 || report.rebuilt > 0 {
                        println!(
                            "  removed {} LanceDB vectors with no sqlite chunk; rebuilt {} missing from sqlite",
                            report.orphans_removed, report.rebuilt
                        );
                    }
                }
                Err(e) => eprintln!("warning: LanceDB repair failed: {}", e),
            }
        }
        lance_rows_after = with_lance_store(|store| lance_store::count(store)).ok();
    }
    if !dry_run {
        conn.execute_batch("PRAGMA optimize;")
            .map_err(|e| format!("database optimize failed: {}", e))?;
    }
    let sqlite_chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM project_chunks", [], |row| row.get(0))
        .map_err(|e| format!("failed counting chunks: {}", e))?;

    println!(
        "prune summary{}:",
        if dry_run {
            " (dry run, nothing written)"
        } else {
            ""
        }
    );
    println!("  projects scanned: {}", projects.len());
    println!("  projects with stale rows: {}", projects_pruned);
    if projects_skipped > 0 {
        println!("  projects skipped (unreadable): {}", projects_skipped);
    }
    println!(
        "  chunks {}: {} (from {} files)",
        verb, total.chunks, total.files
    );
    println!(
        "  manifest rows: {}, symbol rows: {}, import rows: {}, dependency edges: {}",
        total.manifest_rows, total.symbol_rows, total.import_rows, total.edge_rows
    );
    println!(
        "  stale projects {}: {} ({} chunks)",
        if dry_run { "to remove" } else { "removed" },
        stale_projects,
        stale_project_chunks
    );
    if lance_ready {
        if dry_run {
            println!(
                "  lancedb vectors to delete: {} from stale rows + {} orphans",
                total.chunks + stale_project_chunks,
                lance_orphans
            );
        } else {
            println!(
                "  lancedb vectors deleted: {} ({} from stale rows, {} orphans{})",
                lance_deleted,
                total.chunks + stale_project_chunks,
                lance_orphans,
                if lance_delete_failed > 0 {
                    format!(", {} failed", lance_delete_failed)
                } else {
                    String::new()
                }
            );
        }
        println!(
            "  lancedb rows {} from sqlite: {}",
            if dry_run { "to rebuild" } else { "rebuilt" },
            lance_rebuilt
        );
        if let Some(n) = lance_rows_after {
            println!("  lancedb rows now: {}", n);
        }
        // Deleted rows are tombstones and every write is a new version kept on disk, so the
        // directory only shrinks once fragments are rewritten and old versions dropped.
        let lance_dir = data_dir(cwd).join("lance");
        let size_before = lance_store::dir_size_bytes(&lance_dir);
        if dry_run {
            println!(
                "  lancedb on disk: {} (a real run compacts it afterwards)",
                format_bytes(size_before)
            );
        } else if !compact {
            println!(
                "  lancedb on disk: {} (compaction skipped: --no-compact)",
                format_bytes(size_before)
            );
        } else {
            let t_compact = Instant::now();
            let versions_before = lance_store::version_count(&lance_dir);
            match with_lance_store(|store| {
                lance_store::optimize(store, cfg.lance_version_grace_secs.max(0) as u64)
            }) {
                Ok(report) => {
                    let size_after = lance_store::dir_size_bytes(&lance_dir);
                    println!(
                        "  lancedb compacted: {} -> {} on disk; versions {} -> {}; rewrote {} fragments into {}, dropped {} old versions ({})",
                        format_bytes(size_before),
                        format_bytes(size_after),
                        versions_before,
                        lance_store::version_count(&lance_dir),
                        report.fragments_removed,
                        report.fragments_added,
                        report.old_versions,
                        format_duration_ms(t_compact.elapsed().as_millis() as u64)
                    );
                }
                Err(e) => eprintln!("warning: LanceDB compaction failed: {}", e),
            }
        }
    } else {
        println!(
            "  lancedb: skipped (no vectors for model {} yet)",
            model_key
        );
    }
    println!("  sqlite chunks now: {}", sqlite_chunks);
    println!(
        "  elapsed: {}",
        format_duration_ms(t_start.elapsed().as_millis() as u64)
    );
    Ok(())
}

/// Tracked roots whose directory cannot be listed right now (unmounted volume, permissions,
/// gone). Their projects are never removed: index and prune both leave them alone.
pub(crate) fn unavailable_roots(roots: &[TrackedRoot]) -> HashSet<PathBuf> {
    roots
        .iter()
        .filter(|r| fs::read_dir(&r.path).is_err())
        .map(|r| r.path.clone())
        .collect()
}

/// Delete every project row whose path is not in `keep_paths`, except rows under a protected
/// (unavailable) root. Returns (projects removed, chunks removed with them).
pub(crate) fn remove_projects_not_in(
    conn: &Connection,
    keep_paths: &[String],
    protected_roots: &HashSet<PathBuf>,
) -> Result<(i64, i64), String> {
    let keep_set: HashSet<String> = keep_paths.iter().cloned().collect();
    let delete_ids: Vec<i64> = {
        let mut stmt = conn
            .prepare("SELECT id, path FROM projects ORDER BY path")
            .map_err(|e| format!("failed preparing project list query: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("failed listing existing projects: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, path) =
                row.map_err(|e| format!("failed reading existing project row: {}", e))?;
            if keep_set.contains(&path) || is_under_any(Path::new(&path), protected_roots) {
                continue;
            }
            out.push(id);
        }
        out
    };
    let mut removed = 0i64;
    let mut stale_chunk_ids: Vec<i64> = Vec::new();
    for id in &delete_ids {
        // sqlite cascades the project row to its chunks and vectors; LanceDB needs the ids.
        stale_chunk_ids.extend(project_chunk_ids(conn, *id)?);
    }
    if !delete_ids.is_empty() {
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("failed starting project removal transaction: {}", e))?;
        if !stale_chunk_ids.is_empty() {
            lance_mark_dirty(&tx)?;
        }
        for id in &delete_ids {
            removed += tx
                .execute("DELETE FROM projects WHERE id = ?1", params![id])
                .map_err(|e| format!("failed deleting stale project row: {}", e))?
                as i64;
        }
        tx.commit()
            .map_err(|e| format!("failed committing project removal: {}", e))?;
    }
    if !stale_chunk_ids.is_empty() && lance_store_is_open() {
        if let Err(e) = lance_delete_marked(conn, &stale_chunk_ids) {
            eprintln!(
                "warning: LanceDB delete of {} vectors from removed projects failed ({}); the next index run repairs it",
                stale_chunk_ids.len(),
                e
            );
        }
    }
    Ok((removed, stale_chunk_ids.len() as i64))
}

// ── Stale-row pruning ─────────────────────────────────────────────────────────
//
// Re-indexing upserts `project_chunks` rows keyed by (project_id, doc_path, chunk_index).
// Rows for files that left the corpus (deleted, newly excluded, under a `skip_dir_names`
// directory, past the per-project caps) or chunk indices past the end of a file that shrank
// would otherwise live forever. These helpers delete them from sqlite (vectors, relation
// feedback and symbol maps cascade via FK) and hand back the chunk ids so callers can drop
// the LanceDB vectors as well. Pruning only runs on a project whose corpus was actually
// collected; incremental runs that skip an unchanged project never reach it.

/// The rows a freshly collected corpus says should survive a prune pass.
#[derive(Debug, Default, Clone)]
pub(crate) struct PruneKeepSet {
    /// doc_path -> chunk indices present in the corpus (re-chunked files).
    by_doc: HashMap<String, HashSet<i64>>,
    /// doc_paths kept whole: files the scan found unchanged, whose stored chunks all survive.
    whole_docs: HashSet<String>,
    /// doc_rel_path values present in the corpus (the `project_files` manifest key).
    pub(crate) rel_paths: HashSet<String>,
}

impl PruneKeepSet {
    /// Keep exactly these chunks (a fully re-chunked corpus, as `prune` collects it).
    fn from_chunks(chunks: &[ProjectChunk]) -> Self {
        let mut keep = PruneKeepSet::default();
        for chunk in chunks {
            keep.by_doc
                .entry(chunk.doc_path.clone())
                .or_default()
                .insert(chunk.chunk_index);
            keep.rel_paths.insert(chunk.doc_rel_path.clone());
        }
        keep
    }

    /// The full keep set of a scan: every chunk of the re-chunked files plus every stored
    /// chunk of the unchanged ones. A re-chunked file that produced no chunks still keeps its
    /// manifest row and per-file rows.
    pub(crate) fn from_corpus(corpus: &ProjectCorpus) -> Self {
        let mut keep = PruneKeepSet::from_chunks(&corpus.chunks);
        for file in &corpus.files {
            keep.rel_paths.insert(file.rel_path.clone());
            if file.rechunked {
                keep.by_doc.entry(file.doc_path.clone()).or_default();
            } else {
                keep.whole_docs.insert(file.doc_path.clone());
            }
        }
        keep
    }

    pub(crate) fn keeps_chunk(&self, doc_path: &str, chunk_index: i64) -> bool {
        self.whole_docs.contains(doc_path)
            || self
                .by_doc
                .get(doc_path)
                .is_some_and(|idx| idx.contains(&chunk_index))
    }

    pub(crate) fn keeps_doc(&self, doc_path: &str) -> bool {
        self.whole_docs.contains(doc_path) || self.by_doc.contains_key(doc_path)
    }
}

/// What a prune pass removed (or, with `dry_run`, would remove) for one project.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PruneOutcome {
    /// `project_chunks` rows.
    pub(crate) chunks: usize,
    /// Distinct doc_paths that lost at least one chunk.
    pub(crate) files: usize,
    /// `project_files` manifest rows.
    pub(crate) manifest_rows: usize,
    /// `symbols` rows.
    pub(crate) symbol_rows: usize,
    /// `file_imports` rows.
    pub(crate) import_rows: usize,
    /// `file_dependency_edges` rows whose source or target file is gone.
    pub(crate) edge_rows: usize,
    /// Ids of the removed chunks, for the LanceDB delete.
    pub(crate) chunk_ids: Vec<i64>,
}

impl PruneOutcome {
    pub(crate) fn is_empty(&self) -> bool {
        self.chunks == 0
            && self.manifest_rows == 0
            && self.symbol_rows == 0
            && self.import_rows == 0
            && self.edge_rows == 0
    }

    fn absorb(&mut self, other: PruneOutcome) {
        self.chunks += other.chunks;
        self.files += other.files;
        self.manifest_rows += other.manifest_rows;
        self.symbol_rows += other.symbol_rows;
        self.import_rows += other.import_rows;
        self.edge_rows += other.edge_rows;
        self.chunk_ids.extend(other.chunk_ids);
    }
}

pub(crate) fn project_chunk_ids(conn: &Connection, project_id: i64) -> Result<Vec<i64>, String> {
    let mut stmt = conn
        .prepare("SELECT id FROM project_chunks WHERE project_id = ?1")
        .map_err(|e| format!("failed preparing project chunk id query: {}", e))?;
    let rows = stmt
        .query_map(params![project_id], |row| row.get::<_, i64>(0))
        .map_err(|e| format!("failed querying project chunk ids: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project chunk id: {}", e))?);
    }
    Ok(out)
}

/// Distinct values of `column` in `table` for one project. `table`/`column` are code
/// constants, never user input.
pub(crate) fn project_distinct_paths(
    conn: &Connection,
    table: &str,
    column: &str,
    project_id: i64,
) -> Result<Vec<String>, String> {
    let sql = format!(
        "SELECT DISTINCT {} FROM {} WHERE project_id = ?1",
        column, table
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("failed preparing {} path query: {}", table, e))?;
    let rows = stmt
        .query_map(params![project_id], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying {} paths: {}", table, e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading {} path: {}", table, e))?);
    }
    Ok(out)
}

/// Delete (or, with `dry_run`, count) rows of `table` whose `column` is one of `paths`.
pub(crate) fn delete_project_rows_by_path(
    conn: &Connection,
    table: &str,
    column: &str,
    project_id: i64,
    paths: &[String],
    dry_run: bool,
) -> Result<usize, String> {
    if paths.is_empty() {
        return Ok(0);
    }
    let sql = if dry_run {
        format!(
            "SELECT COUNT(*) FROM {} WHERE project_id = ?1 AND {} = ?2",
            table, column
        )
    } else {
        format!(
            "DELETE FROM {} WHERE project_id = ?1 AND {} = ?2",
            table, column
        )
    };
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("failed preparing {} prune: {}", table, e))?;
    let mut n = 0usize;
    for path in paths {
        if dry_run {
            let count: i64 = stmt
                .query_row(params![project_id, path], |row| row.get(0))
                .map_err(|e| format!("failed counting {} rows: {}", table, e))?;
            n += count.max(0) as usize;
        } else {
            n += stmt
                .execute(params![project_id, path])
                .map_err(|e| format!("failed pruning {} rows: {}", table, e))?;
        }
    }
    Ok(n)
}

/// Remove the rows of `project_id` that `keep` (built from the freshly collected corpus) no
/// longer covers: `project_chunks` whose (doc_path, chunk_index) is absent, and the per-file
/// `project_files`, `symbols`, `file_imports` and `file_dependency_edges` rows of doc_paths
/// that are absent altogether. `project_chunk_vectors` and `chunk_relation_feedback` follow
/// through `ON DELETE CASCADE`. sqlite only: the returned
/// `chunk_ids` are for the caller's LanceDB delete. With `dry_run` nothing is written and
/// the counts describe what would be removed.
pub(crate) fn prune_stale_project_rows(
    conn: &Connection,
    project_id: i64,
    keep: &PruneKeepSet,
    dry_run: bool,
) -> Result<PruneOutcome, String> {
    let tx = conn
        .unchecked_transaction()
        .map_err(|e| format!("failed starting prune transaction: {}", e))?;
    let out = prune_stale_project_rows_in(&tx, project_id, keep, dry_run)?;
    tx.commit()
        .map_err(|e| format!("failed committing prune transaction: {}", e))?;
    Ok(out)
}

/// [`prune_stale_project_rows`] inside the caller's transaction (`conn` is the transaction):
/// no BEGIN or COMMIT of its own, so a project's publish can include it.
pub(crate) fn prune_stale_project_rows_in(
    tx: &Connection,
    project_id: i64,
    keep: &PruneKeepSet,
    dry_run: bool,
) -> Result<PruneOutcome, String> {
    let conn = tx;
    let mut out = PruneOutcome::default();

    // Chunks the corpus no longer produces.
    let mut touched_docs: HashSet<String> = HashSet::new();
    {
        let mut stmt = conn
            .prepare("SELECT id, doc_path, chunk_index FROM project_chunks WHERE project_id = ?1")
            .map_err(|e| format!("failed preparing stale chunk query: {}", e))?;
        let rows = stmt
            .query_map(params![project_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| format!("failed querying stale chunks: {}", e))?;
        for row in rows {
            let (id, doc_path, chunk_index) =
                row.map_err(|e| format!("failed reading stale chunk row: {}", e))?;
            if keep.keeps_chunk(&doc_path, chunk_index) {
                continue;
            }
            out.chunk_ids.push(id);
            touched_docs.insert(doc_path);
        }
    }
    out.chunks = out.chunk_ids.len();
    out.files = touched_docs.len();

    // Per-file rows for doc_paths that are gone entirely (not merely shorter). These tables
    // are refreshed per file during indexing, so nothing else deletes rows for vanished files.
    let stale_docs = |paths: Vec<String>| -> Vec<String> {
        paths.into_iter().filter(|p| !keep.keeps_doc(p)).collect()
    };
    let stale_symbol_docs = stale_docs(project_distinct_paths(
        conn, "symbols", "doc_path", project_id,
    )?);
    let stale_import_docs = stale_docs(project_distinct_paths(
        conn,
        "file_imports",
        "source_doc_path",
        project_id,
    )?);
    // Dependency edges go when either end is gone: `helper.py -> util.py` must not outlive
    // util.py. Sources are doc_paths; targets are the resolver's project-relative form (older
    // rows may hold absolute doc_paths, matched against the doc set).
    let stale_edges: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT source_doc_path, target_doc_path FROM file_dependency_edges WHERE project_id = ?1",
            )
            .map_err(|e| format!("failed preparing dependency edge query: {}", e))?;
        let rows = stmt
            .query_map(params![project_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("failed querying dependency edges: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let (source, target) =
                row.map_err(|e| format!("failed reading dependency edge: {}", e))?;
            let target_kept = if target.starts_with('/') {
                keep.keeps_doc(&target)
            } else {
                keep.rel_paths.contains(&target)
            };
            if !keep.keeps_doc(&source) || !target_kept {
                out.push((source, target));
            }
        }
        out
    };
    let stale_manifest: Vec<String> =
        project_distinct_paths(conn, "project_files", "rel_path", project_id)?
            .into_iter()
            .filter(|rel| !keep.rel_paths.contains(rel))
            .collect();

    if !dry_run && !out.chunk_ids.is_empty() {
        // LanceDB still holds these rows until the caller deletes them: committed with the
        // sqlite deletes, so a crash in between leaves the marker set for the repair step.
        lance_mark_dirty(tx)?;
    }
    if !dry_run {
        for batch in out.chunk_ids.chunks(500) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let sql = format!("DELETE FROM project_chunks WHERE id IN ({})", placeholders);
            tx.execute(&sql, rusqlite::params_from_iter(batch.iter()))
                .map_err(|e| format!("failed deleting stale chunks: {}", e))?;
        }
    }
    out.symbol_rows = delete_project_rows_by_path(
        tx,
        "symbols",
        "doc_path",
        project_id,
        &stale_symbol_docs,
        dry_run,
    )?;
    out.import_rows = delete_project_rows_by_path(
        tx,
        "file_imports",
        "source_doc_path",
        project_id,
        &stale_import_docs,
        dry_run,
    )?;
    out.edge_rows = stale_edges.len();
    if !dry_run {
        for (source, target) in &stale_edges {
            tx.execute(
                "DELETE FROM file_dependency_edges WHERE project_id = ?1 AND source_doc_path = ?2 AND target_doc_path = ?3",
                params![project_id, source, target],
            )
            .map_err(|e| format!("failed pruning file_dependency_edges rows: {}", e))?;
        }
    }
    out.manifest_rows = delete_project_rows_by_path(
        tx,
        "project_files",
        "rel_path",
        project_id,
        &stale_manifest,
        dry_run,
    )?;
    Ok(out)
}

#[cfg(test)]
mod prune_tests {
    use super::*;
    use crate::db::init_schema;
    use crate::index::{upsert_project_chunk, ProjectChunk};
    use rusqlite::Connection;
    use std::collections::HashSet;

    const KEEP_DOC: &str = "/p/alpha/src/keep.rs";
    const STALE_DOC: &str = "/p/alpha/tmp/stale.md";

    fn seeded_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        init_schema(&conn).expect("init schema");
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/alpha', 'alpha', 'alpha', 0, 0), (2, '/p/beta', 'beta', 'beta', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (10, 1, '/p/alpha/src/keep.rs', 'src/keep.rs', 0, 0, 5, 'h10', 'keep zero', 0),
       (11, 1, '/p/alpha/src/keep.rs', 'src/keep.rs', 0, 1, 5, 'h11', 'keep one', 0),
       (12, 1, '/p/alpha/tmp/stale.md', 'tmp/stale.md', 0, 0, 5, 'h12', 'stale', 0),
       (20, 2, '/p/beta/tmp/other.md', 'tmp/other.md', 0, 0, 5, 'h20', 'other project', 0);
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector)
VALUES (10, 'm', 1, 1.0, x'00000000'), (11, 'm', 1, 1.0, x'00000000'),
       (12, 'm', 1, 1.0, x'00000000'), (20, 'm', 1, 1.0, x'00000000');
INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed)
VALUES (1, 'src/keep.rs', '/p/alpha/src/keep.rs', 1, 0, 'x', 2, 0),
       (1, 'tmp/stale.md', '/p/alpha/tmp/stale.md', 1, 0, 'y', 1, 0);
INSERT INTO symbols(project_id, doc_path, doc_rel_path, name, kind, line_start, line_end, updated_at)
VALUES (1, '/p/alpha/src/keep.rs', 'src/keep.rs', 'keep', 'function', 1, 2, 0),
       (1, '/p/alpha/tmp/stale.md', 'tmp/stale.md', 'stale', 'function', 1, 2, 0);
INSERT INTO file_imports(project_id, source_doc_path, import_kind, raw_specifier, updated_at)
VALUES (1, '/p/alpha/tmp/stale.md', 'use', 'keep', 0);
INSERT INTO file_dependency_edges(project_id, source_doc_path, target_doc_path, edge_kind, updated_at)
VALUES (1, '/p/alpha/tmp/stale.md', '/p/alpha/src/keep.rs', 'import', 0),
       (1, '/p/alpha/src/keep.rs', '/p/alpha/tmp/stale.md', 'import', 0);
INSERT INTO chunk_relation_feedback(src_chunk_id, dst_chunk_id, relation, decision, created_at, updated_at)
VALUES (12, 10, 'related', 'active', 0, 0);
"#,
        )
        .expect("seed rows");
        conn
    }

    fn count(conn: &Connection, sql: &str) -> i64 {
        conn.query_row(sql, [], |row| row.get(0)).expect(sql)
    }

    fn chunk(doc_path: &str, rel: &str, index: i64) -> ProjectChunk {
        ProjectChunk {
            doc_path: doc_path.to_string(),
            doc_rel_path: rel.to_string(),
            doc_mtime: 0.0,
            chunk_index: index,
            token_count: 1,
            text_hash: String::new(),
            text: String::new(),
            chunk_kind: "text_window".to_string(),
            symbol_name: String::new(),
            parent_context: String::new(),
            line_start: 0,
            line_end: 0,
            context_header: String::new(),
        }
    }

    #[test]
    fn keep_set_groups_indices_by_doc_and_tracks_rel_paths() {
        let keep = PruneKeepSet::from_chunks(&[
            chunk(KEEP_DOC, "src/keep.rs", 0),
            chunk(KEEP_DOC, "src/keep.rs", 1),
        ]);
        assert!(keep.keeps_chunk(KEEP_DOC, 0) && keep.keeps_chunk(KEEP_DOC, 1));
        assert!(!keep.keeps_chunk(KEEP_DOC, 2));
        assert!(keep.keeps_doc(KEEP_DOC) && !keep.keeps_doc(STALE_DOC));
        assert!(keep.rel_paths.contains("src/keep.rs"));
    }

    #[test]
    fn prune_removes_rows_for_paths_outside_corpus_and_keeps_survivors() {
        let conn = seeded_conn();
        let keep = PruneKeepSet::from_chunks(&[
            chunk(KEEP_DOC, "src/keep.rs", 0),
            chunk(KEEP_DOC, "src/keep.rs", 1),
        ]);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        assert_eq!(out.chunks, 1);
        assert_eq!(out.files, 1);
        assert_eq!(out.chunk_ids, vec![12]);
        assert_eq!(out.manifest_rows, 1);
        assert_eq!(out.symbol_rows, 1);
        assert_eq!(out.import_rows, 1);
        // Both edges touching the vanished file: the one it imported with and the one that
        // pointed at it (`keep.rs -> stale.md` must not outlive stale.md).
        assert_eq!(out.edge_rows, 2);

        // Survivors are untouched, with their original ids.
        let survivors: Vec<i64> = {
            let mut stmt = conn
                .prepare("SELECT id FROM project_chunks WHERE project_id = 1 ORDER BY id")
                .unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .map(|r| r.unwrap())
                .collect()
        };
        assert_eq!(survivors, vec![10, 11]);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE project_id = 1"
            ),
            1
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE rel_path = 'src/keep.rs'"
            ),
            1
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM symbols WHERE project_id = 1"),
            1
        );
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM file_imports"), 0);
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM file_dependency_edges WHERE source_doc_path = '/p/alpha/src/keep.rs'"),
            0,
            "the edge into the vanished file is gone too"
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM file_dependency_edges"),
            0
        );
        // FK cascade dropped the vector and the feedback row of chunk 12.
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE chunk_id = 12"
            ),
            0
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM project_chunk_vectors"),
            3
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM chunk_relation_feedback"),
            0
        );
        // The FTS shadow table dropped the stale row too.
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'stale'"
            ),
            0
        );
        // Other projects are never touched.
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE project_id = 2"
            ),
            1
        );
    }

    #[test]
    fn prune_trims_chunk_indices_past_the_end_of_a_shrunk_file() {
        let conn = seeded_conn();
        let keep = PruneKeepSet::from_chunks(&[
            chunk(KEEP_DOC, "src/keep.rs", 0),
            chunk(STALE_DOC, "tmp/stale.md", 0),
        ]);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        assert_eq!(out.chunk_ids, vec![11]);
        assert_eq!(out.files, 1);
        // The file is still in the corpus, so its per-file rows stay.
        assert_eq!(out.manifest_rows, 0);
        assert_eq!(out.symbol_rows, 0);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE project_id = 1"
            ),
            2
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE project_id = 1"
            ),
            2
        );
    }

    #[test]
    fn prune_dry_run_reports_without_writing() {
        let conn = seeded_conn();
        let keep = PruneKeepSet::from_chunks(&[chunk(KEEP_DOC, "src/keep.rs", 0)]);
        let out = prune_stale_project_rows(&conn, 1, &keep, true).expect("dry run");
        assert_eq!(out.chunks, 2);
        assert_eq!(out.files, 2);
        assert_eq!(out.manifest_rows, 1);
        assert_eq!(out.symbol_rows, 1);
        assert_eq!(out.import_rows, 1);
        assert_eq!(out.edge_rows, 2);
        assert!(!out.is_empty());
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 4);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_files"), 2);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM symbols"), 2);
    }

    #[test]
    fn prune_with_empty_corpus_clears_the_project() {
        let conn = seeded_conn();
        let out =
            prune_stale_project_rows(&conn, 1, &PruneKeepSet::default(), false).expect("prune");
        assert_eq!(out.chunks, 3);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE project_id = 1"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE project_id = 1"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE project_id = 2"
            ),
            1
        );
    }

    #[test]
    fn prune_is_a_no_op_when_the_corpus_matches() {
        let conn = seeded_conn();
        let keep = PruneKeepSet::from_chunks(&[
            chunk(KEEP_DOC, "src/keep.rs", 0),
            chunk(KEEP_DOC, "src/keep.rs", 1),
            chunk(STALE_DOC, "tmp/stale.md", 0),
        ]);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        assert!(out.is_empty());
        assert!(out.chunk_ids.is_empty());
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 4);
    }

    #[test]
    fn upsert_keeps_chunk_id_but_drops_feedback_when_text_changes() {
        let conn = seeded_conn();
        let same = ProjectChunk {
            text_hash: "h12".to_string(),
            ..chunk(STALE_DOC, "tmp/stale.md", 0)
        };
        let id = upsert_project_chunk(&conn, 1, &same, 1.0).expect("upsert same hash");
        assert_eq!(id, 12);
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM chunk_relation_feedback"),
            1
        );

        let changed = ProjectChunk {
            text_hash: "h12-changed".to_string(),
            ..chunk(STALE_DOC, "tmp/stale.md", 0)
        };
        let id = upsert_project_chunk(&conn, 1, &changed, 2.0).expect("upsert new hash");
        assert_eq!(
            id, 12,
            "the row keeps its id so its vector is updated in place"
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM chunk_relation_feedback"),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE chunk_id = 12"
            ),
            1
        );
    }

    #[test]
    fn remove_projects_not_in_cascades_chunks_and_vectors() {
        let conn = seeded_conn();
        let (removed, removed_chunks) =
            remove_projects_not_in(&conn, &["/p/alpha".to_string()], &HashSet::new())
                .expect("remove stale");
        assert_eq!(removed, 1);
        assert_eq!(removed_chunks, 1);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM projects"), 1);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE project_id = 2"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE chunk_id = 20"
            ),
            0
        );
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 3);
    }
}
