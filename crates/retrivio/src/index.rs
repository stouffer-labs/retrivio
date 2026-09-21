//! The index run: entry points and the writer lock, per-project publish (chunks, vectors, symbols, imports), live progress, LanceDB synchronisation and repair, and the relationship edges.

use std::collections::{HashMap, HashSet};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{env, thread};

use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use sha1::{Digest, Sha1};

use crate::config::{data_dir, db_path, ConfigValues, ScanSettings};
use crate::config_tui::{clipped, terminal_size_stty};
use crate::db::{
    app_state_bool, app_state_get, app_state_set, ensure_reembed_ready,
    ensure_retrieval_backend_ready, get_or_open_lance, lance_store_is_open, open_db_writer,
    persist_reembed_requirement, vector_dim_from_sqlite, with_lance_store, WriterLock,
    APP_STATE_ACTIVE_MODEL_KEY, APP_STATE_EMBED_FINGERPRINT, APP_STATE_LANCE_DIRTY,
    APP_STATE_REEMBED_REQUIRED, APP_STATE_SCAN_CAPS_FINGERPRINT,
};
use crate::embed::{
    build_embedder, embed_runtime_snapshot, ensure_native_embed_backend,
    reset_embed_runtime_metrics, Embedder,
};
use crate::prune::{
    prune_stale_project_rows_in, remove_projects_not_in, unavailable_roots, PruneKeepSet,
    PruneOutcome,
};
use crate::scan::{
    collect_project_corpus_verifying, load_file_manifest, project_excludes_for_path, project_scan,
    read_for_index, resolve_index_targets, upsert_file_manifest, verify_rel_paths, FileManifest,
    IndexScope, IndexTargets, ProjectScan, ScanCaps,
};
use crate::util::{
    blob_to_f32_vec, cosine_raw, f32_blob, format_duration_ms, is_under_any, non_empty_env,
    normalize_path, now_ts, vector_norm, word_tokens,
};
use crate::{code_intel, lance_store};

/// Bump whenever chunking or context-header logic changes the string handed to the embedder.
/// Stored vectors carrying another version are never reused, and the next full run re-chunks
/// every project so the new inputs reach the identity check.
pub(crate) const EMBEDDING_PIPELINE_VERSION: i64 = 1;

/// Set once a LanceDB write failed in this process: the dirty marker is then never cleared by
/// a later successful write (the earlier failure's rows are still missing).
pub(crate) static LANCE_WRITE_FAILED: AtomicBool = AtomicBool::new(false);

pub(crate) static PROGRESS_IO_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn progress_heartbeat_interval() -> Duration {
    non_empty_env("RETRIVIO_PROGRESS_HEARTBEAT_MS")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|ms| Duration::from_millis(ms.clamp(500, 60_000)))
        .unwrap_or_else(|| Duration::from_secs(5))
}

pub(crate) fn progress_io_lock() -> &'static Mutex<()> {
    PROGRESS_IO_LOCK.get_or_init(|| Mutex::new(()))
}

pub(crate) fn progress_use_single_line() -> bool {
    std::io::stderr().is_terminal()
}

pub(crate) fn progress_clear_line() {
    if !progress_use_single_line() {
        return;
    }
    if let Ok(_g) = progress_io_lock().lock() {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

/// The knobs every entry of the index-run family takes ([`run_index_with_strategy`],
/// [`run_index_with_lock`], [`run_native_index`], [`run_native_index_verifying`],
/// [`run_native_index_with_embedder`]): what the run covers, what it forces, whether it prunes,
/// and the reason shown in progress and logs. `verify_paths` and `emit_progress` stay positional
/// because the wrappers decide them (`run_index_with_lock` always draws progress,
/// `run_native_index` never verifies).
pub(crate) struct IndexRunOptions<'a> {
    pub(crate) scope: IndexScope,
    /// Re-collect every project in scope regardless of its scan signature.
    pub(crate) force_all: bool,
    /// Projects re-collected regardless of their signature even when `force_all` is off.
    pub(crate) force_paths: HashSet<PathBuf>,
    /// Drop project rows whose directory is gone or no longer under a tracked root.
    pub(crate) remove_missing: bool,
    pub(crate) reason: &'a str,
}

pub(crate) fn run_index_with_strategy(
    cwd: &Path,
    cfg: &ConfigValues,
    opts: IndexRunOptions<'_>,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, opts.reason)?;
    ensure_native_embed_backend(cfg, opts.reason)?;
    let writer = WriterLock::try_acquire(&data_dir(cwd))?;
    run_index_with_lock(cwd, cfg, &writer, opts)
}

/// [`run_index_with_strategy`] for a caller that already holds the writer lock.
pub(crate) fn run_index_with_lock(
    cwd: &Path,
    cfg: &ConfigValues,
    writer: &WriterLock,
    opts: IndexRunOptions<'_>,
) -> Result<(), String> {
    let stats = run_native_index(cwd, cfg, writer, opts, true)?;
    print_index_stats(&stats, cfg);
    index_run_verdict(&stats)
}

/// `Err` (a non-zero exit for the CLI) when the run left work behind: a failed project, a run
/// stopped early, or a LanceDB open, repair or write failure. Everything sqlite holds is
/// committed either way; the message says what the next run will retry.
pub(crate) fn index_run_verdict(stats: &IndexStats) -> Result<(), String> {
    let mut problems: Vec<String> = Vec::new();
    if stats.projects_failed > 0 {
        problems.push(format!(
            "{} project(s) failed and keep their previous state",
            stats.projects_failed
        ));
    }
    if !stats.stopped.is_empty() {
        problems.push(format!("run stopped early: {}", stats.stopped));
    }
    if !stats.lance_error.is_empty() {
        problems.push(format!(
            "LanceDB: {} (dirty marker set; repaired on the next run)",
            stats.lance_error
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "index finished with problems: {}; sqlite content is committed, rerun `retrivio index` to retry",
            problems.join("; ")
        ))
    }
}

#[derive(Default, Debug)]
pub(crate) struct IndexStats {
    pub(crate) total_projects: i64,
    pub(crate) updated_projects: i64,
    pub(crate) skipped_projects: i64,
    pub(crate) removed_projects: i64,
    pub(crate) vectorized_projects: i64,
    pub(crate) vector_failures: i64,
    pub(crate) tracked_roots: i64,
    pub(crate) graph_edges: i64,
    pub(crate) chunk_rows: i64,
    pub(crate) chunk_vectors: i64,
    pub(crate) pruned_chunks: i64,
    pruned_files: i64,
    /// Indexable files the scanned projects selected (skipped projects contribute nothing).
    pub(crate) files_selected: i64,
    /// Selected files the manifest showed unchanged: kept as stored, never read or chunked.
    pub(crate) files_unchanged: i64,
    /// Selected files that were new or changed and were read and chunked.
    pub(crate) files_rechunked: i64,
    /// Directory entries the scans could not read; such a project is indexed from what was
    /// readable but nothing is pruned and its signature is not advanced.
    pub(crate) files_unreadable: i64,
    /// Scanned projects with at least one unreadable entry.
    pub(crate) projects_incomplete: i64,
    /// Candidate files a cap left out of the index entirely.
    pub(crate) files_evicted_by_cap: i64,
    /// Files indexed only in part because a cap cut their chunks or text.
    pub(crate) files_truncated_by_cap: i64,
    /// Chunks sent to the embedder.
    pub(crate) chunks_embedded: i64,
    /// Chunks whose stored vector matched the embedding identity and was kept.
    pub(crate) chunks_reused: i64,
    /// Chunk rows deleted: pruned from scanned projects plus those of removed projects.
    pub(crate) chunks_deleted: i64,
    /// LanceDB rows rebuilt from sqlite vectors by the repair step (no embedding).
    pub(crate) lance_repaired: i64,
    /// LanceDB rows with no sqlite vector removed by the repair step.
    pub(crate) lance_orphans_removed: i64,
    /// Documents (docx, pptx, odt, odp, xlsx, pdf, html) whose text was extracted this run.
    pub(crate) documents_extracted: i64,
    /// Documents that yielded no text (over a bound, corrupt, parser error or panic, PDF
    /// child killed). A previously indexed one keeps its old content.
    pub(crate) documents_failed: i64,
    /// Projects whose run failed (collector panic, sqlite or code-intelligence error, the
    /// embedding failure that stopped the run). Each keeps its previous state: old signature,
    /// nothing pruned, nothing published; the fingerprints do not advance. `index` exits
    /// non-zero when this is non-zero.
    pub(crate) projects_failed: i64,
    /// `<project>: <reason>` for every failed project, in run order.
    pub(crate) failures: Vec<String>,
    /// Non-empty when the run stopped before visiting every project (the embedding backend
    /// failed); names the reason and how many projects were left for the next run.
    pub(crate) stopped: String,
    /// Non-empty when LanceDB could not be opened, repaired or written this run. The dirty
    /// marker is set and the next writer run repairs LanceDB from sqlite; sqlite content is
    /// complete and committed. `index` exits non-zero when this is non-empty.
    pub(crate) lance_error: String,
    pub(crate) retrieval_backend: String,
    pub(crate) retrieval_synced_chunks: i64,
    pub(crate) retrieval_error: String,
    elapsed_index_ms: u64,
    pub(crate) elapsed_sync_ms: u64,
    pub(crate) elapsed_total_ms: u64,
}

#[derive(Clone)]
pub(crate) struct ProjectDoc {
    pub(crate) path: PathBuf,
    pub(crate) title: String,
    pub(crate) summary: String,
    pub(crate) mtime: f64,
}

#[derive(Clone)]
pub(crate) struct ProjectChunk {
    pub(crate) doc_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) doc_mtime: f64,
    pub(crate) chunk_index: i64,
    pub(crate) token_count: i64,
    pub(crate) text_hash: String,
    pub(crate) text: String,
    // Code intelligence metadata (populated by AST chunker, empty for fallback chunks)
    pub(crate) chunk_kind: String,
    pub(crate) symbol_name: String,
    pub(crate) parent_context: String,
    pub(crate) line_start: i64,
    pub(crate) line_end: i64,
    pub(crate) context_header: String,
}

/// One indexable file a project scan selected.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScannedFile {
    pub(crate) rel_path: String,
    /// `<canonical project path>/<rel_path>`: the `doc_path` of its chunks and symbols. Derived
    /// from the project path, never canonicalised per file, so the keep set and the stored rows
    /// always agree (a file that is or becomes a symlink keeps its identity).
    pub(crate) doc_path: String,
    pub(crate) size: i64,
    pub(crate) mtime: f64,
    pub(crate) content_hash: String,
    /// Chunks the file contributes: freshly produced when `rechunked`, else the manifest's.
    pub(crate) chunk_count: i64,
    /// True when the file was read and chunked this run (new or changed); false when the
    /// manifest showed it unchanged and its stored chunks, vectors, symbols and imports stand.
    pub(crate) rechunked: bool,
    /// True when the stat differed from the manifest but the content hash did not (touch,
    /// restored copy): the manifest row gets the new stat so the file is not re-hashed on
    /// every run, and nothing is re-chunked.
    pub(crate) stat_changed: bool,
}

pub(crate) struct ProjectCorpus {
    pub(crate) doc: ProjectDoc,
    /// Chunks of the re-chunked files only (the work set).
    pub(crate) chunks: Vec<ProjectChunk>,
    /// Every selected file, unchanged ones included (the full keep set).
    pub(crate) files: Vec<ScannedFile>,
    /// The scan signature of the selected files, stored on the project row once the whole
    /// project succeeded (see [`scan_signature_for`]).
    pub(crate) scan_signature: String,
    /// False when the walk could not read every directory entry: prune and the signature
    /// update are skipped for this project, so nothing unseen is treated as deleted.
    pub(crate) complete: bool,
    /// Directory entries the walk could not read.
    pub(crate) files_unreadable: i64,
    /// Selected candidates left out entirely by `max_files_per_project` or
    /// `max_chunks_per_project`.
    pub(crate) files_evicted_by_cap: i64,
    /// Files indexed only partially: chunks cut by `max_chunks_per_file`, text cut by
    /// `max_file_chars`, or the project chunk cap reached mid-file.
    pub(crate) files_truncated_by_cap: i64,
    /// One line naming the caps that bit, empty when none did.
    pub(crate) caps_note: String,
    /// Documents (docx, pptx, odt, odp, xlsx, pdf, html) whose text was extracted this run.
    pub(crate) documents_extracted: i64,
    /// Documents in the work set that yielded no text: over a bound, corrupt, a parser error
    /// or panic, or the PDF child killed. One indexed before stays exactly as indexed (carried
    /// as unchanged, manifest entry not advanced); one never indexed is absent.
    pub(crate) documents_failed: i64,
    /// (rel_path, reason) for every failed document, for the run's warnings.
    pub(crate) document_failures: Vec<(String, String)>,
}

impl ProjectCorpus {
    fn files_unchanged(&self) -> i64 {
        self.files.iter().filter(|f| !f.rechunked).count() as i64
    }

    fn files_rechunked(&self) -> i64 {
        self.files.iter().filter(|f| f.rechunked).count() as i64
    }
}

pub(crate) struct ExistingProject {
    pub(crate) id: i64,
    path: String,
    pub(crate) title: String,
    pub(crate) summary: String,
    project_mtime: f64,
    /// Signature of the last complete, successful scan ('' before the first one).
    pub(crate) scan_signature: String,
    /// True when a previous run started writing this project and never finished.
    pub(crate) index_in_progress: bool,
}

#[derive(Default)]
pub(crate) struct LiveIndexProgress {
    completed_projects: AtomicUsize,
    current_project_idx: AtomicUsize,
    current_project_chunks_done: AtomicU64,
    current_project_chunks_total: AtomicU64,
    current_project_tokens_done: AtomicU64,
    current_project_tokens_total: AtomicU64,
    current_project_name: Mutex<String>,
    current_phase: Mutex<String>,
}

impl LiveIndexProgress {
    fn mark_project_done(&self) {
        self.completed_projects.fetch_add(1, Ordering::Relaxed);
    }

    fn start_project(
        &self,
        project_idx_1based: usize,
        project_name: &str,
        total_chunks: usize,
        total_tokens: u64,
    ) {
        self.set_phase("embed");
        self.current_project_idx
            .store(project_idx_1based, Ordering::Relaxed);
        self.current_project_chunks_done.store(0, Ordering::Relaxed);
        self.current_project_chunks_total
            .store(total_chunks as u64, Ordering::Relaxed);
        self.current_project_tokens_done.store(0, Ordering::Relaxed);
        self.current_project_tokens_total
            .store(total_tokens, Ordering::Relaxed);
        if let Ok(mut name) = self.current_project_name.lock() {
            *name = project_name.to_string();
        }
    }

    fn add_chunks_done(&self, n: usize) {
        self.current_project_chunks_done
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    fn add_tokens_done(&self, n: u64) {
        self.current_project_tokens_done
            .fetch_add(n, Ordering::Relaxed);
    }

    fn set_phase(&self, phase: &str) {
        if let Ok(mut cur) = self.current_phase.lock() {
            *cur = phase.to_string();
        }
    }
}

pub(crate) fn progress_spinner(elapsed: Duration) -> char {
    match ((elapsed.as_millis() / 250) % 4) as u8 {
        0 => '|',
        1 => '/',
        2 => '-',
        _ => '\\',
    }
}

pub(crate) fn format_ascii_bar(width: usize, ratio: f64) -> String {
    let w = width.max(8);
    let r = ratio.clamp(0.0, 1.0);
    let filled = ((w as f64) * r).round() as usize;
    let filled = filled.min(w);
    let mut out = String::with_capacity(w + 2);
    out.push('[');
    out.push_str(&"#".repeat(filled));
    out.push_str(&"-".repeat(w - filled));
    out.push(']');
    out
}

pub(crate) fn compact_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", (n as f64) / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", (n as f64) / 1_000.0)
    } else {
        n.to_string()
    }
}

pub(crate) fn render_live_progress_line(
    reason: &str,
    total_projects: usize,
    started_at: Instant,
    progress: &LiveIndexProgress,
) {
    if !progress_use_single_line() {
        return;
    }
    let completed = progress.completed_projects.load(Ordering::Relaxed);
    let current_idx = progress.current_project_idx.load(Ordering::Relaxed);
    let chunks_done = progress.current_project_chunks_done.load(Ordering::Relaxed);
    let chunks_total = progress
        .current_project_chunks_total
        .load(Ordering::Relaxed);
    let tokens_done = progress.current_project_tokens_done.load(Ordering::Relaxed);
    let tokens_total = progress
        .current_project_tokens_total
        .load(Ordering::Relaxed);
    let phase = progress
        .current_phase
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| "index".to_string());
    let m = embed_runtime_snapshot();
    let avg_ms = m.latency_sum_ms.checked_div(m.latency_samples).unwrap_or(0);
    let elapsed = started_at.elapsed();
    let elapsed_s = elapsed.as_secs_f64().max(0.001);
    let req_rate = m.requests_succeeded as f64 / elapsed_s;
    let chunk_ratio = if chunks_total > 0 {
        chunks_done as f64 / (chunks_total as f64)
    } else {
        0.0
    };
    let project_ratio = if total_projects > 0 {
        ((completed as f64) + chunk_ratio) / (total_projects as f64)
    } else {
        1.0
    };
    let done_projects = completed.min(total_projects);
    let done_all = total_projects == 0 || done_projects >= total_projects;
    let bar_ratio = if done_all {
        project_ratio
    } else {
        project_ratio.min(0.999)
    };
    let mut pct = project_ratio.clamp(0.0, 1.0) * 100.0;
    if !done_all {
        pct = pct.min(99.9);
    }
    let bar = format_ascii_bar(18, bar_ratio);
    let spin = progress_spinner(elapsed);
    let line = format!(
        "{spin} {reason} {bar} {pct:5.1}% | phase {phase} | proj {done}/{total} act#{cur}/{total} | active ch {ch_done}/{ch_total} tok {tok_done}/{tok_total} emb {emb} | req {ok}/{fail} if {inflight} rt {retry} th {thr} | {rate:.1}/s avg {avg}ms max {max}ms",
        spin = spin,
        reason = reason,
        bar = bar,
        pct = pct,
        phase = phase,
        done = done_projects,
        total = total_projects,
        cur = current_idx.min(total_projects),
        ch_done = compact_count(chunks_done),
        ch_total = compact_count(chunks_total),
        tok_done = compact_count(tokens_done),
        tok_total = compact_count(tokens_total),
        emb = compact_count(m.texts_embedded),
        ok = compact_count(m.requests_succeeded),
        fail = compact_count(m.requests_failed),
        inflight = m.in_flight.max(0),
        retry = m.request_retries,
        thr = m.throttles,
        rate = req_rate,
        avg = avg_ms,
        max = m.latency_max_ms
    );
    let (cols, _) = terminal_size_stty();
    let display = clipped(&line, cols.saturating_sub(1));
    if let Ok(_g) = progress_io_lock().lock() {
        eprint!("\r\x1b[2K{}", display);
        let _ = std::io::stderr().flush();
    }
}

pub(crate) struct LiveProgressReporter {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LiveProgressReporter {
    fn start(
        reason: &str,
        total_projects: usize,
        started_at: Instant,
        progress: Arc<LiveIndexProgress>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let reason = reason.to_string();
        let interval = progress_heartbeat_interval();
        let handle = thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                thread::sleep(interval);
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                render_live_progress_line(&reason, total_projects, started_at, &progress);
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for LiveProgressReporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        progress_clear_line();
    }
}

/// One indexing run. The caller holds the writer lock (`writer`), so this is the only process
/// writing to the store and the only place a schema migration can happen.
pub(crate) fn run_native_index(
    cwd: &Path,
    cfg: &ConfigValues,
    writer: &WriterLock,
    opts: IndexRunOptions<'_>,
    emit_progress: bool,
) -> Result<IndexStats, String> {
    run_native_index_verifying(cwd, cfg, writer, opts, HashSet::new(), emit_progress)
}

/// [`run_native_index`] with `verify_paths`: files (absolute) whose content hash is checked
/// against the manifest even when their size and mtime are unchanged, and whose project is
/// not skipped by the change gate on that account. The watcher passes the paths of the events
/// it received, so an edit that keeps the byte length and restores the timestamp (the one
/// change the stat-based gate cannot see) is still picked up, at the cost of reading and
/// hashing exactly those files.
pub(crate) fn run_native_index_verifying(
    cwd: &Path,
    cfg: &ConfigValues,
    writer: &WriterLock,
    opts: IndexRunOptions<'_>,
    verify_paths: HashSet<PathBuf>,
    emit_progress: bool,
) -> Result<IndexStats, String> {
    let embedder = build_embedder(cfg)?;
    run_native_index_with_embedder(
        cwd,
        cfg,
        writer,
        embedder.as_ref(),
        opts,
        verify_paths,
        emit_progress,
    )
}

/// [`run_native_index`] with the embedder supplied (tests inject a local or failing one).
///
/// Per project the order is: mark the row in progress; upsert chunks, reuse or embed their
/// vectors, prune (complete scans only) and write the manifest; refresh code intelligence;
/// embed the summary when it changed; then publish title, summary, mtime, summary vector,
/// scan signature and the cleared marker in one transaction. A failure anywhere leaves the
/// old signature (and the marker), so the next run rescans the project; nothing of the
/// unchanged files is ever deleted by a failed run.
pub(crate) fn run_native_index_with_embedder(
    cwd: &Path,
    cfg: &ConfigValues,
    writer: &WriterLock,
    embedder: &dyn Embedder,
    opts: IndexRunOptions<'_>,
    verify_paths: HashSet<PathBuf>,
    emit_progress: bool,
) -> Result<IndexStats, String> {
    let IndexRunOptions {
        scope,
        force_all,
        force_paths,
        remove_missing,
        reason,
    } = opts;
    let t_start = Instant::now();
    let verify_paths: Vec<PathBuf> = verify_paths
        .iter()
        .map(|p| normalize_path(&p.to_string_lossy()))
        .collect();
    reset_embed_runtime_metrics();
    let dbp = db_path(cwd);
    let conn = open_db_writer(&dbp, writer)?;
    if reason != "reembed" {
        // A writer: record a model change the read paths could only report.
        persist_reembed_requirement(&conn, cfg)?;
        ensure_reembed_ready(&conn, cfg, &format!("{} indexing", reason))?;
    }
    // `remove_missing` drops every project row not visited by this run; on a scoped run that
    // would delete every project outside the scope.
    if remove_missing && scope != IndexScope::AllRoots {
        return Err("internal error: remove_missing requires the all-roots scope".to_string());
    }
    let IndexTargets {
        roots,
        projects,
        incomplete_roots,
        shallow: shallow_projects,
    } = resolve_index_targets(&conn, cfg, &scope)?;
    if roots.is_empty() {
        return Err("No tracked roots configured. Add one with `retrivio add <path>`.".to_string());
    }
    let model_key = embedder.model_key();
    let mode = if force_all {
        "forced refresh"
    } else {
        "incremental index"
    };
    if emit_progress {
        progress_clear_line();
        println!(
            "{}: {} started (backend={}, model={})",
            reason, mode, cfg.embed_backend, cfg.embed_model
        );
    }

    let mut stats = IndexStats {
        tracked_roots: roots.len() as i64,
        retrieval_backend: cfg.retrieval_backend.clone(),
        ..Default::default()
    };
    stats.total_projects = projects.len() as i64;
    let live_progress = Arc::new(LiveIndexProgress::default());
    live_progress.set_phase("scan");
    let live_started_at = Instant::now();
    let _live_reporter = if emit_progress {
        Some(LiveProgressReporter::start(
            reason,
            projects.len(),
            live_started_at,
            Arc::clone(&live_progress),
        ))
    } else {
        None
    };
    if emit_progress {
        render_live_progress_line(reason, projects.len(), live_started_at, &live_progress);
    }

    // If this model already has vectors, pre-open LanceDB at that known dimension.
    // Otherwise defer opening until we have real embeddings from the current run. A failure
    // is an error of this run (non-zero exit): sqlite is still written in full, the dirty
    // marker is set and the next writer run repairs LanceDB from it.
    if let Some(lance_dim) = vector_dim_from_sqlite(&conn, &model_key) {
        if let Err(e) = get_or_open_lance(cwd, lance_dim) {
            lance_mark_dirty(&conn)?;
            LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
            progress_clear_line();
            eprintln!(
                "error: LanceDB open failed: {}; vectors are written to sqlite only, the dirty marker is set and the next index run repairs LanceDB",
                e
            );
            stats.lance_error = format!("open failed: {}", e);
        }
    }
    // Reconcile LanceDB with sqlite before writing anything, when a previous write failed or
    // was interrupted (dirty marker) or the store has never been checked. `reembed` rebuilds
    // LanceDB wholesale at its end and skips this. A repair that cannot complete (malformed
    // or missing sqlite vector, LanceDB error) fails closed: marker and pending ids stay.
    if reason != "reembed" {
        match repair_lance_from_sqlite(&conn, &model_key, false) {
            Ok(report) => {
                stats.lance_repaired = report.rebuilt as i64;
                stats.lance_orphans_removed = report.orphans_removed as i64;
                if emit_progress && (report.rebuilt > 0 || report.orphans_removed > 0) {
                    progress_clear_line();
                    println!(
                        "{}: LanceDB repaired from sqlite: {} rows rebuilt, {} orphan rows removed (no embedding)",
                        reason, report.rebuilt, report.orphans_removed
                    );
                }
            }
            Err(e) => {
                lance_mark_dirty(&conn)?;
                progress_clear_line();
                eprintln!(
                    "error: LanceDB repair failed: {}; the dirty marker stays set and the next index run retries",
                    e
                );
                if stats.lance_error.is_empty() {
                    stats.lance_error = format!("repair failed: {}", e);
                }
            }
        }
    }

    let identity = EmbedIdentity::for_run(&conn, embedder);
    // `reembed` is the only path that discards matching stored vectors.
    let force_embed = reason == "reembed";
    // The embedding fingerprint (model, dimension, normalisation, pipeline version) of the
    // last complete run. A change means stored vectors may no longer match the current
    // identity even for unchanged files, so every project is re-chunked this run and the
    // per-chunk identity check decides what to embed (a read and a hash per chunk when
    // nothing changed, never an embed). Written after a complete all-roots run.
    let stored_fingerprint = app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT)?;
    let current_fingerprint = identity.fingerprint();
    let fingerprint_changed =
        stored_fingerprint.is_none() || stored_fingerprint != current_fingerprint;
    if emit_progress && fingerprint_changed && !projects.is_empty() {
        progress_clear_line();
        println!(
            "{}: embedding fingerprint {} -> {}; every project is revisited (vectors are reused where their identity matches)",
            reason,
            stored_fingerprint.as_deref().unwrap_or("none"),
            current_fingerprint.as_deref().unwrap_or("unknown")
        );
    }
    let caps = ScanCaps::from_cfg(cfg);
    let settings = ScanSettings::from_cfg(cfg);
    // The caps are not part of any file's stat, so a tighter or looser cap (or a change to
    // document indexing) would otherwise wait for the next edit in each project. Like the
    // embedding fingerprint: a change revisits every project once, re-chunking so per-file
    // caps apply too; vectors are reused where their identity matches.
    let stored_caps = app_state_get(&conn, APP_STATE_SCAN_CAPS_FINGERPRINT)?;
    let current_caps = caps.fingerprint(&settings);
    let caps_changed = stored_caps.as_deref() != Some(current_caps.as_str());
    if emit_progress && caps_changed && !projects.is_empty() {
        progress_clear_line();
        println!(
            "{}: scan caps {} -> {}; every project is revisited (files past a cap are pruned, vectors are reused where their identity matches)",
            reason,
            stored_caps.as_deref().unwrap_or("none"),
            current_caps
        );
    }

    let mut keep_paths: Vec<String> = Vec::new();
    let mut docs_by_id: HashMap<i64, ProjectDoc> = HashMap::new();
    let now = now_ts();
    let project_label = |dir: &Path| -> String {
        let base = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("project")
            .to_string();
        if shallow_projects.contains(dir) {
            format!("{} (root files)", base)
        } else {
            base
        }
    };

    // ── Phase 1: Determine which projects need updating (sequential, needs conn) ──
    struct ProjectWork {
        idx: usize,
        dir: PathBuf,
        /// The gate's walk and selection, handed to the collector so it never walks again.
        scan: ProjectScan,
        /// The stored per-file manifest; empty for a project never indexed with one.
        manifest: FileManifest,
        /// Read and chunk every file, not only the changed ones.
        rechunk_all: bool,
        /// Relative paths whose content hash is checked even when their stat is unchanged.
        verify: HashSet<String>,
        /// The stored row, whose title, summary and vector are reused when unchanged.
        existing: Option<ExistingProject>,
    }
    let mut needs_update: Vec<ProjectWork> = Vec::new();

    for (idx, project_dir) in projects.iter().enumerate() {
        let project_path = normalize_path(&project_dir.to_string_lossy())
            .to_string_lossy()
            .to_string();
        keep_paths.push(project_path.clone());
        let forced = force_all || is_under_any(project_dir, &force_paths);
        let project_excludes = project_excludes_for_path(project_dir, &roots);
        let scan = project_scan(
            project_dir,
            &project_excludes,
            &caps,
            shallow_projects.contains(project_dir),
            &settings,
        );
        // Event paths inside this project that the scan selected: a deleted or excluded path
        // verifies nothing (a deletion moves the signature anyway).
        let mut verify = verify_rel_paths(Path::new(&project_path), &verify_paths);
        verify.retain(|rel| scan.selected.iter().any(|c| c.rel_path == *rel));
        if !scan.complete() && emit_progress {
            progress_clear_line();
            println!(
                "[{}/{}] warning: {}: {} directory entries unreadable; nothing of it is pruned and its signature is not advanced",
                idx + 1,
                projects.len(),
                project_label(project_dir),
                scan.listing.unreadable
            );
        }
        let existing = get_project_by_path(&conn, &project_path)?;
        let mut manifest = FileManifest::new();
        let mut rechunk_all = forced || fingerprint_changed || caps_changed;
        if let Some(row) = &existing {
            // The gate: signature differs, vectors incomplete for the current identity, the
            // fingerprint changed, or a previous run was interrupted.
            let signature_unchanged =
                !row.scan_signature.is_empty() && row.scan_signature == scan.signature;
            if !forced
                && !fingerprint_changed
                && !caps_changed
                && signature_unchanged
                && !row.index_in_progress
            {
                let project_vec_ready = project_vector_matches(&conn, row.id, &identity)?;
                let chunk_count = count_project_chunks(&conn, row.id)?;
                let chunk_vec_count = count_project_chunk_vectors(&conn, row.id, &identity)?;
                let chunk_vec_ready = chunk_count == 0 || chunk_vec_count >= chunk_count;
                // A file named by an event is verified by content hash even though the
                // signature did not move: the project goes to the collector, which reads
                // and hashes exactly that file (`verify`), nothing else.
                if project_vec_ready && chunk_vec_ready && verify.is_empty() {
                    stats.skipped_projects += 1;
                    if !scan.complete() {
                        stats.projects_incomplete += 1;
                        stats.files_unreadable += scan.listing.unreadable as i64;
                    }
                    live_progress.mark_project_done();
                    docs_by_id.insert(
                        row.id,
                        ProjectDoc {
                            path: PathBuf::from(row.path.clone()),
                            title: row.title.clone(),
                            summary: row.summary.clone(),
                            mtime: row.project_mtime,
                        },
                    );
                    if emit_progress {
                        progress_clear_line();
                        println!(
                            "[{}/{}] skip {}",
                            idx + 1,
                            projects.len(),
                            project_label(project_dir)
                        );
                        render_live_progress_line(
                            reason,
                            projects.len(),
                            live_started_at,
                            &live_progress,
                        );
                    }
                    continue;
                }
                // Vectors are missing for chunks no file change would revisit: check every
                // file's chunks against the stored identities.
                if !(project_vec_ready && chunk_vec_ready) {
                    rechunk_all = true;
                }
                if emit_progress && !chunk_vec_ready && chunk_count > 0 {
                    progress_clear_line();
                    println!(
                        "[{}/{}] re-indexing {} (incomplete vectors: {}/{} chunks have embeddings)",
                        idx + 1,
                        projects.len(),
                        project_label(project_dir),
                        chunk_vec_count,
                        chunk_count,
                    );
                    render_live_progress_line(
                        reason,
                        projects.len(),
                        live_started_at,
                        &live_progress,
                    );
                }
            }
            if row.index_in_progress {
                // A previous run stopped part-way: every file goes through the identity check
                // (reads and hashes, no embedding where the stored vector matches).
                rechunk_all = true;
                if emit_progress {
                    progress_clear_line();
                    println!(
                        "[{}/{}] re-indexing {} (previous run was interrupted)",
                        idx + 1,
                        projects.len(),
                        project_label(project_dir)
                    );
                }
            }
            manifest = load_file_manifest(&conn, row.id)?;
        }
        needs_update.push(ProjectWork {
            idx,
            dir: project_dir.clone(),
            scan,
            manifest,
            rechunk_all,
            verify,
            existing,
        });
    }

    // ── Phase 2+3: Collect corpus + embed/store in bounded batches ──
    // Keep peak memory bounded by processing only INDEX_PARALLELISM corpora at once. A
    // project that fails anywhere (collector panic, embedding, sqlite, code intelligence)
    // keeps its previous state and is counted in `projects_failed`; an embedding failure
    // stops the run, since every later project would fail the same way.
    live_progress.set_phase("collect");
    let max_chars = cfg.max_chars_per_project as usize;
    const INDEX_PARALLELISM: usize = 4;
    let total = projects.len();
    let mut visited_work = 0usize;
    'batches: for work_batch in needs_update.chunks(INDEX_PARALLELISM) {
        let results = std::sync::Mutex::new(Vec::<(usize, ProjectCorpus)>::new());
        let mut panics: Vec<(usize, String)> = Vec::new();
        std::thread::scope(|s| {
            let handles: Vec<(usize, _)> = work_batch
                .iter()
                .map(|pw| {
                    let results = &results;
                    let caps = &caps;
                    let settings = &settings;
                    let handle = s.spawn(move || {
                        let corpus = collect_project_corpus_verifying(
                            &pw.dir,
                            &pw.scan,
                            caps,
                            max_chars,
                            &pw.manifest,
                            pw.rechunk_all,
                            &pw.verify,
                            settings,
                        );
                        if let Ok(mut vec) = results.lock() {
                            vec.push((pw.idx, corpus));
                        }
                    });
                    (pw.idx, handle)
                })
                .collect();
            for (idx, handle) in handles {
                if let Err(payload) = handle.join() {
                    panics.push((idx, panic_payload_message(payload.as_ref())));
                }
            }
        });
        let mut corpora: HashMap<usize, ProjectCorpus> = results
            .into_inner()
            .unwrap_or_default()
            .into_iter()
            .collect();

        for work in work_batch {
            visited_work += 1;
            let idx = work.idx;
            let project_dir = &work.dir;
            let project_name = project_label(project_dir);
            let existing_doc = work.existing.as_ref().map(|row| ProjectDoc {
                path: PathBuf::from(row.path.clone()),
                title: row.title.clone(),
                summary: row.summary.clone(),
                mtime: row.project_mtime,
            });
            let Some(corpus) = corpora.remove(&idx) else {
                let msg = panics
                    .iter()
                    .find(|(i, _)| *i == idx)
                    .map(|(_, m)| format!("collecting the project panicked: {}", m))
                    .unwrap_or_else(|| "collecting the project produced nothing".to_string());
                record_project_failure(&mut stats, idx, total, &project_name, &msg, emit_progress);
                if let (Some(row), Some(doc)) = (&work.existing, existing_doc) {
                    docs_by_id.insert(row.id, doc);
                }
                live_progress.mark_project_done();
                continue;
            };
            let project_path = corpus.doc.path.to_string_lossy().to_string();
            let project_total_tokens: u64 = corpus
                .chunks
                .iter()
                .map(|c| c.token_count.max(0) as u64)
                .sum();
            live_progress.start_project(
                idx + 1,
                &project_name,
                corpus.chunks.len(),
                project_total_tokens,
            );
            // Scan-level counters describe what the scan saw, whatever happens next.
            stats.files_selected += corpus.files.len() as i64;
            stats.files_unchanged += corpus.files_unchanged();
            stats.files_rechunked += corpus.files_rechunked();
            stats.files_unreadable += corpus.files_unreadable;
            stats.files_evicted_by_cap += corpus.files_evicted_by_cap;
            stats.files_truncated_by_cap += corpus.files_truncated_by_cap;
            stats.documents_extracted += corpus.documents_extracted;
            stats.documents_failed += corpus.documents_failed;
            if !corpus.complete {
                stats.projects_incomplete += 1;
            }
            if emit_progress && !corpus.caps_note.is_empty() {
                progress_clear_line();
                println!(
                    "[{}/{}] warning: {}: {}",
                    idx + 1,
                    total,
                    project_name,
                    corpus.caps_note
                );
            }
            if emit_progress && !corpus.document_failures.is_empty() {
                // Five per project, or every one with RETRIVIO_DOCUMENT_FAILURES=all.
                let shown: usize = if env::var("RETRIVIO_DOCUMENT_FAILURES")
                    .map(|v| v == "all")
                    .unwrap_or(false)
                {
                    usize::MAX
                } else {
                    5
                };
                progress_clear_line();
                for (rel, reason) in corpus.document_failures.iter().take(shown) {
                    println!(
                        "[{}/{}] warning: {}: document not extracted: {}: {} (previously indexed content, if any, is kept)",
                        idx + 1,
                        total,
                        project_name,
                        rel,
                        reason
                    );
                }
                if corpus.document_failures.len() > shown {
                    println!(
                        "[{}/{}] warning: {}: {} more documents not extracted (RETRIVIO_DOCUMENT_FAILURES=all lists every one)",
                        idx + 1,
                        total,
                        project_name,
                        corpus.document_failures.len() - shown
                    );
                }
            }

            let outcome = index_one_project(
                cwd,
                &conn,
                &caps,
                &identity,
                embedder,
                &corpus,
                project_dir,
                &project_path,
                work.existing.as_ref(),
                now,
                Some(&live_progress),
                force_embed,
            );
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(failure) => {
                    let (msg, stop) = match failure {
                        ProjectFailure::Project(msg) => (msg, false),
                        ProjectFailure::Embedding(msg) => (msg, true),
                    };
                    record_project_failure(
                        &mut stats,
                        idx,
                        total,
                        &project_name,
                        &msg,
                        emit_progress,
                    );
                    if let (Some(row), Some(doc)) = (&work.existing, existing_doc) {
                        docs_by_id.insert(row.id, doc);
                    }
                    live_progress.mark_project_done();
                    if stop {
                        let left = needs_update.len() - visited_work;
                        stats.stopped = format!(
                            "embedding failed while indexing {} ({}); {} project(s) not visited this run",
                            project_name, msg, left
                        );
                        if emit_progress {
                            progress_clear_line();
                            println!("error: {}", stats.stopped);
                        }
                        // Unvisited projects keep their rows and edges as they are.
                        for pending in &needs_update[visited_work..] {
                            if let Some(row) = &pending.existing {
                                docs_by_id.insert(
                                    row.id,
                                    ProjectDoc {
                                        path: PathBuf::from(row.path.clone()),
                                        title: row.title.clone(),
                                        summary: row.summary.clone(),
                                        mtime: row.project_mtime,
                                    },
                                );
                            }
                        }
                        break 'batches;
                    }
                    continue;
                }
            };

            stats.updated_projects += 1;
            stats.chunk_rows += outcome.rows;
            stats.chunk_vectors += outcome.vectors;
            stats.chunks_embedded += outcome.vectors;
            stats.chunks_reused += outcome.reused;
            stats.pruned_chunks += outcome.pruned.chunks as i64;
            stats.pruned_files += outcome.pruned.files as i64;
            stats.chunks_deleted += outcome.pruned.chunks as i64;
            if outcome.summary_embedded {
                stats.vectorized_projects += 1;
            }
            if emit_progress && outcome.pruned.chunks > 0 {
                progress_clear_line();
                println!(
                    "[{}/{}] pruned {} chunks from {} files in {}",
                    idx + 1,
                    total,
                    outcome.pruned.chunks,
                    outcome.pruned.files,
                    project_name
                );
            }
            docs_by_id.insert(
                outcome.project_id,
                ProjectDoc {
                    path: corpus.doc.path.clone(),
                    title: outcome.title,
                    summary: outcome.summary,
                    mtime: corpus.doc.mtime,
                },
            );

            if emit_progress {
                progress_clear_line();
                println!(
                    "[{}/{}] index {} files={} (unchanged={}, rechunked={}) chunks={} embedded={} reused={} deleted={} chunk_vecs={} vector_failures={}{}",
                    idx + 1,
                    total,
                    project_name,
                    corpus.files.len(),
                    corpus.files_unchanged(),
                    corpus.files_rechunked(),
                    outcome.rows,
                    outcome.vectors,
                    outcome.reused,
                    outcome.pruned.chunks,
                    outcome.vectors,
                    0,
                    if corpus.complete {
                        String::new()
                    } else {
                        format!(" unreadable={} (incomplete)", corpus.files_unreadable)
                    }
                );
                render_live_progress_line(reason, total, live_started_at, &live_progress);
            }
            live_progress.mark_project_done();
            if emit_progress {
                render_live_progress_line(reason, total, live_started_at, &live_progress);
            }
        }
    }

    // A run that stopped early or lost a project has not seen everything: it neither removes
    // projects nor advances the fingerprints (the next run revisits under the same rules).
    let run_ok = stats.projects_failed == 0 && stats.stopped.is_empty();
    if remove_missing && stats.stopped.is_empty() {
        live_progress.set_phase("cleanup");
        // Projects under a tracked root that cannot be listed right now, or whose listing
        // was incomplete, are not "missing".
        let mut protected = unavailable_roots(&roots);
        protected.extend(incomplete_roots.iter().cloned());
        if emit_progress {
            for root in &protected {
                progress_clear_line();
                println!(
                    "note: tracked root {} is not fully readable now; its projects are left alone",
                    root.display()
                );
            }
        }
        let (removed_projects, removed_chunks) =
            remove_projects_not_in(&conn, &keep_paths, &protected)?;
        stats.removed_projects = removed_projects;
        stats.chunks_deleted += removed_chunks;
    }
    // The fingerprint advances only when every project was revisited under it: a complete
    // all-roots run with no incomplete scan and no failed project.
    if fingerprint_changed
        && scope == IndexScope::AllRoots
        && stats.projects_incomplete == 0
        && run_ok
    {
        // The dimension may have become known only during this run (first vectors).
        if let Some(fp) = EmbedIdentity::for_run(&conn, embedder).fingerprint() {
            if stored_fingerprint.as_deref() != Some(fp.as_str()) {
                app_state_set(&conn, APP_STATE_EMBED_FINGERPRINT, &fp)?;
            }
        }
    }
    if caps_changed && scope == IndexScope::AllRoots && stats.projects_incomplete == 0 && run_ok {
        app_state_set(&conn, APP_STATE_SCAN_CAPS_FINGERPRINT, &current_caps)?;
    }
    if LANCE_WRITE_FAILED.load(Ordering::SeqCst) && stats.lance_error.is_empty() {
        stats.lance_error = "LanceDB writes failed during this run".to_string();
    }
    live_progress.set_phase("graph-edges");
    stats.graph_edges = rebuild_relationship_edges(&conn, &docs_by_id)?;
    live_progress.set_phase("finalize");
    let t_before_sync = Instant::now();
    stats.elapsed_index_ms = t_before_sync.duration_since(t_start).as_millis() as u64;
    // LanceDB vectors are written inline during reindex_project_chunks;
    // report count for diagnostics.
    if let Ok(n) = with_lance_store(|store| lance_store::count(store)) {
        stats.retrieval_synced_chunks = n as i64;
    }
    stats.elapsed_sync_ms = t_before_sync.elapsed().as_millis() as u64;
    if !app_state_bool(&conn, APP_STATE_REEMBED_REQUIRED)? {
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &model_key)?;
    }

    conn.execute_batch("PRAGMA optimize;")
        .map_err(|e| format!("database optimize failed: {}", e))?;
    stats.elapsed_total_ms = t_start.elapsed().as_millis() as u64;
    Ok(stats)
}

pub(crate) fn print_index_stats(stats: &IndexStats, cfg: &ConfigValues) {
    println!("config root: {}", cfg.root.display());
    println!("tracked roots indexed: {}", stats.tracked_roots);
    println!("projects found: {}", stats.total_projects);
    println!("projects updated: {}", stats.updated_projects);
    println!("projects skipped (unchanged): {}", stats.skipped_projects);
    println!("projects removed: {}", stats.removed_projects);
    println!("vectors refreshed: {}", stats.vectorized_projects);
    println!(
        "files selected: {} (unchanged {}, rechunked {})",
        stats.files_selected, stats.files_unchanged, stats.files_rechunked
    );
    println!(
        "files unreadable: {} (projects incomplete: {})",
        stats.files_unreadable, stats.projects_incomplete
    );
    println!(
        "files evicted by caps: {} (truncated: {})",
        stats.files_evicted_by_cap, stats.files_truncated_by_cap
    );
    println!("chunks indexed: {}", stats.chunk_rows);
    println!("chunk vectors refreshed: {}", stats.chunk_vectors);
    println!(
        "chunks embedded: {}, reused: {}, deleted: {}",
        stats.chunks_embedded, stats.chunks_reused, stats.chunks_deleted
    );
    println!(
        "stale chunks pruned: {} (from {} files)",
        stats.pruned_chunks, stats.pruned_files
    );
    println!(
        "lance repaired: {} (orphans removed: {})",
        stats.lance_repaired, stats.lance_orphans_removed
    );
    println!(
        "documents extracted: {} (failed: {})",
        stats.documents_extracted, stats.documents_failed
    );
    if stats.projects_failed > 0 {
        println!(
            "projects failed: {} (each keeps its previous state and is retried next run)",
            stats.projects_failed
        );
        for line in &stats.failures {
            println!("  - {}", line);
        }
    }
    if !stats.stopped.is_empty() {
        println!("run stopped early: {}", stats.stopped);
    }
    if !stats.lance_error.is_empty() {
        println!(
            "lancedb: {} (dirty marker set; the next index run repairs LanceDB from sqlite)",
            stats.lance_error
        );
    }
    println!("graph edges refreshed: {}", stats.graph_edges);
    println!("retrieval backend: {}", stats.retrieval_backend);
    println!("retrieval chunks synced: {}", stats.retrieval_synced_chunks);
    if !stats.retrieval_error.is_empty() {
        println!("retrieval sync warning: {}", stats.retrieval_error);
    }
    if stats.vector_failures > 0 {
        println!("vector failures: {}", stats.vector_failures);
    }
    println!(
        "timing: index={}, sync={}, total={}",
        format_duration_ms(stats.elapsed_index_ms),
        format_duration_ms(stats.elapsed_sync_ms),
        format_duration_ms(stats.elapsed_total_ms),
    );
}

pub(crate) fn get_project_by_path(
    conn: &Connection,
    path: &str,
) -> Result<Option<ExistingProject>, String> {
    conn.query_row(
        "SELECT id, path, title, summary, project_mtime, scan_signature, index_in_progress FROM projects WHERE path = ?1",
        params![path],
        |row| {
            Ok(ExistingProject {
                id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                summary: row.get(3)?,
                project_mtime: row.get(4)?,
                scan_signature: row.get(5)?,
                index_in_progress: row.get::<_, i64>(6)? != 0,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed fetching project row: {}", e))
}

/// Start writing a project: return its row id (inserting a placeholder for a new project) with
/// `index_in_progress` set. Title, summary, mtime and the scan signature of an existing row are
/// left as they were: a run that fails after this leaves the project looking exactly as
/// before, plus the marker, so the next run rescans it.
pub(crate) fn begin_project_update(
    conn: &Connection,
    path: &str,
    title: &str,
) -> Result<i64, String> {
    conn.execute(
        r#"
INSERT INTO projects(path, title, summary, project_mtime, last_indexed, scan_signature, index_in_progress)
VALUES (?1, ?2, '', 0, 0, '', 1)
ON CONFLICT(path) DO UPDATE SET index_in_progress = 1
"#,
        params![path, title],
    )
    .map_err(|e| format!("failed starting project update: {}", e))?;
    conn.query_row(
        "SELECT id FROM projects WHERE path = ?1",
        params![path],
        |row| row.get(0),
    )
    .map_err(|e| format!("failed reading project id: {}", e))
}

/// Publish the project row after every step of its run succeeded, inside the caller's publish
/// transaction (`tx`): the summary vector (when a new one was embedded), title, summary,
/// mtime, `last_indexed`, the scan signature (only for a complete scan; an incomplete one
/// keeps the old signature so the project is rescanned) and `index_in_progress = 0`.
#[allow(clippy::too_many_arguments)] // one project-row publish, one argument per column
pub(crate) fn finalize_project_row_in(
    tx: &Connection,
    project_id: i64,
    title: &str,
    summary: &str,
    project_mtime: f64,
    last_indexed: f64,
    scan_signature: Option<&str>,
    summary_vector: Option<(&EmbedIdentity, &[f32])>,
) -> Result<(), String> {
    if let Some((identity, vector)) = summary_vector {
        set_project_vector(tx, project_id, identity, vector)?;
    }
    match scan_signature {
        Some(signature) => tx.execute(
            r#"
UPDATE projects
SET title = ?1, summary = ?2, project_mtime = ?3, last_indexed = ?4, scan_signature = ?5,
    index_in_progress = 0
WHERE id = ?6
"#,
            params![
                title,
                summary,
                project_mtime,
                last_indexed,
                signature,
                project_id
            ],
        ),
        None => tx.execute(
            r#"
UPDATE projects
SET title = ?1, summary = ?2, project_mtime = ?3, last_indexed = ?4, index_in_progress = 0
WHERE id = ?5
"#,
            params![title, summary, project_mtime, last_indexed, project_id],
        ),
    }
    .map_err(|e| format!("failed publishing project row: {}", e))?;
    Ok(())
}

/// True when the project's summary vector exists and was produced under exactly `identity`
/// (model, dimension, normalisation, pipeline version).
pub(crate) fn project_vector_matches(
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
) -> Result<bool, String> {
    let row: Option<(String, i64, i64, i64)> = conn
        .query_row(
            "SELECT model, dim, normalized, pipeline_version FROM project_vectors WHERE project_id = ?1",
            params![project_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()
        .map_err(|e| format!("failed checking project vector: {}", e))?;
    Ok(match row {
        Some((model, dim, normalized, pipeline_version)) => {
            model == identity.model
                && identity.dim == Some(dim)
                && (normalized != 0) == identity.normalized
                && pipeline_version == identity.pipeline_version
        }
        None => false,
    })
}

pub(crate) fn count_project_chunks(conn: &Connection, project_id: i64) -> Result<i64, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM project_chunks WHERE project_id = ?1",
        params![project_id],
        |row| row.get(0),
    )
    .map_err(|e| format!("failed counting project chunks: {}", e))
}

/// Chunk vectors of the project that match `identity` exactly. With no known dimension (no
/// vectors for the model yet) nothing can match.
pub(crate) fn count_project_chunk_vectors(
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
) -> Result<i64, String> {
    let Some(dim) = identity.dim else {
        return Ok(0);
    };
    conn.query_row(
        r#"
SELECT COUNT(*)
FROM project_chunks pc
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pc.project_id = ?1 AND pcv.model = ?2 AND pcv.dim = ?3
  AND pcv.normalized = ?4 AND pcv.pipeline_version = ?5
"#,
        params![
            project_id,
            identity.model,
            dim,
            identity.normalized as i64,
            identity.pipeline_version
        ],
        |row| row.get(0),
    )
    .map_err(|e| format!("failed counting chunk vectors: {}", e))
}

/// Returns (project_name, vectors_present, total_chunks) for projects with incomplete vectors.
pub(crate) fn count_incomplete_vector_projects(
    conn: &Connection,
    model: &str,
) -> Result<Vec<(String, i64, i64)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT p.path,
       (SELECT COUNT(*) FROM project_chunks pc WHERE pc.project_id = p.id) AS chunk_count,
       (SELECT COUNT(*) FROM project_chunks pc
        JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
        WHERE pc.project_id = p.id AND pcv.model = ?1) AS vec_count
FROM projects p
HAVING chunk_count > 0 AND vec_count < chunk_count
"#,
        )
        .map_err(|e| format!("failed preparing incomplete vector query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let path: String = row.get(0)?;
            let chunk_count: i64 = row.get(1)?;
            let vec_count: i64 = row.get(2)?;
            Ok((path, vec_count, chunk_count))
        })
        .map_err(|e| format!("failed querying incomplete vectors: {}", e))?;
    let mut result = Vec::new();
    for row in rows {
        let (path, vec_count, chunk_count) =
            row.map_err(|e| format!("failed reading incomplete vector row: {}", e))?;
        let name = Path::new(&path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&path)
            .to_string();
        result.push((name, vec_count, chunk_count));
    }
    Ok(result)
}

pub(crate) fn set_project_vector(
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_vectors(project_id, model, dim, norm, vector, normalized, pipeline_version)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT(project_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector,
    normalized = excluded.normalized,
    pipeline_version = excluded.pipeline_version
"#,
        params![
            project_id,
            identity.model,
            vector.len() as i64,
            norm,
            blob,
            identity.normalized as i64,
            identity.pipeline_version
        ],
    )
    .map_err(|e| format!("failed upserting project vector: {}", e))?;
    Ok(())
}

/// Store extracted symbols for a project. Clears existing symbols first.
pub(crate) fn store_project_symbols(
    conn: &Connection,
    project_id: i64,
    symbols: &[code_intel::ExtractedSymbol],
    doc_path: &str,
    doc_rel_path: &str,
) -> Result<usize, String> {
    let now = now_ts();
    // Clear existing symbols for this doc
    conn.execute(
        "DELETE FROM symbols WHERE project_id = ?1 AND doc_path = ?2",
        params![project_id, doc_path],
    )
    .map_err(|e| format!("failed clearing symbols: {}", e))?;

    let mut count = 0;
    for sym in symbols {
        conn.execute(
            r#"
INSERT INTO symbols(project_id, doc_path, doc_rel_path, name, qualified_name, kind,
    line_start, line_end, signature, doc_comment, visibility, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
"#,
            params![
                project_id,
                doc_path,
                doc_rel_path,
                sym.name,
                sym.qualified_name,
                sym.kind,
                sym.line_start as i64,
                sym.line_end as i64,
                sym.signature,
                sym.doc_comment,
                sym.visibility,
                now
            ],
        )
        .map_err(|e| format!("failed inserting symbol: {}", e))?;
        count += 1;
    }
    Ok(count)
}

/// Store extracted imports for a source file, resolve them to file paths,
/// and create file_dependency_edges.
#[allow(clippy::too_many_arguments)] // one file's import resolution: source identity, language, roots and the project file list
pub(crate) fn store_file_imports(
    conn: &Connection,
    project_id: i64,
    source_doc_path: &str,
    lang: code_intel::LanguageId,
    source_file: &Path,
    project_root: &Path,
    raw_imports: &[code_intel::RawImport],
    project_files: &[String],
) -> Result<usize, String> {
    let now = now_ts();

    // Clear existing imports for this source file
    conn.execute(
        "DELETE FROM file_imports WHERE project_id = ?1 AND source_doc_path = ?2",
        params![project_id, source_doc_path],
    )
    .map_err(|e| format!("failed clearing file imports: {}", e))?;

    // Clear existing dependency edges from this source
    conn.execute(
        "DELETE FROM file_dependency_edges WHERE project_id = ?1 AND source_doc_path = ?2",
        params![project_id, source_doc_path],
    )
    .map_err(|e| format!("failed clearing file dep edges: {}", e))?;

    let mut count = 0;
    let mut resolved_targets: HashMap<String, usize> = HashMap::new(); // target_path -> import_count

    for imp in raw_imports {
        let resolved = code_intel::resolve_import_path(
            lang,
            &imp.raw_specifier,
            source_file,
            project_root,
            project_files,
        );

        let names_joined = imp.imported_names.join(", ");

        conn.execute(
            r#"
INSERT INTO file_imports(project_id, source_doc_path, import_kind, raw_specifier,
    resolved_path, imported_names, line_number, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
"#,
            params![
                project_id,
                source_doc_path,
                imp.import_kind,
                imp.raw_specifier,
                resolved,
                names_joined,
                imp.line_number as i64,
                now
            ],
        )
        .map_err(|e| format!("failed inserting file import: {}", e))?;
        count += 1;

        // Track resolved targets for dependency edge creation
        if !resolved.is_empty() {
            let entry = resolved_targets.entry(resolved).or_insert(0);
            *entry += imp.imported_names.len().max(1);
        }
    }

    // Create file_dependency_edges from resolved imports
    for (target_path, symbol_count) in &resolved_targets {
        let weight = match *symbol_count {
            1 => 0.5,
            2..=5 => 0.7,
            _ => 0.9,
        };
        conn.execute(
            r#"
INSERT OR REPLACE INTO file_dependency_edges(
    project_id, source_doc_path, target_doc_path, edge_kind, weight,
    imported_symbol_count, updated_at)
VALUES (?1, ?2, ?3, 'imports', ?4, ?5, ?6)
"#,
            params![
                project_id,
                source_doc_path,
                target_path,
                weight,
                *symbol_count as i64,
                now
            ],
        )
        .map_err(|e| format!("failed inserting file dep edge: {}", e))?;
    }

    Ok(count)
}

pub(crate) fn upsert_project_chunk(
    conn: &Connection,
    project_id: i64,
    chunk: &ProjectChunk,
    updated_at: f64,
) -> Result<i64, String> {
    let existing: Option<(i64, String)> = conn
        .query_row(
            r#"
SELECT id, text_hash
FROM project_chunks
WHERE project_id = ?1 AND doc_path = ?2 AND chunk_index = ?3
"#,
            params![project_id, chunk.doc_path, chunk.chunk_index],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|e| format!("failed checking project chunk row: {}", e))?;
    if let Some((chunk_id, old_hash)) = existing {
        // The row keeps its id (its LanceDB vector is updated in place), but relation
        // feedback describes the old text: drop it when the content changed.
        if old_hash != chunk.text_hash {
            conn.execute(
                "DELETE FROM chunk_relation_feedback WHERE src_chunk_id = ?1 OR dst_chunk_id = ?1",
                params![chunk_id],
            )
            .map_err(|e| format!("failed clearing stale chunk relation feedback: {}", e))?;
        }
        conn.execute(
            r#"
UPDATE project_chunks
SET doc_rel_path = ?1, doc_mtime = ?2, token_count = ?3, text_hash = ?4, text = ?5, updated_at = ?6,
    chunk_kind = ?7, symbol_name = ?8, parent_context = ?9, line_start = ?10, line_end = ?11, context_header = ?12
WHERE id = ?13
"#,
            params![
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at,
                chunk.chunk_kind,
                chunk.symbol_name,
                chunk.parent_context,
                chunk.line_start,
                chunk.line_end,
                chunk.context_header,
                chunk_id
            ],
        )
        .map_err(|e| format!("failed updating project chunk row: {}", e))?;
        Ok(chunk_id)
    } else {
        conn.execute(
            r#"
INSERT INTO project_chunks(
    project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at,
    chunk_kind, symbol_name, parent_context, line_start, line_end, context_header
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
"#,
            params![
                project_id,
                chunk.doc_path,
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.chunk_index,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at,
                chunk.chunk_kind,
                chunk.symbol_name,
                chunk.parent_context,
                chunk.line_start,
                chunk.line_end,
                chunk.context_header
            ],
        )
        .map_err(|e| format!("failed inserting project chunk row: {}", e))?;
        Ok(conn.last_insert_rowid())
    }
}

// ── Embedding identity ────────────────────────────────────────────────────────
//
// A stored chunk vector is reused instead of re-embedded only when everything that shaped it
// still holds: the exact embedder input (context header + text), the model, the vector
// dimension, the normalisation setting and the chunking pipeline version. Text hash alone is
// not enough: the same text under a different header, model or pipeline is a different vector.

/// The exact string handed to the embedder for a chunk: context header, newline, chunk text;
/// the bare text when there is no header.
pub(crate) fn embed_input_for(context_header: &str, text: &str) -> String {
    if context_header.is_empty() {
        text.to_string()
    } else {
        format!("{}\n{}", context_header, text)
    }
}

/// SHA-1 hex digest of an embedder input string.
pub(crate) fn embed_input_hash(input: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// What the current run writes on every new vector and requires of a stored one to reuse it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmbedIdentity {
    model: String,
    /// Dimension of this model's stored vectors. `None` when none exist yet, in which case
    /// nothing can be reused and the first embedded batch fixes the dimension.
    dim: Option<i64>,
    normalized: bool,
    pipeline_version: i64,
}

impl EmbedIdentity {
    fn for_run(conn: &Connection, embedder: &dyn Embedder) -> Self {
        let model = embedder.model_key();
        let dim = vector_dim_from_sqlite(conn, &model).map(|d| d as i64);
        EmbedIdentity {
            model,
            dim,
            normalized: embedder.normalizes_output(),
            pipeline_version: EMBEDDING_PIPELINE_VERSION,
        }
    }

    /// True when `stored` was produced from `input_hash` under exactly this identity.
    fn reuses(&self, stored: &StoredVectorIdentity, input_hash: &str) -> bool {
        !input_hash.is_empty()
            && stored.embed_input_hash == input_hash
            && stored.model == self.model
            && self.dim == Some(stored.dim)
            && stored.normalized == self.normalized
            && stored.pipeline_version == self.pipeline_version
    }

    /// `model|dim|normalized|pipeline_version`, the store-wide fingerprint kept in app_state.
    /// `None` until the dimension is known (no vectors for the model yet).
    fn fingerprint(&self) -> Option<String> {
        self.dim.map(|dim| {
            format!(
                "{}|{}|{}|{}",
                self.model, dim, self.normalized as u8, self.pipeline_version
            )
        })
    }
}

/// The identity columns of one `project_chunk_vectors` row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StoredVectorIdentity {
    embed_input_hash: String,
    model: String,
    dim: i64,
    normalized: bool,
    pipeline_version: i64,
}

/// Identity of every stored vector of one project, keyed by chunk id.
#[cfg(test)]
pub(crate) fn load_project_vector_identities(
    conn: &Connection,
    project_id: i64,
) -> Result<HashMap<i64, StoredVectorIdentity>, String> {
    let rows = load_project_vector_identity_rows(conn, project_id)?;
    Ok(rows
        .into_iter()
        .map(|(id, _, _, identity)| (id, identity))
        .collect())
}

/// Identity of every stored vector of one project, keyed by the chunk row's position
/// (`doc_path`, `chunk_index`), the key an upsert of the same chunk resolves to.
pub(crate) fn load_project_vector_identities_by_position(
    conn: &Connection,
    project_id: i64,
) -> Result<HashMap<(String, i64), StoredVectorIdentity>, String> {
    let rows = load_project_vector_identity_rows(conn, project_id)?;
    Ok(rows
        .into_iter()
        .map(|(_, doc_path, chunk_index, identity)| ((doc_path, chunk_index), identity))
        .collect())
}

pub(crate) fn load_project_vector_identity_rows(
    conn: &Connection,
    project_id: i64,
) -> Result<Vec<(i64, String, i64, StoredVectorIdentity)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT v.chunk_id, c.doc_path, c.chunk_index, v.embed_input_hash, v.model, v.dim, v.normalized, v.pipeline_version
FROM project_chunk_vectors v
JOIN project_chunks c ON c.id = v.chunk_id
WHERE c.project_id = ?1
"#,
        )
        .map_err(|e| format!("failed preparing vector identity query: {}", e))?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                StoredVectorIdentity {
                    embed_input_hash: row.get::<_, String>(3)?,
                    model: row.get::<_, String>(4)?,
                    dim: row.get::<_, i64>(5)?,
                    normalized: row.get::<_, i64>(6)? != 0,
                    pipeline_version: row.get::<_, i64>(7)?,
                },
            ))
        })
        .map_err(|e| format!("failed querying vector identities: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading vector identity: {}", e))?);
    }
    Ok(out)
}

pub(crate) fn set_project_chunk_vector(
    conn: &Connection,
    chunk_id: i64,
    identity: &EmbedIdentity,
    embed_input_hash: &str,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector, embed_input_hash, normalized, pipeline_version)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT(chunk_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector,
    embed_input_hash = excluded.embed_input_hash,
    normalized = excluded.normalized,
    pipeline_version = excluded.pipeline_version
"#,
        params![
            chunk_id,
            identity.model,
            vector.len() as i64,
            norm,
            blob,
            embed_input_hash,
            identity.normalized as i64,
            identity.pipeline_version
        ],
    )
    .map_err(|e| format!("failed upserting chunk vector: {}", e))?;
    Ok(())
}

/// Embed `texts` with retries; the vectors come back in order, all of one non-zero width.
pub(crate) fn embed_batch_with_retry(
    embedder: &dyn Embedder,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    const BATCH_MAX_RETRIES: usize = 3;
    let mut last_err = String::new();
    for attempt in 0..=BATCH_MAX_RETRIES {
        if attempt > 0 {
            let backoff_ms = 500u64 * 2u64.pow((attempt - 1) as u32);
            thread::sleep(Duration::from_millis(backoff_ms));
        }
        match embedder.embed_many(texts) {
            Ok(vectors) => {
                if vectors.len() != texts.len() {
                    last_err = format!(
                        "embed_many returned {} vectors for {} chunks",
                        vectors.len(),
                        texts.len()
                    );
                    continue;
                }
                let batch_dim = vectors.first().map(|v| v.len()).unwrap_or(0);
                if batch_dim == 0 {
                    last_err = "embed_many returned empty vectors".to_string();
                    continue;
                }
                if vectors.iter().any(|v| v.len() != batch_dim) {
                    return Err(format!(
                        "embed_many returned mixed vector dimensions in a single batch (expected {})",
                        batch_dim
                    ));
                }
                return Ok(vectors);
            }
            Err(e) => {
                last_err = e;
            }
        }
    }
    Err(format!(
        "embedding batch failed after {} retries: {}",
        BATCH_MAX_RETRIES, last_err
    ))
}

/// Which of a project's chunks keep their stored vector and which are embedded (with the
/// exact embedder input), decided before anything is written. A chunk whose stored vector
/// still matches the exact embedder input is reused (its row is rewritten with the same
/// text); every other chunk is embedded. `reembed` is the force path: it ignores stored
/// identities and embeds every chunk of the work set.
pub(crate) struct ChunkPlan<'a> {
    reuse: Vec<&'a ProjectChunk>,
    embed: Vec<(&'a ProjectChunk, String)>,
}

pub(crate) fn plan_chunk_reuse<'a>(
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
    corpus: &'a ProjectCorpus,
    force_embed: bool,
) -> Result<ChunkPlan<'a>, String> {
    let stored: HashMap<(String, i64), StoredVectorIdentity> = if force_embed {
        HashMap::new()
    } else {
        load_project_vector_identities_by_position(conn, project_id)?
    };
    let mut plan = ChunkPlan {
        reuse: Vec::new(),
        embed: Vec::new(),
    };
    for chunk in &corpus.chunks {
        // The embedder sees the context header (file path, parent type, symbol name) above
        // the chunk text; that exact string is what the stored identity hashes.
        let embed_text = embed_input_for(&chunk.context_header, &chunk.text);
        let reusable = stored
            .get(&(chunk.doc_path.clone(), chunk.chunk_index))
            .map(|existing| identity.reuses(existing, &embed_input_hash(&embed_text)))
            .unwrap_or(false);
        if reusable {
            plan.reuse.push(chunk);
        } else {
            plan.embed.push((chunk, embed_text));
        }
    }
    Ok(plan)
}

/// A chunk embedded this run, held in memory until the project is published: the chunk, the
/// exact input its identity hashes, the vector (about 4 KB at 1024 dimensions; a project holds
/// at most `max_chunks_per_project` of them).
pub(crate) struct EmbeddedChunk<'a> {
    chunk: &'a ProjectChunk,
    input: String,
    vector: Vec<f32>,
}

/// Embed the plan's chunks in batches of 512 with no sqlite transaction open. All or nothing
/// for the project: a failed batch fails the project before anything is written.
pub(crate) fn embed_planned_chunks<'a>(
    embedder: &dyn Embedder,
    to_embed: &[(&'a ProjectChunk, String)],
    live_progress: Option<&Arc<LiveIndexProgress>>,
) -> Result<Vec<EmbeddedChunk<'a>>, String> {
    const CHUNK_EMBED_BATCH: usize = 512;
    let mut out: Vec<EmbeddedChunk<'a>> = Vec::with_capacity(to_embed.len());
    for batch in to_embed.chunks(CHUNK_EMBED_BATCH) {
        let texts: Vec<String> = batch.iter().map(|(_, text)| text.clone()).collect();
        let vectors = embed_batch_with_retry(embedder, &texts)?;
        for (((chunk, input), vector), _) in batch.iter().zip(vectors).zip(0..) {
            out.push(EmbeddedChunk {
                chunk,
                input: input.clone(),
                vector,
            });
        }
        if let Some(lp) = live_progress {
            lp.add_chunks_done(batch.len());
            lp.add_tokens_done(batch.iter().map(|(c, _)| c.token_count.max(0) as u64).sum());
        }
    }
    Ok(out)
}

/// Symbols and imports of one re-chunked code file, extracted (tree-sitter) before the
/// publish transaction so the write lock is never held while parsing.
pub(crate) struct CodeIntelExtraction {
    lang: code_intel::LanguageId,
    doc_path: String,
    rel_path: String,
    symbols: Vec<code_intel::ExtractedSymbol>,
    imports: Vec<code_intel::RawImport>,
}

/// Extract symbols and imports for every code file re-chunked this run. Replaced
/// unconditionally at publish time, so a file edited down to zero symbols or imports loses
/// its stale rows; unchanged files keep theirs. A file with no indexable text (whitespace
/// only) yields the empty extraction. Files are read through the project directory (follows
/// symlinks) and keyed by their derived doc_path.
pub(crate) fn extract_project_code_intel(
    project_dir: &Path,
    corpus: &ProjectCorpus,
    caps: &ScanCaps,
) -> Vec<CodeIntelExtraction> {
    let mut out = Vec::new();
    for file in corpus.files.iter().filter(|f| f.rechunked) {
        let path = Path::new(&file.doc_path);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let Some(lang) = code_intel::language_for_extension(&ext) else {
            continue;
        };
        let source = read_for_index(&project_dir.join(&file.rel_path), caps.max_file_chars)
            .map(|(s, _)| s)
            .unwrap_or_default();
        out.push(extract_code_intel(
            lang,
            &file.doc_path,
            &file.rel_path,
            &source,
        ));
    }
    out
}

/// The parse half of a code file's code-intelligence refresh: symbols and imports of
/// `source`, keyed by the file's derived doc_path.
pub(crate) fn extract_code_intel(
    lang: code_intel::LanguageId,
    doc_path: &str,
    rel_path: &str,
    source: &str,
) -> CodeIntelExtraction {
    let path = Path::new(doc_path);
    CodeIntelExtraction {
        lang,
        doc_path: doc_path.to_string(),
        rel_path: rel_path.to_string(),
        symbols: code_intel::extract_symbols(path, source),
        imports: code_intel::extract_imports(path, source),
    }
}

/// The store half: replace the file's symbol, import and dependency-edge rows with the
/// extraction, unconditionally, so an extraction that came back empty removes the stale rows
/// a previous version of the file left behind. Returns (symbols, imports) stored.
pub(crate) fn store_code_intel_extraction(
    conn: &Connection,
    project_id: i64,
    x: &CodeIntelExtraction,
    project_root: &Path,
    project_files: &[String],
) -> Result<(usize, usize), String> {
    let stored_symbols =
        store_project_symbols(conn, project_id, &x.symbols, &x.doc_path, &x.rel_path)?;
    let stored_imports = store_file_imports(
        conn,
        project_id,
        &x.doc_path,
        x.lang,
        Path::new(&x.doc_path),
        project_root,
        &x.imports,
        project_files,
    )?;
    Ok((stored_symbols, stored_imports))
}

/// The project row's published state, written in the publish transaction.
pub(crate) struct ProjectPublication<'a> {
    title: &'a str,
    summary: &'a str,
    project_mtime: f64,
    /// The scan signature to store; `None` for an incomplete scan (the old one stays).
    scan_signature: Option<&'a str>,
    /// A freshly embedded summary vector, or `None` when the stored one is reused.
    summary_vector: Option<&'a [f32]>,
}

/// What [`publish_project`] committed, for the LanceDB write that follows.
pub(crate) struct Published {
    /// Chunk ids of the rows embedded this run, in `lance_batch` order.
    embedded_ids: Vec<i64>,
    lance_batch: Vec<(i64, Vec<f32>)>,
    pruned: PruneOutcome,
}

/// Publish one project in ONE sqlite transaction: the rows of reused chunks, the rows and
/// vectors of embedded chunks (identity hashed from the very input sent), the Lance dirty
/// marker and pending ids for those vectors, the prune of rows the scan no longer covers
/// (complete scans only; the FULL keep set, so unchanged files never lose anything), the
/// manifest rows of re-chunked and touched files, the symbols and imports of re-chunked code
/// files, and (when `publication` is given) the project row itself: title, summary, mtime,
/// signature, summary vector and the cleared in-progress marker.
///
/// A reader therefore sees the project either exactly as it was or exactly as it is now; a
/// failure anywhere rolls everything back and the previous state stands. No embedding and no
/// parsing happens inside: the write lock is held for the writes alone.
#[allow(clippy::too_many_arguments)] // the whole publish transaction: reuse set, embedded chunks, code intelligence and the row
pub(crate) fn publish_project(
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
    corpus: &ProjectCorpus,
    reuse: &[&ProjectChunk],
    embedded: Vec<EmbeddedChunk<'_>>,
    code_intel: &[CodeIntelExtraction],
    publication: Option<ProjectPublication<'_>>,
    now: f64,
) -> Result<Published, String> {
    let tx = conn
        .unchecked_transaction()
        .map_err(|e| format!("failed starting project publish transaction: {}", e))?;

    for chunk in reuse {
        upsert_project_chunk(&tx, project_id, chunk, now)?;
    }
    let mut embedded_ids: Vec<i64> = Vec::with_capacity(embedded.len());
    let mut lance_batch: Vec<(i64, Vec<f32>)> = Vec::with_capacity(embedded.len());
    for item in embedded {
        let chunk_id = upsert_project_chunk(&tx, project_id, item.chunk, now)?;
        set_project_chunk_vector(
            &tx,
            chunk_id,
            identity,
            &embed_input_hash(&item.input),
            &item.vector,
        )?;
        embedded_ids.push(chunk_id);
        lance_batch.push((chunk_id, item.vector));
    }
    if !embedded_ids.is_empty() {
        // The marker and the pending ids commit with the rows and vectors: a crash or a
        // failed LanceDB write leaves both, and the repair step rewrites exactly these rows
        // (a stale same-id row in LanceDB is not "present", it is pending).
        lance_mark_dirty(&tx)?;
        lance_pending_add(&tx, &embedded_ids)?;
    }

    // Drop rows the scan no longer accounts for (deleted files, newly excluded or skipped
    // directories, files past the caps, chunk indices past the end of a file that shrank).
    // Only for a complete scan: whatever an unreadable directory hid is not deleted. The
    // marker is set inside this transaction when LanceDB rows are affected.
    let pruned = if corpus.complete {
        let keep = PruneKeepSet::from_corpus(corpus);
        prune_stale_project_rows_in(&tx, project_id, &keep, false)?
    } else {
        PruneOutcome::default()
    };

    // Manifest rows for the files re-chunked this run and for touched-but-identical files
    // (new stat, same content); unchanged and failed-document files keep theirs.
    for file in corpus
        .files
        .iter()
        .filter(|f| f.rechunked || f.stat_changed)
    {
        upsert_file_manifest(
            &tx,
            project_id,
            &file.rel_path,
            &file.doc_path,
            file.size,
            file.mtime,
            &file.content_hash,
            file.chunk_count,
        )?;
    }

    // Symbols and imports. Import resolution sees every selected file, unchanged included.
    if !code_intel.is_empty() {
        let project_file_list: Vec<String> =
            corpus.files.iter().map(|f| f.rel_path.clone()).collect();
        for x in code_intel {
            store_code_intel_extraction(&tx, project_id, x, &corpus.doc.path, &project_file_list)?;
        }
    }

    if let Some(p) = publication {
        finalize_project_row_in(
            &tx,
            project_id,
            p.title,
            p.summary,
            p.project_mtime,
            now,
            p.scan_signature,
            p.summary_vector.map(|v| (identity, v)),
        )?;
    }
    tx.commit()
        .map_err(|e| format!("failed committing project publish: {}", e))?;
    Ok(Published {
        embedded_ids,
        lance_batch,
        pruned,
    })
}

/// Why one project's run failed.
pub(crate) enum ProjectFailure {
    /// The embedding backend failed (after retries): the run stops, every later project would
    /// fail the same way. The project keeps its previous state and is retried next run.
    Embedding(String),
    /// Anything else (sqlite, code intelligence, an injected test failure): this project only;
    /// it keeps its previous state and the run goes on.
    Project(String),
}

/// What one successfully published project reports.
pub(crate) struct ProjectOutcome {
    project_id: i64,
    /// Chunks the project holds after the run: re-chunked files' chunks plus the stored
    /// chunks of unchanged files.
    rows: i64,
    /// Chunks embedded this run.
    vectors: i64,
    /// Chunks whose stored vector matched the identity and was kept.
    reused: i64,
    pruned: PruneOutcome,
    summary_embedded: bool,
    title: String,
    summary: String,
}

#[cfg(test)]
pub(crate) static INJECT_FAIL_BEFORE_PUBLISH: AtomicBool = AtomicBool::new(false);

/// Test hook: the next collector run for a project directory with this basename panics.
#[cfg(test)]
pub(crate) static INJECT_COLLECTOR_PANIC: Mutex<Option<String>> = Mutex::new(None);

pub(crate) fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

pub(crate) fn record_project_failure(
    stats: &mut IndexStats,
    idx: usize,
    total: usize,
    project_name: &str,
    message: &str,
    emit_progress: bool,
) {
    stats.projects_failed += 1;
    stats
        .failures
        .push(format!("{}: {}", project_name, message));
    if emit_progress {
        progress_clear_line();
        println!(
            "[{}/{}] error: {}: {} (the project keeps its previous state and is retried next run)",
            idx + 1,
            total,
            project_name,
            message
        );
    }
}

/// Embed a project summary with retries.
pub(crate) fn embed_summary_with_retry(
    embedder: &dyn Embedder,
    title: &str,
    summary: &str,
    project_dir: &Path,
) -> Result<Vec<f32>, String> {
    let summary_text = format!("{}\n{}", title, summary);
    let mut last_err = String::new();
    for attempt in 0..=3usize {
        if attempt > 0 {
            thread::sleep(Duration::from_millis(500 * 2u64.pow((attempt - 1) as u32)));
        }
        match embedder.embed_one(&summary_text) {
            Ok(vector) => return Ok(vector),
            Err(e) => last_err = e,
        }
    }
    Err(format!(
        "failed embedding project summary for '{}' after retries: {}",
        project_dir.display(),
        last_err
    ))
}

/// One project's write pipeline after its corpus is collected: mark the row in progress; plan
/// vector reuse; embed the rest (no transaction open); parse code intelligence; embed the
/// summary when it changed; then publish everything in one transaction and write LanceDB.
/// A failure before the publish leaves the project exactly as it was (plus the in-progress
/// marker, so the next run rescans it); the publish itself is atomic.
#[allow(clippy::too_many_arguments)] // one project's run: store handles, identity, corpus, row state, progress and the force switch
pub(crate) fn index_one_project(
    cwd: &Path,
    conn: &Connection,
    caps: &ScanCaps,
    identity: &EmbedIdentity,
    embedder: &dyn Embedder,
    corpus: &ProjectCorpus,
    project_dir: &Path,
    project_path: &str,
    existing: Option<&ExistingProject>,
    now: f64,
    live_progress: Option<&Arc<LiveIndexProgress>>,
    force_embed: bool,
) -> Result<ProjectOutcome, ProjectFailure> {
    // 1. The row, marked in progress; an existing row keeps its title, summary and signature
    //    until the publish.
    let project_id = begin_project_update(conn, project_path, &corpus.doc.title)
        .map_err(ProjectFailure::Project)?;

    // 2. Chunks: decide reuse, embed the rest into memory.
    let plan = plan_chunk_reuse(conn, project_id, identity, corpus, force_embed)
        .map_err(ProjectFailure::Project)?;
    let reused = plan.reuse.len() as i64;
    if let Some(lp) = live_progress {
        lp.add_chunks_done(plan.reuse.len());
        lp.add_tokens_done(plan.reuse.iter().map(|c| c.token_count.max(0) as u64).sum());
    }
    let embedded = embed_planned_chunks(embedder, &plan.embed, live_progress)
        .map_err(ProjectFailure::Embedding)?;
    let vectors = embedded.len() as i64;

    // 3. Symbols and imports of the re-chunked code files, parsed outside any transaction.
    if let Some(lp) = live_progress {
        lp.set_phase("code-intel");
    }
    let code_intel = extract_project_code_intel(project_dir, corpus, caps);

    // 4. The project summary and its vector. An incomplete scan of a known project keeps the
    //    stored summary (it would list fewer files than exist), so nothing is re-embedded for
    //    a directory that is merely unreadable right now. The vector is reused when the
    //    summary text is unchanged and the stored vector matches the current identity;
    //    `reembed` always regenerates it.
    let keep_stored_summary = !corpus.complete && existing.is_some();
    let (title, summary) = match (existing, keep_stored_summary) {
        (Some(row), true) => (row.title.clone(), row.summary.clone()),
        _ => (corpus.doc.title.clone(), corpus.doc.summary.clone()),
    };
    let summary_reusable = !force_embed
        && existing.map(|r| r.summary.as_str()) == Some(summary.as_str())
        && project_vector_matches(conn, project_id, identity).map_err(ProjectFailure::Project)?;
    let summary_vector = if summary_reusable {
        None
    } else {
        Some(
            embed_summary_with_retry(embedder, &title, &summary, project_dir)
                .map_err(ProjectFailure::Embedding)?,
        )
    };

    #[cfg(test)]
    if INJECT_FAIL_BEFORE_PUBLISH.swap(false, Ordering::SeqCst) {
        return Err(ProjectFailure::Project(
            "injected failure between embedding and publish".to_string(),
        ));
    }

    // 5. Publish: everything of this project succeeded. The signature moves only for a
    //    complete scan; an incomplete one keeps the old signature and is rescanned.
    ensure_lance_open_for(cwd, &embedded);
    if let Some(lp) = live_progress {
        lp.set_phase("publish");
    }
    let signature = corpus.complete.then_some(corpus.scan_signature.as_str());
    let published = publish_project(
        conn,
        project_id,
        identity,
        corpus,
        &plan.reuse,
        embedded,
        &code_intel,
        Some(ProjectPublication {
            title: &title,
            summary: &summary,
            project_mtime: corpus.doc.mtime,
            scan_signature: signature,
            summary_vector: summary_vector.as_deref(),
        }),
        now,
    )
    .map_err(ProjectFailure::Project)?;
    sync_lance_after_publish(conn, &published).map_err(ProjectFailure::Project)?;

    let carried: i64 = corpus
        .files
        .iter()
        .filter(|f| !f.rechunked)
        .map(|f| f.chunk_count.max(0))
        .sum();
    Ok(ProjectOutcome {
        project_id,
        rows: corpus.chunks.len() as i64 + carried,
        vectors,
        reused,
        pruned: published.pruned,
        summary_embedded: summary_vector.is_some(),
        title,
        summary,
    })
}

/// Open LanceDB at the dimension of the vectors about to be written, when it is not open yet
/// (the first embedded batch of a fresh store fixes the dimension). A failure is reported
/// once and leaves the marker for the repair step.
pub(crate) fn ensure_lance_open_for(cwd: &Path, embedded: &[EmbeddedChunk<'_>]) {
    let Some(dim) = embedded.first().map(|e| e.vector.len()) else {
        return;
    };
    if let Err(e) = get_or_open_lance(cwd, dim) {
        LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
        eprintln!(
            "error: LanceDB open failed ({}); vectors are written to sqlite only and the next index run repairs LanceDB",
            e
        );
    }
}

/// Bring LanceDB in line with what [`publish_project`] committed: upsert the embedded
/// vectors (then clear their pending ids and, if nothing else failed in this process, the
/// marker) and delete the pruned rows. Failures are reported and left for the repair step;
/// sqlite already holds the truth.
pub(crate) fn sync_lance_after_publish(
    conn: &Connection,
    published: &Published,
) -> Result<(), String> {
    if !published.lance_batch.is_empty() {
        match with_lance_store(|store| lance_store::upsert_chunks(store, &published.lance_batch)) {
            Ok(()) => {
                lance_pending_remove(conn, &published.embedded_ids)?;
                lance_mark_clean(conn)?;
            }
            Err(e) => {
                LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
                eprintln!(
                    "warning: LanceDB upsert of {} vectors failed ({}); sqlite has them and the next index run repairs LanceDB",
                    published.lance_batch.len(),
                    e
                );
            }
        }
    }
    if !published.pruned.chunk_ids.is_empty() && lance_store_is_open() {
        if let Err(e) = lance_delete_marked(conn, &published.pruned.chunk_ids) {
            eprintln!(
                "warning: LanceDB delete of {} stale vectors failed ({}); the next index run repairs it",
                published.pruned.chunk_ids.len(),
                e
            );
        }
    }
    Ok(())
}

/// Result of re-indexing one project: row/vector counts plus what was pruned (chunk-level
/// tests only; `run_native_index` reports through [`IndexStats`]).
#[cfg(test)]
pub(crate) struct ReindexOutcome {
    /// Chunks the project holds after the run: re-chunked files' chunks plus the stored
    /// chunks of unchanged files.
    rows: i64,
    /// Chunks embedded this run.
    vectors: i64,
    /// Chunks whose stored vector matched the current embedding identity and was kept.
    reused: i64,
    /// What the publish pruned.
    pruned: PruneOutcome,
}

/// The chunk part of a project's run, without the project row: plan, embed, publish in one
/// transaction, then write LanceDB. `run_native_index` goes through [`publish_project`] with
/// the project row included; this entry point serves the chunk-level tests and mirrors it.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // test entry mirroring index_one_project's chunk half
pub(crate) fn reindex_project_chunks(
    cwd: &Path,
    conn: &Connection,
    project_id: i64,
    identity: &EmbedIdentity,
    embedder: &dyn Embedder,
    corpus: &ProjectCorpus,
    now: f64,
    live_progress: Option<&Arc<LiveIndexProgress>>,
    force_embed: bool,
) -> Result<ReindexOutcome, String> {
    let plan = plan_chunk_reuse(conn, project_id, identity, corpus, force_embed)?;
    let reused = plan.reuse.len() as i64;
    if let Some(lp) = live_progress {
        lp.add_chunks_done(plan.reuse.len());
        lp.add_tokens_done(plan.reuse.iter().map(|c| c.token_count.max(0) as u64).sum());
    }
    let embedded = embed_planned_chunks(embedder, &plan.embed, live_progress)?;
    let vectors = embedded.len() as i64;
    ensure_lance_open_for(cwd, &embedded);
    let published = publish_project(
        conn,
        project_id,
        identity,
        corpus,
        &plan.reuse,
        embedded,
        &[],
        None,
        now,
    )?;
    sync_lance_after_publish(conn, &published)?;
    let carried: i64 = corpus
        .files
        .iter()
        .filter(|f| !f.rechunked)
        .map(|f| f.chunk_count.max(0))
        .sum();
    Ok(ReindexOutcome {
        rows: corpus.chunks.len() as i64 + carried,
        vectors,
        reused,
        pruned: published.pruned,
    })
}

/// Set the Lance dirty marker inside the caller's sqlite transaction, right before the rows it
/// covers are committed. Cleared by [`lance_mark_clean`] once the LanceDB write succeeded.
pub(crate) fn lance_mark_dirty(conn: &Connection) -> Result<(), String> {
    app_state_set(conn, APP_STATE_LANCE_DIRTY, "1")
}

/// Clear the dirty marker after a successful LanceDB write, unless an earlier write in this
/// process failed (its rows are still missing; the next writer run repairs them).
pub(crate) fn lance_mark_clean(conn: &Connection) -> Result<(), String> {
    if LANCE_WRITE_FAILED.load(Ordering::SeqCst) {
        return Ok(());
    }
    app_state_set(conn, APP_STATE_LANCE_DIRTY, "0")
}

/// Record chunk ids whose sqlite vector is about to be written to LanceDB (same transaction
/// as the vectors). Removed once the LanceDB write succeeded; whatever remains is rewritten
/// by the repair step even when LanceDB already holds a (stale) row for the id.
pub(crate) fn lance_pending_add(conn: &Connection, chunk_ids: &[i64]) -> Result<(), String> {
    let mut stmt = conn
        .prepare("INSERT OR REPLACE INTO lance_pending(chunk_id) VALUES (?1)")
        .map_err(|e| format!("failed preparing lance_pending insert: {}", e))?;
    for id in chunk_ids {
        stmt.execute(params![id])
            .map_err(|e| format!("failed recording pending lance row: {}", e))?;
    }
    Ok(())
}

pub(crate) fn lance_pending_remove(conn: &Connection, chunk_ids: &[i64]) -> Result<(), String> {
    for batch in chunk_ids.chunks(500) {
        let placeholders = vec!["?"; batch.len()].join(", ");
        conn.execute(
            &format!(
                "DELETE FROM lance_pending WHERE chunk_id IN ({})",
                placeholders
            ),
            params_from_iter(batch.iter()),
        )
        .map_err(|e| format!("failed clearing pending lance rows: {}", e))?;
    }
    Ok(())
}

pub(crate) fn lance_pending_ids(conn: &Connection) -> Result<Vec<i64>, String> {
    let mut stmt = conn
        .prepare("SELECT chunk_id FROM lance_pending ORDER BY chunk_id")
        .map_err(|e| format!("failed preparing lance_pending query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, i64>(0))
        .map_err(|e| format!("failed querying pending lance rows: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading pending lance row: {}", e))?);
    }
    Ok(out)
}

/// Delete LanceDB rows with the dirty marker around the write: set before, cleared after
/// success, left set (and the process flagged) on failure.
pub(crate) fn lance_delete_marked(conn: &Connection, chunk_ids: &[i64]) -> Result<usize, String> {
    if chunk_ids.is_empty() {
        return Ok(0);
    }
    lance_mark_dirty(conn)?;
    match with_lance_store(|store| lance_store::delete_chunks(store, chunk_ids)) {
        Ok(()) => {
            lance_mark_clean(conn)?;
            Ok(chunk_ids.len())
        }
        Err(e) => {
            LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
            Err(e)
        }
    }
}

/// The counts [`repair_lance_from_sqlite`] would act on, without writing: (rows missing from
/// LanceDB, LanceDB rows with no sqlite vector). For `prune --dry-run`.
pub(crate) fn lance_reconcile_preview(
    conn: &Connection,
    model_key: &str,
) -> Result<(usize, usize), String> {
    let (lance_ids, dim) = with_lance_store(|store| {
        Ok((lance_store::list_chunk_ids(store)?, lance_store::dim(store)))
    })?;
    let lance_set: HashSet<i64> = lance_ids.into_iter().collect();
    let mut stmt = conn
        .prepare("SELECT chunk_id FROM project_chunk_vectors WHERE model = ?1 AND dim = ?2")
        .map_err(|e| format!("failed preparing chunk vector id query: {}", e))?;
    let rows = stmt
        .query_map(params![model_key, dim as i64], |row| row.get::<_, i64>(0))
        .map_err(|e| format!("failed querying chunk vector ids: {}", e))?;
    let mut sqlite_set: HashSet<i64> = HashSet::new();
    for row in rows {
        sqlite_set.insert(row.map_err(|e| format!("failed reading chunk vector id: {}", e))?);
    }
    let pending: HashSet<i64> = lance_pending_ids(conn)?.into_iter().collect();
    let missing = sqlite_set
        .iter()
        .filter(|id| !lance_set.contains(id) || pending.contains(id))
        .count();
    let orphans = lance_set.difference(&sqlite_set).count();
    Ok((missing, orphans))
}

/// What [`repair_lance_from_sqlite`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct LanceRepairReport {
    /// Rows written to LanceDB from the sqlite blobs: ids LanceDB lacked plus ids whose write
    /// was pending (a stale same-id row may have been there).
    pub(crate) rebuilt: usize,
    /// LanceDB rows with no sqlite vector, deleted.
    pub(crate) orphans_removed: usize,
    /// True when the marker said clean and nothing was compared.
    skipped: bool,
}

/// Bring LanceDB back in line with sqlite without embedding anything: diff the chunk ids in
/// LanceDB against `project_chunk_vectors` for the active model at the store's dimension,
/// rebuild the missing rows from the sqlite blobs and delete the orphans. Runs at the start of
/// index and refresh when the dirty marker is set or absent (`force` false), and always in
/// `prune` (`force` true). Requires the store to be open ([`get_or_open_lance`]); with no
/// vectors for the model yet there is nothing to reconcile and the marker is cleared.
pub(crate) fn repair_lance_from_sqlite(
    conn: &Connection,
    model_key: &str,
    force: bool,
) -> Result<LanceRepairReport, String> {
    let marker = app_state_get(conn, APP_STATE_LANCE_DIRTY)?;
    if !force && marker.as_deref() == Some("0") {
        return Ok(LanceRepairReport {
            skipped: true,
            ..LanceRepairReport::default()
        });
    }
    if !lance_store_is_open() {
        if vector_dim_from_sqlite(conn, model_key).is_none() {
            app_state_set(conn, APP_STATE_LANCE_DIRTY, "0")?;
            return Ok(LanceRepairReport::default());
        }
        return Err("LanceDB is not open".to_string());
    }
    // A crash mid-repair must leave the marker set.
    lance_mark_dirty(conn)?;
    let pending: HashSet<i64> = lance_pending_ids(conn)?.into_iter().collect();
    let (lance_ids, dim) = with_lance_store(|store| {
        Ok((lance_store::list_chunk_ids(store)?, lance_store::dim(store)))
    })?;
    let lance_set: HashSet<i64> = lance_ids.into_iter().collect();
    let sqlite_ids: Vec<i64> = {
        let mut stmt = conn
            .prepare(
                "SELECT chunk_id FROM project_chunk_vectors WHERE model = ?1 AND dim = ?2 ORDER BY chunk_id",
            )
            .map_err(|e| format!("failed preparing chunk vector id query: {}", e))?;
        let rows = stmt
            .query_map(params![model_key, dim as i64], |row| row.get::<_, i64>(0))
            .map_err(|e| format!("failed querying chunk vector ids: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| format!("failed reading chunk vector id: {}", e))?);
        }
        out
    };
    let sqlite_set: HashSet<i64> = sqlite_ids.iter().copied().collect();
    // Rows to write: absent from LanceDB, or pending (their last write did not complete, so
    // whatever LanceDB holds under that id may be stale).
    let missing: Vec<i64> = sqlite_ids
        .iter()
        .copied()
        .filter(|id| !lance_set.contains(id) || pending.contains(id))
        .collect();
    let mut orphans: Vec<i64> = lance_set
        .iter()
        .copied()
        .filter(|id| !sqlite_set.contains(id))
        .collect();
    orphans.sort();

    let mut report = LanceRepairReport::default();
    for batch in missing.chunks(500) {
        let placeholders = vec!["?"; batch.len()].join(", ");
        let sql = format!(
            "SELECT chunk_id, vector FROM project_chunk_vectors WHERE chunk_id IN ({})",
            placeholders
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed preparing vector rebuild query: {}", e))?;
        let rows = stmt
            .query_map(params_from_iter(batch.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| format!("failed querying vectors for rebuild: {}", e))?;
        let mut lance_batch: Vec<(i64, Vec<f32>)> = Vec::with_capacity(batch.len());
        for row in rows {
            let (chunk_id, blob) =
                row.map_err(|e| format!("failed reading vector for rebuild: {}", e))?;
            let vector = blob_to_f32_vec(&blob);
            if vector.len() != dim {
                // Fail closed: a malformed vector is not skipped over. The marker stays set,
                // the pending ids stay, and the run reports the failure.
                LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
                return Err(format!(
                    "sqlite vector for chunk {} holds {} floats ({} bytes) but LanceDB stores {}-dimensional vectors; repair refused, dirty marker and pending ids kept",
                    chunk_id,
                    vector.len(),
                    blob.len(),
                    dim
                ));
            }
            lance_batch.push((chunk_id, vector));
        }
        if lance_batch.len() != batch.len() {
            LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
            return Err(format!(
                "{} of {} required sqlite vectors are missing; repair refused, dirty marker and pending ids kept",
                batch.len() - lance_batch.len(),
                batch.len()
            ));
        }
        with_lance_store(|store| lance_store::upsert_chunks(store, &lance_batch)).map_err(|e| {
            LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
            format!(
                "LanceDB rebuild of {} rows failed: {}",
                lance_batch.len(),
                e
            )
        })?;
        report.rebuilt += lance_batch.len();
    }
    if !orphans.is_empty() {
        with_lance_store(|store| lance_store::delete_chunks(store, &orphans)).map_err(|e| {
            LANCE_WRITE_FAILED.store(true, Ordering::SeqCst);
            format!(
                "LanceDB orphan delete of {} rows failed: {}",
                orphans.len(),
                e
            )
        })?;
        report.orphans_removed = orphans.len();
    }
    // Every pending id was either rewritten above or no longer has a sqlite vector (its
    // LanceDB row, if any, was an orphan and is gone).
    conn.execute("DELETE FROM lance_pending", [])
        .map_err(|e| format!("failed clearing pending lance rows: {}", e))?;
    app_state_set(conn, APP_STATE_LANCE_DIRTY, "0")?;
    // LanceDB now matches sqlite: later successful writes may clear the marker again.
    LANCE_WRITE_FAILED.store(false, Ordering::SeqCst);
    Ok(report)
}

#[cfg(test)]
mod incremental_index_tests {
    use super::*;
    use crate::code_intel;
    use crate::config::{selection_tier, ConfigValues, ScanSettings};
    use crate::db::{
        app_state_get, app_state_set, init_schema, table_has_column, APP_STATE_LANCE_DIRTY,
    };
    use crate::prune::{prune_stale_project_rows, remove_projects_not_in, PruneKeepSet};
    use crate::scan::{
        collect_project_corpus, content_hash_xxh64, file_has_changed, manifest_stat_match,
        project_scan, scan_signature_for, select_scan_files, upsert_file_manifest, CandidateFile,
        FileManifest, FileManifestEntry, ProjectListing, ProjectScan, ScanCaps,
    };
    use crate::util::{bool_env, normalize_path, now_ts};
    use rusqlite::Connection;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    const BEDROCK_MODEL: &str = "bedrock:amazon.titan-embed-text-v2:0";

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        init_schema(&conn).expect("init schema");
        conn
    }

    fn count(conn: &Connection, sql: &str) -> i64 {
        conn.query_row(sql, [], |row| row.get(0)).expect(sql)
    }

    fn scanned(rel: &str, doc_path: &str, rechunked: bool, chunk_count: i64) -> ScannedFile {
        ScannedFile {
            rel_path: rel.to_string(),
            doc_path: doc_path.to_string(),
            size: 1,
            mtime: 0.0,
            content_hash: "h".to_string(),
            chunk_count,
            rechunked,
            stat_changed: false,
        }
    }

    fn corpus(chunks: Vec<ProjectChunk>, files: Vec<ScannedFile>) -> ProjectCorpus {
        ProjectCorpus {
            doc: ProjectDoc {
                path: PathBuf::from("/p/alpha"),
                title: "alpha".to_string(),
                summary: "alpha".to_string(),
                mtime: 0.0,
            },
            chunks,
            files,
            scan_signature: String::new(),
            complete: true,
            files_unreadable: 0,
            files_evicted_by_cap: 0,
            files_truncated_by_cap: 0,
            caps_note: String::new(),
            documents_extracted: 0,
            documents_failed: 0,
            document_failures: Vec::new(),
        }
    }

    fn scan_of(root: &Path) -> ProjectScan {
        project_scan(
            root,
            &HashSet::new(),
            &ScanCaps::default(),
            false,
            &ScanSettings::default(),
        )
    }

    fn collect(root: &Path, manifest: &FileManifest, rechunk_all: bool) -> ProjectCorpus {
        let scan = scan_of(root);
        collect_project_corpus(
            root,
            &scan,
            &ScanCaps::default(),
            100_000,
            manifest,
            rechunk_all,
            &ScanSettings::default(),
        )
    }

    fn identity(dim: Option<i64>) -> EmbedIdentity {
        EmbedIdentity {
            model: BEDROCK_MODEL.to_string(),
            dim,
            normalized: true,
            pipeline_version: EMBEDDING_PIPELINE_VERSION,
        }
    }

    fn stored(hash: &str) -> StoredVectorIdentity {
        StoredVectorIdentity {
            embed_input_hash: hash.to_string(),
            model: BEDROCK_MODEL.to_string(),
            dim: 1024,
            normalized: true,
            pipeline_version: EMBEDDING_PIPELINE_VERSION,
        }
    }

    #[test]
    fn embed_input_hash_is_stable_and_hashes_the_exact_embedder_input() {
        // SHA-1 of "hello": the value must never drift, or every stored vector is orphaned.
        assert_eq!(
            embed_input_hash("hello"),
            "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
        );
        assert_eq!(embed_input_for("", "body"), "body");
        assert_eq!(
            embed_input_for("// File: a.rs", "body"),
            "// File: a.rs\nbody"
        );
        assert_eq!(
            embed_input_hash(&embed_input_for("h", "t")),
            embed_input_hash("h\nt")
        );
        assert_ne!(
            embed_input_hash(&embed_input_for("", "t")),
            embed_input_hash(&embed_input_for("h", "t")),
            "the header is part of the embedder input, so it is part of the identity"
        );
    }

    #[test]
    fn a_stored_vector_is_reused_only_when_every_identity_field_matches() {
        let current = identity(Some(1024));
        let hash = embed_input_hash("x");
        let same = stored(&hash);
        assert!(current.reuses(&same, &hash));
        assert!(!current.reuses(&stored(&embed_input_hash("y")), &hash));
        assert!(!current.reuses(
            &StoredVectorIdentity {
                model: "ollama:qwen3-embedding".to_string(),
                ..same.clone()
            },
            &hash
        ));
        assert!(!current.reuses(
            &StoredVectorIdentity {
                dim: 512,
                ..same.clone()
            },
            &hash
        ));
        assert!(!current.reuses(
            &StoredVectorIdentity {
                normalized: false,
                ..same.clone()
            },
            &hash
        ));
        assert!(!current.reuses(
            &StoredVectorIdentity {
                pipeline_version: EMBEDDING_PIPELINE_VERSION + 1,
                ..same.clone()
            },
            &hash
        ));
        // No vectors exist for the model yet: nothing can be reused.
        assert!(!identity(None).reuses(&same, &hash));
        // A row that never got an identity is never reused.
        assert!(!current.reuses(&stored(""), ""));
    }

    #[test]
    fn migration_backfills_identity_from_stored_text_without_reembedding() {
        let conn = conn();
        // Roll the vectors table back to its 0.1.x shape and seed it the old way.
        for column in ["embed_input_hash", "normalized", "pipeline_version"] {
            conn.execute(
                &format!("ALTER TABLE project_chunk_vectors DROP COLUMN {}", column),
                [],
            )
            .expect("drop identity column");
        }
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/alpha', 'alpha', 'alpha', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at, context_header)
VALUES (10, 1, '/p/alpha/a.md', 'a.md', 0, 0, 1, 'h', 'prose body', 0, ''),
       (11, 1, '/p/alpha/b.rs', 'b.rs', 0, 0, 1, 'h', 'fn b() {}', 0, '// File: b.rs'),
       (12, 1, '/p/alpha/c.md', 'c.md', 0, 0, 1, 'h', 'local body', 0, '');
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector)
VALUES (10, 'bedrock:amazon.titan-embed-text-v2:0', 1024, 1.0, x'00000000'),
       (11, 'bedrock:amazon.titan-embed-text-v2:0', 1024, 1.0, x'00000000'),
       (12, 'ollama:qwen3-embedding', 1024, 1.0, x'00000000');
"#,
        )
        .expect("seed 0.1.x rows");
        assert!(!table_has_column(&conn, "project_chunk_vectors", "embed_input_hash").unwrap());

        init_schema(&conn).expect("open migrates");

        assert!(table_has_column(&conn, "project_chunk_vectors", "embed_input_hash").unwrap());
        let ids = load_project_vector_identities(&conn, 1).expect("identities");
        assert_eq!(ids.len(), 3);
        let bedrock_normalized = bool_env("RETRIVIO_BEDROCK_NORMALIZE", true);
        assert_eq!(
            ids[&10],
            StoredVectorIdentity {
                embed_input_hash: embed_input_hash("prose body"),
                model: BEDROCK_MODEL.to_string(),
                dim: 1024,
                normalized: bedrock_normalized,
                pipeline_version: 1,
            }
        );
        assert_eq!(
            ids[&11].embed_input_hash,
            embed_input_hash("// File: b.rs\nfn b() {}"),
            "the header the embedder saw is part of the backfilled hash"
        );
        assert!(
            !ids[&12].normalized,
            "only bedrock rows carry the bedrock normalisation flag"
        );
        assert_eq!(ids[&12].pipeline_version, 1);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE embed_input_hash = '' OR pipeline_version = 0"
            ),
            0
        );

        // Opening again is a no-op.
        init_schema(&conn).expect("second open");
        assert_eq!(load_project_vector_identities(&conn, 1).unwrap(), ids);

        // The upgrade rule: the current identity reuses every backfilled bedrock row for the
        // exact input the run would build from the stored chunk.
        let current = EmbedIdentity {
            normalized: bedrock_normalized,
            ..identity(Some(1024))
        };
        assert!(current.reuses(
            &ids[&10],
            &embed_input_hash(&embed_input_for("", "prose body"))
        ));
        assert!(current.reuses(
            &ids[&11],
            &embed_input_hash(&embed_input_for("// File: b.rs", "fn b() {}"))
        ));
    }

    #[test]
    fn work_set_classifies_unchanged_touched_changed_and_new_files() {
        let mut manifest = FileManifest::new();
        manifest.insert(
            "a.md".to_string(),
            FileManifestEntry {
                size: 5,
                mtime: 100.0,
                content_hash: content_hash_xxh64(b"hello"),
                chunk_count: 2,
            },
        );
        // Same size and mtime: unchanged on the fast path, without reading the file.
        assert_eq!(
            manifest_stat_match(&manifest, "a.md", 5, 100.0).map(|e| e.chunk_count),
            Some(2)
        );
        assert!(manifest_stat_match(&manifest, "a.md", 5, 101.0).is_none());
        assert!(manifest_stat_match(&manifest, "a.md", 6, 100.0).is_none());
        assert!(manifest_stat_match(&manifest, "b.md", 5, 100.0).is_none());
        // Touched: the mtime moved but the bytes did not. Unchanged, hash kept.
        assert_eq!(
            file_has_changed(&manifest, "a.md", 5, 101.0, b"hello"),
            (false, content_hash_xxh64(b"hello"))
        );
        // Edited in place, same size: changed.
        assert_eq!(
            file_has_changed(&manifest, "a.md", 5, 101.0, b"hellp"),
            (true, content_hash_xxh64(b"hellp"))
        );
        // New file: changed.
        assert_eq!(
            file_has_changed(&manifest, "b.md", 5, 100.0, b"hello"),
            (true, content_hash_xxh64(b"hello"))
        );
    }

    #[test]
    fn collect_reads_and_chunks_only_the_changed_work_set() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("incremental-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("docs")).expect("mk docs");
        fs::write(root.join("docs").join("keep.md"), "keep me exactly as I am").expect("keep");
        fs::write(root.join("docs").join("edit.md"), "first version").expect("edit v1");

        // First run: no manifest, everything is read and chunked.
        let full = collect(&root, &FileManifest::new(), false);
        assert_eq!(full.files.len(), 2);
        assert!(full.files.iter().all(|f| f.rechunked));
        assert_eq!(full.chunks.len(), 2);
        assert_eq!(full.files_rechunked(), 2);
        assert_eq!(full.files_unchanged(), 0);
        assert!(full.complete);
        assert!(full.scan_signature.starts_with("2:"));

        // The manifest the run stores; then edit one file and add another.
        let manifest: FileManifest = full
            .files
            .iter()
            .map(|f| {
                (
                    f.rel_path.clone(),
                    FileManifestEntry {
                        size: f.size,
                        mtime: f.mtime,
                        content_hash: f.content_hash.clone(),
                        chunk_count: f.chunk_count,
                    },
                )
            })
            .collect();
        fs::write(
            root.join("docs").join("edit.md"),
            "second version, longer than before",
        )
        .expect("edit v2");
        fs::write(root.join("docs").join("new.md"), "brand new").expect("new");

        let incremental = collect(&root, &manifest, false);
        let by_rel: HashMap<&str, &ScannedFile> = incremental
            .files
            .iter()
            .map(|f| (f.rel_path.as_str(), f))
            .collect();
        assert_eq!(incremental.files.len(), 3);
        assert!(!by_rel["docs/keep.md"].rechunked);
        assert!(!by_rel["docs/keep.md"].stat_changed);
        assert_eq!(by_rel["docs/keep.md"].chunk_count, 1);
        assert!(by_rel["docs/edit.md"].rechunked);
        assert!(by_rel["docs/new.md"].rechunked);
        let chunked: HashSet<&str> = incremental
            .chunks
            .iter()
            .map(|c| c.doc_rel_path.as_str())
            .collect();
        assert_eq!(chunked, HashSet::from(["docs/edit.md", "docs/new.md"]));
        assert_ne!(
            incremental.scan_signature, full.scan_signature,
            "a new file changes the signature"
        );
        assert!(incremental.scan_signature.starts_with("3:"));
        assert_eq!(
            incremental.doc.summary.matches("indexed_files 3").count(),
            1,
            "unchanged files still count as indexed in the summary"
        );
        // doc_path is the canonical project path joined with rel_path, for every file.
        let project_path = normalize_path(&root.to_string_lossy());
        for f in &incremental.files {
            assert_eq!(f.doc_path, project_path.join(&f.rel_path).to_string_lossy());
        }
        for c in &incremental.chunks {
            assert_eq!(
                c.doc_path,
                project_path.join(&c.doc_rel_path).to_string_lossy()
            );
        }

        // `rechunk_all` (refresh, reembed, incomplete vectors) ignores the manifest.
        let forced = collect(&root, &manifest, true);
        assert_eq!(forced.files_rechunked(), 3);
        assert_eq!(forced.chunks.len(), 3);

        // The full keep set: the unchanged file is kept whole, re-chunked ones by index.
        let keep = PruneKeepSet::from_corpus(&incremental);
        assert!(keep.keeps_chunk(&by_rel["docs/keep.md"].doc_path, 7));
        assert!(keep.keeps_doc(&by_rel["docs/keep.md"].doc_path));
        assert!(keep.keeps_chunk(&by_rel["docs/edit.md"].doc_path, 0));
        assert!(!keep.keeps_chunk(&by_rel["docs/edit.md"].doc_path, 1));
        assert!(keep.rel_paths.contains("docs/keep.md"));
        assert!(keep.rel_paths.contains("docs/new.md"));

        // Deleting a file changes the signature even when no surviving file's mtime moved.
        let before = scan_of(&root);
        fs::remove_file(root.join("docs").join("new.md")).expect("delete");
        let after = scan_of(&root);
        assert_ne!(before.signature, after.signature);
        assert!(after.signature.starts_with("2:"));
        let _ = fs::remove_dir_all(&root);
    }

    fn candidate(rel: &str, size: u64, mtime_ns: i128) -> CandidateFile {
        let suffix = Path::new(rel)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        CandidateFile {
            rel_path: rel.to_string(),
            fs_path: PathBuf::from("/p").join(rel),
            size,
            mtime: mtime_ns as f64 / 1e9,
            mtime_ns,
            tier: selection_tier(&suffix),
        }
    }

    #[test]
    fn selection_tiers_put_notes_before_code_before_config_newest_first_within_a_tier() {
        let listing = ProjectListing {
            candidates: vec![
                candidate("ci/pipeline.yml", 1, 900),
                candidate("src/app.py", 1, 800),
                candidate("README.md", 1, 100),
                candidate("data/config.json", 1, 950),
                candidate("docs/plan.docx", 1, 50),
                candidate("src/lib.rs", 1, 800),
                candidate("notes/old.txt", 1, 20),
            ],
            ..ProjectListing::default()
        };
        let (selected, evicted) = select_scan_files(&listing, &ScanCaps::default());
        let order: Vec<&str> = selected.iter().map(|c| c.rel_path.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "README.md",
                "docs/plan.docx",
                "notes/old.txt",
                "src/app.py",
                "src/lib.rs",
                "data/config.json",
                "ci/pipeline.yml",
            ]
        );
        assert_eq!(evicted, 0);
        assert_eq!(selection_tier(".sql"), 3);
        assert_eq!(selection_tier(".sh"), 2);
        assert_eq!(selection_tier(".pdf"), 1);
        assert_eq!(selection_tier(".htm"), 1);
    }

    #[test]
    fn scan_signature_ignores_order_and_tracks_membership_size_and_mtime() {
        let a = scan_signature_for(&[candidate("b.md", 5, 100), candidate("a.md", 7, 200)]);
        let b = scan_signature_for(&[candidate("a.md", 7, 200), candidate("b.md", 5, 100)]);
        assert_eq!(a, b, "order of discovery does not matter");
        assert!(a.starts_with("2:"));
        assert_ne!(a, scan_signature_for(&[candidate("a.md", 7, 200)]));
        assert_ne!(
            a,
            scan_signature_for(&[candidate("b.md", 5, 100), candidate("a.md", 8, 200)]),
            "a size change with the same mtime is a change"
        );
        assert_ne!(
            a,
            scan_signature_for(&[candidate("b.md", 5, 100), candidate("a.md", 7, 201)]),
            "an mtime change with the same size is a change"
        );
        assert_ne!(
            a,
            scan_signature_for(&[candidate("b.md", 5, 100), candidate("c.md", 7, 200)]),
            "a rename is a change"
        );
    }

    #[test]
    fn selection_is_deterministic_docs_first_newest_first_then_path() {
        let listing = ProjectListing {
            candidates: vec![
                candidate("z.rs", 1, 300),
                candidate("b.md", 1, 100),
                candidate("a.md", 1, 100),
                candidate("c.md", 1, 200),
            ],
            ..ProjectListing::default()
        };
        let caps = ScanCaps {
            max_files_per_project: 3,
            ..ScanCaps::default()
        };
        let (selected, evicted) = select_scan_files(&listing, &caps);
        let order: Vec<&str> = selected.iter().map(|c| c.rel_path.as_str()).collect();
        assert_eq!(order, vec!["c.md", "a.md", "b.md"]);
        assert_eq!(evicted, 1, "the code file is newest but docs come first");
        let (again, _) = select_scan_files(&listing, &caps);
        assert_eq!(again, selected, "same input, same selection");
    }

    fn tmp_root(name: &str) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mk root");
        root
    }

    fn write_file(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(path, text).expect("write");
    }

    fn selected_rels(scan: &ProjectScan) -> Vec<String> {
        let mut rels: Vec<String> = scan.selected.iter().map(|c| c.rel_path.clone()).collect();
        rels.sort();
        rels
    }

    #[test]
    fn built_in_skip_dirs_cover_worktrees_and_app_bundles_and_shallow_scans_stay_at_the_top() {
        assert!(ScanSettings::default().is_skip_dir("worktrees"));
        assert!(ScanSettings::default().is_skip_dir(".worktrees"));
        assert!(ScanSettings::default().is_skip_dir("Amazon Quick 1.0.3377.app"));
        assert!(ScanSettings::default().is_skip_dir("Foo.app"));
        assert!(!ScanSettings::default().is_skip_dir("app"));
        assert!(!ScanSettings::default().is_skip_dir("myapp"));
        assert!(!ScanSettings::default().is_skip_dir("application"));
        assert!(
            ScanSettings::default().is_skip_dir("node_modules"),
            "the old built-ins stay"
        );

        let root = tmp_root("skipdirs-walk");
        write_file(&root.join("notes.md"), "notes at the top");
        write_file(&root.join("src").join("main.rs"), "fn main() {}");
        write_file(
            &root.join("worktrees").join("x").join("ci.yml"),
            "stages: []",
        );
        write_file(
            &root.join(".worktrees").join("y").join("a.md"),
            "hidden worktree",
        );
        write_file(
            &root
                .join("Foo.app")
                .join("Contents")
                .join("Resources")
                .join("readme.md"),
            "bundle text",
        );
        // An Office owner file is neither a document nor worth a line in the summary.
        write_file(&root.join("~$deck.pptx"), "estouff");
        let full = project_scan(
            &root,
            &HashSet::new(),
            &ScanCaps::default(),
            false,
            &ScanSettings::default(),
        );
        assert_eq!(selected_rels(&full), vec!["notes.md", "src/main.rs"]);
        assert_eq!(full.listing.file_names, vec!["notes.md", "src/main.rs"]);
        assert!(!full.shallow);

        // A root-files project: the directory's own files, no subdirectories at all.
        let shallow = project_scan(
            &root,
            &HashSet::new(),
            &ScanCaps::default(),
            true,
            &ScanSettings::default(),
        );
        assert_eq!(selected_rels(&shallow), vec!["notes.md"]);
        assert_eq!(shallow.listing.file_names, vec!["notes.md"]);
        assert!(shallow.shallow);
        assert_ne!(shallow.signature, full.signature);
        let corpus = collect_project_corpus(
            &root,
            &shallow,
            &ScanCaps::default(),
            100_000,
            &FileManifest::new(),
            true,
            &ScanSettings::default(),
        );
        let base = root.file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(
            corpus.doc.title,
            format!("{} (root files)", base.replace('-', " "))
        );
        assert!(
            corpus
                .doc
                .summary
                .starts_with(&format!("project {} (root files)\nindexed_files 1\n", base)),
            "{}",
            corpus.doc.summary
        );
        assert_eq!(corpus.files.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_project_summary_quotes_files_in_path_order_and_ignores_touches() {
        let root = tmp_root("summary-order");
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        write_file(&root.join("a.md"), "alpha text about otters");
        write_file(&root.join("b.md"), "beta text about zebras");
        crate::test_support::set_mtime(&root.join("a.md"), t0);
        crate::test_support::set_mtime(&root.join("b.md"), t0 + Duration::from_secs(60));
        let collect = || {
            let scan = scan_of(&root);
            collect_project_corpus(
                &root,
                &scan,
                &ScanCaps::default(),
                100_000,
                &FileManifest::new(),
                true,
                &ScanSettings::default(),
            )
        };
        let before = collect();
        // Selection order is newest first (b.md), the summary is path order (a.md first).
        assert!(
            before.doc.summary.find("a.md\nalpha").unwrap()
                < before.doc.summary.find("b.md\nbeta").unwrap(),
            "{}",
            before.doc.summary
        );
        // Touching a.md makes it the newest selected file; the summary text does not move.
        crate::test_support::set_mtime(&root.join("a.md"), t0 + Duration::from_secs(600));
        let after = collect();
        assert_eq!(after.doc.summary, before.doc.summary);
        assert_ne!(
            after.scan_signature, before.scan_signature,
            "the gate still sees the touch"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn excludes_apply_when_the_project_is_reached_through_a_symlinked_parent() {
        let base = tmp_root("excl-symlink");
        let real = base.join("real");
        write_file(&real.join("proj").join("src").join("a.md"), "kept");
        write_file(&real.join("proj").join("scratch").join("b.md"), "excluded");
        let link = base.join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let mut excludes: HashSet<PathBuf> = HashSet::new();
        excludes.insert(normalize_path(
            &real.join("proj").join("scratch").to_string_lossy(),
        ));
        let via_link = project_scan(
            &link.join("proj"),
            &excludes,
            &ScanCaps::default(),
            false,
            &ScanSettings::default(),
        );
        assert_eq!(selected_rels(&via_link), vec!["src/a.md"]);
        let direct = project_scan(
            &real.join("proj"),
            &excludes,
            &ScanCaps::default(),
            false,
            &ScanSettings::default(),
        );
        assert_eq!(direct.signature, via_link.signature);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn on_disk_signature_catches_preserved_timestamps_and_ignores_non_indexable_files() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("signature-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mk root");
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let set = |name: &str, text: &str, when: SystemTime| {
            fs::write(root.join(name), text).expect("write");
            crate::test_support::set_mtime(&root.join(name), when);
        };
        set("a.md", "four", t0);
        set("b.md", "beta", t0);
        let base = scan_of(&root);

        // An edit whose timestamp was preserved (size changed): detected.
        set("a.md", "four plus more", t0);
        let preserved = scan_of(&root);
        assert_ne!(preserved.signature, base.signature);

        // A future-dated file does not mask a later edit to another file.
        set(
            "future.md",
            "from the future",
            t0 + Duration::from_secs(10 * 365 * 86_400),
        );
        let with_future = scan_of(&root);
        set("b.md", "beta edited", t0 + Duration::from_secs(60));
        let edited_under_future = scan_of(&root);
        assert_ne!(edited_under_future.signature, with_future.signature);
        assert_eq!(
            edited_under_future.latest_mtime(),
            with_future.latest_mtime(),
            "the old max-mtime gate would have seen nothing"
        );

        // A non-indexable file appearing or changing does not trigger a scan.
        fs::write(root.join("image.png"), b"\x89PNG not text").expect("png");
        let with_png = scan_of(&root);
        assert_eq!(with_png.signature, edited_under_future.signature);
        assert!(with_png
            .listing
            .file_names
            .contains(&"image.png".to_string()));
        assert_eq!(with_png.selected.len(), 3);
        assert!(with_png.complete());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn doc_paths_come_from_the_canonical_project_path_even_through_symlinks() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("symlink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let real = base.join("real").join("proj");
        let elsewhere = base.join("elsewhere");
        fs::create_dir_all(&real).expect("real");
        fs::create_dir_all(&elsewhere).expect("elsewhere");
        fs::write(real.join("f.md"), "a regular file in the project").expect("f");
        fs::write(elsewhere.join("target.md"), "the linked file's content").expect("target");
        std::os::unix::fs::symlink(elsewhere.join("target.md"), real.join("link.md"))
            .expect("file symlink");
        std::os::unix::fs::symlink(base.join("real"), base.join("link-dir")).expect("dir symlink");

        // The project is reached through a symlinked parent directory. The symlinked file
        // inside it is not indexed (regular files only, as before), so nothing outside the
        // project tree can enter it.
        let via_link = base.join("link-dir").join("proj");
        let corpus = collect(&via_link, &FileManifest::new(), true);
        let canonical = normalize_path(&real.to_string_lossy());
        assert_eq!(corpus.doc.path, canonical);
        assert_eq!(corpus.files.len(), 1);
        assert_eq!(corpus.files[0].rel_path, "f.md");
        for f in &corpus.files {
            assert_eq!(f.doc_path, canonical.join(&f.rel_path).to_string_lossy());
            assert!(!f.doc_path.contains("link-dir"));
        }
        for c in &corpus.chunks {
            assert_eq!(
                c.doc_path,
                canonical.join(&c.doc_rel_path).to_string_lossy()
            );
        }
        // The same project collected through its real path yields the same keys, so rows
        // written one way are kept by a scan done the other way.
        let direct = collect(&real, &FileManifest::new(), true);
        let mut a: Vec<&str> = corpus.files.iter().map(|f| f.doc_path.as_str()).collect();
        let mut b: Vec<&str> = direct.files.iter().map(|f| f.doc_path.as_str()).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert_eq!(corpus.scan_signature, direct.scan_signature);
        let keep = PruneKeepSet::from_corpus(&corpus);
        for f in &direct.files {
            assert!(keep.keeps_doc(&f.doc_path));
        }
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn a_selected_file_that_cannot_be_read_makes_the_scan_incomplete() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("readfail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mk root");
        fs::write(root.join("a.md"), "readable file").expect("a");
        fs::write(root.join("b.md"), "this one will be unreadable").expect("b");
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(root.join("b.md"), fs::Permissions::from_mode(0o000)).expect("chmod");
        if fs::read(root.join("b.md")).is_ok() {
            // Running as root: permissions do not bite; nothing to test here.
            let _ = fs::remove_dir_all(&root);
            return;
        }
        let scan = scan_of(&root);
        assert!(scan.complete(), "the walk itself saw both files");
        assert_eq!(scan.selected.len(), 2);
        let corpus = collect_project_corpus(
            &root,
            &scan,
            &ScanCaps::default(),
            100_000,
            &FileManifest::new(),
            true,
            &ScanSettings::default(),
        );
        fs::set_permissions(root.join("b.md"), fs::Permissions::from_mode(0o644)).expect("restore");
        assert!(
            !corpus.complete,
            "a read failure after the walk is an incomplete scan"
        );
        assert_eq!(corpus.files_unreadable, 1);
        assert_eq!(corpus.files.len(), 1);
        assert_eq!(corpus.files[0].rel_path, "a.md");
        // Nothing of b.md would be pruned: reindex skips prune for incomplete corpora and the
        // signature is not published (see run_native_index_with_embedder).
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_whitespace_only_file_is_selected_with_zero_chunks_by_the_real_collector() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("blank-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mk root");
        fs::write(root.join("notes.md"), "some notes").expect("notes");
        fs::write(root.join("blank.py"), "   \n\t\n\n").expect("blank");
        let corpus = collect(&root, &FileManifest::new(), true);
        assert!(corpus.complete);
        assert_eq!(corpus.files.len(), 2);
        let blank = corpus
            .files
            .iter()
            .find(|f| f.rel_path == "blank.py")
            .expect("blank.py is a selected file");
        assert!(blank.rechunked);
        assert_eq!(blank.chunk_count, 0);
        assert!(corpus.chunks.iter().all(|c| c.doc_rel_path != "blank.py"));
        let keep = PruneKeepSet::from_corpus(&corpus);
        assert!(keep.keeps_doc(&blank.doc_path));
        assert!(keep.rel_paths.contains("blank.py"));
        assert!(!keep.keeps_chunk(&blank.doc_path, 0), "its old chunks go");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn prune_deletes_commit_the_lance_dirty_marker_with_them() {
        // The marker must stay set: `remove_projects_not_in` deletes from LanceDB and clears it
        // whenever the process-global handle is open, so no other test may hold it open.
        let _lance = crate::test_support::lance_isolation();
        let conn = conn();
        seed_two_file_project(&conn);
        assert_eq!(app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(), None);
        let scan = corpus(Vec::new(), vec![scanned("a.md", "/p/alpha/a.md", false, 1)]);
        let keep = PruneKeepSet::from_corpus(&scan);
        let dry = prune_stale_project_rows(&conn, 1, &keep, true).expect("dry run");
        assert_eq!(dry.chunks, 2);
        assert_eq!(
            app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
            None,
            "a dry run marks nothing"
        );
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        assert_eq!(out.chunks, 2);
        assert_eq!(
            app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
            Some("1".to_string()),
            "the marker is set in the same transaction as the sqlite deletes"
        );
        // Removing a project sets it too.
        conn.execute_batch(
            "INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed) VALUES (2, '/p/beta', 'beta', 'beta', 0, 0);
             INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
             VALUES (20, 2, '/p/beta/x.md', 'x.md', 0, 0, 1, 'h', 'x', 0);",
        )
        .unwrap();
        app_state_set(&conn, APP_STATE_LANCE_DIRTY, "0").unwrap();
        let (removed, chunks) =
            remove_projects_not_in(&conn, &["/p/alpha".to_string()], &HashSet::new()).unwrap();
        assert_eq!((removed, chunks), (1, 1));
        assert_eq!(
            app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
            Some("1".to_string())
        );
    }

    #[test]
    fn a_rechunked_file_with_zero_chunks_keeps_its_manifest_row_and_code_intel() {
        let conn = conn();
        seed_two_file_project(&conn);
        // tools/b.py was re-read and produced no chunks (nothing embeddable), but it is still a
        // selected file: its manifest row, symbols and imports come from the file, not from
        // chunks, and stay. Only its stored chunks go.
        let scan = corpus(
            Vec::new(),
            vec![
                scanned("a.md", "/p/alpha/a.md", false, 1),
                scanned("tools/b.py", "/p/alpha/tools/b.py", true, 0),
            ],
        );
        let keep = PruneKeepSet::from_corpus(&scan);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        let mut removed = out.chunk_ids.clone();
        removed.sort();
        assert_eq!(removed, vec![11, 12]);
        assert_eq!(out.manifest_rows, 0);
        assert_eq!(out.symbol_rows, 0);
        assert_eq!(out.import_rows, 0);
        assert_eq!(out.edge_rows, 0);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE rel_path = 'tools/b.py'"
            ),
            1
        );
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM symbols"), 2);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM file_imports"), 1);
        // A file that is not selected at all loses everything (the deletion case).
        let gone = corpus(Vec::new(), vec![scanned("a.md", "/p/alpha/a.md", false, 1)]);
        let out = prune_stale_project_rows(&conn, 1, &PruneKeepSet::from_corpus(&gone), false)
            .expect("prune gone");
        assert_eq!(out.manifest_rows, 1);
        assert_eq!(out.symbol_rows, 2);
        assert_eq!(out.import_rows, 1);
    }

    #[test]
    fn a_touched_identical_file_is_marked_stat_changed_and_its_manifest_row_is_refreshed() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("restat-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mk root");
        fs::write(root.join("a.md"), "same content before and after").expect("a");
        let full = collect(&root, &FileManifest::new(), false);
        let manifest: FileManifest = full
            .files
            .iter()
            .map(|f| {
                (
                    f.rel_path.clone(),
                    FileManifestEntry {
                        size: f.size,
                        mtime: f.mtime,
                        content_hash: f.content_hash.clone(),
                        chunk_count: f.chunk_count,
                    },
                )
            })
            .collect();
        let later = SystemTime::now() + Duration::from_secs(300);
        crate::test_support::set_mtime(&root.join("a.md"), later);
        let touched = collect(&root, &manifest, false);
        let f = &touched.files[0];
        assert!(!f.rechunked);
        assert!(f.stat_changed);
        assert!(touched.chunks.is_empty());
        assert!(f.mtime > manifest["a.md"].mtime);
        assert_eq!(f.content_hash, manifest["a.md"].content_hash);
        assert_eq!(f.chunk_count, 1);

        // Through reindex: no embedder call, no prune, the manifest row carries the new stat.
        // `reindex_project_chunks` opens and syncs the process-global LanceDB handle when
        // anything is embedded or pruned; nothing is here, but the path takes the store lock.
        let _lance = crate::test_support::lance_isolation();
        let conn = conn();
        conn.execute_batch(
            "INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed) VALUES (1, '/p/x', 'x', 'x', 0, 0);",
        )
        .unwrap();
        let cfg = ConfigValues::from_map(HashMap::new());
        let embedder = crate::test_support::TestEmbedder::new(&cfg, false);
        upsert_file_manifest(
            &conn,
            1,
            "a.md",
            &f.doc_path,
            f.size,
            manifest["a.md"].mtime,
            &f.content_hash,
            1,
        )
        .unwrap();
        let out = reindex_project_chunks(
            &root,
            &conn,
            1,
            &identity(Some(64)),
            &embedder,
            &touched,
            now_ts(),
            None,
            false,
        )
        .expect("reindex");
        assert_eq!(out.vectors, 0);
        assert_eq!(out.rows, 1);
        assert_eq!(
            out.reused, 0,
            "nothing was re-chunked, so nothing was reused"
        );
        assert_eq!(out.pruned.chunks, 0);
        assert!(embedder.embedded_texts().is_empty());
        let stored: f64 = conn
            .query_row(
                "SELECT file_mtime FROM project_files WHERE rel_path = 'a.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((stored - f.mtime).abs() < 0.001);
        let _ = fs::remove_dir_all(&root);
    }

    fn seed_two_file_project(conn: &Connection) {
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/alpha', 'alpha', 'alpha', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (10, 1, '/p/alpha/a.md', 'a.md', 0, 0, 5, 'h10', 'a zero', 0),
       (11, 1, '/p/alpha/tools/b.py', 'tools/b.py', 0, 0, 5, 'h11', 'b zero', 0),
       (12, 1, '/p/alpha/tools/b.py', 'tools/b.py', 0, 1, 5, 'h12', 'b one', 0);
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector, embed_input_hash, normalized, pipeline_version)
VALUES (10, 'm', 1, 1.0, x'00000000', 'i10', 1, 1),
       (11, 'm', 1, 1.0, x'00000000', 'i11', 1, 1),
       (12, 'm', 1, 1.0, x'00000000', 'i12', 1, 1);
INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed)
VALUES (1, 'a.md', '/p/alpha/a.md', 1, 0, 'x', 1, 0),
       (1, 'tools/b.py', '/p/alpha/tools/b.py', 1, 0, 'y', 2, 0);
INSERT INTO symbols(project_id, doc_path, doc_rel_path, name, kind, line_start, line_end, updated_at)
VALUES (1, '/p/alpha/tools/b.py', 'tools/b.py', 'load', 'function', 1, 2, 0),
       (1, '/p/alpha/tools/b.py', 'tools/b.py', 'Config', 'class', 4, 9, 0);
INSERT INTO file_imports(project_id, source_doc_path, import_kind, raw_specifier, updated_at)
VALUES (1, '/p/alpha/tools/b.py', 'import', 'os', 0);
INSERT INTO file_dependency_edges(project_id, source_doc_path, target_doc_path, edge_kind, updated_at)
VALUES (1, '/p/alpha/tools/b.py', 'a.md', 'imports', 0);
"#,
        )
        .expect("seed two-file project");
    }

    #[test]
    fn a_vanished_file_loses_every_row_and_unchanged_files_keep_theirs() {
        let conn = conn();
        seed_two_file_project(&conn);
        // The scan lists a.md as unchanged and no longer sees tools/b.py.
        let scan = corpus(Vec::new(), vec![scanned("a.md", "/p/alpha/a.md", false, 1)]);
        let keep = PruneKeepSet::from_corpus(&scan);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");

        let mut removed = out.chunk_ids.clone();
        removed.sort();
        assert_eq!(removed, vec![11, 12]);
        assert_eq!(out.files, 1);
        assert_eq!(out.manifest_rows, 1);
        assert_eq!(out.symbol_rows, 2);
        assert_eq!(out.import_rows, 1);
        assert_eq!(out.edge_rows, 1);

        // Everything of tools/b.py is gone ...
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'tools/b.py'"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE chunk_id IN (11, 12)"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE rel_path = 'tools/b.py'"
            ),
            0
        );
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM symbols"), 0);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM file_imports"), 0);
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM file_dependency_edges"),
            0
        );
        // ... and the unchanged file, which the scan never re-chunked, is untouched.
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM project_chunks WHERE id = 10"),
            1
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_chunk_vectors WHERE chunk_id = 10"
            ),
            1
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM project_files WHERE rel_path = 'a.md'"
            ),
            1
        );
    }

    #[test]
    fn an_unchanged_file_is_never_pruned_when_only_another_file_was_rechunked() {
        let conn = conn();
        seed_two_file_project(&conn);
        // a.md was edited (its chunk 0 re-produced); tools/b.py is unchanged and kept whole.
        let edited = ProjectChunk {
            doc_path: "/p/alpha/a.md".to_string(),
            doc_rel_path: "a.md".to_string(),
            doc_mtime: 0.0,
            chunk_index: 0,
            token_count: 1,
            text_hash: "h10b".to_string(),
            text: "a zero edited".to_string(),
            chunk_kind: "text_window".to_string(),
            symbol_name: String::new(),
            parent_context: String::new(),
            line_start: 0,
            line_end: 0,
            context_header: String::new(),
        };
        let scan = corpus(
            vec![edited],
            vec![
                scanned("a.md", "/p/alpha/a.md", true, 1),
                scanned("tools/b.py", "/p/alpha/tools/b.py", false, 2),
            ],
        );
        let keep = PruneKeepSet::from_corpus(&scan);
        let out = prune_stale_project_rows(&conn, 1, &keep, false).expect("prune");
        assert!(out.is_empty(), "{:?}", out);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 3);
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM project_chunk_vectors"),
            3
        );
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM symbols"), 2);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM file_imports"), 1);
        assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_files"), 2);
    }

    #[test]
    fn code_intel_refresh_replaces_rows_even_when_the_extraction_is_empty() {
        let conn = conn();
        seed_two_file_project(&conn);
        let path = Path::new("/p/alpha/tools/b.py");
        let files = vec!["a.md".to_string(), "tools/b.py".to_string()];

        // Edited down to a docstring: no symbols, no imports. The stale rows must go.
        let empty = extract_code_intel(
            code_intel::LanguageId::Python,
            &path.to_string_lossy(),
            "tools/b.py",
            "\"\"\"Kept for old notebooks; defines nothing.\"\"\"\n",
        );
        let (symbols, imports) =
            store_code_intel_extraction(&conn, 1, &empty, Path::new("/p/alpha"), &files)
                .expect("refresh empty");
        assert_eq!((symbols, imports), (0, 0));
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM symbols WHERE doc_path = '/p/alpha/tools/b.py'"
            ),
            0
        );
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM file_imports WHERE source_doc_path = '/p/alpha/tools/b.py'"
            ),
            0
        );
        assert_eq!(
            count(&conn, "SELECT COUNT(*) FROM file_dependency_edges WHERE source_doc_path = '/p/alpha/tools/b.py'"),
            0
        );

        // Real code populates them again.
        let real = extract_code_intel(
            code_intel::LanguageId::Python,
            &path.to_string_lossy(),
            "tools/b.py",
            "import os\n\ndef load(path):\n    return os.path.exists(path)\n",
        );
        let (symbols, imports) =
            store_code_intel_extraction(&conn, 1, &real, Path::new("/p/alpha"), &files)
                .expect("refresh populated");
        assert!(symbols >= 1, "symbols={}", symbols);
        assert!(imports >= 1, "imports={}", imports);
        assert_eq!(
            count(
                &conn,
                "SELECT COUNT(*) FROM symbols WHERE doc_path = '/p/alpha/tools/b.py'"
            ),
            symbols as i64
        );
    }
}

pub(crate) fn rebuild_relationship_edges(
    conn: &Connection,
    docs_by_id: &HashMap<i64, ProjectDoc>,
) -> Result<i64, String> {
    if docs_by_id.is_empty() {
        return Ok(0);
    }
    let mut node_meta: HashMap<i64, (String, HashSet<String>, HashSet<String>)> = HashMap::new();
    for (project_id, doc) in docs_by_id {
        let signal_tokens: HashSet<String> = word_tokens(&doc.summary)
            .into_iter()
            .filter(|t| !is_graph_stopword(t))
            .collect();
        let mut name_tokens: HashSet<String> = word_tokens(&doc.title).into_iter().collect();
        if let Some(name) = doc.path.file_name().and_then(|s| s.to_str()) {
            for t in word_tokens(name) {
                name_tokens.insert(t);
            }
        }
        node_meta.insert(
            *project_id,
            (
                doc.path.to_string_lossy().to_string(),
                signal_tokens,
                name_tokens,
            ),
        );
    }

    // Load project centroid vectors for embedding-based similarity
    let project_vectors = load_project_vectors(conn)?;

    // Load import-based cross-project edges: for each project, find which other
    // projects its files depend on via resolved file_dependency_edges.
    let import_cross_edges = load_import_cross_project_edges(conn)?;

    let mut total_edges = 0i64;
    for (src_id, (_src_path, src_tokens, src_name_tokens)) in &node_meta {
        let mut edges: Vec<(String, String, f64)> = Vec::new();

        for (dst_id, (dst_path, _dst_tokens, dst_name_tokens)) in &node_meta {
            if src_id == dst_id {
                continue;
            }

            // Jaccard/mention-based edges (existing logic)
            let mention = src_tokens.intersection(dst_name_tokens).count();
            let mention_score = if mention >= 2 {
                0.8
            } else if mention == 1 && dst_name_tokens.len() <= 2 {
                0.4
            } else {
                0.0
            };
            let overlap = jaccard(src_name_tokens, dst_name_tokens);
            let overlap_score = if overlap >= 0.25 { overlap } else { 0.0 };
            let jaccard_weight = mention_score + overlap_score;
            if jaccard_weight >= 0.45 {
                let rounded = ((jaccard_weight.min(2.0) * 1000.0).round()) / 1000.0;
                edges.push((dst_path.clone(), "semantic_related".to_string(), rounded));
            }

            // Embedding-based similarity: cosine between project centroid vectors
            if let (Some((src_vec, src_norm)), Some((dst_vec, dst_norm))) =
                (project_vectors.get(src_id), project_vectors.get(dst_id))
            {
                let sim = cosine_raw(src_vec, dst_vec, *src_norm, *dst_norm);
                if sim >= 0.40 {
                    let rounded = ((sim.min(1.0) * 1000.0).round()) / 1000.0;
                    edges.push((
                        dst_path.clone(),
                        "embedding_similarity".to_string(),
                        rounded,
                    ));
                }
            }

            // Import-derived edges: file A in src_project imports file B in dst_project
            if let Some(import_count) = import_cross_edges
                .get(src_id)
                .and_then(|targets| targets.get(dst_id))
            {
                // Weight scales with number of import relationships
                let weight = match *import_count {
                    1 => 0.55,
                    2..=5 => 0.75,
                    _ => 0.90,
                };
                edges.push((dst_path.clone(), "imports_from".to_string(), weight));
            }
        }

        edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        if edges.len() > 60 {
            edges.truncate(60);
        }
        set_project_edges(conn, *src_id, &edges)?;
        total_edges += edges.len() as i64;
    }
    Ok(total_edges)
}

pub(crate) fn set_project_edges(
    conn: &Connection,
    project_id: i64,
    edges: &[(String, String, f64)],
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM project_edges WHERE src_project_id = ?1",
        params![project_id],
    )
    .map_err(|e| format!("failed clearing project edges: {}", e))?;
    for (dst, kind, weight) in edges {
        conn.execute(
            r#"
INSERT INTO project_edges(src_project_id, dst, kind, weight)
VALUES (?1, ?2, ?3, ?4)
"#,
            params![project_id, dst, kind, weight],
        )
        .map_err(|e| format!("failed inserting project edge: {}", e))?;
    }
    Ok(())
}

/// Load all project centroid vectors from SQLite.
/// Returns project_id -> (vector, norm) pairs.
pub(crate) fn load_project_vectors(
    conn: &Connection,
) -> Result<HashMap<i64, (Vec<f32>, f64)>, String> {
    let mut stmt = conn
        .prepare("SELECT project_id, vector, norm FROM project_vectors")
        .map_err(|e| format!("failed preparing project_vectors query: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed querying project_vectors: {}", e))?;
    let mut out: HashMap<i64, (Vec<f32>, f64)> = HashMap::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating project_vectors: {}", e))?
    {
        let project_id: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading project_id: {}", e))?;
        let blob: Vec<u8> = row
            .get(1)
            .map_err(|e| format!("failed reading vector blob: {}", e))?;
        let norm: f64 = row
            .get(2)
            .map_err(|e| format!("failed reading norm: {}", e))?;
        let vector = blob_to_f32_vec(&blob);
        out.insert(project_id, (vector, norm));
    }
    Ok(out)
}

/// Load cross-project import relationships from file_dependency_edges.
/// Returns src_project_id -> { dst_project_id -> import_count }.
/// This finds cases where files in one project import files in another project.
///
/// The target is resolved to an absolute path before the join: a relative target (the
/// resolver's form) is the source project's path plus that relative path, an absolute one is
/// taken as is. Joining on the bare relative path matched any project with a file of the
/// same relative name (`src/util.py` in two unrelated projects) and produced `imports_from`
/// edges that were not there. The count is edges, not the target's chunk rows.
pub(crate) fn load_import_cross_project_edges(
    conn: &Connection,
) -> Result<HashMap<i64, HashMap<i64, usize>>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    e.project_id AS src_project_id,
    pc.project_id AS dst_project_id,
    COUNT(DISTINCT e.source_doc_path || char(10) || e.target_doc_path) AS edge_count
FROM file_dependency_edges e
JOIN projects p ON p.id = e.project_id
JOIN project_chunks pc
    ON pc.doc_path = CASE
        WHEN substr(e.target_doc_path, 1, 1) = '/' THEN e.target_doc_path
        ELSE p.path || '/' || e.target_doc_path
    END
    AND pc.project_id != e.project_id
GROUP BY e.project_id, pc.project_id
"#,
        )
        .map_err(|e| format!("failed preparing import cross-project query: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed querying import cross-project edges: {}", e))?;
    let mut out: HashMap<i64, HashMap<i64, usize>> = HashMap::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating import cross-project rows: {}", e))?
    {
        let src: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading src_project_id: {}", e))?;
        let dst: i64 = row
            .get(1)
            .map_err(|e| format!("failed reading dst_project_id: {}", e))?;
        let count: i64 = row
            .get(2)
            .map_err(|e| format!("failed reading edge_count: {}", e))?;
        out.entry(src).or_default().insert(dst, count as usize);
    }
    Ok(out)
}

pub(crate) fn is_graph_stopword(tok: &str) -> bool {
    matches!(
        tok,
        "the"
            | "and"
            | "for"
            | "with"
            | "from"
            | "into"
            | "this"
            | "that"
            | "project"
            | "file"
            | "files"
            | "readme"
            | "docs"
            | "notes"
            | "src"
            | "test"
            | "tests"
    )
}

pub(crate) fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    if inter == 0 {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

#[cfg(test)]
mod small_fix_tests {
    use super::*;
    use crate::db::init_schema;
    use crate::embed::{BedrockEmbedder, LocalHashEmbedder};
    use rusqlite::Connection;

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        init_schema(&conn).expect("init schema");
        conn
    }

    #[test]
    fn cross_project_import_edges_join_on_absolute_paths_not_relative_names() {
        let conn = conn();
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/alpha', 'alpha', 'a', 0, 0), (2, '/p/beta', 'beta', 'b', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (10, 1, '/p/alpha/main.py', 'main.py', 0, 0, 5, 'h10', 'alpha main', 0),
       (11, 1, '/p/alpha/src/util.py', 'src/util.py', 0, 0, 5, 'h11', 'alpha util', 0),
       (20, 2, '/p/beta/src/util.py', 'src/util.py', 0, 0, 5, 'h20', 'beta util zero', 0),
       (21, 2, '/p/beta/src/util.py', 'src/util.py', 0, 1, 5, 'h21', 'beta util one', 0);
INSERT INTO file_dependency_edges(project_id, source_doc_path, target_doc_path, edge_kind, updated_at)
VALUES (1, '/p/alpha/main.py', 'src/util.py', 'imports', 0);
"#,
        )
        .expect("seed");
        // alpha's main.py imports alpha's own src/util.py: the same relative name exists in
        // beta, which used to produce a spurious alpha -> beta edge.
        let edges = load_import_cross_project_edges(&conn).expect("edges");
        assert!(edges.is_empty(), "{:?}", edges);

        // A target that really lives in another project (absolute form) still counts, once
        // per edge rather than once per chunk of the target file.
        conn.execute(
            "INSERT INTO file_dependency_edges(project_id, source_doc_path, target_doc_path, edge_kind, updated_at) VALUES (1, '/p/alpha/main.py', '/p/beta/src/util.py', 'imports', 0)",
            [],
        )
        .unwrap();
        let edges = load_import_cross_project_edges(&conn).expect("edges");
        assert_eq!(
            edges.get(&1).and_then(|m| m.get(&2)),
            Some(&1),
            "{:?}",
            edges
        );
        assert!(!edges.contains_key(&2));
    }

    #[test]
    fn cohere_queries_are_sent_as_search_query_and_documents_as_search_document() {
        let cohere = BedrockEmbedder::new_with_config("cohere.embed-english-v3", None);
        let docs = cohere.request_payload_batch(&["chunk text".to_string()]);
        assert_eq!(docs["input_type"], "search_document");
        let query = cohere.cohere_payload(&["what is the plan".to_string()], "search_query");
        assert_eq!(query["input_type"], "search_query");
        assert_eq!(query["texts"][0], "what is the plan");
        let single = cohere.request_payload("chunk text");
        assert_eq!(single["input_type"], "search_document");
        let titan = BedrockEmbedder::new_with_config("amazon.titan-embed-text-v2:0", None);
        assert!(titan.request_payload("x").get("inputText").is_some());
        assert!(!titan.is_cohere_model() && cohere.is_cohere_model());
        // Models without asymmetric inputs embed queries exactly like documents.
        let hash = LocalHashEmbedder::new(64);
        assert_eq!(
            hash.embed_query("otters").unwrap(),
            hash.embed_one("otters").unwrap()
        );
    }
}
