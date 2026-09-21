//! The watcher: fswatch streaming and polling loops, watch-target derivation and the path filter, the event loop, and LanceDB compaction.

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};
use std::{env, process, thread};

use crate::api::path_basename;
use crate::config::{
    config_path, data_dir, db_path, load_config_values, ConfigValues, ScanSettings,
};
use crate::db::{
    acquire_writer_lock_for_watch, ensure_retrieval_backend_ready, lance_store_is_open, open_db_rw,
    with_lance_store, TrackedRoot,
};
use crate::embed::ensure_native_embed_backend;
use crate::index::{run_native_index, run_native_index_verifying, IndexRunOptions, IndexStats};
use crate::lance_store;
use crate::scan::{discover_projects_full, resolve_roots, IndexScope};
use crate::util::{
    arg_value, chrono_like_now, command_exists, format_bytes, format_duration_ms,
    normalize_lexical, normalize_path,
};

pub(crate) struct FswatchStream {
    child: std::process::Child,
    rx: Receiver<PathBuf>,
}

impl Drop for FswatchStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(crate) fn start_fswatch_stream(roots: &[PathBuf]) -> Result<FswatchStream, String> {
    if roots.is_empty() {
        return Err("no watch roots configured".to_string());
    }
    let mut cmd = Command::new("fswatch");
    cmd.arg("-0").arg("-r").arg("--latency").arg("0.2");
    for root in roots {
        cmd.arg(root);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed launching fswatch: {}", e))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed acquiring fswatch stdout".to_string())?;
    let (tx, rx) = mpsc::channel::<PathBuf>();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<u8> = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(0, &mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    while matches!(buf.last(), Some(0 | b'\n' | b'\r')) {
                        buf.pop();
                    }
                    if buf.is_empty() {
                        continue;
                    }
                    let raw = String::from_utf8_lossy(&buf).trim().to_string();
                    if raw.is_empty() {
                        continue;
                    }
                    if tx.send(PathBuf::from(raw)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(FswatchStream { child, rx })
}

pub(crate) fn watch_stats_changed(stats: &IndexStats) -> bool {
    stats.updated_projects > 0
        || stats.removed_projects > 0
        || stats.vectorized_projects > 0
        || stats.chunk_vectors > 0
        || stats.vector_failures > 0
        || stats.projects_failed > 0
        || !stats.stopped.is_empty()
        || !stats.lance_error.is_empty()
}

pub(crate) fn print_watch_tick(label: &str, stats: &IndexStats, quiet: bool) {
    if !watch_stats_changed(stats) && quiet {
        return;
    }
    let now = chrono_like_now();
    println!(
        "[{}] {} updated={} removed={} vectorized={} chunk_vectors={} skipped={} files={}/{}/{} unreadable={} evicted={} documents={}/{} chunks_embedded={} chunks_reused={} chunks_deleted={} lance_repaired={} failed={}",
        now,
        label,
        stats.updated_projects,
        stats.removed_projects,
        stats.vectorized_projects,
        stats.chunk_vectors,
        stats.skipped_projects,
        stats.files_selected,
        stats.files_unchanged,
        stats.files_rechunked,
        stats.files_unreadable,
        stats.files_evicted_by_cap,
        stats.documents_extracted,
        stats.documents_failed,
        stats.chunks_embedded,
        stats.chunks_reused,
        stats.chunks_deleted,
        stats.lance_repaired,
        stats.projects_failed
    );
    if stats.vector_failures > 0 {
        println!(
            "[{}] {} vector_failures={}",
            now, label, stats.vector_failures
        );
    }
    for line in &stats.failures {
        println!("[{}] {} failed: {}", now, label, line);
    }
    if !stats.stopped.is_empty() {
        println!("[{}] {} stopped early: {}", now, label, stats.stopped);
    }
    if !stats.lance_error.is_empty() {
        println!(
            "[{}] {} lancedb: {} (dirty marker set; repaired on the next run)",
            now, label, stats.lance_error
        );
    }
}

pub(crate) fn path_depth(path: &Path) -> usize {
    path.components().count()
}

pub(crate) fn longest_prefix_match<'a>(
    path: &Path,
    candidates: &'a [PathBuf],
) -> Option<&'a PathBuf> {
    let mut best: Option<&PathBuf> = None;
    let mut best_depth = 0usize;
    for candidate in candidates {
        if path == candidate || path.starts_with(candidate) {
            let d = path_depth(candidate);
            if d >= best_depth {
                best_depth = d;
                best = Some(candidate);
            }
        }
    }
    best
}

pub(crate) fn normalize_watch_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return normalize_lexical(path);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    normalize_lexical(&cwd.join(path))
}

/// Whether a changed path can affect the index at all. Dropped before it is even queued:
/// anything under a hidden directory, a built-in or `skip_dir_names` directory or a macOS
/// `.app` bundle, hidden files, and files whose suffix the indexer never reads (per
/// `settings`). A path without a suffix (a directory being created, moved or deleted) stays
/// relevant, because that is how a new or vanished project announces itself.
pub(crate) fn watch_path_relevant(path: &Path, settings: &ScanSettings) -> bool {
    for comp in path.components() {
        if let Component::Normal(name) = comp {
            let seg = name.to_string_lossy();
            if seg.starts_with('.') || settings.is_skip_dir(&seg) {
                return false;
            }
        }
    }
    let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext.is_empty() {
        return true;
    }
    settings.is_indexable_suffix(&format!(".{}", ext))
}

/// One line naming what a watch run is about to scan, printed even with `--quiet` so the log
/// always shows which project a tick belonged to.
pub(crate) fn watch_scope_line(scope: &IndexScope) -> String {
    let mut names: Vec<String> = Vec::new();
    if let IndexScope::Targets { roots, projects } = scope {
        for root in roots {
            names.push(format!(
                "{} (discovery)",
                path_basename(&root.to_string_lossy())
            ));
        }
        for project in projects {
            names.push(path_basename(&project.to_string_lossy()));
        }
    }
    format!(
        "[{}] watch: changes in {}",
        chrono_like_now(),
        names.join(", ")
    )
}

/// Compact LanceDB when it holds more than `threshold` versions on disk; 0 disables. Called
/// from the watcher's periodic sweep, so a busy day of small writes cannot grow the store
/// without bound (Slice 0 measured 1,793 versions and 3.7 GB before this existed).
/// Compaction is due when the table holds more versions *or* more data fragments than the
/// threshold. Versions count writes since the last cleanup; fragments count the files a
/// vector search has to open, which keep growing while the version count is held down by
/// the prune pass below.
pub(crate) fn lance_compaction_due(versions: usize, fragments: usize, threshold: i64) -> bool {
    threshold > 0 && (versions as i64 > threshold || fragments as i64 > threshold)
}

/// Watcher-side LanceDB hygiene after a sweep or a polling pass.
///
/// Two steps. When the compaction threshold is passed, fragments are rewritten and every
/// version older than the grace period is dropped (`lance_store::optimize`). Otherwise, when
/// the store still holds versions older than the grace period, they are dropped on their own
/// (`lance_store::prune_versions`), which is what returns the disk space a compaction could
/// not: the versions committed within the grace window before a compaction keep every
/// pre-compaction fragment alive (measured on the live store: 625 MB -> 1.04 GB right after
/// a compaction, with 148 dead files of 652 MB still on disk 45 minutes later), and only a
/// later cleanup can remove them once those versions have aged.
pub(crate) fn maybe_compact_lance(cwd: &Path, cfg: &ConfigValues, label: &str) {
    if !lance_store_is_open() {
        return;
    }
    let lance_dir = data_dir(cwd).join("lance");
    let grace = cfg.lance_version_grace_secs.max(0) as u64;
    let versions_before = lance_store::version_count(&lance_dir);
    let fragments_before = lance_store::fragment_count(&lance_dir);
    if lance_compaction_due(
        versions_before,
        fragments_before,
        cfg.lance_compact_versions,
    ) {
        let size_before = lance_store::dir_size_bytes(&lance_dir);
        let t = Instant::now();
        match with_lance_store(|store| lance_store::optimize(store, grace)) {
            Ok(report) => println!(
                "[{}] {} lancedb compacted: versions {} -> {}, fragments {} -> {} (threshold {}), {} -> {} on disk, rewrote {} fragments into {}, dropped {} old versions ({})",
                chrono_like_now(),
                label,
                versions_before,
                lance_store::version_count(&lance_dir),
                fragments_before,
                lance_store::fragment_count(&lance_dir),
                cfg.lance_compact_versions,
                format_bytes(size_before),
                format_bytes(lance_store::dir_size_bytes(&lance_dir)),
                report.fragments_removed,
                report.fragments_added,
                report.old_versions,
                format_duration_ms(t.elapsed().as_millis() as u64)
            ),
            Err(e) => eprintln!(
                "[{}] {} warning: LanceDB compaction failed ({} versions, {} fragments on disk): {}",
                chrono_like_now(),
                label,
                versions_before,
                fragments_before,
                e
            ),
        }
        return;
    }
    if lance_store::stale_version_count(&lance_dir, grace) == 0 {
        return;
    }
    let size_before = lance_store::dir_size_bytes(&lance_dir);
    let t = Instant::now();
    match with_lance_store(|store| lance_store::prune_versions(store, grace)) {
        Ok(report) => println!(
            "[{}] {} lancedb pruned: versions {} -> {}, fragments {} -> {}, {} -> {} on disk, dropped {} old versions ({})",
            chrono_like_now(),
            label,
            versions_before,
            lance_store::version_count(&lance_dir),
            fragments_before,
            lance_store::fragment_count(&lance_dir),
            format_bytes(size_before),
            format_bytes(lance_store::dir_size_bytes(&lance_dir)),
            report.old_versions,
            format_duration_ms(t.elapsed().as_millis() as u64)
        ),
        Err(e) => eprintln!(
            "[{}] {} warning: LanceDB version prune failed ({} versions on disk): {}",
            chrono_like_now(),
            label,
            versions_before,
            e
        ),
    }
}

/// Map changed paths onto what to index: a path inside a known project targets that project
/// (as a project, never as a root to discover); a path under a tracked root but outside every
/// known project (a new directory) sends discovery to that root.
///
/// Nothing is forced: the targeted project goes through the ordinary change gate and the
/// manifest fast path, so only the files whose size, mtime or content changed are read and
/// chunked, and the full keep set still prunes what left the corpus (a deleted file changes
/// the scan signature, so the project is re-collected and its rows go). Only `refresh`
/// forces a full re-chunk.
pub(crate) fn derive_watch_targets(
    pending_paths: &HashSet<PathBuf>,
    tracked_roots: &[TrackedRoot],
    settings: &ScanSettings,
) -> IndexScope {
    if pending_paths.is_empty() || tracked_roots.is_empty() {
        return IndexScope::projects(Vec::new());
    }
    let discovery = discover_projects_full(tracked_roots, settings);
    let root_paths: Vec<PathBuf> = tracked_roots.iter().map(|r| r.path.clone()).collect();
    let mut root_set: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();
    let mut project_set: std::collections::BTreeSet<PathBuf> = std::collections::BTreeSet::new();

    for raw in pending_paths {
        let path = normalize_watch_path(raw);
        if !watch_path_relevant(&path, settings) {
            continue;
        }
        let Some(root) = longest_prefix_match(&path, &root_paths) else {
            continue;
        };
        match longest_prefix_match(&path, &discovery.projects) {
            // A root-files project owns only the files directly under the root; a change
            // deeper down is a new or unknown directory that discovery has to place.
            Some(project)
                if !(discovery.is_shallow(project) && path.parent() != Some(project.as_path())) =>
            {
                project_set.insert(normalize_path(&project.to_string_lossy()));
            }
            _ => {
                root_set.insert(normalize_path(&root.to_string_lossy()));
            }
        }
    }

    IndexScope::Targets {
        roots: root_set.into_iter().collect(),
        projects: project_set.into_iter().collect(),
    }
}

/// The tracked roots as the store has them now. Reloaded before every watcher run so a root
/// removed meanwhile is ignored (its events map to nothing) and an added one is watched.
pub(crate) fn load_watch_roots(cwd: &Path, cfg: &ConfigValues) -> Result<Vec<TrackedRoot>, String> {
    let conn = open_db_rw(&db_path(cwd))?;
    resolve_roots(&conn, cfg, None)
}

/// Reload the roots and, when the set of root paths changed, restart the fswatch stream on
/// the new set (fswatch watches the paths it was started with; a root added later would be
/// silent, a removed one would keep sending events).
pub(crate) fn refresh_watch_roots(
    cwd: &Path,
    cfg: &ConfigValues,
    stream: &mut FswatchStream,
    root_paths: &mut Vec<PathBuf>,
) -> Result<Vec<TrackedRoot>, String> {
    let roots = load_watch_roots(cwd, cfg)?;
    let current: Vec<PathBuf> = roots.iter().map(|r| r.path.clone()).collect();
    if current != *root_paths {
        println!(
            "[{}] watch: tracked roots changed ({} -> {}); restarting the file watcher",
            chrono_like_now(),
            root_paths.len(),
            current.len()
        );
        let _ = stream.child.kill();
        let _ = stream.child.wait();
        *stream = start_fswatch_stream(&current)?;
        *root_paths = current;
    }
    Ok(roots)
}

/// Event paths verified by content hash in one watcher batch at most (see
/// [`run_native_index_verifying`]); a larger batch relies on the stat gate alone.
pub(crate) const WATCH_VERIFY_MAX: usize = 200;

pub(crate) fn run_watch_event_loop(
    cwd: &Path,
    cfg: &ConfigValues,
    tracked_roots: Vec<TrackedRoot>,
    interval_seconds: f64,
    debounce_ms: u64,
    quiet: bool,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, "watch event-loop")?;
    let settings = ScanSettings::from_cfg(cfg);
    let mut root_paths: Vec<PathBuf> = tracked_roots.iter().map(|r| r.path.clone()).collect();
    let mut stream = start_fswatch_stream(&root_paths)?;
    if !quiet {
        println!(
            "watch mode: fswatch event stream (debounce={}ms, full sweep every {:.1}s)",
            debounce_ms, interval_seconds
        );
    }
    let debounce = Duration::from_millis(debounce_ms.max(100));
    let sweep_every = Duration::from_secs_f64(interval_seconds.max(1.0));
    let mut pending_paths: HashSet<PathBuf> = HashSet::new();
    let mut last_event_at: Option<Instant> = None;
    let mut last_sweep_at = Instant::now();

    loop {
        match stream.rx.recv_timeout(Duration::from_millis(200)) {
            Ok(path) => {
                // Filtered on receipt: an event under `.git`, `node_modules`, a `.app`
                // bundle, a `skip_dir_names` directory or for a `.png` never starts the
                // debounce clock, let alone a scan.
                if watch_path_relevant(&normalize_watch_path(&path), &settings) {
                    pending_paths.insert(path);
                    last_event_at = Some(Instant::now());
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err("fswatch stream disconnected".to_string());
            }
        }
        if let Some(status) = stream
            .child
            .try_wait()
            .map_err(|e| format!("failed checking fswatch status: {}", e))?
        {
            return Err(format!("fswatch exited unexpectedly: {}", status));
        }

        if !pending_paths.is_empty()
            && last_event_at
                .map(|t| t.elapsed() >= debounce)
                .unwrap_or(false)
        {
            // Roots as of now: events under a root removed since the last run map to no
            // target and are dropped here.
            let roots = refresh_watch_roots(cwd, cfg, &mut stream, &mut root_paths)?;
            let scope = derive_watch_targets(&pending_paths, &roots, &settings);
            // The event paths themselves: their content hash is checked even when size and
            // mtime are unchanged (an edit that restores the timestamp). A storm of events
            // (a checkout, a generated tree) falls back to the stat gate instead of hashing
            // hundreds of files.
            let mut verify_paths: HashSet<PathBuf> = pending_paths
                .iter()
                .map(|p| normalize_watch_path(p))
                .filter(|p| watch_path_relevant(p, &settings))
                .collect();
            if verify_paths.len() > WATCH_VERIFY_MAX {
                println!(
                    "[{}] watch: {} event paths in one batch; content verification skipped for this batch (size and mtime gate only)",
                    chrono_like_now(),
                    verify_paths.len()
                );
                verify_paths.clear();
            }
            pending_paths.clear();
            last_event_at = None;
            if scope.is_empty() {
                continue;
            }
            println!("{}", watch_scope_line(&scope));
            ensure_retrieval_backend_ready(cfg, true, "watch events")?;
            let writer = acquire_writer_lock_for_watch(cwd)?;
            // Event scans use the manifest fast path (nothing forced): a project with one
            // edited file reads and embeds that file only; the files the events named are
            // verified by content hash.
            let stats = run_native_index_verifying(
                cwd,
                cfg,
                &writer,
                IndexRunOptions {
                    scope,
                    force_all: false,
                    force_paths: HashSet::new(),
                    remove_missing: false,
                    reason: "watch events",
                },
                verify_paths,
                !quiet,
            )?;
            print_watch_tick("event", &stats, quiet);
        }

        if last_sweep_at.elapsed() >= sweep_every {
            refresh_watch_roots(cwd, cfg, &mut stream, &mut root_paths)?;
            ensure_retrieval_backend_ready(cfg, true, "watch sweep")?;
            let writer = acquire_writer_lock_for_watch(cwd)?;
            let stats = run_native_index(
                cwd,
                cfg,
                &writer,
                IndexRunOptions {
                    scope: IndexScope::AllRoots,
                    force_all: false,
                    force_paths: HashSet::new(),
                    remove_missing: true,
                    reason: "watch sweep",
                },
                !quiet,
            )?;
            print_watch_tick("sweep", &stats, quiet);
            maybe_compact_lance(cwd, cfg, "sweep");
            last_sweep_at = Instant::now();
        }
    }
}

/// One polling pass: take the writer lock, index every root (roots are read from the store
/// by the run itself), compact when due, and release the lock before returning, so nothing is
/// held while the loop sleeps and a manual `index`, `prune` or root change can run between
/// passes.
pub(crate) fn run_watch_poll_once(
    cwd: &Path,
    cfg: &ConfigValues,
    quiet: bool,
) -> Result<IndexStats, String> {
    ensure_retrieval_backend_ready(cfg, true, "watch poll")?;
    let writer = acquire_writer_lock_for_watch(cwd)?;
    let stats = run_native_index(
        cwd,
        cfg,
        &writer,
        IndexRunOptions {
            scope: IndexScope::AllRoots,
            force_all: false,
            force_paths: HashSet::new(),
            remove_missing: true,
            reason: "watch poll",
        },
        !quiet,
    )?;
    print_watch_tick("poll", &stats, quiet);
    maybe_compact_lance(cwd, cfg, "poll");
    drop(writer);
    Ok(stats)
}

pub(crate) fn run_watch_polling_loop(
    cwd: &Path,
    cfg: &ConfigValues,
    interval_seconds: f64,
    quiet: bool,
) {
    if !quiet {
        println!("watch mode: polling");
    }
    loop {
        if let Err(e) = run_watch_poll_once(cwd, cfg, quiet) {
            eprintln!("error: watch poll failed: {}", e);
            process::exit(1);
        }
        thread::sleep(Duration::from_secs_f64(interval_seconds));
    }
}

pub(crate) fn run_watch_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]"
        );
        return;
    }

    let mut interval_seconds = 30.0f64;
    let mut debounce_ms: u64 = 900;
    let mut once = false;
    let mut quiet = false;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--once" => {
                once = true;
            }
            "--quiet" => {
                quiet = true;
            }
            "--interval" => {
                i += 1;
                let raw = arg_value(args, i, "--interval");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            "--debounce-ms" => {
                i += 1;
                let raw = arg_value(args, i, "--debounce-ms");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--interval=") => {
                let raw = other.trim_start_matches("--interval=");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            other if other.starts_with("--debounce-ms=") => {
                let raw = other.trim_start_matches("--debounce-ms=");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    interval_seconds = interval_seconds.max(1.0);
    debounce_ms = debounce_ms.clamp(100, 5000);

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_native_embed_backend(&cfg, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!("hint: use `retrivio init --embed-backend <ollama|bedrock>`");
        process::exit(1);
    });
    ensure_retrieval_backend_ready(&cfg, true, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let tracked_roots = resolve_roots(&conn, &cfg, None).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if !quiet {
        println!(
            "watching tracked roots (config root: {}, tracked={})",
            cfg.root.display(),
            tracked_roots.len()
        );
        println!(
            "interval: {:.1}s  debounce: {}ms",
            interval_seconds, debounce_ms
        );
        println!("press Ctrl-C to stop");
    }

    let writer = acquire_writer_lock_for_watch(&cwd).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let bootstrap = run_native_index(
        &cwd,
        &cfg,
        &writer,
        IndexRunOptions {
            scope: IndexScope::AllRoots,
            force_all: false,
            force_paths: HashSet::new(),
            remove_missing: true,
            reason: "watch bootstrap",
        },
        !quiet,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: watch bootstrap failed: {}", e);
        process::exit(1);
    });
    print_watch_tick("bootstrap", &bootstrap, quiet);
    drop(writer);
    if once {
        return;
    }

    if command_exists("fswatch") {
        match run_watch_event_loop(
            &cwd,
            &cfg,
            tracked_roots,
            interval_seconds,
            debounce_ms,
            quiet,
        ) {
            Ok(()) => return,
            Err(e) => {
                if !quiet {
                    eprintln!("watch: fswatch mode failed ({}); using polling fallback", e);
                }
            }
        }
    }
    if !quiet {
        println!("watch: fswatch unavailable; using polling fallback");
    }
    run_watch_polling_loop(&cwd, &cfg, interval_seconds, quiet);
}

#[cfg(test)]
mod watcher_and_compaction_tests {
    use super::*;
    use crate::config::{config_set_value, ConfigValues, ScanSettings};
    use crate::db::{init_schema, TrackedRoot};
    use crate::rank::{
        document_text, summary_dir_evidence, summary_page_factor, summary_page_stem,
        text_references_project, SummaryPageJudge, SUMMARY_PAGE_FACTOR,
    };
    use crate::scan::IndexScope;
    use crate::util::normalize_path;
    use rusqlite::Connection;
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::{fs, process};

    #[test]
    fn watch_filter_drops_noise_before_any_scan() {
        let settings = ScanSettings {
            extra_skip_dirs: ["tmp".to_string()].into_iter().collect(),
            index_documents: true,
        };
        let dropped = [
            "/r/proj/tmp/scratch.md",
            "/r/proj/.git/index",
            "/r/proj/.git/refs/heads/main.md",
            "/r/proj/node_modules/pkg/README.md",
            "/r/proj/Foo.app/Contents/Resources/notes.md",
            "/r/proj/worktrees/x/ci/pipeline.yml",
            "/r/proj/.worktrees/y/a.md",
            "/r/proj/docs/diagram.png",
            "/r/proj/.hidden/notes.md",
            "/r/proj/.DS_Store",
            "/r/proj/build/out.md",
            "/r/proj/target/debug/x.rs",
        ];
        for p in dropped {
            assert!(
                !watch_path_relevant(Path::new(p), &settings),
                "{} must not trigger a scan",
                p
            );
        }
        let kept = [
            "/r/proj/docs/notes.md",
            "/r/proj/src/main.rs",
            "/r/proj/report.docx",
            "/r/proj/deck.pptx",
            "/r/proj/newdir",
            "/r/proj/Makefile",
        ];
        for p in kept {
            assert!(
                watch_path_relevant(Path::new(p), &settings),
                "{} must trigger a scan",
                p
            );
        }
        // Without `tmp` in the skip set the scratch file is a normal note.
        assert!(watch_path_relevant(
            Path::new("/r/proj/tmp/scratch.md"),
            &ScanSettings::default()
        ));
        // With documents off, a document edit is not a relevant event either.
        let no_docs = ScanSettings {
            index_documents: false,
            ..ScanSettings::default()
        };
        assert!(!watch_path_relevant(
            Path::new("/r/proj/report.docx"),
            &no_docs
        ));
        assert!(watch_path_relevant(
            Path::new("/r/proj/page.html"),
            &no_docs
        ));
        let line = watch_scope_line(&IndexScope::Targets {
            roots: vec![PathBuf::from("/r")],
            projects: vec![PathBuf::from("/r/proj"), PathBuf::from("/r/other")],
        });
        assert!(
            line.ends_with("watch: changes in r (discovery), proj, other"),
            "{}",
            line
        );
    }

    #[test]
    fn events_under_a_root_no_longer_tracked_map_to_nothing() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("watch-roots-{}", process::id()));
        let _ = fs::remove_dir_all(&base);
        let root_a = base.join("root-a");
        let root_b = base.join("root-b");
        fs::create_dir_all(root_a.join("proj")).unwrap();
        fs::create_dir_all(root_b.join("other")).unwrap();
        fs::write(root_a.join("proj").join("notes.md"), "a").unwrap();
        fs::write(root_b.join("other").join("notes.md"), "b").unwrap();
        let tracked = |paths: &[&Path]| -> Vec<TrackedRoot> {
            paths
                .iter()
                .map(|p| TrackedRoot {
                    path: normalize_path(&p.to_string_lossy()),
                    exclude_patterns: Vec::new(),
                })
                .collect()
        };
        let mut pending: HashSet<PathBuf> = HashSet::new();
        pending.insert(root_b.join("other").join("notes.md"));

        let targets = |scope: &IndexScope| -> Vec<PathBuf> { scope.target_paths() };
        // Both roots tracked: the event maps to root-b's project.
        let scope = derive_watch_targets(
            &pending,
            &tracked(&[&root_a, &root_b]),
            &ScanSettings::default(),
        );
        assert!(!scope.is_empty());
        assert!(
            targets(&scope).iter().any(|p| p.ends_with("other")),
            "{:?}",
            scope
        );

        // root-b removed since: the same event maps to nothing, so nothing is re-indexed.
        let scope = derive_watch_targets(&pending, &tracked(&[&root_a]), &ScanSettings::default());
        assert!(scope.is_empty(), "{:?}", scope);

        // An event under the remaining root still scans its project.
        pending.insert(root_a.join("proj").join("notes.md"));
        let scope = derive_watch_targets(&pending, &tracked(&[&root_a]), &ScanSettings::default());
        assert!(!scope.is_empty());
        assert_eq!(targets(&scope).len(), 1);
        assert!(
            targets(&scope).iter().any(|p| p.ends_with("proj")),
            "{:?}",
            scope
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn summary_pages_are_files_named_after_another_project() {
        let stems: HashSet<String> = [
            "202608-acme-rollout",
            "widget-service",
            "ai-activity-widget-service",
            "globex",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // The stem half: named after another indexed project, not after its own.
        assert_eq!(
            summary_page_stem(
                "projects/202608-Acme-Rollout.md",
                "/r/202609-handoff",
                &stems
            ),
            Some("202608-acme-rollout".to_string())
        );
        assert_eq!(
            summary_page_stem("AI-Activity-widget-service.md", "/r/202609-handoff", &stems),
            Some("ai-activity-widget-service".to_string())
        );
        assert!(
            summary_page_stem("202608-acme-rollout.md", "/r/202608-acme-rollout", &stems).is_none()
        );
        assert!(summary_page_stem("projects/notes.md", "/r/202609-handoff", &stems).is_none());
        assert!(summary_page_stem("README.md", "/r/202609-handoff", &stems).is_none());
        assert!(summary_page_stem("", "/r/x", &stems).is_none());
        // Directory evidence: projects/, summaries/, handoff*/ at any depth; nothing else.
        assert!(summary_dir_evidence("projects/202608-acme-rollout.md"));
        assert!(summary_dir_evidence("notes/Summaries/globex.md"));
        assert!(summary_dir_evidence("handoffs/2026-09/globex.md"));
        assert!(summary_dir_evidence("Handoff-notes/globex.md"));
        assert!(!summary_dir_evidence("docs/Globex.md"));
        assert!(!summary_dir_evidence("integrations/Acme.md"));
        assert!(!summary_dir_evidence("Globex.md"));
        // Text evidence: three distinct paths of the named project.
        let digest = "Summary of globex: see globex/docs/design.md, globex/src/main.rs and globex/README.md for details; also /Users/me/globex/notes.txt";
        assert!(text_references_project(digest, "globex"));
        assert!(!text_references_project(
            "Globex versus Initech: throughput modes, globex/ pricing, and one link globex/docs/design.md",
            "globex"
        ));
        assert!(
            !text_references_project("globex/a.md globex/a.md globex/a.md", "globex"),
            "distinct paths"
        );
        assert!(
            !text_references_project("myglobex/a.md myglobex/b.md myglobex/c.md", "globex"),
            "word boundary"
        );
        assert!(!text_references_project("", "globex"));

        // End to end against a store: the same stem, three verdicts.
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/r/vendor-compare', 'sc', 's', 0, 0), (2, '/r/202609-handoff', 'h', 'h', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (10, 1, '/r/vendor-compare/docs/Globex.md', 'docs/Globex.md', 0, 0, 5, 'h10', 'Globex throughput modes compared with Initech for the vendor workshop', 0),
       (11, 2, '/r/202609-handoff/projects/globex.md', 'projects/globex.md', 0, 0, 5, 'h11', 'Digest of the Globex work', 0),
       (12, 2, '/r/202609-handoff/globex.md', 'globex.md', 0, 0, 5, 'h12', 'Digest: globex/docs/design.md, globex/src/lib.rs, globex/README.md', 0),
       (13, 2, '/r/202609-handoff/globex-notes/globex.md', 'globex-notes/globex.md', 0, 0, 5, 'h13', 'Just a note that mentions globex once', 0),
       (14, 2, '/r/202609-handoff/notes/globex.md', 'notes/globex.md', 0, 0, 5, 'h14', 'Digest part one, see globex/docs/design.md', 0),
       (15, 2, '/r/202609-handoff/notes/globex.md', 'notes/globex.md', 0, 1, 5, 'h15', 'part two: globex/src/lib.rs', 0),
       (16, 2, '/r/202609-handoff/notes/globex.md', 'notes/globex.md', 0, 2, 5, 'h16', 'part three: globex/README.md and a closing remark', 0);
"#,
        )
        .unwrap();
        let mut judge = SummaryPageJudge {
            stems: stems.clone(),
            verdicts: HashMap::new(),
        };
        // A document that shares the name, in an ordinary directory, no path references.
        assert!(!judge.is_summary_page(
            &conn,
            "/r/vendor-compare/docs/Globex.md",
            "docs/Globex.md",
            "/r/vendor-compare"
        ));
        // The digest by directory.
        assert!(judge.is_summary_page(
            &conn,
            "/r/202609-handoff/projects/globex.md",
            "projects/globex.md",
            "/r/202609-handoff"
        ));
        // The digest by text, in one chunk.
        assert!(judge.is_summary_page(
            &conn,
            "/r/202609-handoff/globex.md",
            "globex.md",
            "/r/202609-handoff"
        ));
        // Name only: neither directory nor text evidence.
        assert!(!judge.is_summary_page(
            &conn,
            "/r/202609-handoff/globex-notes/globex.md",
            "globex-notes/globex.md",
            "/r/202609-handoff"
        ));
        // Paths spread over three chunks: the document is judged whole, every chunk of it
        // gets the same verdict, and the text is read once (one cached verdict per document).
        for _ in 0..3 {
            assert!(judge.is_summary_page(
                &conn,
                "/r/202609-handoff/notes/globex.md",
                "notes/globex.md",
                "/r/202609-handoff"
            ));
        }
        // Five documents judged (the name-only vendor file included), each once.
        assert_eq!(judge.verdicts.len(), 5);
        assert_eq!(
            document_text(&conn, "/r/202609-handoff/notes/globex.md")
                .matches("globex/")
                .count(),
            3
        );
        // The factor: the constant unless the sweep override is set (read once per process).
        assert!((SUMMARY_PAGE_FACTOR - 0.85).abs() < 1e-12);
        let f = summary_page_factor();
        assert!((0.5..=1.0).contains(&f));
    }

    #[test]
    fn compaction_is_due_only_above_the_threshold() {
        assert!(!lance_compaction_due(0, 0, 200));
        assert!(!lance_compaction_due(200, 0, 200));
        assert!(lance_compaction_due(201, 0, 200));
        // Fragments count too: many small writes with the versions pruned between sweeps.
        assert!(!lance_compaction_due(3, 200, 200));
        assert!(lance_compaction_due(3, 201, 200));
        assert!(!lance_compaction_due(10_000, 10_000, 0), "0 disables");
        assert!(!lance_compaction_due(10_000, 10_000, -1));
        let mut cfg = ConfigValues::from_map(HashMap::new());
        assert_eq!(cfg.lance_compact_versions, 200);
        config_set_value(&mut cfg, "lance_compact_versions", "50").unwrap();
        assert_eq!(cfg.lance_compact_versions, 50);
        config_set_value(&mut cfg, "lance_compact_versions", "-4").unwrap();
        assert_eq!(cfg.lance_compact_versions, 0);
        assert!(config_set_value(&mut cfg, "lance_compact_versions", "many").is_err());
    }
}
