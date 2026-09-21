//! End-to-end tests of the index run family through `TestStore` and `TestEmbedder`: incremental
//! runs, forced refreshes, failure handling, the writer lock and LanceDB repair.

use crate::config::ScanSettings;
use crate::db::{
    app_state_get, app_state_set, is_index_busy_error, with_lance_store, TrackedRoot, WriterLock,
    APP_STATE_EMBED_FINGERPRINT, APP_STATE_LANCE_DIRTY, APP_STATE_SCAN_CAPS_FINGERPRINT,
};
use crate::embed::Embedder;
use crate::embed::HASH_BACKEND_MODEL;
use crate::index::*;
use crate::lance_store;
use crate::scan::{
    collect_project_corpus, project_scan, resolve_index_targets, FileManifest, IndexScope, ScanCaps,
};
use crate::test_support::*;
use crate::util::{blob_to_f32_vec, normalize_path};
use crate::watch::{derive_watch_targets, run_watch_poll_once};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashSet;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs, thread};

fn write(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, text).expect("write");
}

fn two_project_root(store: &TestStore, name: &str) -> PathBuf {
    let root = store.corpus_root(name);
    write(
        &root.join("alpha").join("a.md"),
        "alpha one talks about storage layers",
    );
    write(
        &root.join("alpha").join("b.md"),
        "alpha two talks about api endpoints",
    );
    write(
        &root.join("beta").join("notes.md"),
        "beta notes on authentication",
    );
    root
}

/// Every chunk row has a vector, and every vector's `embed_input_hash` is the hash of the
/// exact embedder input of the text the row holds now.
fn assert_vectors_match_their_chunk_text(conn: &Connection, expected_rows: usize) {
    let without_vector = count(
        conn,
        "SELECT COUNT(*) FROM project_chunks c LEFT JOIN project_chunk_vectors v ON v.chunk_id = c.id WHERE v.chunk_id IS NULL",
    );
    assert_eq!(without_vector, 0, "every chunk has a vector");
    let mut stmt = conn
        .prepare("SELECT c.context_header, c.text, v.embed_input_hash FROM project_chunks c JOIN project_chunk_vectors v ON v.chunk_id = c.id")
        .unwrap();
    let rows: Vec<(String, String, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(rows.len(), expected_rows);
    for (header, text, hash) in rows {
        assert_eq!(hash, embed_input_hash(&embed_input_for(&header, &text)));
    }
}

fn chunk_rows_for(conn: &Connection, rel: &str) -> (i64, i64) {
    let chunks = count(
        conn,
        &format!(
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = '{}'",
            rel
        ),
    );
    let vectors = count(
        conn,
        &format!(
            "SELECT COUNT(*) FROM project_chunk_vectors v JOIN project_chunks c ON c.id = v.chunk_id WHERE c.doc_rel_path = '{}'",
            rel
        ),
    );
    (chunks, vectors)
}

/// The watcher's event scan (slice 4): an edited file is the only one read and embedded;
/// a touch without an edit embeds nothing; a deleted file loses exactly its rows. Before
/// this, event scans forced a full re-chunk of the touched project (`files=81/0/81`).
#[test]
fn watch_event_scans_take_the_manifest_fast_path_and_keep_deletions() {
    let store = TestStore::new("watch-fast-path");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.chunks_embedded, 3);
    let conn = store.conn();

    let root_norm = normalize_path(&root.to_string_lossy());
    let alpha = root_norm.join("alpha");
    let tracked = vec![TrackedRoot {
        path: root_norm.clone(),
        exclude_patterns: Vec::new(),
    }];
    // `verify` is what the event loop passes: the event paths themselves.
    let event_with = |paths: &[PathBuf], verify: bool| -> IndexStats {
        let pending: HashSet<PathBuf> = paths.iter().cloned().collect();
        let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
        assert_eq!(scope, IndexScope::projects(vec![alpha.clone()]));
        let writer = WriterLock::try_acquire(&store.dir).expect("writer lock");
        run_native_index_with_embedder(
            &store.dir,
            &cfg,
            &writer,
            &embedder,
            IndexRunOptions {
                scope,
                force_all: false,
                force_paths: HashSet::new(),
                remove_missing: false,
                reason: "watch events",
            },
            if verify { pending } else { HashSet::new() },
            false,
        )
        .expect("event scan")
    };
    let event = |paths: &[PathBuf]| -> IndexStats { event_with(paths, true) };

    // One edited file: read, chunked and embedded alone; the other file is carried.
    write(
        &alpha.join("b.md"),
        "alpha two rewritten about ui and frontend widgets",
    );
    let edited = event(&[alpha.join("b.md")]);
    assert_eq!(edited.updated_projects, 1);
    assert_eq!(edited.skipped_projects, 0, "beta is outside the scope");
    assert_eq!(
        (
            edited.files_selected,
            edited.files_unchanged,
            edited.files_rechunked
        ),
        (2, 1, 1)
    );
    assert_eq!(edited.chunks_embedded, 1);
    assert_eq!(edited.chunks_deleted, 0);
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'b.md' AND text LIKE '%frontend widgets%'"
        ),
        1
    );

    // A touch without an edit: the stat differs, the content hash says unchanged, nothing
    // is chunked or embedded.
    let touched = fs::OpenOptions::new()
        .write(true)
        .open(alpha.join("a.md"))
        .expect("open a.md");
    touched
        .set_modified(SystemTime::now() + Duration::from_secs(7))
        .expect("touch a.md");
    drop(touched);
    let touched = event(&[alpha.join("a.md")]);
    assert_eq!(touched.updated_projects, 1);
    assert_eq!(
        (
            touched.files_selected,
            touched.files_unchanged,
            touched.files_rechunked
        ),
        (2, 2, 0)
    );
    assert_eq!(touched.chunks_embedded, 0);
    assert_eq!(touched.chunks_deleted, 0);

    // A deleted file: the event scan removes exactly its rows and reads nothing else.
    fs::remove_file(alpha.join("a.md")).expect("rm a.md");
    let deleted = event(&[alpha.join("a.md")]);
    assert_eq!(deleted.updated_projects, 1);
    assert_eq!(
        (
            deleted.files_selected,
            deleted.files_unchanged,
            deleted.files_rechunked
        ),
        (1, 1, 0)
    );
    assert_eq!(deleted.chunks_embedded, 0);
    assert_eq!(deleted.chunks_deleted, 1);
    assert_eq!(chunk_rows_for(&conn, "a.md"), (0, 0));
    assert_eq!(chunk_rows_for(&conn, "b.md"), (1, 1));
    assert_eq!(chunk_rows_for(&conn, "notes.md"), (1, 1));

    // Steady state afterwards: a scope without a file to verify (a directory event) is
    // skipped by the gate; an event naming an unchanged file verifies it (read, hashed)
    // and embeds nothing.
    let steady = event_with(&[alpha.join("b.md")], false);
    assert_eq!(steady.skipped_projects, 1);
    assert_eq!(steady.chunks_embedded, 0);
    let steady = event(&[alpha.join("b.md")]);
    assert_eq!(steady.skipped_projects, 0);
    assert_eq!(
        (
            steady.files_selected,
            steady.files_unchanged,
            steady.files_rechunked
        ),
        (1, 1, 0),
        "alpha holds b.md alone since the deletion"
    );
    assert_eq!(steady.chunks_embedded, 0);

    // The edit the stat gate cannot see: same byte length, timestamp restored exactly.
    // Without the event path as a verify target the project is skipped and the index
    // keeps the old text; with it, the file is read, hashed, re-chunked and re-embedded.
    let before = fs::read_to_string(alpha.join("b.md")).unwrap();
    let after = "alpha two rewritten about ux and frontend gadgets";
    assert_eq!(
        before.len(),
        after.len(),
        "the replacement keeps the byte length"
    );
    let mtime = fs::metadata(alpha.join("b.md"))
        .unwrap()
        .modified()
        .unwrap();
    write(&alpha.join("b.md"), after);
    fs::OpenOptions::new()
        .write(true)
        .open(alpha.join("b.md"))
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    assert_eq!(
        fs::metadata(alpha.join("b.md"))
            .unwrap()
            .modified()
            .unwrap(),
        mtime
    );
    let missed = event_with(&[alpha.join("b.md")], false);
    assert_eq!(
        missed.skipped_projects, 1,
        "the stat gate alone misses the edit"
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'b.md' AND text LIKE '%frontend gadgets%'"
        ),
        0
    );
    let verified = event(&[alpha.join("b.md")]);
    assert_eq!(verified.skipped_projects, 0);
    assert_eq!(verified.updated_projects, 1);
    assert_eq!(
        (
            verified.files_selected,
            verified.files_unchanged,
            verified.files_rechunked
        ),
        (1, 0, 1)
    );
    assert_eq!(verified.chunks_embedded, 1);
    assert_eq!(verified.chunks_deleted, 0);
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'b.md' AND text LIKE '%frontend gadgets%'"
        ),
        1
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'b.md' AND text LIKE '%frontend widgets%'"
        ),
        0
    );
    // An event for an unchanged file: verified (read and hashed), nothing re-chunked,
    // nothing embedded; beta, outside the scope, is not touched at all.
    let same = event(&[alpha.join("b.md")]);
    assert_eq!(
        same.skipped_projects, 0,
        "the named file is verified, not skipped"
    );
    assert_eq!(
        (
            same.files_selected,
            same.files_unchanged,
            same.files_rechunked
        ),
        (1, 1, 0)
    );
    assert_eq!(same.chunks_embedded, 0);
    // An event path outside the project's files (a deleted or foreign path) verifies
    // nothing and the gate skips as before.
    let foreign = event_with(&[alpha.join("zzz.md")], true);
    assert_eq!(foreign.skipped_projects, 1);
}

#[test]
fn a_failed_embedding_keeps_the_old_signature_deletes_nothing_and_is_retried_next_run() {
    let store = TestStore::new("fail-embed");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);

    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.updated_projects, 2);
    assert_eq!(first.chunks_embedded, 3);
    let conn = store.conn();
    let alpha_before = project_row(&conn, &root.join("alpha"));
    assert!(alpha_before.scan_signature.starts_with("2:"));
    assert!(!alpha_before.index_in_progress);

    // Edit one file, add another, and make the embedder fail.
    write(
        &root.join("alpha").join("b.md"),
        "alpha two rewritten about ui and frontend",
    );
    write(
        &root.join("alpha").join("c.md"),
        "alpha three brand new file",
    );
    embedder.fail_from_now();
    // The run completes with the failure recorded (a non-zero exit for the CLI): alpha is
    // counted failed, the embedding outage stops the run, nothing else is touched.
    let failed_run = store
        .index(&cfg, &embedder, false)
        .expect("the run reports the failure instead of aborting");
    assert_eq!(failed_run.projects_failed, 1);
    assert_eq!(failed_run.updated_projects, 0);
    assert!(
        failed_run.failures[0].contains("injected embedder failure"),
        "{:?}",
        failed_run.failures
    );
    assert!(
        failed_run.stopped.contains("embedding failed"),
        "{}",
        failed_run.stopped
    );
    let verdict = index_run_verdict(&failed_run).expect_err("non-zero exit");
    assert!(verdict.contains("1 project(s) failed"), "{}", verdict);

    // The project looks exactly as before to the gate, plus the in-progress marker; the
    // unchanged file lost nothing. Chunk rows commit with their vectors, so the failed
    // batch left no row at all: c.md has none, b.md still holds its old text with the
    // vector made from that text, and no row anywhere lacks a matching vector.
    let alpha_failed = project_row(&conn, &root.join("alpha"));
    assert_eq!(alpha_failed.scan_signature, alpha_before.scan_signature);
    assert_eq!(alpha_failed.summary, alpha_before.summary);
    assert!(alpha_failed.index_in_progress);
    assert_eq!(chunk_rows_for(&conn, "a.md"), (1, 1));
    assert_eq!(chunk_rows_for(&conn, "notes.md"), (1, 1));
    assert_eq!(chunk_rows_for(&conn, "b.md"), (1, 1));
    assert_eq!(
        chunk_rows_for(&conn, "c.md"),
        (0, 0),
        "no row without its vector"
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = 'b.md' AND text LIKE '%api endpoints%'"
        ),
        1,
        "the old text stays until its new vector exists"
    );
    assert_vectors_match_their_chunk_text(&conn, 3);
    assert_eq!(
        count(&conn, "SELECT COUNT(*) FROM project_files"),
        3,
        "the manifest was not written for the failed project"
    );

    // Next run: the project is rescanned, only the two changed chunks are embedded, the
    // unchanged one is reused, and the signature moves.
    embedder.heal();
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.updated_projects, 1, "alpha only; beta is skipped");
    assert_eq!(third.skipped_projects, 1);
    assert_eq!(third.chunks_embedded, 2);
    assert_eq!(third.chunks_reused, 1);
    assert_eq!(third.chunks_deleted, 0);
    let alpha_after = project_row(&conn, &root.join("alpha"));
    assert_ne!(alpha_after.scan_signature, alpha_before.scan_signature);
    assert!(alpha_after.scan_signature.starts_with("3:"));
    assert!(!alpha_after.index_in_progress);
    assert_eq!(chunk_rows_for(&conn, "c.md"), (1, 1));
    assert_vectors_match_their_chunk_text(&conn, 4);

    let fourth = store.index(&cfg, &embedder, false).expect("steady state");
    assert_eq!(fourth.skipped_projects, 2);
    assert_eq!(fourth.chunks_embedded, 0);
}

#[test]
fn an_unreadable_subdirectory_prunes_nothing_and_is_counted() {
    let store = TestStore::new("unreadable");
    let root = store.corpus_root("root");
    write(&root.join("alpha").join("a.md"), "alpha top level file");
    write(
        &root.join("alpha").join("sub").join("b.md"),
        "alpha nested file",
    );
    write(&root.join("beta").join("notes.md"), "beta notes");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.chunks_embedded, 3);
    assert_eq!(first.files_unreadable, 0);
    let conn = store.conn();
    let alpha_before = project_row(&conn, &root.join("alpha"));

    let sub = root.join("alpha").join("sub");
    fs::set_permissions(&sub, fs::Permissions::from_mode(0o000)).expect("chmod 000");
    if fs::read_dir(&sub).is_ok() {
        // Running as root: permissions do not bite; nothing to test here.
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).expect("restore");
        return;
    }
    let second = store.index(&cfg, &embedder, false);
    fs::set_permissions(&sub, fs::Permissions::from_mode(0o755)).expect("restore");
    let second = second.expect("second run");
    assert_eq!(
        second.updated_projects, 1,
        "alpha's signature changed (sub/b.md unseen)"
    );
    assert_eq!(second.files_unreadable, 1);
    assert_eq!(second.projects_incomplete, 1);
    assert_eq!(second.files_selected, 1);
    assert_eq!(second.chunks_deleted, 0);
    assert_eq!(second.pruned_chunks, 0);
    assert_eq!(
        chunk_rows_for(&conn, "sub/b.md"),
        (1, 1),
        "the hidden file kept its rows"
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_files WHERE rel_path = 'sub/b.md'"
        ),
        1
    );
    let alpha_incomplete = project_row(&conn, &root.join("alpha"));
    assert_eq!(
        alpha_incomplete.scan_signature, alpha_before.scan_signature,
        "an incomplete scan never advances the signature"
    );
    assert_eq!(
        alpha_incomplete.summary, alpha_before.summary,
        "and keeps the stored summary"
    );
    assert!(!alpha_incomplete.index_in_progress);

    // Readable again: the on-disk signature equals the stored one, so nothing happens.
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.skipped_projects, 2);
    assert_eq!(third.files_unreadable, 0);
}

#[test]
fn projects_under_an_unavailable_root_survive_an_all_roots_index() {
    let store = TestStore::new("root-gone");
    let root_a = two_project_root(&store, "root-a");
    let root_b = store.corpus_root("root-b");
    write(
        &root_b.join("gamma").join("g.md"),
        "gamma on a volume that comes and goes",
    );
    write(
        &root_b.join("delta").join("d.md"),
        "delta on the same volume",
    );
    store.track(&root_a);
    store.track(&root_b);
    let cfg = store.cfg(&root_a, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.tracked_roots, 2);
    assert_eq!(first.updated_projects, 4);

    // The whole root disappears (unmounted volume): its projects must not be removed.
    fs::remove_dir_all(&root_b).expect("remove root b");
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.removed_projects, 0);
    assert_eq!(second.chunks_deleted, 0);
    let conn = store.conn();
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM projects"), 4);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 5);

    // A project directory that vanished under an available root is removed as before.
    fs::remove_dir_all(root_a.join("beta")).expect("remove beta");
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.removed_projects, 1);
    assert_eq!(third.chunks_deleted, 1);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM projects"), 3);
}

#[test]
fn root_level_files_form_a_root_files_project_that_vanishes_with_them() {
    let store = TestStore::new("rootfiles");
    let root = two_project_root(&store, "root");
    write(
        &root.join("loose-notes.md"),
        "loose notes about giraffe enclosure budgets",
    );
    write(&root.join("todo.txt"), "buy hay for the giraffes");
    write(&root.join("logo.png"), "not indexable");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let root_norm = normalize_path(&root.to_string_lossy());
    let base = root_norm.file_name().unwrap().to_string_lossy().to_string();

    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(
        first.total_projects, 3,
        "alpha, beta and the root's own files"
    );
    assert_eq!(first.files_selected, 5);
    assert_eq!(first.chunks_embedded, 5);
    let conn = store.conn();
    let row = project_row(&conn, &root);
    assert_eq!(
        row.title,
        format!("{} (root files)", base.replace('-', " "))
    );
    assert!(
        row.summary
            .starts_with(&format!("project {} (root files)\nindexed_files 2\n", base)),
        "{}",
        row.summary
    );
    let rels = |conn: &Connection, project_id: i64| -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT doc_rel_path FROM project_chunks WHERE project_id = ?1 ORDER BY doc_rel_path")
            .unwrap();
        stmt.query_map(params![project_id], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(rels(&conn, row.id), vec!["loose-notes.md", "todo.txt"]);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 5);
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.skipped_projects, 3);

    // The watcher: a loose file maps to the root-files project; a file in a new directory
    // under the root maps to root discovery.
    let tracked = vec![TrackedRoot {
        path: root_norm.clone(),
        exclude_patterns: Vec::new(),
    }];
    let pending: HashSet<PathBuf> = [root_norm.join("loose-notes.md")].into_iter().collect();
    let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
    assert_eq!(scope, IndexScope::projects(vec![root_norm.clone()]));
    let pending: HashSet<PathBuf> = [root_norm.join("gamma").join("new.md")]
        .into_iter()
        .collect();
    let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
    assert_eq!(scope, IndexScope::roots(vec![root_norm.clone()]));
    let pending: HashSet<PathBuf> = [root_norm.join("alpha").join("a.md")].into_iter().collect();
    let scope = derive_watch_targets(&pending, &tracked, &ScanSettings::default());
    assert_eq!(scope, IndexScope::projects(vec![root_norm.join("alpha")]));
    // A scoped run naming the root-files project scans it shallow.
    let targets =
        resolve_index_targets(&conn, &cfg, &IndexScope::projects(vec![root_norm.clone()]))
            .expect("resolve");
    assert_eq!(targets.projects, vec![root_norm.clone()]);
    assert!(targets.shallow.contains(&root_norm));
    let targets = resolve_index_targets(&conn, &cfg, &IndexScope::AllRoots).expect("resolve");
    assert_eq!(targets.projects.len(), 3);
    assert!(targets.shallow.contains(&root_norm));
    assert!(!targets.shallow.contains(&root_norm.join("alpha")));

    // One loose file deleted: its rows go; the last one deleted: the project goes.
    fs::remove_file(root.join("todo.txt")).expect("rm todo");
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.updated_projects, 1);
    assert_eq!(third.chunks_deleted, 1);
    assert_eq!(third.removed_projects, 0);
    assert_eq!(rels(&conn, row.id), vec!["loose-notes.md"]);
    fs::remove_file(root.join("loose-notes.md")).expect("rm notes");
    let fourth = store.index(&cfg, &embedder, false).expect("fourth run");
    assert_eq!(fourth.total_projects, 2);
    assert_eq!(fourth.removed_projects, 1);
    assert_eq!(fourth.chunks_deleted, 1);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM projects"), 2);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM project_chunks"), 3);
    assert!(get_project_by_path(&conn, &root_norm.to_string_lossy())
        .unwrap()
        .is_none());
}

#[test]
fn the_chunk_cap_evicts_config_and_data_files_before_code_and_notes() {
    let store = TestStore::new("tiercap");
    let root = store.corpus_root("root");
    let project = root.join("orion");
    let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    write(
        &project.join("HANDOFF-2026-06-01.md"),
        "june handoff: the demo needs the bedrock region fixed",
    );
    write(
        &project.join("tools").join("helper.py"),
        "def helper():\n    return 1\n",
    );
    write(
        &project.join("ci").join("pipeline.yml"),
        "stages:\n  - build\n  - test\n",
    );
    write(
        &project.join("ci").join("deploy.yml"),
        "stages:\n  - deploy\n",
    );
    write(&root.join("beta").join("notes.md"), "beta notes");
    // The config files are the newest: the old docs-first order counted .yml as documents
    // and would have kept them over the handoff.
    set_mtime(&project.join("HANDOFF-2026-06-01.md"), t0);
    set_mtime(
        &project.join("tools").join("helper.py"),
        t0 + Duration::from_secs(10),
    );
    set_mtime(
        &project.join("ci").join("pipeline.yml"),
        t0 + Duration::from_secs(100),
    );
    set_mtime(
        &project.join("ci").join("deploy.yml"),
        t0 + Duration::from_secs(200),
    );
    store.track(&root);

    // Learn how many chunks the notes and the code produce, then cap exactly there.
    let wide = ScanCaps::default();
    let full = collect_project_corpus(
        &project,
        &project_scan(
            &project,
            &HashSet::new(),
            &wide,
            false,
            &ScanSettings::default(),
        ),
        &wide,
        100_000,
        &FileManifest::new(),
        true,
        &ScanSettings::default(),
    );
    let kept: i64 = full
        .files
        .iter()
        .filter(|f| !f.rel_path.ends_with(".yml"))
        .map(|f| f.chunk_count)
        .sum();
    assert!(kept >= 2, "{:?}", full.files);
    let cap = kept.to_string();
    let cfg = store.cfg(&root, &[("max_chunks_per_project", cap.as_str())]);
    let caps = ScanCaps::from_cfg(&cfg);
    assert_eq!(caps.max_chunks_per_project as i64, kept);
    let capped = collect_project_corpus(
        &project,
        &project_scan(
            &project,
            &HashSet::new(),
            &caps,
            false,
            &ScanSettings::default(),
        ),
        &caps,
        100_000,
        &FileManifest::new(),
        true,
        &ScanSettings::default(),
    );
    assert_eq!(
        capped.caps_note,
        format!("2 files not indexed (max_chunks_per_project={})", kept)
    );
    assert_eq!(capped.files_evicted_by_cap, 2);
    let kept_rels: Vec<&str> = capped.files.iter().map(|f| f.rel_path.as_str()).collect();
    assert!(
        kept_rels.contains(&"HANDOFF-2026-06-01.md"),
        "{:?}",
        kept_rels
    );
    assert!(kept_rels.contains(&"tools/helper.py"), "{:?}", kept_rels);
    assert!(
        kept_rels.iter().all(|r| !r.ends_with(".yml")),
        "{:?}",
        kept_rels
    );

    // Index everything under the default caps first, then tighten: the cap change alone
    // (no file changed) must make the next incremental run prune the config files.
    let embedder = TestEmbedder::new(&cfg, false);
    let uncapped = store.cfg(&root, &[]);
    let zero = store
        .index(&uncapped, &embedder, false)
        .expect("uncapped run");
    assert_eq!(zero.files_evicted_by_cap, 0);
    let conn = store.conn();
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path LIKE '%.yml'"
        ),
        2
    );
    assert_eq!(
        app_state_get(&conn, APP_STATE_SCAN_CAPS_FINGERPRINT).unwrap(),
        Some(ScanCaps::default().fingerprint(&ScanSettings::default()))
    );
    let first = store
        .index(&cfg, &embedder, false)
        .expect("first capped run");
    assert_eq!(
        first.skipped_projects, 0,
        "a cap change revisits every project"
    );
    assert_eq!(first.files_evicted_by_cap, 2);
    assert_eq!(first.files_truncated_by_cap, 0);
    assert_eq!(first.chunks_deleted, 2, "the two .yml files' chunks");
    assert_eq!(first.chunks_embedded, 0);
    assert_eq!(
        app_state_get(&conn, APP_STATE_SCAN_CAPS_FINGERPRINT).unwrap(),
        Some(caps.fingerprint(&ScanSettings::default()))
    );
    let manifest = |conn: &Connection| -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT rel_path FROM project_files WHERE project_id = (SELECT id FROM projects WHERE path LIKE '%/orion') ORDER BY rel_path")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(
        manifest(&conn),
        vec!["HANDOFF-2026-06-01.md", "tools/helper.py"]
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path LIKE '%.yml'"
        ),
        0
    );

    // Same cap again: the gate skips, nothing is deleted; a forced run evicts the same two
    // files and deletes and embeds nothing.
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.skipped_projects, 2);
    assert_eq!(second.chunks_deleted, 0);
    assert_eq!(
        second.files_evicted_by_cap, 0,
        "a skipped project reports nothing"
    );
    let forced = store.index(&cfg, &embedder, true).expect("forced run");
    assert_eq!(forced.files_evicted_by_cap, 2);
    assert_eq!(forced.chunks_deleted, 0);
    assert_eq!(forced.chunks_embedded, 0);
    assert_eq!(
        manifest(&conn),
        vec!["HANDOFF-2026-06-01.md", "tools/helper.py"]
    );
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, bytes).expect("write bytes");
}

fn chunks_containing(conn: &Connection, needle: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM project_chunks WHERE instr(text, ?1) > 0",
        params![needle],
        |r| r.get(0),
    )
    .expect("count chunks by text")
}

#[test]
fn documents_are_extracted_once_and_failures_are_counted_without_aborting_the_project() {
    use crate::documents::fixtures;
    let store = TestStore::new("documents");
    let root = store.corpus_root("root");
    let project = root.join("docs");
    write(
        &project.join("readme.md"),
        "plain notes about the giraffe project",
    );
    write_bytes(
        &project.join("report.docx"),
        &fixtures::docx(
            &[
                "Giraffe enclosure budget approved by the board.",
                "Second paragraph.",
            ],
            Some("Budget Report"),
        ),
    );
    write_bytes(
        &project.join("deck.pptx"),
        &fixtures::pptx(
            &[&["Welcome slide"], &["Agenda slide"]],
            &[Some("Speaker note about the okapi fence contractor"), None],
        ),
    );
    write_bytes(
        &project.join("book.xlsx"),
        &fixtures::xlsx(&[("Costs", &[&["Item", "Cost"][..], &["Fence", "1200"][..]])]),
    );
    write_bytes(
        &project.join("page.html"),
        b"<html><head><title>Otter Page</title><style>p{color:red}</style>\
<script>var hidden = 'scriptonly';</script></head>\
<body><h1>Otter habitat plan</h1><p>The pond needs a filter.</p></body></html>",
    );
    write_bytes(
        &project.join("notes.pdf"),
        &fixtures::pdf(&[&["Pangolin transport checklist"]]),
    );
    write_bytes(
        &project.join("corrupt.pptx"),
        b"PK\x03\x04 definitely not a zip archive",
    );
    write_bytes(&project.join("big.docx"), &vec![b'x'; 60_000]);
    write(&root.join("beta").join("notes.md"), "beta notes");
    store.track(&root);
    let cfg = store.cfg(&root, &[("max_document_bytes", "50000")]);
    assert_eq!(ScanCaps::from_cfg(&cfg).max_document_bytes, 50_000);
    let embedder = TestEmbedder::new(&cfg, false);

    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.updated_projects, 2);
    assert_eq!(
        first.files_selected, 7,
        "6 in docs + beta; the two never-indexed documents that failed are absent"
    );
    assert_eq!(first.documents_extracted, 5, "docx, pptx, xlsx, html, pdf");
    assert_eq!(
        first.documents_failed, 2,
        "corrupt zip and over the size cap"
    );
    assert_eq!(first.projects_failed, 0);
    let conn = store.conn();
    assert!(
        chunks_containing(&conn, "okapi fence contractor") >= 1,
        "pptx notes"
    );
    assert!(
        chunks_containing(&conn, "Giraffe enclosure budget approved") >= 1,
        "docx"
    );
    assert!(chunks_containing(&conn, "Budget Report") >= 1, "docx title");
    assert!(
        chunks_containing(&conn, "Otter habitat plan") >= 1,
        "html as text"
    );
    assert_eq!(chunks_containing(&conn, "scriptonly"), 0, "script dropped");
    assert_eq!(chunks_containing(&conn, "<style>"), 0, "markup dropped");
    assert!(
        chunks_containing(&conn, "Pangolin transport checklist") >= 1,
        "pdf"
    );
    assert!(
        chunks_containing(&conn, "Fence 1200") >= 1,
        "xlsx row (the tab collapses to a space like all prose whitespace)"
    );
    let manifest = |rel: &str| -> Option<(i64, String, f64)> {
        conn.query_row(
            "SELECT chunk_count, content_hash, file_mtime FROM project_files WHERE rel_path = ?1",
            params![rel],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .expect("manifest query")
    };
    // Never indexed and failed: absent (no manifest row, nothing to lose).
    assert_eq!(manifest("corrupt.pptx"), None);
    assert_eq!(manifest("big.docx"), None);
    assert!(manifest("deck.pptx").expect("deck row").0 >= 1);
    let row = project_row(&conn, &project);
    assert!(
        row.summary.contains("readme.md\nplain notes"),
        "{}",
        row.summary
    );
    assert!(
        !row.summary.contains("PK"),
        "no document bytes in the summary"
    );

    // Nothing changed: skipped, nothing extracted.
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.skipped_projects, 2);
    assert_eq!(second.documents_extracted, 0);
    assert_eq!(second.documents_failed, 0);

    // A touched document (new stat, same bytes) is neither re-extracted nor re-embedded.
    set_mtime(
        &project.join("report.docx"),
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_600_000_000),
    );
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.updated_projects, 1);
    assert_eq!(third.files_rechunked, 0);
    assert_eq!(third.documents_extracted, 0);
    assert_eq!(third.chunks_embedded, 0);
    assert_eq!(
        third.vectorized_projects, 0,
        "the summary does not quote documents"
    );

    // An edited document is extracted and embedded again; its old text is gone.
    write_bytes(
        &project.join("report.docx"),
        &fixtures::docx(
            &["Giraffe enclosure budget rejected; revise by June."],
            None,
        ),
    );
    let fourth = store.index(&cfg, &embedder, false).expect("fourth run");
    assert_eq!(fourth.files_rechunked, 1);
    assert_eq!(fourth.documents_extracted, 1);
    assert!(fourth.chunks_embedded >= 1);
    assert!(chunks_containing(&conn, "budget rejected") >= 1);
    assert_eq!(chunks_containing(&conn, "budget approved"), 0);

    // A repaired file stops failing.
    write_bytes(
        &project.join("corrupt.pptx"),
        &fixtures::pptx(&[&["Fixed deck"]], &[None]),
    );
    let fifth = store.index(&cfg, &embedder, false).expect("fifth run");
    assert_eq!(fifth.documents_extracted, 1);
    assert_eq!(
        fifth.documents_failed, 1,
        "big.docx was never indexed, so every rescan of the project refuses it again"
    );
    assert!(manifest("corrupt.pptx").expect("repaired row").0 >= 1);
    assert_eq!(manifest("big.docx"), None);

    // A previously indexed document that becomes unparseable keeps everything it had:
    // chunks, vectors and manifest row (with the old hash and stat, so it is retried on
    // the next full revisit), while the project itself stays complete: its signature
    // advances and the gate skips it next run. Only the failure counter and warning say
    // anything happened.
    let report_before = manifest("report.docx").expect("report row");
    let report_rows = chunk_rows_for(&conn, "report.docx");
    assert!(report_rows.0 >= 1 && report_rows.0 == report_rows.1);
    write_bytes(
        &project.join("report.docx"),
        b"PK\x03\x04 the document got corrupted on disk",
    );
    let sixth = store.index(&cfg, &embedder, false).expect("sixth run");
    assert_eq!(sixth.updated_projects, 1);
    assert_eq!(
        sixth.documents_failed, 2,
        "report.docx now, big.docx as always"
    );
    assert_eq!(sixth.documents_extracted, 0);
    assert_eq!(sixth.chunks_deleted, 0);
    assert_eq!(sixth.chunks_embedded, 0);
    assert_eq!(sixth.projects_incomplete, 0, "the project is complete");
    assert_eq!(
        sixth.files_selected, 7,
        "docs only (beta is skipped): the carried report.docx counts, big.docx is absent"
    );
    assert_eq!(
        sixth.files_unchanged, 7,
        "report.docx is carried as unchanged"
    );
    assert_eq!(chunk_rows_for(&conn, "report.docx"), report_rows);
    assert!(
        chunks_containing(&conn, "budget rejected") >= 1,
        "old text kept"
    );
    assert_eq!(
        manifest("report.docx"),
        Some(report_before.clone()),
        "manifest entry not advanced"
    );
    let docs_row = project_row(&conn, &project);
    assert!(!docs_row.index_in_progress);
    let seventh = store.index(&cfg, &embedder, false).expect("seventh run");
    assert_eq!(
        seventh.skipped_projects, 2,
        "signature published; not retried"
    );
    assert_eq!(seventh.documents_failed, 0);
    // A forced revisit reports the failures again and still keeps the old content.
    let forced = store.index(&cfg, &embedder, true).expect("forced run");
    assert_eq!(forced.documents_failed, 2);
    assert_eq!(forced.chunks_deleted, 0);
    assert_eq!(chunk_rows_for(&conn, "report.docx"), report_rows);
    assert_eq!(manifest("report.docx"), Some(report_before));
}

#[test]
fn with_index_documents_off_documents_are_not_selected_and_html_stays_raw() {
    use crate::documents::fixtures;
    let store = TestStore::new("documents-off");
    let root = store.corpus_root("root");
    let project = root.join("docs");
    write(&project.join("readme.md"), "plain notes");
    write_bytes(
        &project.join("report.docx"),
        &fixtures::docx(&["Giraffe enclosure budget"], None),
    );
    write_bytes(
        &project.join("page.html"),
        b"<html><body><script>var hidden = 'scriptonly';</script><p>Otter plan</p></body></html>",
    );
    store.track(&root);
    let cfg = store.cfg(&root, &[("index_documents", "false")]);
    assert!(!cfg.index_documents);
    // The run's settings come from its config alone.
    let settings = ScanSettings::from_cfg(&cfg);
    assert!(!settings.is_indexable_suffix(".docx"));
    assert!(settings.is_indexable_suffix(".html"));
    assert!(ScanSettings::default().is_indexable_suffix(".docx"));
    let embedder = TestEmbedder::new(&cfg, false);
    let stats = store.index(&cfg, &embedder, false).expect("run");
    assert_eq!(stats.files_selected, 2, "readme and html only");
    assert_eq!(stats.documents_extracted, 0);
    assert_eq!(stats.documents_failed, 0);
    let conn = store.conn();
    assert_eq!(chunks_containing(&conn, "Giraffe"), 0);
    assert!(
        chunks_containing(&conn, "<script>") >= 1,
        "raw markup, as before"
    );
}

#[test]
fn caps_evict_deterministically_and_report_it() {
    let store = TestStore::new("caps");
    let root = store.corpus_root("root");
    let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    for (name, text) in [
        ("a.md", "alpha first file"),
        ("b.md", "alpha second file"),
        ("c.md", "alpha third file"),
    ] {
        write(&root.join("alpha").join(name), text);
        set_mtime(&root.join("alpha").join(name), t0);
    }
    write(&root.join("beta").join("notes.md"), "beta notes");
    store.track(&root);
    let cfg = store.cfg(&root, &[("max_files_per_project", "2")]);
    assert_eq!(ScanCaps::from_cfg(&cfg).max_files_per_project, 2);
    let embedder = TestEmbedder::new(&cfg, false);

    // Equal mtimes: the path tie-break selects a.md and b.md; c.md is evicted.
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.files_selected, 3, "2 in alpha + 1 in beta");
    assert_eq!(first.files_evicted_by_cap, 1);
    let conn = store.conn();
    let selected = |conn: &Connection| -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT rel_path FROM project_files WHERE project_id = (SELECT id FROM projects WHERE path LIKE '%/alpha') ORDER BY rel_path")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(selected(&conn), vec!["a.md", "b.md"]);

    // Editing c.md makes it the newest: it displaces exactly one file, b.md (a.md wins the
    // tie on path), whose rows are pruned; counters name the eviction.
    write(&root.join("alpha").join("c.md"), "alpha third file, edited");
    set_mtime(
        &root.join("alpha").join("c.md"),
        t0 + Duration::from_secs(60),
    );
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.updated_projects, 1);
    assert_eq!(second.files_evicted_by_cap, 1);
    assert_eq!(second.chunks_embedded, 1, "c.md");
    assert_eq!(second.chunks_deleted, 1, "b.md");
    assert_eq!(selected(&conn), vec!["a.md", "c.md"]);
    assert_eq!(chunk_rows_for(&conn, "b.md"), (0, 0));
    assert_eq!(chunk_rows_for(&conn, "a.md"), (1, 1));

    // A second identical run is skipped by the gate; a forced one evicts the same file
    // again and deletes nothing.
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.skipped_projects, 2);
    assert_eq!(third.chunks_deleted, 0);
    let forced = store.index(&cfg, &embedder, true).expect("forced run");
    assert_eq!(forced.files_evicted_by_cap, 1);
    assert_eq!(forced.chunks_deleted, 0);
    assert_eq!(forced.chunks_embedded, 0);
    assert_eq!(forced.chunks_reused, 3);
    assert_eq!(selected(&conn), vec!["a.md", "c.md"]);
}

#[test]
fn a_changed_embedding_fingerprint_revisits_unchanged_projects_and_reembeds_only_mismatches() {
    let store = TestStore::new("fingerprint");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let plain = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &plain, false).expect("first run");
    assert_eq!(first.chunks_embedded, 3);
    let conn = store.conn();
    assert_eq!(
        app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT).unwrap(),
        Some("ollama:test-local|64|0|1".to_string())
    );
    let second = store.index(&cfg, &plain, false).expect("second run");
    assert_eq!(second.skipped_projects, 2);

    // One chunk already carries a vector produced under the new identity.
    let a_chunk: i64 = conn
        .query_row(
            "SELECT id FROM project_chunks WHERE doc_rel_path = 'a.md'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "UPDATE project_chunk_vectors SET normalized = 1 WHERE chunk_id = ?1",
        params![a_chunk],
    )
    .unwrap();

    // Same files, same model, normalisation flipped: every project is revisited, the one
    // matching vector is reused, the other two (and both summaries) are re-embedded.
    let normalized = TestEmbedder::new(&cfg, true);
    let third = store.index(&cfg, &normalized, false).expect("third run");
    assert_eq!(third.skipped_projects, 0);
    assert_eq!(third.updated_projects, 2);
    assert_eq!(third.files_rechunked, 3);
    assert_eq!(third.chunks_reused, 1);
    assert_eq!(third.chunks_embedded, 2);
    assert_eq!(third.vectorized_projects, 2);
    assert_eq!(
        app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT).unwrap(),
        Some("ollama:test-local|64|1|1".to_string())
    );
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM project_vectors WHERE normalized = 1"
        ),
        2
    );
    let fourth = store.index(&cfg, &normalized, false).expect("fourth run");
    assert_eq!(fourth.skipped_projects, 2);
}

#[test]
fn lance_rows_are_repaired_from_sqlite_without_embedding() {
    let store = TestStore::new("lance-repair");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.chunks_embedded, 3);
    assert_eq!(first.retrieval_synced_chunks, 3);
    let conn = store.conn();
    assert_eq!(
        app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
        Some("0".to_string())
    );
    let ids: Vec<i64> = {
        let mut stmt = conn
            .prepare("SELECT chunk_id FROM project_chunk_vectors ORDER BY chunk_id")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };

    // Damage LanceDB directly: two rows gone, one orphan added; mark dirty as an
    // interrupted write would have.
    with_lance_store(|s| lance_store::delete_chunks(s, &ids[..2])).expect("delete");
    with_lance_store(|s| lance_store::upsert_chunks(s, &[(999_999, vec![0.5f32; 64])]))
        .expect("orphan");
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 2);
    app_state_set(&conn, APP_STATE_LANCE_DIRTY, "1").unwrap();
    let before = embedder.embedded_texts().len();

    let second = store.index(&cfg, &embedder, false).expect("repair run");
    assert_eq!(second.lance_repaired, 2);
    assert_eq!(second.lance_orphans_removed, 1);
    assert_eq!(second.chunks_embedded, 0);
    assert_eq!(second.skipped_projects, 2);
    assert_eq!(
        embedder.embedded_texts().len(),
        before,
        "no embedding call was made"
    );
    let mut lance_ids = with_lance_store(|s| lance_store::list_chunk_ids(s)).unwrap();
    lance_ids.sort();
    assert_eq!(lance_ids, ids);
    assert_eq!(
        app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
        Some("0".to_string())
    );

    // A failed *update*: sqlite committed a new vector and the pending id, LanceDB still
    // holds the old row under the same id. The repair rewrites it (ids alone would not).
    let sqlite_vec: Vec<f32> = {
        let blob: Vec<u8> = conn
            .query_row(
                "SELECT vector FROM project_chunk_vectors WHERE chunk_id = ?1",
                params![ids[0]],
                |r| r.get(0),
            )
            .unwrap();
        blob_to_f32_vec(&blob)
    };
    let mut wrong = vec![0.0f32; 64];
    wrong[0] = 1.0;
    with_lance_store(|s| lance_store::upsert_chunks(s, &[(ids[0], wrong.clone())]))
        .expect("stale row");
    lance_pending_add(&conn, &ids[..1]).unwrap();
    app_state_set(&conn, APP_STATE_LANCE_DIRTY, "1").unwrap();
    let before_stale =
        with_lance_store(|s| lance_store::search_vectors(s, &sqlite_vec, 1)).unwrap();
    assert!(
        !before_stale.contains_key(&ids[0]) || before_stale.len() > 1,
        "precondition: the stale row does not match its sqlite vector best"
    );
    let stale_run = store.index(&cfg, &embedder, false).expect("stale-row run");
    assert_eq!(stale_run.lance_repaired, 1);
    assert_eq!(stale_run.chunks_embedded, 0);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 3);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM lance_pending"), 0);
    let after = with_lance_store(|s| lance_store::search_vectors(s, &sqlite_vec, 1)).unwrap();
    assert_eq!(after.keys().copied().collect::<Vec<i64>>(), vec![ids[0]]);

    // A clean marker means no comparison on the fast path ...
    with_lance_store(|s| lance_store::delete_chunks(s, &ids[..1])).expect("delete one");
    let third = store.index(&cfg, &embedder, false).expect("clean run");
    assert_eq!(third.lance_repaired, 0);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 2);
    // ... while prune's forced reconciliation always repairs.
    let report =
        repair_lance_from_sqlite(&conn, &embedder.model_key(), true).expect("forced repair");
    assert_eq!(report.rebuilt, 1);
    assert_eq!(report.orphans_removed, 0);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 3);

    // An absent marker (a store from before the marker existed) reconciles once.
    conn.execute(
        "DELETE FROM app_state WHERE key = ?1",
        params![APP_STATE_LANCE_DIRTY],
    )
    .unwrap();
    with_lance_store(|s| lance_store::delete_chunks(s, &ids[2..])).expect("delete last");
    let fourth = store.index(&cfg, &embedder, false).expect("first-time run");
    assert_eq!(fourth.lance_repaired, 1);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 3);
}

/// Every row of a project that a failed run must leave alone, as printable lines.
fn project_snapshot(conn: &Connection, project_path: &Path) -> Vec<String> {
    let path = normalize_path(&project_path.to_string_lossy())
        .to_string_lossy()
        .to_string();
    let mut out: Vec<String> = Vec::new();
    let mut push_rows = |sql: &str| {
        let mut stmt = conn.prepare(sql).expect(sql);
        let cols = stmt.column_count();
        let rows = stmt
            .query_map(params![path], |r| {
                let mut line = Vec::new();
                for i in 0..cols {
                    let v: rusqlite::types::Value = r.get(i)?;
                    line.push(match v {
                        rusqlite::types::Value::Null => "NULL".to_string(),
                        rusqlite::types::Value::Integer(n) => n.to_string(),
                        rusqlite::types::Value::Real(f) => format!("{}", f),
                        rusqlite::types::Value::Text(t) => t,
                        rusqlite::types::Value::Blob(b) => format!("blob:{}", b.len()),
                    });
                }
                Ok(line.join("|"))
            })
            .expect("query");
        for row in rows {
            out.push(row.expect("row"));
        }
    };
    push_rows("SELECT 'project', title, summary, project_mtime, scan_signature FROM projects WHERE path = ?1");
    push_rows("SELECT 'chunk', c.id, c.doc_path, c.chunk_index, c.text_hash, c.text FROM project_chunks c JOIN projects p ON p.id = c.project_id WHERE p.path = ?1 ORDER BY c.id");
    push_rows("SELECT 'vector', v.chunk_id, v.model, v.dim, v.embed_input_hash, v.normalized, v.pipeline_version, length(v.vector) FROM project_chunk_vectors v JOIN project_chunks c ON c.id = v.chunk_id JOIN projects p ON p.id = c.project_id WHERE p.path = ?1 ORDER BY v.chunk_id");
    push_rows("SELECT 'manifest', f.rel_path, f.file_size, f.file_mtime, f.content_hash, f.chunk_count FROM project_files f JOIN projects p ON p.id = f.project_id WHERE p.path = ?1 ORDER BY f.rel_path");
    push_rows("SELECT 'pvector', pv.model, pv.dim, length(pv.vector) FROM project_vectors pv JOIN projects p ON p.id = pv.project_id WHERE p.path = ?1");
    push_rows("SELECT 'edge', e.dst, e.kind, e.weight FROM project_edges e JOIN projects p ON p.id = e.src_project_id WHERE p.path = ?1 ORDER BY e.dst, e.kind");
    out
}

#[test]
fn a_polling_pass_releases_the_writer_lock_before_the_loop_sleeps() {
    let store = TestStore::new("poll-lock");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(
        &root,
        &[
            ("embed_backend", "hash"),
            ("embed_model", HASH_BACKEND_MODEL),
        ],
    );
    // One real polling pass (the hash backend embeds offline).
    let stats = run_watch_poll_once(&store.dir, &cfg, true).expect("poll pass");
    assert_eq!(stats.updated_projects, 2);
    assert!(index_run_verdict(&stats).is_ok());

    // A second process takes the writer lock right away: the pass did not keep it for the
    // sleep that follows in the loop.
    let exe = env::current_exe().expect("test exe");
    let mut child = Command::new(&exe)
        .args([
            "db::writer_lock_tests::hold_writer_lock_helper",
            "--exact",
            "--nocapture",
        ])
        .env("RETRIVIO_TEST_HOLD_LOCK", &store.dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn helper");
    let deadline = Instant::now() + Duration::from_secs(30);
    while !store.dir.join("held").exists() {
        assert!(
            Instant::now() < deadline,
            "the other process never got the lock: the polling pass must have kept it"
        );
        thread::sleep(Duration::from_millis(20));
    }
    let err = WriterLock::try_acquire(&store.dir).expect_err("held by the other process");
    assert!(is_index_busy_error(&err), "{}", err);
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGKILL);
    }
    let _ = child.wait();
    // And once it is gone the next pass runs again (nothing to do).
    let again = run_watch_poll_once(&store.dir, &cfg, true).expect("second pass");
    assert_eq!(again.skipped_projects, 2);
}

#[test]
fn a_failure_between_embedding_and_publish_leaves_the_project_untouched() {
    let store = TestStore::new("publish-atomic");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    let first = store.index(&cfg, &embedder, false).expect("first run");
    assert_eq!(first.chunks_embedded, 3);
    let conn = store.conn();
    let alpha = root.join("alpha");
    let before = project_snapshot(&conn, &alpha);
    assert!(before.iter().filter(|l| l.starts_with("chunk|")).count() == 2);

    // Edit one file, add another, delete nothing; the run embeds both changed chunks and
    // then fails right before the publish transaction.
    write(
        &alpha.join("b.md"),
        "alpha two rewritten about ui and frontend",
    );
    write(&alpha.join("c.md"), "alpha three brand new file");
    let embedded_before = embedder.embedded_texts().len();
    INJECT_FAIL_BEFORE_PUBLISH.store(true, Ordering::SeqCst);
    let failed = store.index(&cfg, &embedder, false).expect("run completes");
    assert!(
        !INJECT_FAIL_BEFORE_PUBLISH.load(Ordering::SeqCst),
        "hook consumed"
    );
    assert_eq!(failed.projects_failed, 1);
    assert_eq!(failed.updated_projects, 0);
    assert_eq!(failed.skipped_projects, 1, "beta");
    assert!(
        failed.stopped.is_empty(),
        "a project failure does not stop the run"
    );
    assert!(
        failed.failures[0].contains("injected failure between embedding and publish"),
        "{:?}",
        failed.failures
    );
    assert_eq!(
        embedder.embedded_texts().len(),
        embedded_before + 3,
        "the failure came after the embedding: two chunks and the changed summary"
    );
    assert!(index_run_verdict(&failed).is_err());

    // Nothing of the project changed: not a chunk, vector, manifest row, summary vector,
    // edge, signature or summary. Only the in-progress marker says a run was here.
    let after = project_snapshot(&conn, &alpha);
    assert_eq!(after, before);
    assert!(project_row(&conn, &alpha).index_in_progress);
    assert_eq!(chunk_rows_for(&conn, "c.md"), (0, 0));
    assert_vectors_match_their_chunk_text(&conn, 3);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM lance_pending"), 0);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 3);

    // The next run publishes it: two chunks embedded again (they were never stored), one
    // reused, the marker cleared, the signature advanced.
    let next = store.index(&cfg, &embedder, false).expect("retry run");
    assert_eq!(next.projects_failed, 0);
    assert_eq!(next.updated_projects, 1);
    assert_eq!(next.chunks_embedded, 2);
    assert_eq!(next.chunks_reused, 1);
    let row = project_row(&conn, &alpha);
    assert!(!row.index_in_progress);
    assert!(row.scan_signature.starts_with("3:"));
    assert_eq!(chunk_rows_for(&conn, "c.md"), (1, 1));
    assert_vectors_match_their_chunk_text(&conn, 4);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 4);
    assert!(index_run_verdict(&next).is_ok());
}

#[test]
fn a_collector_panic_fails_only_that_project_and_freezes_the_fingerprints() {
    let store = TestStore::new("collector-panic");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    store.index(&cfg, &embedder, false).expect("first run");
    let conn = store.conn();
    let alpha = root.join("alpha");
    let before = project_snapshot(&conn, &alpha);
    assert!(app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT)
        .unwrap()
        .is_some());

    // Force a revisit of every project (fingerprint absent) and make alpha's collector
    // panic; beta gets a real edit so it has work to do.
    conn.execute(
        "DELETE FROM app_state WHERE key = ?1",
        params![APP_STATE_EMBED_FINGERPRINT],
    )
    .unwrap();
    write(
        &alpha.join("a.md"),
        "alpha one edited while the collector is broken",
    );
    write(&root.join("beta").join("notes.md"), "beta notes, edited");
    *INJECT_COLLECTOR_PANIC.lock().unwrap() = Some("alpha".to_string());
    let panicked = store.index(&cfg, &embedder, false).expect("run completes");
    *INJECT_COLLECTOR_PANIC.lock().unwrap() = None;
    assert_eq!(panicked.projects_failed, 1);
    assert!(
        panicked.failures[0].contains("collecting the project panicked")
            && panicked.failures[0].contains("injected collector panic"),
        "{:?}",
        panicked.failures
    );
    assert_eq!(panicked.updated_projects, 1, "beta went through");
    assert_eq!(panicked.chunks_embedded, 1, "beta's edited chunk");
    assert!(panicked.stopped.is_empty());
    assert!(index_run_verdict(&panicked).is_err());
    // Alpha: untouched, not even marked in progress (its row was never begun), and its
    // edges to beta survive because the failed project stays in the graph.
    assert_eq!(project_snapshot(&conn, &alpha), before);
    assert!(!project_row(&conn, &alpha).index_in_progress);
    assert_eq!(
        app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT).unwrap(),
        None,
        "a run with a failed project never advances the fingerprint"
    );

    // Healed: alpha is revisited (the fingerprint is still absent), its edit lands, the
    // fingerprint is written.
    let healed = store.index(&cfg, &embedder, false).expect("healed run");
    assert_eq!(healed.projects_failed, 0);
    assert!(healed.updated_projects >= 1);
    assert!(chunks_containing(&conn, "edited while the collector is broken") >= 1);
    assert!(app_state_get(&conn, APP_STATE_EMBED_FINGERPRINT)
        .unwrap()
        .is_some());
    assert!(index_run_verdict(&healed).is_ok());
}

#[test]
fn lance_repair_fails_closed_on_a_malformed_sqlite_vector() {
    let store = TestStore::new("lance-fail-closed");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    store.index(&cfg, &embedder, false).expect("first run");
    let conn = store.conn();
    let ids: Vec<i64> = {
        let mut stmt = conn
            .prepare("SELECT chunk_id FROM project_chunk_vectors ORDER BY chunk_id")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    let good_blob: Vec<u8> = conn
        .query_row(
            "SELECT vector FROM project_chunk_vectors WHERE chunk_id = ?1",
            params![ids[1]],
            |r| r.get(0),
        )
        .unwrap();

    // One sqlite vector is malformed (two bytes, not 64 floats), its LanceDB row is gone,
    // its id is pending and the marker is set: exactly the state a repair must refuse.
    conn.execute(
        "UPDATE project_chunk_vectors SET vector = x'0000' WHERE chunk_id = ?1",
        params![ids[0]],
    )
    .unwrap();
    with_lance_store(|s| lance_store::delete_chunks(s, &ids[..1])).expect("delete");
    lance_pending_add(&conn, &ids[..1]).unwrap();
    app_state_set(&conn, APP_STATE_LANCE_DIRTY, "1").unwrap();

    let refused = store.index(&cfg, &embedder, false).expect("run completes");
    assert!(
        refused.lance_error.contains("repair refused")
            && refused.lance_error.contains(&format!("chunk {}", ids[0])),
        "{}",
        refused.lance_error
    );
    assert_eq!(refused.lance_repaired, 0);
    assert_eq!(refused.skipped_projects, 2, "sqlite content is untouched");
    assert!(index_run_verdict(&refused).is_err());
    assert_eq!(
        app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
        Some("1".to_string()),
        "marker kept"
    );
    assert_eq!(
        count(&conn, "SELECT COUNT(*) FROM lance_pending"),
        1,
        "pending id kept"
    );
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 2);

    // With the blob restored the next run repairs and clears both.
    conn.execute(
        "UPDATE project_chunk_vectors SET vector = ?1 WHERE chunk_id = ?2",
        params![good_blob, ids[0]],
    )
    .unwrap();
    let repaired = store.index(&cfg, &embedder, false).expect("repair run");
    assert!(repaired.lance_error.is_empty(), "{}", repaired.lance_error);
    assert_eq!(repaired.lance_repaired, 1);
    assert_eq!(
        app_state_get(&conn, APP_STATE_LANCE_DIRTY).unwrap(),
        Some("0".to_string())
    );
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM lance_pending"), 0);
    assert_eq!(with_lance_store(|s| lance_store::count(s)).unwrap(), 3);
    assert!(index_run_verdict(&repaired).is_ok());
}

#[test]
fn a_touched_but_identical_file_gets_its_new_stat_written_without_rechunking() {
    let store = TestStore::new("restat");
    let root = two_project_root(&store, "root");
    store.track(&root);
    let cfg = store.cfg(&root, &[]);
    let embedder = TestEmbedder::new(&cfg, false);
    store.index(&cfg, &embedder, false).expect("first run");
    let conn = store.conn();
    let stored_mtime = |conn: &Connection| -> f64 {
        conn.query_row(
            "SELECT file_mtime FROM project_files WHERE rel_path = 'a.md'",
            [],
            |r| r.get(0),
        )
        .unwrap()
    };
    let before = stored_mtime(&conn);

    // Touch: the signature changes (mtime), the content does not.
    let later = SystemTime::now() + Duration::from_secs(120);
    set_mtime(&root.join("alpha").join("a.md"), later);
    let second = store.index(&cfg, &embedder, false).expect("second run");
    assert_eq!(second.updated_projects, 1);
    assert_eq!(second.files_rechunked, 0);
    assert_eq!(second.files_unchanged, 2);
    assert_eq!(second.chunks_embedded, 0);
    assert_eq!(second.chunks_reused, 0, "nothing entered the work set");
    let after = stored_mtime(&conn);
    assert!(
        after > before,
        "manifest carries the new stat ({} -> {})",
        before,
        after
    );
    let expected = later.duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    assert!((after - expected).abs() < 0.001);

    // The next run finds the manifest stat equal again: no read, no hash, skipped.
    let third = store.index(&cfg, &embedder, false).expect("third run");
    assert_eq!(third.skipped_projects, 2);
}
