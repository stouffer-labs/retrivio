//! Shared test fixtures: the per-test data directory, `TestStore`, `TestEmbedder` and small
//! sqlite helpers used by the index, prune, rank and recall test modules.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::time::SystemTime;
use std::{fs, process};

use rusqlite::Connection;

use crate::config::{db_path, ConfigValues};
use crate::db::{ensure_tracked_root_conn, open_db_rw, WriterLock, LANCE_STORE};
use crate::embed::{model_key_for_cfg, Embedder, LocalHashEmbedder};
use crate::index::{
    get_project_by_path, run_native_index_with_embedder, ExistingProject, IndexRunOptions,
    IndexStats, LANCE_WRITE_FAILED,
};
use crate::scan::IndexScope;
use crate::util::{normalize_path, now_ts};

thread_local! {
    static DATA_DIR: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}
/// Process-wide data dir for a test binary that re-runs itself as a child command (the
/// recall end-to-end tests): the command spawns worker threads, which do not inherit the
/// thread-local above.
static PROCESS_DATA_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
/// The LanceDB handle is process-global and not tied to a store path, so every test that can
/// open it, act on it or read the dirty marker takes turns: [`TestStore`] and
/// [`lance_isolation`] both hold this lock for the test's lifetime.
static STORE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn data_dir_for_test() -> PathBuf {
    if let Some(d) = DATA_DIR.with(|d| d.borrow().clone()) {
        return d;
    }
    if let Some(d) = PROCESS_DATA_DIR
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
    {
        return d;
    }
    panic!("data_dir() used in a test without a TestStore; tests must never touch ~/.retrivio")
}

/// Install `dir` as this process's data dir for every thread (child-process helpers only).
pub(crate) fn install_process_data_dir(dir: &Path) {
    *PROCESS_DATA_DIR.lock().unwrap_or_else(|p| p.into_inner()) = Some(dir.to_path_buf());
}

fn reset_lance_store() {
    if let Some(lock) = LANCE_STORE.get() {
        *lock.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// Isolation for a test that needs no data dir but runs a path consulting the process-global
/// LanceDB handle: `remove_projects_not_in`, `sync_lance_after_publish` and
/// `repair_lance_from_sqlite` act on whatever handle is open (`lance_store_is_open`), and
/// `lance_delete_marked` clears the dirty marker after a successful delete. Without the lock a
/// concurrent indexer test's open handle makes the test delete that test's vectors and read a
/// cleared marker. Takes the same lock as [`TestStore`] and leaves the handle closed; hold the
/// returned guard for the whole test (`let _lance = lance_isolation();`).
pub(crate) fn lance_isolation() -> MutexGuard<'static, ()> {
    let guard = STORE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    reset_lance_store();
    LANCE_WRITE_FAILED.store(false, Ordering::SeqCst);
    guard
}

/// An isolated data directory under `repo/tmp` for one test, installed as this thread's
/// data dir for the store's lifetime.
pub(crate) struct TestStore {
    pub(crate) dir: PathBuf,
    _guard: MutexGuard<'static, ()>,
}

impl TestStore {
    pub(crate) fn new(name: &str) -> Self {
        let guard = STORE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("it-{}-{}", name, process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test data dir");
        DATA_DIR.with(|d| *d.borrow_mut() = Some(dir.clone()));
        reset_lance_store();
        LANCE_WRITE_FAILED.store(false, Ordering::SeqCst);
        TestStore { dir, _guard: guard }
    }

    /// A corpus root directory inside the store dir (outside the data files).
    pub(crate) fn corpus_root(&self, name: &str) -> PathBuf {
        let root = self.dir.join("corpus").join(name);
        fs::create_dir_all(&root).expect("corpus root");
        root
    }

    /// Config for a local, deterministic embedder; `extra` adds or overrides keys.
    pub(crate) fn cfg(&self, root: &Path, extra: &[(&str, &str)]) -> ConfigValues {
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("root".into(), root.to_string_lossy().to_string());
        map.insert("embed_backend".into(), "ollama".into());
        map.insert("embed_model".into(), "test-local".into());
        map.insert("retrieval_backend".into(), "lancedb".into());
        map.insert("local_embed_dim".into(), "64".into());
        for (k, v) in extra {
            map.insert((*k).to_string(), (*v).to_string());
        }
        ConfigValues::from_map(map)
    }

    pub(crate) fn conn(&self) -> Connection {
        open_db_rw(&db_path(&self.dir)).expect("open store")
    }

    pub(crate) fn track(&self, root: &Path) {
        let conn = self.conn();
        ensure_tracked_root_conn(&conn, &normalize_path(&root.to_string_lossy()), now_ts())
            .expect("track root");
    }

    /// One all-roots run with `remove_missing`, the way `retrivio index` / `refresh` run.
    pub(crate) fn index(
        &self,
        cfg: &ConfigValues,
        embedder: &dyn Embedder,
        force_all: bool,
    ) -> Result<IndexStats, String> {
        let writer = WriterLock::try_acquire(&self.dir).expect("writer lock");
        run_native_index_with_embedder(
            &self.dir,
            cfg,
            &writer,
            embedder,
            IndexRunOptions {
                scope: IndexScope::AllRoots,
                force_all,
                force_paths: HashSet::new(),
                remove_missing: true,
                reason: if force_all { "refresh" } else { "index" },
            },
            HashSet::new(),
            false,
        )
    }
}

impl Drop for TestStore {
    fn drop(&mut self) {
        reset_lance_store();
        DATA_DIR.with(|d| *d.borrow_mut() = None);
    }
}

/// Deterministic local embedder whose model key matches the test config, with a switchable
/// normalisation flag and an injectable failure (every call from `fail_from_now` on errors).
pub(crate) struct TestEmbedder {
    inner: LocalHashEmbedder,
    key: String,
    normalized: bool,
    calls: AtomicUsize,
    fail_from_call: AtomicUsize,
    texts: Mutex<Vec<String>>,
}

impl TestEmbedder {
    pub(crate) fn new(cfg: &ConfigValues, normalized: bool) -> Self {
        TestEmbedder {
            inner: LocalHashEmbedder::new(cfg.local_embed_dim as usize),
            key: model_key_for_cfg(cfg),
            normalized,
            calls: AtomicUsize::new(0),
            fail_from_call: AtomicUsize::new(usize::MAX),
            texts: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn fail_from_now(&self) {
        self.fail_from_call
            .store(self.calls.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    pub(crate) fn heal(&self) {
        self.fail_from_call.store(usize::MAX, Ordering::SeqCst);
    }

    /// Every text embedded so far, in order.
    pub(crate) fn embedded_texts(&self) -> Vec<String> {
        self.texts.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn account(&self, texts: &[String]) -> Result<(), String> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n >= self.fail_from_call.load(Ordering::SeqCst) {
            return Err("injected embedder failure".to_string());
        }
        self.texts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend(texts.iter().cloned());
        Ok(())
    }
}

impl Embedder for TestEmbedder {
    fn model_key(&self) -> String {
        self.key.clone()
    }

    fn normalizes_output(&self) -> bool {
        self.normalized
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.account(texts)?;
        Ok(texts
            .iter()
            .map(|t| self.inner.embed_one_local(t))
            .collect())
    }
}

pub(crate) fn set_mtime(path: &Path, when: SystemTime) {
    let f = OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open for mtime");
    f.set_modified(when).expect("set mtime");
}

pub(crate) fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).expect(sql)
}

pub(crate) fn project_row(conn: &Connection, path: &Path) -> ExistingProject {
    get_project_by_path(
        conn,
        &normalize_path(&path.to_string_lossy()).to_string_lossy(),
    )
    .expect("project query")
    .expect("project row")
}
