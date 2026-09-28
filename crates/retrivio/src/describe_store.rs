//! SQLite side of the description layer (picker spec, Phase 1a): the `file_meta` and
//! `project_meta` tables, their upserts and the batched lookups the picker runs per keystroke.
//! Readers check [`describe_tables_present`] first: a store last written by an older binary
//! has neither table until the next writer migrates it, and the picker must keep working.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

use rusqlite::types::Value;
use rusqlite::{params, params_from_iter, Connection};

use crate::config::ScanSettings;
use crate::db::db_has_table;
use crate::describe::{
    describe_file, head_text, readme_synopsis, FileMeta, EXTRACT_VERSION, HEAD_CHARS,
};
use crate::documents::{self, ExtractLimits};
use crate::scan::suffix_with_dot;
use crate::util::now_ts;

/// Create the description tables. Called from `init_schema_in_tx` (writers only).
pub(crate) fn ensure_describe_tables(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS file_meta (
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    title_source TEXT NOT NULL DEFAULT '',
    doc_kind TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    extract_version INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (project_id, rel_path)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS project_meta (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    synopsis TEXT NOT NULL DEFAULT '',
    synopsis_source TEXT NOT NULL DEFAULT '',
    synopsis_signature TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);
"#,
    )
    .map_err(|e| format!("failed creating description tables: {}", e))
}

/// True when both description tables exist. Read paths call this once per run and skip every
/// description lookup when it is false.
pub(crate) fn describe_tables_present(conn: &Connection) -> bool {
    matches!(db_has_table(conn, "file_meta"), Ok(true))
        && matches!(db_has_table(conn, "project_meta"), Ok(true))
}

pub(crate) fn upsert_file_meta(
    conn: &Connection,
    project_id: i64,
    rel_path: &str,
    content_hash: &str,
    meta: &FileMeta,
) -> Result<(), String> {
    upsert_file_meta_with_version(
        conn,
        project_id,
        rel_path,
        content_hash,
        meta,
        EXTRACT_VERSION,
    )
}

/// [`upsert_file_meta`] with an explicit `extract_version`. A row written after a read failure
/// carries 0, so the pending predicate (`extract_version < EXTRACT_VERSION`) picks it up
/// again next run.
pub(crate) fn upsert_file_meta_with_version(
    conn: &Connection,
    project_id: i64,
    rel_path: &str,
    content_hash: &str,
    meta: &FileMeta,
    extract_version: i64,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO file_meta(project_id, rel_path, title, title_source, doc_kind, content_hash, extract_version, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT(project_id, rel_path) DO UPDATE SET
    title = excluded.title,
    title_source = excluded.title_source,
    doc_kind = excluded.doc_kind,
    content_hash = excluded.content_hash,
    extract_version = excluded.extract_version,
    updated_at = excluded.updated_at
"#,
        params![
            project_id,
            rel_path,
            meta.title,
            meta.title_source,
            meta.doc_kind,
            content_hash,
            extract_version,
            now_ts()
        ],
    )
    .map_err(|e| format!("failed upserting file_meta: {}", e))?;
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StoredFileMeta {
    pub(crate) title: String,
    pub(crate) title_source: String,
    pub(crate) doc_kind: String,
}

fn placeholders(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(", ")
}

fn project_ids_for_paths(
    conn: &Connection,
    paths: &[&str],
) -> Result<HashMap<String, i64>, String> {
    let mut out = HashMap::new();
    if paths.is_empty() {
        return Ok(out);
    }
    let sql = format!(
        "SELECT id, path FROM projects WHERE path IN ({})",
        placeholders(paths.len())
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("failed preparing project id lookup: {}", e))?;
    let rows = stmt
        .query_map(params_from_iter(paths.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| format!("failed querying project ids: {}", e))?;
    for row in rows {
        let (id, path) = row.map_err(|e| format!("failed reading project id row: {}", e))?;
        out.insert(path, id);
    }
    Ok(out)
}

/// Titles and kinds for `(project_path, rel_path)` pairs, keyed by `<project_path>/<rel_path>`
/// (a chunk's `doc_path`). Two queries whatever the count.
pub(crate) fn load_file_meta(
    conn: &Connection,
    keys: &[(String, String)],
) -> Result<HashMap<String, StoredFileMeta>, String> {
    let mut out = HashMap::new();
    if keys.is_empty() {
        return Ok(out);
    }
    let mut project_paths: Vec<&str> = keys.iter().map(|(p, _)| p.as_str()).collect();
    project_paths.sort_unstable();
    project_paths.dedup();
    let ids = project_ids_for_paths(conn, &project_paths)?;
    if ids.is_empty() {
        return Ok(out);
    }
    let mut rel_paths: Vec<&str> = keys.iter().map(|(_, r)| r.as_str()).collect();
    rel_paths.sort_unstable();
    rel_paths.dedup();
    let id_list: Vec<i64> = ids.values().copied().collect();
    let path_by_id: HashMap<i64, &str> = ids.iter().map(|(p, id)| (*id, p.as_str())).collect();
    let sql = format!(
        "SELECT project_id, rel_path, title, title_source, doc_kind FROM file_meta WHERE project_id IN ({}) AND rel_path IN ({})",
        placeholders(id_list.len()),
        placeholders(rel_paths.len())
    );
    let mut bind: Vec<Value> = id_list.iter().map(|id| Value::Integer(*id)).collect();
    bind.extend(rel_paths.iter().map(|r| Value::Text(r.to_string())));
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("failed preparing file_meta lookup: {}", e))?;
    let rows = stmt
        .query_map(params_from_iter(bind.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })
        .map_err(|e| format!("failed querying file_meta: {}", e))?;
    for row in rows {
        let (pid, rel, title, title_source, doc_kind) =
            row.map_err(|e| format!("failed reading file_meta row: {}", e))?;
        if let Some(pp) = path_by_id.get(&pid) {
            out.insert(
                format!("{}/{}", pp, rel),
                StoredFileMeta {
                    title,
                    title_source,
                    doc_kind,
                },
            );
        }
    }
    Ok(out)
}

pub(crate) fn upsert_project_meta(
    conn: &Connection,
    project_id: i64,
    synopsis: &str,
    source: &str,
    signature: &str,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO project_meta(project_id, synopsis, synopsis_source, synopsis_signature, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(project_id) DO UPDATE SET
    synopsis = excluded.synopsis,
    synopsis_source = excluded.synopsis_source,
    synopsis_signature = excluded.synopsis_signature,
    updated_at = excluded.updated_at
"#,
        params![project_id, synopsis, source, signature, now_ts()],
    )
    .map_err(|e| format!("failed upserting project_meta: {}", e))?;
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StoredProjectMeta {
    pub(crate) synopsis: String,
    pub(crate) synopsis_source: String,
    pub(crate) project_mtime: f64,
    pub(crate) file_count: i64,
}

/// Synopsis (when `with_synopsis`), mtime and file count per project path, in one query.
/// With `with_synopsis = false` the query never names `project_meta`, so it works on a store
/// that has no such table.
pub(crate) fn load_project_meta(
    conn: &Connection,
    project_paths: &[String],
    with_synopsis: bool,
) -> Result<HashMap<String, StoredProjectMeta>, String> {
    let mut out = HashMap::new();
    if project_paths.is_empty() {
        return Ok(out);
    }
    let sql = if with_synopsis {
        format!(
            "SELECT p.path, COALESCE(pm.synopsis, ''), COALESCE(pm.synopsis_source, ''), p.project_mtime, (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id = p.id) FROM projects p LEFT JOIN project_meta pm ON pm.project_id = p.id WHERE p.path IN ({})",
            placeholders(project_paths.len())
        )
    } else {
        format!(
            "SELECT p.path, '', '', p.project_mtime, (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id = p.id) FROM projects p WHERE p.path IN ({})",
            placeholders(project_paths.len())
        )
    };
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("failed preparing project_meta lookup: {}", e))?;
    let rows = stmt
        .query_map(params_from_iter(project_paths.iter()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })
        .map_err(|e| format!("failed querying project_meta: {}", e))?;
    for row in rows {
        let (path, synopsis, synopsis_source, project_mtime, file_count) =
            row.map_err(|e| format!("failed reading project_meta row: {}", e))?;
        out.insert(
            path,
            StoredProjectMeta {
                synopsis,
                synopsis_source,
                project_mtime,
                file_count,
            },
        );
    }
    Ok(out)
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct DescribeReport {
    /// Rows written this run, the failed ones included.
    pub(crate) files_described: usize,
    /// Rows written with the file-name title because the file could not be read; stored at
    /// extract version 0 so they are retried next run.
    pub(crate) files_failed: usize,
    pub(crate) projects_described: usize,
    /// `file_meta` rows deleted because their manifest row is gone.
    pub(crate) rows_pruned: usize,
}

/// Largest document read for its title; bigger Office/HTML files keep the file-name title.
const MAX_DESCRIBE_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;
const BATCH: usize = 500;

/// Title and kind for one file from its bytes: the head of a text file; the extracted text and
/// metadata title of an Office/HTML document (a bounded extraction); PDFs are never opened
/// (their extraction yields no title and spawns a child process), they keep the file name.
fn describe_one(
    abs_path: &Path,
    rel_path: &str,
    settings: &ScanSettings,
    limits: &ExtractLimits,
) -> Result<FileMeta, String> {
    let suffix = suffix_with_dot(abs_path);
    if suffix == ".pdf" {
        return Ok(describe_file(rel_path, "", None));
    }
    if settings.is_document_suffix(&suffix) {
        // Fast path: a file the metadata already shows over the bound is never opened. The
        // bounded read below is the check that holds when the size changes underneath.
        if fs::metadata(abs_path).is_ok_and(|m| m.len() > MAX_DESCRIBE_DOCUMENT_BYTES) {
            return Ok(describe_file(rel_path, "", None));
        }
        let raw = read_at_most(abs_path, MAX_DESCRIBE_DOCUMENT_BYTES + 1)?;
        if raw.len() as u64 > MAX_DESCRIBE_DOCUMENT_BYTES {
            return Ok(describe_file(rel_path, "", None));
        }
        let small = ExtractLimits {
            text_bytes: HEAD_CHARS * 4,
            ..limits.clone()
        };
        return match documents::extract_from_bytes(abs_path, &raw, &small) {
            Ok(Some(doc)) => Ok(describe_file(
                rel_path,
                &head_text(doc.text.as_bytes()),
                doc.title.as_deref(),
            )),
            Ok(None) => Ok(describe_file(rel_path, "", None)),
            Err(e) => Err(e),
        };
    }
    let buf = read_at_most(abs_path, (HEAD_CHARS * 4) as u64)?;
    Ok(describe_file(rel_path, &head_text(&buf), None))
}

/// The first `max` bytes of a file (fewer when it is shorter), however many reads that takes.
fn read_at_most(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|e| format!("cannot open: {}", e))?;
    let mut buf = Vec::new();
    file.take(max)
        .read_to_end(&mut buf)
        .map_err(|e| format!("cannot read: {}", e))?;
    Ok(buf)
}

const README_NAMES: &[&str] = &[
    "README.md",
    "readme.md",
    "README",
    "README.txt",
    "README.rst",
    "Readme.md",
];

fn project_readme_synopsis(project_path: &str) -> Option<String> {
    for name in README_NAMES {
        let candidate = Path::new(project_path).join(name);
        let Ok(buf) = read_at_most(&candidate, 8192) else {
            continue;
        };
        let text = String::from_utf8_lossy(&buf);
        return readme_synopsis(&text);
    }
    None
}

/// Describe every manifest file whose `file_meta` row is missing, older than
/// [`EXTRACT_VERSION`] or for another content hash (at most `max_files` this run), then every
/// project whose synopsis signature is not its scan signature, then drop the `file_meta` rows
/// whose manifest row is gone. Metadata only: reads the head of each file, touches no chunk,
/// vector or embedding. Idempotent and resumable: each batch of 500 is read in full before
/// its one short write transaction, so no file IO happens while the store is locked; a run
/// that stops leaves what it wrote, the next continues. A file that cannot be read gets its
/// file-name title at extract version 0, is counted in `files_failed`, and is retried next
/// run.
pub(crate) fn describe_pending(
    conn: &Connection,
    settings: &ScanSettings,
    limits: &ExtractLimits,
    max_files: usize,
) -> Result<DescribeReport, String> {
    let mut report = DescribeReport::default();
    let pending: Vec<(i64, String, String, String)> = {
        let mut stmt = conn
            .prepare(
                r#"
SELECT pf.project_id, pf.rel_path, pf.abs_path, pf.content_hash
FROM project_files pf
LEFT JOIN file_meta fm ON fm.project_id = pf.project_id AND fm.rel_path = pf.rel_path
WHERE fm.rel_path IS NULL OR fm.extract_version < ?1 OR fm.content_hash <> pf.content_hash
ORDER BY pf.project_id, pf.rel_path
LIMIT ?2
"#,
            )
            .map_err(|e| format!("failed preparing describe query: {}", e))?;
        let rows = stmt
            .query_map(params![EXTRACT_VERSION, max_files as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| format!("failed querying files to describe: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| format!("failed reading describe row: {}", e))?);
        }
        out
    };
    for batch in pending.chunks(BATCH) {
        // All the reading first, outside any transaction.
        let described: Vec<(i64, &str, &str, FileMeta, bool)> = batch
            .iter()
            .map(|(project_id, rel_path, abs_path, content_hash)| {
                match describe_one(Path::new(abs_path), rel_path, settings, limits) {
                    Ok(meta) => (
                        *project_id,
                        rel_path.as_str(),
                        content_hash.as_str(),
                        meta,
                        false,
                    ),
                    Err(_) => (
                        *project_id,
                        rel_path.as_str(),
                        content_hash.as_str(),
                        describe_file(rel_path, "", None),
                        true,
                    ),
                }
            })
            .collect();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("failed starting describe transaction: {}", e))?;
        for (project_id, rel_path, content_hash, meta, failed) in &described {
            if *failed {
                upsert_file_meta_with_version(&tx, *project_id, rel_path, content_hash, meta, 0)?;
                report.files_failed += 1;
            } else {
                upsert_file_meta(&tx, *project_id, rel_path, content_hash, meta)?;
            }
            report.files_described += 1;
        }
        tx.commit()
            .map_err(|e| format!("failed committing describe batch: {}", e))?;
    }

    let projects: Vec<(i64, String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT p.id, p.path, p.scan_signature FROM projects p LEFT JOIN project_meta pm ON pm.project_id = p.id WHERE pm.project_id IS NULL OR pm.synopsis_signature <> p.scan_signature",
            )
            .map_err(|e| format!("failed preparing project describe query: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| format!("failed querying projects to describe: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| format!("failed reading project describe row: {}", e))?);
        }
        out
    };
    if !projects.is_empty() {
        // Every README is read before the one write transaction.
        let synopses: Vec<(i64, String, &str, &str)> = projects
            .iter()
            .map(
                |(project_id, path, signature)| match project_readme_synopsis(path) {
                    Some(s) => (*project_id, s, "readme", signature.as_str()),
                    None => (*project_id, String::new(), "none", signature.as_str()),
                },
            )
            .collect();
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("failed starting project describe transaction: {}", e))?;
        for (project_id, synopsis, source, signature) in &synopses {
            upsert_project_meta(&tx, *project_id, synopsis, source, signature)?;
            report.projects_described += 1;
        }
        tx.commit()
            .map_err(|e| format!("failed committing project describe batch: {}", e))?;
    }

    // Rows for files the manifest no longer lists (a removed file leaves its `file_meta` row
    // behind; there is no composite foreign key to cascade it).
    report.rows_pruned = conn
        .execute(
            "DELETE FROM file_meta WHERE NOT EXISTS (SELECT 1 FROM project_files pf WHERE pf.project_id = file_meta.project_id AND pf.rel_path = file_meta.rel_path)",
            [],
        )
        .map_err(|e| format!("failed pruning orphaned file_meta rows: {}", e))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::init_schema;
    use crate::describe::FileMeta;
    use rusqlite::{params, Connection};

    fn store_with_project() -> (Connection, i64) {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute(
            "INSERT INTO projects(path, title, summary, project_mtime, last_indexed, scan_signature) VALUES ('/p/alpha', 'alpha', '', 1000.0, 1000.0, 'sig1')",
            [],
        )
        .unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM projects", [], |r| r.get(0))
            .unwrap();
        conn.execute(
            "INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed) VALUES (?1, 'docs/a.md', '/p/alpha/docs/a.md', 10, 1000.0, 'h1', 1, 1000.0)",
            params![id],
        )
        .unwrap();
        (conn, id)
    }

    #[test]
    fn tables_are_created_by_init_schema_and_detected() {
        let (conn, _) = store_with_project();
        assert!(describe_tables_present(&conn));
        conn.execute_batch("DROP TABLE file_meta; DROP TABLE project_meta;")
            .unwrap();
        assert!(!describe_tables_present(&conn));
    }

    #[test]
    fn old_store_without_description_tables_still_serves_lookups() {
        let (conn, _) = store_with_project();
        conn.execute_batch("DROP TABLE file_meta; DROP TABLE project_meta;")
            .unwrap();
        assert!(!describe_tables_present(&conn));
        // The synopsis join fails without the table; that is why readers check the gate first.
        assert!(load_project_meta(&conn, &["/p/alpha".to_string()], true).is_err());
        // What jump-feed does when the tables are absent: no file_meta query at all, and the
        // project lookup without the synopsis join.
        let pm = load_project_meta(&conn, &["/p/alpha".to_string()], false).unwrap();
        assert_eq!(pm["/p/alpha"].file_count, 1);
        assert_eq!(pm["/p/alpha"].synopsis, "");
        // The next writer migration recreates them.
        init_schema(&conn).unwrap();
        assert!(describe_tables_present(&conn));
    }

    #[test]
    fn file_meta_round_trips_by_doc_path() {
        let (conn, id) = store_with_project();
        let meta = FileMeta {
            title: "Alpha notes".to_string(),
            title_source: "h1",
            doc_kind: "md",
        };
        upsert_file_meta(&conn, id, "docs/a.md", "h1", &meta).unwrap();
        let loaded = load_file_meta(
            &conn,
            &[
                ("/p/alpha".to_string(), "docs/a.md".to_string()),
                ("/p/alpha".to_string(), "missing.md".to_string()),
            ],
        )
        .unwrap();
        assert_eq!(loaded.len(), 1);
        let got = &loaded["/p/alpha/docs/a.md"];
        assert_eq!(got.title, "Alpha notes");
        assert_eq!(got.title_source, "h1");
        assert_eq!(got.doc_kind, "md");
        // Upsert replaces.
        let meta2 = FileMeta {
            title: "Alpha notes v2".to_string(),
            title_source: "frontmatter",
            doc_kind: "md",
        };
        upsert_file_meta(&conn, id, "docs/a.md", "h2", &meta2).unwrap();
        let loaded =
            load_file_meta(&conn, &[("/p/alpha".to_string(), "docs/a.md".to_string())]).unwrap();
        assert_eq!(loaded["/p/alpha/docs/a.md"].title, "Alpha notes v2");
        assert!(load_file_meta(&conn, &[]).unwrap().is_empty());
    }

    #[test]
    fn project_meta_round_trips_and_works_without_the_table() {
        let (conn, id) = store_with_project();
        upsert_project_meta(&conn, id, "Alpha — the first project", "readme", "sig1").unwrap();
        let loaded = load_project_meta(&conn, &["/p/alpha".to_string()], true).unwrap();
        let got = &loaded["/p/alpha"];
        assert_eq!(got.synopsis, "Alpha — the first project");
        assert_eq!(got.synopsis_source, "readme");
        assert_eq!(got.file_count, 1);
        assert!((got.project_mtime - 1000.0).abs() < f64::EPSILON);
        conn.execute_batch("DROP TABLE project_meta;").unwrap();
        let loaded = load_project_meta(&conn, &["/p/alpha".to_string()], false).unwrap();
        assert_eq!(loaded["/p/alpha"].synopsis, "");
        assert_eq!(loaded["/p/alpha"].file_count, 1);
    }

    #[test]
    fn describe_pending_fills_missing_rows_and_is_idempotent() {
        use crate::config::{ConfigValues, ScanSettings};
        use crate::documents::ExtractLimits;
        use std::fs;
        let dir = std::env::current_dir()
            .unwrap()
            .join("tmp")
            .join(format!("describe-{}", std::process::id()));
        fs::create_dir_all(dir.join("docs")).unwrap();
        fs::write(dir.join("docs/a.md"), "# Alpha plan\n\nbody\n").unwrap();
        fs::write(dir.join("README.md"), "# Alpha\n\nThe first project.\n").unwrap();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let root = dir.to_string_lossy().to_string();
        conn.execute(
            "INSERT INTO projects(path, title, summary, project_mtime, last_indexed, scan_signature) VALUES (?1, 'alpha', '', 1.0, 1.0, 'sig1')",
            params![root],
        ).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM projects", [], |r| r.get(0))
            .unwrap();
        conn.execute(
            "INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed) VALUES (?1, 'docs/a.md', ?2, 10, 1.0, 'h1', 1, 1.0)",
            params![id, dir.join("docs/a.md").to_string_lossy().to_string()],
        ).unwrap();
        let settings =
            ScanSettings::from_cfg(&ConfigValues::from_map(std::collections::HashMap::new()));
        let limits = ExtractLimits {
            text_bytes: 16_384,
            ..ExtractLimits::default()
        };

        let report = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(
            report,
            DescribeReport {
                files_described: 1,
                files_failed: 0,
                projects_described: 1,
                rows_pruned: 0
            }
        );
        let meta = load_file_meta(&conn, &[(root.clone(), "docs/a.md".to_string())]).unwrap();
        assert_eq!(meta[&format!("{}/docs/a.md", root)].title, "Alpha plan");
        let pm = load_project_meta(&conn, std::slice::from_ref(&root), true).unwrap();
        assert_eq!(pm[&root].synopsis, "Alpha — The first project.");
        assert_eq!(pm[&root].synopsis_source, "readme");

        // Nothing pending on the second run.
        let again = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(again, DescribeReport::default());

        // A changed content hash re-describes the file; a changed scan signature re-describes the project.
        conn.execute("UPDATE project_files SET content_hash = 'h2'", [])
            .unwrap();
        conn.execute("UPDATE projects SET scan_signature = 'sig2'", [])
            .unwrap();
        fs::write(dir.join("docs/a.md"), "# Alpha plan v2\n").unwrap();
        let third = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(third.files_described, 1);
        assert_eq!(third.projects_described, 1);
        let meta = load_file_meta(&conn, &[(root.clone(), "docs/a.md".to_string())]).unwrap();
        assert_eq!(meta[&format!("{}/docs/a.md", root)].title, "Alpha plan v2");
        fs::remove_dir_all(&dir).unwrap();
    }

    fn scan_settings_and_limits() -> (crate::config::ScanSettings, crate::documents::ExtractLimits)
    {
        use crate::config::{ConfigValues, ScanSettings};
        use crate::documents::ExtractLimits;
        (
            ScanSettings::from_cfg(&ConfigValues::from_map(std::collections::HashMap::new())),
            ExtractLimits {
                text_bytes: 16_384,
                ..ExtractLimits::default()
            },
        )
    }

    fn stored_version(conn: &Connection, rel_path: &str) -> i64 {
        conn.query_row(
            "SELECT extract_version FROM file_meta WHERE rel_path = ?1",
            params![rel_path],
            |r| r.get(0),
        )
        .unwrap()
    }

    #[test]
    fn a_missing_file_gets_a_file_name_title_and_is_retried_next_run() {
        use std::fs;
        let dir = std::env::current_dir()
            .unwrap()
            .join("tmp")
            .join(format!("describe-missing-{}", std::process::id()));
        fs::create_dir_all(dir.join("docs")).unwrap();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let root = dir.to_string_lossy().to_string();
        conn.execute(
            "INSERT INTO projects(path, title, summary, project_mtime, last_indexed, scan_signature) VALUES (?1, 'alpha', '', 1.0, 1.0, 'sig1')",
            params![root],
        ).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM projects", [], |r| r.get(0))
            .unwrap();
        let abs = dir.join("docs/gone-plan.md");
        conn.execute(
            "INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed) VALUES (?1, 'docs/gone-plan.md', ?2, 10, 1.0, 'h1', 1, 1.0)",
            params![id, abs.to_string_lossy().to_string()],
        ).unwrap();
        let (settings, limits) = scan_settings_and_limits();

        // The file is not on disk: a file-name title, counted as failed, stored at version 0.
        let report = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!((report.files_described, report.files_failed), (1, 1));
        let key = format!("{}/docs/gone-plan.md", root);
        let meta =
            load_file_meta(&conn, &[(root.clone(), "docs/gone-plan.md".to_string())]).unwrap();
        assert_eq!(meta[&key].title, "gone plan");
        assert_eq!(meta[&key].title_source, "filename");
        assert_eq!(stored_version(&conn, "docs/gone-plan.md"), 0);

        // Still missing: the row is pending again and re-described, not left as final.
        let again = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!((again.files_described, again.files_failed), (1, 1));
        assert_ne!(again, DescribeReport::default());

        // Once readable, the retry succeeds and the row becomes current.
        fs::write(&abs, "# The plan that was gone\n\nbody\n").unwrap();
        let third = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!((third.files_described, third.files_failed), (1, 0));
        let meta =
            load_file_meta(&conn, &[(root.clone(), "docs/gone-plan.md".to_string())]).unwrap();
        assert_eq!(meta[&key].title, "The plan that was gone");
        assert_eq!(stored_version(&conn, "docs/gone-plan.md"), EXTRACT_VERSION);
        let fourth = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(fourth.files_described, 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn orphaned_file_meta_rows_are_pruned() {
        let (conn, id) = store_with_project();
        let meta = FileMeta {
            title: "Kept".to_string(),
            title_source: "h1",
            doc_kind: "md",
        };
        // `docs/a.md` has a manifest row; `docs/removed.md` no longer does.
        upsert_file_meta(&conn, id, "docs/a.md", "h1", &meta).unwrap();
        upsert_file_meta(&conn, id, "docs/removed.md", "h9", &meta).unwrap();
        let (settings, limits) = scan_settings_and_limits();
        let report = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(report.rows_pruned, 1);
        let rows: Vec<String> = conn
            .prepare("SELECT rel_path FROM file_meta ORDER BY rel_path")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(rows, vec!["docs/a.md".to_string()]);
        // Nothing left to prune.
        let again = describe_pending(&conn, &settings, &limits, 1000).unwrap();
        assert_eq!(again.rows_pruned, 0);
    }

    #[test]
    fn a_failed_row_is_written_at_version_zero() {
        let (conn, id) = store_with_project();
        let meta = FileMeta {
            title: "a".to_string(),
            title_source: "filename",
            doc_kind: "md",
        };
        upsert_file_meta_with_version(&conn, id, "docs/a.md", "h1", &meta, 0).unwrap();
        assert_eq!(stored_version(&conn, "docs/a.md"), 0);
        upsert_file_meta(&conn, id, "docs/a.md", "h1", &meta).unwrap();
        assert_eq!(stored_version(&conn, "docs/a.md"), EXTRACT_VERSION);
    }
}
