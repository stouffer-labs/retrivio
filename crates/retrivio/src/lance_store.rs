use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::array::FixedSizeListArray;
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use tokio::runtime::Runtime;

/// Shared Tokio runtime for LanceDB async operations (lazy, 2 worker threads).
fn runtime() -> &'static Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to create LanceDB tokio runtime")
    })
}

pub struct LanceStore {
    table: Table,
    dim: usize,
}

pub fn dim(store: &LanceStore) -> usize {
    store.dim
}

/// Open or create a LanceDB store at the given directory path.
///
/// Creates a "chunks" table with schema: chunk_id (Int64), vector (FixedSizeList<Float32>).
/// If the table already exists, it is opened.
pub fn open(path: &Path, dim: usize) -> Result<LanceStore, String> {
    runtime().block_on(async {
        // Strong read consistency: every read re-checks the table's latest version, so a
        // long-lived process (daemon, MCP server) sees rows committed by another process
        // (the watcher) without reopening the table. Writes are always consistent.
        let db = connect(path.to_string_lossy().as_ref())
            .read_consistency_interval(std::time::Duration::from_secs(0))
            .execute()
            .await
            .map_err(|e| format!("failed to open LanceDB at '{}': {}", path.display(), e))?;

        let table_names = db
            .table_names()
            .execute()
            .await
            .map_err(|e| format!("failed to list LanceDB tables: {}", e))?;

        let table = if table_names.iter().any(|n| n == "chunks") {
            db.open_table("chunks")
                .execute()
                .await
                .map_err(|e| format!("failed to open LanceDB 'chunks' table: {}", e))?
        } else {
            let schema = make_schema(dim);
            let batch = empty_batch(&schema, dim)?;
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            db.create_table("chunks", Box::new(batches))
                .execute()
                .await
                .map_err(|e| format!("failed to create LanceDB 'chunks' table: {}", e))?
        };

        Ok(LanceStore { table, dim })
    })
}

/// Batch upsert chunk vectors into LanceDB.
///
/// Each entry is (chunk_id, vector). Processes in batches of 1024.
/// Uses merge-insert (upsert) on chunk_id to handle both new and updated vectors.
pub fn upsert_chunks(store: &mut LanceStore, chunks: &[(i64, Vec<f32>)]) -> Result<(), String> {
    if chunks.is_empty() {
        return Ok(());
    }
    let dim = store.dim;
    runtime().block_on(async {
        for batch_slice in chunks.chunks(1024) {
            let ids: Vec<i64> = batch_slice.iter().map(|(id, _)| *id).collect();
            if batch_slice.iter().any(|(_, v)| v.len() != dim) {
                let got = batch_slice
                    .iter()
                    .map(|(_, v)| v.len())
                    .find(|n| *n != dim)
                    .unwrap_or(0);
                return Err(format!(
                    "vector dimension mismatch for LanceDB upsert: expected {}, got {}",
                    dim, got
                ));
            }
            let vectors: Vec<f32> = batch_slice
                .iter()
                .flat_map(|(_, v)| v.iter().copied())
                .collect();

            let id_array = Int64Array::from(ids);
            let vector_array = make_fixed_list_array(&vectors, dim)?;

            let schema = make_schema(dim);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(id_array) as ArrayRef,
                    Arc::new(vector_array) as ArrayRef,
                ],
            )
            .map_err(|e| format!("failed to build RecordBatch for upsert: {}", e))?;

            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

            let mut op = store.table.merge_insert(&["chunk_id"]);
            op.when_matched_update_all(None)
                .when_not_matched_insert_all();
            op.execute(Box::new(batches))
                .await
                .map_err(|e| format!("failed to upsert chunks into LanceDB: {}", e))?;
        }
        Ok(())
    })
}

/// Delete chunks by their IDs from LanceDB.
pub fn delete_chunks(store: &mut LanceStore, ids: &[i64]) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    runtime().block_on(async {
        for batch in ids.chunks(500) {
            let id_list: Vec<String> = batch.iter().map(|id| id.to_string()).collect();
            let filter = format!("chunk_id IN ({})", id_list.join(", "));
            store
                .table
                .delete(&filter)
                .await
                .map_err(|e| format!("failed to delete chunks from LanceDB: {}", e))?;
        }
        Ok(())
    })
}

/// Every chunk_id currently stored in LanceDB (reads only the id column).
pub fn list_chunk_ids(store: &LanceStore) -> Result<Vec<i64>, String> {
    use futures::TryStreamExt;
    use lancedb::query::Select;
    runtime().block_on(async {
        let stream = store
            .table
            .query()
            .select(Select::Columns(vec!["chunk_id".to_string()]))
            .execute()
            .await
            .map_err(|e| format!("failed to scan LanceDB chunk ids: {}", e))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| format!("failed to collect LanceDB chunk ids: {}", e))?;
        let mut out: Vec<i64> = Vec::new();
        for batch in &batches {
            let col = batch
                .column_by_name("chunk_id")
                .ok_or("LanceDB scan missing 'chunk_id' column")?;
            let ids = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("chunk_id column is not Int64")?;
            out.extend((0..ids.len()).map(|i| ids.value(i)));
        }
        Ok(out)
    })
}

/// ANN vector search returning chunk_id → normalized similarity score.
///
/// Scores are normalized to [0, 1] where 1 is most similar.
/// LanceDB returns L2 distances by default; we convert to similarity.
pub fn search_vectors(
    store: &LanceStore,
    query_vector: &[f32],
    limit: usize,
) -> Result<HashMap<i64, f64>, String> {
    let use_limit = limit.max(1);
    runtime().block_on(async {
        let results = store
            .table
            .vector_search(query_vector.to_vec())
            .map_err(|e| format!("failed to build LanceDB vector query: {}", e))?
            .limit(use_limit)
            .execute()
            .await
            .map_err(|e| format!("LanceDB vector search failed: {}", e))?;

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| format!("failed to collect LanceDB search results: {}", e))?;

        let mut raw: Vec<(i64, f64)> = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("chunk_id")
                .ok_or("LanceDB result missing 'chunk_id' column")?;
            let ids = id_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("chunk_id column is not Int64")?;
            let dist_col = batch
                .column_by_name("_distance")
                .ok_or("LanceDB result missing '_distance' column")?;

            for i in 0..ids.len() {
                let chunk_id = ids.value(i);
                let distance = if let Some(d) = dist_col.as_any().downcast_ref::<Float32Array>() {
                    d.value(i) as f64
                } else if let Some(d) = dist_col.as_any().downcast_ref::<Float64Array>() {
                    d.value(i)
                } else {
                    continue;
                };
                raw.push((chunk_id, distance));
            }
        }

        Ok(normalize_distances(&raw))
    })
}

/// Return the total number of vectors stored.
pub fn count(store: &LanceStore) -> Result<usize, String> {
    runtime().block_on(async {
        let n = store
            .table
            .count_rows(None)
            .await
            .map_err(|e| format!("failed to count LanceDB rows: {}", e))?;
        Ok(n)
    })
}

/// What [`optimize`] removed: compaction rewrites fragments, version pruning drops files.
#[derive(Debug, Default, Clone)]
pub struct OptimizeReport {
    pub fragments_removed: usize,
    pub fragments_added: usize,
    pub files_removed: usize,
    pub files_added: usize,
    pub old_versions: u64,
    pub bytes_removed: u64,
}

/// Reclaim disk space after deletes.
///
/// Lance deletes are tombstones (the rows stay in their data files) and every write commits a
/// new dataset version while the old one stays on disk, so the directory never shrinks on its
/// own. This compacts fragments (merging small ones and rewriting those with deleted rows) and
/// then drops every version but the latest so the superseded files are removed.
///
/// Version pruning keeps `delete_unverified` off: a file referenced by any manifest, including
/// the versions being dropped, is known to be dead and is removed; an unreferenced file
/// younger than 7 days may belong to another process's in-flight write and is left alone.
/// What this does not protect is a process still reading an old snapshot at that instant:
/// its one in-flight query or merge can fail with a missing file. Every Retrivio process
/// opens the table with strong read consistency (see [`open`]) and so re-checks the latest
/// version before each operation, which bounds that window to a single operation.
pub fn optimize(store: &LanceStore) -> Result<OptimizeReport, String> {
    use lancedb::table::{CompactionOptions, Duration as LanceDuration, OptimizeAction};
    runtime().block_on(async {
        let mut report = OptimizeReport::default();
        let compacted = store
            .table
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .map_err(|e| format!("LanceDB compaction failed: {}", e))?;
        if let Some(c) = compacted.compaction {
            report.fragments_removed = c.fragments_removed;
            report.fragments_added = c.fragments_added;
            report.files_removed = c.files_removed;
            report.files_added = c.files_added;
        }
        let pruned = store
            .table
            .optimize(OptimizeAction::Prune {
                older_than: Some(LanceDuration::zero()),
                delete_unverified: Some(false),
                error_if_tagged_old_versions: Some(false),
            })
            .await
            .map_err(|e| format!("LanceDB version prune failed: {}", e))?;
        if let Some(p) = pruned.prune {
            report.old_versions = p.old_versions;
            report.bytes_removed = p.bytes_removed;
        }
        Ok(report)
    })
}

/// Bytes used by every regular file under `path` (0 when it does not exist). Symlinks are
/// not followed.
pub fn dir_size_bytes(path: &Path) -> u64 {
    fn walk(dir: &Path, acc: &mut u64) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                walk(&entry.path(), acc);
            } else if meta.is_file() {
                *acc += meta.len();
            }
        }
    }
    let mut total = 0u64;
    walk(path, &mut total);
    total
}

/// Rebuild the LanceDB store from SQLite's `project_chunk_vectors` table.
///
/// This is used for the `reembed` command — drops all existing data and re-ingests
/// every vector from SQLite.
pub fn rebuild_from_sqlite(
    conn: &rusqlite::Connection,
    model_key: &str,
    path: &Path,
) -> Result<LanceStore, String> {
    let dim: usize = conn
        .query_row(
            "SELECT dim FROM project_chunk_vectors WHERE model = ?1 LIMIT 1",
            rusqlite::params![model_key],
            |row| row.get::<_, i64>(0),
        )
        .map(|d| d.max(1) as usize)
        .unwrap_or(384);

    // Delete existing lance directory and recreate
    if path.exists() {
        std::fs::remove_dir_all(path).map_err(|e| {
            format!(
                "failed removing existing LanceDB at '{}': {}",
                path.display(),
                e
            )
        })?;
    }
    std::fs::create_dir_all(path).map_err(|e| {
        format!(
            "failed creating LanceDB directory '{}': {}",
            path.display(),
            e
        )
    })?;

    let mut store = open(path, dim)?;

    let mut stmt = conn
        .prepare(
            "SELECT chunk_id, vector FROM project_chunk_vectors WHERE model = ?1 ORDER BY chunk_id",
        )
        .map_err(|e| format!("failed preparing rebuild query: {}", e))?;
    let rows = stmt
        .query_map(rusqlite::params![model_key], |row| {
            let chunk_id: i64 = row.get(0)?;
            let vector_blob: Vec<u8> = row.get(1)?;
            Ok((chunk_id, vector_blob))
        })
        .map_err(|e| format!("failed querying vectors for rebuild: {}", e))?;

    let mut batch: Vec<(i64, Vec<f32>)> = Vec::with_capacity(1024);
    let mut total = 0usize;
    for row in rows {
        let (chunk_id, blob) = row.map_err(|e| format!("failed reading rebuild row: {}", e))?;
        let vector = blob_to_f32_vec(&blob);
        if vector.len() == dim {
            batch.push((chunk_id, vector));
            total += 1;
        }
        if batch.len() >= 1024 {
            upsert_chunks(&mut store, &batch)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        upsert_chunks(&mut store, &batch)?;
    }
    eprintln!("  lancedb: rebuilt {} vectors from sqlite", total);
    Ok(store)
}

// --- internal helpers ---

fn make_schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
    ]))
}

fn make_fixed_list_array(flat_values: &[f32], dim: usize) -> Result<FixedSizeListArray, String> {
    if dim == 0 {
        return Err("invalid LanceDB vector dim: 0".to_string());
    }
    if flat_values.len() % dim != 0 {
        return Err(format!(
            "invalid vector buffer length for dim {}: {}",
            dim,
            flat_values.len()
        ));
    }
    let values = Float32Array::from(flat_values.to_vec());
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    FixedSizeListArray::try_new(field, dim as i32, Arc::new(values), None)
        .map_err(|e| format!("failed building FixedSizeListArray: {}", e))
}

fn empty_batch(schema: &SchemaRef, dim: usize) -> Result<RecordBatch, String> {
    let ids = Int64Array::from(Vec::<i64>::new());
    let vectors = make_fixed_list_array(&[], dim)?;
    RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids) as ArrayRef, Arc::new(vectors) as ArrayRef],
    )
    .map_err(|e| format!("failed to create empty RecordBatch: {}", e))
}

fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Convert L2 distances to normalized similarity scores [0, 1].
fn normalize_distances(rows: &[(i64, f64)]) -> HashMap<i64, f64> {
    if rows.is_empty() {
        return HashMap::new();
    }
    let sims: Vec<(i64, f64)> = rows
        .iter()
        .map(|(id, dist)| (*id, 1.0 / (1.0 + dist)))
        .collect();

    let mut dedup: HashMap<i64, f64> = HashMap::new();
    for (id, sim) in &sims {
        let prev = dedup.get(id).copied().unwrap_or(f64::NEG_INFINITY);
        if *sim > prev {
            dedup.insert(*id, *sim);
        }
    }
    if dedup.is_empty() {
        return HashMap::new();
    }

    let lo = dedup.values().copied().fold(f64::INFINITY, f64::min);
    let hi = dedup.values().copied().fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < f64::EPSILON {
        return dedup.into_keys().map(|id| (id, 1.0)).collect();
    }
    let span = hi - lo;
    dedup
        .into_iter()
        .map(|(id, sim)| (id, ((sim - lo) / span).clamp(0.0, 1.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    /// A fresh directory under the workspace `tmp/` (never the system temp dir).
    fn temp_lance_dir(prefix: &str) -> PathBuf {
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("lance-test-{}-{}-{}", prefix, std::process::id(), n));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create temp lance dir");
        p
    }

    fn vec_for(id: i64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|k| ((id as f32) * 0.37 + (k as f32) * 0.11).sin())
            .collect()
    }

    fn version_count(dir: &Path) -> usize {
        std::fs::read_dir(dir.join("chunks.lance").join("_versions"))
            .map(|rd| rd.count())
            .unwrap_or(0)
    }

    /// Two `LanceStore` handles on one directory stand in for two processes (the watcher
    /// writing, a daemon reading): each holds its own dataset snapshot, so without strong
    /// read consistency the reader would stay pinned to the version it opened.
    #[test]
    fn a_second_handle_sees_rows_written_through_the_first() {
        let dir = temp_lance_dir("fresh");
        let dim = 8;
        let mut writer = open(&dir, dim).expect("open writer");
        let reader = open(&dir, dim).expect("open reader");
        assert_eq!(count(&reader).unwrap(), 0);

        upsert_chunks(&mut writer, &[(1, vec_for(1, dim)), (2, vec_for(2, dim))]).unwrap();
        assert_eq!(count(&reader).unwrap(), 2, "reader must see the writer's commit");
        let hits = search_vectors(&reader, &vec_for(2, dim), 1).unwrap();
        assert!(hits.contains_key(&2), "search on the stale handle: {:?}", hits);

        delete_chunks(&mut writer, &[1]).unwrap();
        assert_eq!(list_chunk_ids(&reader).unwrap(), vec![2]);

        // And the other way round: a handle that was written through sees a later writer.
        let mut late = open(&dir, dim).expect("open late writer");
        upsert_chunks(&mut late, &[(3, vec_for(3, dim))]).unwrap();
        let mut ids = list_chunk_ids(&writer).unwrap();
        ids.sort();
        assert_eq!(ids, vec![2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optimize_after_deletes_keeps_rows_and_drops_old_versions() {
        let dir = temp_lance_dir("optimize");
        let dim = 8;
        let mut store = open(&dir, dim).expect("open");
        // Several commits so there are many fragments and versions to fold.
        for batch in 0..5i64 {
            let rows: Vec<(i64, Vec<f32>)> = (0..40)
                .map(|k| {
                    let id = batch * 40 + k;
                    (id, vec_for(id, dim))
                })
                .collect();
            upsert_chunks(&mut store, &rows).unwrap();
        }
        let doomed: Vec<i64> = (0..200).filter(|id| id % 2 == 0).collect();
        delete_chunks(&mut store, &doomed).unwrap();
        assert_eq!(count(&store).unwrap(), 100);
        let versions_before = version_count(&dir);
        assert!(versions_before > 2, "versions before: {}", versions_before);
        let size_before = dir_size_bytes(&dir);
        assert!(size_before > 0);

        let report = optimize(&store).expect("optimize");
        assert!(report.old_versions > 0, "{:?}", report);
        assert!(report.fragments_removed > 0, "{:?}", report);
        assert!(report.bytes_removed > 0, "{:?}", report);
        let versions_after = version_count(&dir);
        assert!(
            versions_after < versions_before,
            "versions {} -> {}",
            versions_before,
            versions_after
        );
        assert!(dir_size_bytes(&dir) < size_before);

        // Data intact: same rows, search still works, further writes still work.
        assert_eq!(count(&store).unwrap(), 100);
        let mut ids = list_chunk_ids(&store).unwrap();
        ids.sort();
        let expected: Vec<i64> = (0..200).filter(|id| id % 2 == 1).collect();
        assert_eq!(ids, expected);
        let hits = search_vectors(&store, &vec_for(7, dim), 1).unwrap();
        assert!(hits.contains_key(&7), "{:?}", hits);
        upsert_chunks(&mut store, &[(500, vec_for(500, dim))]).unwrap();
        assert_eq!(count(&store).unwrap(), 101);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_size_counts_regular_files_only() {
        let dir = temp_lance_dir("dirsize");
        assert_eq!(dir_size_bytes(&dir.join("missing")), 0);
        std::fs::create_dir_all(dir.join("a").join("b")).unwrap();
        std::fs::write(dir.join("a").join("x.bin"), vec![0u8; 10]).unwrap();
        std::fs::write(dir.join("a").join("b").join("y.bin"), vec![0u8; 5]).unwrap();
        assert_eq!(dir_size_bytes(&dir), 15);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
