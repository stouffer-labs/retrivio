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
        let db = connect(path.to_string_lossy().as_ref())
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
