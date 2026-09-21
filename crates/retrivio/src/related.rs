//! Related chunks and context packs: relation feedback (suppress, restore, quality), chunk lookups, project neighbours, related-chunk retrieval and the context pack builder.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

use crate::config::ConfigValues;
use crate::freshness;
use crate::rank::{
    rank_chunks_native_opts, rank_chunks_native_with, related_chunk_from_ranked,
    related_chunk_result_json, RankedChunkResult, RelatedChunkResult,
};
use crate::util::{truncate_text_chars, word_tokens};

#[derive(Clone)]
pub(crate) struct SourceChunk {
    pub(crate) chunk_id: i64,
    pub(crate) chunk_index: i64,
    pub(crate) project_path: String,
    pub(crate) doc_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) text: String,
}

#[derive(Clone)]
pub(crate) struct IndexedChunkRow {
    pub(crate) chunk_id: i64,
    pub(crate) chunk_index: i64,
    pub(crate) project_path: String,
    pub(crate) doc_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) doc_mtime: f64,
    pub(crate) token_count: i64,
    pub(crate) text: String,
}

#[derive(Clone)]
pub(crate) struct ChunkRelationFeedbackRow {
    pub(crate) src_chunk_id: i64,
    pub(crate) dst_chunk_id: i64,
    pub(crate) relation: String,
    pub(crate) decision: String,
    pub(crate) quality_label: String,
    pub(crate) note: String,
    pub(crate) source: String,
    pub(crate) created_at: f64,
    pub(crate) updated_at: f64,
    pub(crate) dst_chunk_index: i64,
    pub(crate) dst_doc_path: String,
    pub(crate) dst_doc_rel_path: String,
    pub(crate) dst_project_path: String,
}

#[allow(clippy::too_many_arguments)] // one feedback row: the edge (src, dst, relation) plus decision, note, source and time
pub(crate) fn apply_chunk_relation_decision(
    conn: &Connection,
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: &str,
    decision: &str,
    note: &str,
    source: &str,
    ts: f64,
) -> Result<(), String> {
    let rel = relation.trim();
    if src_chunk_id <= 0 || dst_chunk_id <= 0 {
        return Err("source_chunk_id and target_chunk_id must be positive integers".to_string());
    }
    if rel.is_empty() {
        return Err("relation must be non-empty".to_string());
    }
    if !matches!(decision, "active" | "suppressed") {
        return Err("decision must be 'active' or 'suppressed'".to_string());
    }
    conn.execute(
        r#"
INSERT INTO chunk_relation_feedback(
    src_chunk_id, dst_chunk_id, relation, decision, note, source, created_at, updated_at
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
ON CONFLICT(src_chunk_id, dst_chunk_id, relation) DO UPDATE SET
    decision = excluded.decision,
    note = excluded.note,
    source = excluded.source,
    updated_at = excluded.updated_at
"#,
        params![
            src_chunk_id,
            dst_chunk_id,
            rel,
            decision,
            note.trim(),
            source.trim(),
            ts
        ],
    )
    .map_err(|e| format!("failed writing relation feedback: {}", e))?;
    Ok(())
}

pub(crate) fn normalize_relation_quality_label(raw: &str) -> Option<&'static str> {
    let v = raw.trim().to_lowercase();
    match v.as_str() {
        "" | "unspecified" | "clear" | "none" => Some("unspecified"),
        "good" => Some("good"),
        "weak" => Some("weak"),
        "wrong" => Some("wrong"),
        _ => None,
    }
}

pub(crate) fn relation_quality_multiplier(cfg: &ConfigValues, quality_label: &str) -> f64 {
    match quality_label {
        "good" => 1.0 + cfg.rank_relation_quality_good_boost.clamp(0.0, 1.0),
        "weak" => (1.0 - cfg.rank_relation_quality_weak_penalty.clamp(0.0, 1.0)).max(0.0),
        "wrong" => (1.0 - cfg.rank_relation_quality_wrong_penalty.clamp(0.0, 1.0)).max(0.0),
        _ => 1.0,
    }
}

#[allow(clippy::too_many_arguments)] // one feedback row: the edge (src, dst, relation) plus label, note, source and time
pub(crate) fn set_chunk_relation_quality(
    conn: &Connection,
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: &str,
    quality_label: &str,
    note: &str,
    source: &str,
    ts: f64,
) -> Result<String, String> {
    let rel = relation.trim();
    if src_chunk_id <= 0 || dst_chunk_id <= 0 {
        return Err("source_chunk_id and target_chunk_id must be positive integers".to_string());
    }
    if rel.is_empty() {
        return Err("relation must be non-empty".to_string());
    }
    let label = normalize_relation_quality_label(quality_label).ok_or_else(|| {
        "quality_label must be one of: good, weak, wrong, unspecified".to_string()
    })?;
    conn.execute(
        r#"
INSERT INTO chunk_relation_feedback(
    src_chunk_id, dst_chunk_id, relation, decision, quality_label, note, source, created_at, updated_at
)
VALUES (?1, ?2, ?3, 'active', ?4, ?5, ?6, ?7, ?7)
ON CONFLICT(src_chunk_id, dst_chunk_id, relation) DO UPDATE SET
    quality_label = excluded.quality_label,
    note = CASE
        WHEN excluded.note <> '' THEN excluded.note
        ELSE chunk_relation_feedback.note
    END,
    source = excluded.source,
    updated_at = excluded.updated_at
"#,
        params![
            src_chunk_id,
            dst_chunk_id,
            rel,
            label,
            note.trim(),
            source.trim(),
            ts
        ],
    )
    .map_err(|e| format!("failed writing relation quality: {}", e))?;
    Ok(label.to_string())
}

pub(crate) fn list_chunk_relation_feedback(
    conn: &Connection,
    src_chunk_id: i64,
    decision: Option<&str>,
    quality_label: Option<&str>,
    limit: usize,
) -> Result<Vec<ChunkRelationFeedbackRow>, String> {
    let decision_filter = decision.unwrap_or("").trim().to_string();
    let quality_filter = quality_label.unwrap_or("").trim().to_string();
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    f.src_chunk_id,
    f.dst_chunk_id,
    f.relation,
    f.decision,
    f.quality_label,
    f.note,
    f.source,
    f.created_at,
    f.updated_at,
    COALESCE(pc.chunk_index, -1) AS dst_chunk_index,
    COALESCE(pc.doc_path, '') AS dst_doc_path,
    COALESCE(pc.doc_rel_path, '') AS dst_doc_rel_path,
    COALESCE(p.path, '') AS dst_project_path
FROM chunk_relation_feedback f
LEFT JOIN project_chunks pc ON pc.id = f.dst_chunk_id
LEFT JOIN projects p ON p.id = pc.project_id
WHERE f.src_chunk_id = ?1
  AND (?2 = '' OR f.decision = ?2)
  AND (?3 = '' OR f.quality_label = ?3)
ORDER BY f.updated_at DESC, f.id DESC
LIMIT ?4
"#,
        )
        .map_err(|e| format!("failed preparing relation feedback query: {}", e))?;
    let rows = stmt
        .query_map(
            params![
                src_chunk_id,
                decision_filter,
                quality_filter,
                limit.clamp(1, 2000) as i64
            ],
            |row| {
                Ok(ChunkRelationFeedbackRow {
                    src_chunk_id: row.get(0)?,
                    dst_chunk_id: row.get(1)?,
                    relation: row.get(2)?,
                    decision: row.get(3)?,
                    quality_label: row.get(4)?,
                    note: row.get(5)?,
                    source: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    dst_chunk_index: row.get(9)?,
                    dst_doc_path: row.get(10)?,
                    dst_doc_rel_path: row.get(11)?,
                    dst_project_path: row.get(12)?,
                })
            },
        )
        .map_err(|e| format!("failed querying relation feedback rows: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading relation feedback row: {}", e))?);
    }
    Ok(out)
}

pub(crate) fn suppressed_relation_set(
    conn: &Connection,
    src_chunk_id: i64,
) -> Result<HashSet<(i64, String)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst_chunk_id, relation
FROM chunk_relation_feedback
WHERE src_chunk_id = ?1
  AND decision = 'suppressed'
"#,
        )
        .map_err(|e| format!("failed preparing suppressed relation query: {}", e))?;
    let rows = stmt
        .query_map(params![src_chunk_id], |row| {
            let dst: i64 = row.get(0)?;
            let relation: String = row.get(1)?;
            Ok((dst, relation))
        })
        .map_err(|e| format!("failed querying suppressed relation rows: {}", e))?;
    let mut out: HashSet<(i64, String)> = HashSet::new();
    for row in rows {
        let (dst, relation) =
            row.map_err(|e| format!("failed reading suppressed relation row: {}", e))?;
        out.insert((dst, relation));
    }
    Ok(out)
}

pub(crate) fn active_relation_quality_map(
    conn: &Connection,
    src_chunk_id: i64,
) -> Result<HashMap<(i64, String), String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst_chunk_id, relation, quality_label
FROM chunk_relation_feedback
WHERE src_chunk_id = ?1
  AND decision = 'active'
"#,
        )
        .map_err(|e| format!("failed preparing relation quality query: {}", e))?;
    let rows = stmt
        .query_map(params![src_chunk_id], |row| {
            let dst: i64 = row.get(0)?;
            let relation: String = row.get(1)?;
            let quality_label: String = row.get(2)?;
            Ok((dst, relation, quality_label))
        })
        .map_err(|e| format!("failed querying relation quality rows: {}", e))?;
    let mut out: HashMap<(i64, String), String> = HashMap::new();
    for row in rows {
        let (dst, relation, quality_label) =
            row.map_err(|e| format!("failed reading relation quality row: {}", e))?;
        out.insert((dst, relation), quality_label);
    }
    Ok(out)
}

pub(crate) fn relation_feedback_row_json(row: &ChunkRelationFeedbackRow) -> Value {
    serde_json::json!({
        "source_chunk_id": row.src_chunk_id,
        "target_chunk_id": row.dst_chunk_id,
        "relation": row.relation,
        "decision": row.decision,
        "quality_label": row.quality_label,
        "note": row.note,
        "source": row.source,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "target_chunk_index": row.dst_chunk_index,
        "target_path": row.dst_doc_path,
        "target_doc_rel_path": row.dst_doc_rel_path,
        "target_project_path": row.dst_project_path,
    })
}

pub(crate) fn source_chunk_json(source: &SourceChunk) -> Value {
    serde_json::json!({
        "chunk_id": source.chunk_id,
        "chunk_index": source.chunk_index,
        "path": source.doc_path,
        "project_path": source.project_path,
        "doc_rel_path": source.doc_rel_path,
    })
}

pub(crate) fn source_chunk_by_id(
    conn: &Connection,
    chunk_id: i64,
) -> Result<Option<SourceChunk>, String> {
    conn.query_row(
        r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id = ?1
"#,
        params![chunk_id],
        |row| {
            Ok(SourceChunk {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                text: row.get(5)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed loading source chunk {}: {}", chunk_id, e))
}

pub(crate) fn indexed_chunk_by_id(
    conn: &Connection,
    chunk_id: i64,
) -> Result<Option<IndexedChunkRow>, String> {
    conn.query_row(
        r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.doc_mtime,
    pc.token_count,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id = ?1
"#,
        params![chunk_id],
        |row| {
            Ok(IndexedChunkRow {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                doc_mtime: row.get(5)?,
                token_count: row.get(6)?,
                text: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed loading chunk {}: {}", chunk_id, e))
}

pub(crate) fn indexed_doc_chunks_by_path(
    conn: &Connection,
    doc_path: &str,
) -> Result<Vec<IndexedChunkRow>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id,
    pc.chunk_index,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.doc_mtime,
    pc.token_count,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.doc_path = ?1
ORDER BY pc.chunk_index ASC
"#,
        )
        .map_err(|e| format!("failed preparing indexed doc chunks query: {}", e))?;
    let rows = stmt
        .query_map(params![doc_path], |row| {
            Ok(IndexedChunkRow {
                chunk_id: row.get(0)?,
                chunk_index: row.get(1)?,
                project_path: row.get(2)?,
                doc_path: row.get(3)?,
                doc_rel_path: row.get(4)?,
                doc_mtime: row.get(5)?,
                token_count: row.get(6)?,
                text: row.get(7)?,
            })
        })
        .map_err(|e| format!("failed querying indexed doc chunks: {}", e))?;
    let mut out: Vec<IndexedChunkRow> = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading indexed doc chunk row: {}", e))?);
    }
    Ok(out)
}

pub(crate) fn project_neighbor_weights(
    conn: &Connection,
    path: &str,
    limit: usize,
) -> Result<HashMap<String, f64>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, MAX(weight) AS w
FROM (
    SELECT pe.dst AS path, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE src.path = ?1
    UNION ALL
    SELECT src.path AS path, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE pe.dst = ?1
)
GROUP BY path
ORDER BY w DESC, path ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project neighbor weights query: {}", e))?;
    let rows = stmt
        .query_map(params![path, limit as i64], |row| {
            let p: String = row.get(0)?;
            let w: f64 = row.get(1)?;
            Ok((p, w))
        })
        .map_err(|e| format!("failed querying project neighbor weights: {}", e))?;
    let mut raw: Vec<(String, f64)> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading project neighbor weight row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.iter().map(|(_, w)| *w).fold(0.0f64, f64::max);
    if hi <= 0.0 {
        return Ok(raw.into_iter().map(|(p, _)| (p, 0.0)).collect());
    }
    Ok(raw.into_iter().map(|(p, w)| (p, w / hi)).collect())
}

/// Multi-hop BFS to find project neighbors with decaying weights.
/// Returns project_path -> (weight, hop_count) map.
///
/// Starting from a seed project, explores neighbors up to `max_hops` deep.
/// Weight decays by `decay` per hop: hop 1 = edge_weight * decay, hop 2 = edge_weight₁ * edge_weight₂ * decay².
/// Prunes paths with weight < 0.05.
pub(crate) fn multi_hop_project_neighbors(
    conn: &Connection,
    seed_path: &str,
    max_hops: usize,
    per_hop_limit: usize,
    decay: f64,
) -> Result<HashMap<String, (f64, usize)>, String> {
    let mut result: HashMap<String, (f64, usize)> = HashMap::new();
    let mut frontier: Vec<(String, f64)> = vec![(seed_path.to_string(), 1.0)];
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(seed_path.to_string());

    for hop in 1..=max_hops {
        let mut next_frontier: Vec<(String, f64)> = Vec::new();

        for (project_path, incoming_weight) in &frontier {
            let neighbors = project_neighbor_weights(conn, project_path, per_hop_limit)?;
            for (neighbor_path, edge_weight) in neighbors {
                if visited.contains(&neighbor_path) {
                    continue;
                }
                let hop_weight = incoming_weight * edge_weight * decay;
                if hop_weight < 0.05 {
                    continue;
                }
                // Keep the best weight if we reach the same project via multiple paths
                let entry = result.entry(neighbor_path.clone()).or_insert((0.0, hop));
                if hop_weight > entry.0 {
                    *entry = (hop_weight, hop);
                }
                next_frontier.push((neighbor_path, hop_weight));
            }
        }

        // Mark all nodes from this hop as visited
        for (path, _) in &next_frontier {
            visited.insert(path.clone());
        }
        frontier = next_frontier;
    }

    Ok(result)
}

pub(crate) fn related_chunks_native(
    conn: &Connection,
    cfg: &ConfigValues,
    chunk_id: i64,
    limit: usize,
) -> Result<(SourceChunk, Vec<RelatedChunkResult>), String> {
    let source = source_chunk_by_id(conn, chunk_id)?
        .ok_or_else(|| format!("chunk {} was not found", chunk_id))?;
    let query_text: String = source.text.chars().take(2400).collect();
    let candidate_limit = std::cmp::max(40, limit.max(1) * 8).min(400);
    // One plain ranker pass over the source text: no HyDE, no reranker (slice 4). The
    // source text is already the best possible query for "chunks like this one".
    let ranked =
        rank_chunks_native_opts(conn, cfg, &query_text, candidate_limit, None, false, false)?;
    let neighbor_weights = project_neighbor_weights(conn, &source.project_path, 120)?;
    let out = related_from_ranked(
        conn,
        cfg,
        source.chunk_id,
        &source.doc_path,
        &source.project_path,
        &ranked,
        &neighbor_weights,
        &HashSet::new(),
        limit,
    )?;
    Ok((source, out))
}

/// Relation of a candidate chunk to a source (file, project or project edge) and its weight;
/// `None` when the two are unrelated.
pub(crate) fn chunk_relation(
    source_doc_path: &str,
    source_project_path: &str,
    row: &RankedChunkResult,
    neighbor_weights: &HashMap<String, f64>,
) -> Option<(String, f64)> {
    if row.path == source_doc_path {
        Some(("same_file".to_string(), 1.0))
    } else if row.project_path == source_project_path {
        Some(("same_project".to_string(), 0.82))
    } else {
        neighbor_weights.get(&row.project_path).map(|weight| {
            (
                "project_edge".to_string(),
                (0.55 + (0.45 * *weight)).clamp(0.0, 1.0),
            )
        })
    }
}

/// Related chunks for one source chosen from an already ranked candidate set: the relation
/// weight blends with the candidate's own score, suppressed relations are dropped, quality
/// feedback multiplies, `exclude` (chunks already placed elsewhere) are skipped. No retrieval
/// happens here; the caller decides where the candidates come from.
#[allow(clippy::too_many_arguments)] // the seed chunk's identity (id, doc, project) plus the ranked pool, weights, exclusions and limit
pub(crate) fn related_from_ranked(
    conn: &Connection,
    cfg: &ConfigValues,
    source_chunk_id: i64,
    source_doc_path: &str,
    source_project_path: &str,
    ranked: &[RankedChunkResult],
    neighbor_weights: &HashMap<String, f64>,
    exclude: &HashSet<i64>,
    limit: usize,
) -> Result<Vec<RelatedChunkResult>, String> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let suppressed = suppressed_relation_set(conn, source_chunk_id)?;
    let quality_feedback = active_relation_quality_map(conn, source_chunk_id)?;
    let mut out: Vec<RelatedChunkResult> = Vec::new();
    for row in ranked {
        if row.chunk_id == source_chunk_id || exclude.contains(&row.chunk_id) {
            continue;
        }
        let Some((relation, relation_weight)) =
            chunk_relation(source_doc_path, source_project_path, row, neighbor_weights)
        else {
            continue;
        };
        if suppressed.contains(&(row.chunk_id, relation.clone())) {
            continue;
        }
        let relation_quality = quality_feedback
            .get(&(row.chunk_id, relation.clone()))
            .cloned()
            .unwrap_or_else(|| "unspecified".to_string());
        let relation_quality_weight = relation_quality_multiplier(cfg, &relation_quality);
        let score = ((0.72 * row.score) + (0.28 * relation_weight)) * relation_quality_weight;
        out.push(related_chunk_from_ranked(
            row,
            relation,
            relation_weight,
            relation_quality,
            relation_quality_weight,
            score,
        ));
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    out.truncate(limit.max(1));
    Ok(out)
}

pub(crate) fn context_pack_schema() -> &'static str {
    "context-pack-v1"
}

#[derive(Clone, Copy)]
pub(crate) struct ContextPackOptions {
    pub(crate) budget_chars: usize,
    pub(crate) seed_limit: usize,
    pub(crate) related_per_seed: usize,
    pub(crate) include_docs: bool,
    pub(crate) doc_max_chars: usize,
}

pub(crate) fn build_context_pack_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    opts: ContextPackOptions,
) -> Result<Value, String> {
    let q = query.trim();
    if q.is_empty() {
        return Err("query must be non-empty".to_string());
    }

    // Diversity limits — prevent any single file/project from dominating the pack
    const MAX_CHUNKS_PER_FILE: usize = 3;
    const MAX_CHUNKS_PER_PROJECT: usize = 8;

    // One ranker pass for the whole pack (slice 4): the pool serves both the seeds and the
    // related chunks of every seed. Before, every seed re-ran the full ranker (an embedding
    // call for the seed's text, HyDE, the reranker) for its related chunks.
    let seed_limit = opts.seed_limit.clamp(1, 80);
    let pool_size = (seed_limit * (2 + opts.related_per_seed.max(1))).clamp(24, 160);
    let pool = rank_chunks_native_with(conn, cfg, q, pool_size, None)?;
    let seeds = &pool;
    let mut used_chars = 0usize;
    let mut included_chunk_ids: HashSet<i64> = HashSet::new();
    let mut included_token_sets: Vec<HashSet<String>> = Vec::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    let mut project_counts: HashMap<String, usize> = HashMap::new();
    let mut docs_for_pack: HashSet<String> = HashSet::new();

    // First pass: filter seeds with dedup + diversity, collect into a staging list
    struct StagedChunk {
        seed: RankedChunkResult,
        chunk: IndexedChunkRow,
        related: Vec<Value>,
    }
    let mut staged: Vec<StagedChunk> = Vec::new();

    for seed in seeds.iter().take(seed_limit * 2) {
        if used_chars >= opts.budget_chars {
            break;
        }
        let Some(chunk) = indexed_chunk_by_id(conn, seed.chunk_id)? else {
            continue;
        };
        if !included_chunk_ids.insert(chunk.chunk_id) {
            continue;
        }

        // Diversity: max chunks per file and per project
        let file_count = file_counts.entry(chunk.doc_path.clone()).or_insert(0);
        if *file_count >= MAX_CHUNKS_PER_FILE {
            continue;
        }
        let project_count = project_counts
            .entry(chunk.project_path.clone())
            .or_insert(0);
        if *project_count >= MAX_CHUNKS_PER_PROJECT {
            continue;
        }

        // Dedup: skip if >50% token overlap with any already-included chunk
        let chunk_tokens: HashSet<String> = word_tokens(&chunk.text).into_iter().collect();
        let is_duplicate = included_token_sets.iter().any(|existing_tokens| {
            if existing_tokens.is_empty() || chunk_tokens.is_empty() {
                return false;
            }
            let intersection = existing_tokens.intersection(&chunk_tokens).count();
            let union = existing_tokens.union(&chunk_tokens).count();
            if union == 0 {
                return false;
            }
            (intersection as f64 / union as f64) > 0.50
        });
        if is_duplicate {
            continue;
        }

        included_token_sets.push(chunk_tokens);
        *file_count += 1;
        *project_count += 1;

        let chunk_budget = std::cmp::min(opts.doc_max_chars.max(256), opts.budget_chars);
        let (_, _, text_chars) = truncate_text_chars(&chunk.text, chunk_budget);
        used_chars = used_chars.saturating_add(text_chars);
        docs_for_pack.insert(chunk.doc_path.clone());

        staged.push(StagedChunk {
            seed: seed.clone(),
            chunk,
            related: Vec::new(),
        });

        if staged.len() >= seed_limit {
            break;
        }
    }

    // Related chunks per seed, from the same pool: the staged seeds are excluded, a chunk is
    // related to one seed at most, project-edge weights are looked up once per project.
    if opts.related_per_seed > 0 {
        let mut taken: HashSet<i64> = staged.iter().map(|s| s.chunk.chunk_id).collect();
        let mut neighbor_cache: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for entry in staged.iter_mut() {
            let project = entry.chunk.project_path.clone();
            if !neighbor_cache.contains_key(&project) {
                let weights = project_neighbor_weights(conn, &project, 120)?;
                neighbor_cache.insert(project.clone(), weights);
            }
            let weights = &neighbor_cache[&project];
            let related = related_from_ranked(
                conn,
                cfg,
                entry.chunk.chunk_id,
                &entry.chunk.doc_path,
                &project,
                &pool,
                weights,
                &taken,
                opts.related_per_seed,
            )?;
            for rel in &related {
                taken.insert(rel.chunk_id);
            }
            entry.related = related.iter().map(related_chunk_result_json).collect();
        }
    }

    // Coherence: group same-file chunks together, sorted by chunk_index within each group.
    // Maintain overall ordering by the best score in each file group.
    staged.sort_by(|a, b| {
        let a_file = &a.chunk.doc_path;
        let b_file = &b.chunk.doc_path;
        if a_file == b_file {
            // Same file: order by chunk_index for coherence
            a.chunk.chunk_index.cmp(&b.chunk.chunk_index)
        } else {
            // Different files: order by score (best first)
            b.seed.score.total_cmp(&a.seed.score)
        }
    });

    // Build the packed output with metadata headers
    let mut packed: Vec<Value> = Vec::new();
    for entry in &staged {
        let chunk_budget = std::cmp::min(opts.doc_max_chars.max(256), opts.budget_chars);
        let (text, truncated, text_chars) = truncate_text_chars(&entry.chunk.text, chunk_budget);
        let returned_chars = text.chars().count();

        // Extract project name from path for the metadata header
        let project_name = Path::new(&entry.chunk.project_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("project");

        packed.push(serde_json::json!({
            "rank": packed.len() + 1,
            "chunk_id": entry.chunk.chunk_id,
            "chunk_index": entry.chunk.chunk_index,
            "path": entry.chunk.doc_path,
            "project_path": entry.chunk.project_path,
            "project_name": project_name,
            "doc_rel_path": entry.chunk.doc_rel_path,
            "score": entry.seed.score,
            "semantic": entry.seed.semantic,
            "lexical": entry.seed.lexical,
            "graph": entry.seed.graph,
            "relation": entry.seed.relation,
            "quality": entry.seed.quality,
            "doc_mtime": entry.seed.doc_mtime,
            "content_date": entry.seed.content_date,
            "date_source": entry.seed.date_source,
            "date_basis": freshness::date_basis(entry.seed.date_source),
            "age_days": entry.seed.age_days,
            "freshness_tier": entry.seed.freshness_tier,
            "is_record": entry.seed.is_record,
            "role": entry.seed.role,
            "verify": entry.seed.verify,
            "noise": entry.seed.noise,
            "raw_similarity": entry.seed.raw_similarity,
            "superseded_by": entry.seed.superseded_by,
            "why": entry.seed.why,
            "text_chars": text_chars,
            "returned_chars": returned_chars,
            "truncated": truncated,
            "text": text,
            "related": entry.related,
            "header": format!("--- {} (chunk {}) in {} ---", entry.chunk.doc_rel_path, entry.chunk.chunk_index + 1, project_name)
        }));
    }

    let mut docs_payload: Vec<Value> = Vec::new();
    if opts.include_docs {
        let mut docs_sorted: Vec<String> = docs_for_pack.into_iter().collect();
        docs_sorted.sort();
        for path in docs_sorted {
            let chunks = indexed_doc_chunks_by_path(conn, &path)?;
            if chunks.is_empty() {
                continue;
            }
            let first = &chunks[0];
            let mut full_text = String::new();
            let mut token_total: i64 = 0;
            for ch in &chunks {
                full_text.push_str(&ch.text);
                token_total += ch.token_count;
            }
            let (text, truncated, text_chars) = truncate_text_chars(&full_text, opts.doc_max_chars);
            let returned_chars = text.chars().count();
            docs_payload.push(serde_json::json!({
                "path": first.doc_path,
                "project_path": first.project_path,
                "doc_rel_path": first.doc_rel_path,
                "doc_mtime": first.doc_mtime,
                "chunk_count": chunks.len(),
                "token_count": token_total,
                "text_chars": text_chars,
                "returned_chars": returned_chars,
                "truncated": truncated,
                "text": text
            }));
        }
    }

    Ok(serde_json::json!({
        "schema": context_pack_schema(),
        "query": q,
        "budget_chars": opts.budget_chars,
        "used_chars": used_chars,
        "seed_limit": opts.seed_limit,
        "related_per_seed": opts.related_per_seed,
        "include_docs": opts.include_docs,
        "chunks": packed,
        "docs": docs_payload
    }))
}
