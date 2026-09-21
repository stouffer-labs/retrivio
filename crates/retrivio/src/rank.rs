//! Ranking: query classification, the file, chunk and project rankers, coverage terms, noise and summary-page judgement, supersession, duplicate collapse, and the result JSON contracts.

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rusqlite::{params, params_from_iter, Connection};
use serde_json::Value;
use xxhash_rust::xxh64;

use crate::config::{is_any_indexable_suffix, ConfigValues};
use crate::db::{
    ensure_reembed_ready, get_or_open_lance, list_project_paths, list_tracked_roots_conn,
    vector_dim_from_sqlite, with_lance_store,
};
use crate::embed::{embed_query_cached, ollama_host, ollama_is_reachable};
use crate::related::multi_hop_project_neighbors;
use crate::roles::{Role, TextShape};
use crate::util::{blob_to_f32_vec, collapse_whitespace, cosine_raw, now_ts, vector_norm};
use crate::{freshness, lance_store, recall, roles};

pub(crate) fn hybrid_search_lance(
    conn: &Connection,
    model_key: &str,
    query: &str,
    query_vector: &[f32],
    semantic_limit: usize,
    lexical_limit: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    // 1. LanceDB ANN vector search
    let dim = vector_dim_from_sqlite(conn, model_key).unwrap_or_else(|| query_vector.len().max(1));
    let dummy = Path::new(""); // data_dir ignores cwd
    get_or_open_lance(dummy, dim)?;
    let semantic_scores =
        with_lance_store(|store| lance_store::search_vectors(store, query_vector, semantic_limit))?;
    if env::var("RETRIVIO_DEBUG_BACKFILL")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for h in semantic_scores.values() {
            lo = lo.min(h.raw_similarity);
            hi = hi.max(h.raw_similarity);
        }
        let mut ids: Vec<i64> = semantic_scores.keys().copied().collect();
        ids.sort_unstable();
        let probe = env::var("RETRIVIO_DEBUG_CHUNK")
            .ok()
            .and_then(|v| v.parse::<i64>().ok());
        eprintln!(
            "lance: {} of {} requested vector hits, cosine {:.3} to {:.3}, id-set hash {:016x}{}",
            semantic_scores.len(),
            semantic_limit,
            lo,
            hi,
            xxh64::xxh64(format!("{:?}", ids).as_bytes(), 0),
            probe
                .map(|id| format!(
                    ", chunk {} {}",
                    id,
                    if semantic_scores.contains_key(&id) {
                        "present"
                    } else {
                        "absent"
                    }
                ))
                .unwrap_or_default()
        );
    }

    // 2. SQLite FTS5 BM25 search (existing function)
    let lexical_signals = search_lexical_chunks_sqlite(conn, query, lexical_limit)?;
    if env::var("RETRIVIO_DEBUG_BACKFILL")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let mut ids: Vec<i64> = lexical_signals.keys().copied().collect();
        ids.sort_unstable();
        let probe = env::var("RETRIVIO_DEBUG_CHUNK")
            .ok()
            .and_then(|v| v.parse::<i64>().ok());
        eprintln!(
            "fts: {} of {} requested keyword hits, id-set hash {:016x}{}",
            lexical_signals.len(),
            lexical_limit,
            xxh64::xxh64(format!("{:?}", ids).as_bytes(), 0),
            probe
                .map(|id| format!(
                    ", chunk {} {}",
                    id,
                    if lexical_signals.contains_key(&id) {
                        "present"
                    } else {
                        "absent"
                    }
                ))
                .unwrap_or_default()
        );
    }
    let lexical_scores: HashMap<i64, f64> = lexical_signals
        .iter()
        .map(|(id, sig)| (*id, sig.lexical))
        .collect();

    // 3. Join with metadata via existing chunk_signals_for_ids(); lexical-only hits get their
    //    cosine from the SQLite vectors so every candidate carries a raw similarity.
    chunk_signals_for_ids(
        conn,
        &semantic_scores,
        &lexical_scores,
        &CoverageTerms::from_query(query),
        Some((model_key, query_vector)),
    )
}

pub(crate) fn list_neighbors_by_path(
    conn: &Connection,
    path: &str,
    limit: usize,
) -> Result<Vec<(String, String, f64)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT pe.dst, pe.kind, pe.weight
FROM projects p
JOIN project_edges pe ON pe.src_project_id = p.id
WHERE p.path = ?1
ORDER BY pe.weight DESC, pe.dst ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing neighbor query: {}", e))?;
    let rows = stmt
        .query_map(params![path, limit as i64], |row| {
            let dst: String = row.get(0)?;
            let kind: String = row.get(1)?;
            let weight: f64 = row.get(2)?;
            Ok((dst, kind, weight))
        })
        .map_err(|e| format!("failed querying neighbors: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading neighbor row: {}", e))?);
    }
    Ok(out)
}

#[derive(Clone)]
pub(crate) struct EvidenceHit {
    chunk_id: i64,
    pub(crate) chunk_index: i64,
    doc_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    pub(crate) relation: String,
    quality: f64,
    pub(crate) excerpt: String,
    // Freshness (spec §4) and role (slice 3)
    content_date: f64,
    date_source: &'static str,
    age_days: f64,
    freshness_tier: String,
    is_record: bool,
    role: &'static str,
    verify: bool,
    noise: bool,
    raw_similarity: Option<f64>,
    recency: f64,
    /// Which signals contributed (slice 4); see [`why_string`].
    why: String,
}

#[derive(Clone)]
pub(crate) struct RankedResult {
    pub(crate) path: String,
    pub(crate) score: f64,
    pub(crate) lexical: f64,
    pub(crate) semantic: f64,
    pub(crate) frecency: f64,
    pub(crate) graph: f64,
    /// Relevance-weighted mean recency of the project's evidence (spec §4).
    recency: f64,
    pub(crate) evidence: Vec<EvidenceHit>,
}

#[derive(Clone)]
pub(crate) struct RankedFileResult {
    pub(crate) path: String,
    pub(crate) project_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) chunk_id: i64,
    pub(crate) chunk_index: i64,
    /// Final ranking score (relevance blended with recency).
    pub(crate) score: f64,
    /// Relevance before the recency blend; `retrivio recall` thresholds on this.
    pub(crate) base_score: f64,
    pub(crate) semantic: f64,
    pub(crate) lexical: f64,
    pub(crate) graph: f64,
    pub(crate) relation: String,
    pub(crate) quality: f64,
    pub(crate) excerpt: String,
    pub(crate) evidence: Vec<EvidenceHit>,
    // Freshness (spec §4)
    pub(crate) doc_mtime: f64,
    pub(crate) content_date: f64,
    pub(crate) date_source: &'static str,
    pub(crate) age_days: f64,
    pub(crate) freshness_tier: String,
    pub(crate) is_record: bool,
    // Roles, honesty and supersession (slice 3)
    /// `state`, `knowledge` or `record`; see `roles.rs`.
    pub(crate) role: &'static str,
    /// True for `state` older than 35 days: the facts were current once and need re-checking.
    pub(crate) verify: bool,
    /// True when the best chunk is a machine artefact (chat dump, lockfile, minified code).
    pub(crate) noise: bool,
    /// Cosine similarity of the best chunk to the query; `None` in lexical-only mode.
    pub(crate) raw_similarity: Option<f64>,
    /// For `state` files, the newest file of the same series when this one is not it.
    pub(crate) superseded_by: Option<String>,
    /// Which signals contributed to the score (slice 4): a compact `+`-joined list such as
    /// `semantic:0.61+lexical:0.40+graph:same_project+recency:fresh`; see [`why_string`].
    pub(crate) why: String,
}

#[derive(Clone)]
pub(crate) struct RankedChunkResult {
    pub(crate) chunk_id: i64,
    pub(crate) chunk_index: i64,
    pub(crate) path: String,
    pub(crate) project_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) score: f64,
    pub(crate) semantic: f64,
    pub(crate) lexical: f64,
    pub(crate) graph: f64,
    pub(crate) relation: String,
    pub(crate) quality: f64,
    pub(crate) excerpt: String,
    // Freshness (spec §4) and role (slice 3)
    pub(crate) doc_mtime: f64,
    pub(crate) content_date: f64,
    pub(crate) date_source: &'static str,
    pub(crate) age_days: f64,
    pub(crate) freshness_tier: String,
    pub(crate) is_record: bool,
    pub(crate) role: &'static str,
    pub(crate) verify: bool,
    pub(crate) noise: bool,
    pub(crate) raw_similarity: Option<f64>,
    /// Label only (slice 4): the newest `state` file of the same series among the files in
    /// this result set when this chunk's file is not it. Chunk search never downranks or
    /// collapses on it.
    pub(crate) superseded_by: Option<String>,
    /// Which signals contributed (slice 4); see [`why_string`].
    pub(crate) why: String,
}

/// Optional knobs for the file/chunk ranking entry points.
#[derive(Clone, Copy, Default)]
pub(crate) struct RankOptions {
    /// Hard filter: drop results whose content date is older than `now - since_days`.
    pub(crate) since_days: Option<f64>,
    /// Skip embeddings entirely and rank from FTS5 lexical signals only (recall fallback).
    pub(crate) lexical_only: bool,
    /// Show superseded `state` files at full strength (also implied by a history query).
    pub(crate) include_superseded: bool,
    /// Raw-cosine floor (0 = off). In semantic mode a candidate passes only with a finite
    /// cosine at or above it: a missing or NaN cosine fails closed. `lexical_only` retrieval
    /// has no cosine and applies no floor.
    pub(crate) min_raw_similarity: f64,
}

/// A raw-cosine floor fails closed: a candidate passes only with a finite cosine at or above
/// the floor. A floor of 0 is off and keeps everything, cosine or not.
pub(crate) fn passes_raw_floor(raw: Option<f64>, floor: f64) -> bool {
    if floor <= 0.0 {
        return true;
    }
    matches!(raw, Some(r) if r.is_finite() && r >= floor)
}

/// Score multiplier for a `state` file that a newer file of the same series supersedes.
pub(crate) const SUPERSEDED_FACTOR: f64 = 0.85;
/// Score multiplier for a summary page about another project (slice 4 experiment): a file
/// whose stem is the name of another indexed project (`projects/202608-acme-rollout.md` next
/// to a project directory `202608-acme-rollout`), living outside that project, with evidence
/// that it is a digest and not a document that happens to share the name (see
/// [`is_summary_page`]). Such pages are AI-written digests of the source project and crowd the
/// source documents out of the top results; the mild penalty lets the sources win ties.
pub(crate) const SUMMARY_PAGE_FACTOR: f64 = 0.85;

/// [`SUMMARY_PAGE_FACTOR`], or `RETRIVIO_SUMMARY_PAGE_FACTOR` (0.5 to 1.0) when set, read
/// once; the override exists so a factor sweep can run one binary against the scorecard.
pub(crate) fn summary_page_factor() -> f64 {
    static FACTOR: OnceLock<f64> = OnceLock::new();
    *FACTOR.get_or_init(|| {
        env::var("RETRIVIO_SUMMARY_PAGE_FACTOR")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite())
            .map(|f| f.clamp(0.5, 1.0))
            .unwrap_or(SUMMARY_PAGE_FACTOR)
    })
}

/// Directories whose files are digests by convention: `projects/`, `summaries/`, `handoff*`.
pub(crate) fn summary_dir_evidence(doc_rel_path: &str) -> bool {
    Path::new(doc_rel_path)
        .parent()
        .map(|dir| {
            dir.components().any(|c| {
                let name = c.as_os_str().to_string_lossy().to_lowercase();
                name == "projects" || name == "summaries" || name.starts_with("handoff")
            })
        })
        .unwrap_or(false)
}

/// Text evidence: the chunk names three or more distinct paths inside the project it is
/// named after (`<name>/docs/plan.md`, `.../<name>/src/main.rs`), the way a digest points
/// back at its sources; a document that merely shares the name does not.
pub(crate) const SUMMARY_TEXT_MIN_PATHS: usize = 3;

pub(crate) fn text_references_project(text: &str, stem: &str) -> bool {
    if stem.is_empty() {
        return false;
    }
    let needle = format!("{}/", stem);
    let mut seen: HashSet<String> = HashSet::new();
    for raw in text.split_whitespace() {
        let token = raw
            .trim_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')));
        let lower = token.to_lowercase();
        if let Some(idx) = lower.find(&needle) {
            let boundary_ok = idx == 0 || lower.as_bytes()[idx - 1] == b'/';
            let tail = &lower[idx + needle.len()..];
            if boundary_ok && !tail.is_empty() && tail.chars().any(|c| c.is_alphanumeric()) {
                seen.insert(lower[idx..].to_string());
                if seen.len() >= SUMMARY_TEXT_MIN_PATHS {
                    return true;
                }
            }
        }
    }
    false
}

/// Lowercased names that identify a summary page: every indexed project's directory name,
/// plus `<root name>-<project name>` for projects that sit directly under a tracked root
/// (the shape a handoff digest uses for `AI-Activity/Foo` -> `AI-Activity-Foo.md`).
pub(crate) fn summary_page_stems(conn: &Connection, project_paths: &[String]) -> HashSet<String> {
    let roots: Vec<PathBuf> = list_tracked_roots_conn(conn).unwrap_or_default();
    let mut stems: HashSet<String> = HashSet::new();
    for p in project_paths {
        let path = Path::new(p);
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        stems.insert(name.to_lowercase());
        if let Some(parent) = path.parent() {
            if roots.iter().any(|r| r == parent) {
                if let Some(root_name) = parent.file_name().and_then(|s| s.to_str()) {
                    stems.insert(format!("{}-{}", root_name, name).to_lowercase());
                }
            }
        }
    }
    stems
}

/// The other project's name the file's stem spells, when it does (lowercased); `None` for a
/// file not named after another indexed project (the project's own README-like `<project>.md`
/// is not a summary of another project).
pub(crate) fn summary_page_stem(
    doc_rel_path: &str,
    project_path: &str,
    stems: &HashSet<String>,
) -> Option<String> {
    let file_name = Path::new(doc_rel_path)
        .file_name()
        .and_then(|s| s.to_str())?;
    let stem = match file_name.rsplit_once('.') {
        Some((s, ext)) if !s.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric()) => s,
        _ => file_name,
    }
    .to_lowercase();
    if !stems.contains(&stem) {
        return None;
    }
    let own = Path::new(project_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    (stem != own).then_some(stem)
}

/// Decides, once per document and for the duration of one ranking call, whether a file is a
/// summary page of another project (see [`SUMMARY_PAGE_FACTOR`]): its stem names another
/// indexed project *and* there is evidence it is a digest: it lives under a `projects/`,
/// `summaries/` or `handoff*` directory, or the file's text (every chunk of it, so paths
/// spread over chunks count) names three or more paths of that project. A `docs/Globex.md`
/// in a vendor comparison or an `integrations/Acme.md` that merely shares a project's name is
/// left alone. The text is read from the store only for the rare documents whose stem
/// matches and whose directory says nothing, once per document whatever the number of its
/// chunks among the candidates; every chunk of a document gets the same verdict.
pub(crate) struct SummaryPageJudge {
    pub(crate) stems: HashSet<String>,
    pub(crate) verdicts: HashMap<String, bool>,
}

impl SummaryPageJudge {
    fn new(conn: &Connection, project_paths: &[String]) -> Self {
        SummaryPageJudge {
            stems: summary_page_stems(conn, project_paths),
            verdicts: HashMap::new(),
        }
    }

    pub(crate) fn is_summary_page(
        &mut self,
        conn: &Connection,
        doc_path: &str,
        doc_rel_path: &str,
        project_path: &str,
    ) -> bool {
        let Some(stem) = summary_page_stem(doc_rel_path, project_path, &self.stems) else {
            return false;
        };
        if let Some(v) = self.verdicts.get(doc_path) {
            return *v;
        }
        let verdict = summary_dir_evidence(doc_rel_path) || {
            // The stem may carry the root prefix (`ai-activity-foo` for `AI-Activity/Foo`);
            // paths in the text use the project directory name, so try the bare name too.
            let bare = self
                .stems
                .iter()
                .filter(|s| stem.ends_with(&format!("-{}", s)) && s.len() < stem.len())
                .max_by_key(|s| s.len())
                .cloned();
            let text = document_text(conn, doc_path);
            text_references_project(&text, &stem)
                || bare
                    .as_deref()
                    .map(|b| text_references_project(&text, b))
                    .unwrap_or(false)
        };
        self.verdicts.insert(doc_path.to_string(), verdict);
        verdict
    }
}

/// Every chunk text of a document, in chunk order, joined by spaces; empty when unknown.
pub(crate) fn document_text(conn: &Connection, doc_path: &str) -> String {
    let Ok(mut stmt) =
        conn.prepare("SELECT text FROM project_chunks WHERE doc_path = ?1 ORDER BY chunk_index")
    else {
        return String::new();
    };
    let Ok(rows) = stmt.query_map(params![doc_path], |row| row.get::<_, String>(0)) else {
        return String::new();
    };
    rows.flatten().collect::<Vec<_>>().join(" ")
}

/// Score multiplier for a file whose role matches what the prompt asks for (records for
/// "what did they say on the call", state for "current status").
pub(crate) const ROLE_HINT_BOOST: f64 = 1.06;

/// Resolved freshness facts for one document (spec §4), plus its role (slice 3).
#[derive(Clone)]
pub(crate) struct FreshnessInfo {
    doc_mtime: f64,
    content_date: f64,
    pub(crate) date_source: &'static str,
    pub(crate) age_days: f64,
    pub(crate) tier: &'static str,
    pub(crate) is_record: bool,
    pub(crate) role: Role,
    pub(crate) verify: bool,
    pub(crate) recency: f64,
}

/// Per-query freshness context: config-derived half-lives, weights and record patterns,
/// pinned to one `now` so every result in a response is judged against the same clock.
pub(crate) struct FreshnessCtx {
    now: f64,
    record_patterns: Vec<String>,
    living_half_life: f64,
    record_half_life: f64,
    living_weight: f64,
    record_weight: f64,
}

impl FreshnessCtx {
    fn new(cfg: &ConfigValues) -> Self {
        Self::at(cfg, now_ts())
    }

    pub(crate) fn at(cfg: &ConfigValues, now: f64) -> Self {
        Self {
            now,
            record_patterns: cfg.record_patterns(),
            living_half_life: cfg.recency_half_life_days,
            record_half_life: cfg.recency_record_half_life_days,
            living_weight: cfg.rank_recency_weight,
            record_weight: cfg.rank_recency_record_weight,
        }
    }

    /// Freshness and role for one document. The role comes from the path *relative to the
    /// project* (components and file name) plus the shape of the retrieved chunk text; the
    /// date comes from the absolute path (project folders carry dates too) and the mtime, by
    /// role.
    pub(crate) fn info(
        &self,
        doc_rel_path: &str,
        doc_path: &str,
        doc_mtime: f64,
        shape: TextShape,
    ) -> FreshnessInfo {
        let role = roles::classify(doc_rel_path, shape, &self.record_patterns);
        let (content_date, date_source) =
            freshness::content_date_for_role(doc_path, doc_mtime, self.now, role);
        let age = freshness::age_days(self.now, content_date);
        let is_record = role.is_record();
        let half_life = if is_record {
            self.record_half_life
        } else {
            self.living_half_life
        };
        FreshnessInfo {
            doc_mtime,
            content_date,
            date_source,
            age_days: age,
            tier: freshness::tier_for_role(age, role),
            is_record,
            role,
            verify: freshness::needs_verify(age, role),
            recency: freshness::recency_score(age, half_life),
        }
    }

    /// [`Self::info`] for a retrieved chunk.
    fn info_for(&self, row: &ChunkSignal) -> FreshnessInfo {
        self.info(&row.doc_rel_path, &row.doc_path, row.doc_mtime, row.shape)
    }

    fn weight(&self, is_record: bool) -> f64 {
        if is_record {
            self.record_weight
        } else {
            self.living_weight
        }
    }

    /// Recency for an already-resolved age and class (avoids re-parsing the path).
    pub(crate) fn recency_for(&self, age_days: f64, is_record: bool) -> f64 {
        let half_life = if is_record {
            self.record_half_life
        } else {
            self.living_half_life
        };
        freshness::recency_score(age_days, half_life)
    }

    /// `final = (1 - w) * score + w * recency`, with `w` chosen by document class.
    pub(crate) fn blend(&self, score: f64, info: &FreshnessInfo) -> f64 {
        freshness::blend(score, info.recency, self.weight(info.is_record))
    }

    /// True unless `since_days` is set and the content date is older than that window.
    pub(crate) fn within_since(&self, content_date: f64, since_days: Option<f64>) -> bool {
        match since_days {
            None => true,
            Some(days) => content_date >= self.now - days.max(0.0) * freshness::DAY_SECS,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RelatedChunkResult {
    pub(crate) chunk_id: i64,
    pub(crate) chunk_index: i64,
    pub(crate) path: String,
    pub(crate) project_path: String,
    pub(crate) doc_rel_path: String,
    pub(crate) relation: String,
    pub(crate) relation_weight: f64,
    pub(crate) relation_quality: String,
    pub(crate) relation_quality_multiplier: f64,
    pub(crate) score: f64,
    pub(crate) semantic: f64,
    pub(crate) lexical: f64,
    pub(crate) quality: f64,
    pub(crate) excerpt: String,
    // The same deterministic fields as every other chunk result (slice 4).
    pub(crate) doc_mtime: f64,
    pub(crate) content_date: f64,
    pub(crate) date_source: &'static str,
    pub(crate) age_days: f64,
    pub(crate) freshness_tier: String,
    pub(crate) is_record: bool,
    pub(crate) role: &'static str,
    pub(crate) verify: bool,
    pub(crate) noise: bool,
    pub(crate) raw_similarity: Option<f64>,
    pub(crate) superseded_by: Option<String>,
    pub(crate) why: String,
}

/// A related chunk built from a ranked chunk of the same query (or of the source text): the
/// ranked fields are carried through, the relation fields are added.
pub(crate) fn related_chunk_from_ranked(
    row: &RankedChunkResult,
    relation: String,
    relation_weight: f64,
    relation_quality: String,
    relation_quality_multiplier: f64,
    score: f64,
) -> RelatedChunkResult {
    RelatedChunkResult {
        chunk_id: row.chunk_id,
        chunk_index: row.chunk_index,
        path: row.path.clone(),
        project_path: row.project_path.clone(),
        doc_rel_path: row.doc_rel_path.clone(),
        relation,
        relation_weight,
        relation_quality,
        relation_quality_multiplier,
        score,
        semantic: row.semantic,
        lexical: row.lexical,
        quality: row.quality,
        excerpt: row.excerpt.clone(),
        doc_mtime: row.doc_mtime,
        content_date: row.content_date,
        date_source: row.date_source,
        age_days: row.age_days,
        freshness_tier: row.freshness_tier.clone(),
        is_record: row.is_record,
        role: row.role,
        verify: row.verify,
        noise: row.noise,
        raw_similarity: row.raw_similarity,
        superseded_by: row.superseded_by.clone(),
        why: row.why.clone(),
    }
}

/// The compact, deterministic account of the signals behind a result's score (slice 4), the
/// same on every surface (CLI `--json`, API, MCP, the hook block, the dossier):
/// `semantic:<cosine>` when the vector search found the chunk (`cosine:<c>` when only the
/// keyword or path search did and the cosine was backfilled), `lexical:<0..1>` when the FTS
/// matched, `graph:<seed|same_project|related_project>` when the project graph raised it,
/// `path` when the path matched query words, `recency:<fresh|aging>` when a young date lifted
/// it, `role` when the prompt's role hint did, `path-penalty` when a scratch or copy directory
/// lowered it, `noise` when the text is a machine artefact; `superseded` is appended by the
/// supersession step. Parts are joined with `+`; the string is empty only when no signal fired.
pub(crate) fn why_string(
    row: &ChunkSignal,
    kw: f64,
    role_boost: bool,
    fresh: &FreshnessInfo,
    path_penalty: f64,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    match row.raw_similarity {
        Some(cos) if row.semantic > 0.0 => parts.push(format!("semantic:{:.2}", cos)),
        Some(cos) => parts.push(format!("cosine:{:.2}", cos)),
        None => {}
    }
    if row.lexical > 0.0 {
        parts.push(format!("lexical:{:.2}", row.lexical));
    }
    if row.graph > 0.0 && row.relation != "direct" && row.relation != "lexical" {
        parts.push(format!("graph:{}", row.relation));
    }
    if kw >= 0.20 {
        parts.push("path".to_string());
    }
    if matches!(fresh.tier, "fresh" | "aging") && fresh.recency >= 0.5 {
        parts.push(format!("recency:{}", fresh.tier));
    }
    if role_boost {
        parts.push("role".to_string());
    }
    if path_penalty < 1.0 {
        parts.push("path-penalty".to_string());
    }
    if row.noise {
        parts.push("noise".to_string());
    }
    parts.join("+")
}

pub(crate) fn why_append(why: &mut String, part: &str) {
    if !why.is_empty() {
        why.push('+');
    }
    why.push_str(part);
}

pub(crate) fn chunk_search_schema() -> &'static str {
    "chunk-search-v2"
}

pub(crate) fn chunk_related_schema() -> &'static str {
    "chunk-related-v1"
}

pub(crate) fn chunk_get_schema() -> &'static str {
    "chunk-get-v1"
}

pub(crate) fn doc_read_schema() -> &'static str {
    "doc-read-v1"
}

pub(crate) fn ranked_chunk_result_json(item: &RankedChunkResult) -> Value {
    serde_json::json!({
        "chunk_id": item.chunk_id,
        "chunk_index": item.chunk_index,
        "path": item.path,
        "project_path": item.project_path,
        "doc_rel_path": item.doc_rel_path,
        "score": item.score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "graph": item.graph,
        "relation": item.relation,
        "quality": item.quality,
        "excerpt": item.excerpt,
        "doc_mtime": item.doc_mtime,
        "content_date": item.content_date,
        "date_source": item.date_source,
        "age_days": item.age_days,
        "freshness_tier": item.freshness_tier,
        "is_record": item.is_record,
        "role": item.role,
        "verify": item.verify,
        "noise": item.noise,
        "raw_similarity": item.raw_similarity,
        "date_basis": freshness::date_basis(item.date_source),
        "superseded_by": item.superseded_by,
        "why": item.why,
    })
}

pub(crate) fn evidence_hit_json(ev: &EvidenceHit) -> Value {
    serde_json::json!({
        "chunk_id": ev.chunk_id,
        "chunk_index": ev.chunk_index,
        "doc_path": ev.doc_path,
        "doc_rel_path": ev.doc_rel_path,
        "score": ev.score,
        "semantic": ev.semantic,
        "lexical": ev.lexical,
        "graph": ev.graph,
        "relation": ev.relation,
        "quality": ev.quality,
        "excerpt": ev.excerpt,
        "content_date": ev.content_date,
        "date_source": ev.date_source,
        "date_basis": freshness::date_basis(ev.date_source),
        "age_days": ev.age_days,
        "freshness_tier": ev.freshness_tier,
        "is_record": ev.is_record,
        "role": ev.role,
        "verify": ev.verify,
        "noise": ev.noise,
        "raw_similarity": ev.raw_similarity,
        "superseded_by": Value::Null,
        "why": ev.why,
    })
}

pub(crate) fn ranked_file_result_json(item: &RankedFileResult) -> Value {
    let evidence: Vec<Value> = item.evidence.iter().map(evidence_hit_json).collect();
    serde_json::json!({
        "path": item.path,
        "project_path": item.project_path,
        "doc_rel_path": item.doc_rel_path,
        "chunk_id": item.chunk_id,
        "chunk_index": item.chunk_index,
        "score": item.score,
        "base_score": item.base_score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "graph": item.graph,
        "relation": item.relation,
        "quality": item.quality,
        "excerpt": item.excerpt,
        "doc_mtime": item.doc_mtime,
        "content_date": item.content_date,
        "date_source": item.date_source,
        "age_days": item.age_days,
        "freshness_tier": item.freshness_tier,
        "is_record": item.is_record,
        "role": item.role,
        "verify": item.verify,
        "noise": item.noise,
        "raw_similarity": item.raw_similarity,
        "superseded_by": item.superseded_by,
        "date_basis": freshness::date_basis(item.date_source),
        "why": item.why,
        "evidence": evidence,
    })
}

pub(crate) fn ranked_project_result_json(item: &RankedResult) -> Value {
    let evidence: Vec<Value> = item.evidence.iter().map(evidence_hit_json).collect();
    serde_json::json!({
        "path": item.path,
        "score": item.score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "frecency": item.frecency,
        "graph": item.graph,
        "recency": item.recency,
        "evidence": evidence,
    })
}

/// Payload of `GET /search?view=files`; also emitted by `retrivio search --json`.
pub(crate) fn search_files_response_json(
    query: &str,
    rows: &[RankedFileResult],
    started: Instant,
) -> Value {
    let results: Vec<Value> = rows.iter().map(ranked_file_result_json).collect();
    serde_json::json!({
        "query": query,
        "view": "files",
        "results": results,
        "timing_ms": started.elapsed().as_secs_f64() * 1000.0
    })
}

/// Payload of `GET /search?view=projects`; also emitted by `retrivio search --json`.
pub(crate) fn search_projects_response_json(
    query: &str,
    rows: &[RankedResult],
    started: Instant,
) -> Value {
    let results: Vec<Value> = rows.iter().map(ranked_project_result_json).collect();
    serde_json::json!({
        "query": query,
        "view": "projects",
        "results": results,
        "timing_ms": started.elapsed().as_secs_f64() * 1000.0
    })
}

pub(crate) fn related_chunk_result_json(item: &RelatedChunkResult) -> Value {
    serde_json::json!({
        "chunk_id": item.chunk_id,
        "chunk_index": item.chunk_index,
        "path": item.path,
        "project_path": item.project_path,
        "doc_rel_path": item.doc_rel_path,
        "relation": item.relation,
        "relation_weight": item.relation_weight,
        "relation_quality": item.relation_quality,
        "relation_quality_multiplier": item.relation_quality_multiplier,
        "score": item.score,
        "semantic": item.semantic,
        "lexical": item.lexical,
        "quality": item.quality,
        "excerpt": item.excerpt,
        "doc_mtime": item.doc_mtime,
        "content_date": item.content_date,
        "date_source": item.date_source,
        "date_basis": freshness::date_basis(item.date_source),
        "age_days": item.age_days,
        "freshness_tier": item.freshness_tier,
        "is_record": item.is_record,
        "role": item.role,
        "verify": item.verify,
        "noise": item.noise,
        "raw_similarity": item.raw_similarity,
        "superseded_by": item.superseded_by,
        "why": item.why,
    })
}

#[derive(Clone)]
pub(crate) struct ChunkSignal {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    doc_mtime: f64,
    /// Min-max normalised over the vector hits of this query (relative signal for fusion).
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
    /// Cosine similarity to the query vector; `None` when no query vector was used.
    raw_similarity: Option<f64>,
    /// Shape of the chunk text (prose, transcript, chat dump).
    shape: TextShape,
    /// Machine artefact: chat dump, lockfile, minified code, `.jsonl`/`.log` dump.
    noise: bool,
    /// Every query term (or a capitalised name from the query) occurs in the chunk text.
    strong_lexical: bool,
}

/// Facts about a chunk's text that the ranker needs at query time, computed once per row.
pub(crate) struct TextFacts {
    quality: f64,
    shape: TextShape,
    noise: bool,
    strong_lexical: bool,
}

pub(crate) fn text_facts(doc_rel_path: &str, text: &str, cover: &CoverageTerms) -> TextFacts {
    let shape = roles::text_shape(text);
    TextFacts {
        quality: content_quality_with_shape(doc_rel_path, text, shape),
        shape,
        noise: is_noise_artifact(doc_rel_path, shape),
        strong_lexical: cover.strong_match(text),
    }
}

/// Verbs and fillers that open task prompts. Capitalised only because they start the prompt or
/// a sentence, they are never names ("Run the tests", "Summarize this", "Please review").
pub(crate) const PROMPT_VERBS: &[&str] = &[
    "run",
    "read",
    "write",
    "fix",
    "check",
    "review",
    "update",
    "create",
    "make",
    "add",
    "remove",
    "delete",
    "show",
    "list",
    "find",
    "explain",
    "tell",
    "give",
    "look",
    "open",
    "start",
    "stop",
    "test",
    "build",
    "deploy",
    "install",
    "describe",
    "compare",
    "help",
    "please",
    "use",
    "try",
    "continue",
    "think",
    "brainstorm",
    "ultrathink",
    "summarize",
    "summarise",
    "draft",
    "plan",
    "implement",
    "refactor",
    "debug",
    "verify",
    "validate",
    "generate",
    "analyze",
    "analyse",
    "investigate",
    "search",
    "print",
    "save",
    "commit",
    "push",
    "pull",
    "merge",
    "rebase",
    "rewrite",
    "edit",
    "change",
    "move",
    "copy",
    "rename",
    "restart",
    "rerun",
    "retry",
    "resume",
    "note",
    "remember",
    "recall",
    "consider",
    "focus",
    "ensure",
    "prepare",
    "propose",
    "suggest",
    "recommend",
    "evaluate",
    "measure",
    "document",
    "report",
    "ship",
    "release",
    "track",
    "watch",
    "follow",
    "handle",
    "address",
    "resolve",
    "close",
    "finish",
    "complete",
    "done",
    "next",
    "also",
    "then",
    "now",
    "today",
    "first",
    "second",
    "last",
    "final",
    "here",
    "there",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "does",
    "should",
    "could",
    "would",
    "will",
    "again",
    "before",
    "after",
    "once",
    "okay",
];

pub(crate) fn is_prompt_verb(lower: &str) -> bool {
    PROMPT_VERBS.contains(&lower)
}

/// Query words used for lexical coverage: distinctive tokens (3+ characters, no stopwords,
/// at most 12) and the capitalised names as typed ("Acme", "Globex"). A capitalised word is a
/// name only when it is not the first word of the prompt or of a sentence, has four or more
/// letters and is neither a stopword nor a prompt verb: "Run the tests" and "Read the handoff"
/// yield no name.
pub(crate) struct CoverageTerms {
    terms: Vec<String>,
    names: Vec<String>,
}

impl CoverageTerms {
    fn from_query(query: &str) -> Self {
        let mut terms: Vec<String> = Vec::new();
        for tok in all_word_tokens(query) {
            if tok.len() < 3 || recall::is_stopword(&tok) || terms.contains(&tok) {
                continue;
            }
            terms.push(tok);
            if terms.len() >= 12 {
                break;
            }
        }
        let mut names: Vec<String> = Vec::new();
        for line in query.lines() {
            let mut sentence_start = true;
            for raw in line.split_whitespace() {
                let ends_sentence = raw.ends_with(['.', '!', '?', ':', ';']);
                let mut first_in_raw = true;
                for word in raw.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
                    if word.is_empty() {
                        continue;
                    }
                    let at_start = sentence_start && first_in_raw;
                    first_in_raw = false;
                    let first_upper = word
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false);
                    let letters = word.chars().filter(|c| c.is_alphabetic()).count();
                    if at_start || !first_upper || letters < 4 {
                        continue;
                    }
                    // Acronyms and ordinary Capitalised words both count.
                    let lower = word.to_lowercase();
                    if recall::is_stopword(&lower)
                        || is_prompt_verb(&lower)
                        || names.contains(&lower)
                    {
                        continue;
                    }
                    names.push(lower);
                }
                sentence_start = ends_sentence;
            }
        }
        CoverageTerms { terms, names }
    }

    fn from_terms(terms: &[String]) -> Self {
        CoverageTerms {
            terms: terms
                .iter()
                .map(|t| t.to_lowercase())
                .filter(|t| t.len() >= 3)
                .take(12)
                .collect(),
            names: Vec::new(),
        }
    }

    /// True when every term (two or more of them) occurs in `text` as a whole token, or a
    /// capitalised name from the query does *together with* at least one other distinctive
    /// query term. A name alone is a mention, not strong evidence.
    fn strong_match(&self, text: &str) -> bool {
        if self.terms.is_empty() && self.names.is_empty() {
            return false;
        }
        let tokens: HashSet<String> = all_word_tokens(text).into_iter().collect();
        if self.terms.len() >= 2 && self.terms.iter().all(|t| tokens.contains(t)) {
            return true;
        }
        self.names
            .iter()
            .any(|n| tokens.contains(n) && self.terms.iter().any(|t| t != n && tokens.contains(t)))
    }
}

pub(crate) fn print_project_results(results: &[RankedResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    score={:.3} semantic={:.3} lexical={:.3} frecency={:.3} graph={:.3} recency={:.3}",
            idx + 1,
            item.path,
            item.score,
            item.semantic,
            item.lexical,
            item.frecency,
            item.graph,
            item.recency
        );
        for ev in item.evidence.iter().take(4) {
            println!(
                "      - {}#{} (chunk {}) score={:.3} sem={:.3} lex={:.3} gscore={:.3} rel={} q={:.2}\n        {}",
                ev.doc_rel_path,
                ev.chunk_index,
                ev.chunk_id,
                ev.score,
                ev.semantic,
                ev.lexical,
                ev.graph,
                ev.relation,
                ev.quality,
                ev.excerpt
            );
        }
    }
}

pub(crate) fn print_file_results(results: &[RankedFileResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    project={}\n    chunk_id={} chunk_index={}\n    score={:.3} cos={} semantic={:.3} lexical={:.3} graph={:.3} relation={} quality={:.2} role={} date={} age={}d tier={} basis={}{}{}\n    why={}\n    {}",
            idx + 1,
            item.path,
            item.project_path,
            item.chunk_id,
            item.chunk_index,
            item.score,
            item.raw_similarity
                .map(|c| format!("{:.3}", c))
                .unwrap_or_else(|| "n/a".to_string()),
            item.semantic,
            item.lexical,
            item.graph,
            item.relation,
            item.quality,
            item.role,
            freshness::format_ymd(item.content_date),
            item.age_days.round() as i64,
            item.freshness_tier,
            freshness::date_basis(item.date_source),
            if item.verify { " verify" } else { "" },
            item.superseded_by
                .as_deref()
                .map(|s| format!(" superseded_by={}", s))
                .unwrap_or_default(),
            item.why,
            item.excerpt
        );
        for ev in item.evidence.iter().take(4) {
            println!(
                "      - {}#{} (chunk {}) score={:.3} sem={:.3} lex={:.3} gscore={:.3} rel={} q={:.2}\n        {}",
                ev.doc_rel_path,
                ev.chunk_index,
                ev.chunk_id,
                ev.score,
                ev.semantic,
                ev.lexical,
                ev.graph,
                ev.relation,
                ev.quality,
                ev.excerpt
            );
        }
    }
}

pub(crate) fn rank_projects_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedResult>, String> {
    ensure_reembed_ready(conn, cfg, "search")?;
    let existing_paths = list_project_paths(conn)?;
    if existing_paths.is_empty() {
        return Ok(Vec::new());
    }
    let q = query.trim();
    if q.is_empty() {
        return rank_by_frecency_only(conn, limit);
    }
    let query_type = QueryType::classify(q);
    let (sem_limit, lex_limit) = query_type.retrieval_limits(
        cfg.vector_candidates.max(1) as usize,
        cfg.lexical_candidates.max(1) as usize,
    );

    let (model_key, query_vector) = embed_query_cached(cfg, q)?;

    let project_semantic = project_semantic_scores(conn, &model_key, &query_vector, sem_limit)?;
    let mut fused = hybrid_search_lance(
        conn,
        &model_key,
        q,
        &query_vector,
        std::cmp::max(80, sem_limit * 3),
        lex_limit,
    )?;
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;
    let fx = FreshnessCtx::new(cfg);
    let project_evidence = project_evidence(&fused, cfg, &fx);
    let project_content = project_content_scores(&project_evidence);
    let frecency = frecency_scores(conn)?;
    let graph = graph_scores(conn)?;
    let path_keywords = path_keyword_scores(&existing_paths, q);
    let project_mtimes = project_mtimes(conn)?;

    let existing_set: HashSet<String> = existing_paths.iter().cloned().collect();
    let mut all_paths: HashSet<String> = HashSet::new();
    for p in project_content.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in project_semantic.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in frecency.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in graph.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    for p in path_keywords.keys() {
        if existing_set.contains(p) {
            all_paths.insert(p.clone());
        }
    }
    if all_paths.is_empty() {
        for p in existing_paths.iter().take(limit.max(1)) {
            all_paths.insert(p.clone());
        }
    }

    let mut out: Vec<RankedResult> = Vec::new();
    for path in all_paths {
        let evidence = project_evidence.get(&path).cloned().unwrap_or_default();
        let lexical = evidence.iter().map(|e| e.lexical).fold(0.0f64, f64::max);
        let sem_from_chunks = evidence.iter().map(|e| e.semantic).fold(0.0f64, f64::max);
        let semantic = sem_from_chunks.max(*project_semantic.get(&path).unwrap_or(&0.0));
        let content = *project_content.get(&path).unwrap_or(&0.0);
        let fr = *frecency.get(&path).unwrap_or(&0.0);
        let gscore = *graph.get(&path).unwrap_or(&0.0);
        let path_kw = *path_keywords.get(&path).unwrap_or(&0.0);
        let w = query_type.project_weights(cfg);
        let mut score = (w.lexical * content)
            + (w.semantic * semantic)
            + (w.path_kw * path_kw)
            + (w.graph * gscore)
            + (w.frecency * fr);
        // Penalize results with no keyword match for symbol/path queries
        if (query_type == QueryType::Symbol || query_type == QueryType::PathQuery)
            && path_kw < 0.20
            && lexical < 0.05
        {
            score *= 0.45;
        }
        if is_generic_container(&path) && path_kw < 0.4 {
            score *= 0.82;
        }
        // Freshness (spec §4): relevance-weighted mean of the evidence recency, falling back
        // to the project's own mtime when no chunk evidence was retrieved. Blended once.
        let weight_sum: f64 = evidence.iter().map(|e| e.score.max(0.0)).sum();
        let recency = if weight_sum > 0.0 {
            evidence
                .iter()
                .map(|e| e.score.max(0.0) * e.recency)
                .sum::<f64>()
                / weight_sum
        } else {
            project_mtimes
                .get(&path)
                .map(|mt| fx.info("", &path, *mt, TextShape::Prose).recency)
                .unwrap_or(0.0)
        };
        score = freshness::blend(score, recency, fx.weight(false));
        out.push(RankedResult {
            path,
            score,
            lexical,
            semantic,
            frecency: fr,
            graph: gscore,
            recency,
            evidence: evidence.into_iter().take(4).collect(),
        });
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    out.truncate(limit.max(1));
    Ok(out)
}

/// `project_mtime` per project path, used as the recency fallback for projects without evidence.
pub(crate) fn project_mtimes(conn: &Connection) -> Result<HashMap<String, f64>, String> {
    let mut stmt = conn
        .prepare("SELECT path, project_mtime FROM projects")
        .map_err(|e| format!("failed preparing project mtime query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .map_err(|e| format!("failed querying project mtimes: {}", e))?;
    let mut out = HashMap::new();
    for row in rows {
        let (path, mtime) = row.map_err(|e| format!("failed reading project mtime row: {}", e))?;
        out.insert(path, mtime);
    }
    Ok(out)
}

pub(crate) fn rank_files_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedFileResult>, String> {
    rank_files_native_with(conn, cfg, query, limit, RankOptions::default())
}

/// File ranking with options: `since_days` hard filter and `lexical_only` (no embeddings;
/// FTS5 signals only, used by `retrivio recall` when the embedding backend is unavailable).
/// The recency blend (spec §4) is applied once per file after base scoring, before truncation.
pub(crate) fn rank_files_native_with(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
    opts: RankOptions,
) -> Result<Vec<RankedFileResult>, String> {
    if !opts.lexical_only {
        ensure_reembed_ready(conn, cfg, "search")?;
    }
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let query_type = QueryType::classify(q);
    let (sem_limit, lex_limit) = query_type.retrieval_limits(
        cfg.vector_candidates.max(1) as usize,
        cfg.lexical_candidates.max(1) as usize,
    );
    let mut query_vector: Option<(String, Vec<f32>)> = None;
    let (project_semantic, mut fused) = if opts.lexical_only {
        let fused = search_lexical_chunks_sqlite(conn, q, std::cmp::max(160, lex_limit * 2))?;
        (HashMap::new(), fused)
    } else {
        let (model_key, vector) = embed_query_cached(cfg, q)?;
        let project_semantic =
            project_semantic_scores(conn, &model_key, &vector, std::cmp::max(120, sem_limit * 2))?;
        let fused = hybrid_search_lance(
            conn,
            &model_key,
            q,
            &vector,
            std::cmp::max(160, sem_limit * 4),
            std::cmp::max(120, lex_limit * 2),
        )?;
        query_vector = Some((model_key, vector));
        (project_semantic, fused)
    };
    if fused.is_empty() {
        return Ok(Vec::new());
    }
    let fx = FreshnessCtx::new(cfg);
    // For Symbol and PathQuery types, boost path-based matching signals
    if query_type == QueryType::Symbol || query_type == QueryType::PathQuery {
        let path_signals = keyword_path_chunk_scores(conn, q, std::cmp::max(220, lex_limit * 3))?;
        if !path_signals.is_empty() {
            fused = fuse_chunk_signals(&path_signals, &fused);
            // Path-keyword hits arrive without a cosine: give them one from the SQLite
            // vectors so every candidate carries `raw_similarity`.
            if let Some((model_key, query_vector)) = query_vector.as_ref() {
                backfill_raw_similarity(conn, &mut fused, model_key, query_vector)?;
            }
        }
    }
    let debug_chunk: Option<i64> = env::var("RETRIVIO_DEBUG_CHUNK")
        .ok()
        .and_then(|v| v.parse::<i64>().ok());
    if let Some(id) = debug_chunk {
        eprintln!(
            "trace {}: after fusion {} (query type {:?}, fused rows {})",
            id,
            fused
                .get(&id)
                .map(|r| format!(
                    "present sem={:.3} lex={:.3} raw={:?} rel={}",
                    r.semantic, r.lexical, r.raw_similarity, r.relation
                ))
                .unwrap_or_else(|| "absent".to_string()),
            query_type,
            fused.len()
        );
    }
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;
    if let Some(id) = debug_chunk {
        eprintln!(
            "trace {}: after graph {}",
            id,
            fused
                .get(&id)
                .map(|r| format!("present graph={:.3} rel={}", r.graph, r.relation))
                .unwrap_or_else(|| "absent".to_string())
        );
    }

    let frecency = frecency_scores(conn)?;
    let project_paths = list_project_paths(conn)?;
    let project_path_keywords = path_keyword_scores(&project_paths, q);
    let mut summary_judge = SummaryPageJudge::new(conn, &project_paths);
    let role_hint = roles::role_hint(q);
    let show_superseded = opts.include_superseded || roles::history_query(q);
    // The absolute floor is on the raw cosine (an honest number), never on the min-max
    // normalised semantic score, whose top is always 1.0. It fails closed: in semantic mode a
    // candidate without a finite cosine (no stored vector, a malformed blob) is dropped.
    // Lexical-only retrieval has no cosines and no floor.
    let raw_floor = if opts.lexical_only {
        0.0
    } else {
        opts.min_raw_similarity
    };
    let mut by_file: HashMap<String, RankedFileResult> = HashMap::new();
    for row in fused.values() {
        if !passes_raw_floor(row.raw_similarity, raw_floor) {
            continue;
        }
        let content = chunk_base_score(row, cfg);
        let project_sem = *project_semantic.get(&row.project_path).unwrap_or(&0.0);
        let fr = *frecency.get(&row.project_path).unwrap_or(&0.0);
        let doc_kw = doc_keyword_score(&row.doc_rel_path, q);
        let project_kw = *project_path_keywords.get(&row.project_path).unwrap_or(&0.0);
        let kw = doc_kw.max(0.72 * project_kw);
        let mut score = match query_type {
            QueryType::Symbol => {
                (0.34 * content)
                    + (0.10 * project_sem)
                    + (0.24 * kw)
                    + (0.12 * row.lexical)
                    + (0.08 * fr)
                    + (0.12 * row.graph)
            }
            QueryType::NaturalLanguage => {
                (0.56 * content)
                    + (0.16 * project_sem)
                    + (0.08 * fr)
                    + (0.08 * kw)
                    + (0.12 * row.graph)
            }
            QueryType::CodePattern => {
                (0.48 * content)
                    + (0.12 * project_sem)
                    + (0.08 * fr)
                    + (0.14 * kw)
                    + (0.18 * row.graph)
            }
            QueryType::PathQuery => {
                (0.22 * content)
                    + (0.06 * project_sem)
                    + (0.42 * kw)
                    + (0.12 * row.lexical)
                    + (0.08 * fr)
                    + (0.10 * row.graph)
            }
        };
        if (query_type == QueryType::Symbol || query_type == QueryType::PathQuery)
            && kw < 0.20
            && row.lexical < 0.05
        {
            score *= 0.40;
        }
        let path_penalty = path_noise_penalty(&row.doc_rel_path);
        score *= path_penalty;
        // Freshness (spec §4): blend once per candidate; the file keeps its best chunk.
        let fresh = fx.info_for(row);
        if !fx.within_since(fresh.content_date, opts.since_days) {
            continue;
        }
        // Role nudge: a prompt about a call or transcript lifts records a little; one about
        // current status lifts state. Small on purpose; relevance still decides.
        let role_boost = role_hint == Some(fresh.role);
        if role_boost {
            score *= ROLE_HINT_BOOST;
        }
        let mut why = why_string(row, kw, role_boost, &fresh, path_penalty);
        if summary_judge.is_summary_page(conn, &row.doc_path, &row.doc_rel_path, &row.project_path)
        {
            score *= summary_page_factor();
            why_append(&mut why, "summary-page");
        }
        let base_score = score;
        score = fx.blend(score, &fresh);
        let candidate = RankedFileResult {
            path: row.doc_path.clone(),
            project_path: row.project_path.clone(),
            doc_rel_path: row.doc_rel_path.clone(),
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            score,
            base_score,
            semantic: row.semantic,
            lexical: row.lexical,
            graph: row.graph,
            relation: row.relation.clone(),
            quality: row.quality,
            excerpt: row.excerpt.clone(),
            evidence: Vec::new(),
            doc_mtime: fresh.doc_mtime,
            content_date: fresh.content_date,
            date_source: fresh.date_source,
            age_days: fresh.age_days,
            freshness_tier: fresh.tier.to_string(),
            is_record: fresh.is_record,
            role: fresh.role.as_str(),
            verify: fresh.verify,
            noise: row.noise,
            raw_similarity: row.raw_similarity,
            superseded_by: None,
            why,
        };
        // The file keeps its best chunk; an exact tie goes to the earlier chunk so the
        // representative (and its cosine) does not depend on hash-map order.
        let prev = by_file.get(&row.doc_path);
        let replace = match prev {
            None => true,
            Some(p) => {
                candidate.score > p.score
                    || (candidate.score == p.score && candidate.chunk_index < p.chunk_index)
            }
        };
        if debug_chunk == Some(row.chunk_id) {
            eprintln!(
                "trace {}: scored {:.4} (base {:.4}, kw {:.3}, content {:.4}) prev {:?} -> {}",
                row.chunk_id,
                candidate.score,
                candidate.base_score,
                kw,
                content,
                prev.map(|p| (p.chunk_id, p.score)),
                if replace { "inserted" } else { "kept previous" }
            );
        }
        if replace {
            by_file.insert(row.doc_path.clone(), candidate);
        }
    }
    if let Some(id) = debug_chunk {
        let holder = by_file.values().find(|r| r.chunk_id == id);
        eprintln!(
            "trace {}: in by_file before collapse: {}",
            id,
            holder
                .map(|r| format!("yes score={:.4}", r.score))
                .unwrap_or_else(|| "no".to_string())
        );
    }
    collapse_duplicate_files(conn, &mut by_file);
    mark_superseded(&mut by_file, show_superseded, fx.now);
    for (doc_path, item) in &mut by_file {
        let mut support: Vec<EvidenceHit> = Vec::new();
        for row in fused.values() {
            let include = row.doc_path == *doc_path
                || (row.project_path == item.project_path
                    && row.doc_path != *doc_path
                    && row.graph >= 0.55)
                || (row.project_path != item.project_path && row.graph >= 0.72);
            if !include {
                continue;
            }
            support.push(evidence_hit_from_chunk(
                row,
                chunk_base_score(row, cfg),
                &fx,
            ));
        }
        support.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.graph.total_cmp(&a.graph))
                .then_with(|| b.chunk_id.cmp(&a.chunk_id))
        });
        support.truncate(4);
        item.evidence = support;
    }

    let mut out: Vec<RankedFileResult> = by_file.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    out.truncate(limit.max(1));
    Ok(out)
}

/// `content_hash` from the file manifest for each of `paths` (absolute), where known. This is
/// the identity of a file's bytes (the whole file, not a chunk) that duplicate collapse uses
/// in search and recall alike.
pub(crate) fn file_content_hashes(conn: &Connection, paths: &[String]) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    for batch in paths.chunks(300) {
        let placeholders = std::iter::repeat_n("?", batch.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT abs_path, content_hash FROM project_files WHERE abs_path IN ({})",
            placeholders
        );
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return out;
        };
        let rows = stmt.query_map(params_from_iter(batch.iter()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        });
        let Ok(rows) = rows else {
            return out;
        };
        for row in rows.flatten() {
            out.insert(row.0, row.1);
        }
    }
    out
}

/// Two files with the same manifest `content_hash` (identical bytes) are copies of one file
/// when they also share their file name, or when one of them sits under a copy directory
/// (`snapshot`, `snapshots`, `memory-snapshot`, `backup`, `backups`, `archive`, `archived`,
/// `copy`). Two byte-identical documents that meet neither rule (a template pasted into two
/// projects under different names) stay two results.
pub(crate) fn same_file_copy(a_rel: &str, b_rel: &str) -> bool {
    roles::file_name(a_rel).eq_ignore_ascii_case(roles::file_name(b_rel))
        || roles::under_noise_dir(a_rel)
        || roles::under_noise_dir(b_rel)
}

/// Copies of one file (see [`same_file_copy`]) collapse to one result: the copy that is not
/// under a copy directory, then the shorter path, then the higher score, then the path. The
/// survivor keeps its own score, cosine and evidence; it never inherits the removed copy's
/// numbers. Two copies of a memory file in a snapshot folder and its source are one result.
pub(crate) fn collapse_duplicate_files(
    conn: &Connection,
    by_file: &mut HashMap<String, RankedFileResult>,
) {
    if by_file.len() < 2 {
        return;
    }
    let paths: Vec<String> = by_file.keys().cloned().collect();
    let hashes = file_content_hashes(conn, &paths);
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for (path, hash) in hashes {
        if !hash.is_empty() && by_file.contains_key(&path) {
            groups.entry(hash).or_default().push(path);
        }
    }
    for members in groups.into_values() {
        if members.len() < 2 {
            continue;
        }
        let mut ordered = members.clone();
        ordered.sort_by(|a, b| {
            let (ra, rb) = (&by_file[a], &by_file[b]);
            roles::under_noise_dir(&ra.doc_rel_path)
                .cmp(&roles::under_noise_dir(&rb.doc_rel_path))
                .then_with(|| a.len().cmp(&b.len()))
                .then_with(|| rb.score.total_cmp(&ra.score))
                .then_with(|| a.cmp(b))
        });
        let mut survivors: Vec<String> = Vec::new();
        for m in ordered {
            let is_copy = survivors
                .iter()
                .any(|s| same_file_copy(&by_file[s].doc_rel_path, &by_file[&m].doc_rel_path));
            if is_copy {
                by_file.remove(&m);
            } else {
                survivors.push(m);
            }
        }
    }
}

/// Supersession, soft and state-only: among `state` files of one series (same project,
/// parent directory and normalised stem) the newest by revision date (the date in the
/// relative path, else the last edit: see [`freshness::revision_date`]) is the head; the
/// others get `superseded_by = head` and, unless the query asks for history, a downrank.
/// Nothing is removed; records and knowledge are never superseded. Ties on the revision date
/// go to the higher score, then to the lexicographically later path.
pub(crate) fn mark_superseded(
    by_file: &mut HashMap<String, RankedFileResult>,
    full_strength: bool,
    now: f64,
) {
    let mut series: HashMap<String, Vec<String>> = HashMap::new();
    for (path, item) in by_file.iter() {
        if item.role == Role::State.as_str() {
            series
                .entry(roles::series_key(&item.project_path, &item.doc_rel_path))
                .or_default()
                .push(path.clone());
        }
    }
    for members in series.into_values() {
        if members.len() < 2 {
            continue;
        }
        let newest = members
            .iter()
            .max_by(|a, b| {
                let (ia, ib) = (&by_file[*a], &by_file[*b]);
                let ra = freshness::revision_date(&ia.doc_rel_path, ia.doc_mtime, now);
                let rb = freshness::revision_date(&ib.doc_rel_path, ib.doc_mtime, now);
                ra.total_cmp(&rb)
                    .then_with(|| ia.score.total_cmp(&ib.score))
                    .then_with(|| a.cmp(b))
            })
            .cloned()
            .expect("non-empty series");
        for m in members {
            if m == newest {
                continue;
            }
            if let Some(item) = by_file.get_mut(&m) {
                item.superseded_by = Some(newest.clone());
                why_append(&mut item.why, "superseded");
                if !full_strength {
                    item.score *= SUPERSEDED_FACTOR;
                    item.base_score *= SUPERSEDED_FACTOR;
                }
            }
        }
    }
}

/// Generate a hypothetical code snippet that would answer the query (HyDE technique).
/// Returns the generated text, or None on failure.
pub(crate) fn generate_hyde_snippet(cfg: &ConfigValues, query: &str) -> Option<String> {
    let host = ollama_host();
    let url = format!("{}/api/generate", host);
    let model = &cfg.reranker_model; // reuse the same small model

    match ollama_is_reachable() {
        Ok(true) => {}
        _ => return None,
    }

    let prompt = format!(
        "Write a short code snippet (10-20 lines) that implements or relates to: {}\n\
         Output ONLY the code, no explanations.",
        query
    );
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.3,
            "num_predict": 256
        }
    });
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(cfg.reranker_timeout_ms.max(2000)))
        .build();
    let resp = agent.post(&url).send_json(&body).ok()?;
    let json: serde_json::Value = resp.into_json().ok()?;
    let text = json["response"].as_str()?.trim().to_string();
    if text.len() < 10 {
        return None;
    }
    Some(text)
}

/// Ask Ollama to score each candidate chunk's relevance to the query on a 0-10 scale.
/// Returns a map of chunk_id -> normalized relevance score (0.0-1.0).
/// On failure (model unavailable, timeout, etc.), returns an empty map so the caller
/// can gracefully skip re-ranking.
pub(crate) fn rerank_with_ollama(
    cfg: &ConfigValues,
    query: &str,
    candidates: &[(i64, String)], // (chunk_id, text excerpt)
) -> HashMap<i64, f64> {
    let host = ollama_host();
    let url = format!("{}/api/generate", host);
    let timeout = Duration::from_millis(cfg.reranker_timeout_ms);
    let model = &cfg.reranker_model;
    let batch_size = cfg.reranker_batch_size.max(1);

    // Pre-check: is Ollama reachable at all?
    match ollama_is_reachable() {
        Ok(true) => {}
        _ => return HashMap::new(),
    }

    let results: std::sync::Mutex<HashMap<i64, f64>> = std::sync::Mutex::new(HashMap::new());

    // Process candidates in parallel batches using scoped threads
    std::thread::scope(|s| {
        for batch in candidates.chunks(batch_size) {
            let batch_handles: Vec<_> = batch
                .iter()
                .map(|(chunk_id, text)| {
                    let url = &url;
                    let results = &results;
                    let cid = *chunk_id;
                    // Truncate text to ~800 chars to keep prompt short
                    let excerpt: String = text.chars().take(800).collect();
                    s.spawn(move || {
                        let prompt = format!(
                            "Rate the relevance of this code snippet to the search query on a scale of 0-10.\n\
                             Query: {}\n\
                             Code:\n{}\n\
                             Output ONLY a single integer 0-10.",
                            query, excerpt
                        );
                        let body = serde_json::json!({
                            "model": model,
                            "prompt": prompt,
                            "stream": false,
                            "options": {
                                "temperature": 0.0,
                                "num_predict": 8
                            }
                        });
                        let agent = ureq::AgentBuilder::new()
                            .timeout(timeout)
                            .build();
                        let resp = match agent.post(url).send_json(&body) {
                            Ok(r) => r,
                            Err(_) => return,
                        };
                        let json: serde_json::Value = match resp.into_json() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        let response_text = json["response"].as_str().unwrap_or("").trim();
                        // Parse the first integer found in the response
                        let score: Option<u32> = response_text
                            .chars()
                            .filter(|c| c.is_ascii_digit())
                            .take(2)
                            .collect::<String>()
                            .parse()
                            .ok();
                        if let Some(s) = score {
                            let normalized = (s.min(10) as f64) / 10.0;
                            if let Ok(mut map) = results.lock() {
                                map.insert(cid, normalized);
                            }
                        }
                    })
                })
                .collect();

            // Wait for current batch to finish before starting next
            for handle in batch_handles {
                let _ = handle.join();
            }
        }
    });

    results.into_inner().unwrap_or_default()
}

/// Chunk ranking with an optional `since_days` hard filter on content date. The recency blend
/// (spec §4) is applied exactly once, after the cross-encoder reranker blend. HyDE and the
/// reranker run as configured.
pub(crate) fn rank_chunks_native_with(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
    since_days: Option<f64>,
) -> Result<Vec<RankedChunkResult>, String> {
    rank_chunks_native_opts(
        conn,
        cfg,
        query,
        limit,
        since_days,
        cfg.hyde_enabled,
        cfg.reranker_enabled,
    )
}

/// [`rank_chunks_native_with`] with HyDE and the reranker switchable per call: the related
/// pass of `get_related_chunks` runs with both off (its query is a chunk's own text, and it
/// must stay cheap), the query pass keeps the configured behaviour.
pub(crate) fn rank_chunks_native_opts(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
    since_days: Option<f64>,
    use_hyde: bool,
    use_reranker: bool,
) -> Result<Vec<RankedChunkResult>, String> {
    ensure_reembed_ready(conn, cfg, "search")?;
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let query_type = QueryType::classify(q);
    let (sem_limit, lex_limit) = query_type.retrieval_limits(
        cfg.vector_candidates.max(1) as usize,
        cfg.lexical_candidates.max(1) as usize,
    );
    let (model_key, query_vector) = embed_query_cached(cfg, q)?;

    // Tiered search: for large codebases (200+ projects), restrict
    // project-level scoring to top-K to reduce downstream work.
    let total_project_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
        .unwrap_or(0);
    let use_tiered = total_project_count >= 200;
    let project_keep_top = if use_tiered {
        30
    } else {
        std::cmp::max(120, sem_limit * 2)
    };
    let project_semantic =
        project_semantic_scores(conn, &model_key, &query_vector, project_keep_top)?;
    let mut fused = hybrid_search_lance(
        conn,
        &model_key,
        q,
        &query_vector,
        std::cmp::max(220, sem_limit * 5),
        std::cmp::max(220, lex_limit * 3),
    )?;
    if fused.is_empty() {
        return Ok(Vec::new());
    }

    // Tiered search: for large codebases (200+ projects), filter chunks
    // to only include those from the top-K most relevant projects.
    // This reduces scoring/expansion/re-ranking work dramatically.
    // Chunks whose text carries every query term, or a capitalised name from the query,
    // survive the cut wherever their project ranks: an exact name such as "Acme" must not
    // vanish because its project is the 31st most similar overall.
    if use_tiered {
        let top_projects: HashSet<&String> = project_semantic.keys().collect();
        let before = fused.len();
        let mut kept_by_lexical = 0usize;
        fused.retain(|_id, row| {
            if top_projects.contains(&row.project_path) {
                return true;
            }
            if row.strong_lexical {
                kept_by_lexical += 1;
                return true;
            }
            false
        });
        if env::var("RETRIVIO_DEBUG_TIERED")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            eprintln!(
                "tiered: {} candidates, {} kept ({} of them outside the top {} projects through lexical coverage)",
                before,
                fused.len(),
                kept_by_lexical,
                project_keep_top
            );
        }
        if fused.is_empty() {
            return Ok(Vec::new());
        }
    }

    // HyDE: for natural language queries, generate a hypothetical code snippet,
    // embed it, run a second vector search, and merge new results into fused.
    if use_hyde && query_type == QueryType::NaturalLanguage {
        if let Some(hyde_text) = generate_hyde_snippet(cfg, q) {
            if let Ok((_hyde_model, hyde_vector)) = embed_query_cached(cfg, &hyde_text) {
                // Run vector-only search with the hypothetical embedding
                if let Ok(hyde_scores) = with_lance_store(|store| {
                    lance_store::search_vectors(store, &hyde_vector, sem_limit)
                }) {
                    // Merge HyDE results: add new chunks or boost existing ones. The cosine
                    // the search reports for a HyDE-only chunk is to the hypothetical text,
                    // not to the query, so every HyDE-only hit gets its `raw_similarity`
                    // recomputed against the real query vector (or `None` when no vector is
                    // stored, which then fails the floor).
                    let hyde_signals = chunk_signals_for_ids(
                        conn,
                        &hyde_scores,
                        &HashMap::new(),
                        &CoverageTerms::from_query(q),
                        None,
                    );
                    if let Ok(mut new_signals) = hyde_signals {
                        let hyde_only: Vec<i64> = new_signals
                            .keys()
                            .copied()
                            .filter(|id| !fused.contains_key(id))
                            .collect();
                        recompute_raw_similarity(
                            conn,
                            &mut new_signals,
                            &hyde_only,
                            &model_key,
                            &query_vector,
                        )?;
                        for (chunk_id, mut signal) in new_signals {
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                fused.entry(chunk_id)
                            {
                                // Scale down HyDE-only results to avoid dominating
                                signal.semantic *= 0.7;
                                e.insert(signal);
                            } else if let Some(existing) = fused.get_mut(&chunk_id) {
                                // Boost existing: chunk found by both original and HyDE
                                existing.semantic =
                                    (existing.semantic + signal.semantic * 0.3).min(1.0);
                            }
                        }
                    }
                }
            }
        }
    }

    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;

    let frecency = frecency_scores(conn)?;
    let fx = FreshnessCtx::new(cfg);
    let mut summary_judge = SummaryPageJudge::new(conn, &list_project_paths(conn)?);
    // The same raw-cosine floor as the files view (`search_min_abs_score`, 0 = off), failing
    // closed on a missing or NaN cosine.
    let raw_floor = cfg.search_min_abs_score;
    let mut out: Vec<RankedChunkResult> = Vec::new();
    for row in fused.values() {
        if !passes_raw_floor(row.raw_similarity, raw_floor) {
            continue;
        }
        let fresh = fx.info_for(row);
        if !fx.within_since(fresh.content_date, since_days) {
            continue;
        }
        let content = chunk_base_score(row, cfg);
        let project_sem = *project_semantic.get(&row.project_path).unwrap_or(&0.0);
        let fr = *frecency.get(&row.project_path).unwrap_or(&0.0);
        let kw = doc_keyword_score(&row.doc_rel_path, q);
        // Chunk-level scoring: blend content quality, project semantic, frecency,
        // keyword match, and graph signals. Weights adjust based on query type.
        let mut score = match query_type {
            QueryType::Symbol => {
                (0.38 * content)
                    + (0.10 * project_sem)
                    + (0.08 * fr)
                    + (0.32 * kw)
                    + (0.12 * row.graph)
            }
            QueryType::NaturalLanguage => {
                (0.58 * content)
                    + (0.18 * project_sem)
                    + (0.08 * fr)
                    + (0.06 * kw)
                    + (0.10 * row.graph)
            }
            QueryType::CodePattern => {
                (0.50 * content)
                    + (0.14 * project_sem)
                    + (0.08 * fr)
                    + (0.14 * kw)
                    + (0.14 * row.graph)
            }
            QueryType::PathQuery => {
                (0.28 * content)
                    + (0.08 * project_sem)
                    + (0.08 * fr)
                    + (0.46 * kw)
                    + (0.10 * row.graph)
            }
        };
        // Same path hygiene as file ranking (scratch, state and copy directories).
        let path_penalty = path_noise_penalty(&row.doc_rel_path);
        score *= path_penalty;
        let mut why = why_string(row, kw, false, &fresh, path_penalty);
        if summary_judge.is_summary_page(conn, &row.doc_path, &row.doc_rel_path, &row.project_path)
        {
            score *= summary_page_factor();
            why_append(&mut why, "summary-page");
        }
        out.push(RankedChunkResult {
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            path: row.doc_path.clone(),
            project_path: row.project_path.clone(),
            doc_rel_path: row.doc_rel_path.clone(),
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            graph: row.graph,
            relation: row.relation.clone(),
            quality: row.quality,
            excerpt: row.excerpt.clone(),
            doc_mtime: fresh.doc_mtime,
            content_date: fresh.content_date,
            date_source: fresh.date_source,
            age_days: fresh.age_days,
            freshness_tier: fresh.tier.to_string(),
            is_record: fresh.is_record,
            role: fresh.role.as_str(),
            verify: fresh.verify,
            noise: row.noise,
            raw_similarity: row.raw_similarity,
            superseded_by: None,
            why,
        });
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    // Cross-encoder re-ranking: take top pool_size candidates, score with LLM,
    // blend re-rank score with original score, then re-sort.
    if use_reranker && !out.is_empty() {
        let pool = out.len().min(cfg.reranker_pool_size);
        let candidates: Vec<(i64, String)> = out[..pool]
            .iter()
            .map(|r| (r.chunk_id, r.excerpt.clone()))
            .collect();
        let rerank_scores = rerank_with_ollama(cfg, q, &candidates);
        if !rerank_scores.is_empty() {
            // Blend: 70% reranker, 30% original (preserves original signal as tiebreaker)
            for item in out[..pool].iter_mut() {
                if let Some(&rs) = rerank_scores.get(&item.chunk_id) {
                    item.score = 0.70 * rs + 0.30 * item.score;
                }
            }
            out[..pool].sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.chunk_id.cmp(&b.chunk_id))
            });
        }
    }

    // Freshness (spec §4): blend recency exactly once, after the reranker, then re-sort.
    for item in out.iter_mut() {
        let recency = fx.recency_for(item.age_days, item.is_record);
        item.score = freshness::blend(item.score, recency, fx.weight(item.is_record));
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    out.truncate(limit.max(1));
    mark_superseded_chunks(&mut out, fx.now);
    Ok(out)
}

/// Supersession as a label on chunk results (slice 4): among the distinct `state` files of
/// one series (project, parent directory, normalised stem) present in the result set, the
/// newest by revision date (the date in the relative path, else the last edit: the same rule
/// as [`mark_superseded`]) is the head; chunks of the other files carry `superseded_by = head`.
/// Chunk search never downranks or collapses on it (an older handoff's paragraph may be the
/// exact answer); the label lets the agent prefer the head.
pub(crate) fn mark_superseded_chunks(out: &mut [RankedChunkResult], now: f64) {
    // Per `state` file in the result set: series key, revision date, best chunk score.
    let mut files: HashMap<String, (String, f64, f64)> = HashMap::new();
    for item in out.iter() {
        if item.role != Role::State.as_str() {
            continue;
        }
        let e = files.entry(item.path.clone()).or_insert_with(|| {
            (
                roles::series_key(&item.project_path, &item.doc_rel_path),
                freshness::revision_date(&item.doc_rel_path, item.doc_mtime, now),
                f64::NEG_INFINITY,
            )
        });
        if item.score > e.2 {
            e.2 = item.score;
        }
    }
    // Head per series: the newest revision, then the higher score, then the lexicographically
    // later path, the same order as `mark_superseded`.
    let mut heads: HashMap<String, (String, f64, f64)> = HashMap::new();
    for (path, (key, revision, score)) in files.iter() {
        let e = heads
            .entry(key.clone())
            .or_insert_with(|| (path.clone(), *revision, *score));
        let better = revision
            .total_cmp(&e.1)
            .then_with(|| score.total_cmp(&e.2))
            .then_with(|| path.cmp(&e.0))
            == std::cmp::Ordering::Greater;
        if better {
            *e = (path.clone(), *revision, *score);
        }
    }
    if heads.is_empty() {
        return;
    }
    for item in out.iter_mut() {
        if item.role != Role::State.as_str() {
            continue;
        }
        let key = roles::series_key(&item.project_path, &item.doc_rel_path);
        if let Some((head, _, _)) = heads.get(&key) {
            if *head != item.path {
                item.superseded_by = Some(head.clone());
                why_append(&mut item.why, "superseded");
            }
        }
    }
}

pub(crate) fn project_semantic_scores(
    conn: &Connection,
    model: &str,
    query_vector: &[f32],
    keep_top: usize,
) -> Result<HashMap<String, f64>, String> {
    let qnorm = vector_norm(query_vector);
    if qnorm == 0.0 {
        return Ok(HashMap::new());
    }
    let mut stmt = conn
        .prepare(
            r#"
SELECT p.path, pv.norm, pv.vector
FROM projects p
JOIN project_vectors pv ON pv.project_id = p.id
WHERE pv.model = ?1
"#,
        )
        .map_err(|e| format!("failed preparing project vector query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let path: String = row.get(0)?;
            let norm: f64 = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            Ok((path, norm, blob))
        })
        .map_err(|e| format!("failed querying project vectors: {}", e))?;

    let mut scores: Vec<(f64, String)> = Vec::new();
    for row in rows {
        let (path, vnorm, blob) =
            row.map_err(|e| format!("failed reading project vector row: {}", e))?;
        if vnorm == 0.0 {
            continue;
        }
        let vec = blob_to_f32_vec(&blob);
        let sim = ((cosine_raw(query_vector, &vec, qnorm, vnorm) + 1.0) / 2.0).clamp(0.0, 1.0);
        scores.push((sim, path));
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    if scores.is_empty() {
        return Ok(HashMap::new());
    }
    scores.truncate(keep_top.max(1));
    let lo = scores.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
    let hi = scores.iter().map(|v| v.0).fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < f64::EPSILON {
        let mut out = HashMap::new();
        for (_, path) in scores {
            out.insert(path, 1.0);
        }
        return Ok(out);
    }
    let span = hi - lo;
    let mut out = HashMap::new();
    for (score, path) in scores {
        out.insert(path, ((score - lo) / span).clamp(0.0, 1.0));
    }
    Ok(out)
}

pub(crate) fn search_lexical_chunks_sqlite(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(HashMap::new());
    }
    lexical_chunk_signals_for_match(conn, &fts, limit, &CoverageTerms::from_query(query))
}

/// FTS5 MATCH expression that ORs quoted terms (double quotes escaped by doubling).
pub(crate) fn fts_or_query(terms: &[String]) -> String {
    terms
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Lexical-only file candidates for `retrivio recall` when embeddings are unavailable.
///
/// Runs one `chunk_fts` query ORing the quoted `terms`, keeps the best chunk per file, fills
/// the freshness fields and scores each file by bm25 normalised to 0..1 (1 = best match),
/// blended once with recency exactly like [`rank_files_native_with`]. Errors yield an empty list.
pub(crate) fn lexical_file_candidates(
    conn: &Connection,
    cfg: &ConfigValues,
    terms: &[String],
    limit: usize,
) -> Vec<RankedFileResult> {
    let fts = fts_or_query(terms);
    if fts.is_empty() {
        return Vec::new();
    }
    let chunk_limit = (limit.max(1) * 8).clamp(40, 600);
    let Ok(signals) =
        lexical_chunk_signals_for_match(conn, &fts, chunk_limit, &CoverageTerms::from_terms(terms))
    else {
        return Vec::new();
    };
    let fx = FreshnessCtx::new(cfg);
    let mut by_file: HashMap<String, RankedFileResult> = HashMap::new();
    for row in signals.values() {
        let fresh = fx.info_for(row);
        let score = fx.blend(row.lexical, &fresh);
        let replace = by_file
            .get(&row.doc_path)
            .map(|prev| score > prev.score)
            .unwrap_or(true);
        if !replace {
            continue;
        }
        by_file.insert(
            row.doc_path.clone(),
            RankedFileResult {
                path: row.doc_path.clone(),
                project_path: row.project_path.clone(),
                doc_rel_path: row.doc_rel_path.clone(),
                chunk_id: row.chunk_id,
                chunk_index: row.chunk_index,
                score,
                base_score: row.lexical,
                semantic: 0.0,
                lexical: row.lexical,
                graph: 0.0,
                relation: "lexical".to_string(),
                quality: row.quality,
                excerpt: row.excerpt.clone(),
                evidence: Vec::new(),
                doc_mtime: fresh.doc_mtime,
                content_date: fresh.content_date,
                date_source: fresh.date_source,
                age_days: fresh.age_days,
                freshness_tier: fresh.tier.to_string(),
                is_record: fresh.is_record,
                role: fresh.role.as_str(),
                verify: fresh.verify,
                noise: row.noise,
                raw_similarity: None,
                superseded_by: None,
                why: format!(
                    "lexical:{:.2}{}",
                    row.lexical,
                    if row.noise { "+noise" } else { "" }
                ),
            },
        );
    }
    let mut out: Vec<RankedFileResult> = by_file.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    out.truncate(limit.max(1));
    out
}

/// (chunk id, project path, doc path, doc rel path, chunk index, text, lexical score, doc mtime).
pub(crate) type RawChunkRow = (i64, String, String, String, i64, String, f64, f64);

pub(crate) fn lexical_chunk_signals_for_match(
    conn: &Connection,
    fts: &str,
    limit: usize,
    cover: &CoverageTerms,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    bm25(chunk_fts) AS lexical_bm25,
    pc.doc_mtime
FROM chunk_fts
JOIN project_chunks pc ON pc.id = chunk_fts.rowid
JOIN projects p ON p.id = pc.project_id
WHERE chunk_fts MATCH ?1
ORDER BY lexical_bm25
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing lexical chunk query: {}", e))?;
    let rows = stmt
        .query_map(params![fts, limit as i64], |row| {
            let chunk_id: i64 = row.get(0)?;
            let project_path: String = row.get(1)?;
            let doc_path: String = row.get(2)?;
            let doc_rel_path: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            let bm25: f64 = row.get(6)?;
            let doc_mtime: f64 = row.get(7)?;
            Ok((
                chunk_id,
                project_path,
                doc_path,
                doc_rel_path,
                chunk_index,
                text,
                bm25,
                doc_mtime,
            ))
        })
        .map_err(|e| format!("failed querying lexical chunks: {}", e))?;

    let mut raw: Vec<RawChunkRow> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading lexical chunk row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let lo = raw.iter().map(|r| r.6).fold(f64::INFINITY, f64::min);
    let hi = raw.iter().map(|r| r.6).fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    for (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, bm25, doc_mtime) in raw
    {
        let lexical = if span.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ((bm25 - lo) / span)
        };
        let facts = text_facts(&doc_rel_path, &text, cover);
        out.insert(
            chunk_id,
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path,
                doc_path,
                doc_rel_path,
                doc_mtime,
                semantic: 0.0,
                lexical: lexical.clamp(0.0, 1.0),
                graph: 0.0,
                relation: "direct".to_string(),
                quality: facts.quality,
                excerpt: clip_text(&text, 190),
                raw_similarity: None,
                shape: facts.shape,
                noise: facts.noise,
                strong_lexical: facts.strong_lexical,
            },
        );
    }
    apply_file_shapes(conn, &mut out)?;
    Ok(out)
}

pub(crate) type PathChunkRow = (i64, String, String, String, i64, String, f64);

pub(crate) fn map_path_chunk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PathChunkRow> {
    Ok((
        row.get::<_, i64>(0)?,
        row.get::<_, String>(1)?,
        row.get::<_, String>(2)?,
        row.get::<_, String>(3)?,
        row.get::<_, i64>(4)?,
        row.get::<_, String>(5)?,
        row.get::<_, f64>(6)?,
    ))
}

/// Path-keyword signal for Symbol and PathQuery searches: chunks whose file path or project
/// path contains the query's path words.
///
/// Bounded on purpose. Only words that look like paths (`query_path_like_tokens`) drive the
/// scans; a Symbol-shaped query (at most two words, none a path) is itself the pattern, and
/// prose never scans the table. Matching files are found through the covering index on
/// `doc_path` (no row reads), capped at `MAX_PATHS_PER_TOKEN` files per word, and the chunk
/// rows are then fetched by exact path; project-path matches are resolved in memory.
pub(crate) fn keyword_path_chunk_scores(
    conn: &Connection,
    query: &str,
    keep_top: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    const MAX_PATHS_PER_TOKEN: usize = 400;
    let path_tokens = query_path_like_tokens(query);
    let word_count = query.split_whitespace().count();
    let source: String = if !path_tokens.is_empty() {
        path_tokens.join(" ")
    } else if word_count <= 2 {
        query.to_string()
    } else {
        return Ok(HashMap::new());
    };
    let q_tokens: Vec<String> = all_word_tokens(&source)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return Ok(HashMap::new());
    }
    let cover = CoverageTerms::from_query(query);
    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    let per_token_limit = (keep_top.max(1) * 3).clamp(50, 2500);
    // File matches take at most three quarters of a word's budget so project-path matches
    // still get a share.
    let doc_budget = (per_token_limit * 3 / 4).max(1);

    // Files whose path contains the word: covering scan of idx_project_chunks_doc.
    // doc_rel_path is a suffix of doc_path, so one pattern covers both columns.
    let mut doc_stmt = conn
        .prepare(
            "SELECT DISTINCT doc_path FROM project_chunks WHERE doc_path LIKE ?1 ORDER BY doc_path LIMIT ?2",
        )
        .map_err(|e| format!("failed preparing keyword path doc query: {}", e))?;
    let mut by_doc_stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    pc.doc_mtime
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.doc_path = ?1
"#,
        )
        .map_err(|e| format!("failed preparing keyword path chunk query: {}", e))?;
    let mut by_project_stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    pc.doc_mtime
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.project_id = ?1
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing keyword project chunk query: {}", e))?;
    let projects: Vec<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, lower(path) FROM projects")
            .map_err(|e| format!("failed preparing project path list: {}", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("failed listing project paths: {}", e))?;
        let mut v = Vec::new();
        for row in rows {
            v.push(row.map_err(|e| format!("failed reading project path row: {}", e))?);
        }
        v
    };

    let q_n = q_tokens.len() as f64;
    let mut absorb = |row: PathChunkRow| {
        let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, doc_mtime) = row;
        let doc_rel_lower = doc_rel_path.to_lowercase();
        let project_lower = project_path.to_lowercase();
        let doc_hits = q_tokens
            .iter()
            .filter(|tok| doc_rel_lower.contains(tok.as_str()))
            .count() as f64;
        let proj_hits = q_tokens
            .iter()
            .filter(|tok| project_lower.contains(tok.as_str()))
            .count() as f64;
        let lexical = ((doc_hits / q_n).max(0.65 * (proj_hits / q_n))).clamp(0.0, 1.0);
        if lexical <= 0.0 {
            return;
        }
        let entry = out.entry(chunk_id).or_insert_with(|| {
            let facts = text_facts(&doc_rel_path, &text, &cover);
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path: project_path.clone(),
                doc_path: doc_path.clone(),
                doc_rel_path: doc_rel_path.clone(),
                doc_mtime,
                semantic: 0.0,
                lexical,
                graph: 0.0,
                relation: "path_keyword".to_string(),
                quality: facts.quality,
                excerpt: clip_text(&text, 190),
                raw_similarity: None,
                shape: facts.shape,
                noise: facts.noise,
                strong_lexical: facts.strong_lexical,
            }
        });
        if lexical > entry.lexical {
            entry.lexical = lexical;
            entry.relation = "path_keyword".to_string();
        }
    };

    for token in &q_tokens {
        let pattern = format!("%{}%", token);
        let mut rows_for_token = 0usize;
        let doc_paths: Vec<String> = {
            let rows = doc_stmt
                .query_map(params![pattern, MAX_PATHS_PER_TOKEN as i64], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(|e| format!("failed querying keyword path docs: {}", e))?;
            let mut v = Vec::new();
            for row in rows {
                v.push(row.map_err(|e| format!("failed reading keyword path doc: {}", e))?);
            }
            v
        };
        for doc_path in &doc_paths {
            if rows_for_token >= doc_budget {
                break;
            }
            let rows = by_doc_stmt
                .query_map(params![doc_path], map_path_chunk_row)
                .map_err(|e| format!("failed querying keyword path chunks: {}", e))?;
            for row in rows {
                absorb(row.map_err(|e| format!("failed reading keyword path chunk row: {}", e))?);
                rows_for_token += 1;
            }
        }
        for (project_id, project_lower) in &projects {
            if rows_for_token >= per_token_limit {
                break;
            }
            if !project_lower.contains(token.as_str()) {
                continue;
            }
            let remaining = (per_token_limit - rows_for_token) as i64;
            let rows = by_project_stmt
                .query_map(params![project_id, remaining], map_path_chunk_row)
                .map_err(|e| format!("failed querying keyword project chunks: {}", e))?;
            for row in rows {
                absorb(
                    row.map_err(|e| format!("failed reading keyword project chunk row: {}", e))?,
                );
                rows_for_token += 1;
            }
        }
    }

    let mut out: HashMap<i64, ChunkSignal> = if out.len() <= keep_top.max(1) {
        out
    } else {
        let mut pairs: Vec<(i64, ChunkSignal)> = out.into_iter().collect();
        pairs.sort_by(|a, b| {
            b.1.lexical
                .total_cmp(&a.1.lexical)
                .then_with(|| a.0.cmp(&b.0))
        });
        pairs.truncate(keep_top.max(1));
        pairs.into_iter().collect()
    };
    // The retained rows take the file-level shape of their `.txt` file, as the FTS and vector
    // candidates do: a path query hits every chunk of the file, and the cut may keep a later
    // chunk whose own text says nothing about the role or the noise flag.
    apply_file_shapes(conn, &mut out)?;
    Ok(out)
}

pub(crate) fn fuse_chunk_signals(
    lexical_chunks: &HashMap<i64, ChunkSignal>,
    semantic_chunks: &HashMap<i64, ChunkSignal>,
) -> HashMap<i64, ChunkSignal> {
    let mut out = semantic_chunks.clone();
    for (chunk_id, lex) in lexical_chunks {
        if let Some(existing) = out.get_mut(chunk_id) {
            existing.lexical = existing.lexical.max(lex.lexical);
            existing.quality = existing.quality.max(lex.quality);
            if existing.excerpt.is_empty() && !lex.excerpt.is_empty() {
                existing.excerpt = lex.excerpt.clone();
            }
        } else {
            out.insert(*chunk_id, lex.clone());
        }
    }
    out
}

pub(crate) fn chunk_base_score(row: &ChunkSignal, cfg: &ConfigValues) -> f64 {
    // Graph is treated as contextual support with hop-penalty baked into row.graph.
    let direct = (cfg.rank_chunk_semantic_weight * row.semantic)
        + (cfg.rank_chunk_lexical_weight * row.lexical)
        + (cfg.rank_chunk_graph_weight * row.graph);
    let quality_mix = cfg.rank_quality_mix.clamp(0.0, 1.0);
    direct * ((1.0 - quality_mix) + (quality_mix * row.quality))
}

pub(crate) fn apply_graph_chunk_expansion(
    conn: &Connection,
    fused: &mut HashMap<i64, ChunkSignal>,
    cfg: &ConfigValues,
) -> Result<(), String> {
    if fused.is_empty() {
        return Ok(());
    }
    let seed_limit = cfg.graph_seed_limit.max(1) as usize;
    let neighbor_limit = cfg.graph_neighbor_limit.max(1) as usize;

    let mut seeds: Vec<(i64, f64, String)> = fused
        .values()
        .map(|row| {
            let seed_score = (0.78 * row.semantic) + (0.22 * row.lexical);
            (row.chunk_id, seed_score, row.project_path.clone())
        })
        .collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    seeds.truncate(seed_limit.max(1));
    if seeds.is_empty() {
        return Ok(());
    }

    let mut seed_projects: HashSet<String> = HashSet::new();
    for (chunk_id, _, project_path) in &seeds {
        seed_projects.insert(project_path.clone());
        if let Some(row) = fused.get_mut(chunk_id) {
            row.graph = row.graph.max(1.0);
            row.relation = "seed".to_string();
        }
    }

    // Multi-hop neighbor discovery: 2 hops deep, decay 0.6 per hop
    let mut neighbor_weights: HashMap<String, f64> = HashMap::new();
    for project_path in &seed_projects {
        let multi_hop =
            multi_hop_project_neighbors(conn, project_path, 2, neighbor_limit.max(1), 0.6)?;
        for (dst, (weight, _hop)) in multi_hop {
            if seed_projects.contains(&dst) {
                continue;
            }
            let hop_penalized = (cfg.graph_related_base + (cfg.graph_related_scale * weight))
                .clamp(0.0, cfg.graph_related_cap.clamp(0.0, 1.0));
            let entry = neighbor_weights.entry(dst).or_insert(0.0);
            if hop_penalized > *entry {
                *entry = hop_penalized;
            }
        }
    }

    for row in fused.values_mut() {
        if row.relation == "seed" {
            continue;
        }
        if seed_projects.contains(&row.project_path) {
            let same_project = if row.semantic >= 0.72 {
                cfg.graph_same_project_high
            } else {
                cfg.graph_same_project_low
            };
            if same_project > row.graph {
                row.graph = same_project;
                row.relation = "same_project".to_string();
            }
            continue;
        }
        if let Some(weight) = neighbor_weights.get(&row.project_path) {
            if *weight > row.graph {
                row.graph = *weight;
                row.relation = "related_project".to_string();
            }
        }
    }

    Ok(())
}

pub(crate) fn evidence_hit_from_chunk(
    row: &ChunkSignal,
    score: f64,
    fx: &FreshnessCtx,
) -> EvidenceHit {
    let fresh = fx.info_for(row);
    EvidenceHit {
        chunk_id: row.chunk_id,
        chunk_index: row.chunk_index,
        doc_path: row.doc_path.clone(),
        doc_rel_path: row.doc_rel_path.clone(),
        score,
        semantic: row.semantic,
        lexical: row.lexical,
        graph: row.graph,
        relation: row.relation.clone(),
        quality: row.quality,
        excerpt: row.excerpt.clone(),
        content_date: fresh.content_date,
        date_source: fresh.date_source,
        age_days: fresh.age_days,
        freshness_tier: fresh.tier.to_string(),
        is_record: fresh.is_record,
        role: fresh.role.as_str(),
        verify: fresh.verify,
        noise: row.noise,
        raw_similarity: row.raw_similarity,
        recency: fresh.recency,
        why: why_string(row, 0.0, false, &fresh, 1.0),
    }
}

/// For `.txt` files the shape that decides the role (and the noise flag) is the *file's*,
/// read once from its first chunk, so every chunk of a long transcript or chat dump is judged
/// alike whichever chunk the query hit. Rows of other file types keep the shape of their own
/// text (only `.txt` roles depend on the shape). The first 4 KB of the file decide: a `.txt`
/// transcript whose "transcript" heading or timecodes start later is not recognised.
pub(crate) fn apply_file_shapes(
    conn: &Connection,
    rows: &mut HashMap<i64, ChunkSignal>,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let mut file_shape: HashMap<String, TextShape> = HashMap::new();
    let mut need: Vec<String> = Vec::new();
    for row in rows.values() {
        if !row.doc_rel_path.to_ascii_lowercase().ends_with(".txt") {
            continue;
        }
        if row.chunk_index == 0 {
            file_shape.insert(row.doc_path.clone(), row.shape);
        } else {
            need.push(row.doc_path.clone());
        }
    }
    need.sort();
    need.dedup();
    need.retain(|p| !file_shape.contains_key(p));
    let looked_up = need.len();
    for batch in need.chunks(300) {
        let placeholders = std::iter::repeat_n("?", batch.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT doc_path, text FROM project_chunks WHERE chunk_index = 0 AND doc_path IN ({})",
            placeholders
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed preparing first-chunk lookup: {}", e))?;
        let found = stmt
            .query_map(params_from_iter(batch.iter()), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("failed querying first chunks: {}", e))?;
        for row in found {
            let (doc_path, text) =
                row.map_err(|e| format!("failed reading first-chunk row: {}", e))?;
            file_shape.insert(doc_path, roles::text_shape(&text));
        }
    }
    if env::var("RETRIVIO_DEBUG_BACKFILL")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!(
            "file-shapes: {} rows, {} .txt files with a first chunk to look up, in {} ms",
            rows.len(),
            looked_up,
            started.elapsed().as_millis()
        );
    }
    if file_shape.is_empty() {
        return Ok(());
    }
    for row in rows.values_mut() {
        let Some(shape) = file_shape.get(&row.doc_path).copied() else {
            continue;
        };
        if shape == row.shape {
            continue;
        }
        // Re-derive what the shape decided: the dump multiplier on quality and the noise flag.
        if row.shape == TextShape::ChatDump && shape != TextShape::ChatDump {
            row.quality = (row.quality / 0.35).clamp(0.08, 1.0);
        } else if row.shape != TextShape::ChatDump && shape == TextShape::ChatDump {
            row.quality = (row.quality * 0.35).clamp(0.08, 1.0);
        }
        row.shape = shape;
        row.noise = is_noise_artifact(&row.doc_rel_path, shape);
    }
    Ok(())
}

pub(crate) fn chunk_signals_for_ids(
    conn: &Connection,
    semantic_scores: &HashMap<i64, lance_store::VectorHit>,
    lexical_scores: &HashMap<i64, f64>,
    cover: &CoverageTerms,
    query_vector: Option<(&str, &[f32])>,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let mut all_ids: Vec<i64> = semantic_scores
        .keys()
        .chain(lexical_scores.keys())
        .copied()
        .collect();
    if all_ids.is_empty() {
        return Ok(HashMap::new());
    }
    all_ids.sort_unstable();
    all_ids.dedup();

    // Cosine for the chunks the vector search did not return (lexical-only hits), from the
    // vectors SQLite keeps: the same number the vector search reports for its own hits.
    let missing: Vec<i64> = all_ids
        .iter()
        .copied()
        .filter(|id| !semantic_scores.contains_key(id))
        .collect();
    let backfilled: HashMap<i64, f64> = match query_vector {
        Some((model_key, qv)) if !missing.is_empty() => {
            cosine_from_sqlite_vectors(conn, model_key, qv, &missing)?
        }
        _ => HashMap::new(),
    };

    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    for batch in all_ids.chunks(300) {
        let placeholders = std::iter::repeat_n("?", batch.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text,
    pc.doc_mtime
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE pc.id IN ({})
"#,
            placeholders
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed preparing chunk metadata lookup: {}", e))?;
        let rows = stmt
            .query_map(params_from_iter(batch.iter().copied()), |row| {
                let chunk_id: i64 = row.get(0)?;
                let project_path: String = row.get(1)?;
                let doc_path: String = row.get(2)?;
                let doc_rel_path: String = row.get(3)?;
                let chunk_index: i64 = row.get(4)?;
                let text: String = row.get(5)?;
                let doc_mtime: f64 = row.get(6)?;
                Ok((
                    chunk_id,
                    project_path,
                    doc_path,
                    doc_rel_path,
                    chunk_index,
                    text,
                    doc_mtime,
                ))
            })
            .map_err(|e| format!("failed querying chunk metadata lookup: {}", e))?;
        for row in rows {
            let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, doc_mtime) =
                row.map_err(|e| format!("failed reading chunk metadata lookup row: {}", e))?;
            let facts = text_facts(&doc_rel_path, &text, cover);
            let hit = semantic_scores.get(&chunk_id).copied();
            let raw_similarity = hit
                .map(|h| h.raw_similarity)
                .or_else(|| backfilled.get(&chunk_id).copied());
            out.insert(
                chunk_id,
                ChunkSignal {
                    chunk_id,
                    chunk_index,
                    project_path,
                    doc_path,
                    doc_rel_path,
                    doc_mtime,
                    semantic: hit.map(|h| h.score).unwrap_or(0.0),
                    lexical: *lexical_scores.get(&chunk_id).unwrap_or(&0.0),
                    graph: 0.0,
                    relation: "direct".to_string(),
                    quality: facts.quality,
                    excerpt: clip_text(&text, 190),
                    raw_similarity,
                    shape: facts.shape,
                    noise: facts.noise,
                    strong_lexical: facts.strong_lexical,
                },
            );
        }
    }
    apply_file_shapes(conn, &mut out)?;
    Ok(out)
}

/// Fill in `raw_similarity` for fused candidates that still lack one (chunks found only by the
/// path-keyword scan), from the vectors SQLite keeps.
pub(crate) fn backfill_raw_similarity(
    conn: &Connection,
    fused: &mut HashMap<i64, ChunkSignal>,
    model_key: &str,
    query_vector: &[f32],
) -> Result<(), String> {
    let missing: Vec<i64> = fused
        .iter()
        .filter(|(_, row)| row.raw_similarity.is_none())
        .map(|(id, _)| *id)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let found = cosine_from_sqlite_vectors(conn, model_key, query_vector, &missing)?;
    for (id, cos) in found {
        if let Some(row) = fused.get_mut(&id) {
            row.raw_similarity = Some(cos);
        }
    }
    Ok(())
}

/// Overwrite `raw_similarity` of the rows named by `ids` with the cosine to `query_vector`
/// read from the vectors SQLite keeps; a row without a stored vector gets `None`. Used for
/// HyDE-only hits, whose search cosine is to the hypothetical text and not to the query.
pub(crate) fn recompute_raw_similarity(
    conn: &Connection,
    rows: &mut HashMap<i64, ChunkSignal>,
    ids: &[i64],
    model_key: &str,
    query_vector: &[f32],
) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    let found = cosine_from_sqlite_vectors(conn, model_key, query_vector, ids)?;
    for id in ids {
        if let Some(row) = rows.get_mut(id) {
            row.raw_similarity = found.get(id).copied();
        }
    }
    Ok(())
}

/// Cosine similarity between `query_vector` and the stored vectors of `chunk_ids` (for the
/// configured model), read from SQLite. Chunks without a stored vector, with a blob of the
/// wrong length or with a non-finite cosine are absent. `RETRIVIO_DEBUG_BACKFILL=1` prints
/// one line per call to stderr with the number of vector blobs read and the time taken.
pub(crate) fn cosine_from_sqlite_vectors(
    conn: &Connection,
    model_key: &str,
    query_vector: &[f32],
    chunk_ids: &[i64],
) -> Result<HashMap<i64, f64>, String> {
    let qnorm = vector_norm(query_vector);
    let mut out: HashMap<i64, f64> = HashMap::new();
    if qnorm <= 0.0 || !qnorm.is_finite() || chunk_ids.is_empty() {
        return Ok(out);
    }
    let started = std::time::Instant::now();
    let mut blobs_read = 0usize;
    for batch in chunk_ids.chunks(300) {
        let placeholders = std::iter::repeat_n("?", batch.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT chunk_id, norm, vector FROM project_chunk_vectors WHERE model = ?1 AND chunk_id IN ({})",
            placeholders
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("failed preparing chunk vector lookup: {}", e))?;
        let mut binds: Vec<rusqlite::types::Value> = Vec::with_capacity(batch.len() + 1);
        binds.push(rusqlite::types::Value::Text(model_key.to_string()));
        binds.extend(batch.iter().map(|id| rusqlite::types::Value::Integer(*id)));
        let rows = stmt
            .query_map(params_from_iter(binds.iter()), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .map_err(|e| format!("failed querying chunk vectors: {}", e))?;
        for row in rows {
            let (chunk_id, norm, blob) =
                row.map_err(|e| format!("failed reading chunk vector row: {}", e))?;
            blobs_read += 1;
            let vec = blob_to_f32_vec(&blob);
            let vnorm = if norm > 0.0 { norm } else { vector_norm(&vec) };
            if vnorm <= 0.0 || !vnorm.is_finite() || vec.len() != query_vector.len() {
                continue;
            }
            let cos = cosine_raw(query_vector, &vec, qnorm, vnorm);
            if !cos.is_finite() {
                continue;
            }
            out.insert(chunk_id, cos.clamp(-1.0, 1.0));
        }
    }
    if env::var("RETRIVIO_DEBUG_BACKFILL")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!(
            "backfill: {} of {} requested vectors read from SQLite, {} with a finite cosine, in {} ms",
            blobs_read,
            chunk_ids.len(),
            out.len(),
            started.elapsed().as_millis()
        );
    }
    Ok(out)
}

pub(crate) fn project_evidence(
    fused_chunks: &HashMap<i64, ChunkSignal>,
    cfg: &ConfigValues,
    fx: &FreshnessCtx,
) -> HashMap<String, Vec<EvidenceHit>> {
    let mut by_project: HashMap<String, Vec<&ChunkSignal>> = HashMap::new();
    for chunk in fused_chunks.values() {
        by_project
            .entry(chunk.project_path.clone())
            .or_default()
            .push(chunk);
    }

    let mut out: HashMap<String, Vec<EvidenceHit>> = HashMap::new();
    for (project_path, rows) in by_project {
        let mut hits: Vec<EvidenceHit> = Vec::new();
        for row in rows {
            hits.push(evidence_hit_from_chunk(row, chunk_base_score(row, cfg), fx));
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.graph.total_cmp(&a.graph))
                .then_with(|| b.chunk_id.cmp(&a.chunk_id))
        });
        hits.truncate(6);
        out.insert(project_path, hits);
    }
    out
}

pub(crate) fn project_content_scores(
    project_evidence: &HashMap<String, Vec<EvidenceHit>>,
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for (project_path, evidence) in project_evidence {
        if evidence.is_empty() {
            continue;
        }
        let top = evidence.iter().map(|e| e.score).fold(0.0f64, f64::max);
        let n = evidence.len().min(3);
        let mean_top = evidence.iter().take(n).map(|e| e.score).sum::<f64>() / n as f64;
        out.insert(project_path.clone(), (0.65 * top) + (0.35 * mean_top));
    }
    out
}

pub(crate) fn fts_query(query: &str) -> String {
    let mut tokens = all_word_tokens(query);
    if tokens.is_empty() {
        return String::new();
    }
    if tokens.len() > 12 {
        tokens.truncate(12);
    }
    tokens
        .into_iter()
        .map(|t| format!("{}*", t))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn frecency_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
    let now = now_ts();
    let since = now - (120.0 * 86400.0);
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, selected_at
FROM selection_events
WHERE selected_at >= ?1
"#,
        )
        .map_err(|e| format!("failed preparing selection events query: {}", e))?;
    let rows = stmt
        .query_map(params![since], |row| {
            let path: String = row.get(0)?;
            let selected_at: f64 = row.get(1)?;
            Ok((path, selected_at))
        })
        .map_err(|e| format!("failed querying selection events: {}", e))?;

    let mut raw: HashMap<String, f64> = HashMap::new();
    for row in rows {
        let (path, selected_at) =
            row.map_err(|e| format!("failed reading selection event row: {}", e))?;
        let age_days = ((now - selected_at) / 86400.0).max(0.0);
        let weight = (-age_days / 14.0).exp();
        *raw.entry(path).or_insert(0.0) += weight;
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.values().fold(0.0f64, |acc, v| acc.max(*v));
    if hi <= 0.0 {
        return Ok(HashMap::new());
    }
    Ok(raw.into_iter().map(|(k, v)| (k, v / hi)).collect())
}

pub(crate) fn rank_by_frecency_only(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<RankedResult>, String> {
    let frecency = frecency_scores(conn)?;
    let graph = graph_scores(conn)?;
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing projects query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying projects: {}", e))?;
    let mut ranked: Vec<RankedResult> = Vec::new();
    for row in rows {
        let path = row.map_err(|e| format!("failed reading project row: {}", e))?;
        let fr = *frecency.get(&path).unwrap_or(&0.0);
        let gscore = *graph.get(&path).unwrap_or(&0.0);
        ranked.push(RankedResult {
            path,
            score: (0.75 * fr) + (0.25 * gscore),
            lexical: 0.0,
            semantic: 0.0,
            frecency: fr,
            graph: gscore,
            recency: 0.0,
            evidence: Vec::new(),
        });
    }
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.path.cmp(&a.path))
    });
    ranked.truncate(limit.max(1));
    Ok(ranked)
}

pub(crate) fn graph_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT dst AS path, SUM(weight) AS edge_score
FROM project_edges
GROUP BY dst
"#,
        )
        .map_err(|e| format!("failed preparing graph score query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            let path: String = row.get(0)?;
            let edge_score: f64 = row.get(1)?;
            Ok((path, edge_score))
        })
        .map_err(|e| format!("failed querying graph scores: {}", e))?;
    let mut raw: Vec<(String, f64)> = Vec::new();
    for row in rows {
        raw.push(row.map_err(|e| format!("failed reading graph score row: {}", e))?);
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    let hi = raw.iter().map(|r| r.1).fold(0.0f64, f64::max);
    if hi <= 0.0 {
        return Ok(HashMap::new());
    }
    Ok(raw.into_iter().map(|(p, s)| (p, s / hi)).collect())
}

pub(crate) fn clip_text(text: &str, max_chars: usize) -> String {
    let value = collapse_whitespace(text);
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut clipped: String = value.chars().take(max_chars).collect();
    clipped = clipped.trim_end().to_string();
    clipped.push_str("...");
    clipped
}

/// Content quality multiplier for a chunk: 1.0 for ordinary text, lower for machine
/// artefacts. Noise is judged from the text and the file type, never from a directory name
/// such as `sessions/`: a handoff in `docs/sessions/` is a note like any other.
#[cfg(test)]
pub(crate) fn content_quality(doc_rel_path: &str, text: &str) -> f64 {
    content_quality_with_shape(doc_rel_path, text, roles::text_shape(text))
}

/// Machine artefacts by file type and text shape: chat dumps (`Human:`/`Assistant:` turns or
/// JSON-lines message records) in any file, `.jsonl`/`.log` files, lockfiles, minified code.
pub(crate) fn is_noise_artifact(doc_rel_path: &str, shape: TextShape) -> bool {
    let name = doc_rel_path.to_lowercase();
    shape == TextShape::ChatDump
        || name.ends_with(".jsonl")
        || name.ends_with(".log")
        || name.ends_with("package-lock.json")
        || name.ends_with("yarn.lock")
        || name.ends_with("pnpm-lock.yaml")
        || name.ends_with(".min.js")
}

pub(crate) fn content_quality_with_shape(doc_rel_path: &str, text: &str, shape: TextShape) -> f64 {
    let n = text.chars().count().max(1) as f64;
    let letters = text.chars().filter(|ch| ch.is_alphabetic()).count() as f64;
    let punctuation_like = text
        .chars()
        .filter(|ch| matches!(ch, '{' | '}' | '[' | ']' | ',' | ':' | '"' | '\\' | '/'))
        .count() as f64;
    let escaped_newlines = text.matches("\\n").count() as f64;
    let tokens = all_word_tokens(text);
    let token_n = tokens.len();
    if token_n == 0 {
        return 0.1;
    }
    let long_tokens = tokens.iter().filter(|tok| tok.len() >= 24).count() as f64;
    let hex_tokens = tokens.iter().filter(|tok| is_hex_token(tok)).count() as f64;
    let unique_ratio = tokens.iter().collect::<HashSet<_>>().len() as f64 / token_n as f64;

    let mut score: f64 = 1.0;
    if token_n >= 80 {
        if (escaped_newlines / n) > 0.008 {
            score *= 0.68;
        }
        if (punctuation_like / n) > 0.24 && (letters / n) < 0.45 {
            score *= 0.70;
        }
        if (long_tokens / token_n as f64) > 0.20 {
            score *= 0.72;
        }
        if (hex_tokens / token_n as f64) > 0.08 {
            score *= 0.68;
        }
        if unique_ratio < 0.18 {
            score *= 0.78;
        }
    }

    let name = doc_rel_path.to_lowercase();
    if name.ends_with("package-lock.json")
        || name.ends_with("yarn.lock")
        || name.ends_with("pnpm-lock.yaml")
    {
        score *= 0.35;
    }
    if name.ends_with(".metadata.json") {
        score *= 0.70;
    }
    // Chat dumps are machine artefacts whatever their extension (`.txt`, `.json`, `.md`
    // exports); `.jsonl` and `.log` files are dumps by construction.
    if shape == TextShape::ChatDump {
        score *= 0.35;
    } else if name.ends_with(".jsonl") || name.ends_with(".log") {
        score *= 0.60;
    }
    if name.ends_with(".min.js") {
        score *= 0.55;
    }
    score.clamp(0.08, 1.0)
}

pub(crate) fn is_hex_token(token: &str) -> bool {
    if token.len() < 16 {
        return false;
    }
    token.chars().all(|ch| ch.is_ascii_hexdigit())
}

pub(crate) fn path_keyword_scores(paths: &[String], query: &str) -> HashMap<String, f64> {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return HashMap::new();
    }
    let q_set: HashSet<String> = q_tokens.into_iter().collect();
    let q_n = q_set.len() as f64;
    let mut out = HashMap::new();
    for path in paths {
        let name = Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let n_tokens: HashSet<String> = all_word_tokens(&name).into_iter().collect();
        if n_tokens.is_empty() {
            continue;
        }
        let exact = q_set.intersection(&n_tokens).count() as f64 / q_n;
        let fuzzy_hits = q_set
            .iter()
            .filter(|tok| name.contains((*tok).as_str()))
            .count() as f64;
        let fuzzy = fuzzy_hits / q_n;
        let score = exact.max(0.8 * fuzzy).clamp(0.0, 1.0);
        if score > 0.0 {
            out.insert(path.clone(), score);
        }
    }
    out
}

pub(crate) fn doc_keyword_score(doc_rel_path: &str, query: &str) -> f64 {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return 0.0;
    }
    let rel = doc_rel_path.to_lowercase();
    let rel_tokens: HashSet<String> = all_word_tokens(&rel).into_iter().collect();
    let q_set: HashSet<String> = q_tokens.into_iter().collect();
    let q_n = q_set.len() as f64;
    let exact_hits = q_set.intersection(&rel_tokens).count() as f64;
    let fuzzy_hits = q_set
        .iter()
        .filter(|token| rel.contains((*token).as_str()))
        .count() as f64;
    let exact = exact_hits / q_n;
    let fuzzy = fuzzy_hits / q_n;
    exact.max(0.8 * fuzzy).clamp(0.0, 1.0)
}

// ── Query type classification ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryType {
    /// Looks like a symbol name: camelCase, snake_case, PascalCase, etc.
    Symbol,
    /// Natural language question or description: "how does auth work", "explain the login flow"
    NaturalLanguage,
    /// Contains code keywords: fn, function, async, impl, class, struct, etc.
    CodePattern,
    /// Contains path separators or file extensions: src/auth, *.rs, middleware.ts
    PathQuery,
}

/// Weights for fusing multiple ranking signals. Each QueryType gets different weights
/// to emphasize the signal most likely to be useful for that query type.
pub(crate) struct QueryWeights {
    semantic: f64,
    lexical: f64,
    path_kw: f64,
    graph: f64,
    frecency: f64,
}

/// File extensions that mark a query word as a file name: the indexable suffixes plus
/// common code/config extensions people type even though those files are not indexed.
pub(crate) fn known_file_extension(ext: &str) -> bool {
    let e = ext.to_ascii_lowercase();
    if e.is_empty() || e.len() > 10 || !e.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    is_any_indexable_suffix(&format!(".{}", e))
        || matches!(
            e.as_str(),
            "rb" | "kt"
                | "kts"
                | "swift"
                | "scala"
                | "php"
                | "cs"
                | "css"
                | "scss"
                | "less"
                | "xml"
                | "csv"
                | "tsv"
                | "ini"
                | "cfg"
                | "conf"
                | "env"
                | "lock"
                | "proto"
                | "tf"
                | "hcl"
                | "ipynb"
                | "pdf"
                | "log"
                | "mjs"
                | "cjs"
                | "vue"
                | "svelte"
                | "dart"
                | "lua"
                | "pl"
                | "pm"
                | "gradle"
                | "cmake"
                | "mk"
                | "bat"
                | "ps1"
                | "psm1"
                | "wasm"
                | "sol"
        )
}

/// Directory names common enough in repositories that `name/other` reads as a path even in
/// the middle of a sentence ("look in docs/sessions"), unlike prose slashes ("update/enhance").
pub(crate) const PATH_LIKE_DIR_NAMES: &[&str] = &[
    "src",
    "docs",
    "doc",
    "lib",
    "libs",
    "bin",
    "test",
    "tests",
    "spec",
    "crates",
    "pkg",
    "cmd",
    "app",
    "apps",
    "api",
    "config",
    "configs",
    "scripts",
    "internal",
    "examples",
    "assets",
    "static",
    "public",
    "include",
    "build",
    "dist",
    "target",
    "tmp",
    "sessions",
    "notes",
    "packages",
    "modules",
    "components",
    "utils",
    "core",
    "data",
    "etc",
    "usr",
    "var",
    "home",
    "opt",
    "users",
    "vendor",
    "node_modules",
    "migrations",
    "templates",
    "fixtures",
    "handoffs",
];

/// Returns the trimmed word when `token` (one whitespace-delimited word of a query) looks
/// like a filesystem path, a file name or a glob rather than prose that happens to contain
/// a slash. `short_query` is true when the token is the whole query, so "auth/token" alone
/// counts while "update/enhance retrivio" or "update/enhance" inside a sentence does not.
pub(crate) fn path_like_token(token: &str, short_query: bool) -> Option<&str> {
    let mut t = token.trim_matches(|c: char| {
        matches!(
            c,
            '"' | '\''
                | '`'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | ','
                | ';'
                | ':'
                | '!'
                | '?'
        )
    });
    if t.len() > 1 && t.ends_with('.') {
        t = &t[..t.len() - 1];
    }
    if t.is_empty() {
        return None;
    }
    // Explicit path prefixes, UNC shares and Windows drive letters.
    if t.starts_with('/')
        || t.starts_with("./")
        || t.starts_with("../")
        || t.starts_with("~/")
        || t.starts_with(".\\")
        || t.starts_with("..\\")
        || t.starts_with("\\\\")
    {
        return Some(t);
    }
    let b = t.as_bytes();
    if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
    {
        return Some(t);
    }
    let normalized = t.replace('\\', "/");
    let trailing_slash = normalized.len() > 1 && normalized.ends_with('/');
    let body = normalized.trim_end_matches('/');
    let segments: Vec<&str> = body.split('/').collect();
    let last = segments.last().copied().unwrap_or("");
    // Globs and file names: "*.rs", ".md", "README.md", "middleware.ts", "src/auth/token.rs".
    let ext = last.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if !ext.is_empty() && known_file_extension(ext) {
        return Some(t);
    }
    // Dotfiles: ".gitignore", ".npmrc" (lowercase, one leading dot, no other dot).
    if segments.len() == 1
        && last.len() >= 3
        && last.starts_with('.')
        && !last[1..].contains('.')
        && last[1..]
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Some(t);
    }
    // Slash-separated segments without an extension need real segment names ("1/IAM" and
    // "w/o" are prose); "src/" keeps its trailing slash as the path signal.
    if segments.len() < 2 && !trailing_slash {
        return None;
    }
    if segments.iter().any(|s| s.len() < 2) {
        return None;
    }
    // Dates and fractions: "09/19", "2026/09/19", "10/20".
    if segments
        .iter()
        .all(|s| s.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }
    if trailing_slash || segments.len() >= 3 {
        return Some(t);
    }
    // Exactly one slash. Path-flavoured punctuation ("202609-ai-handoff/notes", "my_pkg/mod"),
    // a well-known directory name ("docs/sessions") or a query that is just this token make
    // it a path; two plain words ("update/enhance", "and/or", "AWS/S3") stay prose.
    let plain_words = segments
        .iter()
        .all(|s| s.chars().all(|c| c.is_ascii_alphanumeric()));
    if !plain_words || short_query {
        return Some(t);
    }
    let known_dir = segments
        .iter()
        .any(|s| PATH_LIKE_DIR_NAMES.contains(&s.to_ascii_lowercase().as_str()));
    if known_dir {
        Some(t)
    } else {
        None
    }
}

/// The words of `query` that look like paths, file names or globs (see `path_like_token`).
pub(crate) fn query_path_like_tokens(query: &str) -> Vec<String> {
    let words: Vec<&str> = query.split_whitespace().collect();
    let short_query = words.len() == 1;
    words
        .iter()
        .filter_map(|w| path_like_token(w, short_query))
        .map(|s| s.to_string())
        .collect()
}

impl QueryType {
    /// Classify a query string into a QueryType using heuristics.
    fn classify(query: &str) -> QueryType {
        let q = query.trim();

        // PathQuery: some word looks like a path, a file name or a glob ("src/auth",
        // "middleware.ts", "*.rs", ".md", "docs/sessions"). A slash inside prose
        // ("update/enhance", "1/IAM") does not count; see `path_like_token`.
        if !query_path_like_tokens(q).is_empty() {
            return QueryType::PathQuery;
        }

        // NaturalLanguage: contains question words or multiple words with spaces
        let lower = q.to_lowercase();
        let question_words = [
            "how ",
            "what ",
            "why ",
            "where ",
            "when ",
            "which ",
            "explain ",
            "describe ",
            "find ",
            "show ",
            "list ",
        ];
        if question_words.iter().any(|w| lower.starts_with(w)) {
            return QueryType::NaturalLanguage;
        }

        // CodePattern: contains language keywords that suggest code search
        let code_keywords = [
            "fn ",
            "function ",
            "async ",
            "impl ",
            "class ",
            "struct ",
            "trait ",
            "interface ",
            "enum ",
            "type ",
            "def ",
            "import ",
            "use ",
            "pub ",
            "private ",
            "protected ",
            "const ",
            "let ",
            "var ",
        ];
        // Whole-word match only: "lets" is not `let`, "users" is not `use`.
        if code_keywords.iter().any(|kw| {
            lower.starts_with(kw)
                || lower.contains(&format!(" {}", kw))
                || lower.ends_with(&format!(" {}", kw.trim()))
        }) {
            return QueryType::CodePattern;
        }

        // Symbol: looks like an identifier (camelCase, snake_case, PascalCase, UPPER_CASE)
        // Heuristic: no spaces, or 1-2 tokens that look like identifiers
        let tokens: Vec<&str> = q.split_whitespace().collect();
        // A slash that survived the path check is prose ("update/enhance", "1/IAM"): words,
        // not an identifier, however short the query.
        if tokens.iter().any(|t| t.contains('/')) {
            return QueryType::NaturalLanguage;
        }
        if tokens.len() <= 2 {
            let all_look_like_symbols = tokens.iter().all(|t| {
                let has_case_transition = t
                    .chars()
                    .zip(t.chars().skip(1))
                    .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
                let has_underscore = t.contains('_');
                let has_colon_colon = t.contains("::");
                let has_dot = t.contains('.');
                let short_enough = t.len() <= 64;
                short_enough
                    && (has_case_transition
                        || has_underscore
                        || has_colon_colon
                        || has_dot
                        || t.len() <= 20)
            });
            if all_look_like_symbols {
                return QueryType::Symbol;
            }
        }

        // Default: if it's a short query with no question words, treat as Symbol
        if tokens.len() <= 2 {
            return QueryType::Symbol;
        }

        QueryType::NaturalLanguage
    }

    /// Ranking weights for project-level scoring. Natural-language queries, the common case
    /// and the one `retrivio autotune` tunes, read the `rank_project_*` config keys
    /// (`rank_project_content_weight` weighs the chunk-evidence content score, which the
    /// `lexical` slot carries); the three query-shape profiles stay fixed.
    fn project_weights(&self, cfg: &ConfigValues) -> QueryWeights {
        match self {
            //                        semantic  lexical  path_kw  graph  frecency
            QueryType::Symbol => QueryWeights {
                semantic: 0.18,
                lexical: 0.42,
                path_kw: 0.24,
                graph: 0.06,
                frecency: 0.10,
            },
            QueryType::NaturalLanguage => QueryWeights {
                semantic: cfg.rank_project_semantic_weight,
                lexical: cfg.rank_project_content_weight,
                path_kw: cfg.rank_project_path_weight,
                graph: cfg.rank_project_graph_weight,
                frecency: cfg.rank_project_frecency_weight,
            },
            QueryType::CodePattern => QueryWeights {
                semantic: 0.40,
                lexical: 0.30,
                path_kw: 0.08,
                graph: 0.12,
                frecency: 0.10,
            },
            QueryType::PathQuery => QueryWeights {
                semantic: 0.08,
                lexical: 0.12,
                path_kw: 0.60,
                graph: 0.10,
                frecency: 0.10,
            },
        }
    }

    /// Get adaptive retrieval candidate limits based on query type.
    /// Returns (semantic_limit, lexical_limit).
    fn retrieval_limits(&self, base_semantic: usize, base_lexical: usize) -> (usize, usize) {
        match self {
            QueryType::Symbol => (base_semantic.min(40), base_lexical.max(200)),
            QueryType::NaturalLanguage => (base_semantic.max(200), base_lexical.max(60)),
            QueryType::CodePattern => (base_semantic.max(120), base_lexical.max(120)),
            QueryType::PathQuery => (base_semantic.min(40), base_lexical.max(80)),
        }
    }
}

#[cfg(test)]
mod query_type_tests {
    use super::*;
    use crate::db::init_schema;
    use crate::roles;
    use crate::roles::{Role, TextShape};
    use rusqlite::{params, Connection};

    #[test]
    fn prose_slashes_are_not_paths() {
        assert_eq!(
            QueryType::classify("update/enhance retrivio"),
            QueryType::NaturalLanguage
        );
        assert_eq!(
            QueryType::classify("ok well lets update/enhance retrivio to suit our needs"),
            QueryType::NaturalLanguage
        );
        assert_ne!(QueryType::classify("1/IAM role"), QueryType::PathQuery);
        assert_ne!(
            QueryType::classify("compare AWS/S3 to GCS buckets"),
            QueryType::PathQuery
        );
        assert_ne!(QueryType::classify("09/19"), QueryType::PathQuery);
        assert_ne!(
            QueryType::classify("meeting on 09/19 at noon"),
            QueryType::PathQuery
        );
        assert_ne!(
            QueryType::classify("3/4 of the fleet"),
            QueryType::PathQuery
        );
        assert!(query_path_like_tokens("2026/09/19").is_empty());
        assert!(query_path_like_tokens("see https://example.com/x for details").is_empty());
        assert!(query_path_like_tokens("ok well lets update/enhance retrivio").is_empty());
        assert!(query_path_like_tokens("either and/or both of them").is_empty());
    }

    #[test]
    fn real_paths_globs_and_file_names_are_path_queries() {
        assert_eq!(
            QueryType::classify("fix src/auth/token.rs"),
            QueryType::PathQuery
        );
        assert_eq!(QueryType::classify("docs/sessions"), QueryType::PathQuery);
        assert_eq!(
            QueryType::classify("look in docs/sessions for the handoff"),
            QueryType::PathQuery
        );
        assert_eq!(QueryType::classify("*.rs"), QueryType::PathQuery);
        assert_eq!(QueryType::classify(".md"), QueryType::PathQuery);
        assert_eq!(QueryType::classify("middleware.ts"), QueryType::PathQuery);
        assert_eq!(
            QueryType::classify("~/.retrivio/config.toml"),
            QueryType::PathQuery
        );
        assert_eq!(QueryType::classify("./scripts"), QueryType::PathQuery);
        assert_eq!(QueryType::classify("auth/token"), QueryType::PathQuery);
        assert_eq!(
            QueryType::classify("see 202609-ai-handoff/notes please"),
            QueryType::PathQuery
        );
        assert_eq!(
            query_path_like_tokens("fix (src/auth/token.rs) now"),
            vec!["src/auth/token.rs".to_string()]
        );
        assert_eq!(QueryType::classify(r"\\server\share"), QueryType::PathQuery);
        assert_eq!(QueryType::classify(r"C:\x\y"), QueryType::PathQuery);
        assert_eq!(QueryType::classify(".gitignore"), QueryType::PathQuery);
        assert_eq!(
            QueryType::classify("open README.markdown"),
            QueryType::PathQuery
        );
    }

    #[test]
    fn other_query_types_are_unchanged() {
        assert_eq!(
            QueryType::classify("reindex_project_chunks"),
            QueryType::Symbol
        );
        assert_eq!(
            QueryType::classify("how does auth work"),
            QueryType::NaturalLanguage
        );
        assert_eq!(
            QueryType::classify("fn keyword_path_chunk_scores"),
            QueryType::CodePattern
        );
        assert_eq!(
            QueryType::classify("impl Display for Foo"),
            QueryType::CodePattern
        );
        assert_eq!(
            QueryType::classify("where is the struct"),
            QueryType::NaturalLanguage
        );
        // Keyword prefixes inside ordinary words are not code keywords.
        assert_eq!(
            QueryType::classify("how do users log in"),
            QueryType::NaturalLanguage
        );
        assert_eq!(
            QueryType::classify("show me the types of storage classes"),
            QueryType::NaturalLanguage
        );
    }

    #[test]
    fn keyword_path_scores_only_scan_for_path_words() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        init_schema(&conn).expect("init schema");
        conn.execute_batch(
            r#"
INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed)
VALUES (1, '/p/retrivio', 'retrivio', 'r', 0, 0), (2, '/p/other', 'other', 'o', 0, 0);
INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (1, 1, '/p/retrivio/docs/sessions/handoff.md', 'docs/sessions/handoff.md', 0, 0, 3, 'a', 'handoff', 0),
       (2, 1, '/p/retrivio/src/main.rs', 'src/main.rs', 0, 0, 3, 'b', 'main', 0),
       (3, 2, '/p/other/notes/update.md', 'notes/update.md', 0, 0, 3, 'c', 'update enhance', 0);
"#,
        )
        .expect("seed");
        // Prose with a slash: no path word, more than two words -> nothing scanned.
        let none = keyword_path_chunk_scores(
            &conn,
            "ok well lets update/enhance retrivio to suit our needs",
            50,
        )
        .expect("scores");
        assert!(none.is_empty());
        // A path word only matches files on that path.
        let hits = keyword_path_chunk_scores(&conn, "open docs/sessions now", 50).expect("scores");
        assert_eq!(hits.keys().copied().collect::<Vec<_>>(), vec![1]);
        assert!((hits[&1].lexical - 1.0).abs() < 1e-9);
        // Symbol-shaped queries still match by file and project path.
        let sym = keyword_path_chunk_scores(&conn, "main", 50).expect("scores");
        assert!(sym.contains_key(&2));
        let proj = keyword_path_chunk_scores(&conn, "retrivio", 50).expect("scores");
        assert!(proj.contains_key(&1) && proj.contains_key(&2) && !proj.contains_key(&3));
    }

    /// A path query hits every chunk of a `.txt` file. The rows the scan keeps take the
    /// file's shape (its first chunk), as the FTS and vector candidates do, so a later chunk of
    /// a transcript is a record and a later chunk of a chat dump is noise, whether or not the
    /// first chunk survived the `keep_top` cut. A `.md` file keeps the shape of its own chunk.
    #[test]
    fn keyword_path_rows_take_the_txt_file_shape() {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        init_schema(&conn).expect("init schema");
        let prose =
            "The team walked through the quarterly roadmap and the staffing plan in detail.";
        let transcript_head =
            "# Meeting Transcript **Date:** Monday, April 13, 2026\n\n## Summary\nThe team discussed the roadmap.\n";
        let dump_head = "Human: plan the migration\n\nAssistant: The migration moves in three waves.\n\nHuman: and the budget?\n\nAssistant: One budget line per wave.\n\nHuman: ok\n\nAssistant: fine\n";
        assert_eq!(roles::text_shape(transcript_head), TextShape::Transcript);
        assert_eq!(roles::text_shape(dump_head), TextShape::ChatDump);
        assert_eq!(roles::text_shape(prose), TextShape::Prose);
        // Chunk 0 of each file gets the highest id: ties in the scan are cut by ascending chunk
        // id, so `keep_top = 1` keeps the last chunk and drops the first.
        conn.execute(
            "INSERT INTO projects(id, path, title, summary, project_mtime, last_indexed) VALUES (1, '/p/acme', 'acme', 'a', 0, 0)",
            [],
        )
        .expect("seed project");
        let rows: [(i64, &str, &str, i64, &str); 7] = [
            (
                13,
                "/p/acme/notes/roadmap-call.txt",
                "notes/roadmap-call.txt",
                0,
                transcript_head,
            ),
            (
                12,
                "/p/acme/notes/roadmap-call.txt",
                "notes/roadmap-call.txt",
                1,
                prose,
            ),
            (
                11,
                "/p/acme/notes/roadmap-call.txt",
                "notes/roadmap-call.txt",
                2,
                prose,
            ),
            (
                23,
                "/p/acme/exports/chat-export.txt",
                "exports/chat-export.txt",
                0,
                dump_head,
            ),
            (
                21,
                "/p/acme/exports/chat-export.txt",
                "exports/chat-export.txt",
                2,
                prose,
            ),
            (
                33,
                "/p/acme/minutes/board-sync.md",
                "minutes/board-sync.md",
                0,
                transcript_head,
            ),
            (
                31,
                "/p/acme/minutes/board-sync.md",
                "minutes/board-sync.md",
                1,
                prose,
            ),
        ];
        for (id, doc_path, rel, index, text) in rows {
            conn.execute(
                "INSERT INTO project_chunks(id, project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at) VALUES (?1, 1, ?2, ?3, 0, ?4, 3, ?5, ?6, 0)",
                params![id, doc_path, rel, index, format!("h{}", id), text],
            )
            .expect("seed chunk");
        }
        let none: Vec<String> = Vec::new();

        // Every chunk retained: the later chunks carry the first chunk's shape.
        let all = keyword_path_chunk_scores(&conn, "notes/roadmap-call", 50).expect("scores");
        let mut ids: Vec<i64> = all.keys().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![11, 12, 13]);
        for id in [11, 12, 13] {
            assert_eq!(all[&id].shape, TextShape::Transcript, "chunk id {}", id);
            assert!(!all[&id].noise);
            assert_eq!(
                roles::classify(&all[&id].doc_rel_path, all[&id].shape, &none),
                Role::Record,
                "chunk id {}",
                id
            );
        }

        // The cut kept only the last chunk: the first chunk is looked up for its shape.
        let cut = keyword_path_chunk_scores(&conn, "notes/roadmap-call", 1).expect("scores");
        assert_eq!(cut.keys().copied().collect::<Vec<_>>(), vec![11]);
        assert_eq!(cut[&11].chunk_index, 2);
        assert_eq!(cut[&11].shape, TextShape::Transcript);
        assert!(!cut[&11].noise);
        assert_eq!(
            roles::classify(&cut[&11].doc_rel_path, cut[&11].shape, &none),
            Role::Record
        );

        // A later chunk of a chat dump is noise with the dump's quality multiplier.
        let dump = keyword_path_chunk_scores(&conn, "exports/chat-export", 1).expect("scores");
        assert_eq!(dump.keys().copied().collect::<Vec<_>>(), vec![21]);
        assert_eq!(dump[&21].shape, TextShape::ChatDump);
        assert!(dump[&21].noise, "noise from the file's first chunk");
        assert!(dump[&21].quality <= 0.36, "{}", dump[&21].quality);
        assert_eq!(
            roles::classify(&dump[&21].doc_rel_path, dump[&21].shape, &none),
            Role::Knowledge
        );

        // Only `.txt` roles depend on the shape: the `.md` chunk keeps its own.
        let md = keyword_path_chunk_scores(&conn, "minutes/board-sync", 1).expect("scores");
        assert_eq!(md.keys().copied().collect::<Vec<_>>(), vec![31]);
        assert_eq!(md[&31].shape, TextShape::Prose);
        assert!(!md[&31].noise);
    }
}

/// Penalty for paths that are unlikely to be the primary copy of anything: scratch and
/// state directories, copies (`snapshot`, `backup`, `archive`, `copy` directories) and data
/// files. Judged on directory *components* of the path relative to the project.
pub(crate) fn path_noise_penalty(doc_rel_path: &str) -> f64 {
    let p = doc_rel_path.replace('\\', "/").to_lowercase();
    let mut dirs: Vec<&str> = p.split('/').filter(|c| !c.is_empty()).collect();
    dirs.pop();
    let mut penalty = 1.0f64;
    if dirs.contains(&"tmp") {
        penalty *= 0.55;
    }
    if dirs.contains(&"state") {
        penalty *= 0.72;
    }
    if dirs.iter().any(|d| roles::is_noise_dir(d)) {
        penalty *= 0.85;
    }
    if p.ends_with(".json") {
        penalty *= 0.92;
    }
    penalty.clamp(0.25, 1.0)
}

pub(crate) fn is_generic_container(path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(
        name.as_str(),
        "archive"
            | "archived"
            | "archives"
            | "misc"
            | "tmp"
            | "temp"
            | "scratch"
            | "old"
            | "backup"
            | "backups"
    )
}

pub(crate) fn all_word_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Search symbols using FTS5 full-text search.
pub(crate) fn search_symbols_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<Value>, String> {
    // Escape FTS5 special characters and build a prefix query
    let escaped = query
        .replace('"', "\"\"")
        .replace('*', "")
        .trim()
        .to_string();
    if escaped.is_empty() {
        return Ok(Vec::new());
    }
    // Use prefix matching for partial symbol names
    let fts_query = format!("\"{}\"*", escaped);

    let mut stmt = conn
        .prepare(
            r#"
SELECT s.id, s.project_id, s.doc_path, s.doc_rel_path, s.name, s.qualified_name,
       s.kind, s.line_start, s.line_end, s.signature, s.doc_comment, s.visibility,
       p.path as project_path
FROM symbol_fts
JOIN symbols s ON s.id = symbol_fts.rowid
JOIN projects p ON p.id = s.project_id
WHERE symbol_fts MATCH ?1
ORDER BY rank
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing symbol search: {}", e))?;

    let rows = stmt
        .query_map(params![fts_query, limit as i64], |row| {
            Ok(serde_json::json!({
                "symbol_id": row.get::<_, i64>(0)?,
                "project_id": row.get::<_, i64>(1)?,
                "doc_path": row.get::<_, String>(2)?,
                "doc_rel_path": row.get::<_, String>(3)?,
                "name": row.get::<_, String>(4)?,
                "qualified_name": row.get::<_, String>(5)?,
                "kind": row.get::<_, String>(6)?,
                "line_start": row.get::<_, i64>(7)?,
                "line_end": row.get::<_, i64>(8)?,
                "signature": row.get::<_, String>(9)?,
                "doc_comment": row.get::<_, String>(10)?,
                "visibility": row.get::<_, String>(11)?,
                "project_path": row.get::<_, String>(12)?,
            }))
        })
        .map_err(|e| format!("failed executing symbol search: {}", e))?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading symbol row: {}", e))?);
    }
    Ok(out)
}

#[cfg(test)]
mod ranking_honesty_tests {
    use super::*;
    use crate::db::with_lance_store;
    use crate::embed::{embed_query_cached, model_key_for_cfg};
    use crate::roles::TextShape;
    use crate::test_support::*;
    use crate::util::{normalize_path, now_ts};
    use crate::{freshness, lance_store};
    use rusqlite::params;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn write(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(path, text).expect("write");
    }

    fn at(y: i64, m: u32, d: u32) -> SystemTime {
        let secs = freshness::days_from_civil(y, m, d) as u64 * 86_400 + 43_200;
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    const SPEC: &str = "# Otter migration design\n\nThe otter migration moves the otter fleet to the new habitat store in three waves, one per region, with a pricing tier per wave.\n";

    /// One project with a three-handoff series, a transcript dated by its path but edited
    /// today, a spec, a snapshot copy of the spec and a chat-log dump.
    fn fixture(store: &TestStore) -> (PathBuf, PathBuf) {
        let root = store.corpus_root("root");
        // Two projects under the root so discovery reads it as a workspace (a lone child
        // directory would be unwrapped as the project itself).
        write(
            &root.join("beta").join("notes.md"),
            "# Beta\n\nUnrelated notes on zebra habitats and quarterly tax filing checklists.\n",
        );
        let proj = root.join("acme");
        write(
            &proj.join("README.md"),
            "# Acme\n\nProject notes for the otter habitat programme.\n",
        );
        let handoffs = [
            ("HANDOFF-2026-06-10.md", (2026, 6, 10), "# Handoff 2026-06-10\n\nOtter migration handoff. State: the otter migration plan is drafted. Next step: the pricing review with the habitat team.\n"),
            ("HANDOFF-2026-06-19.md", (2026, 6, 19), "# Handoff 2026-06-19\n\nOtter migration handoff. State: pricing review done, two tiers agreed. Next step: the pilot cut-over of the otter migration.\n"),
            ("HANDOFF-2026-06-24.md", (2026, 6, 24), "# Handoff 2026-06-24\n\nOtter migration handoff. State: pilot cut-over of the otter migration complete. Next step: the retrospective and wave two.\n"),
        ];
        for (name, (y, m, d), text) in handoffs {
            let p = proj.join("docs").join("sessions").join(name);
            write(&p, text);
            set_mtime(&p, at(y, m, d));
        }
        // A transcript in a plain notes folder: the role comes from the text shape, the date
        // from the path prefix; the file itself was edited today.
        write(
            &proj.join("notes").join("20260413-otter-pricing-call.txt"),
            "# Meeting Transcript **Date:** Monday, April 13, 2026\n\n## Summary\nThe team discussed otter migration pricing tiers.\n\nAlice: what does the otter pricing look like\nBob: three tiers, one per wave\nAlice: and the migration timeline\nBob: pilot in June\n",
        );
        write(&proj.join("specs").join("otter-migration-design.md"), SPEC);
        write(
            &proj
                .join("memory-snapshot")
                .join("specs")
                .join("otter-migration-design.md"),
            SPEC,
        );
        write(
            &proj.join("exports").join("otter-chat-dump.txt"),
            "Human: plan the otter migration in three waves\n\nAssistant: The otter migration moves the otter fleet to the new habitat store in three waves.\n\nHuman: and the pricing tier per wave?\n\nAssistant: One pricing tier per wave.\n",
        );
        (root, proj)
    }

    fn find<'a>(
        rows: &'a [RankedFileResult],
        suffix: &str,
    ) -> Option<(usize, &'a RankedFileResult)> {
        rows.iter()
            .enumerate()
            .find(|(_, r)| r.path.ends_with(suffix))
    }

    #[test]
    fn roles_supersession_dedup_noise_and_floor_on_a_fixture_project() {
        let store = TestStore::new("ranking-honesty");
        let (root, _proj) = fixture(&store);
        store.track(&root);
        let cfg = store.cfg(&root, &[("embed_backend", "hash")]);
        let embedder = TestEmbedder::new(&cfg, true);
        let stats = store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        let projects: Vec<String> = {
            let mut st = conn
                .prepare("SELECT path FROM projects ORDER BY path")
                .unwrap();
            st.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .map(|r| r.unwrap())
                .collect()
        };
        assert_eq!(stats.updated_projects, 2, "projects: {:?}", projects);
        assert_eq!(stats.files_selected, 9);

        // Handoffs: state, never penalised as "sessions" noise, newest first, the older two
        // carry `superseded_by`, and at 89+ days they are `verify`, not `stale`.
        let q = "otter migration handoff next step";
        let rows =
            rank_files_native_with(&conn, &cfg, q, 10, RankOptions::default()).expect("rank");
        let (i24, h24) = find(&rows, "HANDOFF-2026-06-24.md").expect("newest handoff");
        let (i19, h19) = find(&rows, "HANDOFF-2026-06-19.md").expect("middle handoff");
        let (i10, h10) = find(&rows, "HANDOFF-2026-06-10.md").expect("oldest handoff");
        for (h, day) in [
            (h24, "2026-06-24"),
            (h19, "2026-06-19"),
            (h10, "2026-06-10"),
        ] {
            assert_eq!(h.role, "state", "{}", h.path);
            assert!(
                h.quality > 0.99,
                "handoff penalised: {} q={}",
                h.path,
                h.quality
            );
            assert!(!h.noise);
            assert!(
                h.raw_similarity.is_some(),
                "every candidate carries a cosine"
            );
            assert_eq!(h.freshness_tier, "verify", "{} age {}", h.path, h.age_days);
            assert!(h.verify);
            // State takes the newer of path date and last edit; both fall on the same day here.
            assert_eq!(freshness::format_ymd(h.content_date), day, "{}", h.path);
        }
        assert!(h24.superseded_by.is_none(), "{:?}", h24.superseded_by);
        assert_eq!(h19.superseded_by.as_deref(), Some(h24.path.as_str()));
        assert_eq!(h10.superseded_by.as_deref(), Some(h24.path.as_str()));
        assert!(
            i24 < i19 && i24 < i10,
            "newest first: {} {} {}",
            i24,
            i19,
            i10
        );

        // Editing the oldest handoff today (a typo fix: new content, mtime now) does not make
        // it the head: series order follows the revision date (the date in the file name),
        // not the last edit, although the content date does move to the edit.
        let oldest = _proj
            .join("docs")
            .join("sessions")
            .join("HANDOFF-2026-06-10.md");
        let original = fs::read_to_string(&oldest).unwrap();
        write(&oldest, &format!("{}\nTypo fixed later.\n", original));
        set_mtime(&oldest, SystemTime::now());
        let stats = store.index(&cfg, &embedder, false).expect("reindex");
        assert!(stats.files_selected >= 1);
        let touched =
            rank_files_native_with(&conn, &cfg, q, 10, RankOptions::default()).expect("rank");
        let (_, t24) = find(&touched, "HANDOFF-2026-06-24.md").unwrap();
        let (_, t10) = find(&touched, "HANDOFF-2026-06-10.md").unwrap();
        assert_eq!(t10.date_source, "mtime", "content date moved to the edit");
        assert!(t10.age_days < 1.0, "{}", t10.age_days);
        assert!(t24.superseded_by.is_none(), "{:?}", t24.superseded_by);
        assert_eq!(
            t10.superseded_by.as_deref(),
            Some(t24.path.as_str()),
            "the edited June 10 handoff stays superseded by June 24"
        );
        write(&oldest, &original);
        set_mtime(&oldest, at(2026, 6, 10));
        store.index(&cfg, &embedder, false).expect("reindex");

        // History or the flag: superseded files stay marked but rank at full strength.
        let strong = rank_files_native_with(
            &conn,
            &cfg,
            q,
            10,
            RankOptions {
                include_superseded: true,
                ..RankOptions::default()
            },
        )
        .expect("rank");
        let (_, s19) = find(&strong, "HANDOFF-2026-06-19.md").unwrap();
        assert_eq!(s19.superseded_by.as_deref(), Some(h24.path.as_str()));
        assert!(
            s19.score > h19.score
                && (s19.base_score - h19.base_score / SUPERSEDED_FACTOR).abs() < 1e-9,
            "flag restores full strength: {} vs {}",
            s19.score,
            h19.score
        );
        let hist = rank_files_native_with(
            &conn,
            &cfg,
            "what did the otter migration handoff say in June",
            10,
            RankOptions::default(),
        )
        .expect("rank");
        let (_, x19) = find(&hist, "HANDOFF-2026-06-19.md").unwrap();
        let (_, x24) = find(&hist, "HANDOFF-2026-06-24.md").unwrap();
        assert!(x19.superseded_by.is_some() && x24.superseded_by.is_none());
        assert!((x19.base_score / x19.quality) > 0.0);
        // Same downrank factor absent: the two base scores are not 0.85 apart by construction.
        assert!(
            (x19.base_score - x24.base_score).abs() < (1.0 - SUPERSEDED_FACTOR) * x24.base_score,
            "history query shows the series at full strength: {} vs {}",
            x19.base_score,
            x24.base_score
        );

        // Transcript: a record dated by the event in its path, although edited today.
        let rows = rank_files_native_with(
            &conn,
            &cfg,
            "otter pricing call transcript",
            10,
            RankOptions::default(),
        )
        .expect("rank");
        let (_, t) = find(&rows, "20260413-otter-pricing-call.txt").expect("transcript");
        assert_eq!(t.role, "record");
        assert_eq!(t.freshness_tier, "record");
        assert_eq!(t.date_source, "path-date");
        assert_eq!(freshness::format_ymd(t.content_date), "2026-04-13");
        assert!(t.age_days > 100.0, "{}", t.age_days);
        assert!(!t.verify);

        // Spec vs snapshot copy vs chat dump.
        let rows = rank_files_native_with(
            &conn,
            &cfg,
            "otter migration three waves habitat store",
            10,
            RankOptions::default(),
        )
        .expect("rank");
        let (ispec, spec) = find(&rows, "specs/otter-migration-design.md").expect("spec");
        assert!(!spec.path.contains("memory-snapshot"));
        assert!(
            find(&rows, "memory-snapshot/specs/otter-migration-design.md").is_none(),
            "identical snapshot copy collapsed into the original: {:?}",
            rows.iter().map(|r| &r.path).collect::<Vec<_>>()
        );
        assert_eq!(spec.role, "knowledge");
        assert!(!spec.noise && spec.quality > 0.99);
        let (idump, dump) = find(&rows, "exports/otter-chat-dump.txt").expect("dump");
        assert!(dump.noise, "chat dump flagged as noise");
        assert!(dump.quality <= 0.36, "dump quality {}", dump.quality);
        assert!(ispec < idump, "spec {} above dump {}", ispec, idump);

        // Raw-cosine floor: an unrelated query returns nothing once the floor sits between
        // the unrelated and the related top cosine; the related query still answers.
        let related = rank_files_native_with(&conn, &cfg, q, 10, RankOptions::default()).unwrap();
        let unrelated_q = "zebra quarterly tax filing checklist";
        let unrelated =
            rank_files_native_with(&conn, &cfg, unrelated_q, 10, RankOptions::default()).unwrap();
        let top_raw = |rows: &[RankedFileResult]| {
            rows.iter()
                .filter_map(|r| r.raw_similarity)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let (rel_top, unrel_top) = (top_raw(&related), top_raw(&unrelated));
        assert!(
            rel_top > unrel_top,
            "related {} unrelated {}",
            rel_top,
            unrel_top
        );
        let floor = (rel_top + unrel_top) / 2.0;
        let opts = RankOptions {
            min_raw_similarity: floor,
            ..RankOptions::default()
        };
        assert!(
            rank_files_native_with(&conn, &cfg, unrelated_q, 10, opts)
                .unwrap()
                .is_empty(),
            "unrelated query under the floor {}",
            floor
        );
        assert!(!rank_files_native_with(&conn, &cfg, q, 10, opts)
            .unwrap()
            .is_empty());
        // The floor is stated in raw cosine, never in the normalised score whose top is 1.0.
        assert!(related.iter().any(|r| (r.semantic - 1.0).abs() < 1e-9));
        assert!(unrelated.iter().any(|r| (r.semantic - 1.0).abs() < 1e-9));
    }

    /// Duplicate collapse uses one identity, the whole-file manifest hash, and two rules: same
    /// file name, or one copy under a copy directory. Byte-identical documents under different
    /// names in different projects stay two results, and the survivor keeps its own score.
    #[test]
    fn duplicate_collapse_uses_file_identity_and_copy_rules() {
        let store = TestStore::new("dup-identity");
        let root = store.corpus_root("root");
        let acme = root.join("acme");
        let beta = root.join("beta");
        let gamma = root.join("gamma");
        write(
            &acme.join("README.md"),
            "# Acme\n\nOtter habitat programme.\n",
        );
        write(
            &beta.join("README.md"),
            "# Beta\n\nOtter partner programme.\n",
        );
        write(
            &gamma.join("README.md"),
            "# Gamma\n\nOtter research programme.\n",
        );
        // Same name and bytes in two projects: copies of one file.
        write(&acme.join("specs").join("otter-migration-design.md"), SPEC);
        write(&beta.join("specs").join("otter-migration-design.md"), SPEC);
        // Same bytes, another name, no copy directory: another document.
        write(&gamma.join("design").join("otter-design-v1.md"), SPEC);
        // Same bytes under a copy directory, yet another name: a copy.
        write(&gamma.join("archive").join("old-otter.md"), SPEC);
        store.track(&root);
        let cfg = store.cfg(&root, &[("embed_backend", "hash")]);
        let embedder = TestEmbedder::new(&cfg, true);
        store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        let q = "otter migration three waves habitat store";
        let rows = rank_files_native_with(&conn, &cfg, q, 20, RankOptions::default()).unwrap();
        let spec_copies: Vec<&RankedFileResult> = rows
            .iter()
            .filter(|r| r.path.ends_with("specs/otter-migration-design.md"))
            .collect();
        assert_eq!(
            spec_copies.len(),
            1,
            "same name and bytes collapse: {:?}",
            rows.iter().map(|r| &r.path).collect::<Vec<_>>()
        );
        assert!(
            find(&rows, "design/otter-design-v1.md").is_some(),
            "a byte-identical document under another name stays"
        );
        assert!(
            find(&rows, "archive/old-otter.md").is_none(),
            "a copy under a copy directory folds into an original"
        );

        // The survivor keeps its own numbers: the archive copy scores higher but loses to the
        // original by the copy-directory rule, and the original's score stays what it was.
        let paths: Vec<String> = rows.iter().map(|r| r.path.clone()).collect();
        let hashes = file_content_hashes(&conn, &paths);
        let (_, kept) = find(&rows, "specs/otter-migration-design.md").unwrap();
        let copy_path =
            normalize_path(&gamma.join("archive").join("old-otter.md").to_string_lossy())
                .to_string_lossy()
                .to_string();
        assert_eq!(
            hashes.get(&kept.path),
            file_content_hashes(&conn, std::slice::from_ref(&copy_path)).get(&copy_path),
            "both copies share the manifest hash"
        );
        let mut copy = kept.clone();
        copy.path = copy_path.clone();
        copy.doc_rel_path = "archive/old-otter.md".to_string();
        copy.project_path = normalize_path(&gamma.to_string_lossy())
            .to_string_lossy()
            .to_string();
        copy.score = kept.score + 0.5;
        copy.base_score = kept.base_score + 0.5;
        copy.raw_similarity = Some(0.999);
        let mut by_file: HashMap<String, RankedFileResult> = HashMap::new();
        by_file.insert(kept.path.clone(), kept.clone());
        by_file.insert(copy_path.clone(), copy);
        collapse_duplicate_files(&conn, &mut by_file);
        assert_eq!(by_file.len(), 1);
        let survivor = by_file.values().next().unwrap();
        assert_eq!(survivor.path, kept.path, "the original survives");
        assert!(
            (survivor.score - kept.score).abs() < 1e-12
                && (survivor.base_score - kept.base_score).abs() < 1e-12
                && survivor.raw_similarity == kept.raw_similarity,
            "the survivor keeps its own score and cosine, not the copy's higher ones"
        );
        assert!(same_file_copy("a/x.md", "b/X.MD"));
        assert!(same_file_copy("a/x.md", "backup/y.md"));
        assert!(same_file_copy(
            "memory-snapshot/-u/MEMORY.md",
            "memory/notes.md"
        ));
        assert!(!same_file_copy("a/x.md", "b/y.md"));
    }

    /// Ranking is reproducible: when candidates tie, the cut and the order follow the chunk
    /// id or the path, never the hash-map iteration order. The path-keyword scan is the case
    /// that showed on the live store (one query's file list changed between two identical
    /// processes because its tied `lexical = 0.4` rows were truncated in hash-map order).
    #[test]
    fn ranking_is_deterministic_across_calls() {
        let store = TestStore::new("deterministic");
        let (root, _proj) = fixture(&store);
        store.track(&root);
        let cfg = store.cfg(&root, &[("embed_backend", "hash")]);
        let embedder = TestEmbedder::new(&cfg, true);
        store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        // "otter" is in the name of five fixture files: every chunk of them ties at
        // lexical 1.0, and keep_top = 2 forces a cut among the ties.
        let first: Vec<i64> = {
            let mut v: Vec<i64> = keyword_path_chunk_scores(&conn, "otter", 2)
                .unwrap()
                .keys()
                .copied()
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(first.len(), 2);
        for _ in 0..6 {
            let mut again: Vec<i64> = keyword_path_chunk_scores(&conn, "otter", 2)
                .unwrap()
                .keys()
                .copied()
                .collect();
            again.sort_unstable();
            assert_eq!(again, first, "the cut among tied rows is stable");
        }
        let order = |q: &str| -> Vec<(String, i64)> {
            rank_files_native_with(&conn, &cfg, q, 20, RankOptions::default())
                .unwrap()
                .into_iter()
                .map(|r| (r.path, r.chunk_id))
                .collect()
        };
        for q in [
            "otter",
            "otter migration handoff next step",
            "docs/sessions",
        ] {
            let a = order(q);
            for _ in 0..4 {
                assert_eq!(order(q), a, "{}", q);
            }
        }
    }

    /// A long `.txt` transcript and a long `.txt` chat dump: whichever chunk a query hits, the
    /// role (record) and the noise flag come from the file's first chunk, so they are the same
    /// for every chunk. Plain `.md` files keep the shape of the retrieved chunk.
    #[test]
    fn txt_role_and_noise_are_stable_across_chunks() {
        let store = TestStore::new("txt-file-shape");
        let root = store.corpus_root("root");
        write(
            &root.join("beta").join("notes.md"),
            "# Beta\n\nUnrelated notes on zebra habitats and quarterly tax filing checklists.\n",
        );
        let proj = root.join("acme");
        write(&proj.join("README.md"), "# Acme\n\nProject notes.\n");
        // Chunks are 1000 characters: the markers sit in the first chunk, the searched-for words
        // ("pelican invoice reconciliation") only in the third.
        let filler =
            "The team walked through the quarterly roadmap and the staffing plan in detail. ";
        let mut transcript = String::from("# Meeting Transcript **Date:** Monday, April 13, 2026\n\n## Summary\nThe team discussed the roadmap.\n\n");
        while transcript.len() < 2300 {
            transcript.push_str(filler);
        }
        transcript.push_str(
            "Then the pelican invoice reconciliation was assigned to the finance lead.\n",
        );
        write(&proj.join("notes").join("roadmap-call.txt"), &transcript);
        let mut dump = String::from("Human: plan the migration\n\nAssistant: The migration moves in three waves.\n\nHuman: and the budget?\n\nAssistant: One budget line per wave.\n\nHuman: ok\n\nAssistant: ");
        while dump.len() < 2300 {
            dump.push_str(filler);
        }
        dump.push_str("The walrus ledger export runs nightly after the close.\n");
        write(&proj.join("exports").join("chat-export.txt"), &dump);
        store.track(&root);
        let cfg = store.cfg(&root, &[("embed_backend", "hash")]);
        let embedder = TestEmbedder::new(&cfg, true);
        store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        let n_chunks = |name: &str| -> i64 {
            conn.query_row(
                "SELECT COUNT(*) FROM project_chunks WHERE doc_rel_path = ?1",
                params![name],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert!(n_chunks("notes/roadmap-call.txt") >= 3);
        assert!(n_chunks("exports/chat-export.txt") >= 3);

        let rows = rank_files_native_with(
            &conn,
            &cfg,
            "pelican invoice reconciliation",
            10,
            RankOptions::default(),
        )
        .unwrap();
        let (_, t) = find(&rows, "roadmap-call.txt").expect("transcript");
        assert!(
            t.chunk_index >= 2,
            "the hit is a later chunk: {}",
            t.chunk_index
        );
        assert_eq!(t.role, "record", "role from the file's first chunk");
        assert!(!t.noise);
        let rows = rank_files_native_with(
            &conn,
            &cfg,
            "walrus ledger export nightly",
            10,
            RankOptions::default(),
        )
        .unwrap();
        let (_, d) = find(&rows, "chat-export.txt").expect("dump");
        assert!(d.chunk_index >= 2, "{}", d.chunk_index);
        assert!(d.noise, "noise from the file's first chunk");
        assert!(d.quality <= 0.36, "{}", d.quality);
        assert_eq!(d.role, "knowledge");
        // Chunk view agrees with the files view.
        let mut quiet = cfg.clone();
        quiet.reranker_enabled = false;
        let chunks =
            rank_chunks_native_with(&conn, &quiet, "walrus ledger export nightly", 10, None)
                .unwrap();
        let c = chunks
            .iter()
            .find(|c| c.path.ends_with("chat-export.txt"))
            .expect("dump chunk");
        assert!(c.noise && c.chunk_index >= 2);
    }

    /// A chunk row with no vector anywhere (found by FTS only) and one whose SQLite vector is
    /// NaN: both carry `raw_similarity = None` and fail any floor above zero, in the files and
    /// the chunks view; a candidate with a real cosine above the floor stays.
    #[test]
    fn raw_floor_fails_closed_on_missing_and_nan_cosines() {
        assert!(passes_raw_floor(None, 0.0));
        assert!(passes_raw_floor(Some(f64::NAN), 0.0), "floor 0 is off");
        assert!(!passes_raw_floor(None, 0.4));
        assert!(!passes_raw_floor(Some(f64::NAN), 0.4));
        assert!(!passes_raw_floor(Some(f64::INFINITY), 0.4));
        assert!(!passes_raw_floor(Some(0.399), 0.4));
        assert!(passes_raw_floor(Some(0.4), 0.4));
        assert!(passes_raw_floor(Some(0.9), 0.4));

        let store = TestStore::new("raw-floor-closed");
        let (root, proj) = fixture(&store);
        store.track(&root);
        let mut cfg = store.cfg(
            &root,
            &[("embed_backend", "hash"), ("reranker_enabled", "false")],
        );
        let embedder = TestEmbedder::new(&cfg, true);
        store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        let project_id: i64 = conn
            .query_row(
                "SELECT id FROM projects WHERE path = ?1",
                params![normalize_path(&proj.to_string_lossy())
                    .to_string_lossy()
                    .to_string()],
                |r| r.get(0),
            )
            .expect("acme project id");
        let now = now_ts();
        let q = "zebra quarterly tax filing checklist";
        let insert_chunk = |name: &str, text: &str| -> i64 {
            let doc_path = proj.join("notes").join(name).to_string_lossy().to_string();
            conn.execute(
                "INSERT INTO project_chunks(project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at) VALUES (?1, ?2, ?3, ?4, 0, 12, ?5, ?6, ?4)",
                params![project_id, doc_path, format!("notes/{}", name), now, format!("h-{}", name), text],
            )
            .expect("insert chunk");
            conn.last_insert_rowid()
        };
        // Found by the keyword search, no vector in SQLite or LanceDB.
        let no_vec = insert_chunk(
            "zebra-no-vector.md",
            "Zebra quarterly tax filing checklist for the zebra habitat, first draft.",
        );
        // A stored vector that is all NaN: the cosine is not finite.
        let nan_vec = insert_chunk(
            "zebra-nan-vector.md",
            "Zebra quarterly tax filing checklist for the zebra habitat, second draft.",
        );
        let model_key = model_key_for_cfg(&cfg);
        let dim = cfg.local_embed_dim.max(1) as usize;
        let nan_blob: Vec<u8> = std::iter::repeat_n(f32::NAN.to_le_bytes(), dim)
            .flatten()
            .collect();
        conn.execute(
            "INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector) VALUES (?1, ?2, ?3, 1.0, ?4)",
            params![nan_vec, model_key, dim as i64, nan_blob],
        )
        .expect("insert nan vector");

        let open = rank_files_native_with(&conn, &cfg, q, 20, RankOptions::default()).unwrap();
        let (_, a) = find(&open, "zebra-no-vector.md").expect("lexical hit without a vector");
        let (_, b) = find(&open, "zebra-nan-vector.md").expect("lexical hit with a NaN vector");
        let (_, beta) = find(&open, "beta/notes.md").expect("the real zebra note");
        assert_eq!(a.raw_similarity, None, "{}", a.path);
        assert_eq!(b.raw_similarity, None, "{}", b.path);
        let beta_raw = beta.raw_similarity.expect("real cosine");
        assert!(beta_raw.is_finite() && beta_raw > 0.0, "{}", beta_raw);

        let floor = beta_raw / 2.0;
        let closed = rank_files_native_with(
            &conn,
            &cfg,
            q,
            20,
            RankOptions {
                min_raw_similarity: floor,
                ..RankOptions::default()
            },
        )
        .unwrap();
        assert!(find(&closed, "beta/notes.md").is_some());
        assert!(
            find(&closed, "zebra-no-vector.md").is_none(),
            "a missing cosine fails the floor: {:?}",
            closed.iter().map(|r| &r.path).collect::<Vec<_>>()
        );
        assert!(find(&closed, "zebra-nan-vector.md").is_none());

        // Chunk view, same floor through `search_min_abs_score`.
        cfg.search_min_abs_score = floor;
        let chunks = rank_chunks_native_with(&conn, &cfg, q, 20, None).unwrap();
        assert!(chunks.iter().any(|c| c.path.ends_with("beta/notes.md")));
        assert!(!chunks.iter().any(|c| c.path.contains("zebra-no-vector")));
        assert!(!chunks.iter().any(|c| c.path.contains("zebra-nan-vector")));
        cfg.search_min_abs_score = 0.0;
        let chunks = rank_chunks_native_with(&conn, &cfg, q, 20, None).unwrap();
        assert!(chunks.iter().any(|c| c.path.contains("zebra-no-vector")));
        let _ = no_vec;
    }

    /// The cosine reported for a hit found through a different vector (HyDE) is recomputed
    /// against the real query vector from the SQLite copy of the chunk vectors, agrees with
    /// LanceDB's cosine for the same query, and is `None` when the vector is gone.
    #[test]
    fn hyde_only_hits_are_rescored_against_the_real_query_vector() {
        let store = TestStore::new("hyde-rescore");
        let (root, _proj) = fixture(&store);
        store.track(&root);
        let cfg = store.cfg(&root, &[("embed_backend", "hash")]);
        let embedder = TestEmbedder::new(&cfg, true);
        store.index(&cfg, &embedder, false).expect("index");
        let conn = store.conn();
        let (model_key, qv) = embed_query_cached(&cfg, "otter migration pricing tiers").unwrap();
        let (_, hyde_vec) =
            embed_query_cached(&cfg, "fn migrate_otters() { let waves = 3; }").unwrap();
        // What the HyDE pass sees: hits scored against the hypothetical text.
        let hyde_hits = with_lance_store(|s| lance_store::search_vectors(s, &hyde_vec, 5)).unwrap();
        assert!(!hyde_hits.is_empty());
        let mut signals = chunk_signals_for_ids(
            &conn,
            &hyde_hits,
            &HashMap::new(),
            &CoverageTerms::from_query(""),
            None,
        )
        .unwrap();
        let ids: Vec<i64> = signals.keys().copied().collect();
        recompute_raw_similarity(&conn, &mut signals, &ids, &model_key, &qv).unwrap();
        let real = hybrid_search_lance(
            &conn,
            &model_key,
            "otter migration pricing tiers",
            &qv,
            50,
            50,
        )
        .unwrap();
        let from_sqlite = cosine_from_sqlite_vectors(&conn, &model_key, &qv, &ids).unwrap();
        let mut differs_from_hyde = 0usize;
        for id in &ids {
            let got = signals[id]
                .raw_similarity
                .expect("cosine to the real query");
            assert!((got - from_sqlite[id]).abs() < 1e-9);
            let lance = real[id].raw_similarity.expect("lance cosine");
            assert!(
                (got - lance).abs() < 1e-5,
                "chunk {}: sqlite {} vs lance {}",
                id,
                got,
                lance
            );
            if (got - hyde_hits[id].raw_similarity).abs() > 1e-6 {
                differs_from_hyde += 1;
            }
        }
        assert!(
            differs_from_hyde > 0,
            "the HyDE cosine is not the query cosine"
        );
        // No stored vector: no cosine, so the floor fails closed.
        let victim = ids[0];
        conn.execute(
            "DELETE FROM project_chunk_vectors WHERE chunk_id = ?1",
            params![victim],
        )
        .unwrap();
        recompute_raw_similarity(&conn, &mut signals, &[victim], &model_key, &qv).unwrap();
        assert_eq!(signals[&victim].raw_similarity, None);
        assert!(!passes_raw_floor(signals[&victim].raw_similarity, 0.01));
    }

    #[test]
    fn noise_is_judged_from_text_and_file_type_not_from_sessions_directories() {
        let prose = "# Handoff\n\nState of the work: the migration plan is drafted and the next step is the pricing review.";
        assert!((content_quality("docs/sessions/HANDOFF-2026-06-10.md", prose) - 1.0).abs() < 1e-9);
        assert!((content_quality("session_notes/x.md", prose) - 1.0).abs() < 1e-9);
        let dump = "Human: fix the tests\n\nAssistant: Running them now.\n\nHuman: ok\n";
        assert!((content_quality("exports/dump.txt", dump) - 0.35).abs() < 1e-9);
        assert!((content_quality("docs/dump.md", dump) - 0.35).abs() < 1e-9);
        let jsonl = "{\"type\":\"user\",\"message\":{\"role\":\"user\"}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\"}}\n";
        assert!((content_quality("logs/session.jsonl", jsonl) - 0.35).abs() < 1e-9);
        assert!((content_quality("logs/run.log", "started\nfinished\n") - 0.60).abs() < 1e-9);
        assert!((content_quality("logs/run.jsonl", "{\"a\": 1}\n") - 0.60).abs() < 1e-9);
        assert!(
            (content_quality(
                "package-lock.json",
                "{\"name\": \"x\", \"lockfileVersion\": 3}"
            ) - 0.35)
                .abs()
                < 1e-9
        );
        assert!(is_noise_artifact("x/package-lock.json", TextShape::Prose));
        assert!(is_noise_artifact("x/app.min.js", TextShape::Prose));
        assert!(is_noise_artifact("x/notes.md", TextShape::ChatDump));
        assert!(!is_noise_artifact(
            "docs/sessions/HANDOFF.md",
            TextShape::Prose
        ));
        assert!(!is_noise_artifact("notes/call.txt", TextShape::Transcript));

        // Path penalties are about directory components, never about the file name.
        assert!((path_noise_penalty("docs/sessions/HANDOFF.md") - 1.0).abs() < 1e-9);
        assert!(
            (path_noise_penalty("memory-snapshot/-Users-x/memory/MEMORY.md") - 0.85).abs() < 1e-9
        );
        assert!((path_noise_penalty("docs/Backups/old.md") - 0.85).abs() < 1e-9);
        assert!((path_noise_penalty("archived/meeting-transcribe/DESIGN.md") - 0.85).abs() < 1e-9);
        assert!((path_noise_penalty("archive/x.md") - 0.85).abs() < 1e-9);
        assert!((path_noise_penalty("snapshot.md") - 1.0).abs() < 1e-9);
        assert!((path_noise_penalty("tmp/x.md") - 0.55).abs() < 1e-9);
        assert!((path_noise_penalty("infra/state/x.tfstate") - 0.72).abs() < 1e-9);
        assert!((path_noise_penalty("data/x.json") - 0.92).abs() < 1e-9);
        assert!((path_noise_penalty("tmp/backup/x.json") - 0.55 * 0.85 * 0.92).abs() < 1e-9);
    }

    #[test]
    fn coverage_terms_mark_strong_lexical_matches() {
        let cover = CoverageTerms::from_query("what do we know about Acme and S3 Tables");
        assert_eq!(cover.names, vec!["acme".to_string(), "tables".to_string()]);
        assert!(cover.terms.contains(&"acme".to_string()));
        assert!(!cover.terms.contains(&"what".to_string()));
        assert!(
            !cover.strong_match("Notes from the Acme briefing on 2026-07-15."),
            "a name alone is a mention"
        );
        assert!(
            cover.strong_match("What we know about the Acme briefing."),
            "a name joined by another distinctive term"
        );
        assert!(!cover.strong_match("Nothing about that customer here."));
        assert!(
            cover.strong_match("acme know tables s3 about"),
            "every term present"
        );
        // Capitalised only because they open the prompt or a sentence: not names.
        assert!(CoverageTerms::from_query("Run the tests").names.is_empty());
        assert!(CoverageTerms::from_query("Read the handoff")
            .names
            .is_empty());
        assert!(CoverageTerms::from_query("Summarize this").names.is_empty());
        assert!(!CoverageTerms::from_query("Run the tests").strong_match("cargo run --release"));
        assert_eq!(
            CoverageTerms::from_query("Read the Acme handoff. Summarize it for Globex").names,
            vec!["acme".to_string(), "globex".to_string()]
        );
        assert_eq!(
            CoverageTerms::from_query("Tell me about Globex\nThen Acme").names,
            vec!["globex".to_string(), "acme".to_string()],
            "a new line starts a sentence"
        );
        assert!(
            CoverageTerms::from_query("what about AWS and EBC")
                .names
                .is_empty(),
            "three letters are not enough"
        );
        assert_eq!(
            CoverageTerms::from_query("please Review the Acme deck").names,
            vec!["acme".to_string()],
            "a prompt verb is not a name even mid-sentence"
        );
        let two = CoverageTerms::from_query("lancedb compaction");
        assert!(two.names.is_empty());
        assert!(two.strong_match("LanceDB compaction runs in the watcher sweep"));
        assert!(
            !two.strong_match("LanceDB versions pile up"),
            "one of two terms"
        );
        let one = CoverageTerms::from_query("compaction");
        assert!(
            !one.strong_match("compaction"),
            "a single common word is not strong evidence"
        );
        assert!(!CoverageTerms::from_query("").strong_match("anything"));
        let lex = CoverageTerms::from_terms(&["S3Tables".to_string(), "cost".to_string()]);
        assert!(lex.strong_match("s3tables cost allocation"));
    }
}
