//! Topic dossier (slice 4): "what do we know about X", answered as a list of projects.
//!
//! One fused file-level retrieval pass (the same ranker `search --view files` and the recall
//! hook use, widened to [`RANKED_POOL`] files, then capped at [`PER_PROJECT_CAP`] files per
//! project and [`CANDIDATES`] in all), grouped by project: for each of the top
//! projects the role-typed best entry file with its date and cosine, the newest evidence
//! date, the number of distinct files (duplicates and superseded handoffs already folded by
//! the ranker), a one-line reason built from the `why` signals and the topic words the entry
//! matched, then the neighbours of the top projects from `project_edges`, and a closing
//! instruction to use `search_files` / `pack_context` for depth. Nothing here calls an LLM
//! or reads files; the grouping is pure over the ranked rows and unit-tested as such.
//!
//! Surfaces: `retrivio dossier <topic>`, the MCP tool `topic_dossier`, and the recall hook's
//! automatic dossier (`recall_dossier = auto`), which reuses the hook's own candidate set.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use rusqlite::Connection;
use serde_json::{json, Value};

use super::{freshness, recall, ConfigValues, RankOptions, RankedFileResult};

/// Files a dossier is built from, after the per-project cap.
pub const CANDIDATES: usize = 60;
/// Files requested from the ranker before the cap: the ranker's own retrieval widths bound
/// the cost, not this number, so asking for more rows is free and lets a topic one project
/// floods (49 of 60 files in a measured run) still show the other projects that hold it.
pub const RANKED_POOL: usize = 240;
/// Files one project may contribute to the candidate set. Breadth is about how many projects
/// hold material, not how many files the largest one has.
pub const PER_PROJECT_CAP: usize = 12;
/// Projects shown by default and at most.
pub const DEFAULT_LIMIT: usize = 6;
pub const MAX_LIMIT: usize = 8;
/// Below this cosine a file never enters a dossier. Measured on the live store (slice 3):
/// unrelated material sits at 0.15-0.26, customers known only through a few transcripts at
/// 0.42-0.49; `search_min_abs_score`, when set, replaces it.
pub const MIN_COSINE: f64 = 0.30;
/// Related projects listed at most.
pub const RELATED_LIMIT: usize = 5;
/// Breadth credit per extra evidence file, capped at five files: a project with several
/// relevant files ranks above one with a single, slightly stronger file.
const BREADTH_CREDIT: f64 = 0.08;

pub const SCHEMA: &str = "topic-dossier-v1";
pub const INSTRUCTION: &str = "For depth, call search_files with a narrower query or pack_context on a project's entry file; read a file before relying on it. Entries marked weak sit under the recall floor.";

/// The best file of a project for the topic, with the same fields every result carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub path: String,
    pub role: &'static str,
    pub content_date: f64,
    pub date_source: &'static str,
    pub age_days: f64,
    pub tier: String,
    pub verify: bool,
    pub noise: bool,
    pub superseded_by: Option<String>,
    pub raw_similarity: Option<f64>,
    pub why: String,
    pub excerpt: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectEntry {
    pub project_path: String,
    pub name: String,
    pub score: f64,
    pub entry: Entry,
    /// Distinct files of the project among the candidates (duplicate copies and superseded
    /// handoffs already folded away by the ranker).
    pub evidence_count: usize,
    /// Newest content date among those files.
    pub evidence_date: f64,
    /// Best cosine of the project's files.
    pub best_cosine: Option<f64>,
    /// Best cosine under the recall floor: shown, but flagged.
    pub weak: bool,
    /// Topic words found in the entry's path or excerpt.
    pub matched: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelatedProject {
    pub project_path: String,
    pub name: String,
    /// Raw edge weight: the centroid cosine for `embedding_similarity`, the mention score for
    /// `semantic_related`.
    pub weight: f64,
    /// Edge kind (`embedding_similarity` first, `semantic_related` as a fallback).
    pub kind: String,
    /// Name of the dossier project it neighbours.
    pub via: String,
}

/// Project breadth of a candidate set for the hook's dossier gate: how many projects hold a
/// file above the dossier floor and the best cosines of the first and third project. The gate
/// (`recall.rs`) fires when three or more projects are present and the third's best cosine is
/// within [`BREADTH_SPREAD`] of the first's: the topic is spread over projects rather than
/// owned by one, whatever the file counts say (one project may hold 49 transcript files about
/// a customer and still not be the only place to look).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Breadth {
    pub projects: usize,
    pub c1: f64,
    pub c3: f64,
}

/// Largest gap between the best and the third-best project cosine that still counts as
/// breadth. Measured on 24 private prompts (slice 4 report): 0.10 separates the multi-project
/// entity prompts (third project within 0.01-0.08 of the first) from single-project topics
/// (third project 0.15-0.19 below).
pub const BREADTH_SPREAD: f64 = 0.10;
/// The best project's cosine may sit this far under the recall floor and still open a dossier
/// (a customer known through scattered notes at 0.42-0.49 while the recall floor was 0.45 in
/// that measurement).
pub const BREADTH_FLOOR_MARGIN: f64 = 0.05;

pub fn breadth(rows: &[RankedFileResult]) -> Breadth {
    let mut best: HashMap<&str, f64> = HashMap::new();
    for r in rows {
        if r.noise {
            continue;
        }
        let Some(c) = r.raw_similarity else {
            continue;
        };
        let e = best.entry(r.project_path.as_str()).or_insert(c);
        if c > *e {
            *e = c;
        }
    }
    let mut cos: Vec<f64> = best.values().copied().collect();
    cos.sort_by(|a, b| b.total_cmp(a));
    Breadth {
        projects: cos.len(),
        c1: cos.first().copied().unwrap_or(0.0),
        c3: cos.get(2).copied().unwrap_or(0.0),
    }
}

/// The breadth half of the hook's dossier gate.
pub fn breadth_fires(b: &Breadth, recall_floor: f64) -> bool {
    b.projects >= 3 && b.c3 >= b.c1 - BREADTH_SPREAD && b.c1 >= recall_floor - BREADTH_FLOOR_MARGIN
}

#[derive(Clone, Debug)]
pub struct Dossier {
    pub topic: String,
    pub projects: Vec<ProjectEntry>,
    pub related: Vec<RelatedProject>,
    /// Files the dossier was built from: above the floor, no noise or superseded rows, at most
    /// [`PER_PROJECT_CAP`] per project, [`CANDIDATES`] in all.
    pub candidates: usize,
    pub floor: f64,
    pub weak_below: f64,
    pub timing_ms: f64,
}

pub fn project_name(project_path: &str) -> String {
    Path::new(project_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| project_path.to_string())
}

/// Topic words worth matching: lowercase tokens of three or more characters that are not
/// stopwords, in prompt order, deduplicated.
pub fn topic_terms(topic: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in topic.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-')) {
        let tok = raw.trim_matches('-').to_lowercase();
        if tok.chars().count() < 3 || recall::is_stopword(&tok) || out.contains(&tok) {
            continue;
        }
        out.push(tok);
    }
    out
}

fn tokens_of(text: &str) -> HashSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches('-').to_lowercase())
        .collect()
}

/// The cosine floor a dossier applies: `search_min_abs_score` when configured, else
/// [`MIN_COSINE`].
pub fn floor_for(cfg: &ConfigValues) -> f64 {
    if cfg.search_min_abs_score > 0.0 {
        cfg.search_min_abs_score
    } else {
        MIN_COSINE
    }
}

/// A dossier's candidate rows out of ranked files (pure): machine artefacts and the older
/// members of a handoff series (`superseded_by` set) are dropped, each project keeps at most
/// `cap` of its best files, and the first `limit` rows survive. The input order (score) is
/// kept. Both the CLI/MCP dossier ([`build`]) and the hook's gate go through this, so a
/// project's file count, breadth and entry are read off the same rows.
pub fn candidates(rows: &[RankedFileResult], cap: usize, limit: usize) -> Vec<RankedFileResult> {
    let mut per_project: HashMap<&str, usize> = HashMap::new();
    let mut out: Vec<RankedFileResult> = Vec::new();
    for row in rows {
        if out.len() >= limit {
            break;
        }
        if row.noise || row.superseded_by.is_some() {
            continue;
        }
        let n = per_project.entry(row.project_path.as_str()).or_insert(0);
        if *n >= cap.max(1) {
            continue;
        }
        *n += 1;
        out.push(row.clone());
    }
    out
}

/// Group ranked files by project (pure). `weak_below` is the recall floor: a project whose
/// best cosine is under it is kept but flagged `weak`. Noise files and superseded series
/// members are left out before any project metric (entry, best cosine, evidence date, file
/// count), so a project known only through a superseded handoff is not listed.
pub fn group_projects(
    rows: &[RankedFileResult],
    topic: &str,
    weak_below: f64,
    limit: usize,
) -> Vec<ProjectEntry> {
    let terms = topic_terms(topic);
    let mut by_project: HashMap<String, Vec<&RankedFileResult>> = HashMap::new();
    for row in rows {
        if row.noise || row.superseded_by.is_some() {
            continue;
        }
        by_project
            .entry(row.project_path.clone())
            .or_default()
            .push(row);
    }
    let mut out: Vec<ProjectEntry> = Vec::new();
    for (project_path, mut files) in by_project {
        files.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.path.cmp(&b.path))
        });
        let best = files[0];
        let evidence_count = files.len();
        let evidence_date = files
            .iter()
            .map(|f| f.content_date)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_cosine = files
            .iter()
            .filter_map(|f| f.raw_similarity)
            .fold(None, |acc: Option<f64>, c| {
                Some(acc.map_or(c, |a| a.max(c)))
            });
        let weak = best_cosine.map(|c| c < weak_below).unwrap_or(false);
        let haystack: HashSet<String> = {
            let mut h = tokens_of(&best.doc_rel_path);
            h.extend(tokens_of(&best.excerpt));
            h
        };
        let matched: Vec<String> = terms
            .iter()
            .filter(|t| haystack.contains(*t))
            .cloned()
            .collect();
        let score = best.score * (1.0 + BREADTH_CREDIT * ((evidence_count - 1).min(5) as f64));
        let mut reason = format!(
            "{} file{}",
            evidence_count,
            if evidence_count == 1 { "" } else { "s" }
        );
        if matched.is_empty() {
            reason.push_str("; no topic word in the entry");
        } else {
            reason.push_str("; matches ");
            reason.push_str(&matched.join(", "));
        }
        if !best.why.is_empty() {
            reason.push_str("; ");
            reason.push_str(&best.why);
        }
        if weak {
            reason.push_str("; weak");
        }
        let name = project_name(&project_path);
        out.push(ProjectEntry {
            project_path,
            name,
            score,
            entry: Entry {
                path: best.path.clone(),
                role: best.role,
                content_date: best.content_date,
                date_source: best.date_source,
                age_days: best.age_days,
                tier: best.freshness_tier.clone(),
                verify: best.verify,
                noise: best.noise,
                superseded_by: best.superseded_by.clone(),
                raw_similarity: best.raw_similarity,
                why: best.why.clone(),
                excerpt: best.excerpt.clone(),
            },
            evidence_count,
            evidence_date,
            best_cosine,
            weak,
            matched,
            reason,
        });
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.evidence_count.cmp(&a.evidence_count))
            .then_with(|| a.project_path.cmp(&b.project_path))
    });
    out.truncate(limit.clamp(1, MAX_LIMIT));
    out
}

/// Edges of one project in both directions with their raw weights: (other path, kind, weight).
fn project_edges_raw(conn: &Connection, path: &str) -> Result<Vec<(String, String, f64)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, kind, MAX(weight) AS w
FROM (
    SELECT pe.dst AS path, pe.kind AS kind, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE src.path = ?1
    UNION ALL
    SELECT src.path AS path, pe.kind AS kind, pe.weight AS weight
    FROM project_edges pe
    JOIN projects src ON src.id = pe.src_project_id
    WHERE pe.dst = ?1
)
GROUP BY path, kind
"#,
        )
        .map_err(|e| format!("failed preparing project edge query: {}", e))?;
    let rows = stmt
        .query_map(rusqlite::params![path], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })
        .map_err(|e| format!("failed querying project edges: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project edge row: {}", e))?);
    }
    Ok(out)
}

fn kind_rank(kind: &str) -> u8 {
    match kind {
        "embedding_similarity" => 0,
        "imports_from" => 1,
        _ => 2,
    }
}

/// Neighbours of the top three dossier projects in `project_edges`, excluding the dossier's
/// own projects and the tracked roots' "root files" projects. Raw weights, never normalised:
/// `embedding_similarity` edges (the cosine between the two projects' centroid vectors, kept
/// from 0.40 up) rank first, then import edges, then the name-mention `semantic_related`
/// edges, which are a weak heuristic and only fill the list when nothing better exists.
pub fn related_projects(
    conn: &Connection,
    projects: &[ProjectEntry],
    limit: usize,
) -> Result<Vec<RelatedProject>, String> {
    if projects.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let listed: HashSet<&str> = projects.iter().map(|p| p.project_path.as_str()).collect();
    let roots: HashSet<String> = super::list_tracked_roots_conn(conn)
        .unwrap_or_default()
        .into_iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let mut best: HashMap<String, (u8, f64, String, String)> = HashMap::new();
    for p in projects.iter().take(3) {
        for (dst, kind, w) in project_edges_raw(conn, &p.project_path)? {
            if listed.contains(dst.as_str()) || roots.contains(&dst) {
                continue;
            }
            let rank = kind_rank(&kind);
            let candidate = (rank, w, kind, p.name.clone());
            match best.get(&dst) {
                Some((r, cw, _, _)) if (*r, -*cw) <= (rank, -w) => {}
                _ => {
                    best.insert(dst, candidate);
                }
            }
        }
    }
    let mut out: Vec<RelatedProject> = best
        .into_iter()
        .map(|(project_path, (_, weight, kind, via))| RelatedProject {
            name: project_name(&project_path),
            project_path,
            weight,
            kind,
            via,
        })
        .collect();
    out.sort_by(|a, b| {
        kind_rank(&a.kind)
            .cmp(&kind_rank(&b.kind))
            .then_with(|| b.weight.total_cmp(&a.weight))
            .then_with(|| a.project_path.cmp(&b.project_path))
    });
    out.truncate(limit);
    Ok(out)
}

/// The explicit dossier: one retrieval pass, grouping, neighbours.
pub fn build(
    conn: &Connection,
    cfg: &ConfigValues,
    topic: &str,
    limit: usize,
) -> Result<Dossier, String> {
    let started = Instant::now();
    let topic = topic.trim();
    if topic.is_empty() {
        return Err("topic must be non-empty".to_string());
    }
    let floor = floor_for(cfg);
    let ranked = super::rank_files_native_with(
        conn,
        cfg,
        topic,
        RANKED_POOL,
        RankOptions {
            min_raw_similarity: floor,
            ..RankOptions::default()
        },
    )?;
    let rows = candidates(&ranked, PER_PROJECT_CAP, CANDIDATES);
    let projects = group_projects(&rows, topic, cfg.recall_min_abs_score, limit);
    let related = related_projects(conn, &projects, RELATED_LIMIT)?;
    Ok(Dossier {
        topic: topic.to_string(),
        projects,
        related,
        candidates: rows.len(),
        floor,
        weak_below: cfg.recall_min_abs_score,
        timing_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn entry_json(e: &Entry) -> Value {
    json!({
        "path": e.path,
        "role": e.role,
        "content_date": e.content_date,
        "content_date_ymd": freshness::format_ymd(e.content_date),
        "date_source": e.date_source,
        "date_basis": freshness::date_basis(e.date_source),
        "age_days": e.age_days,
        "freshness_tier": e.tier,
        "verify": e.verify,
        "noise": e.noise,
        "superseded_by": e.superseded_by,
        "raw_similarity": e.raw_similarity,
        "why": e.why,
        "excerpt": e.excerpt,
    })
}

pub fn to_json(d: &Dossier) -> Value {
    let projects: Vec<Value> = d
        .projects
        .iter()
        .map(|p| {
            json!({
                "project_path": p.project_path,
                "name": p.name,
                "score": p.score,
                "evidence_count": p.evidence_count,
                "evidence_date": p.evidence_date,
                "evidence_date_ymd": freshness::format_ymd(p.evidence_date),
                "best_cosine": p.best_cosine,
                "weak": p.weak,
                "matched": p.matched,
                "reason": p.reason,
                "entry": entry_json(&p.entry),
            })
        })
        .collect();
    let related: Vec<Value> = d
        .related
        .iter()
        .map(|r| {
            json!({
                "project_path": r.project_path,
                "name": r.name,
                "weight": r.weight,
                "kind": r.kind,
                "via": r.via,
            })
        })
        .collect();
    json!({
        "schema": SCHEMA,
        "topic": d.topic,
        "count": d.projects.len(),
        "candidates": d.candidates,
        "floor": d.floor,
        "weak_below": d.weak_below,
        "projects": projects,
        "related_projects": related,
        "instruction": INSTRUCTION,
        "timing_ms": d.timing_ms,
    })
}

/// One line per project: rank, name, entry path, date, role and age (plus `verify`), file
/// count, reason. `compact` drops the reason's signal part for the hook block.
pub fn project_line(n: usize, p: &ProjectEntry, compact: bool) -> String {
    let mut label = format!(
        "{} · {}d",
        p.entry.role,
        p.entry.age_days.max(0.0).floor() as i64
    );
    if matches!(p.entry.tier.as_str(), "verify" | "stale") {
        label.push_str(" · ");
        label.push_str(&p.entry.tier);
    }
    let files = format!(
        "{} file{}",
        p.evidence_count,
        if p.evidence_count == 1 { "" } else { "s" }
    );
    if compact {
        let mut line = format!(
            "{}. {} — {} — {} ({}) — {}",
            n,
            p.name,
            p.entry.path,
            freshness::format_ymd(p.entry.content_date),
            label,
            files
        );
        if !p.matched.is_empty() {
            line.push_str(&format!(" — matches {}", p.matched.join(", ")));
        }
        if p.weak {
            line.push_str(" — weak");
        }
        line
    } else {
        format!(
            "{}. {} — {} — {} ({}) — {}",
            n,
            p.name,
            p.entry.path,
            freshness::format_ymd(p.entry.content_date),
            label,
            p.reason
        )
    }
}

pub fn related_line(related: &[RelatedProject]) -> Option<String> {
    if related.is_empty() {
        return None;
    }
    let items: Vec<String> = related
        .iter()
        .map(|r| format!("{} ({} {:.2} via {})", r.name, r.kind, r.weight, r.via))
        .collect();
    Some(format!("Related projects: {}", items.join(", ")))
}

/// The CLI rendering.
pub fn render_text(d: &Dossier) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "Topic dossier: {} — {} project{} from {} file{} (cosine >= {:.2}; weak under {:.2}; {:.0} ms)",
        d.topic,
        d.projects.len(),
        if d.projects.len() == 1 { "" } else { "s" },
        d.candidates,
        if d.candidates == 1 { "" } else { "s" },
        d.floor,
        d.weak_below,
        d.timing_ms
    ));
    if d.projects.is_empty() {
        lines.push("No project holds material about this topic above the floor.".to_string());
    }
    for (i, p) in d.projects.iter().enumerate() {
        lines.push(project_line(i + 1, p, false));
    }
    if let Some(r) = related_line(&d.related) {
        lines.push(r);
    }
    lines.push(INSTRUCTION.to_string());
    lines.join("\n")
}

/// Synthetic ranked file for the tests here and in `recall`: cosine `cos`, the given role,
/// not noise, not superseded, dated by mtime three days ago.
#[cfg(test)]
pub(crate) fn test_row(
    path: &str,
    project: &str,
    score: f64,
    cos: f64,
    role: &'static str,
) -> RankedFileResult {
    let rel = path
        .strip_prefix(project)
        .map(|r| r.trim_start_matches('/').to_string())
        .unwrap_or_else(|| path.to_string());
    RankedFileResult {
        path: path.to_string(),
        project_path: project.to_string(),
        doc_rel_path: rel,
        chunk_id: 1,
        chunk_index: 0,
        score,
        base_score: score,
        semantic: 1.0,
        lexical: 0.0,
        graph: 0.0,
        relation: "direct".to_string(),
        quality: 1.0,
        excerpt: "Acme pricing notes from the workshop".to_string(),
        evidence: Vec::new(),
        doc_mtime: 1_800_000_000.0,
        content_date: 1_800_000_000.0,
        date_source: "mtime",
        age_days: 3.0,
        freshness_tier: "fresh".to_string(),
        is_record: role == "record",
        role,
        verify: false,
        noise: false,
        raw_similarity: Some(cos),
        superseded_by: None,
        why: format!("semantic:{:.2}", cos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        path: &str,
        project: &str,
        score: f64,
        cos: f64,
        role: &'static str,
    ) -> RankedFileResult {
        super::test_row(path, project, score, cos, role)
    }

    #[test]
    fn topic_terms_drop_stopwords_and_short_tokens() {
        assert_eq!(
            topic_terms("what do we know about Acme pricing"),
            vec!["know", "acme", "pricing"]
        );
        assert!(topic_terms("the and of").is_empty());
    }

    #[test]
    fn projects_group_rank_by_score_with_breadth_credit_and_flag_weak_ones() {
        let mut rows = vec![
            row("/r/a/notes.md", "/r/a", 0.90, 0.62, "knowledge"),
            row(
                "/r/b/docs/sessions/HANDOFF-2026-09-01.md",
                "/r/b",
                0.88,
                0.58,
                "state",
            ),
            row(
                "/r/b/docs/sessions/HANDOFF-2026-08-01.md",
                "/r/b",
                0.80,
                0.55,
                "state",
            ),
            row("/r/b/spec.md", "/r/b", 0.70, 0.50, "knowledge"),
            row("/r/c/transcripts/call.txt", "/r/c", 0.60, 0.41, "record"),
        ];
        rows[2].superseded_by = Some(rows[1].path.clone());
        let mut dump = row("/r/d/chat.txt", "/r/d", 0.99, 0.9, "knowledge");
        dump.noise = true;
        rows.push(dump);

        // A project known only through a superseded handoff is not listed, and a superseded
        // member with the best score never becomes another project's entry.
        let mut only_old = row(
            "/r/e/docs/sessions/HANDOFF-2026-07-01.md",
            "/r/e",
            0.95,
            0.7,
            "state",
        );
        only_old.superseded_by = Some("/r/e/docs/sessions/HANDOFF-2026-09-01.md".to_string());
        rows.push(only_old);
        let mut old_best = row("/r/a/HANDOFF-2026-01-01.md", "/r/a", 0.99, 0.9, "state");
        old_best.superseded_by = Some("/r/a/HANDOFF-2026-09-01.md".to_string());
        rows.push(old_best);

        let projects = group_projects(&rows, "what do we know about acme pricing", 0.45, 8);
        assert_eq!(
            projects.len(),
            3,
            "the noise file's project and the superseded-only project are not listed"
        );
        assert!(projects.iter().all(|p| p.entry.superseded_by.is_none()));
        assert!(projects
            .iter()
            .all(|p| p.best_cosine.unwrap() < 0.9 && p.evidence_date <= 1_800_000_000.0));
        // b has two distinct files (the superseded handoff is folded) and the breadth credit
        // lifts it above a's single, slightly stronger file.
        assert_eq!(projects[0].name, "b");
        assert_eq!(projects[0].evidence_count, 2);
        assert_eq!(projects[0].entry.role, "state");
        assert!((projects[0].score - 0.88 * 1.08).abs() < 1e-9);
        assert_eq!(projects[1].name, "a");
        assert_eq!(projects[1].evidence_count, 1);
        assert_eq!(projects[1].matched, vec!["acme", "pricing"]);
        assert!(projects[1]
            .reason
            .starts_with("1 file; matches acme, pricing; semantic:0.62"));
        assert!(!projects[1].weak);
        assert_eq!(projects[2].name, "c");
        assert!(projects[2].weak, "0.41 is under the 0.45 recall floor");
        assert!(projects[2].reason.ends_with("; weak"));
        assert_eq!(projects[2].entry.role, "record");
        assert_eq!(
            projects[1].entry.path, "/r/a/notes.md",
            "not the superseded handoff"
        );
        // The limit caps the list and never exceeds MAX_LIMIT.
        assert_eq!(group_projects(&rows, "acme", 0.45, 1).len(), 1);
        assert_eq!(group_projects(&rows, "acme", 0.45, 99).len(), 3);
        // The candidate set: noise and superseded rows out, at most `cap` files per project
        // (the best ones, input order is score order), `limit` rows in all.
        let flood: Vec<RankedFileResult> = (0..30)
            .map(|i| {
                row(
                    &format!("/r/big/f{}.md", i),
                    "/r/big",
                    0.9 - i as f64 * 0.01,
                    0.6,
                    "knowledge",
                )
            })
            .chain(rows.iter().cloned())
            .collect();
        let capped = candidates(&flood, 12, 60);
        assert_eq!(
            capped.iter().filter(|r| r.project_path == "/r/big").count(),
            12
        );
        assert!(capped.iter().all(|r| !r.noise && r.superseded_by.is_none()));
        assert_eq!(
            capped.iter().filter(|r| r.project_path == "/r/b").count(),
            2
        );
        assert_eq!(candidates(&flood, 12, 5).len(), 5);
        assert_eq!(
            candidates(&flood, 1, 60)
                .iter()
                .filter(|r| r.project_path == "/r/big")
                .count(),
            1
        );
        // Rendering: role and age as two fields, the file count, the reason.
        let line = project_line(1, &projects[0], false);
        assert!(line.starts_with("1. b — /r/b/docs/sessions/HANDOFF-2026-09-01.md — "));
        assert!(line.contains("(state · 3d) — 2 files; matches acme, pricing; semantic:0.58"));
        let compact = project_line(3, &projects[2], true);
        assert!(compact.ends_with("(record · 3d) — 1 file — matches acme, pricing — weak"));
        let json = to_json(&Dossier {
            topic: "acme".to_string(),
            projects: projects.clone(),
            related: vec![RelatedProject {
                project_path: "/r/z".to_string(),
                name: "z".to_string(),
                weight: 0.7,
                kind: "embedding_similarity".to_string(),
                via: "b".to_string(),
            }],
            candidates: rows.len(),
            floor: 0.30,
            weak_below: 0.45,
            timing_ms: 12.0,
        });
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["count"], 3);
        assert_eq!(json["projects"][0]["entry"]["date_basis"], "mtime");
        assert_eq!(json["projects"][2]["weak"], true);
        assert_eq!(json["related_projects"][0]["via"], "b");
        assert_eq!(json["instruction"], INSTRUCTION);
        assert!(
            related_line(&[]).is_none(),
            "no related line without neighbours"
        );
    }

    #[test]
    fn breadth_counts_projects_and_compares_first_and_third_cosines() {
        // Five projects within 0.05 of each other (a customer spread over folders): fires,
        // even with the best cosine under the recall floor by less than the margin.
        let spread = vec![
            row("/r/a/1.md", "/r/a", 0.9, 0.462, "knowledge"),
            row("/r/a/2.md", "/r/a", 0.8, 0.40, "knowledge"),
            row("/r/b/1.md", "/r/b", 0.8, 0.421, "knowledge"),
            row("/r/c/1.md", "/r/c", 0.7, 0.449, "knowledge"),
            row("/r/d/1.md", "/r/d", 0.6, 0.447, "record"),
            row("/r/e/1.md", "/r/e", 0.5, 0.416, "record"),
        ];
        let b = breadth(&spread);
        assert_eq!(b.projects, 5);
        assert!((b.c1 - 0.462).abs() < 1e-9 && (b.c3 - 0.447).abs() < 1e-9);
        assert!(breadth_fires(&b, 0.45));
        assert!(!breadth_fires(&b, 0.55), "too far under the floor");
        // One project owns the topic: the third project is far below.
        let owned = vec![
            row("/r/a/1.md", "/r/a", 0.9, 0.607, "knowledge"),
            row("/r/b/1.md", "/r/b", 0.8, 0.439, "knowledge"),
            row("/r/c/1.md", "/r/c", 0.7, 0.417, "knowledge"),
            row("/r/d/1.md", "/r/d", 0.6, 0.306, "knowledge"),
        ];
        let b = breadth(&owned);
        assert_eq!((b.projects, b.c1, b.c3), (4, 0.607, 0.417));
        assert!(!breadth_fires(&b, 0.45));
        // Two projects are not breadth; noise and cosine-less rows do not count.
        let mut two = vec![
            row("/r/a/1.md", "/r/a", 0.9, 0.7, "knowledge"),
            row("/r/b/1.md", "/r/b", 0.8, 0.69, "knowledge"),
        ];
        let mut noisy = row("/r/c/1.md", "/r/c", 0.8, 0.69, "knowledge");
        noisy.noise = true;
        two.push(noisy);
        let mut lexical = row("/r/d/1.md", "/r/d", 0.8, 0.0, "knowledge");
        lexical.raw_similarity = None;
        two.push(lexical);
        let b = breadth(&two);
        assert_eq!(b.projects, 2);
        assert!(!breadth_fires(&b, 0.45));
        assert_eq!(breadth(&[]), Breadth::default());
    }
}
