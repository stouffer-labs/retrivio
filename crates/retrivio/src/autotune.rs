//! The autotune command: selection-event examples, candidate MRR evaluation, the recommendation and its report.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::{env, fs, process};

use rusqlite::{params, Connection};
use serde_json::Value;

use crate::config::{
    config_path, config_rows, config_value_string, data_dir, db_path, load_config_values,
    write_config_file, ConfigValues,
};
use crate::db::{ensure_db_schema, list_project_paths, open_db_rw};
use crate::rank::rank_projects_native;
use crate::util::{arg_value, normalize_path, now_ts, yes_no};

#[derive(Clone, Debug)]
pub(crate) struct AutotuneOptions {
    dry_run: bool,
    deep: bool,
    max_events: usize,
    limit: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct AutotuneExample {
    query: String,
    path: String,
    weight: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct AutotuneOutcome {
    pub(crate) cfg: ConfigValues,
    pub(crate) examples_used: usize,
    pub(crate) baseline_mrr: f64,
    pub(crate) baseline_hit1: f64,
    pub(crate) baseline_hit3: f64,
    pub(crate) best_mrr: f64,
    pub(crate) best_hit1: f64,
    pub(crate) best_hit3: f64,
    pub(crate) candidates_tested: usize,
    pub(crate) used_history: bool,
}

pub(crate) fn parse_autotune_options(args: &[OsString]) -> AutotuneOptions {
    let mut dry_run = false;
    let mut deep = false;
    let mut max_events = 320usize;
    let mut limit = 40usize;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--dry-run" => {
                dry_run = true;
            }
            "--deep" => {
                deep = true;
            }
            "--max-events" => {
                i += 1;
                let raw = arg_value(args, i, "--max-events");
                max_events = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-events must be an integer");
                    process::exit(2);
                });
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--max-events=") => {
                let raw = other.trim_start_matches("--max-events=");
                max_events = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-events must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--limit=") => {
                let raw = other.trim_start_matches("--limit=");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown autotune option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }
    AutotuneOptions {
        dry_run,
        deep,
        max_events: max_events.clamp(20, 5000),
        limit: limit.clamp(5, 120),
    }
}

pub(crate) fn event_path_to_project(raw: &str, known_projects: &HashSet<String>) -> Option<String> {
    let mut cur = normalize_path(raw);
    loop {
        let candidate = cur.to_string_lossy().to_string();
        if known_projects.contains(&candidate) {
            return Some(candidate);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

pub(crate) fn load_autotune_examples(
    conn: &Connection,
    max_events: usize,
    known_projects: &HashSet<String>,
) -> Result<Vec<AutotuneExample>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT query, path, COUNT(*) AS c, MAX(selected_at) AS last_seen
FROM selection_events
WHERE length(trim(query)) > 0 AND length(trim(path)) > 0
GROUP BY query, path
ORDER BY last_seen DESC
LIMIT ?1
"#,
        )
        .map_err(|e| format!("failed preparing autotune event query: {}", e))?;
    let rows = stmt
        .query_map(params![max_events as i64], |row| {
            let query: String = row.get(0)?;
            let path: String = row.get(1)?;
            let count: i64 = row.get(2)?;
            Ok((query, path, count))
        })
        .map_err(|e| format!("failed querying autotune events: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        let (query, path, count) =
            row.map_err(|e| format!("failed reading autotune event row: {}", e))?;
        let Some(project_path) = event_path_to_project(&path, known_projects) else {
            continue;
        };
        out.push(AutotuneExample {
            query,
            path: project_path,
            weight: (count.max(1) as f64).sqrt(),
        });
    }
    Ok(out)
}

pub(crate) fn evaluate_candidate_mrr(
    conn: &Connection,
    cfg: &ConfigValues,
    examples: &[AutotuneExample],
    limit: usize,
) -> Result<(f64, f64, f64), String> {
    if examples.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }
    let mut grouped: HashMap<String, Vec<&AutotuneExample>> = HashMap::new();
    for ex in examples {
        grouped.entry(ex.query.clone()).or_default().push(ex);
    }
    let mut ranks_by_query: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for query in grouped.keys() {
        let rows = rank_projects_native(conn, cfg, query, limit)?;
        let mut rank_map: HashMap<String, usize> = HashMap::new();
        for (idx, row) in rows.iter().enumerate() {
            rank_map.insert(row.path.clone(), idx + 1);
        }
        ranks_by_query.insert(query.clone(), rank_map);
    }
    let mut total_weight = 0.0f64;
    let mut mrr_sum = 0.0f64;
    let mut hit1_sum = 0.0f64;
    let mut hit3_sum = 0.0f64;
    for ex in examples {
        total_weight += ex.weight;
        let rank = ranks_by_query
            .get(&ex.query)
            .and_then(|m| m.get(&ex.path))
            .copied();
        if let Some(r) = rank {
            mrr_sum += ex.weight * (1.0 / r as f64);
            if r == 1 {
                hit1_sum += ex.weight;
            }
            if r <= 3 {
                hit3_sum += ex.weight;
            }
        }
    }
    if total_weight <= 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    Ok((
        mrr_sum / total_weight,
        hit1_sum / total_weight,
        hit3_sum / total_weight,
    ))
}

pub(crate) fn unique_i64(values: Vec<i64>) -> Vec<i64> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for v in values {
        if seen.insert(v) {
            out.push(v);
        }
    }
    out
}

pub(crate) fn unique_f64(values: Vec<f64>) -> Vec<f64> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for v in values {
        let norm = (v * 10000.0).round() / 10000.0;
        let key = format!("{:.4}", norm);
        if seen.insert(key) {
            out.push(norm);
        }
    }
    out
}

pub(crate) fn normalize3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let av = a.max(0.0);
    let bv = b.max(0.0);
    let cv = c.max(0.0);
    let sum = av + bv + cv;
    if sum <= 0.0 {
        return (0.66, 0.24, 0.10);
    }
    (av / sum, bv / sum, cv / sum)
}

pub(crate) fn normalize5(vals: [f64; 5]) -> [f64; 5] {
    let mut v = vals;
    for item in &mut v {
        *item = item.max(0.0);
    }
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return [0.58, 0.14, 0.10, 0.10, 0.08];
    }
    [v[0] / sum, v[1] / sum, v[2] / sum, v[3] / sum, v[4] / sum]
}

pub(crate) fn autotune_key(cfg: &ConfigValues) -> String {
    format!(
        "{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{:.4}|{}|{}|{}|{}",
        cfg.rank_chunk_semantic_weight,
        cfg.rank_chunk_lexical_weight,
        cfg.rank_chunk_graph_weight,
        cfg.rank_project_content_weight,
        cfg.rank_project_semantic_weight,
        cfg.rank_project_path_weight,
        cfg.rank_project_graph_weight,
        cfg.rank_project_frecency_weight,
        cfg.graph_same_project_high,
        cfg.graph_same_project_low,
        cfg.graph_seed_limit,
        cfg.graph_neighbor_limit,
        cfg.lexical_candidates,
        cfg.vector_candidates
    )
}

#[allow(clippy::too_many_arguments)] // autotune search state: the candidate plus the running best and its metrics
pub(crate) fn maybe_promote_candidate(
    conn: &Connection,
    examples: &[AutotuneExample],
    limit: usize,
    candidate: ConfigValues,
    tested: &mut usize,
    seen: &mut HashSet<String>,
    best_cfg: &mut ConfigValues,
    best_metrics: &mut (f64, f64, f64),
) -> Result<(), String> {
    let key = autotune_key(&candidate);
    if !seen.insert(key) {
        return Ok(());
    }
    let metrics = evaluate_candidate_mrr(conn, &candidate, examples, limit)?;
    *tested += 1;
    if metrics.0 > best_metrics.0 + 1e-9
        || ((metrics.0 - best_metrics.0).abs() <= 1e-9 && metrics.1 > best_metrics.1 + 1e-9)
    {
        *best_cfg = candidate;
        *best_metrics = metrics;
    }
    Ok(())
}

pub(crate) fn autotune_recommendation(
    conn: &Connection,
    base_cfg: &ConfigValues,
    max_events: usize,
    limit: usize,
    deep: bool,
) -> Result<AutotuneOutcome, String> {
    let known_projects: HashSet<String> = list_project_paths(conn)?.into_iter().collect();
    let mut examples = load_autotune_examples(conn, max_events, &known_projects)?;
    if examples.len() > 120 {
        examples.truncate(120);
    }
    let used_history = examples.len() >= 6;
    let baseline = evaluate_candidate_mrr(conn, base_cfg, &examples, limit)?;

    if !used_history {
        let mut heuristic = base_cfg.clone();
        let project_count = known_projects.len() as i64;
        heuristic.graph_seed_limit = if project_count < 80 { 8 } else { 12 };
        heuristic.graph_neighbor_limit = (project_count / 2).clamp(80, 180);
        heuristic.rank_chunk_graph_weight = if project_count > 200 { 0.12 } else { 0.10 };
        heuristic.rank_project_graph_weight = if project_count > 200 { 0.12 } else { 0.10 };
        let (cs, cl, cg) = normalize3(
            heuristic.rank_chunk_semantic_weight,
            heuristic.rank_chunk_lexical_weight,
            heuristic.rank_chunk_graph_weight,
        );
        heuristic.rank_chunk_semantic_weight = cs;
        heuristic.rank_chunk_lexical_weight = cl;
        heuristic.rank_chunk_graph_weight = cg;
        let p = normalize5([
            heuristic.rank_project_content_weight,
            heuristic.rank_project_semantic_weight,
            heuristic.rank_project_path_weight,
            heuristic.rank_project_graph_weight,
            heuristic.rank_project_frecency_weight,
        ]);
        heuristic.rank_project_content_weight = p[0];
        heuristic.rank_project_semantic_weight = p[1];
        heuristic.rank_project_path_weight = p[2];
        heuristic.rank_project_graph_weight = p[3];
        heuristic.rank_project_frecency_weight = p[4];
        let heuristic_metrics = evaluate_candidate_mrr(conn, &heuristic, &examples, limit)?;
        return Ok(AutotuneOutcome {
            cfg: heuristic,
            examples_used: examples.len(),
            baseline_mrr: baseline.0,
            baseline_hit1: baseline.1,
            baseline_hit3: baseline.2,
            best_mrr: heuristic_metrics.0,
            best_hit1: heuristic_metrics.1,
            best_hit3: heuristic_metrics.2,
            candidates_tested: 1,
            used_history: false,
        });
    }

    let mut best_cfg = base_cfg.clone();
    let mut best_metrics = baseline;
    let mut tested = 0usize;
    let mut seen = HashSet::new();
    seen.insert(autotune_key(&best_cfg));
    let rounds = if deep { 2 } else { 1 };

    for _round in 0..rounds {
        let chunk_graph_vals = unique_f64(vec![
            (best_cfg.rank_chunk_graph_weight * 0.75).clamp(0.02, 0.35),
            best_cfg.rank_chunk_graph_weight.clamp(0.02, 0.35),
            (best_cfg.rank_chunk_graph_weight * 1.25).clamp(0.02, 0.35),
        ]);
        let project_graph_vals = unique_f64(vec![
            (best_cfg.rank_project_graph_weight * 0.75).clamp(0.02, 0.45),
            best_cfg.rank_project_graph_weight.clamp(0.02, 0.45),
            (best_cfg.rank_project_graph_weight * 1.25).clamp(0.02, 0.45),
        ]);
        let seed_vals = unique_i64(vec![
            (best_cfg.graph_seed_limit - 2).clamp(2, 64),
            best_cfg.graph_seed_limit.clamp(2, 64),
            (best_cfg.graph_seed_limit + 2).clamp(2, 64),
        ]);
        let neighbor_vals = unique_i64(vec![
            (best_cfg.graph_neighbor_limit - 20).clamp(8, 500),
            best_cfg.graph_neighbor_limit.clamp(8, 500),
            (best_cfg.graph_neighbor_limit + 20).clamp(8, 500),
        ]);

        for chunk_graph in &chunk_graph_vals {
            for project_graph in &project_graph_vals {
                for seed in &seed_vals {
                    for neighbor in &neighbor_vals {
                        let mut candidate = best_cfg.clone();
                        candidate.rank_chunk_graph_weight = *chunk_graph;
                        candidate.rank_project_graph_weight = *project_graph;
                        candidate.graph_seed_limit = *seed;
                        candidate.graph_neighbor_limit = *neighbor;
                        let (cs, cl, cg) = normalize3(
                            candidate.rank_chunk_semantic_weight,
                            candidate.rank_chunk_lexical_weight,
                            candidate.rank_chunk_graph_weight,
                        );
                        candidate.rank_chunk_semantic_weight = cs;
                        candidate.rank_chunk_lexical_weight = cl;
                        candidate.rank_chunk_graph_weight = cg;
                        let p = normalize5([
                            candidate.rank_project_content_weight,
                            candidate.rank_project_semantic_weight,
                            candidate.rank_project_path_weight,
                            candidate.rank_project_graph_weight,
                            candidate.rank_project_frecency_weight,
                        ]);
                        candidate.rank_project_content_weight = p[0];
                        candidate.rank_project_semantic_weight = p[1];
                        candidate.rank_project_path_weight = p[2];
                        candidate.rank_project_graph_weight = p[3];
                        candidate.rank_project_frecency_weight = p[4];
                        maybe_promote_candidate(
                            conn,
                            &examples,
                            limit,
                            candidate,
                            &mut tested,
                            &mut seen,
                            &mut best_cfg,
                            &mut best_metrics,
                        )?;
                    }
                }
            }
        }

        let same_high_vals = unique_f64(vec![
            (best_cfg.graph_same_project_high - 0.06).clamp(0.20, 0.98),
            best_cfg.graph_same_project_high.clamp(0.20, 0.98),
            (best_cfg.graph_same_project_high + 0.06).clamp(0.20, 0.98),
        ]);
        let same_low_vals = unique_f64(vec![
            (best_cfg.graph_same_project_low - 0.06).clamp(0.10, 0.95),
            best_cfg.graph_same_project_low.clamp(0.10, 0.95),
            (best_cfg.graph_same_project_low + 0.06).clamp(0.10, 0.95),
        ]);
        let related_scale_vals = unique_f64(vec![
            (best_cfg.graph_related_scale - 0.10).clamp(0.10, 1.40),
            best_cfg.graph_related_scale.clamp(0.10, 1.40),
            (best_cfg.graph_related_scale + 0.10).clamp(0.10, 1.40),
        ]);
        let related_base_vals = unique_f64(vec![
            (best_cfg.graph_related_base - 0.05).clamp(0.02, 0.50),
            best_cfg.graph_related_base.clamp(0.02, 0.50),
            (best_cfg.graph_related_base + 0.05).clamp(0.02, 0.50),
        ]);
        let related_cap_vals = unique_f64(vec![
            (best_cfg.graph_related_cap - 0.08).clamp(0.20, 0.99),
            best_cfg.graph_related_cap.clamp(0.20, 0.99),
            (best_cfg.graph_related_cap + 0.08).clamp(0.20, 0.99),
        ]);
        for high in &same_high_vals {
            for low in &same_low_vals {
                if low > high {
                    continue;
                }
                for scale in &related_scale_vals {
                    for base in &related_base_vals {
                        for cap in &related_cap_vals {
                            let mut candidate = best_cfg.clone();
                            candidate.graph_same_project_high = *high;
                            candidate.graph_same_project_low = *low;
                            candidate.graph_related_scale = *scale;
                            candidate.graph_related_base = *base;
                            candidate.graph_related_cap = (*cap).max(*base);
                            maybe_promote_candidate(
                                conn,
                                &examples,
                                limit,
                                candidate,
                                &mut tested,
                                &mut seen,
                                &mut best_cfg,
                                &mut best_metrics,
                            )?;
                        }
                    }
                }
            }
        }

        let chunk_templates = vec![
            normalize3(
                best_cfg.rank_chunk_semantic_weight,
                best_cfg.rank_chunk_lexical_weight,
                best_cfg.rank_chunk_graph_weight,
            ),
            (0.72, 0.18, 0.10),
            (0.62, 0.23, 0.15),
            (0.54, 0.36, 0.10),
            (0.56, 0.20, 0.24),
        ];
        for (s, l, g) in chunk_templates {
            let mut candidate = best_cfg.clone();
            let (ns, nl, ng) = normalize3(s, l, g);
            candidate.rank_chunk_semantic_weight = ns;
            candidate.rank_chunk_lexical_weight = nl;
            candidate.rank_chunk_graph_weight = ng;
            maybe_promote_candidate(
                conn,
                &examples,
                limit,
                candidate,
                &mut tested,
                &mut seen,
                &mut best_cfg,
                &mut best_metrics,
            )?;
        }

        let project_templates = vec![
            normalize5([
                best_cfg.rank_project_content_weight,
                best_cfg.rank_project_semantic_weight,
                best_cfg.rank_project_path_weight,
                best_cfg.rank_project_graph_weight,
                best_cfg.rank_project_frecency_weight,
            ]),
            [0.62, 0.12, 0.08, 0.12, 0.06],
            [0.50, 0.10, 0.08, 0.24, 0.08],
            [0.52, 0.14, 0.18, 0.10, 0.06],
            [0.48, 0.12, 0.08, 0.10, 0.22],
        ];
        for template in project_templates {
            let mut candidate = best_cfg.clone();
            let p = normalize5(template);
            candidate.rank_project_content_weight = p[0];
            candidate.rank_project_semantic_weight = p[1];
            candidate.rank_project_path_weight = p[2];
            candidate.rank_project_graph_weight = p[3];
            candidate.rank_project_frecency_weight = p[4];
            maybe_promote_candidate(
                conn,
                &examples,
                limit,
                candidate,
                &mut tested,
                &mut seen,
                &mut best_cfg,
                &mut best_metrics,
            )?;
        }

        let lexical_vals = unique_i64(vec![
            (best_cfg.lexical_candidates - 40).clamp(20, 1000),
            best_cfg.lexical_candidates.clamp(20, 1000),
            (best_cfg.lexical_candidates + 40).clamp(20, 1000),
        ]);
        let vector_vals = unique_i64(vec![
            (best_cfg.vector_candidates - 40).clamp(20, 1000),
            best_cfg.vector_candidates.clamp(20, 1000),
            (best_cfg.vector_candidates + 40).clamp(20, 1000),
        ]);
        for lexical in &lexical_vals {
            for vector in &vector_vals {
                let mut candidate = best_cfg.clone();
                candidate.lexical_candidates = *lexical;
                candidate.vector_candidates = *vector;
                maybe_promote_candidate(
                    conn,
                    &examples,
                    limit,
                    candidate,
                    &mut tested,
                    &mut seen,
                    &mut best_cfg,
                    &mut best_metrics,
                )?;
            }
        }
    }

    Ok(AutotuneOutcome {
        cfg: best_cfg,
        examples_used: examples.len(),
        baseline_mrr: baseline.0,
        baseline_hit1: baseline.1,
        baseline_hit3: baseline.2,
        best_mrr: best_metrics.0,
        best_hit1: best_metrics.1,
        best_hit3: best_metrics.2,
        candidates_tested: tested.max(1),
        used_history: true,
    })
}

pub(crate) fn autotune_snapshot(cfg: &ConfigValues) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, _) in config_rows() {
        if let Some(value) = config_value_string(cfg, key) {
            out.insert(key.to_string(), value);
        }
    }
    out
}

pub(crate) fn write_autotune_report(
    cwd: &Path,
    opts: &AutotuneOptions,
    baseline_cfg: &ConfigValues,
    outcome: &AutotuneOutcome,
    applied: bool,
    cfg_path: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    let report_dir = data_dir(cwd).join("autotune");
    fs::create_dir_all(&report_dir)
        .map_err(|e| format!("failed creating autotune report dir: {}", e))?;

    let ts = now_ts();
    let ts_ms = (ts * 1000.0).round() as i64;
    let run_json = report_dir.join(format!("report-{}.json", ts_ms));
    let run_md = report_dir.join(format!("report-{}.md", ts_ms));
    let latest_json = report_dir.join("latest.json");
    let latest_md = report_dir.join("latest.md");

    let before = autotune_snapshot(baseline_cfg);
    let after = autotune_snapshot(&outcome.cfg);
    let mut changed: Vec<Value> = Vec::new();
    for (key, after_value) in &after {
        let before_value = before.get(key).cloned().unwrap_or_default();
        if before_value != *after_value {
            changed.push(serde_json::json!({
                "key": key,
                "before": before_value,
                "after": after_value
            }));
        }
    }

    let report = serde_json::json!({
        "schema": "autotune-report-v1",
        "generated_at": ts,
        "generated_at_ms": ts_ms,
        "applied": applied,
        "config_path": cfg_path,
        "options": {
            "dry_run": opts.dry_run,
            "deep": opts.deep,
            "max_events": opts.max_events,
            "limit": opts.limit
        },
        "summary": {
            "examples_used": outcome.examples_used,
            "used_history": outcome.used_history,
            "candidates_tested": outcome.candidates_tested,
            "baseline_mrr": outcome.baseline_mrr,
            "best_mrr": outcome.best_mrr,
            "delta_mrr": outcome.best_mrr - outcome.baseline_mrr,
            "baseline_hit1": outcome.baseline_hit1,
            "best_hit1": outcome.best_hit1,
            "delta_hit1": outcome.best_hit1 - outcome.baseline_hit1,
            "baseline_hit3": outcome.baseline_hit3,
            "best_hit3": outcome.best_hit3,
            "delta_hit3": outcome.best_hit3 - outcome.baseline_hit3
        },
        "baseline_config": before,
        "recommended_config": after,
        "changed": changed
    });
    let report_raw = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize autotune report: {}", e))?;
    fs::write(&run_json, format!("{}\n", report_raw)).map_err(|e| {
        format!(
            "failed writing autotune report '{}': {}",
            run_json.display(),
            e
        )
    })?;
    fs::write(&latest_json, format!("{}\n", report_raw)).map_err(|e| {
        format!(
            "failed writing autotune latest report '{}': {}",
            latest_json.display(),
            e
        )
    })?;

    let mut md_lines: Vec<String> = Vec::new();
    md_lines.push("# Retrivio Autotune Report".to_string());
    md_lines.push(String::new());
    md_lines.push(format!("- generated_at: {:.3}", ts));
    md_lines.push(format!("- applied: {}", yes_no(applied)));
    md_lines.push(format!("- config_path: `{}`", cfg_path.display()));
    md_lines.push(format!(
        "- options: dry_run={} deep={} max_events={} limit={}",
        yes_no(opts.dry_run),
        yes_no(opts.deep),
        opts.max_events,
        opts.limit
    ));
    md_lines.push(String::new());
    md_lines.push("## Metrics".to_string());
    md_lines.push(String::new());
    md_lines.push(format!(
        "- examples_used: {} (history={})",
        outcome.examples_used,
        yes_no(outcome.used_history)
    ));
    md_lines.push(format!(
        "- candidates_tested: {}",
        outcome.candidates_tested
    ));
    md_lines.push(format!(
        "- mrr: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_mrr,
        outcome.best_mrr,
        outcome.best_mrr - outcome.baseline_mrr
    ));
    md_lines.push(format!(
        "- hit@1: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_hit1,
        outcome.best_hit1,
        outcome.best_hit1 - outcome.baseline_hit1
    ));
    md_lines.push(format!(
        "- hit@3: {:.4} -> {:.4} (delta {:+.4})",
        outcome.baseline_hit3,
        outcome.best_hit3,
        outcome.best_hit3 - outcome.baseline_hit3
    ));
    md_lines.push(String::new());
    md_lines.push("## Changed Settings".to_string());
    md_lines.push(String::new());
    if changed.is_empty() {
        md_lines.push("- none".to_string());
    } else {
        for row in &changed {
            let key = row.get("key").and_then(|v| v.as_str()).unwrap_or_default();
            let before_v = row
                .get("before")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let after_v = row
                .get("after")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            md_lines.push(format!("- `{}`: `{}` -> `{}`", key, before_v, after_v));
        }
    }
    let md_raw = format!("{}\n", md_lines.join("\n"));
    fs::write(&run_md, &md_raw).map_err(|e| {
        format!(
            "failed writing autotune markdown '{}': {}",
            run_md.display(),
            e
        )
    })?;
    fs::write(&latest_md, &md_raw).map_err(|e| {
        format!(
            "failed writing autotune latest markdown '{}': {}",
            latest_md.display(),
            e
        )
    })?;
    Ok((latest_json, latest_md))
}

pub(crate) fn run_autotune_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]");
        println!("notes:");
        println!("  - tunes ranking settings from historical selection events");
        println!("  - use --deep for a larger candidate sweep (slower)");
        println!("  - writes tuned config unless --dry-run is used");
        return;
    }
    let opts = parse_autotune_options(args);
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let outcome = autotune_recommendation(&conn, &cfg, opts.max_events, opts.limit, opts.deep)
        .unwrap_or_else(|e| {
            eprintln!("error: autotune failed: {}", e);
            process::exit(1);
        });

    println!(
        "autotune: examples={} baseline_mrr={:.4} best_mrr={:.4} candidates={}",
        outcome.examples_used, outcome.baseline_mrr, outcome.best_mrr, outcome.candidates_tested
    );
    println!(
        "autotune: baseline_hit1={:.4} best_hit1={:.4} baseline_hit3={:.4} best_hit3={:.4}",
        outcome.baseline_hit1, outcome.best_hit1, outcome.baseline_hit3, outcome.best_hit3
    );
    if !outcome.used_history {
        println!("autotune: not enough history; applied heuristic defaults");
    }
    println!(
        "autotune: rank_chunk_graph_weight={:.4} rank_project_graph_weight={:.4} graph_seed_limit={} graph_neighbor_limit={}",
        outcome.cfg.rank_chunk_graph_weight,
        outcome.cfg.rank_project_graph_weight,
        outcome.cfg.graph_seed_limit,
        outcome.cfg.graph_neighbor_limit
    );

    if opts.dry_run {
        println!("autotune: dry-run enabled; config not written");
    } else {
        write_config_file(&cfg_path, &outcome.cfg).unwrap_or_else(|e| {
            eprintln!("error: failed writing tuned config: {}", e);
            process::exit(1);
        });
        println!("autotune: config updated -> {}", cfg_path.display());
    }
    let (report_json, report_md) =
        write_autotune_report(&cwd, &opts, &cfg, &outcome, !opts.dry_run, &cfg_path)
            .unwrap_or_else(|e| {
                eprintln!("error: failed writing autotune report: {}", e);
                process::exit(1);
            });
    println!("autotune: report_json={}", report_json.display());
    println!("autotune: report_md={}", report_md.display());
}
