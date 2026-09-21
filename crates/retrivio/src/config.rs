//! Configuration: ConfigValues and its file format, the config rows and setters, the CLI store-location overrides and ScanSettings.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::{env, fs};

use crate::embed::{
    default_embed_model_for_backend, is_known_embed_backend,
    migrate_isengard_add_profile_to_credential_cmd, EMBED_BACKENDS,
};
#[cfg(test)]
use crate::test_support;
use crate::util::{expand_tilde, normalize_path};
use crate::{documents, roles};

pub(crate) static CLI_DATA_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();
pub(crate) static CLI_CONFIG_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();

pub(crate) fn set_cli_data_dir_override(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("--data-dir requires a non-empty path".to_string());
    }
    let resolved = expand_tilde(trimmed);
    if let Some(existing) = CLI_DATA_DIR_OVERRIDE.get() {
        if existing != &resolved {
            return Err(format!(
                "--data-dir specified multiple times with different values ('{}' vs '{}')",
                existing.display(),
                resolved.display()
            ));
        }
        return Ok(());
    }
    let _ = CLI_DATA_DIR_OVERRIDE.set(resolved);
    Ok(())
}

pub(crate) fn set_cli_config_override(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("--config requires a non-empty path".to_string());
    }
    let resolved = expand_tilde(trimmed);
    if let Some(existing) = CLI_CONFIG_OVERRIDE.get() {
        if existing != &resolved {
            return Err(format!(
                "--config specified multiple times with different values ('{}' vs '{}')",
                existing.display(),
                resolved.display()
            ));
        }
        return Ok(());
    }
    let _ = CLI_CONFIG_OVERRIDE.set(resolved);
    Ok(())
}

pub(crate) fn consume_global_path_overrides(args: Vec<OsString>) -> Result<Vec<OsString>, String> {
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--data-dir" {
            i += 1;
            if i >= args.len() {
                return Err("--data-dir requires a path".to_string());
            }
            let raw = args[i].to_string_lossy().to_string();
            set_cli_data_dir_override(&raw)?;
            i += 1;
            continue;
        }
        if let Some(raw) = s.strip_prefix("--data-dir=") {
            set_cli_data_dir_override(raw)?;
            i += 1;
            continue;
        }
        if s == "--config" {
            i += 1;
            if i >= args.len() {
                return Err("--config requires a path".to_string());
            }
            let raw = args[i].to_string_lossy().to_string();
            set_cli_config_override(&raw)?;
            i += 1;
            continue;
        }
        if let Some(raw) = s.strip_prefix("--config=") {
            set_cli_config_override(raw)?;
            i += 1;
            continue;
        }
        break;
    }
    Ok(args.into_iter().skip(i).collect())
}

pub(crate) fn config_rows() -> Vec<(&'static str, &'static str)> {
    vec![
        ("root", "Root path hint (not auto-tracked)"),
        (
            "embed_backend",
            "Embedding backend: ollama, bedrock, or hash (offline, tests and smoke checks only; not semantic)",
        ),
        ("embed_model", "Embedding model id"),
        ("aws_profile", "AWS profile for Bedrock"),
        ("aws_region", "AWS region for Bedrock"),
        ("aws_refresh_cmd", "Credential refresh command (optional)"),
        (
            "aws_credential_cmd",
            "On-demand credential command, must print credential_process JSON (optional)",
        ),
        ("bedrock_concurrency", "Bedrock invoke concurrency"),
        ("bedrock_max_retries", "Bedrock max retry attempts"),
        ("bedrock_retry_base_ms", "Bedrock retry base backoff (ms)"),
        ("retrieval_backend", "Retrieval backend"),
        ("local_embed_dim", "Embedding dimension for local models"),
        ("max_chars_per_project", "Indexing cap per project"),
        ("max_files_per_project", "Files indexed per project at most"),
        (
            "max_chunks_per_project",
            "Chunks indexed per project at most",
        ),
        ("max_chunks_per_file", "Chunks indexed per file at most"),
        ("max_file_chars", "Characters read per file at most"),
        (
            "index_documents",
            "Extract text from docx, pptx, odt, odp, xlsx, pdf (text-based) and html",
        ),
        (
            "max_document_bytes",
            "Documents larger than this are skipped (counted as failed)",
        ),
        (
            "max_document_uncompressed_bytes",
            "Office/OpenDocument archives declaring more uncompressed bytes than this are refused unread",
        ),
        (
            "document_extract_timeout_ms",
            "PDF extraction runs in a child process killed after this many milliseconds",
        ),
        (
            "lance_compact_versions",
            "Watcher compacts LanceDB when it holds more versions than this (0 = never)",
        ),
        (
            "lance_version_grace_secs",
            "Compaction keeps LanceDB versions younger than this so concurrent readers finish",
        ),
        ("lexical_candidates", "Lexical candidates"),
        ("vector_candidates", "Vector candidates"),
        ("rank_chunk_semantic_weight", "Chunk score semantic weight"),
        ("rank_chunk_lexical_weight", "Chunk score lexical weight"),
        ("rank_chunk_graph_weight", "Chunk score graph weight"),
        ("rank_quality_mix", "Chunk quality mixing factor"),
        (
            "rank_relation_quality_good_boost",
            "Related score boost for quality=good",
        ),
        (
            "rank_relation_quality_weak_penalty",
            "Related score penalty for quality=weak",
        ),
        (
            "rank_relation_quality_wrong_penalty",
            "Related score penalty for quality=wrong",
        ),
        ("rank_project_content_weight", "Project rank content weight"),
        (
            "rank_project_semantic_weight",
            "Project rank semantic weight",
        ),
        (
            "rank_project_path_weight",
            "Project rank path keyword weight",
        ),
        ("rank_project_graph_weight", "Project rank graph weight"),
        (
            "rank_project_frecency_weight",
            "Project rank frecency weight",
        ),
        ("graph_seed_limit", "Graph expansion seed chunk count"),
        ("graph_neighbor_limit", "Graph neighbor traversal limit"),
        (
            "graph_same_project_high",
            "Same-project graph weight (high semantic)",
        ),
        (
            "graph_same_project_low",
            "Same-project graph weight (low semantic)",
        ),
        ("graph_related_base", "Related-project graph base"),
        ("graph_related_scale", "Related-project graph scale"),
        ("graph_related_cap", "Related-project graph cap"),
        (
            "hyde_enabled",
            "HyDE hypothetical-document query expansion (opt-in)",
        ),
        (
            "reranker_enabled",
            "Ollama cross-encoder re-ranking of chunk results",
        ),
        ("reranker_model", "Ollama model used for re-ranking"),
        ("reranker_pool_size", "Re-ranker candidate pool size"),
        ("reranker_batch_size", "Re-ranker parallel batch size"),
        ("reranker_timeout_ms", "Re-ranker per-request timeout (ms)"),
        (
            "rank_recency_weight",
            "Recency blend weight for living docs",
        ),
        (
            "rank_recency_record_weight",
            "Recency blend weight for records",
        ),
        (
            "recency_half_life_days",
            "Recency half-life for living docs (days)",
        ),
        (
            "recency_record_half_life_days",
            "Recency half-life for records (days)",
        ),
        (
            "recency_record_patterns",
            "Comma-separated path patterns that mark records",
        ),
        (
            "skip_dir_names",
            "Extra directory names skipped at discovery/indexing (comma-separated)",
        ),
        (
            "recall_max_leads",
            "Max leads injected by `retrivio recall`",
        ),
        (
            "recall_min_score_ratio",
            "Recall: min score ratio vs top lead",
        ),
        (
            "recall_min_abs_score",
            "Recall: min raw cosine similarity a lead must reach (semantic mode)",
        ),
        (
            "search_min_abs_score",
            "Search: min raw cosine similarity for file results (0 = off)",
        ),
        (
            "recall_band_ratio",
            "Recall: score band ratio for grouping leads",
        ),
        (
            "recall_roots",
            "Recall: comma-separated roots (empty = all tracked)",
        ),
        ("recall_excerpts", "Recall: include excerpts in leads"),
        ("recall_system_message", "Recall: emit as system message"),
        ("recall_semantic", "Recall: semantic retrieval mode"),
        (
            "recall_dossier",
            "Recall: automatic topic dossier (shadow logs the gate, auto applies it, off)",
        ),
        (
            "recall_session_ttl_days",
            "Recall: session memory TTL (days)",
        ),
    ]
}

pub(crate) fn config_enum_options(key: &str) -> Option<Vec<&'static str>> {
    match key {
        "embed_backend" => Some(EMBED_BACKENDS.to_vec()),
        "retrieval_backend" => Some(vec!["lancedb"]),
        "recall_semantic" => Some(vec!["auto", "on", "off"]),
        "recall_dossier" => Some(vec!["shadow", "auto", "off"]),
        "hyde_enabled"
        | "reranker_enabled"
        | "recall_excerpts"
        | "recall_system_message"
        | "index_documents" => Some(vec!["false", "true"]),
        _ => None,
    }
}

pub(crate) fn config_value_string(cfg: &ConfigValues, key: &str) -> Option<String> {
    match key {
        "root" => Some(cfg.root.to_string_lossy().to_string()),
        "embed_backend" => Some(cfg.embed_backend.clone()),
        "embed_model" => Some(cfg.embed_model.clone()),
        "aws_profile" => Some(cfg.aws_profile.clone()),
        "aws_region" => Some(cfg.aws_region.clone()),
        "aws_refresh_cmd" => Some(cfg.aws_refresh_cmd.clone()),
        "aws_credential_cmd" => Some(cfg.aws_credential_cmd.clone()),
        "bedrock_concurrency" => Some(cfg.bedrock_concurrency.to_string()),
        "bedrock_max_retries" => Some(cfg.bedrock_max_retries.to_string()),
        "bedrock_retry_base_ms" => Some(cfg.bedrock_retry_base_ms.to_string()),
        "retrieval_backend" => Some(cfg.retrieval_backend.clone()),
        "local_embed_dim" => Some(cfg.local_embed_dim.to_string()),
        "max_chars_per_project" => Some(cfg.max_chars_per_project.to_string()),
        "max_files_per_project" => Some(cfg.max_files_per_project.to_string()),
        "max_chunks_per_project" => Some(cfg.max_chunks_per_project.to_string()),
        "max_chunks_per_file" => Some(cfg.max_chunks_per_file.to_string()),
        "max_file_chars" => Some(cfg.max_file_chars.to_string()),
        "index_documents" => Some(cfg.index_documents.to_string()),
        "max_document_bytes" => Some(cfg.max_document_bytes.to_string()),
        "max_document_uncompressed_bytes" => Some(cfg.max_document_uncompressed_bytes.to_string()),
        "document_extract_timeout_ms" => Some(cfg.document_extract_timeout_ms.to_string()),
        "lance_compact_versions" => Some(cfg.lance_compact_versions.to_string()),
        "lance_version_grace_secs" => Some(cfg.lance_version_grace_secs.to_string()),
        "lexical_candidates" => Some(cfg.lexical_candidates.to_string()),
        "vector_candidates" => Some(cfg.vector_candidates.to_string()),
        "rank_chunk_semantic_weight" => Some(format!("{:.6}", cfg.rank_chunk_semantic_weight)),
        "rank_chunk_lexical_weight" => Some(format!("{:.6}", cfg.rank_chunk_lexical_weight)),
        "rank_chunk_graph_weight" => Some(format!("{:.6}", cfg.rank_chunk_graph_weight)),
        "rank_quality_mix" => Some(format!("{:.6}", cfg.rank_quality_mix)),
        "rank_relation_quality_good_boost" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_good_boost))
        }
        "rank_relation_quality_weak_penalty" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_weak_penalty))
        }
        "rank_relation_quality_wrong_penalty" => {
            Some(format!("{:.6}", cfg.rank_relation_quality_wrong_penalty))
        }
        "rank_project_content_weight" => Some(format!("{:.6}", cfg.rank_project_content_weight)),
        "rank_project_semantic_weight" => Some(format!("{:.6}", cfg.rank_project_semantic_weight)),
        "rank_project_path_weight" => Some(format!("{:.6}", cfg.rank_project_path_weight)),
        "rank_project_graph_weight" => Some(format!("{:.6}", cfg.rank_project_graph_weight)),
        "rank_project_frecency_weight" => Some(format!("{:.6}", cfg.rank_project_frecency_weight)),
        "graph_seed_limit" => Some(cfg.graph_seed_limit.to_string()),
        "graph_neighbor_limit" => Some(cfg.graph_neighbor_limit.to_string()),
        "graph_same_project_high" => Some(format!("{:.6}", cfg.graph_same_project_high)),
        "graph_same_project_low" => Some(format!("{:.6}", cfg.graph_same_project_low)),
        "graph_related_base" => Some(format!("{:.6}", cfg.graph_related_base)),
        "graph_related_scale" => Some(format!("{:.6}", cfg.graph_related_scale)),
        "graph_related_cap" => Some(format!("{:.6}", cfg.graph_related_cap)),
        "hyde_enabled" => Some(cfg.hyde_enabled.to_string()),
        "reranker_enabled" => Some(cfg.reranker_enabled.to_string()),
        "reranker_model" => Some(cfg.reranker_model.clone()),
        "reranker_pool_size" => Some(cfg.reranker_pool_size.to_string()),
        "reranker_batch_size" => Some(cfg.reranker_batch_size.to_string()),
        "reranker_timeout_ms" => Some(cfg.reranker_timeout_ms.to_string()),
        "rank_recency_weight" => Some(format!("{:.6}", cfg.rank_recency_weight)),
        "rank_recency_record_weight" => Some(format!("{:.6}", cfg.rank_recency_record_weight)),
        "recency_half_life_days" => Some(format!("{:.6}", cfg.recency_half_life_days)),
        "recency_record_half_life_days" => {
            Some(format!("{:.6}", cfg.recency_record_half_life_days))
        }
        "recency_record_patterns" => Some(cfg.recency_record_patterns.clone()),
        "skip_dir_names" => Some(cfg.skip_dir_names.clone()),
        "recall_max_leads" => Some(cfg.recall_max_leads.to_string()),
        "recall_min_score_ratio" => Some(format!("{:.6}", cfg.recall_min_score_ratio)),
        "recall_min_abs_score" => Some(format!("{:.6}", cfg.recall_min_abs_score)),
        "search_min_abs_score" => Some(format!("{:.6}", cfg.search_min_abs_score)),
        "recall_band_ratio" => Some(format!("{:.6}", cfg.recall_band_ratio)),
        "recall_roots" => Some(cfg.recall_roots.clone()),
        "recall_excerpts" => Some(cfg.recall_excerpts.to_string()),
        "recall_system_message" => Some(cfg.recall_system_message.to_string()),
        "recall_semantic" => Some(cfg.recall_semantic.clone()),
        "recall_dossier" => Some(cfg.recall_dossier.clone()),
        "recall_session_ttl_days" => Some(format!("{:.6}", cfg.recall_session_ttl_days)),
        _ => None,
    }
}

/// A float setting must be a finite number. `nan`, `inf` and `infinity` parse as `f64`, but
/// `clamp` passes NaN through and pins an infinity to a bound the user did not name, and a
/// written NaN is then dropped at load with the default applied silently.
pub(crate) fn parse_finite_f64(key: &str, value: &str) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(format!("{} must be a finite number", key)),
    }
}

pub(crate) fn config_set_value(cfg: &mut ConfigValues, key: &str, raw: &str) -> Result<(), String> {
    let value = raw.trim();
    match key {
        "root" => {
            if value.is_empty() {
                return Err("root must not be empty".to_string());
            }
            cfg.root = normalize_path(value);
        }
        "embed_backend" => {
            let old_backend = cfg.embed_backend.clone();
            let old_default = default_embed_model_for_backend(&old_backend).to_string();
            let v = value.to_lowercase();
            if !is_known_embed_backend(&v) {
                return Err(format!(
                    "embed_backend must be one of: {}",
                    EMBED_BACKENDS.join(", ")
                ));
            }
            cfg.embed_backend = v;
            let current_model = cfg.embed_model.trim();
            if current_model.is_empty() || current_model == old_default {
                cfg.embed_model = default_embed_model_for_backend(&cfg.embed_backend).to_string();
            }
        }
        "embed_model" => {
            if value.is_empty() {
                return Err("embed_model must not be empty".to_string());
            }
            cfg.embed_model = value.to_string();
        }
        "aws_profile" => {
            cfg.aws_profile = value.to_string();
        }
        "aws_region" => {
            cfg.aws_region = value.to_string();
        }
        "aws_refresh_cmd" => {
            cfg.aws_refresh_cmd = value.to_string();
        }
        "aws_credential_cmd" => {
            cfg.aws_credential_cmd = value.to_string();
        }
        "bedrock_concurrency" => {
            cfg.bedrock_concurrency = value
                .parse::<i64>()
                .map_err(|_| "bedrock_concurrency must be an integer".to_string())?
                .clamp(1, 128);
        }
        "bedrock_max_retries" => {
            cfg.bedrock_max_retries = value
                .parse::<i64>()
                .map_err(|_| "bedrock_max_retries must be an integer".to_string())?
                .clamp(0, 12);
        }
        "bedrock_retry_base_ms" => {
            cfg.bedrock_retry_base_ms = value
                .parse::<i64>()
                .map_err(|_| "bedrock_retry_base_ms must be an integer".to_string())?
                .clamp(50, 10_000);
        }
        "retrieval_backend" => {
            // LanceDB is the only backend; silently accept any value
            cfg.retrieval_backend = "lancedb".to_string();
        }
        "local_embed_dim" => {
            cfg.local_embed_dim = value
                .parse::<i64>()
                .map_err(|_| "local_embed_dim must be an integer".to_string())?
                .max(64);
        }
        "max_chars_per_project" => {
            cfg.max_chars_per_project = value
                .parse::<i64>()
                .map_err(|_| "max_chars_per_project must be an integer".to_string())?
                .clamp(1000, 500_000);
        }
        "max_files_per_project" => {
            cfg.max_files_per_project = value
                .parse::<i64>()
                .map_err(|_| "max_files_per_project must be an integer".to_string())?
                .clamp(1, 1_000_000);
        }
        "max_chunks_per_project" => {
            cfg.max_chunks_per_project = value
                .parse::<i64>()
                .map_err(|_| "max_chunks_per_project must be an integer".to_string())?
                .clamp(1, 10_000_000);
        }
        "max_chunks_per_file" => {
            cfg.max_chunks_per_file = value
                .parse::<i64>()
                .map_err(|_| "max_chunks_per_file must be an integer".to_string())?
                .clamp(1, 100_000);
        }
        "max_file_chars" => {
            cfg.max_file_chars = value
                .parse::<i64>()
                .map_err(|_| "max_file_chars must be an integer".to_string())?
                .clamp(1000, 50_000_000);
        }
        "index_documents" => {
            cfg.index_documents = parse_bool_setting(key, value)?;
        }
        "max_document_bytes" => {
            cfg.max_document_bytes = value
                .parse::<i64>()
                .map_err(|_| "max_document_bytes must be an integer".to_string())?
                .clamp(1000, 2_000_000_000);
        }
        "max_document_uncompressed_bytes" => {
            cfg.max_document_uncompressed_bytes = value
                .parse::<i64>()
                .map_err(|_| "max_document_uncompressed_bytes must be an integer".to_string())?
                .clamp(1000, 20_000_000_000);
        }
        "document_extract_timeout_ms" => {
            cfg.document_extract_timeout_ms = value
                .parse::<i64>()
                .map_err(|_| "document_extract_timeout_ms must be an integer".to_string())?
                .clamp(500, 600_000);
        }
        "lance_compact_versions" => {
            cfg.lance_compact_versions = value
                .parse::<i64>()
                .map_err(|_| "lance_compact_versions must be an integer".to_string())?
                .clamp(0, 1_000_000);
        }
        "lance_version_grace_secs" => {
            cfg.lance_version_grace_secs = value
                .parse::<i64>()
                .map_err(|_| "lance_version_grace_secs must be an integer".to_string())?
                .clamp(0, 86_400);
        }
        "lexical_candidates" => {
            cfg.lexical_candidates = value
                .parse::<i64>()
                .map_err(|_| "lexical_candidates must be an integer".to_string())?
                .clamp(20, 1000);
        }
        "vector_candidates" => {
            cfg.vector_candidates = value
                .parse::<i64>()
                .map_err(|_| "vector_candidates must be an integer".to_string())?
                .clamp(20, 1000);
        }
        "rank_chunk_semantic_weight" => {
            cfg.rank_chunk_semantic_weight =
                parse_finite_f64("rank_chunk_semantic_weight", value)?.clamp(0.0, 1.0);
        }
        "rank_chunk_lexical_weight" => {
            cfg.rank_chunk_lexical_weight =
                parse_finite_f64("rank_chunk_lexical_weight", value)?.clamp(0.0, 1.0);
        }
        "rank_chunk_graph_weight" => {
            cfg.rank_chunk_graph_weight =
                parse_finite_f64("rank_chunk_graph_weight", value)?.clamp(0.0, 1.0);
        }
        "rank_quality_mix" => {
            cfg.rank_quality_mix = parse_finite_f64("rank_quality_mix", value)?.clamp(0.0, 1.0);
        }
        "rank_relation_quality_good_boost" => {
            cfg.rank_relation_quality_good_boost =
                parse_finite_f64("rank_relation_quality_good_boost", value)?.clamp(0.0, 1.0);
        }
        "rank_relation_quality_weak_penalty" => {
            cfg.rank_relation_quality_weak_penalty =
                parse_finite_f64("rank_relation_quality_weak_penalty", value)?.clamp(0.0, 1.0);
        }
        "rank_relation_quality_wrong_penalty" => {
            cfg.rank_relation_quality_wrong_penalty =
                parse_finite_f64("rank_relation_quality_wrong_penalty", value)?.clamp(0.0, 1.0);
        }
        "rank_project_content_weight" => {
            cfg.rank_project_content_weight =
                parse_finite_f64("rank_project_content_weight", value)?.clamp(0.0, 2.0);
        }
        "rank_project_semantic_weight" => {
            cfg.rank_project_semantic_weight =
                parse_finite_f64("rank_project_semantic_weight", value)?.clamp(0.0, 2.0);
        }
        "rank_project_path_weight" => {
            cfg.rank_project_path_weight =
                parse_finite_f64("rank_project_path_weight", value)?.clamp(0.0, 2.0);
        }
        "rank_project_graph_weight" => {
            cfg.rank_project_graph_weight =
                parse_finite_f64("rank_project_graph_weight", value)?.clamp(0.0, 2.0);
        }
        "rank_project_frecency_weight" => {
            cfg.rank_project_frecency_weight =
                parse_finite_f64("rank_project_frecency_weight", value)?.clamp(0.0, 2.0);
        }
        "graph_seed_limit" => {
            cfg.graph_seed_limit = value
                .parse::<i64>()
                .map_err(|_| "graph_seed_limit must be an integer".to_string())?
                .clamp(2, 64);
        }
        "graph_neighbor_limit" => {
            cfg.graph_neighbor_limit = value
                .parse::<i64>()
                .map_err(|_| "graph_neighbor_limit must be an integer".to_string())?
                .clamp(8, 500);
        }
        "graph_same_project_high" => {
            cfg.graph_same_project_high =
                parse_finite_f64("graph_same_project_high", value)?.clamp(0.0, 1.0);
        }
        "graph_same_project_low" => {
            cfg.graph_same_project_low =
                parse_finite_f64("graph_same_project_low", value)?.clamp(0.0, 1.0);
        }
        "graph_related_base" => {
            cfg.graph_related_base = parse_finite_f64("graph_related_base", value)?.clamp(0.0, 1.0);
        }
        "graph_related_scale" => {
            cfg.graph_related_scale =
                parse_finite_f64("graph_related_scale", value)?.clamp(0.0, 2.0);
        }
        "graph_related_cap" => {
            cfg.graph_related_cap = parse_finite_f64("graph_related_cap", value)?.clamp(0.0, 1.0);
        }
        "hyde_enabled" => cfg.hyde_enabled = parse_bool_setting(key, value)?,
        "reranker_enabled" => cfg.reranker_enabled = parse_bool_setting(key, value)?,
        "reranker_model" => {
            if value.is_empty() {
                return Err("reranker_model must not be empty".to_string());
            }
            cfg.reranker_model = value.to_string();
        }
        "reranker_pool_size" => {
            cfg.reranker_pool_size = value
                .parse::<usize>()
                .map_err(|_| "reranker_pool_size must be an integer".to_string())?
                .clamp(10, 200);
        }
        "reranker_batch_size" => {
            cfg.reranker_batch_size = value
                .parse::<usize>()
                .map_err(|_| "reranker_batch_size must be an integer".to_string())?
                .clamp(1, 32);
        }
        "reranker_timeout_ms" => {
            cfg.reranker_timeout_ms = value
                .parse::<u64>()
                .map_err(|_| "reranker_timeout_ms must be an integer".to_string())?
                .clamp(500, 30_000);
        }
        "rank_recency_weight" => {
            cfg.rank_recency_weight =
                parse_finite_f64("rank_recency_weight", value)?.clamp(0.0, 0.5);
        }
        "rank_recency_record_weight" => {
            cfg.rank_recency_record_weight =
                parse_finite_f64("rank_recency_record_weight", value)?.clamp(0.0, 0.5);
        }
        "recency_half_life_days" => {
            cfg.recency_half_life_days =
                parse_finite_f64("recency_half_life_days", value)?.clamp(1.0, 3650.0);
        }
        "recency_record_half_life_days" => {
            cfg.recency_record_half_life_days =
                parse_finite_f64("recency_record_half_life_days", value)?.clamp(1.0, 3650.0);
        }
        "recency_record_patterns" => {
            cfg.recency_record_patterns = split_csv_setting(value).join(",");
        }
        "skip_dir_names" => {
            for name in split_csv_setting(value) {
                if name.contains('/') || name == "." || name == ".." {
                    return Err(format!(
                        "skip_dir_names entries must be bare directory names, got '{}'",
                        name
                    ));
                }
            }
            cfg.skip_dir_names = split_csv_setting(value).join(",");
        }
        "recall_max_leads" => {
            cfg.recall_max_leads = value
                .parse::<usize>()
                .map_err(|_| "recall_max_leads must be an integer".to_string())?
                .clamp(1, 5);
        }
        "recall_min_score_ratio" => {
            cfg.recall_min_score_ratio =
                parse_finite_f64("recall_min_score_ratio", value)?.clamp(0.1, 1.0);
        }
        "recall_min_abs_score" => {
            cfg.recall_min_abs_score =
                parse_finite_f64("recall_min_abs_score", value)?.clamp(0.0, 1.0);
        }
        "search_min_abs_score" => {
            cfg.search_min_abs_score =
                parse_finite_f64("search_min_abs_score", value)?.clamp(0.0, 1.0);
        }
        "recall_band_ratio" => {
            cfg.recall_band_ratio = parse_finite_f64("recall_band_ratio", value)?.clamp(0.5, 1.0);
        }
        "recall_roots" => {
            cfg.recall_roots = split_csv_setting(value).join(",");
        }
        "recall_excerpts" => cfg.recall_excerpts = parse_bool_setting(key, value)?,
        "recall_system_message" => cfg.recall_system_message = parse_bool_setting(key, value)?,
        "recall_semantic" => {
            let v = value.to_lowercase();
            if !matches!(v.as_str(), "auto" | "on" | "off") {
                return Err("recall_semantic must be one of: auto, on, off".to_string());
            }
            cfg.recall_semantic = v;
        }
        "recall_dossier" => {
            let v = value.to_lowercase();
            if !matches!(v.as_str(), "shadow" | "auto" | "off") {
                return Err("recall_dossier must be one of: shadow, auto, off".to_string());
            }
            cfg.recall_dossier = v;
        }
        "recall_session_ttl_days" => {
            cfg.recall_session_ttl_days =
                parse_finite_f64("recall_session_ttl_days", value)?.clamp(1.0, 30.0);
        }
        _ => return Err(format!("unknown config key '{}'", key)),
    }
    Ok(())
}

pub(crate) fn print_config_values(cfg: &ConfigValues) {
    for (key, hint) in config_rows() {
        if let Some(v) = config_value_string(cfg, key) {
            println!("{:<30} = {:<24} # {}", key, v, hint);
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConfigValues {
    pub(crate) root: PathBuf,
    pub(crate) embed_backend: String,
    pub(crate) embed_model: String,
    pub(crate) aws_profile: String,
    pub(crate) aws_region: String,
    pub(crate) aws_refresh_cmd: String,
    pub(crate) aws_credential_cmd: String,
    pub(crate) bedrock_concurrency: i64,
    pub(crate) bedrock_max_retries: i64,
    pub(crate) bedrock_retry_base_ms: i64,
    pub(crate) retrieval_backend: String,
    pub(crate) local_embed_dim: i64,
    pub(crate) max_chars_per_project: i64,
    // Scan caps (Slice 1): what one project scan may index. Defaults unchanged from the
    // former constants; `retrivio index` warns per project when one of them bites.
    pub(crate) max_files_per_project: i64,
    pub(crate) max_chunks_per_project: i64,
    pub(crate) max_chunks_per_file: i64,
    pub(crate) max_file_chars: i64,
    // Document extraction (documents.rs): docx, pptx, odt, odp, xlsx, pdf and html-as-text.
    pub(crate) index_documents: bool,
    pub(crate) max_document_bytes: i64,
    /// Declared uncompressed total an Office/OpenDocument archive may have before it is
    /// refused unread (zip-bomb bound).
    pub(crate) max_document_uncompressed_bytes: i64,
    /// Deadline for the PDF extraction child process.
    pub(crate) document_extract_timeout_ms: i64,
    // LanceDB compaction threshold for the watcher (versions on disk); 0 disables.
    pub(crate) lance_compact_versions: i64,
    /// Compaction keeps LanceDB versions younger than this many seconds, so a reader that
    /// opened an older snapshot can finish its query (longer than the recall hook's 4 s
    /// deadline by a wide margin).
    pub(crate) lance_version_grace_secs: i64,
    pub(crate) lexical_candidates: i64,
    pub(crate) vector_candidates: i64,
    pub(crate) rank_chunk_semantic_weight: f64,
    pub(crate) rank_chunk_lexical_weight: f64,
    pub(crate) rank_chunk_graph_weight: f64,
    pub(crate) rank_quality_mix: f64,
    pub(crate) rank_relation_quality_good_boost: f64,
    pub(crate) rank_relation_quality_weak_penalty: f64,
    pub(crate) rank_relation_quality_wrong_penalty: f64,
    pub(crate) rank_project_content_weight: f64,
    pub(crate) rank_project_semantic_weight: f64,
    pub(crate) rank_project_path_weight: f64,
    pub(crate) rank_project_graph_weight: f64,
    pub(crate) rank_project_frecency_weight: f64,
    pub(crate) graph_seed_limit: i64,
    pub(crate) graph_neighbor_limit: i64,
    pub(crate) graph_same_project_high: f64,
    pub(crate) graph_same_project_low: f64,
    pub(crate) graph_related_base: f64,
    pub(crate) graph_related_scale: f64,
    pub(crate) graph_related_cap: f64,
    // HyDE: Hypothetical Document Embedding (opt-in, adds ~800ms)
    pub(crate) hyde_enabled: bool,
    // Cross-encoder re-ranking via Ollama
    pub(crate) reranker_enabled: bool,
    pub(crate) reranker_model: String,
    pub(crate) reranker_pool_size: usize,
    pub(crate) reranker_batch_size: usize,
    pub(crate) reranker_timeout_ms: u64,
    // Freshness model (spec §4): recency blend weights, half-lives and record patterns
    pub(crate) rank_recency_weight: f64,
    pub(crate) rank_recency_record_weight: f64,
    pub(crate) recency_half_life_days: f64,
    pub(crate) recency_record_half_life_days: f64,
    pub(crate) recency_record_patterns: String,
    // Hygiene (spec §7): extra directory names skipped at discovery and corpus walk
    pub(crate) skip_dir_names: String,
    // Proactive recall (spec §5)
    pub(crate) recall_max_leads: usize,
    pub(crate) recall_min_score_ratio: f64,
    pub(crate) recall_min_abs_score: f64,
    pub(crate) recall_band_ratio: f64,
    /// Raw-cosine floor for `search` results (0 = off); slice 3.
    pub(crate) search_min_abs_score: f64,
    pub(crate) recall_roots: String,
    pub(crate) recall_excerpts: bool,
    pub(crate) recall_system_message: bool,
    pub(crate) recall_semantic: String,
    /// Automatic topic dossier in the hook (slice 4): `shadow` (default; the gate decision is
    /// logged, leads are shown), `auto` (a compact dossier replaces the leads when the gate
    /// fires) or `off`.
    pub(crate) recall_dossier: String,
    pub(crate) recall_session_ttl_days: f64,
}

/// Default comma-separated path patterns that mark a document as a point-in-time record.
/// Matched against path components relative to the project (see `roles.rs`), in addition to
/// the built-in record rules. Handoffs and `docs/sessions` are `state`, not records, and the
/// built-in state rules win over these patterns.
pub(crate) const DEFAULT_RECENCY_RECORD_PATTERNS: &str =
    "transcript,customer-signals,meeting,call-notes,.srt";

pub(crate) fn parse_bool_config(
    map: &std::collections::HashMap<String, String>,
    key: &str,
    default: bool,
) -> bool {
    map.get(key)
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(default)
}

pub(crate) fn parse_bool_setting(key: &str, value: &str) -> Result<bool, String> {
    match value.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("{} must be true or false", key)),
    }
}

/// Split a comma-separated config value into trimmed, non-empty items.
pub(crate) fn split_csv_setting(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub(crate) fn default_config_root() -> PathBuf {
    env::current_dir()
        .ok()
        .map(|p| normalize_path(&p.to_string_lossy()))
        .unwrap_or_else(|| expand_tilde("~"))
}

impl ConfigValues {
    pub(crate) fn from_map(map: std::collections::HashMap<String, String>) -> Self {
        let mut embed_backend = map
            .get("embed_backend")
            .cloned()
            .unwrap_or_else(|| "ollama".to_string())
            .trim()
            .to_lowercase();
        if !is_known_embed_backend(&embed_backend) {
            embed_backend = "ollama".to_string();
        }

        // LanceDB is the default (and only) retrieval backend.
        // Silently convert "falkordb" for backward compatibility.
        let retrieval_backend = "lancedb".to_string();

        let mut embed_model = map
            .get("embed_model")
            .cloned()
            .unwrap_or_else(|| default_embed_model_for_backend(&embed_backend).to_string());
        if embed_model.trim().is_empty() || embed_model == "sentence-transformers/all-MiniLM-L6-v2"
        {
            embed_model = default_embed_model_for_backend(&embed_backend).to_string();
        }
        let aws_profile = map
            .get("aws_profile")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
        let aws_region = map
            .get("aws_region")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
        let mut aws_refresh_cmd = map
            .get("aws_refresh_cmd")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
        let mut aws_credential_cmd = map
            .get("aws_credential_cmd")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
        // Silent migration: legacy `isengardcli add-profile EMAIL --role ROLE` was a
        // one-shot that wrote static keys; those are no longer issued. Convert it to
        // an on-demand `isengardcli credentials --awscli EMAIL --role ROLE` invocation.
        if aws_credential_cmd.is_empty() {
            if let Some(migrated) = migrate_isengard_add_profile_to_credential_cmd(&aws_refresh_cmd)
            {
                aws_credential_cmd = migrated;
                aws_refresh_cmd.clear();
            }
        }
        let bedrock_concurrency = map
            .get("bedrock_concurrency")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(32)
            .clamp(1, 128);
        let bedrock_max_retries = map
            .get("bedrock_max_retries")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(3)
            .clamp(0, 12);
        let bedrock_retry_base_ms = map
            .get("bedrock_retry_base_ms")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(250)
            .clamp(50, 10_000);

        let local_embed_dim = map
            .get("local_embed_dim")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(384)
            .max(64);
        let max_chars_per_project = map
            .get("max_chars_per_project")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(12000);
        let max_files_per_project = map
            .get("max_files_per_project")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(2000)
            .clamp(1, 1_000_000);
        let max_chunks_per_project = map
            .get("max_chunks_per_project")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(6000)
            .clamp(1, 10_000_000);
        let max_chunks_per_file = map
            .get("max_chunks_per_file")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(28)
            .clamp(1, 100_000);
        let max_file_chars = map
            .get("max_file_chars")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(80_000)
            .clamp(1000, 50_000_000);
        let index_documents = parse_bool_config(&map, "index_documents", true);
        let max_document_bytes = map
            .get("max_document_bytes")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(200_000_000)
            .clamp(1000, 2_000_000_000);
        let max_document_uncompressed_bytes = map
            .get("max_document_uncompressed_bytes")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(documents::DEFAULT_MAX_DOCUMENT_UNCOMPRESSED_BYTES as i64)
            .clamp(1000, 20_000_000_000);
        let document_extract_timeout_ms = map
            .get("document_extract_timeout_ms")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(documents::DEFAULT_DOCUMENT_EXTRACT_TIMEOUT_MS as i64)
            .clamp(500, 600_000);
        let lance_compact_versions = map
            .get("lance_compact_versions")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(200)
            .clamp(0, 1_000_000);
        let lance_version_grace_secs = map
            .get("lance_version_grace_secs")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(900)
            .clamp(0, 86_400);
        // Candidate counts are clamped at load so a hand-edited config cannot make one query
        // read an unbounded number of vector blobs (the bounds are documented in the README
        // under "Retrieval Pipeline").
        let lexical_candidates = map
            .get("lexical_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120)
            .clamp(20, 1000);
        let vector_candidates = map
            .get("vector_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120)
            .clamp(20, 1000);
        let rank_chunk_semantic_weight = map
            .get("rank_chunk_semantic_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.66)
            .clamp(0.0, 1.0);
        let rank_chunk_lexical_weight = map
            .get("rank_chunk_lexical_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.24)
            .clamp(0.0, 1.0);
        let rank_chunk_graph_weight = map
            .get("rank_chunk_graph_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 1.0);
        let rank_quality_mix = map
            .get("rank_quality_mix")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.70)
            .clamp(0.0, 1.0);
        let rank_relation_quality_good_boost = map
            .get("rank_relation_quality_good_boost")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.08)
            .clamp(0.0, 1.0);
        let rank_relation_quality_weak_penalty = map
            .get("rank_relation_quality_weak_penalty")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.20)
            .clamp(0.0, 1.0);
        let rank_relation_quality_wrong_penalty = map
            .get("rank_relation_quality_wrong_penalty")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.65)
            .clamp(0.0, 1.0);
        let rank_project_content_weight = map
            .get("rank_project_content_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.58)
            .clamp(0.0, 2.0);
        let rank_project_semantic_weight = map
            .get("rank_project_semantic_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.14)
            .clamp(0.0, 2.0);
        let rank_project_path_weight = map
            .get("rank_project_path_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 2.0);
        let rank_project_graph_weight = map
            .get("rank_project_graph_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.10)
            .clamp(0.0, 2.0);
        let rank_project_frecency_weight = map
            .get("rank_project_frecency_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.08)
            .clamp(0.0, 2.0);
        let graph_seed_limit = map
            .get("graph_seed_limit")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(10)
            .clamp(2, 64);
        let graph_neighbor_limit = map
            .get("graph_neighbor_limit")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(100)
            .clamp(8, 500);
        let graph_same_project_high = map
            .get("graph_same_project_high")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.88)
            .clamp(0.0, 1.0);
        let graph_same_project_low = map
            .get("graph_same_project_low")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.76)
            .clamp(0.0, 1.0);
        let graph_related_base = map
            .get("graph_related_base")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.20)
            .clamp(0.0, 1.0);
        let graph_related_scale = map
            .get("graph_related_scale")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.70)
            .clamp(0.0, 2.0);
        let graph_related_cap = map
            .get("graph_related_cap")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.90)
            .clamp(0.0, 1.0);

        // HyDE config
        let hyde_enabled = map
            .get("hyde_enabled")
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(false); // opt-in, default off

        // Cross-encoder re-ranking config
        let reranker_enabled = map
            .get("reranker_enabled")
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(true);
        let reranker_model = map
            .get("reranker_model")
            .cloned()
            .unwrap_or_else(|| "qwen3:0.6b".to_string())
            .trim()
            .to_string();
        let reranker_pool_size = map
            .get("reranker_pool_size")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(60)
            .clamp(10, 200);
        let reranker_batch_size = map
            .get("reranker_batch_size")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, 32);
        let reranker_timeout_ms = map
            .get("reranker_timeout_ms")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(3000)
            .clamp(500, 30_000);

        // Freshness model (spec §4)
        let rank_recency_weight = map
            .get("rank_recency_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.12)
            .clamp(0.0, 0.5);
        let rank_recency_record_weight = map
            .get("rank_recency_record_weight")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.04)
            .clamp(0.0, 0.5);
        let recency_half_life_days = map
            .get("recency_half_life_days")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(21.0)
            .clamp(1.0, 3650.0);
        let recency_record_half_life_days = map
            .get("recency_record_half_life_days")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(90.0)
            .clamp(1.0, 3650.0);
        let recency_record_patterns = map
            .get("recency_record_patterns")
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| DEFAULT_RECENCY_RECORD_PATTERNS.to_string());

        // Hygiene (spec §7)
        let skip_dir_names = map
            .get("skip_dir_names")
            .map(|v| v.trim().to_string())
            .unwrap_or_default();

        // Proactive recall (spec §5)
        let recall_max_leads = map
            .get("recall_max_leads")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .clamp(1, 5);
        let recall_min_score_ratio = map
            .get("recall_min_score_ratio")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.80)
            .clamp(0.1, 1.0);
        // Raw-cosine floor for recall leads (slice 3). Calibrated on the private scorecard with
        // Titan Text Embeddings v2 (floor -> leads on the seven negative prompts / lead-hit on
        // 29 positives): 0.30 -> 13/14, 0.35 -> 7/14, 0.40 -> 4/14, 0.45 -> 4/12,
        // 0.50 -> 1/12, 0.55 -> 0/10. 0.40 keeps every positive lead the lower floors find,
        // and 0.45 is dominated by it (the same four negative leads, two positives fewer), so
        // 0.40 is the default. The negative leads that survive at 0.40 are instruction-shaped
        // prompts ("read the handoff", "run the tests") whose cosines overlap with real
        // questions; no floor separates them. The recall gate (slice 4) handles those by prompt
        // shape, not the floor. 0.1.x configs carry 0.40 from when this key compared a fusion
        // score; under the new meaning that value is the default, so they need no change.
        // A non-finite value (`nan`, `inf`) is rejected and the default applies; both floors are
        // clamped to [0, 1].
        let recall_min_abs_score = map
            .get("recall_min_abs_score")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.40)
            .clamp(0.0, 1.0);
        let search_min_abs_score = map
            .get("search_min_abs_score")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let recall_band_ratio = map
            .get("recall_band_ratio")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.90)
            .clamp(0.5, 1.0);
        let recall_roots = map
            .get("recall_roots")
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let recall_excerpts = parse_bool_config(&map, "recall_excerpts", true);
        let recall_system_message = parse_bool_config(&map, "recall_system_message", false);
        let mut recall_semantic = map
            .get("recall_semantic")
            .map(|v| v.trim().to_lowercase())
            .unwrap_or_else(|| "auto".to_string());
        if !matches!(recall_semantic.as_str(), "auto" | "on" | "off") {
            recall_semantic = "auto".to_string();
        }
        let mut recall_dossier = map
            .get("recall_dossier")
            .map(|v| v.trim().to_lowercase())
            .unwrap_or_else(|| "shadow".to_string());
        if !matches!(recall_dossier.as_str(), "shadow" | "auto" | "off") {
            recall_dossier = "shadow".to_string();
        }
        let recall_session_ttl_days = map
            .get("recall_session_ttl_days")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(3.0)
            .clamp(1.0, 30.0);

        Self {
            root: map
                .get("root")
                .map(|v| normalize_path(v))
                .unwrap_or_else(default_config_root),
            embed_backend,
            embed_model,
            aws_profile,
            aws_region,
            aws_refresh_cmd,
            aws_credential_cmd,
            bedrock_concurrency,
            bedrock_max_retries,
            bedrock_retry_base_ms,
            retrieval_backend,
            local_embed_dim,
            max_chars_per_project,
            max_files_per_project,
            max_chunks_per_project,
            max_chunks_per_file,
            max_file_chars,
            index_documents,
            max_document_bytes,
            max_document_uncompressed_bytes,
            document_extract_timeout_ms,
            lance_compact_versions,
            lance_version_grace_secs,
            lexical_candidates,
            vector_candidates,
            rank_chunk_semantic_weight,
            rank_chunk_lexical_weight,
            rank_chunk_graph_weight,
            rank_quality_mix,
            rank_relation_quality_good_boost,
            rank_relation_quality_weak_penalty,
            rank_relation_quality_wrong_penalty,
            rank_project_content_weight,
            rank_project_semantic_weight,
            rank_project_path_weight,
            rank_project_graph_weight,
            rank_project_frecency_weight,
            graph_seed_limit,
            graph_neighbor_limit,
            graph_same_project_high,
            graph_same_project_low,
            graph_related_base,
            graph_related_scale,
            graph_related_cap,
            hyde_enabled,
            reranker_enabled,
            reranker_model,
            reranker_pool_size,
            reranker_batch_size,
            reranker_timeout_ms,
            rank_recency_weight,
            rank_recency_record_weight,
            recency_half_life_days,
            recency_record_half_life_days,
            recency_record_patterns,
            skip_dir_names,
            recall_max_leads,
            recall_min_score_ratio,
            recall_min_abs_score,
            recall_band_ratio,
            search_min_abs_score,
            recall_roots,
            recall_excerpts,
            recall_system_message,
            recall_semantic,
            recall_dossier,
            recall_session_ttl_days,
        }
    }

    /// Path patterns (trimmed, non-empty) that classify a document as a record.
    pub(crate) fn record_patterns(&self) -> Vec<String> {
        split_csv_setting(&self.recency_record_patterns)
    }

    /// Extra directory names to skip during discovery and corpus walks.
    pub(crate) fn skip_dir_name_set(&self) -> HashSet<String> {
        split_csv_setting(&self.skip_dir_names)
            .into_iter()
            .collect()
    }

    /// Absolute roots that `retrivio recall` searches; empty means every tracked root.
    pub(crate) fn recall_root_list(&self) -> Vec<PathBuf> {
        split_csv_setting(&self.recall_roots)
            .iter()
            .map(|s| normalize_path(s))
            .collect()
    }
}

pub(crate) fn write_config_file(path: &Path, cfg: &ConfigValues) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating config dir: {}", e))?;
    }
    let lines = [
        format!("root = \"{}\"", toml_escape(&cfg.root.to_string_lossy())),
        format!("embed_backend = \"{}\"", toml_escape(&cfg.embed_backend)),
        format!("embed_model = \"{}\"", toml_escape(&cfg.embed_model)),
        format!("aws_profile = \"{}\"", toml_escape(&cfg.aws_profile)),
        format!("aws_region = \"{}\"", toml_escape(&cfg.aws_region)),
        format!(
            "aws_refresh_cmd = \"{}\"",
            toml_escape(&cfg.aws_refresh_cmd)
        ),
        format!(
            "aws_credential_cmd = \"{}\"",
            toml_escape(&cfg.aws_credential_cmd)
        ),
        format!("bedrock_concurrency = {}", cfg.bedrock_concurrency),
        format!("bedrock_max_retries = {}", cfg.bedrock_max_retries),
        format!("bedrock_retry_base_ms = {}", cfg.bedrock_retry_base_ms),
        format!(
            "retrieval_backend = \"{}\"",
            toml_escape(&cfg.retrieval_backend)
        ),
        format!("local_embed_dim = {}", cfg.local_embed_dim),
        format!("max_chars_per_project = {}", cfg.max_chars_per_project),
        format!("max_files_per_project = {}", cfg.max_files_per_project),
        format!("max_chunks_per_project = {}", cfg.max_chunks_per_project),
        format!("max_chunks_per_file = {}", cfg.max_chunks_per_file),
        format!("max_file_chars = {}", cfg.max_file_chars),
        format!("index_documents = {}", cfg.index_documents),
        format!("max_document_bytes = {}", cfg.max_document_bytes),
        format!(
            "max_document_uncompressed_bytes = {}",
            cfg.max_document_uncompressed_bytes
        ),
        format!(
            "document_extract_timeout_ms = {}",
            cfg.document_extract_timeout_ms
        ),
        format!("lance_compact_versions = {}", cfg.lance_compact_versions),
        format!(
            "lance_version_grace_secs = {}",
            cfg.lance_version_grace_secs
        ),
        format!("lexical_candidates = {}", cfg.lexical_candidates),
        format!("vector_candidates = {}", cfg.vector_candidates),
        format!(
            "rank_chunk_semantic_weight = {:.6}",
            cfg.rank_chunk_semantic_weight
        ),
        format!(
            "rank_chunk_lexical_weight = {:.6}",
            cfg.rank_chunk_lexical_weight
        ),
        format!(
            "rank_chunk_graph_weight = {:.6}",
            cfg.rank_chunk_graph_weight
        ),
        format!("rank_quality_mix = {:.6}", cfg.rank_quality_mix),
        format!(
            "rank_relation_quality_good_boost = {:.6}",
            cfg.rank_relation_quality_good_boost
        ),
        format!(
            "rank_relation_quality_weak_penalty = {:.6}",
            cfg.rank_relation_quality_weak_penalty
        ),
        format!(
            "rank_relation_quality_wrong_penalty = {:.6}",
            cfg.rank_relation_quality_wrong_penalty
        ),
        format!(
            "rank_project_content_weight = {:.6}",
            cfg.rank_project_content_weight
        ),
        format!(
            "rank_project_semantic_weight = {:.6}",
            cfg.rank_project_semantic_weight
        ),
        format!(
            "rank_project_path_weight = {:.6}",
            cfg.rank_project_path_weight
        ),
        format!(
            "rank_project_graph_weight = {:.6}",
            cfg.rank_project_graph_weight
        ),
        format!(
            "rank_project_frecency_weight = {:.6}",
            cfg.rank_project_frecency_weight
        ),
        format!("graph_seed_limit = {}", cfg.graph_seed_limit),
        format!("graph_neighbor_limit = {}", cfg.graph_neighbor_limit),
        format!(
            "graph_same_project_high = {:.6}",
            cfg.graph_same_project_high
        ),
        format!("graph_same_project_low = {:.6}", cfg.graph_same_project_low),
        format!("graph_related_base = {:.6}", cfg.graph_related_base),
        format!("graph_related_scale = {:.6}", cfg.graph_related_scale),
        format!("graph_related_cap = {:.6}", cfg.graph_related_cap),
        format!("hyde_enabled = {}", cfg.hyde_enabled),
        format!("reranker_enabled = {}", cfg.reranker_enabled),
        format!("reranker_model = \"{}\"", toml_escape(&cfg.reranker_model)),
        format!("reranker_pool_size = {}", cfg.reranker_pool_size),
        format!("reranker_batch_size = {}", cfg.reranker_batch_size),
        format!("reranker_timeout_ms = {}", cfg.reranker_timeout_ms),
        format!("rank_recency_weight = {:.6}", cfg.rank_recency_weight),
        format!(
            "rank_recency_record_weight = {:.6}",
            cfg.rank_recency_record_weight
        ),
        format!("recency_half_life_days = {:.6}", cfg.recency_half_life_days),
        format!(
            "recency_record_half_life_days = {:.6}",
            cfg.recency_record_half_life_days
        ),
        format!(
            "recency_record_patterns = \"{}\"",
            toml_escape(&cfg.recency_record_patterns)
        ),
        format!("skip_dir_names = \"{}\"", toml_escape(&cfg.skip_dir_names)),
        format!("recall_max_leads = {}", cfg.recall_max_leads),
        format!("recall_min_score_ratio = {:.6}", cfg.recall_min_score_ratio),
        format!("recall_min_abs_score = {:.6}", cfg.recall_min_abs_score),
        format!("search_min_abs_score = {:.6}", cfg.search_min_abs_score),
        format!("recall_band_ratio = {:.6}", cfg.recall_band_ratio),
        format!("recall_roots = \"{}\"", toml_escape(&cfg.recall_roots)),
        format!("recall_excerpts = {}", cfg.recall_excerpts),
        format!("recall_system_message = {}", cfg.recall_system_message),
        format!(
            "recall_semantic = \"{}\"",
            toml_escape(&cfg.recall_semantic)
        ),
        format!("recall_dossier = \"{}\"", toml_escape(&cfg.recall_dossier)),
        format!(
            "recall_session_ttl_days = {:.6}",
            cfg.recall_session_ttl_days
        ),
        String::new(),
    ];
    fs::write(path, lines.join("\n")).map_err(|e| format!("failed writing config: {}", e))
}

pub(crate) fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// What one scan skips and admits, from the config: `skip_dir_names` on top of the built-in
/// skip list, and `index_documents`. Built with [`ScanSettings::from_cfg`] once per run
/// (`index`, `refresh`, `prune`, `watch`, and every API and MCP refresh, which load the config
/// per request) and passed explicitly to discovery, the corpus walk, the scan-caps fingerprint
/// and the watcher filter. There is no process-wide copy: two runs in one process (the tests,
/// a long-running server whose config changed) never see each other's settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScanSettings {
    /// Directory names skipped in addition to the built-in list.
    pub(crate) extra_skip_dirs: HashSet<String>,
    /// Whether the document formats (see [`documents`]) are indexable.
    pub(crate) index_documents: bool,
}

impl Default for ScanSettings {
    /// What an empty config yields: the built-in skip list alone, documents on.
    fn default() -> Self {
        ScanSettings {
            extra_skip_dirs: HashSet::new(),
            index_documents: true,
        }
    }
}

impl ScanSettings {
    pub(crate) fn from_cfg(cfg: &ConfigValues) -> Self {
        ScanSettings {
            extra_skip_dirs: cfg.skip_dir_name_set(),
            index_documents: cfg.index_documents,
        }
    }

    /// A directory the walks never enter: a built-in name, a `.app` bundle or a configured one.
    pub(crate) fn is_skip_dir(&self, name: &str) -> bool {
        is_builtin_skip_dir(name) || self.extra_skip_dirs.contains(name)
    }

    /// Suffixes the indexer reads: the text and code formats always, the document formats
    /// when `index_documents` is on. Lower-case with the dot.
    pub(crate) fn is_indexable_suffix(&self, suffix: &str) -> bool {
        is_text_indexable_suffix(suffix)
            || (self.index_documents && documents::is_document_suffix(suffix))
    }

    /// A file the collector extracts (see [`documents`]) rather than reads as text.
    pub(crate) fn is_document_suffix(&self, suffix: &str) -> bool {
        self.index_documents && documents::is_document_suffix(suffix)
    }
}

/// Every suffix the indexer can read with documents on. The query classifier uses this, so a
/// query naming `deck.pptx` reads as a file name whatever `index_documents` says.
pub(crate) fn is_any_indexable_suffix(suffix: &str) -> bool {
    is_text_indexable_suffix(suffix) || documents::is_document_suffix(suffix)
}

pub(crate) fn is_builtin_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "__pycache__"
            | ".cache"
            | ".mypy_cache"
            | ".pytest_cache"
            | "node_modules"
            | "site-packages"
            | ".venv"
            | "venv"
            | ".idea"
            | "cdk.out"
            | ".next"
            | "dist"
            | "build"
            | "target"
            | "worktrees"
            | ".worktrees"
    ) || name.ends_with(".app")
}

/// Suffixes read as text (or code) without any extraction step.
pub(crate) fn is_text_indexable_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        ".md"
            | ".markdown"
            | ".txt"
            | ".rst"
            | ".adoc"
            | ".html"
            | ".htm"
            | ".py"
            | ".js"
            | ".ts"
            | ".tsx"
            | ".jsx"
            | ".go"
            | ".rs"
            | ".java"
            | ".c"
            | ".cc"
            | ".cpp"
            | ".h"
            | ".hpp"
            | ".sh"
            | ".bash"
            | ".zsh"
            | ".yaml"
            | ".yml"
            | ".toml"
            | ".json"
            | ".sql"
    )
}

/// The order in which a project's files are selected when a cap bites: 1 human documents
/// (notes and extracted document formats), 2 code, 3 config and data. `suffix` is lower-case
/// with its dot.
pub(crate) fn selection_tier(suffix: &str) -> u8 {
    match suffix {
        ".md" | ".markdown" | ".txt" | ".rst" | ".adoc" | ".html" | ".htm" => 1,
        s if documents::is_document_suffix(s) => 1,
        ".json" | ".yaml" | ".yml" | ".toml" | ".cfg" | ".ini" | ".sql" => 3,
        _ => 2,
    }
}

pub(crate) fn config_path(cwd: &Path) -> PathBuf {
    if let Some(path) = CLI_CONFIG_OVERRIDE.get() {
        return path.clone();
    }
    data_dir(cwd).join("config.toml")
}

pub(crate) fn db_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("retrivio.db")
}

pub(crate) fn data_dir(_cwd: &Path) -> PathBuf {
    // Tests never touch the real store: every test that needs a data directory installs one
    // through `test_support::TestStore`; anything else reaching here is a bug in the test.
    #[cfg(test)]
    {
        test_support::data_dir_for_test()
    }
    #[cfg(not(test))]
    {
        if let Some(path) = CLI_DATA_DIR_OVERRIDE.get() {
            return path.clone();
        }
        expand_tilde("~/.retrivio")
    }
}

pub(crate) fn load_config_values(path: &Path) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let data = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => return out,
    };
    for raw in data.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim().to_string();
        let mut val = v.trim().to_string();
        if let Some(comment_idx) = val.find('#') {
            val = val[..comment_idx].trim().to_string();
        }
        if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
            val = val[1..val.len() - 1].to_string();
        }
        out.insert(key, val);
    }
    if let Some(patterns) = out.get("recency_record_patterns") {
        warn_legacy_record_patterns(patterns);
    }
    out
}

/// One note per process, on stderr, when `recency_record_patterns` carries an entry written
/// for the pre-0.2.0 rule (a substring of the absolute path): since 0.2.0 patterns match path
/// components relative to the project, so `a/b` means consecutive directories and an absolute
/// or `~` path never matches.
pub(crate) fn warn_legacy_record_patterns(patterns: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let legacy: Vec<String> = split_csv_setting(patterns)
        .into_iter()
        .filter(|p| roles::legacy_record_pattern(p))
        .collect();
    if legacy.is_empty() || WARNED.swap(true, Ordering::SeqCst) {
        return;
    }
    eprintln!(
        "retrivio: recency_record_patterns {} matched as path components relative to the project since 0.2.0 (`a/b` = consecutive directories; an absolute or `~` path never matches), no longer as substrings of the absolute path; handoffs and docs/sessions are state regardless. Edit the key to silence this note.",
        legacy
            .iter()
            .map(|p| format!("\"{}\"", p))
            .collect::<Vec<_>>()
            .join(", ")
    );
}
