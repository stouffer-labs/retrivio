//! Cross-cutting contract tests: config defaults and clamps, chunk and relation JSON shapes,
//! MCP tool specs, the typo and picker helpers, shell splitting and the Bedrock payloads.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use rusqlite::Connection;

use crate::api::parse_since_days;
use crate::cli::likely_command_typo;
use crate::config::{
    config_rows, config_set_value, config_value_string, load_config_values, write_config_file,
    ConfigValues,
};
use crate::db::init_schema;
use crate::embed::{
    bedrock_embedding_space_key, default_embed_model_for_backend,
    migrate_isengard_add_profile_to_credential_cmd, parse_iso8601_to_unix, sigv4_authorize,
    sigv4_canonical_uri, unix_to_amz_date, uri_encode_path_segment, AwsCredentials,
    BedrockEmbedder, Embedder,
};
use crate::mcp::{mcp_tool_needs_rw, mcp_tool_specs};
use crate::pick::pick_query_changed;
use crate::rank::{
    chunk_get_schema, chunk_related_schema, chunk_search_schema, doc_read_schema, fts_or_query,
    mark_superseded_chunks, ranked_chunk_result_json, related_chunk_result_json, FreshnessCtx,
    RankedChunkResult, RelatedChunkResult,
};
use crate::related::{
    apply_chunk_relation_decision, context_pack_schema, list_chunk_relation_feedback,
    normalize_relation_quality_label, relation_feedback_row_json, set_chunk_relation_quality,
    source_chunk_json, ChunkRelationFeedbackRow, SourceChunk,
};
use crate::roles::{Role, TextShape};
use crate::util::{shell_split, strip_terminal_control_sequences, truncate_text_chars};

#[test]
fn freshness_and_recall_config_defaults() {
    let cfg = ConfigValues::from_map(std::collections::HashMap::new());
    assert!((cfg.rank_recency_weight - 0.12).abs() < 1e-12);
    assert!((cfg.rank_recency_record_weight - 0.04).abs() < 1e-12);
    assert_eq!(cfg.recency_half_life_days, 21.0);
    assert_eq!(cfg.recency_record_half_life_days, 90.0);
    assert_eq!(
        cfg.record_patterns(),
        vec![
            "transcript",
            "customer-signals",
            "meeting",
            "call-notes",
            ".srt"
        ]
    );
    assert!(cfg.skip_dir_name_set().is_empty());
    assert_eq!(cfg.recall_max_leads, 3);
    assert_eq!(cfg.recall_min_score_ratio, 0.80);
    assert_eq!(cfg.recall_min_abs_score, 0.40);
    assert_eq!(cfg.search_min_abs_score, 0.0);
    assert_eq!(cfg.recall_band_ratio, 0.90);
    assert_eq!(cfg.recall_roots, "");
    assert!(cfg.recall_root_list().is_empty());
    assert!(cfg.recall_excerpts);
    assert!(!cfg.recall_system_message);
    assert_eq!(cfg.recall_semantic, "auto");
    assert_eq!(cfg.recall_dossier, "shadow");
    assert_eq!(cfg.recall_session_ttl_days, 3.0);
}

/// A non-finite floor in the config is rejected (the default applies) and finite values
/// are clamped to [0, 1]: a floor must never be a number that every comparison fails.
#[test]
fn floors_reject_non_finite_values_and_clamp() {
    let with = |k: &str, v: &str| {
        let mut m = std::collections::HashMap::new();
        m.insert(k.to_string(), v.to_string());
        ConfigValues::from_map(m)
    };
    assert_eq!(
        with("recall_min_abs_score", "nan").recall_min_abs_score,
        0.40
    );
    assert_eq!(
        with("recall_min_abs_score", "NaN").recall_min_abs_score,
        0.40
    );
    assert_eq!(
        with("recall_min_abs_score", "inf").recall_min_abs_score,
        0.40
    );
    assert_eq!(
        with("recall_min_abs_score", "-inf").recall_min_abs_score,
        0.40
    );
    // Candidate counts are clamped to the documented range.
    assert_eq!(with("lexical_candidates", "5000").lexical_candidates, 1000);
    assert_eq!(with("vector_candidates", "1").vector_candidates, 20);
    assert_eq!(with("lexical_candidates", "300").lexical_candidates, 300);
    assert_eq!(with("reranker_pool_size", "999").reranker_pool_size, 200);
    assert_eq!(
        ConfigValues::from_map(std::collections::HashMap::new()).lexical_candidates,
        120
    );
    assert_eq!(with("recall_min_abs_score", "2").recall_min_abs_score, 1.0);
    assert_eq!(with("recall_min_abs_score", "-1").recall_min_abs_score, 0.0);
    assert_eq!(
        with("recall_min_abs_score", "0.5").recall_min_abs_score,
        0.5
    );
    assert_eq!(
        with("search_min_abs_score", "nan").search_min_abs_score,
        0.0
    );
    assert_eq!(
        with("search_min_abs_score", "inf").search_min_abs_score,
        0.0
    );
    assert_eq!(
        with("search_min_abs_score", "0.3").search_min_abs_score,
        0.3
    );
}

/// Every float setting rejects a non-finite value with one clear error and is left as it
/// was: `nan` passes `clamp` unchanged and `inf` pins to a bound the user never named, and
/// a written NaN is dropped at load with the default applied silently. Finite values still
/// parse and clamp as before.
#[test]
fn float_settings_reject_non_finite_values() {
    const FLOAT_KEYS: &[&str] = &[
        "rank_chunk_semantic_weight",
        "rank_chunk_lexical_weight",
        "rank_chunk_graph_weight",
        "rank_quality_mix",
        "rank_relation_quality_good_boost",
        "rank_relation_quality_weak_penalty",
        "rank_relation_quality_wrong_penalty",
        "rank_project_content_weight",
        "rank_project_semantic_weight",
        "rank_project_path_weight",
        "rank_project_graph_weight",
        "rank_project_frecency_weight",
        "graph_same_project_high",
        "graph_same_project_low",
        "graph_related_base",
        "graph_related_scale",
        "graph_related_cap",
        "rank_recency_weight",
        "rank_recency_record_weight",
        "recency_half_life_days",
        "recency_record_half_life_days",
        "recall_min_score_ratio",
        "recall_min_abs_score",
        "search_min_abs_score",
        "recall_band_ratio",
        "recall_session_ttl_days",
    ];
    let mut cfg = ConfigValues::from_map(std::collections::HashMap::new());
    for key in FLOAT_KEYS {
        let shown = config_value_string(&cfg, key)
            .unwrap_or_else(|| panic!("config_value_string missing {}", key));
        let before = format!("{:?}", cfg);
        for bad in [
            "nan",
            "NaN",
            "inf",
            "-inf",
            "+infinity",
            "Infinity",
            "abc",
            "",
        ] {
            let err = config_set_value(&mut cfg, key, bad)
                .expect_err(&format!("{} accepted {:?}", key, bad));
            assert_eq!(err, format!("{} must be a finite number", key), "{:?}", bad);
            assert_eq!(
                config_value_string(&cfg, key).unwrap(),
                shown,
                "{} {:?}",
                key,
                bad
            );
            assert_eq!(
                format!("{:?}", cfg),
                before,
                "nothing changed on {} {:?}",
                key,
                bad
            );
        }
    }
    // The two floors, which the load path also guards, still take finite values and clamp.
    config_set_value(&mut cfg, "recall_min_abs_score", "0.35").unwrap();
    assert_eq!(cfg.recall_min_abs_score, 0.35);
    config_set_value(&mut cfg, "recall_min_abs_score", "-3").unwrap();
    assert_eq!(cfg.recall_min_abs_score, 0.0);
    config_set_value(&mut cfg, "search_min_abs_score", "1e9").unwrap();
    assert_eq!(cfg.search_min_abs_score, 1.0);
    config_set_value(&mut cfg, "search_min_abs_score", " 0.4 ").unwrap();
    assert_eq!(cfg.search_min_abs_score, 0.4);
    // Written and read back, the floors are the finite values, never a default fallback.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tmp")
        .join(format!("test-config-finite-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create tmp config dir");
    let path = dir.join("config.toml");
    write_config_file(&path, &cfg).expect("write config");
    let back = ConfigValues::from_map(load_config_values(&path));
    assert_eq!(back.recall_min_abs_score, 0.0);
    assert_eq!(back.search_min_abs_score, 0.4);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn freshness_and_recall_config_clamps_set_show_and_roundtrip() {
    let mut map = std::collections::HashMap::new();
    map.insert("rank_recency_weight".to_string(), "0.9".to_string());
    map.insert("recency_half_life_days".to_string(), "0".to_string());
    map.insert("recall_max_leads".to_string(), "99".to_string());
    map.insert("recall_semantic".to_string(), "sometimes".to_string());
    map.insert("recall_excerpts".to_string(), "no".to_string());
    let cfg = ConfigValues::from_map(map);
    assert_eq!(cfg.rank_recency_weight, 0.5);
    assert_eq!(cfg.recency_half_life_days, 1.0);
    assert_eq!(cfg.recall_max_leads, 5);
    assert_eq!(cfg.recall_semantic, "auto");
    assert!(!cfg.recall_excerpts);

    let mut cfg = ConfigValues::from_map(std::collections::HashMap::new());
    for key in [
        "max_files_per_project",
        "max_chunks_per_project",
        "max_chunks_per_file",
        "max_file_chars",
        "index_documents",
        "max_document_bytes",
        "max_document_uncompressed_bytes",
        "document_extract_timeout_ms",
        "lance_compact_versions",
        "lance_version_grace_secs",
        "rank_recency_weight",
        "rank_recency_record_weight",
        "recency_half_life_days",
        "recency_record_half_life_days",
        "recency_record_patterns",
        "skip_dir_names",
        "recall_max_leads",
        "recall_min_score_ratio",
        "recall_min_abs_score",
        "search_min_abs_score",
        "recall_band_ratio",
        "recall_roots",
        "recall_excerpts",
        "recall_system_message",
        "recall_semantic",
        "recall_dossier",
        "recall_session_ttl_days",
        "hyde_enabled",
        "reranker_enabled",
        "reranker_model",
        "reranker_pool_size",
        "reranker_batch_size",
        "reranker_timeout_ms",
    ] {
        assert!(
            config_rows().iter().any(|(k, _)| *k == key),
            "config_rows missing {}",
            key
        );
        assert!(
            config_value_string(&cfg, key).is_some(),
            "config_value_string missing {}",
            key
        );
    }
    config_set_value(&mut cfg, "skip_dir_names", " demo-data, tmp ,marketplaces ").unwrap();
    assert_eq!(
        config_value_string(&cfg, "skip_dir_names").unwrap(),
        "demo-data,tmp,marketplaces"
    );
    assert!(config_set_value(&mut cfg, "skip_dir_names", "a/b").is_err());
    config_set_value(&mut cfg, "recall_semantic", "OFF").unwrap();
    assert_eq!(cfg.recall_semantic, "off");
    assert!(config_set_value(&mut cfg, "recall_semantic", "maybe").is_err());
    config_set_value(&mut cfg, "recall_dossier", "AUTO").unwrap();
    assert_eq!(cfg.recall_dossier, "auto");
    assert!(config_set_value(&mut cfg, "recall_dossier", "sometimes").is_err());
    config_set_value(&mut cfg, "recall_excerpts", "false").unwrap();
    assert!(!cfg.recall_excerpts);
    assert!(config_set_value(&mut cfg, "recall_excerpts", "maybe").is_err());
    config_set_value(&mut cfg, "rank_recency_weight", "2").unwrap();
    assert_eq!(cfg.rank_recency_weight, 0.5);
    config_set_value(&mut cfg, "reranker_enabled", "0").unwrap();
    assert!(!cfg.reranker_enabled);
    config_set_value(&mut cfg, "recall_roots", "~/a, ~/b").unwrap();
    assert_eq!(cfg.recall_root_list().len(), 2);

    // write_config_file -> load_config_values -> from_map preserves the new keys.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tmp")
        .join(format!("test-config-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create tmp config dir");
    let path = dir.join("config.toml");
    write_config_file(&path, &cfg).expect("write config");
    let back = ConfigValues::from_map(load_config_values(&path));
    assert_eq!(back.skip_dir_names, "demo-data,tmp,marketplaces");
    assert_eq!(back.recall_semantic, "off");
    assert_eq!(back.recall_dossier, "auto");
    assert!(!back.recall_excerpts);
    assert_eq!(back.rank_recency_weight, 0.5);
    assert!(!back.reranker_enabled);
    assert_eq!(back.recency_record_patterns, cfg.recency_record_patterns);
    assert_eq!(back.recall_roots, cfg.recall_roots);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn freshness_ctx_blends_by_class_and_filters_since() {
    let cfg = ConfigValues::from_map(std::collections::HashMap::new());
    let now = 1_800_000_000.0;
    let fx = FreshnessCtx::at(&cfg, now);
    let living = fx.info(
        "notes/plan.md",
        "/p/notes/plan.md",
        now - 21.0 * 86_400.0,
        TextShape::Prose,
    );
    assert_eq!(living.date_source, "mtime");
    assert!(!living.is_record);
    assert_eq!(living.role, Role::Knowledge);
    assert_eq!(living.tier, "aging");
    assert!(!living.verify);
    assert!((living.recency - 0.5).abs() < 1e-9);
    assert!((fx.blend(1.0, &living) - (0.88 + 0.12 * 0.5)).abs() < 1e-9);
    let record = fx.info(
        "transcripts/call.md",
        "/p/transcripts/call.md",
        now - 90.0 * 86_400.0,
        TextShape::Prose,
    );
    assert!(record.is_record);
    assert_eq!(record.role, Role::Record);
    assert_eq!(record.tier, "record");
    assert!((record.recency - 0.5).abs() < 1e-9);
    assert!((fx.blend(1.0, &record) - (0.96 + 0.04 * 0.5)).abs() < 1e-9);
    assert!((fx.recency_for(record.age_days, true) - record.recency).abs() < 1e-12);
    // State: living decay, `verify` past 35 days; the role never comes from the project
    // folder name (`202609-ai-handoff` here), only from the relative path.
    let state = fx.info(
        "docs/sessions/HANDOFF-2026-01-01.md",
        "/p/202609-ai-handoff/docs/sessions/HANDOFF-2026-01-01.md",
        now - 40.0 * 86_400.0,
        TextShape::Prose,
    );
    assert_eq!(state.role, Role::State);
    assert_eq!(state.tier, "verify");
    assert!(state.verify);
    assert!(!state.is_record);
    let plain = fx.info(
        "AGENTS.md",
        "/p/202609-ai-handoff/AGENTS.md",
        now - 1.0 * 86_400.0,
        TextShape::Prose,
    );
    assert_eq!(plain.role, Role::Knowledge);
    assert_eq!(plain.tier, "fresh");
    // Records take the event date from the path even when edited later.
    let event = fx.info(
        "customer-signals/Acme/20260715-call/transcript.txt",
        "/p/x/customer-signals/Acme/20260715-call/transcript.txt",
        now - 2.0 * 86_400.0,
        TextShape::Prose,
    );
    assert_eq!(event.date_source, "path-date");
    assert!(event.age_days > 60.0, "{}", event.age_days);
    assert!(fx.within_since(now - 5.0 * 86_400.0, Some(7.0)));
    assert!(!fx.within_since(now - 8.0 * 86_400.0, Some(7.0)));
    assert!(fx.within_since(now - 800.0 * 86_400.0, None));
    assert_eq!(
        fts_or_query(&[
            "alpha".to_string(),
            "con\"text".to_string(),
            " ".to_string()
        ]),
        "\"alpha\" OR \"con\"\"text\""
    );
    assert_eq!(parse_since_days(Some(&"30".to_string())), Some(30.0));
    assert_eq!(parse_since_days(Some(&"-1".to_string())), None);
    assert_eq!(parse_since_days(None), None);
}

#[test]
fn terminal_escape_sequences_are_stripped_from_prompt_input() {
    assert_eq!(strip_terminal_control_sequences("\u{1b}[A\u{1b}[A1"), "1");
    assert_eq!(strip_terminal_control_sequences("\u{1b}[B"), "");
    assert_eq!(strip_terminal_control_sequences("1\u{1b}[D"), "1");
}

#[test]
fn picker_query_change_detection_ignores_case_and_spacing() {
    assert!(!pick_query_changed(
        "speech to text",
        "  Speech   To   Text  "
    ));
    assert!(pick_query_changed("call", "speech to text"));
    assert!(pick_query_changed("call", ""));
}

#[test]
fn likely_command_typos_get_suggestions() {
    assert_eq!(likely_command_typo("setuo"), Some("setup"));
    assert_eq!(likely_command_typo("refesh"), Some("refresh"));
    assert_eq!(likely_command_typo("src/auth"), None);
    assert_eq!(likely_command_typo("stops"), None);
}

#[test]
fn chunk_schema_constants_are_stable() {
    assert_eq!(chunk_search_schema(), "chunk-search-v2");
    assert_eq!(chunk_related_schema(), "chunk-related-v1");
    assert_eq!(chunk_get_schema(), "chunk-get-v1");
    assert_eq!(doc_read_schema(), "doc-read-v1");
    assert_eq!(context_pack_schema(), "context-pack-v1");
}

fn state_chunk(path: &str, rel: &str, mtime: f64, score: f64) -> RankedChunkResult {
    RankedChunkResult {
        chunk_id: 1,
        chunk_index: 0,
        path: path.to_string(),
        project_path: "/r/p".to_string(),
        doc_rel_path: rel.to_string(),
        score,
        semantic: score,
        lexical: 0.0,
        graph: 0.0,
        relation: "direct".to_string(),
        quality: 1.0,
        excerpt: String::new(),
        doc_mtime: mtime,
        content_date: mtime,
        date_source: "mtime",
        age_days: 1.0,
        freshness_tier: "fresh".to_string(),
        is_record: false,
        role: "state",
        verify: false,
        noise: false,
        raw_similarity: Some(score),
        superseded_by: None,
        why: String::new(),
    }
}

/// Chunk results label the series head by the same rule as file search
/// (`mark_superseded`): the newest revision date (a date in the path beats a fresh mtime),
/// then the higher score of the file's best chunk, then the lexicographically later path.
#[test]
fn chunk_supersession_head_follows_the_file_rule() {
    let now = 1_800_000_000.0;
    let day = 86_400.0;
    // A June handoff edited today (newer mtime) stays behind September's (path date).
    let mut out = vec![
        state_chunk(
            "/r/p/docs/sessions/HANDOFF-2026-06-10.md",
            "docs/sessions/HANDOFF-2026-06-10.md",
            now,
            0.9,
        ),
        state_chunk(
            "/r/p/docs/sessions/HANDOFF-2026-09-01.md",
            "docs/sessions/HANDOFF-2026-09-01.md",
            now - 20.0 * day,
            0.5,
        ),
    ];
    mark_superseded_chunks(&mut out, now);
    assert_eq!(
        out[0].superseded_by.as_deref(),
        Some("/r/p/docs/sessions/HANDOFF-2026-09-01.md")
    );
    assert!(out[0].why.contains("superseded"));
    assert!(out[1].superseded_by.is_none());
    // Undated names with one mtime: the file whose best chunk scores higher is the head,
    // whatever the chunk order.
    let mut out = vec![
        state_chunk(
            "/r/p/notes/HANDOFF-final.md",
            "notes/HANDOFF-final.md",
            now,
            0.4,
        ),
        state_chunk(
            "/r/p/notes/HANDOFF-draft.md",
            "notes/HANDOFF-draft.md",
            now,
            0.7,
        ),
        state_chunk(
            "/r/p/notes/HANDOFF-final.md",
            "notes/HANDOFF-final.md",
            now,
            0.6,
        ),
    ];
    mark_superseded_chunks(&mut out, now);
    assert!(out[1].superseded_by.is_none());
    assert_eq!(
        out[0].superseded_by.as_deref(),
        Some("/r/p/notes/HANDOFF-draft.md")
    );
    assert_eq!(
        out[2].superseded_by.as_deref(),
        Some("/r/p/notes/HANDOFF-draft.md")
    );
    // Same revision date and score: the lexicographically later path is the head.
    let mut out = vec![
        state_chunk("/r/p/notes/HANDOFF-v2.md", "notes/HANDOFF-v2.md", now, 0.5),
        state_chunk("/r/p/notes/HANDOFF-v3.md", "notes/HANDOFF-v3.md", now, 0.5),
    ];
    mark_superseded_chunks(&mut out, now);
    assert_eq!(
        out[0].superseded_by.as_deref(),
        Some("/r/p/notes/HANDOFF-v3.md")
    );
    assert!(out[1].superseded_by.is_none());
}

#[test]
fn ranked_chunk_json_contract_fields() {
    let item = RankedChunkResult {
        chunk_id: 11,
        chunk_index: 3,
        path: "/tmp/a.md".to_string(),
        project_path: "/tmp".to_string(),
        doc_rel_path: "a.md".to_string(),
        score: 0.9,
        semantic: 0.8,
        lexical: 0.7,
        graph: 0.6,
        relation: "direct".to_string(),
        quality: 1.0,
        excerpt: "hello".to_string(),
        doc_mtime: 1_700_000_000.0,
        content_date: 1_700_000_000.0,
        date_source: "mtime",
        age_days: 3.5,
        freshness_tier: "fresh".to_string(),
        is_record: false,
        role: "knowledge",
        verify: false,
        noise: false,
        raw_similarity: Some(0.41),
        superseded_by: None,
        why: "semantic:0.41+lexical:0.70".to_string(),
    };
    let json = ranked_chunk_result_json(&item);
    let obj = json.as_object().expect("expected object");
    assert_eq!(obj.len(), 25);
    for key in [
        "chunk_id",
        "chunk_index",
        "path",
        "project_path",
        "doc_rel_path",
        "score",
        "semantic",
        "lexical",
        "graph",
        "relation",
        "quality",
        "excerpt",
        "doc_mtime",
        "content_date",
        "date_source",
        "age_days",
        "freshness_tier",
        "is_record",
        "role",
        "verify",
        "noise",
        "raw_similarity",
        "date_basis",
        "superseded_by",
        "why",
    ] {
        assert!(obj.contains_key(key), "missing key: {}", key);
    }
    assert_eq!(obj["role"], "knowledge");
    assert_eq!(obj["raw_similarity"], 0.41);
    assert_eq!(obj["date_basis"], "mtime");
    assert_eq!(obj["why"], "semantic:0.41+lexical:0.70");
    assert!(obj["superseded_by"].is_null());
}

#[test]
fn related_chunk_json_contract_fields() {
    let item = RelatedChunkResult {
        chunk_id: 19,
        chunk_index: 7,
        path: "/tmp/b.md".to_string(),
        project_path: "/tmp".to_string(),
        doc_rel_path: "b.md".to_string(),
        relation: "same_project".to_string(),
        relation_weight: 0.82,
        relation_quality: "good".to_string(),
        relation_quality_multiplier: 1.08,
        score: 0.88,
        semantic: 0.77,
        lexical: 0.66,
        quality: 0.95,
        excerpt: "world".to_string(),
        doc_mtime: 1_700_000_000.0,
        content_date: 1_700_000_000.0,
        date_source: "path-date",
        age_days: 40.0,
        freshness_tier: "verify".to_string(),
        is_record: false,
        role: "state",
        verify: true,
        noise: false,
        raw_similarity: Some(0.52),
        superseded_by: Some("/tmp/b2.md".to_string()),
        why: "semantic:0.52+superseded".to_string(),
    };
    let json = related_chunk_result_json(&item);
    let obj = json.as_object().expect("expected object");
    assert_eq!(obj.len(), 27);
    for key in [
        "content_date",
        "date_source",
        "date_basis",
        "age_days",
        "freshness_tier",
        "role",
        "verify",
        "noise",
        "raw_similarity",
        "superseded_by",
        "why",
    ] {
        assert!(obj.contains_key(key), "missing key: {}", key);
    }
    assert_eq!(obj["date_basis"], "path");
    assert_eq!(obj["superseded_by"], "/tmp/b2.md");
    for key in [
        "chunk_id",
        "chunk_index",
        "path",
        "project_path",
        "doc_rel_path",
        "relation",
        "relation_weight",
        "relation_quality",
        "relation_quality_multiplier",
        "score",
        "semantic",
        "lexical",
        "quality",
        "excerpt",
    ] {
        assert!(obj.contains_key(key), "missing key: {}", key);
    }
}

#[test]
fn relation_feedback_json_contract_fields() {
    let row = ChunkRelationFeedbackRow {
        src_chunk_id: 10,
        dst_chunk_id: 12,
        relation: "same_project".to_string(),
        decision: "active".to_string(),
        quality_label: "weak".to_string(),
        note: "test".to_string(),
        source: "unit".to_string(),
        created_at: 1.0,
        updated_at: 2.0,
        dst_chunk_index: 3,
        dst_doc_path: "/tmp/a.md".to_string(),
        dst_doc_rel_path: "a.md".to_string(),
        dst_project_path: "/tmp".to_string(),
    };
    let json = relation_feedback_row_json(&row);
    let obj = json.as_object().expect("expected object");
    assert!(obj.contains_key("quality_label"));
    assert_eq!(
        obj.get("quality_label")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "weak"
    );
}

#[test]
fn relation_quality_label_normalization_contract() {
    assert_eq!(normalize_relation_quality_label("good"), Some("good"));
    assert_eq!(normalize_relation_quality_label("weak"), Some("weak"));
    assert_eq!(normalize_relation_quality_label("wrong"), Some("wrong"));
    assert_eq!(
        normalize_relation_quality_label("clear"),
        Some("unspecified")
    );
    assert_eq!(normalize_relation_quality_label(""), Some("unspecified"));
    assert_eq!(normalize_relation_quality_label("bogus"), None);
}

#[test]
fn relation_feedback_quality_filters_and_preserves_decision() {
    let conn = Connection::open_in_memory().expect("open in-memory sqlite");
    init_schema(&conn).expect("init schema");
    conn.execute_batch(
        r#"
INSERT INTO projects(path, title, summary, project_mtime, last_indexed)
VALUES ('/tmp/p', 'p', 'p', 0, 0);
INSERT INTO project_chunks(project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (1, '/tmp/p/a.md', 'a.md', 0, 0, 10, 'h1', 'alpha', 0);
INSERT INTO project_chunks(project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at)
VALUES (1, '/tmp/p/b.md', 'b.md', 0, 1, 10, 'h2', 'beta', 0);
"#,
    )
    .expect("seed rows");

    apply_chunk_relation_decision(
        &conn,
        1,
        2,
        "same_project",
        "suppressed",
        "suppress first",
        "test",
        10.0,
    )
    .expect("suppress relation");
    let set_label = set_chunk_relation_quality(
        &conn,
        1,
        2,
        "same_project",
        "good",
        "quality set",
        "test",
        11.0,
    )
    .expect("set quality");
    assert_eq!(set_label, "good");

    let rows = list_chunk_relation_feedback(&conn, 1, Some("suppressed"), Some("good"), 20)
        .expect("query feedback");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].decision, "suppressed");
    assert_eq!(rows[0].quality_label, "good");

    let cleared =
        set_chunk_relation_quality(&conn, 1, 2, "same_project", "unspecified", "", "test", 12.0)
            .expect("clear quality");
    assert_eq!(cleared, "unspecified");
    let rows_after =
        list_chunk_relation_feedback(&conn, 1, None, Some("good"), 20).expect("query cleared");
    assert_eq!(rows_after.len(), 0);
}

#[test]
fn mcp_specs_include_relation_quality_tools() {
    let specs = mcp_tool_specs();
    let mut names: HashSet<String> = HashSet::new();
    for spec in specs {
        if let Some(name) = spec.get("name").and_then(|v| v.as_str()) {
            names.insert(name.to_string());
        }
    }
    assert!(names.contains("list_relation_feedback"));
    assert!(names.contains("set_relation_quality"));
    assert!(names.contains("read_chunk"));
    assert!(names.contains("read_document"));
    assert!(names.contains("pack_context"));
    assert!(names.contains("topic_dossier"));
    assert!(!mcp_tool_needs_rw("topic_dossier"), "the dossier is a read");
}

#[test]
fn source_chunk_json_contract_fields() {
    let source = SourceChunk {
        chunk_id: 5,
        chunk_index: 2,
        project_path: "/tmp".to_string(),
        doc_path: "/tmp/c.md".to_string(),
        doc_rel_path: "c.md".to_string(),
        text: "body".to_string(),
    };
    let json = source_chunk_json(&source);
    let obj = json.as_object().expect("expected object");
    assert_eq!(obj.len(), 5);
    for key in [
        "chunk_id",
        "chunk_index",
        "path",
        "project_path",
        "doc_rel_path",
    ] {
        assert!(obj.contains_key(key), "missing key: {}", key);
    }
}

#[test]
fn truncate_text_chars_contract() {
    let (full, full_truncated, full_chars) = truncate_text_chars("abcdef", 12);
    assert_eq!(full, "abcdef");
    assert!(!full_truncated);
    assert_eq!(full_chars, 6);

    let (clipped, clipped_truncated, clipped_chars) = truncate_text_chars("abcdef", 3);
    assert_eq!(clipped, "abc");
    assert!(clipped_truncated);
    assert_eq!(clipped_chars, 6);
}

#[test]
fn shell_split_handles_quoted_args() {
    let tokens = shell_split("foo 'bar baz' --qux \"quoted value\"").unwrap();
    assert_eq!(tokens, vec!["foo", "bar baz", "--qux", "quoted value"]);
}

#[test]
fn shell_split_rejects_unbalanced_quotes() {
    assert!(shell_split("foo 'bar").is_none());
    assert!(shell_split("foo \"bar").is_none());
}

#[test]
fn migrate_isengard_add_profile_rewrites_to_credentials() {
    let legacy =
        "'isengardcli' add-profile 'wwso-strategics-data-ai-fusion@amazon.com' --role Admin";
    let migrated = migrate_isengard_add_profile_to_credential_cmd(legacy).expect("should migrate");
    // shell_split should round-trip the migrated form back to the same tokens.
    let tokens = shell_split(&migrated).expect("migrated form must be parseable");
    assert_eq!(
        tokens,
        vec![
            "isengardcli",
            "credentials",
            "--awscli",
            "wwso-strategics-data-ai-fusion@amazon.com",
            "--role",
            "Admin"
        ]
    );
}

#[test]
fn migrate_isengard_add_profile_keeps_absolute_path() {
    let legacy = "/Users/x/Scripts/isengardcli/isengardcli add-profile foo@bar.com --role ReadOnly";
    let migrated = migrate_isengard_add_profile_to_credential_cmd(legacy).expect("should migrate");
    let tokens = shell_split(&migrated).expect("migrated form must be parseable");
    assert_eq!(
        tokens,
        vec![
            "/Users/x/Scripts/isengardcli/isengardcli",
            "credentials",
            "--awscli",
            "foo@bar.com",
            "--role",
            "ReadOnly"
        ]
    );
}

#[test]
fn migrate_isengard_add_profile_ignores_non_isengard() {
    // aws-sso login command should not be rewritten
    assert!(
        migrate_isengard_add_profile_to_credential_cmd("aws sso login --profile foo").is_none()
    );
    // isengardcli credentials (already correct) should not be rewritten
    assert!(migrate_isengard_add_profile_to_credential_cmd(
        "isengardcli credentials --awscli foo --role Admin"
    )
    .is_none());
    assert!(migrate_isengard_add_profile_to_credential_cmd("").is_none());
}

#[test]
fn config_migrates_legacy_aws_refresh_cmd_to_credential_cmd() {
    let mut map = std::collections::HashMap::new();
    map.insert("embed_backend".to_string(), "bedrock".to_string());
    map.insert(
        "aws_refresh_cmd".to_string(),
        "'isengardcli' add-profile 'foo@amazon.com' --role Admin".to_string(),
    );
    let cfg = ConfigValues::from_map(map);
    assert_eq!(cfg.aws_refresh_cmd, "");
    let tokens = shell_split(&cfg.aws_credential_cmd).expect("migrated form must be parseable");
    assert_eq!(
        tokens,
        vec![
            "isengardcli",
            "credentials",
            "--awscli",
            "foo@amazon.com",
            "--role",
            "Admin"
        ]
    );
}

#[test]
fn config_preserves_explicit_aws_credential_cmd_over_migration() {
    let mut map = std::collections::HashMap::new();
    map.insert("embed_backend".to_string(), "bedrock".to_string());
    map.insert(
        "aws_refresh_cmd".to_string(),
        "'isengardcli' add-profile 'foo@amazon.com' --role Admin".to_string(),
    );
    map.insert(
        "aws_credential_cmd".to_string(),
        "/usr/local/bin/my-creds.sh".to_string(),
    );
    let cfg = ConfigValues::from_map(map);
    // Explicit aws_credential_cmd wins; legacy aws_refresh_cmd is left alone.
    assert_eq!(cfg.aws_credential_cmd, "/usr/local/bin/my-creds.sh");
    assert_eq!(
        cfg.aws_refresh_cmd,
        "'isengardcli' add-profile 'foo@amazon.com' --role Admin"
    );
}

#[test]
fn bedrock_defaults_and_payload_contract() {
    assert_eq!(
        default_embed_model_for_backend("bedrock"),
        "amazon.titan-embed-text-v2:0"
    );
    assert_eq!(
        bedrock_embedding_space_key("amazon.titan-embed-text-v2:0"),
        "bedrock:amazon.titan-embed-text-v2:0"
    );
    assert_eq!(
        bedrock_embedding_space_key(""),
        "bedrock:amazon.titan-embed-text-v2:0"
    );
    let embedder = BedrockEmbedder::new("");
    let payload = embedder.request_payload("hello world");
    assert_eq!(
        payload
            .get("inputText")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "hello world"
    );
    assert_eq!(embedder.model_key(), "bedrock:amazon.titan-embed-text-v2:0");
}

#[test]
fn bedrock_response_parsing_contract() {
    let direct = serde_json::json!({"embedding": [0.1, 0.2, 0.3]});
    let parsed = BedrockEmbedder::parse_vector(&direct).expect("parse direct embedding");
    assert_eq!(parsed.len(), 3);

    let batched = serde_json::json!({"embeddings": [[0.4, 0.5]]});
    let parsed_batch = BedrockEmbedder::parse_vector(&batched).expect("parse batched embedding");
    assert_eq!(parsed_batch.len(), 2);
}

#[test]
fn bedrock_sigv4_date_conversion() {
    // 2024-01-15 12:30:00 UTC = 1705321800
    let (ts, ds) = unix_to_amz_date(1705321800);
    assert_eq!(ts, "20240115T123000Z");
    assert_eq!(ds, "20240115");

    // Unix epoch
    let (ts0, ds0) = unix_to_amz_date(0);
    assert_eq!(ts0, "19700101T000000Z");
    assert_eq!(ds0, "19700101");
}

#[test]
fn bedrock_iso8601_parsing() {
    let ts = parse_iso8601_to_unix("2024-01-15T12:30:00Z");
    assert_eq!(ts, Some(1705321800));

    let ts2 = parse_iso8601_to_unix("2024-01-15T12:30:00+00:00");
    assert_eq!(ts2, Some(1705321800));

    assert_eq!(parse_iso8601_to_unix("short"), None);
}

#[test]
fn bedrock_sigv4_signing_deterministic() {
    let creds = AwsCredentials {
        access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
        secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        session_token: None,
        expires_at: None,
    };
    let body = b"{\"inputText\":\"hello\",\"normalize\":true}";
    let headers = sigv4_authorize(
        "POST",
        "bedrock-runtime.us-east-1.amazonaws.com",
        "/model/amazon.titan-embed-text-v2%3A0/invoke",
        body,
        "us-east-1",
        "bedrock",
        &creds,
    );
    let auth = headers.iter().find(|(k, _)| k == "Authorization").unwrap();
    assert!(auth
        .1
        .starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"));
    assert!(auth
        .1
        .contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date"));
}

#[test]
fn bedrock_uri_encoding() {
    assert_eq!(
        uri_encode_path_segment("amazon.titan-embed-text-v2:0"),
        "amazon.titan-embed-text-v2%3A0"
    );
    assert_eq!(uri_encode_path_segment("simple"), "simple");

    // SigV4 canonical URI double-encodes the already-encoded path
    let url_path = "/model/amazon.titan-embed-text-v2%3A0/invoke";
    let canonical = sigv4_canonical_uri(url_path);
    assert_eq!(canonical, "/model/amazon.titan-embed-text-v2%253A0/invoke");
}

#[test]
fn bedrock_batch_parse_vectors() {
    let cohere = serde_json::json!({"embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]});
    let vecs = BedrockEmbedder::parse_vectors(&cohere).expect("parse cohere batch");
    assert_eq!(vecs.len(), 3);
    assert_eq!(vecs[2], vec![0.5f32, 0.6f32]);

    let titan = serde_json::json!({"embedding": [0.7, 0.8]});
    let vecs_t = BedrockEmbedder::parse_vectors(&titan).expect("parse titan single");
    assert_eq!(vecs_t.len(), 1);
}
