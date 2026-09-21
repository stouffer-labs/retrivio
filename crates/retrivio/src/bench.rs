//! The bench command: dataset planning, doctor and chunk-dataset export.

use std::ffi::OsString;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::{env, fs, process};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use rusqlite::{params, Connection};

use crate::config::{config_path, data_dir, db_path, load_config_values, ConfigValues};
use crate::db::{open_db_read_only, open_db_rw};
use crate::embed::model_key_for_cfg;
use crate::util::{arg_value, normalize_path, now_ts};

#[derive(Clone, Debug)]
pub(crate) struct BenchDatasetMeta {
    created_at: f64,
    source_db: String,
    model: String,
    dim: i64,
    chunks: i64,
}

pub(crate) fn run_bench_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_bench_help();
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bench_root = data_dir(&cwd).join("bench");
    fs::create_dir_all(&bench_root).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to create bench dir '{}': {}",
            bench_root.display(),
            e
        );
        process::exit(1);
    });

    let mut action = "plan".to_string();
    let mut action_set = false;
    let mut model_key_opt: Option<String> = None;
    let mut export_limit: Option<usize> = None;
    let mut dataset_path_opt: Option<PathBuf> = None;
    let mut queries_path_opt: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if !action_set && !s.starts_with('-') && matches!(s.as_str(), "plan" | "doctor" | "export")
        {
            action = s;
            action_set = true;
            i += 1;
            continue;
        }
        match s.as_str() {
            "--model-key" => {
                i += 1;
                model_key_opt = Some(arg_value(args, i, "--model-key").trim().to_string());
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
                export_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            "--dataset" => {
                i += 1;
                dataset_path_opt = Some(normalize_path(&arg_value(args, i, "--dataset")));
            }
            "--queries" => {
                i += 1;
                queries_path_opt = Some(normalize_path(&arg_value(args, i, "--queries")));
            }
            other if other.starts_with("--model-key=") => {
                model_key_opt = Some(other.trim_start_matches("--model-key=").trim().to_string());
            }
            other if other.starts_with("--limit=") => {
                let raw = other.trim_start_matches("--limit=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
                export_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other if other.starts_with("--dataset=") => {
                let raw = other.trim_start_matches("--dataset=").trim();
                dataset_path_opt = Some(normalize_path(raw));
            }
            other if other.starts_with("--queries=") => {
                let raw = other.trim_start_matches("--queries=").trim();
                queries_path_opt = Some(normalize_path(raw));
            }
            other => {
                eprintln!("error: unknown bench action/option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let dataset_path = dataset_path_opt.unwrap_or_else(|| bench_root.join("dataset.jsonl"));
    let queries_path = queries_path_opt.unwrap_or_else(|| bench_root.join("queries.txt"));
    let meta_path = bench_meta_path(&dataset_path);

    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let model_key = model_key_opt.unwrap_or_else(|| default_bench_model_key(&cfg));

    match action.as_str() {
        "plan" => {
            println!("benchmark plan: docs/BENCHMARK_PLAN.md");
            println!("bench dir: {}", bench_root.display());
            println!("next:");
            println!("  retrivio bench doctor");
            println!("  retrivio bench export");
        }
        "doctor" => {
            let cfg_path = config_path(&cwd);
            let dbp = db_path(&cwd);
            println!("config: {}", cfg_path.display());
            println!("db: {}", dbp.display());
            println!("bench dir: {}", bench_root.display());
            println!("model key: {}", model_key);

            if dbp.exists() {
                match open_db_read_only(&dbp) {
                    Ok(conn) => {
                        let chunks: i64 = conn
                            .query_row("SELECT COUNT(*) FROM project_chunks", [], |row| row.get(0))
                            .unwrap_or(0);
                        let vectors: i64 = conn
                            .query_row(
                                "SELECT COUNT(*) FROM project_chunk_vectors WHERE model = ?1",
                                params![model_key.clone()],
                                |row| row.get(0),
                            )
                            .unwrap_or(0);
                        println!("chunks in sqlite: {}", chunks);
                        println!("chunk vectors for model: {}", vectors);
                    }
                    Err(err) => println!("sqlite: error ({})", err),
                }
            } else {
                println!("sqlite: error (database file missing)");
            }
            println!(
                "queries file: {} ({})",
                queries_path.display(),
                if queries_path.exists() {
                    "exists"
                } else {
                    "missing"
                }
            );
            println!("retrieval backend: lancedb (embedded)");
        }
        "export" => {
            write_default_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let dbp = db_path(&cwd);
            let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let meta = export_chunk_dataset_native(
                &conn,
                &dbp,
                &model_key,
                &dataset_path,
                &meta_path,
                export_limit,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            println!("dataset: {}", dataset_path.display());
            println!("meta: {}", meta_path.display());
            println!("chunks: {}", meta.chunks);
            println!("dim: {}", meta.dim);
            println!("model: {}", meta.model);
            println!("queries: {}", queries_path.display());
        }
        other => {
            eprintln!("error: unknown bench action '{}'", other);
            process::exit(2);
        }
    }
}

pub(crate) fn print_bench_help() {
    println!("usage: retrivio bench [plan|doctor|export] [options]");
    println!("options:");
    println!("  --model-key <key>        Vector model key for dataset export");
    println!("  --limit <n>              Optional max rows during export (0=all)");
    println!("  --dataset <path>         Dataset JSONL path");
    println!("  --queries <path>         Queries file path");
}

pub(crate) fn default_bench_model_key(cfg: &ConfigValues) -> String {
    let backend = cfg.embed_backend.trim().to_lowercase();
    if backend == "hash" {
        return model_key_for_cfg(cfg);
    }
    if backend == "ollama" {
        let model = if cfg.embed_model.trim().is_empty() {
            "qwen3-embedding".to_string()
        } else {
            cfg.embed_model.trim().to_string()
        };
        return format!("ollama:{}", model);
    }
    if cfg.embed_model.trim().is_empty() {
        format!("{}:{}", backend, "qwen3-embedding")
    } else {
        format!("{}:{}", backend, cfg.embed_model.trim())
    }
}

pub(crate) fn bench_meta_path(dataset_path: &Path) -> PathBuf {
    dataset_path.with_extension("meta.json")
}

pub(crate) fn write_default_bench_queries(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create queries directory '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    let body = [
        "# One query per line. Lines starting with # are ignored.",
        "storage replication",
        "semantic layer",
        "auth flow",
        "inference pricing",
        "vector search",
    ]
    .join("\n")
        + "\n";
    fs::write(path, body)
        .map_err(|e| format!("failed writing default queries '{}': {}", path.display(), e))
}

pub(crate) fn export_chunk_dataset_native(
    conn: &Connection,
    source_db: &Path,
    model_key: &str,
    out_jsonl: &Path,
    out_meta_json: &Path,
    limit: Option<usize>,
) -> Result<BenchDatasetMeta, String> {
    if let Some(parent) = out_jsonl.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating dataset dir '{}': {}", parent.display(), e))?;
    }
    if let Some(parent) = out_meta_json.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating metadata dir '{}': {}", parent.display(), e))?;
    }

    let sql = if limit.is_some() {
        r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text,
    pc.chunk_index,
    pcv.dim,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
ORDER BY pc.id
LIMIT ?2
"#
    } else {
        r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.text,
    pc.chunk_index,
    pcv.dim,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
ORDER BY pc.id
"#
    };

    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| format!("failed preparing benchmark export query: {}", e))?;
    let mut rows = if let Some(v) = limit {
        stmt.query(params![model_key, v as i64])
            .map_err(|e| format!("failed running benchmark export query: {}", e))?
    } else {
        stmt.query(params![model_key])
            .map_err(|e| format!("failed running benchmark export query: {}", e))?
    };

    let out_file = fs::File::create(out_jsonl).map_err(|e| {
        format!(
            "failed creating dataset file '{}': {}",
            out_jsonl.display(),
            e
        )
    })?;
    let mut writer = BufWriter::new(out_file);
    let mut dim = 0i64;
    let mut chunks = 0i64;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed reading benchmark export row: {}", e))?
    {
        let chunk_id: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading chunk_id: {}", e))?;
        let project_path: String = row
            .get(1)
            .map_err(|e| format!("failed reading project_path: {}", e))?;
        let doc_path: String = row
            .get(2)
            .map_err(|e| format!("failed reading doc_path: {}", e))?;
        let doc_rel_path: String = row
            .get(3)
            .map_err(|e| format!("failed reading doc_rel_path: {}", e))?;
        let text: String = row
            .get(4)
            .map_err(|e| format!("failed reading text: {}", e))?;
        let chunk_index: i64 = row
            .get(5)
            .map_err(|e| format!("failed reading chunk_index: {}", e))?;
        let row_dim: i64 = row
            .get(6)
            .map_err(|e| format!("failed reading dim: {}", e))?;
        let vector_blob: Vec<u8> = row
            .get(7)
            .map_err(|e| format!("failed reading vector blob: {}", e))?;

        if dim <= 0 {
            dim = row_dim.max(1);
        }
        let rec = serde_json::json!({
            "chunk_id": chunk_id,
            "project_path": project_path,
            "doc_path": doc_path,
            "doc_rel_path": doc_rel_path,
            "chunk_index": chunk_index,
            "text": text,
            "vector_b64_f32": BASE64_STANDARD.encode(vector_blob),
        });
        writer
            .write_all(rec.to_string().as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|e| format!("failed writing dataset record: {}", e))?;
        chunks += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("failed flushing dataset output: {}", e))?;

    if chunks == 0 {
        return Err(format!(
            "No chunk vectors found for model key '{}'. Run `retrivio index` first or pass --model-key.",
            model_key
        ));
    }

    let meta = BenchDatasetMeta {
        created_at: now_ts(),
        source_db: source_db.to_string_lossy().to_string(),
        model: model_key.to_string(),
        dim: dim.max(1),
        chunks,
    };
    let meta_json = serde_json::json!({
        "created_at": meta.created_at,
        "source_db": meta.source_db,
        "model": meta.model,
        "dim": meta.dim,
        "chunks": meta.chunks,
    });
    let text = serde_json::to_string_pretty(&meta_json)
        .map_err(|e| format!("failed serializing dataset metadata: {}", e))?;
    fs::write(out_meta_json, format!("{}\n", text)).map_err(|e| {
        format!(
            "failed writing metadata '{}': {}",
            out_meta_json.display(),
            e
        )
    })?;
    Ok(meta)
}
