//! Embedding backends: the Embedder trait, backend selection, runtime metrics, hook mode and
//! the query-embedding cache. `bedrock`, `ollama` and `hash` hold the three backends.

use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use rusqlite::{params, Connection};

use crate::config::{db_path, ConfigValues};
use crate::db::{open_db_read_only, open_db_side_writer};
use crate::util::{blob_to_f32_vec, f32_blob, now_ts};

mod bedrock;
mod hash;
mod ollama;

pub(crate) use bedrock::*;
pub(crate) use hash::*;
pub(crate) use ollama::*;

/// Set by `retrivio recall` (editor hook): credential refresh commands are never spawned and
/// the non-interactive credential export is time-boxed. Process-global and one-way.
pub(crate) static HOOK_MODE: AtomicBool = AtomicBool::new(false);

/// Enter hook mode for the rest of the process (see [`HOOK_MODE`]).
pub(crate) fn set_hook_mode() {
    HOOK_MODE.store(true, Ordering::SeqCst);
}

pub(crate) fn hook_mode_active() -> bool {
    HOOK_MODE.load(Ordering::SeqCst)
}

#[cfg(test)]
mod hook_mode_tests {
    use super::*;
    use crate::config::ConfigValues;
    use std::collections::HashMap;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    fn hook_mode_never_spawns_the_refresh_command() {
        set_hook_mode();
        assert!(hook_mode_active());
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("aws_refresh_cmd".to_string(), "sleep 30".to_string());
        let cfg = ConfigValues::from_map(map);
        assert_eq!(cfg.aws_refresh_cmd, "sleep 30");
        let started = Instant::now();
        let err = refresh_aws_credentials_if_configured(Some(&cfg)).unwrap_err();
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "took {:?}",
            started.elapsed()
        );
        assert!(err.contains("hook mode"), "{}", err);
        let started = Instant::now();
        assert!(run_refresh_command_once("sleep 30").is_err());
        assert!(started.elapsed() < Duration::from_millis(100));
        // An empty command is still a no-op.
        assert!(run_refresh_command_once("   ").is_ok());
    }

    #[test]
    fn bounded_credential_command_is_killed_on_timeout() {
        let mut slow = Command::new("sleep");
        slow.arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let err = bounded_command_output(&mut slow, Duration::from_millis(150)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "took {:?}",
            started.elapsed()
        );

        let mut fast = Command::new("sh");
        fast.arg("-c")
            .arg("printf '{\"AccessKeyId\":\"x\"}'")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let out = bounded_command_output(&mut fast, Duration::from_secs(5)).unwrap();
        assert!(out.status.success());
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"AccessKeyId\":\"x\"}"
        );

        // A grandchild that inherited the pipe does not stall the read past the budget.
        let mut orphan = Command::new("sh");
        orphan
            .arg("-c")
            .arg("sleep 2 & printf ok")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let err = bounded_command_output(&mut orphan, Duration::from_millis(300)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert!(
            started.elapsed() < Duration::from_millis(1500),
            "took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn hook_mode_disables_daemon_autostart_and_cli_fallback() {
        set_hook_mode();
        let started = Instant::now();
        assert_eq!(maybe_autostart_ollama("http://127.0.0.1:1"), Ok(false));
        assert!(started.elapsed() < Duration::from_millis(100));
        let embedder = BedrockEmbedder::new("amazon.titan-embed-text-v2:0");
        let err = embedder
            .invoke_model_cli(&serde_json::json!({ "inputText": "x" }))
            .unwrap_err();
        assert!(err.contains("hook mode"), "{}", err);
    }
}

#[derive(Default)]
pub(crate) struct EmbedRuntimeMetrics {
    requests_started: AtomicU64,
    requests_succeeded: AtomicU64,
    requests_failed: AtomicU64,
    request_retries: AtomicU64,
    throttles: AtomicU64,
    texts_embedded: AtomicU64,
    in_flight: AtomicI64,
    latency_sum_ms: AtomicU64,
    latency_samples: AtomicU64,
    latency_max_ms: AtomicU64,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct EmbedRuntimeSnapshot {
    pub(crate) requests_succeeded: u64,
    pub(crate) requests_failed: u64,
    pub(crate) request_retries: u64,
    pub(crate) throttles: u64,
    pub(crate) texts_embedded: u64,
    pub(crate) in_flight: i64,
    pub(crate) latency_sum_ms: u64,
    pub(crate) latency_samples: u64,
    pub(crate) latency_max_ms: u64,
}

pub(crate) fn embed_runtime_metrics() -> &'static EmbedRuntimeMetrics {
    static METRICS: OnceLock<EmbedRuntimeMetrics> = OnceLock::new();
    METRICS.get_or_init(EmbedRuntimeMetrics::default)
}

pub(crate) fn atomic_update_max(dst: &AtomicU64, value: u64) {
    let mut cur = dst.load(Ordering::Relaxed);
    while value > cur {
        match dst.compare_exchange_weak(cur, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => cur = next,
        }
    }
}

pub(crate) fn reset_embed_runtime_metrics() {
    let m = embed_runtime_metrics();
    m.requests_started.store(0, Ordering::Relaxed);
    m.requests_succeeded.store(0, Ordering::Relaxed);
    m.requests_failed.store(0, Ordering::Relaxed);
    m.request_retries.store(0, Ordering::Relaxed);
    m.throttles.store(0, Ordering::Relaxed);
    m.texts_embedded.store(0, Ordering::Relaxed);
    m.in_flight.store(0, Ordering::Relaxed);
    m.latency_sum_ms.store(0, Ordering::Relaxed);
    m.latency_samples.store(0, Ordering::Relaxed);
    m.latency_max_ms.store(0, Ordering::Relaxed);
}

pub(crate) fn embed_metric_request_start() {
    let m = embed_runtime_metrics();
    m.requests_started.fetch_add(1, Ordering::Relaxed);
    m.in_flight.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn embed_metric_request_end(success: bool, elapsed: Duration) {
    let m = embed_runtime_metrics();
    if success {
        m.requests_succeeded.fetch_add(1, Ordering::Relaxed);
    } else {
        m.requests_failed.fetch_add(1, Ordering::Relaxed);
    }
    m.in_flight.fetch_sub(1, Ordering::Relaxed);
    let elapsed_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
    m.latency_sum_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
    m.latency_samples.fetch_add(1, Ordering::Relaxed);
    atomic_update_max(&m.latency_max_ms, elapsed_ms);
}

pub(crate) fn embed_metric_retry() {
    embed_runtime_metrics()
        .request_retries
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn embed_metric_throttle() {
    embed_runtime_metrics()
        .throttles
        .fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn embed_metric_texts(count: usize) {
    embed_runtime_metrics()
        .texts_embedded
        .fetch_add(count as u64, Ordering::Relaxed);
}

pub(crate) fn embed_runtime_snapshot() -> EmbedRuntimeSnapshot {
    let m = embed_runtime_metrics();
    EmbedRuntimeSnapshot {
        requests_succeeded: m.requests_succeeded.load(Ordering::Relaxed),
        requests_failed: m.requests_failed.load(Ordering::Relaxed),
        request_retries: m.request_retries.load(Ordering::Relaxed),
        throttles: m.throttles.load(Ordering::Relaxed),
        texts_embedded: m.texts_embedded.load(Ordering::Relaxed),
        in_flight: m.in_flight.load(Ordering::Relaxed),
        latency_sum_ms: m.latency_sum_ms.load(Ordering::Relaxed),
        latency_samples: m.latency_samples.load(Ordering::Relaxed),
        latency_max_ms: m.latency_max_ms.load(Ordering::Relaxed),
    }
}

pub(crate) fn default_embed_model_for_backend(backend: &str) -> &'static str {
    match backend {
        "bedrock" => "amazon.titan-embed-text-v2:0",
        "hash" => HASH_BACKEND_MODEL,
        _ => "qwen3-embedding",
    }
}

/// Backends `retrivio index` can embed with.
pub(crate) const EMBED_BACKENDS: &[&str] = &["ollama", "bedrock", "hash"];

pub(crate) fn is_known_embed_backend(name: &str) -> bool {
    EMBED_BACKENDS.contains(&name)
}

pub(crate) fn ensure_native_embed_backend(cfg: &ConfigValues, context: &str) -> Result<(), String> {
    match cfg.embed_backend.as_str() {
        "ollama" | "hash" => Ok(()),
        "bedrock" => bedrock_preflight_credentials(cfg, context),
        other => Err(format!(
            "{} requires native embed_backend in [{}] (current='{}')",
            context,
            EMBED_BACKENDS.join(", "),
            other
        )),
    }
}

#[derive(Clone)]
pub(crate) struct QueryEmbedCacheEntry {
    cached_at: Instant,
    vector: Vec<f32>,
}

pub(crate) static QUERY_EMBED_CACHE: OnceLock<Mutex<HashMap<String, QueryEmbedCacheEntry>>> =
    OnceLock::new();

pub(crate) fn query_embed_cache() -> &'static Mutex<HashMap<String, QueryEmbedCacheEntry>> {
    QUERY_EMBED_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn query_embed_cache_limit() -> usize {
    env::var("RETRIVIO_QUERY_EMBED_CACHE_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(4096)
        .clamp(64, 100_000)
}

pub(crate) fn query_embed_cache_ttl() -> Duration {
    let sec = env::var("RETRIVIO_QUERY_EMBED_CACHE_TTL_SEC")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(3600); // 1 hour (was 15 minutes)
    Duration::from_secs(sec.clamp(15, 86_400))
}

pub(crate) fn model_key_for_cfg(cfg: &ConfigValues) -> String {
    match cfg.embed_backend.as_str() {
        "ollama" => {
            let model = if cfg.embed_model.trim().is_empty() {
                "qwen3-embedding".to_string()
            } else {
                cfg.embed_model.trim().to_string()
            };
            format!("ollama:{}", model)
        }
        "bedrock" => bedrock_embedding_space_key(&cfg.embed_model),
        "hash" => hash_model_key(cfg.local_embed_dim),
        other => {
            let model = if cfg.embed_model.trim().is_empty() {
                "qwen3-embedding".to_string()
            } else {
                cfg.embed_model.trim().to_string()
            };
            format!("{}:{}", other, model)
        }
    }
}

pub(crate) fn bedrock_embedding_space_key(model: &str) -> String {
    let normalized = if model.trim().is_empty() {
        default_embed_model_for_backend("bedrock").to_string()
    } else {
        model.trim().to_string()
    };
    format!("bedrock:{}", normalized)
}

/// Key of the query-embedding cache, in memory and on disk: SHA-256 over the model key and
/// the normalised (trimmed, ASCII-lowercased) query, as hex. The cache never holds the query
/// text itself; two queries that differ only in case or surrounding whitespace share a key.
pub(crate) fn query_cache_key(model_key: &str, normalized_query: &str) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    h.update(model_key.as_bytes());
    h.update([0u8]);
    h.update(normalized_query.as_bytes());
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

pub(crate) fn embed_query_cached(
    cfg: &ConfigValues,
    query: &str,
) -> Result<(String, Vec<f32>), String> {
    let q = query.trim();
    if q.is_empty() {
        return Err("query embedding requested for empty query".to_string());
    }
    let model_key_guess = model_key_for_cfg(cfg);
    let normalized_query = q.to_ascii_lowercase();
    let cache_key_guess = query_cache_key(&model_key_guess, &normalized_query);
    let ttl = query_embed_cache_ttl();
    {
        let mut cache = query_embed_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = cache.get(&cache_key_guess) {
            if entry.cached_at.elapsed() <= ttl {
                return Ok((model_key_guess, entry.vector.clone()));
            }
        }
        cache.retain(|_, entry| entry.cached_at.elapsed() <= ttl);
    }

    // Persistent disk cache (survives process restarts). Read-only and never migrating: this
    // runs inside `recall` on every prompt, and a store that predates the cache table or is
    // absent simply misses.
    let db_p = db_path(&cfg.root);
    if let Ok(conn) = open_db_read_only(&db_p) {
        if let Ok(cached_vec) = disk_cache_lookup(&conn, &normalized_query, &model_key_guess) {
            // Found in disk cache — populate in-memory cache and return
            let cache_key = cache_key_guess.clone();
            let mut cache = query_embed_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.len() >= query_embed_cache_limit() {
                cache.clear();
            }
            cache.insert(
                cache_key,
                QueryEmbedCacheEntry {
                    cached_at: Instant::now(),
                    vector: cached_vec.clone(),
                },
            );
            return Ok((model_key_guess, cached_vec));
        }
    }

    let embedder = build_embedder(cfg)?;
    let model_key = embedder.model_key();
    let vector = embedder.embed_query(q)?;
    let cache_key = query_cache_key(&model_key, &normalized_query);
    {
        let mut cache = query_embed_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.len() >= query_embed_cache_limit() {
            cache.clear();
        }
        cache.insert(
            cache_key,
            QueryEmbedCacheEntry {
                cached_at: Instant::now(),
                vector: vector.clone(),
            },
        );
    }

    // Best-effort write to the disk cache on a non-migrating connection with a short busy
    // timeout: an absent store, an old table shape or a watcher holding the write lock is
    // skipped silently (a search or the recall hook never waits on the cache write).
    if db_p.is_file() {
        if let Ok(conn) = open_db_side_writer(&db_p) {
            let _ = disk_cache_store(&conn, &normalized_query, &model_key, &vector);
        }
    }

    Ok((model_key, vector))
}

/// Lookup a query embedding from the persistent SQLite cache. The row is found by the hash
/// of (model key, normalised query); a store whose table still has the pre-0.2.1 shape
/// (`query_normalized` text column) has no such column and simply misses.
pub(crate) fn disk_cache_lookup(
    conn: &Connection,
    query_normalized: &str,
    model_key: &str,
) -> Result<Vec<f32>, String> {
    let key = query_cache_key(model_key, query_normalized);
    let (blob, cached_at): (Vec<u8>, f64) = conn
        .query_row(
            "SELECT vector, cached_at FROM query_embed_cache WHERE query_hash = ?1 AND model_key = ?2",
            params![key, model_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| format!("disk cache miss: {}", e))?;
    // Check TTL (1 hour = 3600s)
    let age = now_ts() - cached_at;
    if age > 3600.0 {
        return Err("disk cache entry expired".to_string());
    }
    Ok(blob_to_f32_vec(&blob))
}

/// Store a query embedding in the persistent SQLite cache: the hash key, the model key, the
/// vector and the time; never the query text. Fails (and is ignored by the caller) on a store
/// whose table has the old shape; the next writer run recreates the table.
pub(crate) fn disk_cache_store(
    conn: &Connection,
    query_normalized: &str,
    model_key: &str,
    vector: &[f32],
) -> Result<(), String> {
    let blob = f32_blob(vector);
    let key = query_cache_key(model_key, query_normalized);
    conn.execute(
        r#"
INSERT INTO query_embed_cache(query_hash, model_key, vector, cached_at)
VALUES (?1, ?2, ?3, ?4)
ON CONFLICT(query_hash) DO UPDATE SET
    model_key = excluded.model_key,
    vector = excluded.vector,
    cached_at = excluded.cached_at
"#,
        params![key, model_key, blob, now_ts()],
    )
    .map_err(|e| format!("failed writing disk cache: {}", e))?;
    Ok(())
}

pub(crate) trait Embedder {
    fn model_key(&self) -> String;
    /// True when the backend is asked to return unit-length vectors. Part of the stored
    /// embedding identity: a vector produced under one setting is not reused under the other.
    fn normalizes_output(&self) -> bool {
        false
    }
    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let rows = self.embed_many(&[text.to_string()])?;
        rows.into_iter()
            .next()
            .ok_or_else(|| "No embedding returned.".to_string())
    }
    /// Embed a search query. Models with asymmetric input types (Cohere) send `search_query`
    /// here and `search_document` for indexed text; every other model embeds both the same way.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_one(text)
    }
}

pub(crate) fn build_embedder(cfg: &ConfigValues) -> Result<Box<dyn Embedder>, String> {
    match cfg.embed_backend.as_str() {
        "ollama" => Ok(Box::new(OllamaEmbedder::new(&cfg.embed_model))),
        "bedrock" => Ok(Box::new(BedrockEmbedder::new_with_config(
            &cfg.embed_model,
            Some(cfg),
        ))),
        // Offline feature hashing: deterministic, no network, not semantic. For tests and
        // smoke checks of the indexer, never for real retrieval quality.
        "hash" => Ok(Box::new(LocalHashEmbedder::new(
            cfg.local_embed_dim.max(0) as usize
        ))),
        other => Err(format!(
            "native index does not support backend '{}' yet; use one of {}",
            other,
            EMBED_BACKENDS.join(", ")
        )),
    }
}
