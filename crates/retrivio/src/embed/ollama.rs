//! Ollama backend: host discovery, autostart, model pull and the embedder.

use std::collections::HashSet;
use std::io::IsTerminal;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{env, thread};

use serde_json::Value;

use crate::config::ConfigValues;
use crate::embed::{
    embed_metric_request_end, embed_metric_request_start, embed_metric_retry, embed_metric_texts,
    hook_mode_active, Embedder,
};
use crate::util::{bool_env, command_exists, non_empty_env, prompt_yes_no};

pub(crate) static OLLAMA_AUTOSTART_ONCE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Known-good ollama embedding models: (model_name, description).
pub(crate) const KNOWN_OLLAMA_EMBEDDING_MODELS: &[(&str, &str)] = &[
    ("qwen3-embedding", "default, good quality, 896-dim"),
    ("nomic-embed-text", "768-dim, popular general-purpose"),
    ("mxbai-embed-large", "1024-dim, high quality"),
    ("all-minilm", "384-dim, fast and small"),
    ("snowflake-arctic-embed", "1024-dim, strong retrieval"),
];

/// Ollama preflight: check reachability and probe embedding with a test string.
pub(crate) fn run_ollama_preflight(cfg: &ConfigValues) -> Result<(), String> {
    let host = ollama_host();
    println!(
        "setup: ollama preflight (model='{}', host='{}')",
        cfg.embed_model, host
    );
    match ollama_is_reachable() {
        Ok(true) => println!("setup: ollama reachable: ok"),
        Ok(false) => {
            match maybe_autostart_ollama(&host) {
                Ok(true) => println!("setup: ollama auto-start: ok"),
                Ok(false) => {}
                Err(e) => {
                    return Err(format!(
                        "ollama is not reachable at '{}'; auto-start failed: {}",
                        host, e
                    ));
                }
            }
            if !matches!(ollama_is_reachable(), Ok(true)) {
                return Err(format!(
                    "ollama is not reachable at '{}'. Is it running? Try: ollama serve",
                    host
                ));
            }
            println!("setup: ollama reachable: ok");
        }
        Err(e) => {
            return Err(format!("ollama reachability check failed: {}", e));
        }
    }
    let embedder = OllamaEmbedder::new(&cfg.embed_model);
    let probe = "retrivio setup embedding probe";
    match embedder.embed_one(probe) {
        Ok(vec) => {
            println!(
                "setup: ollama model probe: ok (embedding_dim={})",
                vec.len()
            );
            return Ok(());
        }
        Err(e) => {
            let msg = e.to_string();
            if !msg.contains("404") && !msg.contains("not found") {
                return Err(format!("ollama model probe failed: {}", msg));
            }
            // Model not found — offer to pull it.
            eprintln!("model '{}' is not available locally.", cfg.embed_model);
            if !std::io::stdin().is_terminal() {
                return Err(format!(
                    "model '{}' not found. run `ollama pull {}` first.",
                    cfg.embed_model, cfg.embed_model
                ));
            }
            match prompt_yes_no(&format!("pull '{}' now?", cfg.embed_model), true) {
                Ok(true) => {
                    ollama_pull_model(&cfg.embed_model)?;
                }
                Ok(false) => {
                    return Err(format!(
                        "model '{}' not pulled. run `ollama pull {}` before using.",
                        cfg.embed_model, cfg.embed_model
                    ));
                }
                Err(e) => {
                    return Err(format!("prompt failed: {}", e));
                }
            }
        }
    }
    // Re-probe after pull
    let vec = embedder
        .embed_one(probe)
        .map_err(|e| format!("ollama model probe failed after pull: {}", e))?;
    println!(
        "setup: ollama model probe: ok (embedding_dim={})",
        vec.len()
    );
    Ok(())
}

pub(crate) fn ensure_ollama_ready_for_add_refresh(cfg: &ConfigValues) -> Result<(), String> {
    let host = ollama_host();
    match ollama_is_reachable() {
        Ok(true) => {}
        Ok(false) => {
            match maybe_autostart_ollama(&host) {
                Ok(true) | Ok(false) => {}
                Err(e) => {
                    return Err(format!(
                        "ollama is not reachable at '{}'; auto-start failed: {}",
                        host, e
                    ));
                }
            }
            if !matches!(ollama_is_reachable(), Ok(true)) {
                return Err(format!(
                    "ollama is not reachable at '{}'. Start it with `ollama serve`.",
                    host
                ));
            }
        }
        Err(e) => {
            return Err(format!("ollama reachability check failed: {}", e));
        }
    }

    let embedder = OllamaEmbedder::new(&cfg.embed_model);
    let probe = "retrivio add refresh embedding probe";
    match embedder.embed_one(probe) {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if !msg.contains("404") && !msg.contains("not found") {
                return Err(format!("ollama model probe failed: {}", msg));
            }
            if !std::io::stdin().is_terminal() {
                return Err(format!(
                    "model '{}' not found. run `ollama pull {}` first.",
                    cfg.embed_model, cfg.embed_model
                ));
            }
            eprintln!("model '{}' is not available locally.", cfg.embed_model);
            match prompt_yes_no(&format!("pull '{}' now?", cfg.embed_model), true) {
                Ok(true) => ollama_pull_model(&cfg.embed_model)?,
                Ok(false) => {
                    return Err(format!(
                        "model '{}' not pulled. run `ollama pull {}` before indexing.",
                        cfg.embed_model, cfg.embed_model
                    ));
                }
                Err(e) => return Err(format!("prompt failed: {}", e)),
            }
            embedder
                .embed_one(probe)
                .map(|_| ())
                .map_err(|e| format!("ollama model probe failed after pull: {}", e))
        }
    }
}

/// Resolve the Ollama API host (respects OLLAMA_HOST env var).
pub(crate) fn ollama_host() -> String {
    env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Check if Ollama is reachable by hitting GET /api/tags with a short timeout.
/// Returns Ok(true) if any HTTP response, Ok(false) on transport error.
pub(crate) fn ollama_is_reachable() -> Result<bool, String> {
    let host = ollama_host();
    let url = format!("{}/api/tags", host);
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(5))
        .build();
    match agent.get(&url).call() {
        Ok(_) => Ok(true),
        Err(ureq::Error::Status(_, _)) => Ok(true), // HTTP error still means reachable
        Err(ureq::Error::Transport(_)) => Ok(false),
    }
}

pub(crate) fn ollama_autostart_once_state() -> &'static Mutex<HashSet<String>> {
    OLLAMA_AUTOSTART_ONCE.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn ollama_autostart_timeout() -> Duration {
    non_empty_env("RETRIVIO_OLLAMA_AUTOSTART_TIMEOUT_SEC")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|secs| Duration::from_secs(secs.clamp(2, 90)))
        .unwrap_or_else(|| Duration::from_secs(12))
}

pub(crate) fn maybe_autostart_ollama(host: &str) -> Result<bool, String> {
    if !bool_env("RETRIVIO_OLLAMA_AUTOSTART", true) {
        return Ok(false);
    }
    // An editor hook never starts daemons; the caller reports the server as unreachable.
    if hook_mode_active() {
        return Ok(false);
    }
    if matches!(ollama_is_reachable(), Ok(true)) {
        return Ok(false);
    }

    let always_retry = bool_env("RETRIVIO_OLLAMA_AUTOSTART_ALWAYS", false);
    if !always_retry {
        let mut done = ollama_autostart_once_state()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if done.contains(host) {
            return Ok(false);
        }
        done.insert(host.to_string());
    }

    if !command_exists("ollama") {
        return Err("`ollama` executable not found for auto-start attempt".to_string());
    }

    eprintln!(
        "ollama: server not reachable at {}; attempting `ollama serve` in background...",
        host
    );
    let mut child = Command::new("ollama")
        .arg("serve")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn `ollama serve`: {}", e))?;

    let timeout = ollama_autostart_timeout();
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if matches!(ollama_is_reachable(), Ok(true)) {
            eprintln!("ollama: auto-start succeeded.");
            return Ok(true);
        }
        if let Ok(Some(status)) = child.try_wait() {
            let code = status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            return Err(format!("`ollama serve` exited early (status {})", code));
        }
        thread::sleep(Duration::from_millis(250));
    }

    Err(format!(
        "timed out waiting {}s for ollama API at {}",
        timeout.as_secs(),
        host
    ))
}

/// List locally-installed Ollama models by querying GET /api/tags.
/// Returns sorted model names on success.
pub(crate) fn ollama_list_local_models() -> Result<Vec<String>, String> {
    let host = ollama_host();
    let url = format!("{}/api/tags", host);
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(5))
        .build();
    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| format!("failed querying ollama models: {}", e))?;
    let raw = resp
        .into_string()
        .map_err(|e| format!("failed reading ollama response: {}", e))?;
    let json: Value =
        serde_json::from_str(&raw).map_err(|e| format!("failed parsing ollama response: {}", e))?;
    let mut names: Vec<String> = Vec::new();
    if let Some(models) = json.get("models").and_then(|v| v.as_array()) {
        for m in models {
            if let Some(name) = m.get("name").and_then(|v| v.as_str()) {
                // Normalize: strip ":latest" suffix so "qwen3-embedding:latest" -> "qwen3-embedding"
                let normalized = name.strip_suffix(":latest").unwrap_or(name).to_string();
                if !names.contains(&normalized) {
                    names.push(normalized);
                }
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Pull an Ollama model by running `ollama pull <model>` as a subprocess.
/// Inherits stdin/stdout/stderr so the user sees native progress display.
pub(crate) fn ollama_pull_model(model: &str) -> Result<(), String> {
    let status = Command::new("ollama")
        .arg("pull")
        .arg(model)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run `ollama pull {}`: {}", model, e))?;
    if !status.success() {
        return Err(format!(
            "`ollama pull {}` exited with status {}",
            model, status
        ));
    }
    Ok(())
}

pub(crate) struct OllamaEmbedder {
    model: String,
    host: String,
    keep_alive: Option<String>,
    timeout_sec: u64,
    max_input_chars: usize,
}

impl OllamaEmbedder {
    pub(crate) fn new(model: &str) -> Self {
        let model_name = if model.trim().is_empty() {
            "qwen3-embedding".to_string()
        } else {
            model.trim().to_string()
        };
        let host = ollama_host();
        let keep_alive = env::var("RETRIVIO_OLLAMA_KEEP_ALIVE")
            .ok()
            .unwrap_or_else(|| "24h".to_string())
            .trim()
            .to_string();
        let keep_alive = if keep_alive.is_empty() {
            None
        } else {
            Some(keep_alive)
        };
        let max_input_chars = Self::probe_max_input_chars(&host, &model_name);
        Self {
            model: model_name,
            host,
            keep_alive,
            timeout_sec: 60,
            max_input_chars,
        }
    }

    /// Query Ollama `/api/show` for the model's context length and derive a
    /// safe character limit. Falls back to a generous default on failure.
    fn probe_max_input_chars(host: &str, model: &str) -> usize {
        const DEFAULT_MAX_CHARS: usize = 8000;
        const CHARS_PER_TOKEN: usize = 2; // conservative for code-heavy content
        let url = format!("{}/api/show", host);
        let payload = serde_json::json!({ "name": model });
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(10))
            .build();
        let resp = match agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&payload.to_string())
        {
            Ok(r) => r,
            Err(_) => return DEFAULT_MAX_CHARS,
        };
        let raw = match resp.into_string() {
            Ok(s) => s,
            Err(_) => return DEFAULT_MAX_CHARS,
        };
        let json: Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => return DEFAULT_MAX_CHARS,
        };
        // num_ctx lives under model_info or model_params depending on Ollama version
        let num_ctx = json
            .pointer("/model_info/general.context_length")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                // fallback: parse from parameters string
                json.get("parameters")
                    .and_then(|v| v.as_str())
                    .and_then(|params| {
                        for line in params.lines() {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() == 2 && parts[0] == "num_ctx" {
                                return parts[1].parse::<u64>().ok();
                            }
                        }
                        None
                    })
            });
        match num_ctx {
            Some(ctx) => {
                let limit = (ctx as usize).saturating_mul(CHARS_PER_TOKEN);
                eprintln!(
                    "ollama model '{}': context_length={}, max_input_chars={}",
                    model, ctx, limit
                );
                limit.max(200) // never go below 200 chars
            }
            None => DEFAULT_MAX_CHARS,
        }
    }

    fn request_embed(
        &self,
        payload: &Value,
        allow_retry_without_keep_alive: bool,
        allow_retry_after_autostart: bool,
    ) -> Result<Value, String> {
        let url = format!("{}/api/embed", self.host);
        let body = payload.to_string();
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(self.timeout_sec))
            .build();
        embed_metric_request_start();
        let req_started = Instant::now();
        let response = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body);
        match response {
            Ok(resp) => {
                let raw = resp
                    .into_string()
                    .map_err(|e| format!("failed reading Ollama response: {}", e))?;
                let parsed = serde_json::from_str::<Value>(&raw)
                    .map_err(|e| format!("failed parsing Ollama response JSON: {}", e));
                embed_metric_request_end(parsed.is_ok(), req_started.elapsed());
                parsed
            }
            Err(ureq::Error::Status(code, resp)) => {
                embed_metric_request_end(false, req_started.elapsed());
                if code == 400
                    && allow_retry_without_keep_alive
                    && payload.get("keep_alive").is_some()
                {
                    embed_metric_retry();
                    let mut retry_payload = payload.clone();
                    if let Some(obj) = retry_payload.as_object_mut() {
                        obj.remove("keep_alive");
                    }
                    return self.request_embed(&retry_payload, false, allow_retry_after_autostart);
                }
                let detail = resp.into_string().unwrap_or_default();
                Err(format!(
                    "Ollama embedding request failed (HTTP {}): {}",
                    code, detail
                ))
            }
            Err(ureq::Error::Transport(e)) => {
                embed_metric_request_end(false, req_started.elapsed());
                if allow_retry_after_autostart {
                    match maybe_autostart_ollama(&self.host) {
                        Ok(true) => {
                            embed_metric_retry();
                            return self.request_embed(
                                payload,
                                allow_retry_without_keep_alive,
                                false,
                            );
                        }
                        Ok(false) => {}
                        Err(start_err) => {
                            return Err(format!(
                                "Ollama embedding request failed: {}. Auto-start failed: {}. Ensure Ollama is running and model '{}' is available.",
                                e, start_err, self.model
                            ));
                        }
                    }
                }
                Err(format!(
                    "Ollama embedding request failed: {}. Ensure Ollama is running and model '{}' is available.",
                    e, self.model
                ))
            }
        }
    }

    fn parse_vectors(data: &Value) -> Result<Vec<Vec<f32>>, String> {
        if let Some(embeddings) = data.get("embeddings").and_then(|v| v.as_array()) {
            let mut out = Vec::new();
            for item in embeddings {
                let Some(arr) = item.as_array() else {
                    continue;
                };
                let mut row = Vec::with_capacity(arr.len());
                for num in arr {
                    if let Some(f) = num.as_f64() {
                        row.push(f as f32);
                    }
                }
                if !row.is_empty() {
                    out.push(row);
                }
            }
            return Ok(out);
        }
        if let Some(single) = data.get("embedding").and_then(|v| v.as_array()) {
            let mut row = Vec::with_capacity(single.len());
            for num in single {
                if let Some(f) = num.as_f64() {
                    row.push(f as f32);
                }
            }
            if !row.is_empty() {
                return Ok(vec![row]);
            }
        }
        Err("Unexpected Ollama embed response format.".to_string())
    }
}

impl Embedder for OllamaEmbedder {
    fn model_key(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Start with the probed limit, then shrink adaptively on context-length errors.
        let mut limit = self.max_input_chars;
        for _shrink in 0..5 {
            let truncated: Vec<String> = texts
                .iter()
                .map(|t| {
                    if t.chars().count() <= limit {
                        t.clone()
                    } else {
                        t.chars().take(limit).collect()
                    }
                })
                .collect();
            let mut payload = serde_json::json!({
                "model": self.model,
                "input": truncated,
                "truncate": true,
            });
            if let Some(keep_alive) = &self.keep_alive {
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("keep_alive".to_string(), Value::String(keep_alive.clone()));
                }
            }
            match self.request_embed(&payload, true, true) {
                Ok(data) => {
                    let out = Self::parse_vectors(&data)?;
                    embed_metric_texts(out.len());
                    return Ok(out);
                }
                Err(e) if e.contains("context length") => {
                    // Tokenizer produced more tokens than expected — shrink and retry.
                    embed_metric_retry();
                    limit = (limit * 2) / 3; // reduce by ~33% each round
                    if limit < 64 {
                        return Err(e);
                    }
                    eprintln!(
                        "warning: input exceeded model context; retrying with max_input_chars={}",
                        limit
                    );
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err(
            "embedding failed: could not fit input within model context after multiple truncations"
                .to_string(),
        )
    }
}

// ---- AWS SigV4 native HTTP for Bedrock (replaces subprocess-per-embedding) ----
