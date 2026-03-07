use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::f64::consts::PI;
use std::ffi::OsString;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use rusqlite::{params, params_from_iter, Connection, OpenFlags, OptionalExtension};
use serde_json::Value;
use sha1::{Digest, Sha1};
use xxhash_rust::xxh64;

mod code_intel;
mod lance_store;

const APP_STATE_ACTIVE_MODEL_KEY: &str = "active_model_key";
const APP_STATE_REEMBED_REQUIRED: &str = "reembed_required";
const APP_STATE_REEMBED_REASON: &str = "reembed_reason";
const SHELL_HOOK_MARKER_START: &str = "# >>> retrivio shell >>>";
const SHELL_HOOK_MARKER_END: &str = "# <<< retrivio shell <<<";
const SHELL_WRAPPER_ENV: &str = "RETRIVIO_SHELL_WRAPPER";
static BEDROCK_REQ_SEQ: AtomicU64 = AtomicU64::new(1);
static BEDROCK_REFRESH_ONCE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
static OLLAMA_AUTOSTART_ONCE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
static LANCE_STORE: OnceLock<Mutex<Option<lance_store::LanceStore>>> = OnceLock::new();
static PROGRESS_IO_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static CLI_DATA_DIR_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();
static CLI_CONFIG_OVERRIDE: OnceLock<PathBuf> = OnceLock::new();
const KNOWN_TOP_LEVEL_COMMANDS: &[&str] = &[
    "doctor",
    "setup",
    "auth",
    "config",
    "autotune",
    "init",
    "install",
    "add",
    "del",
    "roots",
    "exclude",
    "include",
    "index",
    "refresh",
    "reembed",
    "watch",
    "search",
    "pick",
    "jump",
    "ui",
    "stop",
    "daemon",
    "bench",
    "api",
    "mcp",
    "self-test",
    "graph",
    "legacy",
];

#[derive(Default)]
struct EmbedRuntimeMetrics {
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
struct EmbedRuntimeSnapshot {
    requests_started: u64,
    requests_succeeded: u64,
    requests_failed: u64,
    request_retries: u64,
    throttles: u64,
    texts_embedded: u64,
    in_flight: i64,
    latency_sum_ms: u64,
    latency_samples: u64,
    latency_max_ms: u64,
}

fn embed_runtime_metrics() -> &'static EmbedRuntimeMetrics {
    static METRICS: OnceLock<EmbedRuntimeMetrics> = OnceLock::new();
    METRICS.get_or_init(EmbedRuntimeMetrics::default)
}

fn strip_terminal_control_sequences(raw: &str) -> String {
    let mut cleaned = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            cleaned.push(ch);
            continue;
        }
        match chars.peek().copied() {
            Some('[') => {
                chars.next();
                while let Some(next) = chars.next() {
                    if next.is_ascii_alphabetic() || next == '~' {
                        break;
                    }
                }
            }
            Some('O') => {
                chars.next();
                let _ = chars.next();
            }
            _ => {}
        }
    }
    cleaned
}

fn damerau_levenshtein_ascii(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b_chars.len() + 1]; a_chars.len() + 1];
    for i in 0..=a_chars.len() {
        dp[i][0] = i;
    }
    for j in 0..=b_chars.len() {
        dp[0][j] = j;
    }
    for i in 1..=a_chars.len() {
        for j in 1..=b_chars.len() {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            let mut best = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                best = best.min(dp[i - 2][j - 2] + 1);
            }
            dp[i][j] = best;
        }
    }
    dp[a_chars.len()][b_chars.len()]
}

fn likely_command_typo(input: &str) -> Option<&'static str> {
    let normalized = input.trim().to_ascii_lowercase();
    if normalized.len() < 4
        || !normalized
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
    {
        return None;
    }
    let mut best: Option<(&'static str, usize)> = None;
    for &candidate in KNOWN_TOP_LEVEL_COMMANDS {
        if candidate
            .chars()
            .next()
            .zip(normalized.chars().next())
            .map(|(a, b)| a != b)
            .unwrap_or(true)
        {
            continue;
        }
        let max_distance = if candidate.len() <= 4 {
            if normalized.len() != candidate.len() {
                continue;
            }
            1
        } else if candidate.len() >= 6 {
            2
        } else {
            1
        };
        let distance = damerau_levenshtein_ascii(&normalized, candidate);
        if distance > max_distance {
            continue;
        }
        match best {
            None => best = Some((candidate, distance)),
            Some((_, best_distance)) if distance < best_distance => {
                best = Some((candidate, distance));
            }
            _ => {}
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn atomic_update_max(dst: &AtomicU64, value: u64) {
    let mut cur = dst.load(Ordering::Relaxed);
    while value > cur {
        match dst.compare_exchange_weak(cur, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => cur = next,
        }
    }
}

fn reset_embed_runtime_metrics() {
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

fn embed_metric_request_start() {
    let m = embed_runtime_metrics();
    m.requests_started.fetch_add(1, Ordering::Relaxed);
    m.in_flight.fetch_add(1, Ordering::Relaxed);
}

fn embed_metric_request_end(success: bool, elapsed: Duration) {
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

fn embed_metric_retry() {
    embed_runtime_metrics()
        .request_retries
        .fetch_add(1, Ordering::Relaxed);
}

fn embed_metric_throttle() {
    embed_runtime_metrics()
        .throttles
        .fetch_add(1, Ordering::Relaxed);
}

fn embed_metric_texts(count: usize) {
    embed_runtime_metrics()
        .texts_embedded
        .fetch_add(count as u64, Ordering::Relaxed);
}

fn embed_runtime_snapshot() -> EmbedRuntimeSnapshot {
    let m = embed_runtime_metrics();
    EmbedRuntimeSnapshot {
        requests_started: m.requests_started.load(Ordering::Relaxed),
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

fn progress_heartbeat_interval() -> Duration {
    non_empty_env("RETRIVIO_PROGRESS_HEARTBEAT_MS")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|ms| Duration::from_millis(ms.clamp(500, 60_000)))
        .unwrap_or_else(|| Duration::from_secs(5))
}

fn progress_io_lock() -> &'static Mutex<()> {
    PROGRESS_IO_LOCK.get_or_init(|| Mutex::new(()))
}

fn progress_use_single_line() -> bool {
    std::io::stderr().is_terminal()
}

fn progress_clear_line() {
    if !progress_use_single_line() {
        return;
    }
    if let Ok(_g) = progress_io_lock().lock() {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

fn get_or_open_lance(cwd: &Path, dim: usize) -> Result<(), String> {
    let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(store) = guard.as_ref() {
        if lance_store::dim(store) == dim {
            return Ok(());
        }
        *guard = None;
    }
    let lance_path = data_dir(cwd).join("lance");
    if !lance_path.exists() {
        std::fs::create_dir_all(&lance_path)
            .map_err(|e| format!("failed creating lance directory: {}", e))?;
    }
    let store = lance_store::open(&lance_path, dim)?;
    *guard = Some(store);
    Ok(())
}

fn with_lance_store<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut lance_store::LanceStore) -> Result<R, String>,
{
    let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    match guard.as_mut() {
        Some(store) => f(store),
        None => Err("LanceDB not initialized; run index first".to_string()),
    }
}

fn set_cli_data_dir_override(raw: &str) -> Result<(), String> {
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

fn set_cli_config_override(raw: &str) -> Result<(), String> {
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

fn consume_global_path_overrides(args: Vec<OsString>) -> Result<Vec<OsString>, String> {
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

fn main() {
    configure_sigpipe();
    let raw_args: Vec<OsString> = env::args_os().skip(1).collect();
    let args = consume_global_path_overrides(raw_args).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(2);
    });
    if args.is_empty() {
        // Default to interactive query flow when invoked without a subcommand.
        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            maybe_prompt_shell_hook_setup_for_shorthand_query();
        }
        run_jump_cmd(&[]);
        return;
    }

    let first = args[0].to_string_lossy().to_string();
    match first.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" | "version" => {
            println!("retrivio {}", env!("CARGO_PKG_VERSION"));
        }
        "doctor" => run_doctor(&args[1..]),
        "setup" => run_setup_cmd(&args[1..]),
        "auth" => run_auth_cmd(&args[1..]),
        "config" => run_config_cmd(&args[1..]),
        "autotune" => run_autotune_cmd(&args[1..]),
        "init" => run_init(&args[1..]),
        "install" => run_install_cmd(&args[1..]),
        "add" => run_add(&args[1..]),
        "del" => run_del(&args[1..]),
        "roots" => run_roots(&args[1..]),
        "exclude" => run_exclude_cmd(&args[1..]),
        "include" => run_include_cmd(&args[1..]),
        "index" => run_index_cmd(&args[1..]),
        "refresh" => run_refresh_cmd(&args[1..]),
        "reembed" => run_reembed_cmd(&args[1..]),
        "watch" => run_watch_cmd(&args[1..]),
        "search" => run_search_cmd(&args[1..]),
        "pick" => run_pick_cmd(&args[1..]),
        "jump" => run_jump_cmd(&args[1..]),
        "ui" => run_ui_cmd(&args[1..]),
        "stop" => run_stop_cmd(&args[1..]),
        "daemon" => run_daemon_cmd(&args[1..]),
        "bench" => run_bench_cmd(&args[1..]),
        "api" => run_api_cmd(&args[1..]),
        "mcp" => run_mcp_cmd(&args[1..]),
        "self-test" => run_self_test_cmd(&args[1..]),
        "graph" => run_graph_cmd(&args[1..]),
        "legacy" => run_legacy_cmd(&args[1..]),
        _ => {
            if first.starts_with('-') {
                eprintln!("error: unknown option '{}'", first);
                eprintln!();
                print_help();
                process::exit(2);
            }
            if args.len() == 1 {
                if let Some(candidate) = likely_command_typo(&first) {
                    eprintln!("error: unknown command '{}'", first);
                    eprintln!("hint: did you mean `retrivio {}`?", candidate);
                    process::exit(2);
                }
            }
            // Shorthand query mode:
            // - interactive terminals: open jump/picker flow
            // - non-interactive contexts: print search results
            if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                maybe_prompt_shell_hook_setup_for_shorthand_query();
                run_jump_cmd(&args);
            } else {
                run_search_cmd(&args);
            }
        }
    }
}

#[cfg(unix)]
fn configure_sigpipe() {
    // Allow shell pipelines like `retrivio doctor | head -n 5` to exit cleanly
    // when the downstream reader closes early.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn configure_sigpipe() {}

fn detect_active_shell_for_init() -> String {
    let shell = non_empty_env("SHELL")
        .and_then(|raw| {
            Path::new(raw.trim())
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "bash".to_string());
    if shell.contains("zsh") {
        "zsh".to_string()
    } else if shell.contains("fish") {
        "fish".to_string()
    } else {
        "bash".to_string()
    }
}

fn shell_rc_path(shell: &str) -> Option<PathBuf> {
    let home = env::var("HOME").ok().map(PathBuf::from)?;
    match shell {
        "zsh" => Some(home.join(".zshrc")),
        "bash" => Some(home.join(".bashrc")),
        "fish" => Some(home.join(".config").join("fish").join("config.fish")),
        _ => None,
    }
}

fn shell_hook_block_present(rc_path: &Path) -> bool {
    fs::read_to_string(rc_path)
        .map(|raw| raw.contains(SHELL_HOOK_MARKER_START) && raw.contains(SHELL_HOOK_MARKER_END))
        .unwrap_or(false)
}

fn default_shell_hook_present_in_rc_files() -> bool {
    let Some(home) = env::var("HOME").ok().map(PathBuf::from) else {
        return false;
    };
    let bash = shell_hook_block_present(&home.join(".bashrc"));
    let zsh = shell_hook_block_present(&home.join(".zshrc"));
    bash || zsh
}

fn print_shell_activation_hint() {
    let shell = detect_active_shell_for_init();
    eprintln!(
        "hint: run `eval \"$(retrivio init {})\"` in this shell to activate now.",
        shell
    );
    if let Some(rc) = shell_rc_path(&shell) {
        eprintln!("hint: or reload your rc file: source {}", rc.display());
    } else {
        eprintln!("hint: or open a new shell after updating your rc file.");
    }
}

fn maybe_prompt_shell_hook_setup_for_shorthand_query() {
    if bool_env(SHELL_WRAPPER_ENV, false) {
        return;
    }
    if bool_env("RETRIVIO_NO_SHELL_HOOK_PROMPT", false) {
        return;
    }
    eprintln!("note: retrivio shell integration is not active in this shell.");
    eprintln!(
        "note: selections can print a path, but only the shell wrapper can `cd` your current shell."
    );

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if default_shell_hook_present_in_rc_files() {
        eprintln!("note: shell hooks already exist in your rc files but are not loaded yet.");
        print_shell_activation_hint();
        return;
    }

    match prompt_yes_no(
        "install shell integration into ~/.bashrc and ~/.zshrc now?",
        false,
    ) {
        Ok(true) => {
            match install_default_shell_hooks(&cwd) {
                Ok(messages) => {
                    for line in messages {
                        eprintln!("shell_hook: {}", line);
                    }
                }
                Err(err) => eprintln!("warning: shell hook setup failed: {}", err),
            }
            print_shell_activation_hint();
        }
        Ok(false) => {
            print_shell_activation_hint();
        }
        Err(err) => {
            eprintln!("warning: prompt failed: {}", err);
            print_shell_activation_hint();
        }
    }
}

/// Known-good ollama embedding models: (model_name, description).
const KNOWN_OLLAMA_EMBEDDING_MODELS: &[(&str, &str)] = &[
    ("qwen3-embedding", "default, good quality, 896-dim"),
    ("nomic-embed-text", "768-dim, popular general-purpose"),
    ("mxbai-embed-large", "1024-dim, high quality"),
    ("all-minilm", "384-dim, fast and small"),
    ("snowflake-arctic-embed", "1024-dim, strong retrieval"),
];

/// Known-good bedrock embedding models: (model_id, description).
const KNOWN_BEDROCK_EMBEDDING_MODELS: &[(&str, &str)] = &[
    (
        "amazon.titan-embed-text-v2:0",
        "default, 1024-dim, AWS native",
    ),
    ("cohere.embed-english-v3", "1024-dim, English-optimized"),
    ("cohere.embed-multilingual-v3", "1024-dim, multilingual"),
];

fn default_embed_model_for_backend(backend: &str) -> &'static str {
    match backend {
        "bedrock" => "amazon.titan-embed-text-v2:0",
        _ => "qwen3-embedding",
    }
}

fn is_probably_bedrock_model_id(model: &str) -> bool {
    let t = model.trim();
    if t.is_empty() {
        return false;
    }
    t.starts_with("arn:aws:bedrock:")
        || t.starts_with("amazon.")
        || t.starts_with("cohere.")
        || t.starts_with("anthropic.")
        || t.starts_with("meta.")
        || t.starts_with("mistral.")
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|v| {
        let t = v.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    })
}

fn bool_env(name: &str, default: bool) -> bool {
    non_empty_env(name)
        .map(|v| {
            let n = v.to_ascii_lowercase();
            matches!(n.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(default)
}

fn print_help() {
    println!("retrivio (rust)");
    println!();
    println!("Global options (must come before subcommand/query):");
    println!("  --data-dir <path>            # override state dir for this run");
    println!("  --config <path>              # override config file for this run");
    println!();
    println!("Shorthand:");
    println!("  retrivio                     # interactive query/picker");
    println!(
        "  retrivio <query...>          # interactive jump/picker (TTY), search output (non-TTY)"
    );
    println!();
    println!("Native commands:");
    println!("  retrivio doctor [--fix]");
    println!("  retrivio setup");
    println!("  retrivio auth [select|status]");
    println!("  retrivio config [edit|show|set <key> <value>|autotune] [options]");
    println!("  retrivio autotune [--dry-run] [--deep] [--max-events <n>] [--limit <n>]");
    println!("  retrivio version");
    println!("  retrivio init [bash|zsh|fish] [--root <path>] [--embed-backend <ollama|bedrock>] [--embed-model <id>]");
    println!("  retrivio install [--no-system-install] [--no-download] [--no-shell-hook] [--legacy] [--venv <path>] [--bench]");
    println!("  retrivio add <path> [path ...] [--exclude <pattern>] [--refresh|--no-refresh]");
    println!("  retrivio del <path> [path ...] [--refresh|--no-refresh]");
    println!("  retrivio roots");
    println!("  retrivio exclude <root> <pattern> [pattern ...]");
    println!("  retrivio include <root> <pattern> [pattern ...]");
    println!("  retrivio index");
    println!("  retrivio refresh [path ...]");
    println!(
        "  retrivio reembed                    # rebuild vectors/graph after embed model change"
    );
    println!("  retrivio watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]");
    println!("  retrivio search [--view projects|files] [--limit <n>] <query...>");
    println!(
        "  retrivio pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
    );
    println!("  retrivio jump [--files|--dirs] [--limit <n>] [query...]");
    println!("  retrivio ui [--host <addr>] [--port <n>]");
    println!("  retrivio stop [--wait <seconds>]");
    println!(
        "  retrivio daemon [start|stop|restart|status|logs [n]] [--host <addr>] [--port <n>] [--timeout <seconds>]"
    );
    println!("  retrivio bench [plan|doctor|export|run] [options]");
    println!("  retrivio graph [doctor|status|start|stop|provision|view|open|neighbors|lineage]  # advanced");
    println!("  retrivio api [--host <addr>] [--port <n>]");
    println!("  retrivio mcp [serve|doctor|register|unregister]");
    println!("  retrivio self-test");
    println!();
    println!("Compatibility:");
    println!("  retrivio legacy <args...>   # explicit Python bridge");
}

fn run_ui_cmd(args: &[OsString]) {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8780;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--host" => {
                i += 1;
                host = arg_value(args, i, "--host");
            }
            "--port" => {
                i += 1;
                let raw = arg_value(args, i, "--port");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            x if x.starts_with("--host=") => {
                host = x.trim_start_matches("--host=").to_string();
            }
            x if x.starts_with("--port=") => {
                let raw = x.trim_start_matches("--port=");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    run_graph_open_cmd(&host, port).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

fn run_stop_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio stop [--wait <seconds>]");
        println!("no-op (LanceDB is embedded; no external process to stop).");
        return;
    }
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--wait" => {
                i += 1;
                let value = arg_value(args, i, "--wait");
                if value.parse::<u64>().is_err() {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                }
            }
            x if x.starts_with("--wait=") => {
                let value = x.trim_start_matches("--wait=");
                if value.parse::<u64>().is_err() {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                }
            }
            other => {
                eprintln!("error: unknown stop option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }
    println!("no longer needed (LanceDB is embedded — no external server process)");
}

fn run_doctor(args: &[OsString]) {
    let mut fix = false;
    for arg in args {
        let s = arg.to_string_lossy();
        match s.as_ref() {
            "-h" | "--help" => {
                println!("usage: retrivio doctor [--fix]");
                println!("  --fix   run active backend preflight checks and remediation probes");
                return;
            }
            "--fix" => {
                fix = true;
            }
            other => {
                eprintln!("error: unknown doctor option '{}'", other);
                process::exit(2);
            }
        }
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);

    let cfg_map = load_config_values(&cfg_path);
    let cfg = ConfigValues::from_map(cfg_map);
    let root = cfg.root.clone();
    let embed_backend = cfg.embed_backend.clone();
    let embed_model = cfg.embed_model.clone();
    let _retrieval_backend = cfg.retrieval_backend.clone();
    let root_exists = root.is_dir();
    let cfg_exists = cfg_path.exists();
    let db_exists = db_path.exists();
    let fzf_installed = command_exists("fzf");
    let fswatch_installed = command_exists("fswatch");
    let tracked_roots = tracked_roots_count(&db_path).unwrap_or(0);
    let db_ready = database_ready(&db_path);
    let reembed_state = if db_exists && db_ready {
        match open_db_rw(&db_path).and_then(|conn| reembed_requirement_reason(&conn, &cfg)) {
            Ok(Some(reason)) => format!("required ({})", reason),
            Ok(None) => "ready".to_string(),
            Err(err) => format!("unknown ({})", err),
        }
    } else {
        "unknown (database not ready)".to_string()
    };

    println!("config: {}", cfg_path.display());
    println!("db: {}", db_path.display());
    println!(
        "config root exists: {} ({})",
        yes_no(root_exists),
        root.to_string_lossy()
    );
    println!("fzf installed: {}", yes_no(fzf_installed));
    println!(
        "fswatch installed: {}{}",
        yes_no(fswatch_installed),
        if fswatch_installed {
            ""
        } else {
            " (watch uses polling fallback)"
        }
    );
    println!("embed backend configured: {}", embed_backend);
    println!("embed model configured: {}", embed_model);
    if embed_backend == "bedrock" {
        let region = bedrock_region_for_cfg(Some(&cfg));
        let profile =
            bedrock_profile_for_cfg(Some(&cfg)).unwrap_or_else(|| "<default>".to_string());
        let aws_cli = bedrock_aws_cli_path();
        let concurrency = bedrock_concurrency_for_cfg(Some(&cfg));
        let max_retries = bedrock_max_retries_for_cfg(Some(&cfg));
        let retry_base_ms = bedrock_retry_base_ms_for_cfg(Some(&cfg));
        let refresh_cmd = bedrock_refresh_cmd_for_cfg(Some(&cfg))
            .map(|_| "yes".to_string())
            .unwrap_or_else(|| "no".to_string());
        println!(
            "bedrock profile configured: {}",
            if cfg.aws_profile.trim().is_empty() {
                "<none>"
            } else {
                cfg.aws_profile.trim()
            }
        );
        println!(
            "bedrock region configured: {}",
            if cfg.aws_region.trim().is_empty() {
                "<none>"
            } else {
                cfg.aws_region.trim()
            }
        );
        println!("bedrock region resolved: {}", region);
        println!("bedrock profile resolved: {}", profile);
        println!("bedrock aws cli: {}", aws_cli);
        println!("bedrock refresh cmd configured: {}", refresh_cmd);
        println!("bedrock invoke concurrency: {}", concurrency);
        println!("bedrock max retries: {}", max_retries);
        println!("bedrock retry base ms: {}", retry_base_ms);
    }
    println!("retrieval backend: lancedb (embedded)");
    println!("embed backend active: {}", embed_backend);
    println!("tracked roots: {}", tracked_roots);
    println!("database ready: {}", yes_no(db_ready && db_exists));
    println!("embedding migration: {}", reembed_state);

    // Check for projects with incomplete chunk vectors
    if db_exists && db_ready {
        if let Ok(conn) = open_db_read_only(&db_path) {
            let model_key = model_key_for_cfg(&cfg);
            match count_incomplete_vector_projects(&conn, &model_key) {
                Ok(gaps) if !gaps.is_empty() => {
                    eprintln!(
                        "warning: {} project(s) have incomplete embeddings (chunks without vectors)",
                        gaps.len()
                    );
                    for (name, have, total) in &gaps {
                        eprintln!("  {} ({}/{} chunks have embeddings)", name, have, total);
                    }
                    eprintln!("  run `retrivio index` to repair");
                }
                _ => {}
            }
        }
    }

    if !cfg_exists {
        eprintln!("warning: config is missing; initialize with: retrivio init");
    }

    if fix {
        match run_doctor_fix(&cfg) {
            Ok(_) => {
                println!("doctor --fix: ok");
            }
            Err(err) => {
                eprintln!("doctor --fix failed: {}", err);
                process::exit(1);
            }
        }
    } else if embed_backend == "bedrock" {
        println!(
            "doctor hint: run `retrivio doctor --fix` to preflight AWS identity/model access."
        );
    }
}

fn command_available(cmd: &str) -> bool {
    if cmd.contains('/') {
        return is_executable_file(Path::new(cmd));
    }
    command_exists(cmd)
}

fn aws_cli_json(
    aws_cli: &str,
    region: &str,
    profile: Option<&str>,
    args: &[&str],
) -> Result<Value, String> {
    let mut cmd = Command::new(aws_cli);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.arg("--region").arg(region).arg("--output").arg("json");
    if let Some(p) = profile {
        cmd.arg("--profile").arg(p);
    }
    let output = cmd
        .output()
        .map_err(|e| format!("failed executing AWS CLI '{}': {}", aws_cli, e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!(
            "AWS CLI command failed: {} {}",
            args.join(" "),
            detail
        ));
    }
    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    serde_json::from_str::<Value>(&raw)
        .map_err(|e| format!("failed parsing AWS CLI JSON response: {}", e))
}

fn run_doctor_fix(cfg: &ConfigValues) -> Result<(), String> {
    if cfg.embed_backend != "bedrock" {
        println!(
            "doctor --fix: no embedding preflight for embed_backend='{}' (bedrock only)",
            cfg.embed_backend
        );
        return Ok(());
    }

    let aws_cli = bedrock_aws_cli_path();
    let region = bedrock_region_for_cfg(Some(cfg));
    let profile = bedrock_profile_for_cfg(Some(cfg));
    println!(
        "doctor --fix: bedrock preflight (model='{}', region='{}', profile='{}')",
        cfg.embed_model,
        region,
        profile.clone().unwrap_or_else(|| "<default>".to_string())
    );

    if !command_available(&aws_cli) {
        return Err(format!(
            "AWS CLI '{}' is not available/executable (set RETRIVIO_AWS_CLI or install aws cli)",
            aws_cli
        ));
    }
    println!("doctor --fix: aws cli executable: ok ({})", aws_cli);

    if bedrock_refresh_cmd_for_cfg(Some(cfg)).is_some() {
        println!("doctor --fix: running RETRIVIO_AWS_REFRESH_CMD...");
        refresh_aws_credentials_if_configured(Some(cfg))?;
        println!("doctor --fix: credential refresh: ok");
    } else {
        println!(
            "doctor --fix: credential refresh hook not set (configure via `retrivio setup` / `retrivio config set aws_refresh_cmd ...`)"
        );
    }

    let identity = aws_cli_json(
        &aws_cli,
        &region,
        profile.as_deref(),
        &["sts", "get-caller-identity"],
    )?;
    let account = identity
        .get("Account")
        .and_then(|v| v.as_str())
        .unwrap_or("<unknown>");
    let arn = identity
        .get("Arn")
        .and_then(|v| v.as_str())
        .unwrap_or("<unknown>");
    println!(
        "doctor --fix: aws identity: account={} arn={}",
        account, arn
    );

    let embedder = BedrockEmbedder::new_with_config(&cfg.embed_model, Some(cfg));
    let probe = "retrivio doctor embedding probe";
    let vec = embedder
        .embed_one(probe)
        .map_err(|e| format!("bedrock model probe failed: {}", e))?;
    println!(
        "doctor --fix: bedrock model probe: ok (embedding_dim={})",
        vec.len()
    );
    Ok(())
}

/// Ollama preflight: check reachability and probe embedding with a test string.
fn run_ollama_preflight(cfg: &ConfigValues) -> Result<(), String> {
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

fn ensure_ollama_ready_for_add_refresh(cfg: &ConfigValues) -> Result<(), String> {
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

#[derive(Clone, Debug)]
struct AwsProfileChoice {
    name: String,
    region: Option<String>,
    source: String,
}

#[derive(Clone, Debug)]
struct IsengardAccountChoice {
    account_ref: String,
    label: String,
    favorite: bool,
}

fn run_setup_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio setup");
        println!("guided first-time setup for backend/model/auth/performance.");
        return;
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!("error: `retrivio setup` requires an interactive terminal");
        process::exit(2);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    eprintln!("setup: using config -> {}", cfg_path.display());
    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let old_cfg = cfg.clone();

    // 1. Select backend
    let backend_options = vec![
        "ollama (local model runtime)".to_string(),
        "bedrock (aws credentials + model API)".to_string(),
    ];
    let default_backend_idx = if cfg.embed_backend == "bedrock" { 1 } else { 0 };
    let selected_backend = select_option(
        "choose embedding backend",
        &backend_options,
        Some(default_backend_idx),
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    })
    .unwrap_or(default_backend_idx);
    cfg.embed_backend = if selected_backend == 1 {
        "bedrock".to_string()
    } else {
        "ollama".to_string()
    };

    // 2. Select model (interactive)
    if cfg.embed_backend == "ollama" {
        configure_ollama_model_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        configure_bedrock_model_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    }

    // 3. Backend-specific auth/performance
    if cfg.embed_backend == "bedrock" {
        configure_bedrock_auth_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        configure_bedrock_performance_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        cfg.aws_profile.clear();
        cfg.aws_region.clear();
        cfg.aws_refresh_cmd.clear();
    }

    // 4. Write config
    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed writing config: {}", e);
        process::exit(1);
    });
    println!("setup: config updated -> {}", cfg_path.display());
    println!("embed_backend: {}", cfg.embed_backend);
    println!("embed_model: {}", cfg.embed_model);
    if cfg.embed_backend == "bedrock" {
        println!(
            "aws_profile: {}",
            if cfg.aws_profile.trim().is_empty() {
                "<default chain>"
            } else {
                cfg.aws_profile.trim()
            }
        );
        println!("aws_region: {}", bedrock_region_for_cfg(Some(&cfg)));
        println!(
            "aws_refresh_cmd: {}",
            if cfg.aws_refresh_cmd.trim().is_empty() {
                "<none>"
            } else {
                cfg.aws_refresh_cmd.trim()
            }
        );
        println!("bedrock_concurrency: {}", cfg.bedrock_concurrency);
        println!("bedrock_max_retries: {}", cfg.bedrock_max_retries);
        println!("bedrock_retry_base_ms: {}", cfg.bedrock_retry_base_ms);
    }

    // 5. Preflight — both backends
    if cfg.embed_backend == "bedrock" {
        match run_doctor_fix(&cfg) {
            Ok(_) => println!("setup: bedrock preflight ok"),
            Err(e) => eprintln!("warning: setup preflight failed: {}", e),
        }
    } else {
        match run_ollama_preflight(&cfg) {
            Ok(_) => println!("setup: ollama preflight ok"),
            Err(e) => {
                eprintln!("warning: setup preflight failed: {}", e);
                if !command_exists("ollama") {
                    eprintln!(
                        "note: `ollama` is not installed on this system. install Ollama or rerun `retrivio setup` and choose `bedrock`."
                    );
                } else {
                    eprintln!(
                        "note: start Ollama with `ollama serve`, or rerun `retrivio setup` and choose `bedrock`."
                    );
                }
            }
        }
    }

    // 6. Reembed detection (config change + quick compatibility check)
    match refresh_reembed_requirement_for_config_change(&cwd, &old_cfg, &cfg) {
        Ok(Some(reason)) => {
            eprintln!();
            eprintln!("warning: {}", reason);
            match prompt_yes_no("run `retrivio reembed` now?", false) {
                Ok(true) => {
                    run_reembed_cmd(&[]);
                }
                Ok(false) => {
                    println!("run `retrivio reembed` before searching.");
                }
                Err(e) => {
                    eprintln!("warning: prompt failed: {}", e);
                    println!("run `retrivio reembed` before searching.");
                }
            }
        }
        Ok(None) => { /* embeddings compatible with current config */ }
        Err(e) => {
            // DB may not exist yet on first setup — that's fine
            if !e.contains("no such table") && !e.contains("unable to open") {
                eprintln!("warning: reembed check failed: {}", e);
            }
        }
    }
}

fn run_auth_cmd(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio auth [select|status]");
        println!("  select  interactive Bedrock auth/profile/region selection");
        println!("  status  show resolved Bedrock auth/profile/region");
        return;
    }
    let sub = args[0].to_string_lossy().to_ascii_lowercase();
    match sub.as_str() {
        "status" => {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            println!("embed_backend: {}", cfg.embed_backend);
            println!(
                "aws_profile (configured): {}",
                if cfg.aws_profile.trim().is_empty() {
                    "<none>"
                } else {
                    cfg.aws_profile.trim()
                }
            );
            println!(
                "aws_profile (resolved): {}",
                bedrock_profile_for_cfg(Some(&cfg))
                    .unwrap_or_else(|| "<default chain>".to_string())
            );
            println!(
                "aws_region (configured): {}",
                if cfg.aws_region.trim().is_empty() {
                    "<none>"
                } else {
                    cfg.aws_region.trim()
                }
            );
            println!(
                "aws_region (resolved): {}",
                bedrock_region_for_cfg(Some(&cfg))
            );
            println!(
                "aws_refresh_cmd: {}",
                bedrock_refresh_cmd_for_cfg(Some(&cfg)).unwrap_or_else(|| "<none>".to_string())
            );
            println!(
                "bedrock_concurrency: {}",
                bedrock_concurrency_for_cfg(Some(&cfg))
            );
            println!(
                "bedrock_max_retries: {}",
                bedrock_max_retries_for_cfg(Some(&cfg))
            );
            println!(
                "bedrock_retry_base_ms: {}",
                bedrock_retry_base_ms_for_cfg(Some(&cfg))
            );
        }
        "select" => {
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                eprintln!("error: `retrivio auth select` requires an interactive terminal");
                process::exit(2);
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg_path = config_path(&cwd);
            let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
            cfg.embed_backend = "bedrock".to_string();
            configure_bedrock_auth_interactive(&mut cfg).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
                eprintln!("error: failed writing config: {}", e);
                process::exit(1);
            });
            println!("auth: config updated -> {}", cfg_path.display());
            println!(
                "aws_profile: {}",
                if cfg.aws_profile.trim().is_empty() {
                    "<default chain>"
                } else {
                    cfg.aws_profile.trim()
                }
            );
            println!("aws_region: {}", bedrock_region_for_cfg(Some(&cfg)));
            println!("bedrock_concurrency: {}", cfg.bedrock_concurrency);
            println!("bedrock_max_retries: {}", cfg.bedrock_max_retries);
            println!("bedrock_retry_base_ms: {}", cfg.bedrock_retry_base_ms);
        }
        other => {
            eprintln!("error: unknown auth subcommand '{}'", other);
            process::exit(2);
        }
    }
}

fn configure_bedrock_auth_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let aws_profiles = list_aws_profiles();
    let isengard_cli = find_isengardcli_binary();
    let isengard_accounts = if let Some(cli) = &isengard_cli {
        list_isengard_accounts(cli).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut use_isengard = false;
    if !isengard_accounts.is_empty() {
        let source_options = vec![
            "aws profile (from local aws config/credentials)".to_string(),
            "isengard account (auto-refresh command)".to_string(),
        ];
        let source_idx = select_option(
            "choose bedrock auth source",
            &source_options,
            Some(if cfg.aws_refresh_cmd.trim().is_empty() {
                0
            } else {
                1
            }),
        )?
        .unwrap_or(0);
        use_isengard = source_idx == 1;
    }

    if use_isengard {
        let labels: Vec<String> = isengard_accounts.iter().map(|a| a.label.clone()).collect();
        if let Some(idx) = select_option("choose isengard account for refresh", &labels, Some(0))? {
            let choice = &isengard_accounts[idx];
            if let Some(cli) = &isengard_cli {
                // Select role so the refresh command is non-interactive
                let role_options = vec![
                    "Admin".to_string(),
                    "ReadOnly".to_string(),
                    "PowerUser".to_string(),
                ];
                let existing_role = extract_role_from_refresh_cmd(&cfg.aws_refresh_cmd);
                let default_role_idx = existing_role
                    .as_ref()
                    .and_then(|r| role_options.iter().position(|o| o.eq_ignore_ascii_case(r)))
                    .unwrap_or(0);
                let role = if let Some(role_idx) = select_option(
                    "choose isengard role",
                    &role_options,
                    Some(default_role_idx),
                )? {
                    role_options[role_idx].clone()
                } else {
                    "Admin".to_string()
                };
                cfg.aws_refresh_cmd = format!(
                    "{} add-profile {} --role {}",
                    shell_escape(cli),
                    shell_escape(&choice.account_ref),
                    role
                );
            }
        }
    } else {
        cfg.aws_refresh_cmd.clear();
    }

    if !aws_profiles.is_empty() {
        let mut profile_options = vec!["<default chain>".to_string()];
        for profile in &aws_profiles {
            let region = profile.region.clone().unwrap_or_else(|| "-".to_string());
            profile_options.push(format!(
                "{}  (region={} source={})",
                profile.name, region, profile.source
            ));
        }
        let default_idx = if cfg.aws_profile.trim().is_empty() {
            0
        } else {
            aws_profiles
                .iter()
                .position(|p| p.name == cfg.aws_profile.trim())
                .map(|i| i + 1)
                .unwrap_or(0)
        };
        let selected = select_option("choose aws profile", &profile_options, Some(default_idx))?
            .unwrap_or(default_idx);
        if selected == 0 {
            cfg.aws_profile.clear();
        } else if let Some(p) = aws_profiles.get(selected - 1) {
            cfg.aws_profile = p.name.clone();
            if cfg.aws_region.trim().is_empty() {
                if let Some(region) = &p.region {
                    cfg.aws_region = region.clone();
                }
            }
        }
    } else {
        let manual = prompt_line("aws profile name (empty for default chain): ")?;
        cfg.aws_profile = manual.trim().to_string();
    }

    let mut region_candidates = vec![
        "us-east-1".to_string(),
        "us-east-2".to_string(),
        "us-west-1".to_string(),
        "us-west-2".to_string(),
        "eu-west-1".to_string(),
        "eu-central-1".to_string(),
        "ap-southeast-1".to_string(),
        "ap-southeast-2".to_string(),
        "ap-northeast-1".to_string(),
    ];
    if let Some(region) = aws_region_for_profile_name(if cfg.aws_profile.trim().is_empty() {
        "default"
    } else {
        cfg.aws_profile.trim()
    }) {
        region_candidates.push(region);
    }
    if !cfg.aws_region.trim().is_empty() {
        region_candidates.push(cfg.aws_region.trim().to_string());
    }
    let resolved_region = bedrock_region_for_cfg(Some(cfg));
    region_candidates.push(resolved_region.clone());
    region_candidates.sort();
    region_candidates.dedup();
    let default_region_idx = region_candidates
        .iter()
        .position(|r| *r == resolved_region)
        .unwrap_or(0);
    if let Some(idx) = select_option(
        "choose aws region for bedrock",
        &region_candidates,
        Some(default_region_idx),
    )? {
        cfg.aws_region = region_candidates[idx].clone();
    }
    Ok(())
}

fn configure_bedrock_performance_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let current_concurrency = cfg.bedrock_concurrency.clamp(1, 32);
    let current_retries = cfg.bedrock_max_retries.clamp(0, 12);
    let current_base_ms = cfg.bedrock_retry_base_ms.clamp(50, 10_000);

    let profiles = vec![
        "balanced (recommended): concurrency=4 retries=3 base_ms=250".to_string(),
        "fast: concurrency=8 retries=2 base_ms=150".to_string(),
        "conservative: concurrency=2 retries=4 base_ms=350".to_string(),
        "aggressive: concurrency=12 retries=1 base_ms=120".to_string(),
        format!(
            "keep current: concurrency={} retries={} base_ms={}",
            current_concurrency, current_retries, current_base_ms
        ),
        "custom...".to_string(),
    ];

    let selected =
        select_option("choose bedrock performance profile", &profiles, Some(4))?.unwrap_or(4);
    match selected {
        0 => {
            cfg.bedrock_concurrency = 4;
            cfg.bedrock_max_retries = 3;
            cfg.bedrock_retry_base_ms = 250;
        }
        1 => {
            cfg.bedrock_concurrency = 8;
            cfg.bedrock_max_retries = 2;
            cfg.bedrock_retry_base_ms = 150;
        }
        2 => {
            cfg.bedrock_concurrency = 2;
            cfg.bedrock_max_retries = 4;
            cfg.bedrock_retry_base_ms = 350;
        }
        3 => {
            cfg.bedrock_concurrency = 12;
            cfg.bedrock_max_retries = 1;
            cfg.bedrock_retry_base_ms = 120;
        }
        4 => {
            cfg.bedrock_concurrency = current_concurrency;
            cfg.bedrock_max_retries = current_retries;
            cfg.bedrock_retry_base_ms = current_base_ms;
        }
        _ => {
            let c = prompt_line(&format!(
                "bedrock_concurrency (1..32) [{}]: ",
                current_concurrency
            ))?;
            let r = prompt_line(&format!(
                "bedrock_max_retries (0..12) [{}]: ",
                current_retries
            ))?;
            let b = prompt_line(&format!(
                "bedrock_retry_base_ms (50..10000) [{}]: ",
                current_base_ms
            ))?;

            let parsed_c = if c.trim().is_empty() {
                current_concurrency
            } else {
                c.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_concurrency must be an integer".to_string())?
            };
            let parsed_r = if r.trim().is_empty() {
                current_retries
            } else {
                r.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_max_retries must be an integer".to_string())?
            };
            let parsed_b = if b.trim().is_empty() {
                current_base_ms
            } else {
                b.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_retry_base_ms must be an integer".to_string())?
            };
            cfg.bedrock_concurrency = parsed_c.clamp(1, 32);
            cfg.bedrock_max_retries = parsed_r.clamp(0, 12);
            cfg.bedrock_retry_base_ms = parsed_b.clamp(50, 10_000);
        }
    }
    Ok(())
}

/// Interactive ollama model selection with availability checks.
fn configure_ollama_model_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let reachable = ollama_is_reachable().unwrap_or(false);
    let local_models: Vec<String> = if reachable {
        ollama_list_local_models().unwrap_or_default()
    } else {
        eprintln!(
            "warning: ollama not reachable at '{}'; install status unknown",
            ollama_host()
        );
        Vec::new()
    };

    // Build the selection list from known-good models
    let mut options: Vec<String> = Vec::new();
    let mut option_model_names: Vec<String> = Vec::new();

    for &(name, desc) in KNOWN_OLLAMA_EMBEDDING_MODELS {
        let installed = local_models
            .iter()
            .any(|m| m == name || m.starts_with(&format!("{}:", name)));
        let status = if !reachable {
            String::new()
        } else if installed {
            ", installed".to_string()
        } else {
            ", not installed".to_string()
        };
        options.push(format!("{} ({}{})", name, desc, status));
        option_model_names.push(name.to_string());
    }

    // Add any locally-installed models not already in the known list
    if reachable {
        for local in &local_models {
            let already = KNOWN_OLLAMA_EMBEDDING_MODELS
                .iter()
                .any(|&(n, _)| local == n || local.starts_with(&format!("{}:", n)));
            if !already {
                options.push(format!("{} (installed, unknown dims)", local));
                option_model_names.push(local.clone());
            }
        }
    }

    // Add custom option
    options.push("custom (enter model name)...".to_string());
    option_model_names.push(String::new()); // sentinel for custom

    // Find default index: current model, or first entry
    let current_model = if cfg.embed_model.trim().is_empty() {
        default_embed_model_for_backend("ollama").to_string()
    } else {
        cfg.embed_model.trim().to_string()
    };
    let default_idx = option_model_names
        .iter()
        .position(|n| {
            !n.is_empty() && (n == &current_model || current_model.starts_with(&format!("{}:", n)))
        })
        .unwrap_or(0);

    let selected = select_option("choose ollama embedding model", &options, Some(default_idx))?
        .unwrap_or(default_idx);

    let chosen_model =
        if selected < option_model_names.len() && !option_model_names[selected].is_empty() {
            option_model_names[selected].clone()
        } else {
            // Custom entry
            let custom = prompt_line("enter ollama model name: ")?;
            let trimmed = custom.trim().to_string();
            if trimmed.is_empty() {
                eprintln!(
                    "warning: empty model name, keeping current: {}",
                    current_model
                );
                return Ok(());
            }
            trimmed
        };

    cfg.embed_model = chosen_model.clone();

    // Offer to pull if not installed and ollama CLI is available
    if reachable {
        let is_installed = local_models
            .iter()
            .any(|m| m == &chosen_model || m.starts_with(&format!("{}:", &chosen_model)));
        if !is_installed && command_available("ollama") {
            match prompt_yes_no(
                &format!("'{}' is not installed locally. pull now?", chosen_model),
                true,
            ) {
                Ok(true) => {
                    if let Err(e) = ollama_pull_model(&chosen_model) {
                        eprintln!("warning: pull failed: {}", e);
                    }
                }
                Ok(false) => {
                    println!(
                        "skipped. run `ollama pull {}` before using this model.",
                        chosen_model
                    );
                }
                Err(e) => {
                    eprintln!("warning: prompt failed: {}", e);
                }
            }
        } else if !is_installed {
            println!(
                "note: '{}' is not installed. run `ollama pull {}` before using this model.",
                chosen_model, chosen_model
            );
        }
    }

    Ok(())
}

/// Interactive bedrock model selection.
fn configure_bedrock_model_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let mut options: Vec<String> = Vec::new();
    let mut option_model_ids: Vec<String> = Vec::new();

    for &(model_id, desc) in KNOWN_BEDROCK_EMBEDDING_MODELS {
        options.push(format!("{} ({})", model_id, desc));
        option_model_ids.push(model_id.to_string());
    }

    options.push("custom (enter model ID or ARN)...".to_string());
    option_model_ids.push(String::new()); // sentinel

    let current_model = if cfg.embed_model.trim().is_empty() {
        default_embed_model_for_backend("bedrock").to_string()
    } else {
        cfg.embed_model.trim().to_string()
    };
    let default_idx = option_model_ids
        .iter()
        .position(|n| !n.is_empty() && n == &current_model)
        .unwrap_or(0);

    let selected = select_option(
        "choose bedrock embedding model",
        &options,
        Some(default_idx),
    )?
    .unwrap_or(default_idx);

    let chosen_model =
        if selected < option_model_ids.len() && !option_model_ids[selected].is_empty() {
            option_model_ids[selected].clone()
        } else {
            let custom = prompt_line("enter bedrock model ID or ARN: ")?;
            let trimmed = custom.trim().to_string();
            if trimmed.is_empty() {
                eprintln!(
                    "warning: empty model ID, keeping current: {}",
                    current_model
                );
                return Ok(());
            }
            trimmed
        };

    cfg.embed_model = chosen_model;
    Ok(())
}

fn select_option(
    prompt: &str,
    options: &[String],
    default: Option<usize>,
) -> Result<Option<usize>, String> {
    if options.is_empty() {
        return Ok(None);
    }
    let default_idx = default.unwrap_or(0).min(options.len().saturating_sub(1));
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() && command_exists("fzf") {
        let mut ordered_indices: Vec<usize> = (0..options.len()).collect();
        ordered_indices.rotate_left(default_idx);

        let mut cmd = Command::new("fzf");
        cmd.arg("--ansi")
            .arg("--prompt")
            .arg(format!("{} > ", prompt))
            .arg("--height")
            .arg("50%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--cycle")
            .arg("--delimiter")
            .arg("\t")
            .arg("--with-nth")
            .arg("2")
            .arg("--no-sort")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching fzf selector: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            for idx in ordered_indices {
                let marker = if idx == default_idx { "[*]" } else { "[ ]" };
                writeln!(stdin, "{}\t{} {}", idx, marker, options[idx])
                    .map_err(|e| format!("failed writing selector input: {}", e))?;
            }
        }
        let output = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for fzf selector: {}", e))?;
        if !output.status.success() {
            return Ok(None);
        }
        let line = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if line.is_empty() {
            return Ok(None);
        }
        if let Some((left, _)) = line.split_once('\t') {
            if let Ok(n) = left.trim().parse::<usize>() {
                if n < options.len() {
                    return Ok(Some(n));
                }
            }
        }
    }

    println!("{}", prompt);
    for (idx, option) in options.iter().enumerate() {
        let marker = if idx == default_idx { "*" } else { " " };
        println!("  {:>2}. [{}] {}", idx + 1, marker, option);
    }
    let raw = prompt_line(&format!("select number (Enter for {}): ", default_idx + 1))?;
    let normalized = strip_terminal_control_sequences(raw.trim());
    if normalized.trim().is_empty() {
        return Ok(Some(default_idx));
    }
    let parsed = normalized
        .trim()
        .parse::<usize>()
        .map_err(|_| "invalid selection: expected a number".to_string())?;
    if parsed < 1 || parsed > options.len() {
        return Err(format!(
            "invalid selection: {} (must be between 1 and {})",
            parsed,
            options.len()
        ));
    }
    Ok(Some(parsed - 1))
}

fn aws_config_file_path() -> PathBuf {
    non_empty_env("AWS_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_tilde("~/.aws/config"))
}

fn aws_credentials_file_path() -> PathBuf {
    non_empty_env("AWS_SHARED_CREDENTIALS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_tilde("~/.aws/credentials"))
}

fn parse_aws_profile_regions() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let path = aws_config_file_path();
    let Ok(raw) = fs::read_to_string(path) else {
        return out;
    };
    let mut current: Option<String> = None;
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') {
            let mut section = t.trim_start_matches('[').trim_end_matches(']').trim();
            if let Some(rest) = section.strip_prefix("profile ") {
                section = rest.trim();
            }
            if section.is_empty() {
                current = None;
            } else {
                current = Some(section.to_string());
            }
            continue;
        }
        if let Some(profile) = &current {
            if let Some((k, v)) = t.split_once('=') {
                if k.trim().eq_ignore_ascii_case("region") {
                    let region = v.trim().to_string();
                    if !region.is_empty() {
                        out.insert(profile.clone(), region);
                    }
                }
            }
        }
    }
    out
}

fn parse_ini_profile_names(path: &Path, treat_profile_prefix: bool) -> HashSet<String> {
    let mut out = HashSet::new();
    let Ok(raw) = fs::read_to_string(path) else {
        return out;
    };
    for line in raw.lines() {
        let t = line.trim();
        if !(t.starts_with('[') && t.ends_with(']')) {
            continue;
        }
        let mut section = t
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim()
            .to_string();
        if treat_profile_prefix {
            if let Some(rest) = section.strip_prefix("profile ") {
                section = rest.trim().to_string();
            }
        }
        if !section.is_empty() {
            out.insert(section);
        }
    }
    out
}

fn aws_cli_list_profiles() -> Vec<String> {
    if !command_available("aws") {
        return Vec::new();
    }
    let output = match Command::new("aws")
        .arg("configure")
        .arg("list-profiles")
        .output()
    {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn list_aws_profiles() -> Vec<AwsProfileChoice> {
    let mut names: HashSet<String> = HashSet::new();
    let mut source_map: HashMap<String, HashSet<String>> = HashMap::new();
    for p in aws_cli_list_profiles() {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-cli".to_string());
    }
    for p in parse_ini_profile_names(&aws_config_file_path(), true) {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-config".to_string());
    }
    for p in parse_ini_profile_names(&aws_credentials_file_path(), false) {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-credentials".to_string());
    }
    let region_map = parse_aws_profile_regions();
    let mut out: Vec<AwsProfileChoice> = names
        .into_iter()
        .map(|name| {
            let source = source_map
                .remove(&name)
                .map(|set| {
                    let mut v: Vec<String> = set.into_iter().collect();
                    v.sort();
                    v.join("+")
                })
                .unwrap_or_else(|| "local".to_string());
            AwsProfileChoice {
                region: region_map.get(&name).cloned(),
                name,
                source,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn aws_region_for_profile_name(profile: &str) -> Option<String> {
    let map = parse_aws_profile_regions();
    map.get(profile).cloned()
}

fn find_isengardcli_binary() -> Option<String> {
    if command_available("isengardcli") {
        return Some("isengardcli".to_string());
    }
    let fallback = expand_tilde("~/Scripts/isengardcli/isengardcli");
    if is_executable_file(&fallback) {
        return Some(fallback.to_string_lossy().to_string());
    }
    None
}

fn extract_role_from_refresh_cmd(cmd: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "--role" {
            if let Some(role) = parts.get(i + 1) {
                return Some(role.to_string());
            }
        }
    }
    None
}

fn list_isengard_accounts(cli: &str) -> Result<Vec<IsengardAccountChoice>, String> {
    let output = Command::new(cli)
        .arg("ls")
        .arg("--output")
        .arg("json")
        .arg("--all")
        .output()
        .map_err(|e| format!("failed running isengardcli ls: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("isengardcli ls failed: {}", detail));
    }
    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    let parsed: Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid isengard json output: {}", e))?;
    let arr = parsed
        .as_array()
        .ok_or_else(|| "unexpected isengard output format".to_string())?;
    let mut out = Vec::new();
    for row in arr {
        let status = row.get("Status").and_then(|v| v.as_str()).unwrap_or("");
        if !status.is_empty() && !status.eq_ignore_ascii_case("ACTIVE") {
            continue;
        }
        let email = row
            .get("Email")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let name = row
            .get("Name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let alias = row
            .get("Alias")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let account_id = row
            .get("AWSAccountID")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let favorite = row
            .get("Favorite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let account_ref = if !email.is_empty() {
            email.to_string()
        } else if !alias.is_empty() {
            alias.to_string()
        } else if !name.is_empty() {
            name.to_string()
        } else {
            continue;
        };
        let display_name = if !alias.is_empty() {
            alias.to_string()
        } else if !name.is_empty() {
            name.to_string()
        } else {
            account_ref.clone()
        };
        let mut label = display_name;
        if !email.is_empty() {
            label.push_str(&format!(" <{}>", email));
        }
        if !account_id.is_empty() {
            label.push_str(&format!(" [{}]", account_id));
        }
        out.push(IsengardAccountChoice {
            account_ref,
            label,
            favorite,
        });
    }
    out.sort_by(|a, b| {
        b.favorite
            .cmp(&a.favorite)
            .then_with(|| a.label.to_lowercase().cmp(&b.label.to_lowercase()))
    });
    out.dedup_by(|a, b| a.account_ref == b.account_ref);
    Ok(out)
}

#[derive(Clone, Debug)]
struct AutotuneOptions {
    dry_run: bool,
    deep: bool,
    max_events: usize,
    limit: usize,
}

#[derive(Clone, Debug)]
struct AutotuneExample {
    query: String,
    path: String,
    weight: f64,
}

#[derive(Clone, Debug)]
struct AutotuneOutcome {
    cfg: ConfigValues,
    examples_used: usize,
    baseline_mrr: f64,
    baseline_hit1: f64,
    baseline_hit3: f64,
    best_mrr: f64,
    best_hit1: f64,
    best_hit3: f64,
    candidates_tested: usize,
    used_history: bool,
}

fn parse_autotune_options(args: &[OsString]) -> AutotuneOptions {
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

fn config_rows() -> Vec<(&'static str, &'static str)> {
    vec![
        ("root", "Root path hint (not auto-tracked)"),
        ("embed_backend", "Embedding backend"),
        ("embed_model", "Embedding model id"),
        ("aws_profile", "AWS profile for Bedrock"),
        ("aws_region", "AWS region for Bedrock"),
        ("aws_refresh_cmd", "Credential refresh command (optional)"),
        ("bedrock_concurrency", "Bedrock invoke concurrency"),
        ("bedrock_max_retries", "Bedrock max retry attempts"),
        ("bedrock_retry_base_ms", "Bedrock retry base backoff (ms)"),
        ("retrieval_backend", "Retrieval backend"),
        ("local_embed_dim", "Embedding dimension for local models"),
        ("max_chars_per_project", "Indexing cap per project"),
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
    ]
}

fn config_enum_options(key: &str) -> Option<Vec<&'static str>> {
    match key {
        "embed_backend" => Some(vec!["ollama", "bedrock"]),
        "retrieval_backend" => Some(vec!["lancedb"]),
        _ => None,
    }
}

fn config_value_string(cfg: &ConfigValues, key: &str) -> Option<String> {
    match key {
        "root" => Some(cfg.root.to_string_lossy().to_string()),
        "embed_backend" => Some(cfg.embed_backend.clone()),
        "embed_model" => Some(cfg.embed_model.clone()),
        "aws_profile" => Some(cfg.aws_profile.clone()),
        "aws_region" => Some(cfg.aws_region.clone()),
        "aws_refresh_cmd" => Some(cfg.aws_refresh_cmd.clone()),
        "bedrock_concurrency" => Some(cfg.bedrock_concurrency.to_string()),
        "bedrock_max_retries" => Some(cfg.bedrock_max_retries.to_string()),
        "bedrock_retry_base_ms" => Some(cfg.bedrock_retry_base_ms.to_string()),
        "retrieval_backend" => Some(cfg.retrieval_backend.clone()),
        "local_embed_dim" => Some(cfg.local_embed_dim.to_string()),
        "max_chars_per_project" => Some(cfg.max_chars_per_project.to_string()),
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
        _ => None,
    }
}

fn config_set_value(cfg: &mut ConfigValues, key: &str, raw: &str) -> Result<(), String> {
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
            if !matches!(v.as_str(), "ollama" | "bedrock") {
                return Err("embed_backend must be one of: ollama, bedrock".to_string());
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
        "lexical_candidates" => {
            cfg.lexical_candidates = value
                .parse::<i64>()
                .map_err(|_| "lexical_candidates must be an integer".to_string())?
                .clamp(10, 5000);
        }
        "vector_candidates" => {
            cfg.vector_candidates = value
                .parse::<i64>()
                .map_err(|_| "vector_candidates must be an integer".to_string())?
                .clamp(10, 5000);
        }
        "rank_chunk_semantic_weight" => {
            cfg.rank_chunk_semantic_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_semantic_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_chunk_lexical_weight" => {
            cfg.rank_chunk_lexical_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_lexical_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_chunk_graph_weight" => {
            cfg.rank_chunk_graph_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_chunk_graph_weight must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_quality_mix" => {
            cfg.rank_quality_mix = value
                .parse::<f64>()
                .map_err(|_| "rank_quality_mix must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_good_boost" => {
            cfg.rank_relation_quality_good_boost = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_good_boost must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_weak_penalty" => {
            cfg.rank_relation_quality_weak_penalty = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_weak_penalty must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_relation_quality_wrong_penalty" => {
            cfg.rank_relation_quality_wrong_penalty = value
                .parse::<f64>()
                .map_err(|_| "rank_relation_quality_wrong_penalty must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "rank_project_content_weight" => {
            cfg.rank_project_content_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_content_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_semantic_weight" => {
            cfg.rank_project_semantic_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_semantic_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_path_weight" => {
            cfg.rank_project_path_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_path_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_graph_weight" => {
            cfg.rank_project_graph_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_graph_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "rank_project_frecency_weight" => {
            cfg.rank_project_frecency_weight = value
                .parse::<f64>()
                .map_err(|_| "rank_project_frecency_weight must be a number".to_string())?
                .clamp(0.0, 2.0);
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
            cfg.graph_same_project_high = value
                .parse::<f64>()
                .map_err(|_| "graph_same_project_high must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_same_project_low" => {
            cfg.graph_same_project_low = value
                .parse::<f64>()
                .map_err(|_| "graph_same_project_low must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_related_base" => {
            cfg.graph_related_base = value
                .parse::<f64>()
                .map_err(|_| "graph_related_base must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        "graph_related_scale" => {
            cfg.graph_related_scale = value
                .parse::<f64>()
                .map_err(|_| "graph_related_scale must be a number".to_string())?
                .clamp(0.0, 2.0);
        }
        "graph_related_cap" => {
            cfg.graph_related_cap = value
                .parse::<f64>()
                .map_err(|_| "graph_related_cap must be a number".to_string())?
                .clamp(0.0, 1.0);
        }
        _ => return Err(format!("unknown config key '{}'", key)),
    }
    Ok(())
}

fn print_config_values(cfg: &ConfigValues) {
    for (key, hint) in config_rows() {
        if let Some(v) = config_value_string(cfg, key) {
            println!("{:<30} = {:<24} # {}", key, v, hint);
        }
    }
}

fn prompt_line_to(prompt: &str, stderr_prompt: bool) -> Result<String, String> {
    if stderr_prompt {
        eprint!("{}", prompt);
        std::io::stderr()
            .flush()
            .map_err(|e| format!("failed flushing stderr: {}", e))?;
    } else {
        print!("{}", prompt);
        std::io::stdout()
            .flush()
            .map_err(|e| format!("failed flushing stdout: {}", e))?;
    }
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| format!("failed reading input: {}", e))?;
    Ok(line.trim_end_matches(&['\r', '\n'][..]).to_string())
}

fn prompt_line(prompt: &str) -> Result<String, String> {
    prompt_line_to(prompt, false)
}

fn prompt_line_stderr(prompt: &str) -> Result<String, String> {
    prompt_line_to(prompt, true)
}

/// Prompt for yes/no with a default. Shows `[Y/n]` or `[y/N]`.
fn prompt_yes_no(prompt: &str, default_yes: bool) -> Result<bool, String> {
    let hint = if default_yes { "[Y/n]" } else { "[y/N]" };
    let raw = prompt_line(&format!("{} {}: ", prompt, hint))?;
    let trimmed = strip_terminal_control_sequences(raw.trim())
        .trim()
        .to_lowercase();
    if trimmed.is_empty() {
        return Ok(default_yes);
    }
    match trimmed.as_str() {
        "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        _ => Ok(default_yes),
    }
}

fn tty_ui_available() -> bool {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return false;
    }
    if !command_exists("stty") {
        return false;
    }
    run_stty_capture(["-g"]).is_ok()
}

#[derive(Clone, Debug)]
enum ConfigTuiMode {
    Navigate,
    Edit { key: String, buffer: String },
}

#[derive(Clone, Debug)]
struct ConfigTuiState {
    selected: usize,
    scroll: usize,
    dirty: bool,
    discard_armed: bool,
    status: String,
    mode: ConfigTuiMode,
}

enum ConfigKey {
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
    Enter,
    Tab,
    Esc,
    Backspace,
    CtrlC,
    Char(char),
    Unknown,
}

struct ConfigTuiGuard {
    stty_state: Option<String>,
}

impl Drop for ConfigTuiGuard {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(b"\x1b[0m\x1b[?25h\x1b[?1049l");
        let _ = stdout.flush();
        if let Some(state) = &self.stty_state {
            let _ = run_stty(args_slice([state.as_str()]));
        }
    }
}

fn enter_config_tui_mode() -> Result<ConfigTuiGuard, String> {
    if !command_exists("stty") {
        return Err("stty is required for full-screen config mode".to_string());
    }
    let state_str = run_stty_capture(["-g"]).map_err(|_| "failed reading tty mode".to_string())?;
    run_stty(["raw", "-echo", "min", "0", "time", "1"])
        .map_err(|_| "failed switching tty to raw mode".to_string())?;

    let mut stdout = std::io::stdout();
    stdout
        .write_all(b"\x1b[?1049h\x1b[?25l")
        .map_err(|e| format!("failed entering alternate screen: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing terminal init: {}", e))?;

    Ok(ConfigTuiGuard {
        stty_state: Some(state_str),
    })
}

fn args_slice<const N: usize>(arr: [&str; N]) -> [&str; N] {
    arr
}

fn run_stty<const N: usize>(args: [&str; N]) -> Result<(), String> {
    let mut cmd = String::from("stty");
    for arg in args {
        cmd.push(' ');
        cmd.push_str(arg);
    }
    cmd.push_str(" < /dev/tty > /dev/tty 2>/dev/null");
    let status = Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .status()
        .map_err(|e| format!("failed running stty: {}", e))?;
    if status.success() {
        Ok(())
    } else {
        Err("stty returned non-zero status".to_string())
    }
}

fn run_stty_capture<const N: usize>(args: [&str; N]) -> Result<String, String> {
    let mut cmd = String::from("stty");
    for arg in args {
        cmd.push(' ');
        cmd.push_str(arg);
    }
    cmd.push_str(" < /dev/tty 2>/dev/null");
    let out = Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .output()
        .map_err(|e| format!("failed running stty capture: {}", e))?;
    if !out.status.success() {
        return Err("stty capture returned non-zero status".to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn read_config_key() -> Result<Option<ConfigKey>, String> {
    let mut stdin = std::io::stdin();
    let mut buf = [0u8; 32];
    let n = stdin
        .read(&mut buf)
        .map_err(|e| format!("failed reading key input: {}", e))?;
    if n == 0 {
        return Ok(None);
    }
    let mut seq = buf[..n].to_vec();
    if seq[0] == 0x1b && seq.len() == 1 {
        let mut extra = [0u8; 16];
        let n2 = stdin
            .read(&mut extra)
            .map_err(|e| format!("failed reading escape sequence: {}", e))?;
        if n2 > 0 {
            seq.extend_from_slice(&extra[..n2]);
        }
    }
    let key = match seq[0] {
        0x03 => ConfigKey::CtrlC,
        0x09 => ConfigKey::Tab,
        b'\r' | b'\n' => ConfigKey::Enter,
        0x7f | 0x08 => ConfigKey::Backspace,
        0x1b => {
            if seq.len() >= 3 && seq[1] == b'[' {
                match seq[2] {
                    b'A' => ConfigKey::Up,
                    b'B' => ConfigKey::Down,
                    b'C' => ConfigKey::Right,
                    b'D' => ConfigKey::Left,
                    b'H' => ConfigKey::Home,
                    b'F' => ConfigKey::End,
                    b'5' if seq.get(3) == Some(&b'~') => ConfigKey::PageUp,
                    b'6' if seq.get(3) == Some(&b'~') => ConfigKey::PageDown,
                    b'1' if seq.get(3) == Some(&b'~') => ConfigKey::Home,
                    b'4' if seq.get(3) == Some(&b'~') => ConfigKey::End,
                    _ => ConfigKey::Esc,
                }
            } else {
                ConfigKey::Esc
            }
        }
        b => {
            if (0x20..=0x7e).contains(&b) {
                ConfigKey::Char(b as char)
            } else {
                ConfigKey::Unknown
            }
        }
    };
    Ok(Some(key))
}

fn terminal_size_fallback() -> (usize, usize) {
    let cols = env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let rows = env::var("LINES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24);
    (cols.max(40), rows.max(12))
}

fn terminal_size_ioctl() -> Option<(usize, usize)> {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        // Try stdout first, then /dev/tty
        let mut ret = libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws);
        if ret != 0 || ws.ws_col == 0 {
            let tty_fd = libc::open(
                b"/dev/tty\0".as_ptr() as *const libc::c_char,
                libc::O_RDONLY,
            );
            if tty_fd >= 0 {
                ret = libc::ioctl(tty_fd, libc::TIOCGWINSZ, &mut ws);
                libc::close(tty_fd);
            } else {
                return None;
            }
        }
        if ret == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            Some((ws.ws_col as usize, ws.ws_row as usize))
        } else {
            None
        }
    }
}

/// Probe actual visible terminal size using ANSI cursor position report.
/// Must be called after raw mode is enabled. Moves cursor to far bottom-right
/// corner and asks the terminal to report the position — this gives the real
/// visible dimensions regardless of what ioctl/stty report.
fn terminal_size_cursor_probe() -> Option<(usize, usize)> {
    use std::io::{Read, Write};
    let mut tty_out = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/tty")
        .ok()?;
    let mut tty_in = std::fs::File::open("/dev/tty").ok()?;

    // Save cursor, move to 999;999, query position, restore cursor
    tty_out
        .write_all(b"\x1b[s\x1b[999;999H\x1b[6n\x1b[u")
        .ok()?;
    tty_out.flush().ok()?;

    // Read response: \x1b[{rows};{cols}R
    let mut buf = [0u8; 32];
    let mut total = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
    loop {
        if std::time::Instant::now() > deadline || total >= buf.len() {
            break;
        }
        match tty_in.read(&mut buf[total..total + 1]) {
            Ok(1) => {
                total += 1;
                if buf[total - 1] == b'R' {
                    break;
                }
            }
            Ok(_) => continue, // VTIME expired, no data yet — keep trying until deadline
            Err(_) => break,
        }
    }
    let resp = std::str::from_utf8(&buf[..total]).ok()?;
    let inner = resp.strip_prefix("\x1b[")?.strip_suffix('R')?;
    let mut parts = inner.split(';');
    let rows: usize = parts.next()?.parse().ok()?;
    let cols: usize = parts.next()?.parse().ok()?;
    if cols > 0 && rows > 0 {
        Some((cols, rows))
    } else {
        None
    }
}

fn terminal_size_stty() -> (usize, usize) {
    // Prefer direct ioctl — no subprocess, no login shell interference
    if let Some((cols, rows)) = terminal_size_ioctl() {
        return (cols.max(40), rows.max(12));
    }
    // Fallback: stty subprocess
    if let Ok(raw) = run_stty_capture(["size"]) {
        let mut parts = raw.split_whitespace();
        if let (Some(rows), Some(cols)) = (parts.next(), parts.next()) {
            if let (Ok(r), Ok(c)) = (rows.parse::<usize>(), cols.parse::<usize>()) {
                return (c.max(40), r.max(12));
            }
        }
    }
    terminal_size_fallback()
}

fn clipped(s: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= max_width {
        return s.to_string();
    }
    if max_width <= 1 {
        return "…".to_string();
    }
    let mut out = String::new();
    for ch in s.chars().take(max_width.saturating_sub(1)) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn config_rows_count() -> usize {
    config_rows().len()
}

fn cycle_enum_setting(cfg: &mut ConfigValues, key: &str, direction: i32) -> Result<bool, String> {
    let Some(options) = config_enum_options(key) else {
        return Ok(false);
    };
    if options.is_empty() {
        return Ok(false);
    }
    let current = config_value_string(cfg, key).unwrap_or_default();
    let mut pos = options.iter().position(|v| *v == current).unwrap_or(0) as i32;
    pos += direction.signum();
    if pos < 0 {
        pos = options.len() as i32 - 1;
    }
    if pos as usize >= options.len() {
        pos = 0;
    }
    let next = options[pos as usize];
    config_set_value(cfg, key, next)?;
    Ok(true)
}

fn draw_config_tui(
    stdout: &mut std::io::Stdout,
    cfg: &ConfigValues,
    cfg_path: &Path,
    state: &mut ConfigTuiState,
    term_size: (usize, usize),
) -> Result<(), String> {
    let rows = config_rows();
    if rows.is_empty() {
        return Err("no config rows available".to_string());
    }
    let (width, height) = term_size;

    let header_rows = 4usize;
    let footer_rows = 3usize;
    let min_height = header_rows + footer_rows + 2;
    let mut visible_rows = height.saturating_sub(header_rows + footer_rows);
    if height < min_height {
        visible_rows = 1;
    }
    if state.selected >= rows.len() {
        state.selected = rows.len().saturating_sub(1);
    }
    if state.selected < state.scroll {
        state.scroll = state.selected;
    }
    if state.selected >= state.scroll + visible_rows {
        state.scroll = state
            .selected
            .saturating_sub(visible_rows.saturating_sub(1));
    }

    let title = "retrivio config";
    let mode_text = match &state.mode {
        ConfigTuiMode::Navigate => "mode: navigate",
        ConfigTuiMode::Edit { .. } => "mode: edit",
    };
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "\x1b[36;1m{}\x1b[0m \x1b[90m{}\x1b[0m",
        clipped(title, 20),
        clipped(mode_text, width.saturating_sub(22))
    ));
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(&format!("config: {}", cfg_path.display()), width)
    ));
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(
            "↑/↓ move  ←/→ cycle enums  Enter edit  a autotune  s save  q discard",
            width
        )
    ));

    // Adaptive column widths based on terminal width
    let key_w = ((width * 40) / 100).clamp(20, 36);
    let value_w = ((width * 14) / 100).clamp(10, 16);
    let desc_w = width.saturating_sub(key_w + value_w + 2).max(1);
    lines.push(format!(
        "\x1b[1m{}\x1b[0m",
        clipped(
            &format!(
                "{:<key_w$} {:<value_w$} {}",
                "key",
                "value",
                "description",
                key_w = key_w,
                value_w = value_w
            ),
            width
        )
    ));

    for i in 0..visible_rows {
        let row_index = state.scroll + i;
        if row_index >= rows.len() {
            break;
        }
        let (key, hint) = rows[row_index];
        let value = config_value_string(cfg, key).unwrap_or_default();
        let key_txt = clipped(key, key_w);
        let val_txt = clipped(&value, value_w);
        let hint_txt = clipped(hint, desc_w);
        let row_text = clipped(
            &format!(
                "{:<key_w$} {:<value_w$} {}",
                key_txt,
                val_txt,
                hint_txt,
                key_w = key_w,
                value_w = value_w
            ),
            width,
        );
        if row_index == state.selected {
            lines.push(format!("\x1b[7m{}\x1b[0m", row_text));
        } else {
            lines.push(row_text);
        }
    }

    let mode_line = match &state.mode {
        ConfigTuiMode::Navigate => {
            "navigate: Enter edit | s save | q discard | a autotune".to_string()
        }
        ConfigTuiMode::Edit { key, buffer } => {
            format!("edit {} = {}  (Enter apply, Esc cancel)", key, buffer)
        }
    };
    while lines.len() + footer_rows < height {
        lines.push(String::new());
    }
    lines.push(format!(
        "\x1b[90m{}\x1b[0m",
        clipped(
            &format!(
                "item {}/{}{}",
                state.selected + 1,
                rows.len(),
                if state.dirty { "  [unsaved]" } else { "" }
            ),
            width
        )
    ));
    lines.push(format!("\x1b[33m{}\x1b[0m", clipped(&mode_line, width)));
    lines.push(format!("\x1b[36m{}\x1b[0m", clipped(&state.status, width)));

    let payload = format!("\x1b[H\x1b[2J{}", lines.join("\r\n"));
    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing TUI frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing TUI frame: {}", e))?;
    Ok(())
}

fn run_config_tui(cwd: &Path, cfg_path: &Path, working: &mut ConfigValues) -> Result<bool, String> {
    let _guard = enter_config_tui_mode()?;
    let mut stdout = std::io::stdout();

    // Probe terminal size — cursor probe is most reliable in raw mode
    let mut term_size = terminal_size_cursor_probe().unwrap_or_else(|| terminal_size_stty());

    let mut state = ConfigTuiState {
        selected: 0,
        scroll: 0,
        dirty: false,
        discard_armed: false,
        status: "full-screen config loaded".to_string(),
        mode: ConfigTuiMode::Navigate,
    };
    let total_rows = config_rows_count();
    if total_rows == 0 {
        return Err("no editable config rows".to_string());
    }
    loop {
        draw_config_tui(&mut stdout, working, cfg_path, &mut state, term_size)?;
        let Some(key_event) = read_config_key()? else {
            continue;
        };
        match &mut state.mode {
            ConfigTuiMode::Navigate => match key_event {
                ConfigKey::Up => {
                    if state.selected > 0 {
                        state.selected -= 1;
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Down => {
                    if state.selected + 1 < total_rows {
                        state.selected += 1;
                    }
                    state.discard_armed = false;
                }
                ConfigKey::PageUp => {
                    state.selected = state.selected.saturating_sub(10);
                    state.discard_armed = false;
                }
                ConfigKey::PageDown => {
                    state.selected = (state.selected + 10).min(total_rows.saturating_sub(1));
                    state.discard_armed = false;
                }
                ConfigKey::Home => {
                    state.selected = 0;
                    state.discard_armed = false;
                }
                ConfigKey::End => {
                    state.selected = total_rows.saturating_sub(1);
                    state.discard_armed = false;
                }
                ConfigKey::Left | ConfigKey::Right => {
                    let rows = config_rows();
                    let key = rows[state.selected].0;
                    let direction = if matches!(key_event, ConfigKey::Left) {
                        -1
                    } else {
                        1
                    };
                    match cycle_enum_setting(working, key, direction) {
                        Ok(true) => {
                            state.dirty = true;
                            state.status = format!(
                                "updated {} -> {}",
                                key,
                                config_value_string(working, key).unwrap_or_default()
                            );
                        }
                        Ok(false) => {
                            state.status =
                                "selected key has no enum options; press Enter to edit".to_string();
                        }
                        Err(e) => {
                            state.status = e;
                        }
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Enter => {
                    let rows = config_rows();
                    let key = rows[state.selected].0.to_string();
                    let value = config_value_string(working, &key).unwrap_or_default();
                    state.mode = ConfigTuiMode::Edit { key, buffer: value };
                    state.status =
                        "editing value; press Enter to apply or Esc to cancel".to_string();
                    state.discard_armed = false;
                }
                ConfigKey::Char('s') => {
                    write_config_file(cfg_path, working)?;
                    state.status = format!("config saved: {}", cfg_path.display());
                    return Ok(true);
                }
                ConfigKey::Char('a') => {
                    state.status = "running autotune...".to_string();
                    draw_config_tui(&mut stdout, working, cfg_path, &mut state, term_size)?;
                    let result = (|| -> Result<AutotuneOutcome, String> {
                        let dbp = db_path(cwd);
                        ensure_db_schema(&dbp)?;
                        let conn = open_db_rw(&dbp)?;
                        autotune_recommendation(&conn, working, 320, 40, false)
                    })();
                    match result {
                        Ok(outcome) => {
                            *working = outcome.cfg;
                            state.dirty = true;
                            state.status = format!(
                                "autotune applied: mrr {:.4}->{:.4}, hit3 {:.4}->{:.4}",
                                outcome.baseline_mrr,
                                outcome.best_mrr,
                                outcome.baseline_hit3,
                                outcome.best_hit3
                            );
                        }
                        Err(e) => {
                            state.status = format!("autotune failed: {}", e);
                        }
                    }
                    state.discard_armed = false;
                }
                ConfigKey::Esc | ConfigKey::Char('q') => {
                    if state.dirty && !state.discard_armed {
                        state.status =
                            "unsaved changes: press q again to discard, or s to save".to_string();
                        state.discard_armed = true;
                    } else {
                        return Ok(false);
                    }
                }
                ConfigKey::CtrlC => {
                    return Ok(false);
                }
                _ => {}
            },
            ConfigTuiMode::Edit { key, buffer } => match key_event {
                ConfigKey::Esc => {
                    state.mode = ConfigTuiMode::Navigate;
                    state.status = "edit cancelled".to_string();
                }
                ConfigKey::Enter => match config_set_value(working, key, buffer) {
                    Ok(_) => {
                        state.dirty = true;
                        state.status = format!(
                            "updated {} = {}",
                            key,
                            config_value_string(working, key).unwrap_or_default()
                        );
                        state.mode = ConfigTuiMode::Navigate;
                    }
                    Err(e) => {
                        state.status = e;
                    }
                },
                ConfigKey::Backspace => {
                    buffer.pop();
                }
                ConfigKey::Char(c) => {
                    buffer.push(c);
                }
                ConfigKey::CtrlC => return Ok(false),
                _ => {}
            },
        }
    }
}

fn select_config_menu_key(cfg: &ConfigValues) -> Result<Option<String>, String> {
    let mut rows: Vec<(String, String, String)> = vec![
        (
            "@save".to_string(),
            "Save and exit".to_string(),
            "Write config to disk".to_string(),
        ),
        (
            "@autotune".to_string(),
            "Autotune ranking".to_string(),
            "Use selection history to tune weights".to_string(),
        ),
        (
            "@discard".to_string(),
            "Discard and exit".to_string(),
            "Exit without saving".to_string(),
        ),
    ];
    for (key, hint) in config_rows() {
        rows.push((
            key.to_string(),
            config_value_string(cfg, key).unwrap_or_default(),
            hint.to_string(),
        ));
    }

    if tty_ui_available() && command_exists("fzf") {
        let mut payload = String::new();
        for (key, value, hint) in &rows {
            payload.push_str(&format!("{}\t{}\t{}\n", key, value, hint));
        }
        let mut cmd = Command::new("fzf");
        cmd.arg("--height=80%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--delimiter=\t")
            .arg("--with-nth=1,2,3")
            .arg("--prompt")
            .arg("retrivio config> ")
            .arg("--header")
            .arg("key | value | description (Enter: select)")
            .arg("--no-sort")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching config picker: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(payload.as_bytes())
                .map_err(|e| format!("failed writing config picker input: {}", e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for config picker: {}", e))?;
        if !out.status.success() {
            return Ok(None);
        }
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if line.is_empty() {
            return Ok(None);
        }
        let key = line.split('\t').next().unwrap_or("").trim().to_string();
        if key.is_empty() {
            return Ok(None);
        }
        return Ok(Some(key));
    }

    for (idx, (_key, value, hint)) in rows.iter().enumerate() {
        println!("{:>2}. {:<26} {:<24} {}", idx + 1, rows[idx].0, value, hint);
    }
    let raw = prompt_line("select item number (empty to cancel): ")?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    let idx = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| "invalid index".to_string())?;
    if idx == 0 || idx > rows.len() {
        return Err("index out of range".to_string());
    }
    Ok(Some(rows[idx - 1].0.clone()))
}

fn select_enum_option(key: &str, current: &str) -> Result<Option<String>, String> {
    let options = match config_enum_options(key) {
        Some(v) => v,
        None => return Ok(None),
    };
    if tty_ui_available() && command_exists("fzf") {
        let mut payload = String::new();
        for option in &options {
            payload.push_str(option);
            payload.push('\n');
        }
        let mut cmd = Command::new("fzf");
        cmd.arg("--height=40%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--prompt")
            .arg(format!("{} (current: {})> ", key, current))
            .arg("--header")
            .arg("choose value")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching enum selector: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(payload.as_bytes())
                .map_err(|e| format!("failed writing enum selector input: {}", e))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for enum selector: {}", e))?;
        if !out.status.success() {
            return Ok(None);
        }
        let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if value.is_empty() {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    println!("{} options: {}", key, options.join(", "));
    let raw = prompt_line(&format!("new value [{}]: ", current))?;
    if raw.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(raw.trim().to_string()))
}

fn edit_config_key_interactive(cfg: &mut ConfigValues, key: &str) -> Result<(), String> {
    let current = config_value_string(cfg, key).unwrap_or_default();
    if let Some(chosen) = select_enum_option(key, &current)? {
        config_set_value(cfg, key, &chosen)?;
        return Ok(());
    }
    println!("editing {} (current='{}')", key, current);
    println!("tip: enter ':empty' to clear string values, empty input keeps current value");
    let raw = prompt_line("new value: ")?;
    let next = if raw.trim().is_empty() {
        current
    } else if raw.trim() == ":empty" {
        String::new()
    } else {
        raw.trim().to_string()
    };
    config_set_value(cfg, key, &next)
}

fn run_config_edit_legacy(cwd: &Path) -> Result<(), String> {
    let cfg_path = config_path(cwd);
    let mut working = ConfigValues::from_map(load_config_values(&cfg_path));
    let original = working.clone();
    loop {
        let selected = select_config_menu_key(&working)?;
        let Some(key) = selected else {
            println!("config edit cancelled (no changes saved)");
            return Ok(());
        };
        match key.as_str() {
            "@save" => {
                write_config_file(&cfg_path, &working)?;
                println!("config saved: {}", cfg_path.display());
                if let Some(reason) =
                    refresh_reembed_requirement_for_config_change(cwd, &original, &working)?
                {
                    println!("warning: {}", reason);
                }
                return Ok(());
            }
            "@discard" => {
                println!("config changes discarded");
                return Ok(());
            }
            "@autotune" => {
                let dbp = db_path(cwd);
                ensure_db_schema(&dbp)?;
                let conn = open_db_rw(&dbp)?;
                let outcome = autotune_recommendation(&conn, &working, 320, 40, false)?;
                working = outcome.cfg;
                println!(
                    "autotune: examples={} baseline_mrr={:.4} best_mrr={:.4} candidates={}",
                    outcome.examples_used,
                    outcome.baseline_mrr,
                    outcome.best_mrr,
                    outcome.candidates_tested
                );
                println!(
                    "autotune: baseline_hit1={:.4} best_hit1={:.4} baseline_hit3={:.4} best_hit3={:.4}",
                    outcome.baseline_hit1,
                    outcome.best_hit1,
                    outcome.baseline_hit3,
                    outcome.best_hit3
                );
                if !outcome.used_history {
                    println!("autotune: used heuristic initialization (not enough history)");
                }
            }
            _ => {
                edit_config_key_interactive(&mut working, &key)?;
                if let Some(v) = config_value_string(&working, &key) {
                    println!("updated {} = {}", key, v);
                }
            }
        }
    }
}

fn run_config_edit(cwd: &Path) -> Result<(), String> {
    let cfg_path = config_path(cwd);
    let mut working = ConfigValues::from_map(load_config_values(&cfg_path));
    let original = working.clone();
    if tty_ui_available() {
        match run_config_tui(cwd, &cfg_path, &mut working) {
            Ok(saved) => {
                if saved {
                    println!("config saved: {}", cfg_path.display());
                    if let Some(reason) =
                        refresh_reembed_requirement_for_config_change(cwd, &original, &working)?
                    {
                        println!("warning: {}", reason);
                    }
                } else {
                    println!("config changes discarded");
                }
                return Ok(());
            }
            Err(e) => {
                eprintln!(
                    "warning: full-screen config unavailable ({}); falling back to legacy editor",
                    e
                );
            }
        }
    }
    run_config_edit_legacy(cwd)
}

fn run_config_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio config [edit|show|set <key> <value>|autotune] [options]");
        println!("examples:");
        println!("  retrivio config                  # interactive editor");
        println!("  keybinds in editor: ↑/↓ move, Enter edit, ←/→ cycle enums, a autotune, s save, q discard");
        println!("  retrivio config show             # print current values");
        println!("  retrivio config set graph_seed_limit 12");
        println!("  retrivio config autotune --dry-run --deep --max-events 500");
        return;
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let action = args
        .first()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_else(|| "edit".to_string());
    match action.as_str() {
        "edit" => {
            eprintln!("config: {}", config_path(&cwd).display());
            run_config_edit(&cwd).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "show" => {
            let cp = config_path(&cwd);
            println!("config: {}", cp.display());
            let cfg = ConfigValues::from_map(load_config_values(&cp));
            print_config_values(&cfg);
        }
        "set" => {
            if args.len() < 3 {
                eprintln!("error: usage: retrivio config set <key> <value>");
                process::exit(2);
            }
            let key = args[1].to_string_lossy().to_string();
            let value = args[2..]
                .iter()
                .map(|v| v.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let cfg_path = config_path(&cwd);
            let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
            let before_cfg = cfg.clone();
            config_set_value(&mut cfg, &key, &value).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(2);
            });
            write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            println!(
                "config updated ({}): {}={}",
                cfg_path.display(),
                key,
                config_value_string(&cfg, &key).unwrap_or_default()
            );
            if let Some(reason) =
                refresh_reembed_requirement_for_config_change(&cwd, &before_cfg, &cfg)
                    .unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    })
            {
                println!("warning: {}", reason);
            }
        }
        "autotune" => {
            run_autotune_cmd(&args[1..]);
        }
        other => {
            eprintln!("error: unknown config action '{}'", other);
            process::exit(2);
        }
    }
}

fn event_path_to_project(raw: &str, known_projects: &HashSet<String>) -> Option<String> {
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

fn load_autotune_examples(
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

fn evaluate_candidate_mrr(
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

fn unique_i64(values: Vec<i64>) -> Vec<i64> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for v in values {
        if seen.insert(v) {
            out.push(v);
        }
    }
    out
}

fn unique_f64(values: Vec<f64>) -> Vec<f64> {
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

fn normalize3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let av = a.max(0.0);
    let bv = b.max(0.0);
    let cv = c.max(0.0);
    let sum = av + bv + cv;
    if sum <= 0.0 {
        return (0.66, 0.24, 0.10);
    }
    (av / sum, bv / sum, cv / sum)
}

fn normalize5(vals: [f64; 5]) -> [f64; 5] {
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

fn autotune_key(cfg: &ConfigValues) -> String {
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

fn maybe_promote_candidate(
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

fn autotune_recommendation(
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
            (best_cfg.lexical_candidates - 40).clamp(20, 5000),
            best_cfg.lexical_candidates.clamp(20, 5000),
            (best_cfg.lexical_candidates + 40).clamp(20, 5000),
        ]);
        let vector_vals = unique_i64(vec![
            (best_cfg.vector_candidates - 40).clamp(20, 5000),
            best_cfg.vector_candidates.clamp(20, 5000),
            (best_cfg.vector_candidates + 40).clamp(20, 5000),
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

fn autotune_snapshot(cfg: &ConfigValues) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (key, _) in config_rows() {
        if let Some(value) = config_value_string(cfg, key) {
            out.insert(key.to_string(), value);
        }
    }
    out
}

fn write_autotune_report(
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

fn run_autotune_cmd(args: &[OsString]) {
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

fn run_graph_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio graph [doctor|status|start|stop|provision|view|open|neighbors|lineage] [--path <project-or-child-path>] [--limit <n>] [--threshold <0..1>] [--depth <1..3>]");
        println!("quick examples:");
        println!("  retrivio ui");
        println!(
            "  retrivio graph neighbors --path ~/projects/sample-project --limit 12"
        );
        println!(
            "  retrivio graph lineage --path ~/projects/sample-project --depth 2 --threshold 0.60"
        );
        println!("note: retrieval uses embedded LanceDB + SQLite FTS5 (no external server needed)");
        return;
    }

    let mut action = "status".to_string();
    let mut wait_seconds: u64 = 12;
    let mut allow_system_install = true;
    let mut allow_download = true;
    let mut force = false;
    let mut source_path: Option<PathBuf> = None;
    let mut view_host = "127.0.0.1".to_string();
    let mut view_port: u16 = 8780;
    let mut graph_focus: Option<String> = None;
    let mut graph_path: Option<String> = None;
    let mut graph_limit: usize = 80;
    let mut graph_threshold: f64 = 0.0;
    let mut graph_depth: usize = 1;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "doctor" | "status" | "start" | "stop" | "provision" | "view" | "open"
            | "neighbors" | "lineage" | "text" | "ui" => {
                action = s;
            }
            "--no-system-install" => {
                allow_system_install = false;
            }
            "--no-download" => {
                allow_download = false;
            }
            "--force" => {
                force = true;
            }
            "--from" => {
                i += 1;
                let raw = arg_value(args, i, "--from");
                source_path = Some(normalize_path(&raw));
            }
            "--wait" => {
                i += 1;
                let value = arg_value(args, i, "--wait");
                wait_seconds = value.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                });
            }
            "--host" => {
                i += 1;
                view_host = arg_value(args, i, "--host");
            }
            "--focus" => {
                i += 1;
                let raw = arg_value(args, i, "--focus");
                let trimmed = raw.trim().to_string();
                if !trimmed.is_empty() {
                    graph_focus = Some(trimmed);
                }
            }
            "--path" => {
                i += 1;
                let raw = arg_value(args, i, "--path");
                let trimmed = raw.trim().to_string();
                if !trimmed.is_empty() {
                    graph_path = Some(trimmed);
                }
            }
            "--limit" => {
                i += 1;
                let value = arg_value(args, i, "--limit");
                graph_limit = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--threshold" => {
                i += 1;
                let value = arg_value(args, i, "--threshold");
                graph_threshold = value.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --threshold must be a decimal between 0 and 1");
                    process::exit(2);
                });
            }
            "--depth" => {
                i += 1;
                let value = arg_value(args, i, "--depth");
                graph_depth = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --depth must be an integer (1..3)");
                    process::exit(2);
                });
            }
            "--port" => {
                i += 1;
                let value = arg_value(args, i, "--port");
                view_port = value.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            x if x.starts_with("--wait=") => {
                let value = x.trim_start_matches("--wait=");
                wait_seconds = value.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --wait must be an integer number of seconds");
                    process::exit(2);
                });
            }
            x if x.starts_with("--host=") => {
                view_host = x.trim_start_matches("--host=").to_string();
            }
            x if x.starts_with("--focus=") => {
                let raw = x.trim_start_matches("--focus=").trim().to_string();
                if !raw.is_empty() {
                    graph_focus = Some(raw);
                }
            }
            x if x.starts_with("--path=") => {
                let raw = x.trim_start_matches("--path=").trim().to_string();
                if !raw.is_empty() {
                    graph_path = Some(raw);
                }
            }
            x if x.starts_with("--limit=") => {
                let value = x.trim_start_matches("--limit=");
                graph_limit = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            x if x.starts_with("--threshold=") => {
                let value = x.trim_start_matches("--threshold=");
                graph_threshold = value.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --threshold must be a decimal between 0 and 1");
                    process::exit(2);
                });
            }
            x if x.starts_with("--depth=") => {
                let value = x.trim_start_matches("--depth=");
                graph_depth = value.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --depth must be an integer (1..3)");
                    process::exit(2);
                });
            }
            x if x.starts_with("--port=") => {
                let value = x.trim_start_matches("--port=");
                view_port = value.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            x if x == "--layout"
                || x.starts_with("--layout=")
                || x == "--renderer"
                || x.starts_with("--renderer=")
                || x == "--ui-backend"
                || x.starts_with("--ui-backend=") =>
            {
                eprintln!(
                    "error: terminal graph renderers were removed; use `retrivio graph open` for visual graph and `retrivio graph neighbors|lineage` for terminal output"
                );
                process::exit(2);
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("error: unknown graph action/option '{}'", other);
                    process::exit(2);
                }
                if graph_path.is_none() {
                    graph_path = Some(other.to_string());
                } else {
                    eprintln!("error: unexpected extra argument '{}'", other);
                    process::exit(2);
                }
            }
        }
        i += 1;
    }

    match action.as_str() {
        "status" | "doctor" => {
            let lance_path = data_dir(Path::new("")).join("lance");
            println!("retrieval backend: lancedb (embedded)");
            println!("lancedb path: {}", lance_path.display());
            if lance_path.exists() {
                println!("lancedb ready: yes");
            } else {
                println!("lancedb ready: no (run `retrivio refresh` to create)");
            }
        }
        "start" | "stop" | "provision" => {
            println!("no longer needed (LanceDB is embedded — no external server process)");
        }
        "view" => {
            eprintln!("warning: `retrivio graph view` is deprecated; use `retrivio ui`");
            serve_graph_viewer(&view_host, view_port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "open" => {
            eprintln!("warning: `retrivio graph open` is deprecated; use `retrivio ui`");
            run_graph_open_cmd(&view_host, view_port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "neighbors" => {
            run_graph_neighbors_cmd(
                graph_path.as_deref().or(graph_focus.as_deref()),
                graph_limit,
                graph_threshold,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "lineage" => {
            run_graph_lineage_cmd(
                graph_path.as_deref().or(graph_focus.as_deref()),
                graph_limit,
                graph_threshold,
                graph_depth,
            )
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        "text" => {
            eprintln!(
                "error: `retrivio graph {}` is removed; use `retrivio ui` (browser UI) or `retrivio graph neighbors|lineage` (terminal tables)",
                action
            );
            process::exit(2);
        }
        "ui" => {
            eprintln!("warning: `retrivio graph ui` is deprecated; use top-level `retrivio ui`");
            run_graph_open_cmd(&view_host, view_port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        _ => unreachable!(),
    }
}

fn display_path_compact(path: &str) -> String {
    if let Ok(home) = env::var("HOME") {
        if path == home {
            return "~".to_string();
        }
        let prefix = format!("{}/", home);
        if let Some(rest) = path.strip_prefix(&prefix) {
            return format!("~/{}", rest);
        }
    }
    path.to_string()
}

fn open_url_in_default_browser(url: &str) -> Result<(), String> {
    if cfg!(target_os = "macos") && command_exists("open") {
        let status = Command::new("open")
            .arg(url)
            .status()
            .map_err(|e| format!("failed launching browser via `open`: {}", e))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("`open` exited with status {}", status));
    }
    if command_exists("xdg-open") {
        let status = Command::new("xdg-open")
            .arg(url)
            .status()
            .map_err(|e| format!("failed launching browser via `xdg-open`: {}", e))?;
        if status.success() {
            return Ok(());
        }
        return Err(format!("`xdg-open` exited with status {}", status));
    }
    Err("no supported browser opener found (tried `open` and `xdg-open`)".to_string())
}

fn graph_viewer_has_nodes(host: &str, port: u16) -> Option<bool> {
    let url = format!("http://{}:{}/graph/view/data?limit=1", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    let body = resp.into_string().ok()?;
    let payload: Value = serde_json::from_str(&body).ok()?;
    let nodes = payload
        .get("nodes")
        .and_then(Value::as_array)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    Some(nodes)
}

fn graph_viewer_state(host: &str, port: u16) -> Option<Value> {
    let url = format!("http://{}:{}/graph/view/state", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    resp.into_json::<Value>().ok()
}

fn graph_view_state_json(conn: &Connection, cwd: &Path) -> Result<Value, String> {
    let mut roots: Vec<String> = list_tracked_roots_conn(conn)?
        .into_iter()
        .map(|p| normalize_path(&p.to_string_lossy()).to_string_lossy().to_string())
        .collect();
    roots.sort();
    roots.dedup();

    let mut hasher = Sha1::new();
    for root in &roots {
        hasher.update(root.as_bytes());
        hasher.update(b"\n");
    }
    let roots_hash = format!("{:x}", hasher.finalize());

    let projects_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
        .unwrap_or(0);

    Ok(serde_json::json!({
        "db_path": db_path(cwd).to_string_lossy().to_string(),
        "tracked_roots_count": roots.len(),
        "tracked_roots_hash": roots_hash,
        "projects_count": projects_count
    }))
}

fn local_graph_state(cwd: &Path) -> Option<Value> {
    let dbp = db_path(cwd);
    ensure_db_schema(&dbp).ok()?;
    let conn = open_db_rw(&dbp).ok()?;
    graph_view_state_json(&conn, cwd).ok()
}

fn graph_state_matches(remote: &Value, local: &Value) -> bool {
    remote.get("db_path") == local.get("db_path")
        && remote.get("tracked_roots_hash") == local.get("tracked_roots_hash")
}

fn graph_viewer_is_retrivio(host: &str, port: u16) -> Option<bool> {
    let url = format!("http://{}:{}/", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    let body = resp.into_string().ok()?;
    Some(body.contains("Retrivio Graph Viewer"))
}

fn local_graph_has_nodes(cwd: &Path) -> bool {
    let dbp = db_path(cwd);
    let Ok(conn) = open_db_read_only(&dbp) else {
        return false;
    };
    conn.query_row("SELECT COUNT(*) FROM project_edges", [], |row| {
        row.get::<_, i64>(0)
    })
    .map(|count| count > 0)
    .unwrap_or(false)
}

fn stop_retrivio_graph_viewer_on_port(port: u16) -> bool {
    let Some(pid) = listener_pid_for_port(port) else {
        return false;
    };
    stop_pid_graceful(pid)
}

fn try_reuse_saved_graph_viewer(
    host: &str,
    requested_port: u16,
    cwd: &Path,
    local_state: Option<&Value>,
) -> Option<u16> {
    let Some(saved) = parse_graph_runtime_state(cwd) else {
        return None;
    };
    if saved.host != host {
        if pid_is_alive(saved.pid) {
            let _ = stop_pid_graceful(saved.pid);
        }
        clear_graph_runtime_state(cwd);
        return None;
    }
    if !pid_is_alive(saved.pid) {
        clear_graph_runtime_state(cwd);
        return None;
    }
    if saved.port == requested_port {
        return None;
    }

    let healthy = api_health_host_port(host, saved.port);
    let retrivio = matches!(graph_viewer_is_retrivio(host, saved.port), Some(true));
    if healthy && retrivio {
        if let Some(local) = local_state {
            if let Some(remote) = graph_viewer_state(host, saved.port) {
                if graph_state_matches(&remote, local) {
                    return Some(saved.port);
                }
            }
        } else {
            // No local state to compare; reusing a healthy saved viewer avoids port fan-out.
            return Some(saved.port);
        }
    }

    let _ = stop_pid_graceful(saved.pid);
    clear_graph_runtime_state(cwd);
    None
}

fn ensure_graph_viewer_running(
    host: &str,
    port: u16,
    prefer_fresh_when_existing_empty: bool,
    cwd: &Path,
) -> Result<u16, String> {
    let local_state = local_graph_state(cwd);
    if let Some(reuse_port) = try_reuse_saved_graph_viewer(host, port, cwd, local_state.as_ref()) {
        return Ok(reuse_port);
    }

    let mut try_restart_existing = false;
    if api_health_host_port(host, port) {
        if !matches!(graph_viewer_is_retrivio(host, port), Some(true)) {
            eprintln!(
                "graph open: existing server on port {} is not Retrivio viewer; launching fresh viewer on a new port",
                port
            );
        } else if prefer_fresh_when_existing_empty
            && matches!(graph_viewer_has_nodes(host, port), Some(false))
        {
            eprintln!(
                "graph open: existing viewer on port {} has no graph nodes; restarting it on the same port",
                port
            );
            try_restart_existing = true;
        } else {
            let remote_state = graph_viewer_state(host, port);
            match (remote_state, local_state.as_ref()) {
                (Some(remote), Some(local)) if graph_state_matches(&remote, local) => {
                    if let Some(pid) = listener_pid_for_port(port) {
                        let _ = persist_graph_runtime_state(cwd, pid, host, port);
                    }
                    return Ok(port);
                }
                (Some(_), Some(_)) => {
                    eprintln!(
                        "graph open: existing viewer on port {} uses a different state; restarting it on the same port",
                        port
                    );
                    try_restart_existing = true;
                }
                _ => {
                    eprintln!(
                        "graph open: unable to verify existing viewer state on port {}; restarting it on the same port",
                        port
                    );
                    try_restart_existing = true;
                }
            }
        }
    }
    if try_restart_existing {
        if stop_retrivio_graph_viewer_on_port(port) {
            clear_graph_runtime_state(cwd);
            thread::sleep(Duration::from_millis(150));
        } else {
            eprintln!(
                "graph open: unable to stop existing viewer on port {}; launching fresh viewer on a new port",
                port
            );
        }
    }
    let mut use_port = port;
    if find_free_port(host, use_port, 1).is_none() {
        let Some(free) = find_free_port(host, use_port.saturating_add(1), 200) else {
            return Err(format!(
                "port {} is busy and no free fallback port was found",
                use_port
            ));
        };
        eprintln!(
            "graph open: port {} busy; launching viewer on {}",
            use_port, free
        );
        use_port = free;
    }

    let exe =
        env::current_exe().map_err(|e| format!("failed resolving current executable: {}", e))?;
    let mut child = Command::new(exe)
        .arg("graph")
        .arg("view")
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(use_port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed launching graph viewer in background: {}", e))?;

    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if api_health_host_port(host, use_port) {
            let _ = persist_graph_runtime_state(cwd, child.id(), host, use_port);
            return Ok(use_port);
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("failed checking graph viewer process: {}", e))?
        {
            clear_graph_runtime_state(cwd);
            return Err(format!("graph viewer exited early with status {}", status));
        }
        thread::sleep(Duration::from_millis(120));
    }
    clear_graph_runtime_state(cwd);
    Err(format!(
        "graph viewer did not become healthy at http://{}:{}/health within timeout",
        host, use_port
    ))
}

fn resolve_graph_target_project(
    conn: &Connection,
    raw_path: Option<&str>,
) -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut candidate = raw_path.map(normalize_path).unwrap_or_else(|| cwd.clone());
    if candidate.is_file() {
        if let Some(parent) = candidate.parent() {
            candidate = parent.to_path_buf();
        }
    }
    let normalized = normalize_path(candidate.to_string_lossy().as_ref())
        .to_string_lossy()
        .to_string();
    if let Some(project_path) = resolve_project_for_target(conn, &normalized)? {
        return Ok(project_path);
    }
    let needle = path_basename(&normalized);
    let suggestions = suggest_project_paths(conn, &needle, 5)?;
    let mut msg = format!(
        "no indexed project found for '{}'; run `retrivio index` or pass a tracked project path",
        display_path_compact(&normalized)
    );
    if !suggestions.is_empty() {
        msg.push_str("\nclosest indexed projects:");
        for s in suggestions {
            msg.push_str(&format!("\n- {}", display_path_compact(&s)));
        }
    }
    Err(msg)
}

fn run_graph_open_cmd(host: &str, port: u16) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bound_port = ensure_graph_viewer_running(host, port, local_graph_has_nodes(&cwd), &cwd)?;
    let url = format!("http://{}:{}/", host, bound_port);
    println!("graph viewer: {}", url);
    if let Err(err) = open_url_in_default_browser(&url) {
        eprintln!("warning: {}", err);
        eprintln!("hint: open this URL manually in your browser");
    }
    Ok(())
}

fn run_graph_neighbors_cmd(
    raw_path: Option<&str>,
    limit: usize,
    min_weight: f64,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_read_only(&dbp)?;
    let project_path = resolve_graph_target_project(&conn, raw_path)?;

    let limit = limit.max(1).min(200);
    let min_weight = min_weight.clamp(0.0, 1.0);
    let rows = list_neighbors_by_path(&conn, &project_path, limit * 3)?;
    let mut filtered: Vec<(String, String, f64)> = rows
        .into_iter()
        .filter(|(_, _, w)| *w >= min_weight)
        .take(limit)
        .collect();
    filtered.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    println!("project: {}", display_path_compact(&project_path));
    println!(
        "neighbors: {} (threshold >= {:.2})",
        filtered.len(),
        min_weight
    );
    if filtered.is_empty() {
        println!("(no outgoing neighbors matched current threshold)");
        return Ok(());
    }
    println!(
        "{:<4} {:>7}  {:<18} {}",
        "rank", "weight", "relation", "target"
    );
    for (idx, (dst, kind, weight)) in filtered.iter().enumerate() {
        println!(
            "{:<4} {:>7.3}  {:<18} {}",
            idx + 1,
            weight,
            clipped(kind, 18),
            display_path_compact(dst)
        );
    }
    Ok(())
}

fn run_graph_lineage_cmd(
    raw_path: Option<&str>,
    limit: usize,
    min_weight: f64,
    depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_read_only(&dbp)?;
    let project_path = resolve_graph_target_project(&conn, raw_path)?;

    let limit = limit.max(1).min(200);
    let min_weight = min_weight.clamp(0.0, 1.0);
    let depth = depth.clamp(1, 3);
    let data = load_graph_text_data(&cwd, Some(&project_path), (limit * 6).max(40))?;
    let subgraph =
        build_focus_subgraph(&data, &project_path, (limit * 3).max(24), min_weight, depth);

    let mut incoming: Vec<(String, String, f64)> = Vec::new();
    let mut outgoing: Vec<(String, String, f64)> = Vec::new();
    let mut relays: Vec<(String, String, String, f64)> = Vec::new();
    for edge in &subgraph.edges {
        if edge.weight < min_weight {
            continue;
        }
        if edge.source == project_path {
            outgoing.push((edge.target.clone(), edge.kind.clone(), edge.weight));
        } else if edge.target == project_path {
            incoming.push((edge.source.clone(), edge.kind.clone(), edge.weight));
        } else {
            relays.push((
                edge.source.clone(),
                edge.target.clone(),
                edge.kind.clone(),
                edge.weight,
            ));
        }
    }
    incoming.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    outgoing.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    relays.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(&b.0)));

    println!("focus: {}", display_path_compact(&project_path));
    println!(
        "lineage: depth={} threshold>={:.2} nodes={} edges={}",
        depth,
        min_weight,
        subgraph.nodes.len(),
        subgraph.edges.len()
    );

    println!();
    println!("direct edges:");
    println!(
        "{:<4} {:<4} {:>7}  {:<18} {}",
        "rank", "dir", "weight", "relation", "path"
    );
    let mut rank = 1usize;
    for (src, kind, weight) in incoming.iter().take(limit) {
        println!(
            "{:<4} {:<4} {:>7.3}  {:<18} {}",
            rank,
            "in",
            weight,
            clipped(kind, 18),
            display_path_compact(src)
        );
        rank += 1;
    }
    for (dst, kind, weight) in outgoing.iter().take(limit) {
        println!(
            "{:<4} {:<4} {:>7.3}  {:<18} {}",
            rank,
            "out",
            weight,
            clipped(kind, 18),
            display_path_compact(dst)
        );
        rank += 1;
    }
    if rank == 1 {
        println!("(no direct lineage edges matched current threshold)");
    }

    if depth > 1 {
        println!();
        println!("neighbor relay edges:");
        println!(
            "{:<4} {:>7}  {:<18} {} -> {}",
            "rank", "weight", "relation", "source", "target"
        );
        let mut relay_rank = 1usize;
        for (src, dst, kind, weight) in relays.iter().take(limit) {
            println!(
                "{:<4} {:>7.3}  {:<18} {} -> {}",
                relay_rank,
                weight,
                clipped(kind, 18),
                clipped(&display_path_compact(src), 42),
                clipped(&display_path_compact(dst), 42)
            );
            relay_rank += 1;
        }
        if relay_rank == 1 {
            println!("(no relay edges at current threshold/depth)");
        }
    }
    Ok(())
}

fn resolve_project_for_target(conn: &Connection, target: &str) -> Result<Option<String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path
FROM projects
WHERE path = ?1 OR ?1 LIKE path || '/%'
ORDER BY length(path) DESC
LIMIT 1
"#,
        )
        .map_err(|e| format!("failed preparing project resolution query: {}", e))?;
    let found: Option<String> = stmt
        .query_row(params![target], |row| row.get(0))
        .optional()
        .map_err(|e| format!("failed resolving target project: {}", e))?;
    Ok(found)
}

fn suggest_project_paths(
    conn: &Connection,
    needle: &str,
    limit: usize,
) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path
FROM projects
WHERE lower(path) LIKE lower(?1)
ORDER BY path ASC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project suggestion query: {}", e))?;
    let pattern = if needle.trim().is_empty() {
        "%".to_string()
    } else {
        format!("%{}%", needle.trim())
    };
    let rows = stmt
        .query_map(params![pattern, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|e| format!("failed querying project suggestions: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project suggestion row: {}", e))?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct GraphTextEdge {
    source: String,
    target: String,
    kind: String,
    weight: f64,
}

#[derive(Clone, Debug)]
struct GraphTextData {
    nodes: Vec<String>,
    edges: Vec<GraphTextEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphTextLayout {
    Cluster,
    Lineage,
    Matrix,
}

impl GraphTextLayout {
    fn name(self) -> &'static str {
        match self {
            GraphTextLayout::Cluster => "cluster",
            GraphTextLayout::Lineage => "lineage",
            GraphTextLayout::Matrix => "matrix",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphTextLayout::Cluster => GraphTextLayout::Lineage,
            GraphTextLayout::Lineage => GraphTextLayout::Matrix,
            GraphTextLayout::Matrix => GraphTextLayout::Cluster,
        }
    }
}

fn parse_graph_text_layout(raw: &str) -> Option<GraphTextLayout> {
    match raw.trim().to_lowercase().as_str() {
        "cluster" => Some(GraphTextLayout::Cluster),
        "lineage" => Some(GraphTextLayout::Lineage),
        "matrix" => Some(GraphTextLayout::Matrix),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphTextRenderer {
    Auto,
    Graphviz,
    Ascii,
}

impl GraphTextRenderer {
    fn name(self) -> &'static str {
        match self {
            GraphTextRenderer::Auto => "auto",
            GraphTextRenderer::Graphviz => "graphviz",
            GraphTextRenderer::Ascii => "ascii",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphTextRenderer::Auto => GraphTextRenderer::Graphviz,
            GraphTextRenderer::Graphviz => GraphTextRenderer::Ascii,
            GraphTextRenderer::Ascii => GraphTextRenderer::Auto,
        }
    }
}

fn parse_graph_text_renderer(raw: &str) -> Option<GraphTextRenderer> {
    match raw.trim().to_lowercase().as_str() {
        "auto" => Some(GraphTextRenderer::Auto),
        "graphviz" => Some(GraphTextRenderer::Graphviz),
        "ascii" => Some(GraphTextRenderer::Ascii),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphUiBackend {
    Auto,
    Vt,
    Kitty,
    Iterm,
    Text,
}

impl GraphUiBackend {
    fn name(self) -> &'static str {
        match self {
            GraphUiBackend::Auto => "auto",
            GraphUiBackend::Vt => "vt",
            GraphUiBackend::Kitty => "kitty",
            GraphUiBackend::Iterm => "iterm",
            GraphUiBackend::Text => "text",
        }
    }

    fn toggle(self) -> Self {
        match self {
            GraphUiBackend::Auto => GraphUiBackend::Vt,
            GraphUiBackend::Vt => GraphUiBackend::Kitty,
            GraphUiBackend::Kitty => GraphUiBackend::Iterm,
            GraphUiBackend::Iterm => GraphUiBackend::Text,
            GraphUiBackend::Text => GraphUiBackend::Auto,
        }
    }
}

fn parse_graph_ui_backend(raw: &str) -> Option<GraphUiBackend> {
    match raw.trim().to_lowercase().as_str() {
        "auto" => Some(GraphUiBackend::Auto),
        "vt" => Some(GraphUiBackend::Vt),
        "kitty" => Some(GraphUiBackend::Kitty),
        "iterm" => Some(GraphUiBackend::Iterm),
        "text" => Some(GraphUiBackend::Text),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphUiBackendResolved {
    Vt,
    Kitty,
    Iterm,
    Text,
}

impl GraphUiBackendResolved {
    fn name(self) -> &'static str {
        match self {
            GraphUiBackendResolved::Vt => "vt-24bit",
            GraphUiBackendResolved::Kitty => "kitty",
            GraphUiBackendResolved::Iterm => "iterm-inline",
            GraphUiBackendResolved::Text => "text-fallback",
        }
    }
}

fn terminal_supports_kitty_graphics() -> bool {
    env::var("KITTY_WINDOW_ID").is_ok()
        || env::var("TERM")
            .map(|v| v.to_lowercase().contains("kitty"))
            .unwrap_or(false)
}

fn terminal_supports_iterm_inline() -> bool {
    env::var("TERM_PROGRAM")
        .map(|v| v == "iTerm.app")
        .unwrap_or(false)
        || env::var("WEZTERM_PANE").is_ok()
}

fn terminal_is_apple_terminal() -> bool {
    env::var("TERM_PROGRAM")
        .map(|v| v == "Apple_Terminal")
        .unwrap_or(false)
}

fn resolve_graph_ui_backend(choice: GraphUiBackend) -> (GraphUiBackendResolved, Option<String>) {
    if !command_exists("dot") {
        return (
            GraphUiBackendResolved::Text,
            Some("dot not found; using text fallback".to_string()),
        );
    }
    match choice {
        GraphUiBackend::Auto => {
            if terminal_supports_kitty_graphics() {
                (GraphUiBackendResolved::Kitty, None)
            } else if terminal_supports_iterm_inline() {
                (GraphUiBackendResolved::Iterm, None)
            } else if terminal_is_apple_terminal() {
                (
                    GraphUiBackendResolved::Text,
                    Some("Apple Terminal detected; defaulting to clean text mode".to_string()),
                )
            } else {
                (GraphUiBackendResolved::Vt, None)
            }
        }
        GraphUiBackend::Vt => (GraphUiBackendResolved::Vt, None),
        GraphUiBackend::Kitty => {
            if terminal_supports_kitty_graphics() {
                (GraphUiBackendResolved::Kitty, None)
            } else {
                (
                    GraphUiBackendResolved::Text,
                    Some("kitty backend unsupported in this terminal; using text".to_string()),
                )
            }
        }
        GraphUiBackend::Iterm => {
            if terminal_supports_iterm_inline() {
                (GraphUiBackendResolved::Iterm, None)
            } else {
                (
                    GraphUiBackendResolved::Text,
                    Some("iterm backend unsupported in this terminal; using text".to_string()),
                )
            }
        }
        GraphUiBackend::Text => (GraphUiBackendResolved::Text, None),
    }
}

#[derive(Clone, Debug)]
struct GraphUiState {
    backend: GraphUiBackend,
    focus_idx: usize,
    limit: usize,
    min_weight: f64,
    depth: usize,
    cluster_cross_edges: bool,
    status: String,
}

fn run_graph_ui_cmd(
    focus_hint: Option<&str>,
    limit: usize,
    initial_backend: Option<GraphUiBackend>,
    initial_min_weight: f64,
    initial_depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut state = GraphUiState {
        backend: initial_backend.unwrap_or(GraphUiBackend::Auto),
        focus_idx: 0,
        limit: limit.max(12).min(360),
        min_weight: initial_min_weight.clamp(0.0, 1.0),
        depth: initial_depth.clamp(1, 3),
        cluster_cross_edges: false,
        status: "graph loaded".to_string(),
    };
    let mut data = load_graph_text_data(&cwd, focus_hint, state.limit)?;
    if data.nodes.is_empty() {
        println!(
            "graph ui: no graph nodes found; run `retrivio add <path>` then `retrivio index` (example: `retrivio add ~/projects && retrivio index`)"
        );
        return Ok(());
    }
    state.focus_idx = graph_text_focus_index(&data.nodes, focus_hint);

    if !tty_ui_available() {
        let (width, mut height) = terminal_size_fallback();
        if height > 40 {
            height = 40;
        }
        let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
        let focus = &data.nodes[focus_idx];
        let subgraph = build_focus_subgraph(
            &data,
            focus,
            state.limit.min(100).max(8),
            state.min_weight,
            state.depth,
        );
        let lines = render_focus_star_cluster(
            focus,
            width.max(80),
            height.saturating_sub(6).max(8),
            &subgraph.nodes,
            &subgraph.edges,
        );
        println!("{}", lines.join("\n"));
        return Ok(());
    }

    let _guard = enter_config_tui_mode()?;
    let mut stdout = std::io::stdout();
    loop {
        draw_graph_ui_tui(&mut stdout, &data, &state)?;
        let Some(key) = read_config_key()? else {
            thread::sleep(Duration::from_millis(120));
            continue;
        };
        match key {
            ConfigKey::Esc | ConfigKey::CtrlC | ConfigKey::Char('q') => return Ok(()),
            ConfigKey::Right | ConfigKey::Down | ConfigKey::Char('j') | ConfigKey::Char('l') => {
                if !data.nodes.is_empty() {
                    state.focus_idx = (state.focus_idx + 1) % data.nodes.len();
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Left | ConfigKey::Up | ConfigKey::Char('k') | ConfigKey::Char('h') => {
                if !data.nodes.is_empty() {
                    if state.focus_idx == 0 {
                        state.focus_idx = data.nodes.len().saturating_sub(1);
                    } else {
                        state.focus_idx -= 1;
                    }
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Char('b') => {
                state.backend = state.backend.toggle();
                state.status = format!("ui backend: {}", state.backend.name());
            }
            ConfigKey::Char('x') => {
                state.cluster_cross_edges = !state.cluster_cross_edges;
                state.status = format!(
                    "cross-edges: {}",
                    if state.cluster_cross_edges {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ConfigKey::Char('w') => {
                state.min_weight = (state.min_weight + 0.05).min(0.95);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char('s') => {
                state.min_weight = (state.min_weight - 0.05).max(0.0);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char(']') => {
                state.depth = (state.depth + 1).min(3);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('[') => {
                state.depth = state.depth.saturating_sub(1).max(1);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('+') | ConfigKey::Char('=') => {
                state.limit = (state.limit + 10).min(360);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('-') => {
                state.limit = state.limit.saturating_sub(10).max(12);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('r') => {
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = "graph reloaded".to_string();
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GraphPoint {
    x: i32,
    y: i32,
}

#[derive(Clone, Debug)]
struct GraphTextState {
    layout: GraphTextLayout,
    renderer: GraphTextRenderer,
    focus_idx: usize,
    limit: usize,
    min_weight: f64,
    depth: usize,
    cluster_cross_edges: bool,
    status: String,
}

fn run_graph_text_cmd(
    focus_hint: Option<&str>,
    limit: usize,
    initial_layout: Option<GraphTextLayout>,
    initial_renderer: Option<GraphTextRenderer>,
    initial_min_weight: f64,
    initial_depth: usize,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut state = GraphTextState {
        layout: initial_layout.unwrap_or(GraphTextLayout::Lineage),
        renderer: initial_renderer.unwrap_or(GraphTextRenderer::Auto),
        focus_idx: 0,
        limit: limit.max(12).min(360),
        min_weight: initial_min_weight.clamp(0.0, 1.0),
        depth: initial_depth.clamp(1, 3),
        cluster_cross_edges: false,
        status: "graph loaded".to_string(),
    };
    let mut data = load_graph_text_data(&cwd, focus_hint, state.limit)?;
    if data.nodes.is_empty() {
        println!("graph text: no graph nodes found; run `retrivio index` first");
        return Ok(());
    }
    state.focus_idx = graph_text_focus_index(&data.nodes, focus_hint);
    if !tty_ui_available() {
        let (width, mut height) = terminal_size_fallback();
        if height > 44 {
            height = 44;
        }
        let lines = render_graph_text_snapshot(&data, &state, width, height);
        println!("{}", lines.join("\n"));
        return Ok(());
    }

    let _guard = match enter_config_tui_mode() {
        Ok(v) => v,
        Err(err) => {
            let (width, mut height) = terminal_size_fallback();
            if height > 44 {
                height = 44;
            }
            let lines = render_graph_text_snapshot(&data, &state, width, height);
            println!("{}", lines.join("\n"));
            eprintln!(
                "graph text: interactive TUI unavailable ({}); rendered static snapshot",
                err
            );
            return Ok(());
        }
    };
    let mut stdout = std::io::stdout();
    loop {
        draw_graph_text_tui(&mut stdout, &data, &state)?;
        let Some(key) = read_config_key()? else {
            continue;
        };
        match key {
            ConfigKey::Esc | ConfigKey::CtrlC | ConfigKey::Char('q') => return Ok(()),
            ConfigKey::Char('t') | ConfigKey::Tab => {
                state.layout = state.layout.toggle();
                state.status = format!("layout: {}", state.layout.name());
            }
            ConfigKey::Char('v') => {
                state.renderer = state.renderer.toggle();
                state.status = format!("renderer: {}", state.renderer.name());
            }
            ConfigKey::Char('x') => {
                state.cluster_cross_edges = !state.cluster_cross_edges;
                state.status = format!(
                    "cluster cross-edges: {}",
                    if state.cluster_cross_edges {
                        "on"
                    } else {
                        "off"
                    }
                );
            }
            ConfigKey::Right | ConfigKey::Down | ConfigKey::Char('j') | ConfigKey::Char('l') => {
                if !data.nodes.is_empty() {
                    state.focus_idx = (state.focus_idx + 1) % data.nodes.len();
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Left | ConfigKey::Up | ConfigKey::Char('k') | ConfigKey::Char('h') => {
                if !data.nodes.is_empty() {
                    if state.focus_idx == 0 {
                        state.focus_idx = data.nodes.len().saturating_sub(1);
                    } else {
                        state.focus_idx -= 1;
                    }
                    state.status =
                        format!("focus: {}", path_basename(&data.nodes[state.focus_idx]));
                }
            }
            ConfigKey::Char('+') | ConfigKey::Char('=') => {
                state.limit = (state.limit + 10).min(360);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('-') => {
                state.limit = state.limit.saturating_sub(10).max(12);
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = format!("node limit: {}", state.limit);
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('r') => {
                let focus = data.nodes.get(state.focus_idx).cloned();
                match load_graph_text_data(&cwd, focus.as_deref(), state.limit) {
                    Ok(next) => {
                        data = next;
                        state.focus_idx = graph_text_focus_index(&data.nodes, focus.as_deref());
                        state.status = "graph reloaded".to_string();
                    }
                    Err(e) => state.status = format!("reload failed: {}", e),
                }
            }
            ConfigKey::Char('w') => {
                state.min_weight = (state.min_weight + 0.05).min(0.95);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char('s') => {
                state.min_weight = (state.min_weight - 0.05).max(0.0);
                state.status = format!("edge threshold: {:.2}", state.min_weight);
            }
            ConfigKey::Char(']') => {
                state.depth = (state.depth + 1).min(3);
                state.status = format!("hop depth: {}", state.depth);
            }
            ConfigKey::Char('[') => {
                state.depth = state.depth.saturating_sub(1).max(1);
                state.status = format!("hop depth: {}", state.depth);
            }
            _ => {}
        }
    }
}

fn graph_text_focus_index(nodes: &[String], focus_hint: Option<&str>) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    if let Some(raw) = focus_hint {
        let target = normalize_path(raw).to_string_lossy().to_string();
        if let Some(idx) = nodes.iter().position(|n| *n == target) {
            return idx;
        }
    }
    0
}

fn load_graph_text_data(
    cwd: &Path,
    focus_hint: Option<&str>,
    limit: usize,
) -> Result<GraphTextData, String> {
    let dbp = db_path(cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_rw(&dbp)?;
    let limit = limit.max(12).min(600);
    let mut payload = graph_view_data_json(&conn, focus_hint, limit)?;

    let mut nodes: Vec<String> = Vec::new();
    if let Some(items) = payload.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        let mut seen: HashSet<String> = HashSet::new();
        for item in items.iter() {
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("id").and_then(|v| v.as_str()))
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                continue;
            }
            let normalized = normalize_path(&path).to_string_lossy().to_string();
            if seen.insert(normalized.clone()) {
                nodes.push(normalized);
            }
        }
    }

    let mut edges: Vec<GraphTextEdge> = Vec::new();
    if let Some(items) = payload.get("edges").and_then(|v| v.as_array()) {
        for item in items {
            let source_raw = item
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim();
            let target_raw = item
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim();
            if source_raw.is_empty() || target_raw.is_empty() {
                continue;
            }
            let source = normalize_path(source_raw).to_string_lossy().to_string();
            let target = normalize_path(target_raw).to_string_lossy().to_string();
            if source == target {
                continue;
            }
            let kind = item
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("related")
                .trim()
                .to_string();
            let weight = item.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.0);
            edges.push(GraphTextEdge {
                source,
                target,
                kind,
                weight,
            });
        }
    }

    if nodes.is_empty() && focus_hint.is_some() {
        return load_graph_text_data(cwd, None, limit);
    }
    Ok(GraphTextData { nodes, edges })
}

fn draw_graph_text_tui(
    stdout: &mut std::io::Stdout,
    data: &GraphTextData,
    state: &GraphTextState,
) -> Result<(), String> {
    let (width, height) = terminal_size_stty();
    let lines = render_graph_text_snapshot(data, state, width, height);
    let payload = format!("\x1b[H\x1b[2J{}", lines.join("\r\n"));
    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing graph text frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing graph text frame: {}", e))
}

fn draw_graph_ui_tui(
    stdout: &mut std::io::Stdout,
    data: &GraphTextData,
    state: &GraphUiState,
) -> Result<(), String> {
    if data.nodes.is_empty() {
        return Err("graph ui: no graph nodes".to_string());
    }
    let (width, height) = terminal_size_stty();
    let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
    let focus = &data.nodes[focus_idx];
    let subgraph = build_focus_subgraph(
        data,
        focus,
        state.limit.min(140).max(8),
        state.min_weight,
        state.depth,
    );
    let (backend, backend_note) = resolve_graph_ui_backend(state.backend);
    let header_rows = 5usize;
    let mut canvas_h = height.saturating_sub(header_rows).max(8);
    if backend == GraphUiBackendResolved::Text {
        canvas_h = canvas_h.min(18);
    }
    let canvas = render_graph_ui_canvas(
        focus,
        &subgraph,
        width.max(80),
        canvas_h,
        backend,
        state.cluster_cross_edges,
    );

    let mut payload = String::new();
    payload.push_str("\x1b[H\x1b[2J");
    payload.push_str(&clipped(
        &format!(
            "\x1b[36;1mretrivio graph[ui]\x1b[0m nodes={} edges={} visible_nodes={} visible_edges={} backend={} w>={:.2} depth={} cross={}",
            data.nodes.len(),
            data.edges.len(),
            subgraph.nodes.len(),
            subgraph.edges.len(),
            backend.name(),
            state.min_weight,
            state.depth,
            if state.cluster_cross_edges { "on" } else { "off" }
        ),
        width,
    ));
    payload.push_str("\r\n");
    payload.push_str(&clipped(
        "keys: b backend  x cross-edges  ←/→ or j/k focus  w/s threshold  [/ ] depth  r reload  +/- scope  q exit",
        width,
    ));
    payload.push_str("\r\n");
    payload.push_str(&clipped(&format!("focus: {}", focus), width));
    payload.push_str("\r\n");
    let mut status = state.status.clone();
    if let Some(note) = backend_note {
        if !status.is_empty() {
            status.push_str(" | ");
        }
        status.push_str(&note);
    }
    payload.push_str(&clipped(&format!("status: {}", status), width));
    payload.push_str("\r\n");
    payload.push_str(&clipped(
        "hint: use iTerm2/Kitty/WezTerm for richer pixel rendering; Terminal.app is limited",
        width,
    ));
    payload.push_str("\r\n");
    payload.push_str(&canvas);
    if !payload.ends_with('\n') {
        payload.push_str("\r\n");
    }

    stdout
        .write_all(payload.as_bytes())
        .map_err(|e| format!("failed writing graph ui frame: {}", e))?;
    stdout
        .flush()
        .map_err(|e| format!("failed flushing graph ui frame: {}", e))
}

fn render_graph_ui_canvas(
    focus: &str,
    subgraph: &GraphTextSubgraph,
    width: usize,
    height: usize,
    backend: GraphUiBackendResolved,
    include_cross_edges: bool,
) -> String {
    match backend {
        GraphUiBackendResolved::Text => {
            render_focus_star_cluster(focus, width, height, &subgraph.nodes, &subgraph.edges)
                .join("\n")
        }
        GraphUiBackendResolved::Vt => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            match run_dot_render(&dot_src, "vt-24bit", "neato") {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: vt render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\r\n")
                }
            }
        }
        GraphUiBackendResolved::Kitty => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            let render = run_dot_render(&dot_src, "kittyz", "neato")
                .or_else(|_| run_dot_render(&dot_src, "kitty", "neato"));
            match render {
                Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: kitty render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\r\n")
                }
            }
        }
        GraphUiBackendResolved::Iterm => {
            let dot_src = graph_ui_dot_source(focus, subgraph, width, height, include_cross_edges);
            match run_dot_render(&dot_src, "png", "neato") {
                Ok(png) => {
                    let encoded = BASE64_STANDARD.encode(png);
                    format!(
                        "\x1b]1337;File=inline=1;width=100%;preserveAspectRatio=1:{}\x07",
                        encoded
                    )
                }
                Err(err) => {
                    let mut lines = render_focus_star_cluster(
                        focus,
                        width,
                        height,
                        &subgraph.nodes,
                        &subgraph.edges,
                    );
                    lines.push(format!(
                        "graph ui: iterm inline render failed: {}",
                        clipped(&err, 120)
                    ));
                    lines.join("\r\n")
                }
            }
        }
    }
}

fn run_dot_render(dot_src: &str, format: &str, engine: &str) -> Result<Vec<u8>, String> {
    let mut child = Command::new("dot")
        .arg(format!("-K{}", engine))
        .arg(format!("-T{}", format))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning dot: {}", e))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(dot_src.as_bytes())
            .map_err(|e| format!("failed writing dot input: {}", e))?;
    } else {
        return Err("dot stdin unavailable".to_string());
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed reading dot output: {}", e))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(format!(
            "dot render failed ({}): {}",
            out.status,
            clipped(&String::from_utf8_lossy(&out.stderr), 200)
        ))
    }
}

fn graph_ui_dot_source(
    focus: &str,
    subgraph: &GraphTextSubgraph,
    width: usize,
    height: usize,
    include_cross_edges: bool,
) -> String {
    let (nodes, edges) =
        cluster_draw_data(focus, &subgraph.nodes, &subgraph.edges, include_cross_edges);
    let w_in = ((width as f64) / 10.0).clamp(8.0, 26.0);
    let h_in = ((height as f64) / 4.0).clamp(4.0, 16.0);
    let mut node_ids: HashMap<String, String> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        node_ids.insert(node.clone(), format!("n{}", idx));
    }
    let mut dot = format!(
        "digraph G {{\n\
         graph [bgcolor=\"#0b0f14\",pad=0.20,overlap=false,splines=true,outputorder=edgesfirst,size=\"{:.2},{:.2}!\",ratio=fill];\n\
         node [shape=circle,style=filled,fontname=\"Menlo\",fontsize=11,fontcolor=\"#e5e7eb\",color=\"#93c5fd\",fillcolor=\"#1f2937\",penwidth=1.4,width=0.84,height=0.84,fixedsize=true];\n\
         edge [color=\"#6b7280\",penwidth=1.2,arrowsize=0.55];\n",
        w_in, h_in
    );
    for node in &nodes {
        let id = node_ids
            .get(node)
            .cloned()
            .unwrap_or_else(|| "n0".to_string());
        let label = short_graph_label(node, 18);
        if node == focus {
            dot.push_str(&format!(
                "{} [label=\"{}\",fillcolor=\"#0f172a\",color=\"#f59e0b\",penwidth=2.6,width=1.02,height=1.02];\n",
                id,
                dot_escape(&label)
            ));
        } else {
            dot.push_str(&format!("{} [label=\"{}\"];\n", id, dot_escape(&label)));
        }
    }
    for edge in &edges {
        let Some(src) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst) = node_ids.get(&edge.target) else {
            continue;
        };
        let focus_edge = edge.source == focus || edge.target == focus;
        if focus_edge {
            dot.push_str(&format!(
                "{} -> {} [color=\"#f59e0b\",penwidth=2.1];\n",
                src, dst
            ));
        } else {
            dot.push_str(&format!("{} -> {};\n", src, dst));
        }
    }
    dot.push_str("}\n");
    dot
}

fn render_graph_text_snapshot(
    data: &GraphTextData,
    state: &GraphTextState,
    width: usize,
    height: usize,
) -> Vec<String> {
    if data.nodes.is_empty() {
        return vec!["graph text: no graph nodes".to_string()];
    }
    let focus_idx = state.focus_idx.min(data.nodes.len().saturating_sub(1));
    let focus = &data.nodes[focus_idx];
    let header_rows = 4usize;
    let footer_rows = if state.layout == GraphTextLayout::Cluster {
        12usize
    } else {
        6usize
    };
    let mut canvas_h = height.saturating_sub(header_rows + footer_rows).max(8);
    if state.layout == GraphTextLayout::Lineage {
        canvas_h = canvas_h.min(18);
    } else if state.layout == GraphTextLayout::Matrix {
        canvas_h = canvas_h.min(20);
    } else if state.layout == GraphTextLayout::Cluster {
        canvas_h = canvas_h.min(18);
    }
    let canvas_w = width.max(48);

    let layout_cap = match state.layout {
        GraphTextLayout::Matrix => 30,
        GraphTextLayout::Cluster => 18,
        GraphTextLayout::Lineage => 16,
    };
    let visible_limit = state.limit.min(layout_cap).max(8);
    let subgraph = build_focus_subgraph(data, focus, visible_limit, state.min_weight, state.depth);
    let mut renderer_used = "n/a".to_string();
    let mut renderer_note: Option<String> = None;
    let mut cluster_legend_nodes: Option<Vec<String>> = None;
    let canvas_lines = match state.layout {
        GraphTextLayout::Lineage => {
            render_lineage_diagram(focus, &subgraph.edges, canvas_w, canvas_h)
        }
        GraphTextLayout::Matrix => {
            render_graph_matrix(focus, &subgraph.nodes, &subgraph.edges, canvas_w, canvas_h)
        }
        GraphTextLayout::Cluster => {
            let (legend_nodes, _) = cluster_draw_data(
                focus,
                &subgraph.nodes,
                &subgraph.edges,
                state.cluster_cross_edges,
            );
            cluster_legend_nodes = Some(legend_nodes);
            let (lines, used, note) = render_graph_cluster(
                focus,
                canvas_w,
                canvas_h,
                &subgraph.nodes,
                &subgraph.edges,
                state.renderer,
                state.cluster_cross_edges,
            );
            renderer_used = used;
            renderer_note = note;
            lines
        }
    };
    let neighbor_limit = if state.layout == GraphTextLayout::Cluster {
        3
    } else {
        4
    };
    let neighbor_lines = focus_edge_summary_lines(focus, &subgraph.edges, neighbor_limit, width);
    let mut out: Vec<String> = Vec::new();
    out.push(format!(
        "\x1b[36;1mretrivio graph[text:{}]\x1b[0m nodes={} edges={} visible_nodes={} visible_edges={} limit={} w>={:.2} depth={} renderer={} cross={}",
        state.layout.name(),
        data.nodes.len(),
        data.edges.len(),
        subgraph.nodes.len(),
        subgraph.edges.len(),
        state.limit
        ,
        state.min_weight,
        state.depth,
        renderer_used,
        if state.cluster_cross_edges {
            "on"
        } else {
            "off"
        }
    ));
    out.push(clipped(
        "keys: t/Tab layout  v renderer  x cross-edges  \u{2190}/\u{2192} or j/k focus  w/s threshold  [/ ] depth  r reload  +/- limit  q exit",
        width,
    ));
    out.push(clipped(&format!("focus: {}", focus), width));
    let mut status_line = state.status.clone();
    if let Some(note) = renderer_note {
        if !status_line.is_empty() {
            status_line.push_str(" | ");
        }
        status_line.push_str(&note);
    }
    out.push(clipped(&format!("status: {}", status_line), width));
    out.extend(canvas_lines);
    for line in neighbor_lines {
        out.push(clipped(&line, width));
    }
    if state.layout == GraphTextLayout::Cluster {
        let legend_nodes = cluster_legend_nodes.as_deref().unwrap_or(&subgraph.nodes);
        let legend_lines = cluster_node_legend_lines(focus, legend_nodes, width, 8);
        for line in legend_lines {
            out.push(clipped(&line, width));
        }
    }
    while out.len() < height {
        out.push(String::new());
    }
    out.truncate(height);
    out
}

#[derive(Clone, Debug)]
struct GraphTextSubgraph {
    nodes: Vec<String>,
    edges: Vec<GraphTextEdge>,
}

fn build_focus_subgraph(
    data: &GraphTextData,
    focus: &str,
    max_nodes: usize,
    min_weight: f64,
    depth: usize,
) -> GraphTextSubgraph {
    let max_nodes = max_nodes.max(6);
    let depth = depth.clamp(1, 3);
    let min_weight = min_weight.clamp(0.0, 1.0);

    let mut eligible_edges: Vec<GraphTextEdge> = data
        .edges
        .iter()
        .filter(|e| e.weight >= min_weight)
        .cloned()
        .collect();
    if eligible_edges.is_empty() {
        let mut fallback: Vec<GraphTextEdge> = data
            .edges
            .iter()
            .filter(|e| e.source == focus || e.target == focus)
            .cloned()
            .collect();
        fallback.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        eligible_edges = fallback.into_iter().take(20).collect();
    }
    eligible_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    let mut selected: HashSet<String> = HashSet::new();
    selected.insert(focus.to_string());
    let mut frontier: HashSet<String> = HashSet::new();
    frontier.insert(focus.to_string());
    for _ in 0..depth {
        if selected.len() >= max_nodes {
            break;
        }
        let mut candidates: Vec<(f64, String)> = Vec::new();
        for edge in &eligible_edges {
            if frontier.contains(&edge.source) && !selected.contains(&edge.target) {
                candidates.push((edge.weight, edge.target.clone()));
            }
            if frontier.contains(&edge.target) && !selected.contains(&edge.source) {
                candidates.push((edge.weight * 0.97, edge.source.clone()));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut next_frontier: HashSet<String> = HashSet::new();
        for (_, node) in candidates {
            if selected.len() >= max_nodes {
                break;
            }
            if selected.insert(node.clone()) {
                next_frontier.insert(node);
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    if selected.len() < max_nodes {
        for edge in &eligible_edges {
            if selected.contains(&edge.source) && selected.insert(edge.target.clone()) {
                if selected.len() >= max_nodes {
                    break;
                }
            }
            if selected.contains(&edge.target) && selected.insert(edge.source.clone()) {
                if selected.len() >= max_nodes {
                    break;
                }
            }
        }
    }

    let mut sub_edges: Vec<GraphTextEdge> = eligible_edges
        .iter()
        .filter(|e| selected.contains(&e.source) && selected.contains(&e.target))
        .cloned()
        .collect();
    sub_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let edge_cap = (max_nodes * 3).max(16);
    if sub_edges.len() > edge_cap {
        sub_edges.truncate(edge_cap);
    }

    let mut score_by_node: HashMap<String, f64> = HashMap::new();
    score_by_node.insert(focus.to_string(), 9999.0);
    for edge in &sub_edges {
        let focus_boost = if edge.source == focus || edge.target == focus {
            1.25
        } else {
            1.0
        };
        *score_by_node.entry(edge.source.clone()).or_insert(0.0) += edge.weight * focus_boost;
        *score_by_node.entry(edge.target.clone()).or_insert(0.0) += edge.weight * focus_boost;
    }

    let mut nodes: Vec<String> = selected.into_iter().collect();
    nodes.sort_by(|a, b| {
        if a == focus {
            return std::cmp::Ordering::Less;
        }
        if b == focus {
            return std::cmp::Ordering::Greater;
        }
        let sa = *score_by_node.get(a).unwrap_or(&0.0);
        let sb = *score_by_node.get(b).unwrap_or(&0.0);
        sb.total_cmp(&sa).then_with(|| a.cmp(b))
    });
    GraphTextSubgraph {
        nodes,
        edges: sub_edges,
    }
}

fn focus_edge_summary_lines(
    focus: &str,
    edges: &[GraphTextEdge],
    max_lines: usize,
    width: usize,
) -> Vec<String> {
    let mut rels: Vec<(f64, String)> = Vec::new();
    let mut out_count = 0usize;
    let mut in_count = 0usize;
    for edge in edges {
        if edge.source == focus {
            out_count += 1;
            rels.push((
                edge.weight,
                format!(
                    "out  -> {}  ({:.2}, {})",
                    path_basename(&edge.target),
                    edge.weight,
                    clipped(&edge.kind, 14)
                ),
            ));
        } else if edge.target == focus {
            in_count += 1;
            rels.push((
                edge.weight,
                format!(
                    "in   <- {}  ({:.2}, {})",
                    path_basename(&edge.source),
                    edge.weight,
                    clipped(&edge.kind, 14)
                ),
            ));
        }
    }
    rels.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut lines = vec![clipped(
        &format!(
            "neighbors (focused subgraph): outgoing={} incoming={}",
            out_count, in_count
        ),
        width,
    )];
    for (_, text) in rels.into_iter().take(max_lines.max(1)) {
        lines.push(clipped(&format!("  {}", text), width));
    }
    if lines.len() == 1 {
        lines.push(clipped("  (no connected edges for current focus)", width));
    }
    lines
}

fn render_graph_cluster(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    renderer: GraphTextRenderer,
    include_cross_edges: bool,
) -> (Vec<String>, String, Option<String>) {
    if !include_cross_edges {
        return (
            render_focus_star_cluster(focus, width, height, visible_nodes, visible_edges),
            "focus-star".to_string(),
            Some("press x to show full cross-edge cluster".to_string()),
        );
    }
    let dot_available = command_exists("dot");
    match renderer {
        GraphTextRenderer::Graphviz => {
            if !dot_available {
                return (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    Some("graphviz not found on PATH; using ascii".to_string()),
                );
            }
            match render_graph_graphviz_plain(
                focus,
                width,
                height,
                visible_nodes,
                visible_edges,
                include_cross_edges,
            ) {
                Ok(lines) => (lines, "graphviz".to_string(), None),
                Err(err) => (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    Some(format!(
                        "graphviz failed; using ascii ({})",
                        clipped(&err, 80)
                    )),
                ),
            }
        }
        GraphTextRenderer::Auto => {
            if dot_available {
                match render_graph_graphviz_plain(
                    focus,
                    width,
                    height,
                    visible_nodes,
                    visible_edges,
                    include_cross_edges,
                ) {
                    Ok(lines) => (lines, "graphviz".to_string(), None),
                    Err(_) => (
                        render_graph_ascii(
                            focus,
                            width,
                            height,
                            visible_nodes,
                            visible_edges,
                            include_cross_edges,
                        ),
                        "ascii".to_string(),
                        Some("graphviz parse fallback to ascii".to_string()),
                    ),
                }
            } else {
                (
                    render_graph_ascii(
                        focus,
                        width,
                        height,
                        visible_nodes,
                        visible_edges,
                        include_cross_edges,
                    ),
                    "ascii".to_string(),
                    None,
                )
            }
        }
        GraphTextRenderer::Ascii => (
            render_graph_ascii(
                focus,
                width,
                height,
                visible_nodes,
                visible_edges,
                include_cross_edges,
            ),
            "ascii".to_string(),
            None,
        ),
    }
}

fn render_focus_star_cluster(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
) -> Vec<String> {
    let mut canvas = vec![vec![' '; width]; height];
    if visible_nodes.is_empty() {
        return vec![String::new(); height];
    }
    let (draw_nodes, draw_edges) = cluster_draw_data(focus, visible_nodes, visible_edges, false);
    let labels = build_cluster_label_map(&draw_nodes, focus);

    let mut in_map: HashMap<String, f64> = HashMap::new();
    let mut out_map: HashMap<String, f64> = HashMap::new();
    for edge in &draw_edges {
        if edge.source == focus {
            let cur = out_map.get(&edge.target).copied().unwrap_or(0.0);
            if edge.weight > cur {
                out_map.insert(edge.target.clone(), edge.weight);
            }
        } else if edge.target == focus {
            let cur = in_map.get(&edge.source).copied().unwrap_or(0.0);
            if edge.weight > cur {
                in_map.insert(edge.source.clone(), edge.weight);
            }
        }
    }
    let mut incoming: Vec<(String, f64)> = in_map.into_iter().collect();
    let mut outgoing: Vec<(String, f64)> = out_map.into_iter().collect();
    incoming.sort_by(|a, b| b.1.total_cmp(&a.1));
    outgoing.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max_side = ((height.saturating_sub(4)) / 2).clamp(3, 8);
    incoming.truncate(max_side);
    outgoing.truncate(max_side);

    let focus_x = if incoming.is_empty() && !outgoing.is_empty() {
        (width as i32 / 3).max(8)
    } else if outgoing.is_empty() && !incoming.is_empty() {
        (width as i32 * 2 / 3).max(8)
    } else {
        (width as i32 / 2).max(8)
    };
    let focus_y = if incoming.is_empty() || outgoing.is_empty() {
        4i32.min(height.saturating_sub(3) as i32).max(2)
    } else {
        (height as i32 / 2).max(3)
    };
    let focus_pt = GraphPoint {
        x: focus_x,
        y: focus_y,
    };
    let left_x = ((focus_pt.x - 44).max(2)).min((focus_pt.x - 8).max(2));
    let right_x = ((focus_pt.x + 44).min(width as i32 - 28)).max(focus_pt.x + 8);
    let left_mid = (focus_pt.x - 8).max(5);
    let right_mid = (focus_pt.x + 8).min(width.saturating_sub(6) as i32);

    let max_neighbors = incoming.len().max(outgoing.len()) as i32;
    let band = (max_neighbors + 1).max(4).min((height as i32 / 4).max(6));
    let min_slot_y = (focus_pt.y - band).max(1);
    let max_slot_y = (focus_pt.y + band).min(height.saturating_sub(2) as i32);
    let left_slots = spread_slots(incoming.len(), min_slot_y, max_slot_y);
    let right_slots = spread_slots(outgoing.len(), min_slot_y, max_slot_y);

    for (idx, (_node, _w)) in incoming.iter().enumerate() {
        let y = left_slots.get(idx).copied().unwrap_or(focus_pt.y);
        draw_manhattan_edge(
            &mut canvas,
            focus_pt,
            GraphPoint { x: left_x + 1, y },
            left_mid,
        );
    }
    for (idx, (_node, _w)) in outgoing.iter().enumerate() {
        let y = right_slots.get(idx).copied().unwrap_or(focus_pt.y);
        draw_manhattan_edge(
            &mut canvas,
            focus_pt,
            GraphPoint {
                x: right_x.saturating_sub(2),
                y,
            },
            right_mid,
        );
    }

    let focus_tag = labels
        .get(focus)
        .cloned()
        .unwrap_or_else(|| "[*]".to_string());
    let focus_title = format!("{} {}", focus_tag, short_graph_label(focus, 18));
    canvas_put_text(
        &mut canvas,
        (focus_pt.x - 1).max(0),
        focus_pt.y,
        &focus_title,
    );

    for (idx, (node, w)) in incoming.iter().enumerate() {
        let y = left_slots.get(idx).copied().unwrap_or(focus_pt.y);
        let tag = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let txt = format!("{} {} ({:.2})", tag, short_graph_label(node, 16), w);
        canvas_put_text(&mut canvas, left_x, y, &txt);
    }
    for (idx, (node, w)) in outgoing.iter().enumerate() {
        let y = right_slots.get(idx).copied().unwrap_or(focus_pt.y);
        let tag = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let txt = format!("{} {} ({:.2})", tag, short_graph_label(node, 16), w);
        canvas_put_text(&mut canvas, right_x, y, &txt);
    }

    canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect()
}

fn cluster_draw_data(
    focus: &str,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> (Vec<String>, Vec<GraphTextEdge>) {
    let mut focus_edges: Vec<GraphTextEdge> = Vec::new();
    let mut non_focus_edges: Vec<GraphTextEdge> = Vec::new();
    for edge in visible_edges {
        if edge.source == focus || edge.target == focus {
            focus_edges.push(edge.clone());
        } else {
            non_focus_edges.push(edge.clone());
        }
    }
    focus_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    non_focus_edges.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    let mut draw_edges: Vec<GraphTextEdge> = focus_edges.into_iter().take(16).collect();
    if include_cross_edges {
        draw_edges.extend(non_focus_edges.into_iter().take(8));
    }

    let mut draw_nodes: Vec<String> = Vec::new();
    let mut seen_nodes: HashSet<String> = HashSet::new();
    let push_node = |out: &mut Vec<String>, seen: &mut HashSet<String>, node: &str| {
        if seen.insert(node.to_string()) {
            out.push(node.to_string());
        }
    };
    push_node(&mut draw_nodes, &mut seen_nodes, focus);
    for edge in &draw_edges {
        push_node(&mut draw_nodes, &mut seen_nodes, &edge.source);
        push_node(&mut draw_nodes, &mut seen_nodes, &edge.target);
    }
    if draw_nodes.len() < 10 {
        for node in visible_nodes {
            push_node(&mut draw_nodes, &mut seen_nodes, node);
            if draw_nodes.len() >= 10 {
                break;
            }
        }
    }
    (draw_nodes, draw_edges)
}

fn render_graph_ascii(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> Vec<String> {
    let mut canvas = vec![vec![' '; width]; height];
    if visible_nodes.is_empty() {
        return vec![String::new(); height];
    }
    let (draw_nodes, draw_edges) =
        cluster_draw_data(focus, visible_nodes, visible_edges, include_cross_edges);
    let positions = layout_cluster_positions(&draw_nodes, focus, width, height);

    for edge in &draw_edges {
        let Some(a) = positions.get(&edge.source) else {
            continue;
        };
        let Some(b) = positions.get(&edge.target) else {
            continue;
        };
        draw_graph_line(&mut canvas, *a, *b);
    }

    let labels = build_cluster_label_map(&draw_nodes, focus);
    let center_x = (width as i32) / 2;
    for node in &draw_nodes {
        let Some(pt) = positions.get(node) else {
            continue;
        };
        let token = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let token_len = token.chars().count() as i32;
        let token_x = (pt.x - (token_len / 2)).clamp(0, width.saturating_sub(1) as i32);
        canvas_put_text(&mut canvas, token_x, pt.y, &token);
        let label = short_graph_label(node, 12);
        let label_len = label.chars().count() as i32;
        let label_x = if pt.x <= center_x {
            pt.x + token_len / 2 + 2
        } else {
            (pt.x - token_len / 2 - label_len - 2).max(0)
        };
        canvas_put_text(&mut canvas, label_x, pt.y, &label);
    }

    canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect()
}

#[derive(Default)]
struct GraphvizPlainLayout {
    width: f64,
    height: f64,
    node_positions: HashMap<String, (f64, f64)>,
}

fn render_graph_graphviz_plain(
    focus: &str,
    width: usize,
    height: usize,
    visible_nodes: &[String],
    visible_edges: &[GraphTextEdge],
    include_cross_edges: bool,
) -> Result<Vec<String>, String> {
    let (draw_nodes, draw_edges) =
        cluster_draw_data(focus, visible_nodes, visible_edges, include_cross_edges);
    if draw_nodes.is_empty() {
        return Ok(vec![String::new(); height]);
    }

    let mut node_ids: HashMap<String, String> = HashMap::new();
    let mut id_to_node: HashMap<String, String> = HashMap::new();
    let mut dot = String::from(
        "digraph G {\n\
         graph [overlap=false,splines=false,outputorder=edgesfirst,pad=0.12,nodesep=0.20,ranksep=0.24];\n\
         node [shape=box,style=rounded,fontname=\"Menlo\",fontsize=10];\n\
         edge [fontname=\"Menlo\",fontsize=8,dir=none];\n",
    );
    for (idx, node) in draw_nodes.iter().enumerate() {
        let id = format!("n{}", idx);
        node_ids.insert(node.clone(), id.clone());
        id_to_node.insert(id.clone(), node.clone());
        let mut label = short_graph_label(node, 22);
        if node == focus {
            label = format!("* {}", label);
        }
        dot.push_str(&format!("{} [label=\"{}\"];\n", id, dot_escape(&label)));
    }
    for edge in &draw_edges {
        let Some(src) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst) = node_ids.get(&edge.target) else {
            continue;
        };
        dot.push_str(&format!("{} -> {};\n", src, dst));
    }
    dot.push_str("}\n");

    let mut child = Command::new("dot")
        .arg("-Kneato")
        .arg("-Tplain")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed spawning dot: {}", e))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(dot.as_bytes())
            .map_err(|e| format!("failed writing graphviz input: {}", e))?;
    } else {
        return Err("dot stdin unavailable".to_string());
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed reading dot output: {}", e))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "dot exited with status {}: {}",
            output.status,
            clipped(err.trim(), 120)
        ));
    }
    let plain = String::from_utf8_lossy(&output.stdout);
    let layout = parse_graphviz_plain_layout(&plain)?;

    let mut canvas = vec![vec![' '; width]; height];
    for edge in &draw_edges {
        let Some(src_id) = node_ids.get(&edge.source) else {
            continue;
        };
        let Some(dst_id) = node_ids.get(&edge.target) else {
            continue;
        };
        let Some((sx, sy)) = layout.node_positions.get(src_id) else {
            continue;
        };
        let Some((dx, dy)) = layout.node_positions.get(dst_id) else {
            continue;
        };
        let a = map_graphviz_point((*sx, *sy), width, height, layout.width, layout.height);
        let b = map_graphviz_point((*dx, *dy), width, height, layout.width, layout.height);
        draw_graph_line(&mut canvas, a, b);
    }

    let labels = build_cluster_label_map(&draw_nodes, focus);
    let center_x = (width as i32) / 2;
    for (id, (x, y)) in &layout.node_positions {
        let Some(node) = id_to_node.get(id) else {
            continue;
        };
        let pt = map_graphviz_point((*x, *y), width, height, layout.width, layout.height);
        let token = labels
            .get(node)
            .cloned()
            .unwrap_or_else(|| "[??]".to_string());
        let token_len = token.chars().count() as i32;
        let token_x = (pt.x - (token_len / 2)).clamp(0, width.saturating_sub(1) as i32);
        canvas_put_text(&mut canvas, token_x, pt.y, &token);
        let label = short_graph_label(node, 16);
        let label_len = label.chars().count() as i32;
        let label_x = if pt.x <= center_x {
            pt.x + token_len / 2 + 2
        } else {
            (pt.x - token_len / 2 - label_len - 2).max(0)
        };
        canvas_put_text(&mut canvas, label_x, pt.y, &label);
    }

    Ok(canvas
        .into_iter()
        .map(|row| row.into_iter().collect::<String>())
        .collect())
}

fn parse_graphviz_plain_layout(raw: &str) -> Result<GraphvizPlainLayout, String> {
    let mut out = GraphvizPlainLayout::default();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "graph" if parts.len() >= 4 => {
                out.width = parts[2].parse::<f64>().unwrap_or(0.0);
                out.height = parts[3].parse::<f64>().unwrap_or(0.0);
            }
            "node" if parts.len() >= 4 => {
                let id = parts[1].to_string();
                let x = parts[2].parse::<f64>().unwrap_or(0.0);
                let y = parts[3].parse::<f64>().unwrap_or(0.0);
                out.node_positions.insert(id, (x, y));
            }
            _ => {}
        }
    }
    if out.node_positions.is_empty() {
        return Err("graphviz plain output had no node positions".to_string());
    }
    if out.width <= 0.0 || out.height <= 0.0 {
        out.width = 1.0;
        out.height = 1.0;
    }
    Ok(out)
}

fn map_graphviz_point(
    pt: (f64, f64),
    width: usize,
    height: usize,
    graph_width: f64,
    graph_height: f64,
) -> GraphPoint {
    let graph_width = graph_width.max(1.0);
    let graph_height = graph_height.max(1.0);
    let margin_x = 2i32;
    let margin_y = 1i32;
    let usable_w = width.saturating_sub(4).max(4) as f64;
    let usable_h = height.saturating_sub(3).max(3) as f64;
    let x = margin_x + ((pt.0 / graph_width) * usable_w).round() as i32;
    let y = margin_y + (((graph_height - pt.1) / graph_height) * usable_h).round() as i32;
    GraphPoint {
        x: x.clamp(0, width.saturating_sub(1) as i32),
        y: y.clamp(0, height.saturating_sub(1) as i32),
    }
}

fn dot_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' | '\r' => out.push(' '),
            _ => out.push(ch),
        }
    }
    out
}

fn layout_cluster_positions(
    nodes: &[String],
    focus: &str,
    width: usize,
    height: usize,
) -> HashMap<String, GraphPoint> {
    let mut out: HashMap<String, GraphPoint> = HashMap::new();
    if nodes.is_empty() {
        return out;
    }
    let cx = (width as i32 / 2).max(4);
    let cy = (height as i32 / 2).max(3);
    out.insert(focus.to_string(), GraphPoint { x: cx, y: cy });

    let others: Vec<&String> = nodes.iter().filter(|n| n.as_str() != focus).collect();
    if others.is_empty() {
        return out;
    }
    let rx = ((width as i32 / 2) - 12).max(8) as f64;
    let ry = ((height as i32 / 2) - 4).max(4) as f64;
    for (idx, node) in others.iter().enumerate() {
        let theta = -PI / 2.0 + ((idx as f64) * (2.0 * PI / others.len() as f64));
        let x = (cx as f64 + rx * theta.cos()).round() as i32;
        let y = (cy as f64 + ry * theta.sin()).round() as i32;
        out.insert(
            (*node).clone(),
            GraphPoint {
                x: x.clamp(1, width.saturating_sub(2) as i32),
                y: y.clamp(1, height.saturating_sub(2) as i32),
            },
        );
    }
    out
}

fn draw_graph_line(canvas: &mut [Vec<char>], a: GraphPoint, b: GraphPoint) {
    let mut x = a.x;
    let mut y = a.y;
    let dx = (b.x - a.x).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let dy = -(b.y - a.y).abs();
    let sy = if a.y < b.y { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x == b.x && y == b.y {
            break;
        }
        let mut nx = x;
        let mut ny = y;
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            nx += sx;
        }
        if e2 <= dx {
            err += dx;
            ny += sy;
        }
        if !(nx == b.x && ny == b.y) && !(nx == a.x && ny == a.y) {
            let step_ch = if nx == x {
                '|'
            } else if ny == y {
                '-'
            } else if (nx - x) == (ny - y) {
                '\\'
            } else {
                '/'
            };
            canvas_put_char(canvas, nx, ny, step_ch, false);
        }
        x = nx;
        y = ny;
    }
}

fn draw_manhattan_edge(canvas: &mut [Vec<char>], from: GraphPoint, to: GraphPoint, mid_x: i32) {
    draw_hline(canvas, from.y, from.x, mid_x);
    draw_vline(canvas, mid_x, from.y, to.y);
    draw_hline(canvas, to.y, mid_x, to.x);
}

fn draw_hline(canvas: &mut [Vec<char>], y: i32, x0: i32, x1: i32) {
    if canvas.is_empty() {
        return;
    }
    let (a, b) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    for x in a..=b {
        canvas_put_char(canvas, x, y, '-', false);
    }
}

fn draw_vline(canvas: &mut [Vec<char>], x: i32, y0: i32, y1: i32) {
    if canvas.is_empty() {
        return;
    }
    let (a, b) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    for y in a..=b {
        canvas_put_char(canvas, x, y, '|', false);
    }
}

fn spread_slots(count: usize, min_y: i32, max_y: i32) -> Vec<i32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(min_y + max_y) / 2];
    }
    let span = (max_y - min_y).max(1) as f64;
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let t = (idx + 1) as f64 / (count + 1) as f64;
        let y = (min_y as f64 + span * t).round() as i32;
        out.push(y.clamp(min_y, max_y));
    }
    out
}

fn canvas_put_char(canvas: &mut [Vec<char>], x: i32, y: i32, ch: char, overwrite: bool) {
    if canvas.is_empty() {
        return;
    }
    let h = canvas.len() as i32;
    let w = canvas[0].len() as i32;
    if x < 0 || y < 0 || x >= w || y >= h {
        return;
    }
    let cell = &mut canvas[y as usize][x as usize];
    if overwrite {
        *cell = ch;
        return;
    }
    if *cell == ' ' {
        *cell = ch;
        return;
    }
    if *cell == ch {
        return;
    }
    if *cell == 'o' || *cell == '*' {
        return;
    }
    *cell = '+';
}

fn canvas_put_text(canvas: &mut [Vec<char>], x: i32, y: i32, text: &str) {
    if canvas.is_empty() {
        return;
    }
    let h = canvas.len() as i32;
    let w = canvas[0].len() as i32;
    if y < 0 || y >= h {
        return;
    }
    let mut cursor = x;
    for ch in text.chars() {
        if cursor >= 0 && cursor < w {
            canvas[y as usize][cursor as usize] = ch;
        }
        cursor += 1;
        if cursor >= w {
            break;
        }
    }
}

fn short_graph_label(path: &str, max_width: usize) -> String {
    clipped(&path_basename(path), max_width.max(4))
}

fn build_cluster_label_map(nodes: &[String], focus: &str) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert(focus.to_string(), "[*]".to_string());
    let mut idx = 1usize;
    for node in nodes {
        if node == focus {
            continue;
        }
        map.insert(node.clone(), format!("[{:02}]", idx));
        idx += 1;
    }
    map
}

fn cluster_node_legend_lines(
    focus: &str,
    nodes: &[String],
    width: usize,
    max_lines: usize,
) -> Vec<String> {
    if max_lines == 0 || nodes.is_empty() {
        return Vec::new();
    }
    let labels = build_cluster_label_map(nodes, focus);
    let mut lines = Vec::new();
    lines.push(clipped("node legend:", width));
    lines.push(clipped(
        &format!(
            "  [*] {}",
            short_graph_label(focus, width.saturating_sub(6))
        ),
        width,
    ));
    let mut count = 0usize;
    for node in nodes {
        if node == focus {
            continue;
        }
        let Some(tag) = labels.get(node) else {
            continue;
        };
        lines.push(clipped(
            &format!(
                "  {} {}",
                tag,
                short_graph_label(node, width.saturating_sub(8))
            ),
            width,
        ));
        count += 1;
        if count >= max_lines.saturating_sub(2) {
            break;
        }
    }
    lines
}

fn render_lineage_diagram(
    focus: &str,
    edges: &[GraphTextEdge],
    width: usize,
    height: usize,
) -> Vec<String> {
    let mut incoming: Vec<(f64, String)> = Vec::new();
    let mut outgoing: Vec<(f64, String)> = Vec::new();
    for edge in edges {
        if edge.target == focus {
            incoming.push((edge.weight, short_graph_label(&edge.source, 30)));
        } else if edge.source == focus {
            outgoing.push((edge.weight, short_graph_label(&edge.target, 30)));
        }
    }
    incoming.sort_by(|a, b| b.0.total_cmp(&a.0));
    outgoing.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut lines: Vec<String> = Vec::new();
    lines.push(clipped(
        "lineage lanes: incoming -> focus -> outgoing (filtered by threshold/depth)",
        width,
    ));
    lines.push(clipped(
        &format!("focus node: {}", short_graph_label(focus, 42)),
        width,
    ));
    lines.push(clipped(
        &format!("incoming={} outgoing={}", incoming.len(), outgoing.len()),
        width,
    ));

    let max_rows = height.saturating_sub(5).max(1);
    if incoming.is_empty() {
        for (idx, (_, node)) in outgoing.iter().take(max_rows).enumerate() {
            lines.push(clipped(
                &format!("  {:>2}. [*] -> {}", idx + 1, node),
                width,
            ));
        }
    } else if outgoing.is_empty() {
        for (idx, (_, node)) in incoming.iter().take(max_rows).enumerate() {
            lines.push(clipped(
                &format!("  {:>2}. {} -> [*]", idx + 1, node),
                width,
            ));
        }
    } else {
        let left_w = ((width.saturating_sub(16)) / 2).clamp(10, 32);
        let right_w = width.saturating_sub(left_w + 10).clamp(12, 44);
        let rows = incoming.len().max(outgoing.len()).max(1).min(max_rows);
        for idx in 0..rows {
            let left = incoming
                .get(idx)
                .map(|(_, s)| clipped(s, left_w))
                .unwrap_or_else(String::new);
            let right = outgoing
                .get(idx)
                .map(|(_, s)| clipped(s, right_w))
                .unwrap_or_else(String::new);
            lines.push(clipped(
                &format!("{:<left_w$} -> [*] -> {}", left, right, left_w = left_w),
                width,
            ));
        }
    }

    if lines.len() < height {
        lines.push(clipped(
            "hint: w/s threshold, [/] depth, t layout, v renderer",
            width,
        ));
    }
    while lines.len() < height {
        lines.push(String::new());
    }
    lines.truncate(height);
    lines
}

fn render_graph_matrix(
    focus: &str,
    nodes: &[String],
    edges: &[GraphTextEdge],
    width: usize,
    height: usize,
) -> Vec<String> {
    if nodes.is_empty() {
        return vec![String::new(); height];
    }
    let mut ordered: Vec<String> = Vec::new();
    ordered.push(focus.to_string());
    for node in nodes {
        if node != focus {
            ordered.push(node.clone());
        }
        if ordered.len() >= 10 {
            break;
        }
    }
    let mut edge_map: HashMap<(String, String), f64> = HashMap::new();
    for edge in edges {
        let k = (edge.source.clone(), edge.target.clone());
        let cur = edge_map.get(&k).copied().unwrap_or(0.0);
        if edge.weight > cur {
            edge_map.insert(k, edge.weight);
        }
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push(clipped(
        "adjacency matrix: value = directed edge weight (source row -> target col)",
        width,
    ));
    let mut header = String::from("     ");
    for idx in 0..ordered.len() {
        header.push_str(&format!("{:>4}", idx));
    }
    lines.push(clipped(&header, width));
    for (row_idx, src) in ordered.iter().enumerate() {
        let mut row = format!("{:>3} ", row_idx);
        for dst in &ordered {
            let value = edge_map
                .get(&(src.clone(), dst.clone()))
                .copied()
                .unwrap_or(0.0);
            if src == dst {
                row.push_str("   .");
            } else if value <= 0.0 {
                row.push_str("   -");
            } else {
                row.push_str(&format!("{:>4}", (value * 10.0).round() as i32));
            }
        }
        lines.push(clipped(&row, width));
    }
    lines.push(clipped("legend:", width));
    for (idx, node) in ordered.iter().enumerate() {
        lines.push(clipped(
            &format!(
                "  {:>2}: {}",
                idx,
                short_graph_label(node, width.saturating_sub(8))
            ),
            width,
        ));
    }
    while lines.len() < height {
        lines.push(String::new());
    }
    lines.truncate(height);
    lines
}

#[derive(Debug)]
struct ShellCommandOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

fn run_shell_capture(command: &str) -> Result<ShellCommandOutput, String> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg(command)
        .output()
        .map_err(|e| format!("failed running shell command '{}': {}", command, e))?;
    Ok(ShellCommandOutput {
        exit_code: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    })
}

fn runtime_config_value(key: &str) -> Option<String> {
    let cwd = env::current_dir().ok()?;
    let map = load_config_values(&config_path(&cwd));
    map.get(key)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn graph_runtime_dir_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("runtime").join("graph")
}

fn graph_runtime_log_path(cwd: &Path) -> PathBuf {
    graph_runtime_dir_path(cwd).join("runtime.log")
}

fn graph_runtime_state_path(cwd: &Path) -> PathBuf {
    graph_runtime_dir_path(cwd).join("runtime-state.json")
}

#[derive(Debug, Clone)]
struct GraphRuntimeState {
    pid: u32,
    host: String,
    port: u16,
}

fn load_graph_runtime_state(cwd: &Path) -> Option<Value> {
    let path = graph_runtime_state_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<Value>(&raw).ok()
}

fn parse_graph_runtime_state(cwd: &Path) -> Option<GraphRuntimeState> {
    let payload = load_graph_runtime_state(cwd)?;
    let pid = payload.get("pid").and_then(Value::as_u64)? as u32;
    let host = payload
        .get("host")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())?;
    let port = payload.get("port").and_then(Value::as_u64)? as u16;
    if pid == 0 || host.is_empty() || port == 0 {
        return None;
    }
    Some(GraphRuntimeState { pid, host, port })
}

fn persist_graph_runtime_state(cwd: &Path, pid: u32, host: &str, port: u16) -> Result<(), String> {
    let runtime_dir = graph_runtime_dir_path(cwd);
    fs::create_dir_all(&runtime_dir).map_err(|e| {
        format!(
            "failed creating graph runtime dir '{}': {}",
            runtime_dir.display(),
            e
        )
    })?;
    let payload = serde_json::json!({
        "pid": pid,
        "host": host,
        "port": port,
        "updated_at": now_ts(),
    });
    let raw = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("failed serializing graph runtime state: {}", e))?;
    fs::write(graph_runtime_state_path(cwd), raw)
        .map_err(|e| format!("failed writing graph runtime state: {}", e))
}

fn clear_graph_runtime_state(cwd: &Path) {
    let _ = fs::remove_file(graph_runtime_state_path(cwd));
}

fn stop_pid_graceful(pid: u32) -> bool {
    let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if !pid_is_alive(pid) {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
    !pid_is_alive(pid)
}

fn shell_words(input: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = input.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(ch) = chars.next() {
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                cur.push(ch);
            }
            continue;
        }
        if in_double {
            match ch {
                '"' => in_double = false,
                '\\' => {
                    if let Some(next) = chars.next() {
                        cur.push(next);
                    }
                }
                _ => cur.push(ch),
            }
            continue;
        }
        match ch {
            '\'' => in_single = true,
            '"' => in_double = true,
            '\\' => {
                if let Some(next) = chars.next() {
                    cur.push(next);
                }
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn extract_loadmodule_path(raw_cmd: &str) -> Option<String> {
    let words = shell_words(raw_cmd);
    for idx in 0..words.len() {
        let tok = words[idx].trim();
        if tok == "--loadmodule" {
            if idx + 1 < words.len() {
                let v = words[idx + 1].trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
            continue;
        }
        if let Some(v) = tok.strip_prefix("--loadmodule=") {
            let vv = v.trim();
            if !vv.is_empty() {
                return Some(vv.to_string());
            }
        }
    }
    None
}

fn listener_pid_for_port(port: u16) -> Option<u32> {
    if !command_exists("lsof") {
        return None;
    }
    let out = run_shell_capture(&format!(
        "lsof -nP -tiTCP:{} -sTCP:LISTEN 2>/dev/null | head -n1",
        port
    ))
    .ok()?;
    if out.exit_code != 0 {
        return None;
    }
    out.stdout
        .lines()
        .find_map(|line| line.trim().parse::<u32>().ok())
}

fn pid_command_line(pid: u32) -> Option<String> {
    let out = run_shell_capture(&format!("ps -p {} -o command= 2>/dev/null", pid)).ok()?;
    if out.exit_code != 0 {
        return None;
    }
    let cmd = out.stdout.trim();
    if cmd.is_empty() {
        None
    } else {
        Some(cmd.to_string())
    }
}

fn pid_is_alive(pid: u32) -> bool {
    let out = run_shell_capture(&format!("kill -0 {} >/dev/null 2>&1", pid));
    matches!(out, Ok(out) if out.exit_code == 0)
}

fn ensure_retrieval_backend_ready(
    _cfg: &ConfigValues,
    _auto_start: bool,
    _context: &str,
) -> Result<(), String> {
    // LanceDB is embedded — no external server process needed.
    // Just ensure the lance directory exists.
    let lance_path = data_dir(Path::new("")).join("lance");
    if !lance_path.exists() {
        fs::create_dir_all(&lance_path)
            .map_err(|e| format!("failed creating lance directory: {}", e))?;
    }
    Ok(())
}

fn run_init(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage:");
        println!("  retrivio init [--root <path>] [--embed-backend <ollama|bedrock>] [--embed-model <id>]");
        println!("  retrivio init <bash|zsh|fish>");
        return;
    }

    if let Some(first) = args.first() {
        let shell = first.to_string_lossy().to_ascii_lowercase();
        if matches!(shell.as_str(), "bash" | "zsh" | "fish") {
            if args.len() > 1 {
                eprintln!(
                    "error: shell init does not take extra arguments (got {})",
                    args[1].to_string_lossy()
                );
                process::exit(2);
            }
            print!("{}", render_shell_init_script(&shell));
            return;
        }
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);

    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let before_cfg = cfg.clone();
    let mut explicit_model = false;
    let mut explicit_embed_backend = false;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].to_string_lossy().to_string();
        match arg.as_str() {
            "--root" => {
                i += 1;
                let v = arg_value(args, i, "--root");
                cfg.root = normalize_path(&v);
            }
            "--embed-backend" => {
                i += 1;
                let mut v = arg_value(args, i, "--embed-backend").to_lowercase();
                if v == "auto" {
                    v = "ollama".to_string();
                }
                if !matches!(v.as_str(), "ollama" | "bedrock") {
                    eprintln!("error: invalid --embed-backend '{}'", v);
                    process::exit(2);
                }
                cfg.embed_backend = v;
                explicit_embed_backend = true;
            }
            "--embed-model" => {
                i += 1;
                cfg.embed_model = arg_value(args, i, "--embed-model");
                explicit_model = true;
            }
            "--retrieval-backend" => {
                i += 1;
                let _raw = arg_value(args, i, "--retrieval-backend");
                // LanceDB is the only backend; silently accept any value
                cfg.retrieval_backend = "lancedb".to_string();
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    if !explicit_model {
        if explicit_embed_backend {
            cfg.embed_model = default_embed_model_for_backend(&cfg.embed_backend).to_string();
        } else {
            let current_model = cfg.embed_model.trim();
            let old_default = default_embed_model_for_backend(&before_cfg.embed_backend);
            if current_model.is_empty()
                || current_model == old_default
                || current_model == "sentence-transformers/all-MiniLM-L6-v2"
            {
                cfg.embed_model = default_embed_model_for_backend(&cfg.embed_backend).to_string();
            }
        }
    }

    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed to write config: {}", e);
        process::exit(1);
    });

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });
    if let Some(reason) = refresh_reembed_requirement_for_config_change(&cwd, &before_cfg, &cfg)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        })
    {
        println!("warning: {}", reason);
    }

    println!("config: {}", cfg_path.display());
    println!("database: {}", db_path.display());
    println!("root: {}", cfg.root.display());
    println!("embed_backend: {}", cfg.embed_backend);
    println!("embed_model: {}", cfg.embed_model);
    println!("retrieval_backend: {}", cfg.retrieval_backend);
}

fn render_shell_init_script(shell: &str) -> String {
    let exec_path = env::current_exe()
        .ok()
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let exec_q = shell_escape(&exec_path);
    let wrapper_env = SHELL_WRAPPER_ENV;
    let passthrough = "init|install|setup|auth|add|del|roots|index|refresh|reembed|watch|search|pick|jump|doctor|config|autotune|version|api|mcp|self-test|graph|bench|daemon|legacy|ui|stop|help|-h|--help";

    match shell {
        "fish" => format!(
            r#"# retrivio shell init (fish)
set -g __retrivio_bin {exec_q}
function __retrivio_exec --description "Execute Retrivio binary"
  if test -n "$__retrivio_bin"; and test -x "$__retrivio_bin"
    env {wrapper_env}=1 "$__retrivio_bin" $argv
  else
    env {wrapper_env}=1 command retrivio $argv
  end
end
function retrivio --description "Retrivio shell wrapper with cwd jump"
  set -l sub ""
  if test (count $argv) -gt 0
    set sub $argv[1]
  end
  if string match -qr '^--(data-dir|config)(=|$)' -- "$sub"
    __retrivio_exec $argv
    return $status
  end
  switch $sub
    case {passthrough}
      __retrivio_exec $argv
      return $status
  end
  set -l mode_flag
  if test (count $argv) -gt 0
    switch $argv[1]
      case --files -f
        set mode_flag --files
        set -e argv[1]
      case --dirs -d --directories
        set mode_flag --dirs
        set -e argv[1]
    end
  end
  set -l target (__retrivio_exec jump $mode_flag $argv)
  if test $status -ne 0
    return $status
  end
  if test -z "$target"
    return 1
  end
  if test -d "$target"
    cd "$target"
    return $status
  end
  if test -f "$target"
    if set -q VISUAL
      $VISUAL "$target"
      return $status
    end
    if set -q EDITOR
      $EDITOR "$target"
      return $status
    end
    if command -v open >/dev/null 2>&1
      open "$target"
      return $status
    end
  end
  return 1
end
function s; retrivio $argv; end
function sd; retrivio --dirs $argv; end
function sf; retrivio --files $argv; end
function cg; retrivio $argv; end
"#
        ),
        _ => format!(
            r#"# retrivio shell init ({shell})
_retrivio_bin={exec_q}
_retrivio_exec() {{
  if [ -n "${{_retrivio_bin:-}}" ] && [ -x "${{_retrivio_bin}}" ]; then
    {wrapper_env}=1 "${{_retrivio_bin}}" "$@"
  else
    {wrapper_env}=1 command retrivio "$@"
  fi
}}
retrivio() {{
  local sub="${{1:-}}"
  case "$sub" in
    --data-dir|--data-dir=*|--config|--config=*)
      _retrivio_exec "$@"
      return $?
      ;;
  esac
  case "$sub" in
    {passthrough})
      _retrivio_exec "$@"
      return $?
      ;;
  esac
  local mode_flag=""
  case "${{1:-}}" in
    --files|-f)
      mode_flag="--files"
      shift
      ;;
    --dirs|-d|--directories)
      mode_flag="--dirs"
      shift
      ;;
  esac
  local target
  target="$(_retrivio_exec jump ${{mode_flag:+$mode_flag}} "$@")" || return $?
  [ -z "$target" ] && return 1
  if [ -d "$target" ]; then
    builtin cd "$target" || return $?
    return 0
  fi
  if [ -f "$target" ]; then
    if [ -n "${{VISUAL:-}}" ]; then "${{VISUAL}}" "$target"; return $?; fi
    if [ -n "${{EDITOR:-}}" ]; then "${{EDITOR}}" "$target"; return $?; fi
    if command -v open >/dev/null 2>&1; then open "$target"; return $?; fi
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$target" >/dev/null 2>&1; return $?; fi
    return 0
  fi
  return 1
}}
s() {{ retrivio "$@"; }}
sd() {{ retrivio --dirs "$@"; }}
sf() {{ retrivio --files "$@"; }}
cg() {{ retrivio "$@"; }}
rv() {{ retrivio ""; }}
"#
        ),
    }
}

fn run_install_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio install [--no-system-install] [--no-download] [--no-shell-hook] [--legacy] [--venv <path>] [--bench]");
        println!("notes:");
        println!("  - uses global Retrivio state under ~/.retrivio by default");
        println!("  - ensures LanceDB directory exists (embedded, no external process)");
        println!("  - attempts to install fswatch for low-latency event-driven watch");
        println!("  - installs shell hook into ~/.bashrc and ~/.zshrc by default (eval \"$(retrivio init <shell>)\")");
        println!("  - native Rust api/mcp runtime is included by default");
        println!("  - use --legacy only if you explicitly want the Python compatibility runtime");
        return;
    }

    let mut legacy_venv: Option<String> = None;
    let mut legacy_bench = false;
    let mut run_legacy_install = false;
    let mut allow_system_install = true;
    let mut allow_download = true;
    let mut install_shell_hook = true;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--venv" => {
                i += 1;
                legacy_venv = Some(arg_value(args, i, "--venv"));
                run_legacy_install = true;
            }
            "--bench" => {
                legacy_bench = true;
                run_legacy_install = true;
            }
            "--legacy" => {
                run_legacy_install = true;
            }
            "--no-legacy" => {
                run_legacy_install = false;
            }
            "--no-system-install" => {
                allow_system_install = false;
            }
            "--no-download" => {
                allow_download = false;
            }
            "--no-shell-hook" => {
                install_shell_hook = false;
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let dbp = db_path(&cwd);
    let state_dir = data_dir(&cwd);

    fs::create_dir_all(&state_dir).unwrap_or_else(|e| {
        eprintln!(
            "error: failed to create state dir '{}': {}",
            state_dir.display(),
            e
        );
        process::exit(1);
    });
    fs::create_dir_all(state_dir.join("bin")).unwrap_or_else(|e| {
        eprintln!("error: failed to create runtime bin dir: {}", e);
        process::exit(1);
    });
    fs::create_dir_all(graph_runtime_dir_path(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: failed to create graph runtime dir: {}", e);
        process::exit(1);
    });

    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_db_schema(&dbp).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut notes: Vec<String> = Vec::new();
    let mut fswatch_ready = command_exists("fswatch");
    if !fswatch_ready {
        if allow_system_install {
            match try_install_fswatch_with_homebrew() {
                Ok(true) => {
                    fswatch_ready = true;
                    notes.push("fswatch installed; watch will use event-driven mode".to_string());
                }
                Ok(false) => {
                    notes
                        .push("fswatch not available; watch will use polling fallback".to_string());
                }
                Err(err) => {
                    notes.push(format!(
                        "fswatch install attempt failed (watch will use polling fallback): {}",
                        err
                    ));
                }
            }
        } else {
            notes.push(
                "fswatch missing and system install disabled; watch will use polling fallback"
                    .to_string(),
            );
        }
    } else {
        notes.push("fswatch detected; watch will use event-driven mode".to_string());
    }
    // Ensure LanceDB directory exists (embedded — no external process needed)
    let lance_path = data_dir(&cwd).join("lance");
    if !lance_path.exists() {
        std::fs::create_dir_all(&lance_path).unwrap_or_else(|e| {
            eprintln!("error: failed to create LanceDB directory: {}", e);
            process::exit(1);
        });
    }
    notes.push(format!(
        "lancedb: directory ready at {}",
        lance_path.display()
    ));

    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed to write config: {}", e);
        process::exit(1);
    });

    println!("install: complete");
    println!("config: {}", cfg_path.display());
    println!("database: {}", dbp.display());
    println!("state_dir: {}", state_dir.display());
    println!("retrieval_backend: lancedb (embedded)");
    println!(
        "watch_mode: {}",
        if fswatch_ready {
            "event-driven (fswatch)"
        } else {
            "polling fallback"
        }
    );
    for note in notes {
        println!("note: {}", note);
    }
    if install_shell_hook {
        match install_default_shell_hooks(&cwd) {
            Ok(messages) => {
                for line in messages {
                    println!("shell_hook: {}", line);
                }
            }
            Err(err) => {
                eprintln!("warning: shell hook setup failed: {}", err);
            }
        }
    } else {
        println!("shell_hook: skipped (--no-shell-hook)");
    }

    let legacy_venv_path = if let Some(raw) = &legacy_venv {
        expand_tilde(raw)
    } else {
        state_dir.join("venv")
    };
    let legacy_cmd_present = legacy_venv_path.join("bin").join("retrivio").exists();
    let should_run_legacy =
        run_legacy_install && (legacy_bench || legacy_venv.is_some() || !legacy_cmd_present);

    if run_legacy_install && !legacy_bench && legacy_venv.is_none() && legacy_cmd_present {
        println!(
            "install: legacy runtime already present at {}; skipping legacy reinstall",
            legacy_venv_path.display()
        );
    }

    if should_run_legacy {
        println!("install: delegating to legacy installer for api/mcp runtime...");
        let mut bridge_args: Vec<OsString> = vec![OsString::from("install")];
        if legacy_bench {
            bridge_args.push(OsString::from("--bench"));
        }
        if let Some(path) = legacy_venv {
            bridge_args.push(OsString::from("--venv"));
            bridge_args.push(OsString::from(path));
        }
        let code = run_legacy_bridge_status(&bridge_args);
        if code != 0 {
            if legacy_bench {
                eprintln!("error: legacy installer failed with status {}", code);
                process::exit(code);
            }
            eprintln!(
                "warning: legacy installer failed with status {} (api/mcp bridge commands may be unavailable)",
                code
            );
        }
    } else {
        println!("install: legacy Python runtime not requested (native-only install complete)");
    }
}

fn render_shell_hook_block(shell: &str, preferred_bin: &Path) -> String {
    let block = format!(
        "{start}\nif [ -x {bin} ]; then\n  eval \"$({bin} init {shell})\"\nelif command -v retrivio >/dev/null 2>&1; then\n  eval \"$(retrivio init {shell})\"\nfi\n{end}\n",
        start = SHELL_HOOK_MARKER_START,
        bin = shell_escape(&preferred_bin.to_string_lossy()),
        shell = shell,
        end = SHELL_HOOK_MARKER_END
    );
    block
}

fn ensure_shell_hook_in_rc_file(
    rc_path: &Path,
    shell: &str,
    preferred_bin: &Path,
) -> Result<String, String> {
    let block = render_shell_hook_block(shell, preferred_bin);
    let existing = fs::read_to_string(rc_path).unwrap_or_default();
    if let Some(start_idx) = existing.find(SHELL_HOOK_MARKER_START) {
        if let Some(end_rel) = existing[start_idx..].find(SHELL_HOOK_MARKER_END) {
            let end_idx = start_idx + end_rel + SHELL_HOOK_MARKER_END.len();
            let mut updated = String::new();
            updated.push_str(&existing[..start_idx]);
            if !updated.is_empty() && !updated.ends_with('\n') {
                updated.push('\n');
            }
            updated.push_str(&block);
            if end_idx < existing.len() {
                let tail = existing[end_idx..].trim_start_matches('\n');
                if !tail.is_empty() {
                    updated.push('\n');
                    updated.push_str(tail);
                    if !updated.ends_with('\n') {
                        updated.push('\n');
                    }
                }
            }
            if updated == existing {
                return Ok("already present".to_string());
            }
            fs::write(rc_path, updated).map_err(|e| {
                format!(
                    "failed writing shell rc file '{}': {}",
                    rc_path.display(),
                    e
                )
            })?;
            return Ok("updated".to_string());
        }
    }
    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    if !updated.is_empty() {
        updated.push('\n');
    }
    updated.push_str(&block);
    fs::write(rc_path, updated).map_err(|e| {
        format!(
            "failed writing shell rc file '{}': {}",
            rc_path.display(),
            e
        )
    })?;
    Ok("installed".to_string())
}

fn install_default_shell_hooks(cwd: &Path) -> Result<Vec<String>, String> {
    let repo = find_repo_root().unwrap_or_else(|| cwd.to_path_buf());
    let preferred_bin = if let Ok(exe) = env::current_exe() {
        if exe.exists() {
            exe
        } else {
            repo.join("retrivio")
        }
    } else {
        repo.join("retrivio")
    };
    let home = env::var("HOME")
        .map(PathBuf::from)
        .map_err(|_| "HOME is not set; cannot install shell hooks".to_string())?;
    let rc_files = [("bash", home.join(".bashrc")), ("zsh", home.join(".zshrc"))];
    let mut messages: Vec<String> = Vec::new();
    for (shell, rc) in rc_files {
        match ensure_shell_hook_in_rc_file(&rc, shell, &preferred_bin) {
            Ok(state) => messages.push(format!("{} -> {}", state, rc.display())),
            Err(err) => messages.push(format!("failed -> {} ({})", rc.display(), err)),
        }
    }
    messages.push(
        "open a new shell (or source your rc file) to activate Retrivio shell integration"
            .to_string(),
    );
    Ok(messages)
}

fn resolve_command_path(name: &str) -> Option<PathBuf> {
    let out = run_shell_capture(&format!("command -v {}", shell_escape(name))).ok()?;
    if out.exit_code != 0 {
        return None;
    }
    let line = out.stdout.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let path = PathBuf::from(line);
    if !path.is_absolute() || !is_executable_file(&path) {
        return None;
    }
    Some(path)
}

fn current_platform_tokens() -> (Vec<&'static str>, Vec<&'static str>) {
    let os_tokens = match env::consts::OS {
        "macos" => vec!["macos", "darwin", "osx"],
        "linux" => vec!["linux"],
        other => vec![other],
    };
    let arch_tokens = match env::consts::ARCH {
        "aarch64" => vec!["aarch64", "arm64"],
        "x86_64" => vec!["x86_64", "amd64"],
        other => vec![other],
    };
    (os_tokens, arch_tokens)
}

fn try_install_fswatch_with_homebrew() -> Result<bool, String> {
    if command_exists("fswatch") {
        return Ok(true);
    }
    if !command_exists("brew") {
        return Ok(false);
    }
    println!("install: attempting Homebrew install for fswatch...");
    let out = run_shell_capture("brew install fswatch >/dev/null 2>&1")?;
    if out.exit_code != 0 {
        return Ok(false);
    }
    Ok(command_exists("fswatch"))
}

fn run_add(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio add <path> [path ...] [--exclude <pattern>] [--refresh|--no-refresh]"
        );
        println!("  --exclude <pattern>  relative directory to exclude from indexing (repeatable)");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut refresh: Option<bool> = None; // None = ask user
    let mut inputs: Vec<String> = Vec::new();
    let mut excludes: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--refresh" => refresh = Some(true),
            "--no-refresh" => refresh = Some(false),
            "--exclude" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --exclude requires an argument");
                    process::exit(2);
                }
                excludes.push(args[i].to_string_lossy().to_string());
            }
            x if x.starts_with("--exclude=") => {
                excludes.push(x.trim_start_matches("--exclude=").to_string());
            }
            x if x.starts_with('-') => {
                eprintln!("error: unknown option '{}'", x);
                process::exit(2);
            }
            _ => inputs.push(s),
        }
        i += 1;
    }

    if inputs.is_empty() {
        eprintln!("error: no paths provided");
        process::exit(2);
    }

    let conn = open_db_rw(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to open database: {}", e);
        process::exit(1);
    });

    let mut added: Vec<PathBuf> = Vec::new();
    for raw in inputs {
        let path = normalize_path(&raw);
        if !path.is_dir() {
            eprintln!("skip (not a directory): {}", path.display());
            continue;
        }
        ensure_tracked_root_conn(&conn, &path, now_ts()).unwrap_or_else(|e| {
            eprintln!("error: failed to add root '{}': {}", path.display(), e);
            process::exit(1);
        });
        if !excludes.is_empty() {
            add_exclude_patterns_conn(&conn, &path, &excludes).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed to set excludes for '{}': {}",
                    path.display(),
                    e
                );
                process::exit(1);
            });
        }
        added.push(path);
    }

    if added.is_empty() {
        eprintln!("error: no valid directories were added.");
        process::exit(1);
    }

    println!("tracked roots added:");
    for p in &added {
        let root_excludes = get_exclude_patterns_conn(&conn, p).unwrap_or_default();
        if root_excludes.is_empty() {
            println!("- {}", p.display());
        } else {
            println!("- {} (excludes: {})", p.display(), root_excludes.join(", "));
        }
    }

    let do_refresh = match refresh {
        Some(v) => v,
        None => {
            if std::io::stdin().is_terminal() {
                prompt_yes_no("index now?", true).unwrap_or(true)
            } else {
                true
            }
        }
    };
    if do_refresh {
        let refresh_was_explicit = refresh.is_some();
        if cfg.embed_backend == "ollama" {
            match ensure_ollama_ready_for_add_refresh(&cfg) {
                Ok(()) => {}
                Err(e) => {
                    if refresh_was_explicit {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                    eprintln!("warning: skipping initial index because Ollama is not ready: {}", e);
                    eprintln!(
                        "note: tracked roots were added. run `retrivio setup`, start Ollama, or change embed_backend; then run `retrivio index`."
                    );
                    return;
                }
            }
        }
        let force_paths: HashSet<PathBuf> = added.iter().cloned().collect();
        run_index_with_strategy(
            &cwd,
            &cfg,
            Some(added.clone()),
            true,
            Some(force_paths),
            false,
            "add refresh",
        )
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        println!("note: run `retrivio index` when ready.");
    }
}

fn run_del(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio del <path> [path ...] [--refresh|--no-refresh]");
        println!("default: --no-refresh (fast remove); use --refresh to compact index immediately");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let db_path = db_path(&cwd);
    let _cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let mut refresh = false;
    let mut inputs: Vec<String> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        match s.as_str() {
            "--refresh" => refresh = true,
            "--no-refresh" => refresh = false,
            x if x.starts_with('-') => {
                eprintln!("error: unknown option '{}'", x);
                process::exit(2);
            }
            _ => inputs.push(s),
        }
    }

    if inputs.is_empty() {
        eprintln!("error: no paths provided");
        process::exit(2);
    }

    let mut removed = 0i64;
    for raw in inputs {
        let path = normalize_path(&raw);
        removed += remove_tracked_root(&db_path, &path).unwrap_or_else(|e| {
            eprintln!("error: failed to remove root '{}': {}", path.display(), e);
            process::exit(1);
        });
    }
    println!("tracked roots removed: {}", removed);

    if !refresh && removed > 0 && std::io::stdin().is_terminal() {
        match prompt_yes_no("refresh index now?", true) {
            Ok(true) => refresh = true,
            Ok(false) => {}
            Err(_) => {}
        }
    }
    if refresh {
        run_index_with_strategy(&cwd, &_cfg, None, false, None, true, "delete refresh")
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
    } else if removed > 0 {
        println!("note: run `retrivio refresh` when convenient.");
    }
}

fn run_roots(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio roots [add <path>|del <path>]");
        println!("lists all tracked roots and their exclude patterns");
        println!("  retrivio roots add <path>  — same as `retrivio add <path>`");
        println!("  retrivio roots del <path>  — same as `retrivio del <path>`");
        return;
    }
    // Delegate subcommands so "retrivio roots add/del" works intuitively.
    if let Some(sub) = args.first().map(|a| a.to_string_lossy().to_string()) {
        match sub.as_str() {
            "add" => {
                run_add(&args[1..]);
                return;
            }
            "del" | "rm" | "remove" => {
                run_del(&args[1..]);
                return;
            }
            _ => {}
        }
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let db_path = db_path(&cwd);

    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });

    let conn = open_db_rw(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to open database: {}", e);
        process::exit(1);
    });

    let rows = list_tracked_roots_full_conn(&conn).unwrap_or_else(|e| {
        eprintln!("error: failed to list tracked roots: {}", e);
        process::exit(1);
    });

    println!("tracked roots: {}", rows.len());
    for r in &rows {
        println!("- {}", r.path.display());
        for excl in &r.exclude_patterns {
            println!("    exclude: {}", excl);
        }
    }
}

fn run_exclude_cmd(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio exclude <root> <pattern> [pattern ...]");
        println!("adds exclude patterns to a tracked root");
        println!("excluded directories are skipped during project discovery and indexing");
        println!();
        println!("example: retrivio exclude ~/projects node_modules .cache dist");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let db_path = db_path(&cwd);
    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });
    let conn = open_db_rw(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to open database: {}", e);
        process::exit(1);
    });

    let root_path = normalize_path(&args[0].to_string_lossy());
    let patterns: Vec<String> = args[1..]
        .iter()
        .map(|a| a.to_string_lossy().to_string())
        .collect();
    if patterns.is_empty() {
        eprintln!("error: no exclude patterns provided");
        process::exit(2);
    }

    // Verify the root is tracked
    let roots = list_tracked_roots_full_conn(&conn).unwrap_or_default();
    if !roots.iter().any(|r| r.path == root_path) {
        eprintln!(
            "error: '{}' is not a tracked root. Add it first with `retrivio add`.",
            root_path.display()
        );
        process::exit(1);
    }

    add_exclude_patterns_conn(&conn, &root_path, &patterns).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let current = get_exclude_patterns_conn(&conn, &root_path).unwrap_or_default();
    println!("excludes for {}:", root_path.display());
    for p in &current {
        println!("  - {}", p);
    }
    println!("note: run `retrivio refresh` to re-index with updated excludes.");
}

fn run_include_cmd(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio include <root> <pattern> [pattern ...]");
        println!("removes exclude patterns from a tracked root (re-includes them)");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let db_path = db_path(&cwd);
    ensure_db_schema(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to initialize database: {}", e);
        process::exit(1);
    });
    let conn = open_db_rw(&db_path).unwrap_or_else(|e| {
        eprintln!("error: failed to open database: {}", e);
        process::exit(1);
    });

    let root_path = normalize_path(&args[0].to_string_lossy());
    let patterns: Vec<String> = args[1..]
        .iter()
        .map(|a| a.to_string_lossy().to_string())
        .collect();
    if patterns.is_empty() {
        eprintln!("error: no patterns to remove");
        process::exit(2);
    }

    let roots = list_tracked_roots_full_conn(&conn).unwrap_or_default();
    if !roots.iter().any(|r| r.path == root_path) {
        eprintln!("error: '{}' is not a tracked root.", root_path.display());
        process::exit(1);
    }

    remove_exclude_patterns_conn(&conn, &root_path, &patterns).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let current = get_exclude_patterns_conn(&conn, &root_path).unwrap_or_default();
    if current.is_empty() {
        println!("no excludes remaining for {}", root_path.display());
    } else {
        println!("excludes for {}:", root_path.display());
        for p in &current {
            println!("  - {}", p);
        }
    }
    println!("note: run `retrivio refresh` to re-index with updated excludes.");
}

fn run_index_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio index");
        return;
    }
    if !args.is_empty() {
        let other = args[0].to_string_lossy();
        eprintln!("error: unknown argument '{}'", other);
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    run_index_with_strategy(&cwd, &cfg, None, false, None, true, "index").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

fn run_refresh_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio refresh [path ...]");
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));

    let mut scoped: Vec<PathBuf> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        if s.starts_with('-') {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        }
        let p = normalize_path(&s);
        if !p.is_dir() {
            eprintln!("skip (not a directory): {}", p.display());
            continue;
        }
        scoped.push(p);
    }

    let (scope_roots, force_paths, remove_missing) = if scoped.is_empty() {
        (None, None, true)
    } else {
        let force_set: HashSet<PathBuf> = scoped.iter().cloned().collect();
        (Some(scoped.clone()), Some(force_set), false)
    };

    run_index_with_strategy(
        &cwd,
        &cfg,
        scope_roots,
        true,
        force_paths,
        remove_missing,
        "refresh",
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

fn run_reembed_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio reembed");
        println!(
            "forces a full embedding rebuild for the current model and rebuilds LanceDB vector index."
        );
        return;
    }
    if !args.is_empty() {
        let other = args[0].to_string_lossy();
        eprintln!("error: unknown argument '{}'", other);
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "reembed").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "reembed").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let mut stats = run_native_index(
        &cwd,
        &cfg,
        None,
        true,
        HashSet::new(),
        true,
        true,
        "reembed",
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let model_key = model_key_for_cfg(&cfg);
    // Rebuild LanceDB from the freshly-embedded vectors in SQLite
    let lance_path = data_dir(&cwd).join("lance");
    let t_sync2 = Instant::now();
    match lance_store::rebuild_from_sqlite(&conn, &model_key, &lance_path) {
        Ok(rebuilt_store) => {
            let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
            let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
            if let Ok(n) = lance_store::count(&rebuilt_store) {
                stats.retrieval_synced_chunks += n as i64;
            }
            *guard = Some(rebuilt_store);
        }
        Err(e) => {
            eprintln!("error: post-reembed LanceDB rebuild failed: {}", e);
            eprintln!("hint: embeddings are saved in sqlite; re-run `retrivio reembed` to retry");
            process::exit(1);
        }
    }
    let sync2_ms = t_sync2.elapsed().as_millis() as u64;
    stats.elapsed_sync_ms += sync2_ms;
    stats.elapsed_total_ms += sync2_ms;
    mark_reembed_completed(&conn, &model_key).unwrap_or_else(|e| {
        eprintln!("error: failed finalizing reembed state: {}", e);
        process::exit(1);
    });

    print_index_stats(&stats, &cfg);
    println!("reembed: complete (model={})", model_key);
}

fn run_search_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio search [--view projects|files] [--limit <n>] <query...>");
        return;
    }

    let mut limit: usize = 20;
    let mut view = "projects".to_string();
    let mut query_parts: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--limit" {
            i += 1;
            let v = arg_value(args, i, "--limit");
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
            i += 1;
            continue;
        }
        if let Some(v) = s.strip_prefix("--limit=") {
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
            i += 1;
            continue;
        }
        if s == "--view" {
            i += 1;
            view = arg_value(args, i, "--view").to_lowercase();
            i += 1;
            continue;
        }
        if let Some(v) = s.strip_prefix("--view=") {
            view = v.to_lowercase();
            i += 1;
            continue;
        }
        if s.starts_with('-') {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        }
        query_parts.push(s);
        i += 1;
    }

    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }
    if query_parts.is_empty() {
        eprintln!("error: query is empty");
        process::exit(2);
    }
    if limit == 0 {
        limit = 1;
    }

    let query = query_parts.join(" ");
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    ensure_native_embed_backend(&cfg, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `retrivio init --embed-backend <ollama|bedrock>`, or run `retrivio legacy search ...`"
        );
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if view == "files" {
        let rows = rank_files_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            println!("No file results found.");
            return;
        }
        print_file_results(&rows);
        return;
    }

    let rows = rank_projects_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    if rows.is_empty() {
        println!("No results found.");
        return;
    }
    print_project_results(&rows);
}

struct FswatchStream {
    child: std::process::Child,
    rx: Receiver<PathBuf>,
}

impl Drop for FswatchStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_fswatch_stream(roots: &[PathBuf]) -> Result<FswatchStream, String> {
    if roots.is_empty() {
        return Err("no watch roots configured".to_string());
    }
    let mut cmd = Command::new("fswatch");
    cmd.arg("-0").arg("-r").arg("--latency").arg("0.2");
    for root in roots {
        cmd.arg(root);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed launching fswatch: {}", e))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed acquiring fswatch stdout".to_string())?;
    let (tx, rx) = mpsc::channel::<PathBuf>();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<u8> = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(0, &mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    while matches!(buf.last(), Some(0 | b'\n' | b'\r')) {
                        buf.pop();
                    }
                    if buf.is_empty() {
                        continue;
                    }
                    let raw = String::from_utf8_lossy(&buf).trim().to_string();
                    if raw.is_empty() {
                        continue;
                    }
                    if tx.send(PathBuf::from(raw)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Ok(FswatchStream { child, rx })
}

fn watch_stats_changed(stats: &IndexStats) -> bool {
    stats.updated_projects > 0
        || stats.removed_projects > 0
        || stats.vectorized_projects > 0
        || stats.chunk_vectors > 0
        || stats.vector_failures > 0
}

fn print_watch_tick(label: &str, stats: &IndexStats, quiet: bool) {
    if !watch_stats_changed(stats) && quiet {
        return;
    }
    let now = chrono_like_now();
    println!(
        "[{}] {} updated={} removed={} vectorized={} chunk_vectors={} skipped={}",
        now,
        label,
        stats.updated_projects,
        stats.removed_projects,
        stats.vectorized_projects,
        stats.chunk_vectors,
        stats.skipped_projects
    );
    if stats.vector_failures > 0 {
        println!(
            "[{}] {} vector_failures={}",
            now, label, stats.vector_failures
        );
    }
}

fn path_depth(path: &Path) -> usize {
    path.components().count()
}

fn longest_prefix_match<'a>(path: &Path, candidates: &'a [PathBuf]) -> Option<&'a PathBuf> {
    let mut best: Option<&PathBuf> = None;
    let mut best_depth = 0usize;
    for candidate in candidates {
        if path == candidate || path.starts_with(candidate) {
            let d = path_depth(candidate);
            if d >= best_depth {
                best_depth = d;
                best = Some(candidate);
            }
        }
    }
    best
}

fn normalize_watch_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return normalize_lexical(path);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    normalize_lexical(&cwd.join(path))
}

fn watch_path_relevant(path: &Path) -> bool {
    for comp in path.components() {
        if let Component::Normal(name) = comp {
            let seg = name.to_string_lossy();
            if seg.starts_with('.') || is_skip_dir(&seg) {
                return false;
            }
        }
    }
    let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    let ext = Path::new(name)
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext.is_empty() {
        return true;
    }
    is_indexable_suffix(&format!(".{}", ext))
}

fn derive_watch_targets(
    pending_paths: &HashSet<PathBuf>,
    tracked_roots: &[TrackedRoot],
) -> (Vec<PathBuf>, HashSet<PathBuf>) {
    if pending_paths.is_empty() || tracked_roots.is_empty() {
        return (Vec::new(), HashSet::new());
    }
    let known_projects = discover_projects(tracked_roots);
    let root_paths: Vec<PathBuf> = tracked_roots.iter().map(|r| r.path.clone()).collect();
    let mut scope_set: HashSet<String> = HashSet::new();
    let mut force_set: HashSet<String> = HashSet::new();

    for raw in pending_paths {
        let path = normalize_watch_path(raw);
        if !watch_path_relevant(&path) {
            continue;
        }
        let Some(root) = longest_prefix_match(&path, &root_paths) else {
            continue;
        };
        if let Some(project) = longest_prefix_match(&path, &known_projects) {
            let key = project.to_string_lossy().to_string();
            scope_set.insert(key.clone());
            force_set.insert(key);
        } else {
            scope_set.insert(root.to_string_lossy().to_string());
        }
    }

    let mut scope_roots: Vec<PathBuf> = scope_set.into_iter().map(|v| normalize_path(&v)).collect();
    scope_roots.sort();
    let force_paths: HashSet<PathBuf> = force_set.into_iter().map(|v| normalize_path(&v)).collect();
    (scope_roots, force_paths)
}

fn run_watch_event_loop(
    cwd: &Path,
    cfg: &ConfigValues,
    tracked_roots: &[TrackedRoot],
    interval_seconds: f64,
    debounce_ms: u64,
    quiet: bool,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, "watch event-loop")?;
    let root_paths: Vec<PathBuf> = tracked_roots.iter().map(|r| r.path.clone()).collect();
    let mut stream = start_fswatch_stream(&root_paths)?;
    if !quiet {
        println!(
            "watch mode: fswatch event stream (debounce={}ms, full sweep every {:.1}s)",
            debounce_ms, interval_seconds
        );
    }
    let debounce = Duration::from_millis(debounce_ms.max(100));
    let sweep_every = Duration::from_secs_f64(interval_seconds.max(1.0));
    let mut pending_paths: HashSet<PathBuf> = HashSet::new();
    let mut last_event_at: Option<Instant> = None;
    let mut last_sweep_at = Instant::now();

    loop {
        match stream.rx.recv_timeout(Duration::from_millis(200)) {
            Ok(path) => {
                pending_paths.insert(path);
                last_event_at = Some(Instant::now());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err("fswatch stream disconnected".to_string());
            }
        }
        if let Some(status) = stream
            .child
            .try_wait()
            .map_err(|e| format!("failed checking fswatch status: {}", e))?
        {
            return Err(format!("fswatch exited unexpectedly: {}", status));
        }

        if !pending_paths.is_empty()
            && last_event_at
                .map(|t| t.elapsed() >= debounce)
                .unwrap_or(false)
        {
            let (scope_roots, force_paths) = derive_watch_targets(&pending_paths, tracked_roots);
            pending_paths.clear();
            last_event_at = None;
            if scope_roots.is_empty() {
                continue;
            }
            ensure_retrieval_backend_ready(cfg, true, "watch events")?;
            let stats = run_native_index(
                cwd,
                cfg,
                Some(scope_roots),
                false,
                force_paths,
                false,
                !quiet,
                "watch events",
            )?;
            print_watch_tick("event", &stats, quiet);
        }

        if last_sweep_at.elapsed() >= sweep_every {
            ensure_retrieval_backend_ready(cfg, true, "watch sweep")?;
            let stats = run_native_index(
                cwd,
                cfg,
                None,
                false,
                HashSet::new(),
                true,
                !quiet,
                "watch sweep",
            )?;
            print_watch_tick("sweep", &stats, quiet);
            last_sweep_at = Instant::now();
        }
    }
}

fn run_watch_polling_loop(cwd: &Path, cfg: &ConfigValues, interval_seconds: f64, quiet: bool) {
    if !quiet {
        println!("watch mode: polling");
    }
    loop {
        ensure_retrieval_backend_ready(cfg, true, "watch poll").unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        let stats = run_native_index(
            cwd,
            cfg,
            None,
            false,
            HashSet::new(),
            true,
            !quiet,
            "watch poll",
        )
        .unwrap_or_else(|e| {
            eprintln!("error: watch poll failed: {}", e);
            process::exit(1);
        });
        print_watch_tick("poll", &stats, quiet);
        thread::sleep(Duration::from_secs_f64(interval_seconds));
    }
}

fn run_watch_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]"
        );
        return;
    }

    let mut interval_seconds = 30.0f64;
    let mut debounce_ms: u64 = 900;
    let mut once = false;
    let mut quiet = false;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--once" => {
                once = true;
            }
            "--quiet" => {
                quiet = true;
            }
            "--interval" => {
                i += 1;
                let raw = arg_value(args, i, "--interval");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            "--debounce-ms" => {
                i += 1;
                let raw = arg_value(args, i, "--debounce-ms");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--interval=") => {
                let raw = other.trim_start_matches("--interval=");
                interval_seconds = raw.parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("error: --interval must be a number");
                    process::exit(2);
                });
            }
            other if other.starts_with("--debounce-ms=") => {
                let raw = other.trim_start_matches("--debounce-ms=");
                debounce_ms = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --debounce-ms must be an integer");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    interval_seconds = interval_seconds.max(1.0);
    debounce_ms = debounce_ms.clamp(100, 5000);

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_native_embed_backend(&cfg, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `retrivio init --embed-backend <ollama|bedrock>`, or run `retrivio legacy watch ...`"
        );
        process::exit(1);
    });
    ensure_retrieval_backend_ready(&cfg, true, "watch").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let tracked_roots = resolve_roots(&conn, &cfg, None).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if !quiet {
        println!(
            "watching tracked roots (config root: {}, tracked={})",
            cfg.root.display(),
            tracked_roots.len()
        );
        println!(
            "interval: {:.1}s  debounce: {}ms",
            interval_seconds, debounce_ms
        );
        println!("press Ctrl-C to stop");
    }

    let bootstrap = run_native_index(
        &cwd,
        &cfg,
        None,
        false,
        HashSet::new(),
        true,
        !quiet,
        "watch bootstrap",
    )
    .unwrap_or_else(|e| {
        eprintln!("error: watch bootstrap failed: {}", e);
        process::exit(1);
    });
    print_watch_tick("bootstrap", &bootstrap, quiet);
    if once {
        return;
    }

    if command_exists("fswatch") {
        match run_watch_event_loop(
            &cwd,
            &cfg,
            &tracked_roots,
            interval_seconds,
            debounce_ms,
            quiet,
        ) {
            Ok(()) => return,
            Err(e) => {
                if !quiet {
                    eprintln!("watch: fswatch mode failed ({}); using polling fallback", e);
                }
            }
        }
    }
    if !quiet {
        println!("watch: fswatch unavailable; using polling fallback");
    }
    run_watch_polling_loop(&cwd, &cfg, interval_seconds, quiet);
}

fn chrono_like_now() -> String {
    let out = run_shell_capture("date '+%Y-%m-%d %H:%M:%S'");
    match out {
        Ok(v) if v.exit_code == 0 && !v.stdout.trim().is_empty() => v.stdout.trim().to_string(),
        _ => "now".to_string(),
    }
}

fn run_jump_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio jump [--files|--dirs] [--view projects|files] [--limit <n>] [--emit-path-file <path>] [query...]"
        );
        println!("prints selected path to stdout (intended for shell wrappers to cd/open).");
        println!("picker keys: Tab=toggle dir/file, Ctrl-D=dirs, Ctrl-F=files, Ctrl-U=clear query");
        return;
    }

    let mut view = "projects".to_string();
    let mut limit: usize = 40;
    let mut emit_path_file = String::new();
    let mut query_parts: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--files" | "-f" => {
                view = "files".to_string();
            }
            "--dirs" | "-d" | "--directories" => {
                view = "projects".to_string();
            }
            "--view" => {
                i += 1;
                view = arg_value(args, i, "--view").to_lowercase();
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--emit-path-file" => {
                i += 1;
                emit_path_file = arg_value(args, i, "--emit-path-file");
            }
            x if x.starts_with("--view=") => {
                view = x.trim_start_matches("--view=").to_lowercase();
            }
            x if x.starts_with("--limit=") => {
                let raw = x.trim_start_matches("--limit=");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            x if x.starts_with("--emit-path-file=") => {
                emit_path_file = x.trim_start_matches("--emit-path-file=").to_string();
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
            other => {
                query_parts.push(other.to_string());
            }
        }
        i += 1;
    }

    if limit == 0 {
        limit = 1;
    }
    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }

    let query = if !query_parts.is_empty() {
        query_parts.join(" ")
    } else {
        if !std::io::stdin().is_terminal() {
            eprintln!("usage: retrivio jump [--files|--dirs] [query...]");
            process::exit(2);
        }
        match prompt_line_stderr("retrivio query: ") {
            Ok(v) => v.trim().to_string(),
            Err(e) => {
                if e.to_ascii_lowercase().contains("interrupted") {
                    process::exit(130);
                }
                eprintln!("error: {}", e);
                process::exit(1);
            }
        }
    };
    if query.is_empty() {
        process::exit(130);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "jump").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "jump").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `retrivio init --embed-backend <ollama|bedrock>`, or run `retrivio legacy jump ...`"
        );
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    let mut active_view = view;
    let mut active_query = query;
    let mut is_first_run = true;
    loop {
        let candidates = if active_view == "files" {
            match rank_files_native(&conn, &cfg, &active_query, limit) {
                Err(e) => {
                    if is_first_run {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                    Vec::new()
                }
                Ok(rows) => {
                    if rows.is_empty() && is_first_run {
                        eprintln!("error: no file matches found.");
                        process::exit(1);
                    }
                    rows.into_iter()
                        .map(|item| make_file_pick_candidate(&item))
                        .collect::<Vec<_>>()
                }
            }
        } else {
            match rank_projects_native(&conn, &cfg, &active_query, limit) {
                Err(e) => {
                    if is_first_run {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                    Vec::new()
                }
                Ok(rows) => {
                    if rows.is_empty() && is_first_run {
                        eprintln!("error: no indexed projects found. run `retrivio index` first.");
                        process::exit(1);
                    }
                    rows.into_iter()
                        .map(|item| make_project_pick_candidate(&item))
                        .collect::<Vec<_>>()
                }
            }
        };
        is_first_run = false;

        if candidates.is_empty() {
            let empty_view = active_view.clone();
            active_view = if empty_view == "files" {
                "projects".to_string()
            } else {
                "files".to_string()
            };
            eprintln!(
                "no {} matches for '{}'; switching to {} mode",
                empty_view, active_query, active_view
            );
            continue;
        }

        let action =
            pick_candidate_path(&candidates, &active_query, &active_view).unwrap_or_else(|e| {
                eprintln!("error: picker failed: {}", e);
                process::exit(1);
            });
        match action {
            PickAction::Cancel => process::exit(130),
            PickAction::Toggle { view, query } => {
                active_view = view;
                active_query = query;
            }
            PickAction::Selected {
                path: selected_path,
                query,
            } => {
                let record_query = if query.trim().is_empty() {
                    active_query.clone()
                } else {
                    query
                };
                record_selection_event(&conn, &record_query, &selected_path, now_ts())
                    .unwrap_or_else(|e| {
                        eprintln!("warning: failed to record selection event: {}", e);
                    });
                maybe_write_emit_path(&cwd, &emit_path_file, &selected_path).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
                if emit_path_file.trim().is_empty() {
                    println!("{}", selected_path);
                }
                return;
            }
        }
    }
}

fn maybe_write_emit_path(
    cwd: &Path,
    emit_path_file: &str,
    selected_path: &str,
) -> Result<(), String> {
    if emit_path_file.trim().is_empty() {
        return Ok(());
    }
    let mut out_path = expand_tilde(emit_path_file);
    if !out_path.is_absolute() {
        out_path = cwd.join(out_path);
    }
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create emit path parent '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    fs::write(&out_path, format!("{}\n", selected_path)).map_err(|e| {
        format!(
            "failed writing emit path file '{}': {}",
            out_path.display(),
            e
        )
    })?;
    Ok(())
}

fn run_pick_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
        );
        return;
    }

    let mut query = String::new();
    let mut view = "projects".to_string();
    let mut limit: usize = 30;
    let mut emit_path_file = String::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--query" => {
                i += 1;
                query = arg_value(args, i, "--query");
            }
            "--view" => {
                i += 1;
                view = arg_value(args, i, "--view").to_lowercase();
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--emit-path-file" => {
                i += 1;
                emit_path_file = arg_value(args, i, "--emit-path-file");
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    if limit == 0 {
        limit = 1;
    }
    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!(
            "hint: use `retrivio init --embed-backend <ollama|bedrock>`, or run `retrivio legacy pick ...`"
        );
        process::exit(1);
    });
    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let candidates = if view == "files" {
        let rows = rank_files_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            eprintln!("error: no file matches found.");
            process::exit(1);
        }
        rows.into_iter()
            .map(|item| make_file_pick_candidate(&item))
            .collect::<Vec<_>>()
    } else {
        let rows = rank_projects_native(&conn, &cfg, &query, limit).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if rows.is_empty() {
            eprintln!("error: no indexed projects found. run `retrivio index` first.");
            process::exit(1);
        }
        rows.into_iter()
            .map(|item| make_project_pick_candidate(&item))
            .collect::<Vec<_>>()
    };

    let selected = pick_candidate_path(&candidates, &query, &view).unwrap_or_else(|e| {
        eprintln!("error: picker failed: {}", e);
        process::exit(1);
    });
    let selected_path = match selected {
        PickAction::Selected { path, .. } => path,
        PickAction::Toggle { .. } | PickAction::Cancel => {
            process::exit(1);
        }
    };
    if selected_path.is_empty() {
        process::exit(1);
    }

    if !emit_path_file.trim().is_empty() {
        let mut out_path = expand_tilde(&emit_path_file);
        if !out_path.is_absolute() {
            out_path = cwd.join(out_path);
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed to create emit path parent '{}': {}",
                    parent.display(),
                    e
                );
                process::exit(1);
            });
        }
        fs::write(&out_path, format!("{}\n", selected_path)).unwrap_or_else(|e| {
            eprintln!(
                "error: failed writing emit path file '{}': {}",
                out_path.display(),
                e
            );
            process::exit(1);
        });
    }

    record_selection_event(&conn, &query, &selected_path, now_ts()).unwrap_or_else(|e| {
        eprintln!("warning: failed to record selection event: {}", e);
    });
    println!("{}", selected_path);
}

struct PickCandidate {
    path: String,
    display_path: String,
    score: String,
    kind: String,
    signals: String,
    preview: String,
}

fn pick_one_line(text: &str, max_len: usize) -> String {
    pick_trunc(&collapse_whitespace(text), max_len).replace('\t', " ")
}

fn pick_signals(pairs: &[(&str, f64)]) -> String {
    let parts: Vec<String> = pairs
        .iter()
        .filter(|(_, v)| *v > 0.0005)
        .map(|(label, v)| format!("{}:{}", label, pick_num3(*v)))
        .collect();
    if parts.is_empty() {
        "-".to_string()
    } else {
        parts.join(" ")
    }
}

fn make_project_pick_candidate(item: &RankedResult) -> PickCandidate {
    let raw = pick_trunc(&pick_home_short(&item.path), 58);
    let display_path = format!("{:<60}", raw);
    let score = pick_num3(item.score);
    let signals = pick_signals(&[
        ("s", item.semantic),
        ("l", item.lexical),
        ("f", item.frecency),
        ("g", item.graph),
    ]);
    let preview = if let Some(ev) = item.evidence.first() {
        format!(
            "top chunk: {}#{} score={} rel={} :: {}",
            ev.doc_rel_path,
            ev.chunk_index,
            pick_num3(ev.score),
            ev.relation,
            pick_one_line(&ev.excerpt, 240)
        )
    } else {
        "no chunk evidence available".to_string()
    };
    PickCandidate {
        path: item.path.clone(),
        display_path,
        score,
        kind: "dir".to_string(),
        signals,
        preview,
    }
}

fn make_file_pick_candidate(item: &RankedFileResult) -> PickCandidate {
    let file_label = if item.doc_rel_path.trim().is_empty() {
        path_basename(&item.path)
    } else {
        item.doc_rel_path.clone()
    };
    let raw = format!(
        "{} [{}]",
        pick_trunc(&file_label, 40),
        pick_trunc(&path_basename(&item.project_path), 16)
    );
    let display_path = format!("{:<60}", raw);
    let score = pick_num3(item.score);
    let sig_pairs: Vec<(&str, f64)> = vec![
        ("s", item.semantic),
        ("l", item.lexical),
        ("g", item.graph),
        ("q", item.quality),
    ];
    let rel_display = pick_trunc(&item.relation, 12);
    let signals = {
        let base = pick_signals(&sig_pairs);
        if item.relation.is_empty() || item.relation == "none" {
            base
        } else {
            format!("{} r:{}", base, rel_display)
        }
    };
    let mut preview = format!(
        "chunk #{} score={} :: {}",
        item.chunk_index,
        pick_num3(item.score),
        pick_one_line(&item.excerpt, 220)
    );
    if let Some(ev) = item.evidence.first() {
        preview.push_str(&format!(
            " | related {}#{} {} {}",
            pick_trunc(&ev.doc_rel_path, 32),
            ev.chunk_index,
            pick_num3(ev.score),
            pick_one_line(&ev.excerpt, 120)
        ));
    }
    PickCandidate {
        path: item.path.clone(),
        display_path,
        score,
        kind: "file".to_string(),
        signals,
        preview,
    }
}

enum PickAction {
    Selected { path: String, query: String },
    Toggle { view: String, query: String },
    Cancel,
}

fn pick_candidate_path(
    candidates: &[PickCandidate],
    query: &str,
    view: &str,
) -> Result<PickAction, String> {
    if candidates.is_empty() {
        return Ok(PickAction::Cancel);
    }
    if !std::io::stdin().is_terminal() {
        return Ok(PickAction::Selected {
            path: candidates[0].path.clone(),
            query: query.to_string(),
        });
    }
    if !command_exists("fzf") {
        eprintln!(
            "warning: `fzf` is not installed; interactive picker UI is unavailable. selecting the top-ranked {} match automatically.",
            if view == "files" { "file" } else { "directory" }
        );
        eprintln!(
            "hint: install `fzf` to enable interactive dir/file switching, previews, and manual selection."
        );
        return Ok(PickAction::Selected {
            path: candidates[0].path.clone(),
            query: query.to_string(),
        });
    }

    let mut payload = String::new();
    for item in candidates {
        let line = format!(
            "{:<60} {:>5}  {:<4} {}",
            item.display_path, item.score, item.kind, item.signals
        );
        payload.push_str(&format!("{}\t{}\t{}\n", item.path, line, item.preview));
    }
    let prompt = if view == "files" {
        "retrivio[file]> ".to_string()
    } else {
        "retrivio[dir]> ".to_string()
    };
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--delimiter=\t")
        // Keep display compact (field 2), but match query against both
        // display line + preview text (fields 2 and 3).
        .arg("--with-nth=2")
        .arg("--nth=2,3")
        .arg("--print-query")
        .arg("--expect=tab,ctrl-d,ctrl-f")
        .arg("--preview")
        .arg("echo {3}")
        .arg("--preview-window=down,6,wrap")
        .arg("--bind")
        .arg("ctrl-u:unix-line-discard")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg(if query.trim().is_empty() {
            "path | score | type | s=sem l=lex g=graph q=qual r=rel | Tab=toggle Ctrl-D=dirs Ctrl-F=files".to_string()
        } else {
            format!(
                "ranked for query='{}' | path | score | type | s=sem l=lex g=graph q=qual r=rel | Tab=toggle Ctrl-D=dirs Ctrl-F=files",
                query
            )
        })
        .arg("--no-sort")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch fzf: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| format!("failed writing picker input: {}", e))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed waiting for fzf: {}", e))?;
    let output = String::from_utf8_lossy(&out.stdout);
    let mut lines = output.lines();
    let query_line = lines.next().unwrap_or("").trim().to_string();
    let effective_query = if query_line.trim().is_empty() {
        query.to_string()
    } else {
        query_line.clone()
    };
    let second_line = lines.next().unwrap_or("").trim().to_string();
    let mut key_line = second_line.clone();
    let mut selected_line = String::new();
    if second_line.contains('\t') {
        // Some fzf versions omit an empty expect-key line for Enter.
        key_line.clear();
        selected_line = second_line;
    }
    // Check expected toggle keys BEFORE exit status — fzf may exit 1 (no match)
    // when the user presses an --expect key with no visible matches, but still
    // prints the key to stdout.
    if key_line == "tab" {
        let toggled = if view == "files" { "projects" } else { "files" };
        return Ok(PickAction::Toggle {
            view: toggled.to_string(),
            query: effective_query.clone(),
        });
    }
    if key_line == "ctrl-d" {
        return Ok(PickAction::Toggle {
            view: "projects".to_string(),
            query: effective_query.clone(),
        });
    }
    if key_line == "ctrl-f" {
        return Ok(PickAction::Toggle {
            view: "files".to_string(),
            query: effective_query.clone(),
        });
    }
    if !out.status.success() {
        return Ok(PickAction::Cancel);
    }
    if selected_line.is_empty() {
        selected_line = lines
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .to_string();
    }
    let line = selected_line;
    if line.trim().is_empty() {
        return Ok(PickAction::Cancel);
    }
    let path = line.split('\t').next().unwrap_or("").trim().to_string();
    if path.is_empty() {
        return Ok(PickAction::Cancel);
    }
    Ok(PickAction::Selected {
        path,
        query: effective_query,
    })
}

#[derive(Clone, Debug)]
struct BenchDatasetMeta {
    created_at: f64,
    source_db: String,
    model: String,
    dim: i64,
    chunks: i64,
}

fn run_bench_cmd(args: &[OsString]) {
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
    let mut backends: Vec<String> = Vec::new();
    let mut k: usize = 20;
    let mut repeats: usize = 10;
    let mut warmup: usize = 1;
    let mut rebuild = false;
    let mut dataset_limit: Option<usize> = None;
    let mut dataset_target: Option<usize> = None;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if !action_set
            && !s.starts_with('-')
            && matches!(s.as_str(), "plan" | "doctor" | "export" | "run")
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
            "--backend" => {
                i += 1;
                let backend = arg_value(args, i, "--backend").trim().to_lowercase();
                if !backend.is_empty() {
                    backends.push(backend);
                }
            }
            "--k" => {
                i += 1;
                let raw = arg_value(args, i, "--k");
                k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --k must be an integer");
                    process::exit(2);
                });
                k = k.max(1);
            }
            "--repeats" => {
                i += 1;
                let raw = arg_value(args, i, "--repeats");
                repeats = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --repeats must be an integer");
                    process::exit(2);
                });
                repeats = repeats.max(1);
            }
            "--warmup" => {
                i += 1;
                let raw = arg_value(args, i, "--warmup");
                warmup = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --warmup must be an integer");
                    process::exit(2);
                });
            }
            "--rebuild" => {
                rebuild = true;
            }
            "--dataset-limit" => {
                i += 1;
                let raw = arg_value(args, i, "--dataset-limit");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --dataset-limit must be an integer");
                    process::exit(2);
                });
                dataset_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            "--target-chunks" => {
                i += 1;
                let raw = arg_value(args, i, "--target-chunks");
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --target-chunks must be an integer");
                    process::exit(2);
                });
                dataset_target = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
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
            other if other.starts_with("--backend=") => {
                let backend = other.trim_start_matches("--backend=").trim().to_lowercase();
                if !backend.is_empty() {
                    backends.push(backend);
                }
            }
            other if other.starts_with("--k=") => {
                let raw = other.trim_start_matches("--k=").trim();
                k = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --k must be an integer");
                    process::exit(2);
                });
                k = k.max(1);
            }
            other if other.starts_with("--repeats=") => {
                let raw = other.trim_start_matches("--repeats=").trim();
                repeats = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --repeats must be an integer");
                    process::exit(2);
                });
                repeats = repeats.max(1);
            }
            other if other.starts_with("--warmup=") => {
                let raw = other.trim_start_matches("--warmup=").trim();
                warmup = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --warmup must be an integer");
                    process::exit(2);
                });
            }
            other if other.starts_with("--dataset-limit=") => {
                let raw = other.trim_start_matches("--dataset-limit=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --dataset-limit must be an integer");
                    process::exit(2);
                });
                dataset_limit = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
            }
            other if other.starts_with("--target-chunks=") => {
                let raw = other.trim_start_matches("--target-chunks=").trim();
                let parsed = raw.parse::<i64>().unwrap_or_else(|_| {
                    eprintln!("error: --target-chunks must be an integer");
                    process::exit(2);
                });
                dataset_target = if parsed <= 0 {
                    None
                } else {
                    Some(parsed as usize)
                };
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
            println!("  retrivio bench run --backend lancedb");
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
        "run" => {
            write_default_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if !dataset_path.exists() {
                println!("bench: dataset missing; exporting from sqlite index...");
                let dbp = db_path(&cwd);
                let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
                export_chunk_dataset_native(
                    &conn,
                    &dbp,
                    &model_key,
                    &dataset_path,
                    &meta_path,
                    None,
                )
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
            }

            let queries = load_bench_queries(&queries_path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if queries.is_empty() {
                eprintln!("error: no queries found in {}", queries_path.display());
                process::exit(1);
            }

            if backends.is_empty() {
                backends.push("lancedb".to_string());
            }
            if dataset_target.is_some() && !rebuild {
                println!("bench: target-chunks requested; forcing --rebuild for correctness");
                rebuild = true;
            }

            let report_path = bench_root.join("report.json");
            let report_md_path = bench_root.join("report.md");

            let dim = bench_dim_from_meta_or_db(&meta_path, &db_path(&cwd), &model_key, 384)
                .unwrap_or(384) as usize;
            let mut backend_results: serde_json::Map<String, Value> = serde_json::Map::new();
            for backend in &backends {
                let value = match backend.as_str() {
                    "falkordb" | "falkordblite" => Ok(serde_json::json!({
                        "status": "unsupported",
                        "error": "FalkorDB has been replaced by LanceDB"
                    })),
                    "lancedb" => Ok(serde_json::json!({
                        "status": "unsupported",
                        "error": "native LanceDB benchmark is not implemented yet"
                    })),
                    _ => Ok(serde_json::json!({"status": "unknown_backend"})),
                }
                .unwrap_or_else(|e: String| serde_json::json!({"status": "error", "error": e}));
                backend_results.insert(backend.clone(), value);
            }

            let report = serde_json::json!({
                "created_at": now_ts(),
                "model_key": model_key,
                "dataset_jsonl": dataset_path.to_string_lossy().to_string(),
                "dataset_meta": meta_path.to_string_lossy().to_string(),
                "queries_file": queries_path.to_string_lossy().to_string(),
                "query_count": queries.len(),
                "k": k,
                "repeats": repeats,
                "warmup": warmup,
                "dataset_limit": dataset_limit,
                "dataset_target": dataset_target,
                "rebuild": rebuild,
                "backends": Value::Object(backend_results.clone()),
            });
            let report_raw = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed to serialize report JSON: {}", e))
                .and_then(|v| {
                    fs::write(&report_path, format!("{}\n", v)).map_err(|e| {
                        format!("failed writing report '{}': {}", report_path.display(), e)
                    })
                });
            if let Err(e) = report_raw {
                eprintln!("error: {}", e);
                process::exit(1);
            }

            let mut lines: Vec<String> = Vec::new();
            lines.push("# Retrivio Bench Report".to_string());
            lines.push(String::new());
            lines.push(format!("- Dataset: `{}`", dataset_path.display()));
            lines.push(format!(
                "- Queries: `{}` ({} queries)",
                queries_path.display(),
                queries.len()
            ));
            lines.push(format!(
                "- k: {}, repeats: {}, warmup: {}",
                k, repeats, warmup
            ));
            if let Some(v) = dataset_limit {
                lines.push(format!("- dataset_limit: {}", v));
            }
            if let Some(v) = dataset_target {
                lines.push(format!("- target_chunks: {}", v));
            }
            lines.push(format!("- Backends: {}", backends.join(", ")));
            lines.push(String::new());
            for backend in &backends {
                let res = backend_results.get(backend).cloned().unwrap_or_else(
                    || serde_json::json!({"status": "error", "error": "missing backend result"}),
                );
                lines.push(format!("## {}", backend));
                lines.push(String::new());
                let status = res
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("error")
                    .to_string();
                lines.push(format!("- status: {}", status));
                if status != "ok" {
                    let err = res
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !err.is_empty() {
                        lines.push(format!("- error: `{}`", err));
                    }
                    lines.push(String::new());
                    continue;
                }
                let total = res
                    .get("total")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                lines.push(format!(
                    "- connect_s: {:.3}",
                    res.get("connect_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- ingest_s: {:.3}",
                    res.get("ingest_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- index_s: {:.3}",
                    res.get("index_s").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- total p50_ms: {:.2}",
                    total.get("p50_ms").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(format!(
                    "- total p95_ms: {:.2}",
                    total.get("p95_ms").and_then(Value::as_f64).unwrap_or(0.0)
                ));
                lines.push(String::new());
            }
            fs::write(&report_md_path, format!("{}\n", lines.join("\n"))).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed writing report markdown '{}': {}",
                    report_md_path.display(),
                    e
                );
                process::exit(1);
            });
            println!("report: {}", report_path.display());
            println!("report_md: {}", report_md_path.display());
        }
        other => {
            eprintln!("error: unknown bench action '{}'", other);
            process::exit(2);
        }
    }
}

fn print_bench_help() {
    println!("usage: retrivio bench [plan|doctor|export|run] [options]");
    println!("options:");
    println!("  --model-key <key>        Vector model key for dataset export");
    println!("  --limit <n>              Optional max rows during export (0=all)");
    println!("  --dataset <path>         Dataset JSONL path");
    println!("  --queries <path>         Queries file path");
    println!("  --backend <name>         Backend to benchmark (repeatable)");
    println!("  --k <n>                  Top-k per query (default: 20)");
    println!("  --repeats <n>            Measured repeats per query (default: 10)");
    println!("  --warmup <n>             Warmup repeats per query (default: 1)");
    println!("  --rebuild                Rebuild LanceDB vectors from SQLite before benchmark");
    println!("  --dataset-limit <n>      Included for compatibility (currently informational)");
    println!("  --target-chunks <n>      Included for compatibility (currently informational)");
}

fn default_bench_model_key(cfg: &ConfigValues) -> String {
    let backend = cfg.embed_backend.trim().to_lowercase();
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

fn bench_meta_path(dataset_path: &Path) -> PathBuf {
    dataset_path.with_extension("meta.json")
}

fn write_default_bench_queries(path: &Path) -> Result<(), String> {
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

fn load_bench_queries(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed reading queries file '{}': {}", path.display(), e))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let q = line.trim();
        if q.is_empty() || q.starts_with('#') {
            continue;
        }
        out.push(q.to_string());
    }
    Ok(out)
}

fn export_chunk_dataset_native(
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

fn bench_dim_from_meta_or_db(
    meta_path: &Path,
    db_path: &Path,
    model_key: &str,
    default_dim: i64,
) -> Option<i64> {
    if meta_path.exists() {
        if let Ok(raw) = fs::read_to_string(meta_path) {
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(dim) = value.get("dim").and_then(Value::as_i64) {
                    if dim > 0 {
                        return Some(dim);
                    }
                }
            }
        }
    }
    if db_path.exists() {
        if let Ok(conn) = open_db_read_only(db_path) {
            if let Ok(dim) = conn.query_row(
                "SELECT dim FROM project_chunk_vectors WHERE model = ?1 LIMIT 1",
                params![model_key],
                |row| row.get::<_, i64>(0),
            ) {
                if dim > 0 {
                    return Some(dim);
                }
            }
        }
    }
    Some(default_dim.max(1))
}

fn summarize_latencies_ms(values: &[f64]) -> Value {
    if values.is_empty() {
        return serde_json::json!({
            "count": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        });
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let count = sorted.len() as f64;
    let min_ms = *sorted.first().unwrap_or(&0.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let mean_ms = sorted.iter().sum::<f64>() / count.max(1.0);
    serde_json::json!({
        "count": count,
        "min_ms": min_ms,
        "p50_ms": percentile_ms(&sorted, 50.0),
        "p95_ms": percentile_ms(&sorted, 95.0),
        "max_ms": max_ms,
        "mean_ms": mean_ms,
    })
}

fn percentile_ms(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let k = ((sorted_values.len() - 1) as f64) * (p / 100.0);
    let floor = k.floor() as usize;
    let ceil = k.ceil() as usize;
    if floor == ceil {
        return sorted_values[floor];
    }
    let d0 = sorted_values[floor] * ((ceil as f64) - k);
    let d1 = sorted_values[ceil] * (k - (floor as f64));
    d0 + d1
}

fn api_pid_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.pid")
}

fn api_port_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.port")
}

fn api_log_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.log")
}

fn daemon_default_host() -> String {
    env::var("RETRIVIO_API_HOST")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

fn daemon_default_port() -> u16 {
    env::var("RETRIVIO_API_PORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(8765)
}

fn daemon_default_timeout() -> u64 {
    env::var("RETRIVIO_API_START_TIMEOUT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(8)
        .max(1)
}

fn read_saved_api_port(cwd: &Path) -> Option<u16> {
    let path = api_port_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u16>().ok()
}

fn read_saved_api_pid(cwd: &Path) -> Option<u32> {
    let path = api_pid_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u32>().ok()
}

fn persist_api_runtime_state(cwd: &Path, pid: u32, port: u16) -> Result<(), String> {
    let data = data_dir(cwd);
    fs::create_dir_all(&data)
        .map_err(|e| format!("failed creating data dir '{}': {}", data.display(), e))?;
    fs::write(api_pid_path(cwd), format!("{}\n", pid))
        .map_err(|e| format!("failed writing api pid file: {}", e))?;
    fs::write(api_port_path(cwd), format!("{}\n", port))
        .map_err(|e| format!("failed writing api port file: {}", e))?;
    Ok(())
}

fn clear_api_runtime_state(cwd: &Path) {
    let _ = fs::remove_file(api_pid_path(cwd));
    let _ = fs::remove_file(api_port_path(cwd));
}

fn api_health_host_port(host: &str, port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect((host, port)) else {
        return false;
    };
    let req = format!(
        "GET /health HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        host, port
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let Ok(n) = stream.read(&mut buf) else {
        return false;
    };
    if n == 0 {
        return false;
    }
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
}

fn port_available(host: &str, port: u16) -> bool {
    match TcpStream::connect((host, port)) {
        Ok(_) => false,
        Err(err) => {
            matches!(
                err.kind(),
                std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::AddrNotAvailable
                    | std::io::ErrorKind::PermissionDenied
                    | std::io::ErrorKind::NotConnected
                    | std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::Other
            )
        }
    }
}

fn find_free_port(host: &str, start: u16, span: usize) -> Option<u16> {
    for offset in 0..span.max(1) {
        let port = start.saturating_add(offset as u16);
        if port == 0 {
            continue;
        }
        if port_available(host, port) {
            return Some(port);
        }
    }
    None
}

fn tail_file_lines(path: &Path, lines: usize) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed reading log '{}': {}", path.display(), e))?;
    let items: Vec<&str> = raw.lines().collect();
    let start = items.len().saturating_sub(lines.max(1));
    for line in &items[start..] {
        println!("{}", line);
    }
    Ok(())
}

fn spawn_api_daemon(cwd: &Path, host: &str, port: u16) -> Result<u32, String> {
    let log_path = api_log_path(cwd);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating api log dir '{}': {}", parent.display(), e))?;
    }
    let log_out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("failed opening api log '{}': {}", log_path.display(), e))?;
    let log_err = log_out.try_clone().map_err(|e| {
        format!(
            "failed cloning api log handle '{}': {}",
            log_path.display(),
            e
        )
    })?;
    let exe = env::current_exe().map_err(|e| format!("failed resolving current exe: {}", e))?;
    let child = Command::new(exe)
        .arg("api")
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::from(log_out))
        .stderr(Stdio::from(log_err))
        .spawn()
        .map_err(|e| format!("failed launching daemon api process: {}", e))?;
    Ok(child.id())
}

fn run_daemon_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio daemon [start|stop|restart|status|logs [n]] [--host <addr>] [--port <n>] [--timeout <seconds>]"
        );
        return;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut action = "status".to_string();
    let mut host = daemon_default_host();
    let mut port = daemon_default_port();
    if let Some(saved) = read_saved_api_port(&cwd) {
        port = saved;
    }
    let mut timeout_s = daemon_default_timeout();
    let mut logs_n = 60usize;
    let mut action_set = false;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "start" | "stop" | "restart" | "status" | "logs" => {
                action = s;
                action_set = true;
            }
            "--host" => {
                i += 1;
                host = arg_value(args, i, "--host");
            }
            "--port" => {
                i += 1;
                let raw = arg_value(args, i, "--port");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            "--timeout" => {
                i += 1;
                let raw = arg_value(args, i, "--timeout");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(1);
            }
            other if other.starts_with("--host=") => {
                host = other.trim_start_matches("--host=").to_string();
            }
            other if other.starts_with("--port=") => {
                let raw = other.trim_start_matches("--port=");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            other if other.starts_with("--timeout=") => {
                let raw = other.trim_start_matches("--timeout=");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(1);
            }
            other => {
                if action == "logs" && !other.starts_with('-') {
                    logs_n = other.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("error: logs count must be an integer");
                        process::exit(2);
                    });
                } else if !action_set {
                    action = other.to_string();
                    action_set = true;
                } else {
                    eprintln!("error: unknown option '{}'", other);
                    process::exit(2);
                }
            }
        }
        i += 1;
    }

    match action.as_str() {
        "start" => {
            if api_health_host_port(&host, port) {
                if let Some(pid) = read_saved_api_pid(&cwd) {
                    let _ = persist_api_runtime_state(&cwd, pid, port);
                } else {
                    let _ = fs::write(api_port_path(&cwd), format!("{}\n", port));
                }
                println!("daemon: running (http://{}:{})", host, port);
                return;
            }
            if !port_available(&host, port) {
                if let Some(free) = find_free_port(&host, daemon_default_port(), 100) {
                    eprintln!("retrivio daemon: port {} busy; using {}", port, free);
                    port = free;
                } else {
                    eprintln!(
                        "error: no free {} port available near {}",
                        host,
                        daemon_default_port()
                    );
                    process::exit(1);
                }
            }
            let pid = spawn_api_daemon(&cwd, &host, port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let deadline = Instant::now() + Duration::from_secs(timeout_s);
            while Instant::now() < deadline {
                if api_health_host_port(&host, port) {
                    persist_api_runtime_state(&cwd, pid, port).unwrap_or_else(|e| {
                        eprintln!("warning: {}", e);
                    });
                    println!("daemon: running (http://{}:{})", host, port);
                    return;
                }
                if !pid_is_alive(pid) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
            clear_api_runtime_state(&cwd);
            eprintln!(
                "retrivio daemon failed to start; see {}",
                api_log_path(&cwd).display()
            );
            process::exit(1);
        }
        "stop" => {
            if let Some(pid) = read_saved_api_pid(&cwd) {
                let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
                let deadline = Instant::now() + Duration::from_secs(timeout_s);
                while Instant::now() < deadline {
                    if !pid_is_alive(pid) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                if pid_is_alive(pid) {
                    let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
                }
            }
            clear_api_runtime_state(&cwd);
            println!("daemon: stopped");
        }
        "restart" => {
            if let Some(pid) = read_saved_api_pid(&cwd) {
                let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
                thread::sleep(Duration::from_millis(150));
                if pid_is_alive(pid) {
                    let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
                }
            }
            clear_api_runtime_state(&cwd);
            if !port_available(&host, port) {
                if let Some(free) = find_free_port(&host, daemon_default_port(), 100) {
                    eprintln!("retrivio daemon: port {} busy; using {}", port, free);
                    port = free;
                } else {
                    eprintln!(
                        "error: no free {} port available near {}",
                        host,
                        daemon_default_port()
                    );
                    process::exit(1);
                }
            }
            let pid = spawn_api_daemon(&cwd, &host, port).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            let deadline = Instant::now() + Duration::from_secs(timeout_s);
            while Instant::now() < deadline {
                if api_health_host_port(&host, port) {
                    persist_api_runtime_state(&cwd, pid, port).unwrap_or_else(|e| {
                        eprintln!("warning: {}", e);
                    });
                    println!("daemon: running (http://{}:{})", host, port);
                    return;
                }
                if !pid_is_alive(pid) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
            clear_api_runtime_state(&cwd);
            eprintln!(
                "retrivio daemon failed to start; see {}",
                api_log_path(&cwd).display()
            );
            process::exit(1);
        }
        "status" => {
            let mut status_port = port;
            if let Some(saved) = read_saved_api_port(&cwd) {
                status_port = saved;
            }
            if api_health_host_port(&host, status_port) {
                println!("daemon: running (http://{}:{})", host, status_port);
            } else {
                println!("daemon: stopped");
                process::exit(1);
            }
        }
        "logs" => {
            let log = api_log_path(&cwd);
            if !log.exists() {
                eprintln!("error: log file not found: {}", log.display());
                process::exit(1);
            }
            tail_file_lines(&log, logs_n).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
        }
        other => {
            eprintln!("error: unknown daemon action '{}'", other);
            process::exit(2);
        }
    }
}

fn graph_viewer_html() -> &'static str {
    r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Retrivio Graph Viewer</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111a2d;
      --line: #223252;
      --text: #d9e4ff;
      --muted: #9fb4d9;
      --accent: #65d9a7;
      --accent2: #7eb5ff;
      --warn: #ffd37a;
    }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace; }
    .wrap { display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }
    .top {
      display: flex; gap: 10px; align-items: center; padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #0f172a, #0c1a35 48%, #0f2c2b);
    }
    input, button, select {
      background: #0b162b; color: var(--text); border: 1px solid var(--line); border-radius: 7px;
      padding: 8px 10px; font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent2); }
    .grid { display: grid; grid-template-columns: 1.4fr 1fr; min-height: 0; }
    .panel { min-height: 0; border-right: 1px solid var(--line); }
    .panel:last-child { border-right: none; }
    #graphBox { position: relative; height: calc(100vh - 58px); }
    svg { width: 100%; height: 100%; display: block; background: radial-gradient(circle at 40% 30%, #14213d 0%, #0b1220 70%); }
    #busyOverlay {
      position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
      background: rgba(6, 10, 22, 0.46); backdrop-filter: blur(1px); z-index: 8;
    }
    #busyOverlay.active { display: flex; }
    .busy-card {
      display: inline-flex; align-items: center; gap: 10px;
      background: rgba(17, 26, 45, 0.92); border: 1px solid var(--line);
      border-radius: 10px; padding: 10px 14px; color: var(--text);
    }
    .spinner {
      width: 14px; height: 14px; border: 2px solid #35527f; border-top-color: #7eb5ff;
      border-radius: 50%; animation: spin 0.9s linear infinite;
    }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .label { fill: var(--text); font-size: 12px; pointer-events: none; }
    .muted { color: var(--muted); }
    .right { height: calc(100vh - 58px); overflow: auto; padding: 10px; }
    .card { border: 1px solid var(--line); border-radius: 10px; padding: 10px; margin-bottom: 10px; background: var(--panel); }
    .small { font-size: 12px; color: var(--muted); }
    .mono { white-space: pre-wrap; word-break: break-word; }
    .chunk-actions { margin-top: 8px; display: flex; gap: 8px; }
    .chip { color: var(--warn); font-size: 12px; }
    a { color: var(--accent2); text-decoration: none; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <strong>Retrivio Graph Viewer</strong>
    <input id="search" placeholder="search query (optional)" style="flex:1.2" />
    <select id="searchView">
      <option value="projects" selected>projects</option>
      <option value="files">files</option>
    </select>
    <input id="focus" placeholder="focus project path (optional)" style="flex:1" />
    <select id="limit">
      <option value="80">80 nodes</option>
      <option value="120" selected>120 nodes</option>
      <option value="200">200 nodes</option>
      <option value="350">350 nodes</option>
    </select>
    <button id="searchBtn">Search</button>
    <button id="clearSearch">Clear</button>
    <button id="reload">Reload</button>
  </div>
  <div class="grid">
    <div class="panel" id="graphBox">
      <svg id="graph"></svg>
      <div id="busyOverlay">
        <div class="busy-card">
          <div class="spinner"></div>
          <div id="busyText">Loading...</div>
        </div>
      </div>
    </div>
    <div class="panel right">
      <div class="card">
        <div><strong>Selected Project</strong></div>
        <div id="selectedProject" class="small">none</div>
      </div>
      <div class="card">
        <div><strong>Search Matches</strong></div>
        <div id="searchMatches" class="small">global graph mode</div>
      </div>
      <div class="card">
        <div><strong>Selected Chunk</strong></div>
        <div id="selectedChunk" class="small">none</div>
      </div>
      <div class="card">
        <div><strong>Project Chunks</strong></div>
        <div id="chunks" class="small">select a project node</div>
      </div>
      <div class="card">
        <div><strong>Related Chunks</strong></div>
        <div id="related" class="small">select a chunk</div>
      </div>
      <div class="card">
        <div><strong>Explainability</strong></div>
        <div id="explain" class="small">select a related chunk</div>
      </div>
      <div class="card">
        <div><strong>Suppressed Relations</strong></div>
        <div id="suppressed" class="small">select a chunk</div>
      </div>
    </div>
  </div>
</div>
<script>
const q = (id) => document.getElementById(id);
const state = {
  nodes: [],
  edges: [],
  searchQuery: "",
  searchView: "projects",
  searchResults: [],
  selectedProject: "",
  selectedChunk: 0,
  relatedRows: [],
  relatedSource: null,
  selectedRelationIdx: -1,
  feedbackRows: [],
  requestSeq: 0,
  busyCount: 0,
  view: {
    zoom: 1.0,
    panX: 0,
    panY: 0,
    bounds: null,
    hasInteracted: false,
    drag: {
      active: false,
      moved: false,
      startX: 0,
      startY: 0,
      pointerId: null
    },
    ignoreClicksUntil: 0,
    initialized: false
  }
};
async function jget(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
async function jpost(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
function esc(v) {
  return String(v ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;" }[c]));
}
function basenamePath(path) {
  const raw = String(path || "");
  if (!raw) return "";
  const parts = raw.split("/");
  return parts[parts.length - 1] || raw;
}
function shortLabel(text, maxLen = 26) {
  const s = String(text || "");
  if (s.length <= maxLen) return s;
  if (maxLen <= 1) return "…";
  return s.slice(0, maxLen - 1) + "…";
}
function setBusy(active, message) {
  if (active) state.busyCount += 1;
  else state.busyCount = Math.max(0, state.busyCount - 1);
  const overlay = q("busyOverlay");
  if (!overlay) return;
  if (message) {
    const t = q("busyText");
    if (t) t.textContent = String(message);
  }
  if (state.busyCount > 0) {
    overlay.classList.add("active");
  } else {
    overlay.classList.remove("active");
  }
}
async function withBusy(message, fn) {
  setBusy(true, message);
  try {
    return await fn();
  } finally {
    setBusy(false);
  }
}
function nextRequestId() {
  state.requestSeq += 1;
  return state.requestSeq;
}
function isStaleRequest(requestId) {
  return requestId !== state.requestSeq;
}
function nodeColor(node) {
  const kind = String((node && node.kind) || "project");
  if (kind === "query") return "#ffd37a";
  if (kind === "match") return "#65d9a7";
  if (kind === "file") return "#a9d4ff";
  if (kind === "related") return "#7eb5ff";
  return "#7eb5ff";
}
function nodeRadius(node) {
  const kind = String((node && node.kind) || "project");
  if (kind === "query") return 9;
  if (kind === "match") return 7;
  if (kind === "file") return 6;
  return 5.5;
}
function renderSearchMatches() {
  const mount = q("searchMatches");
  if (!mount) return;
  if (!state.searchQuery) {
    mount.innerHTML = `<div class="small">global graph mode</div>`;
    return;
  }
  if (!state.searchResults.length) {
    mount.innerHTML = `<div class="small">no matches for "${esc(state.searchQuery)}"</div>`;
    return;
  }
  mount.innerHTML = state.searchResults.slice(0, 14).map((r, i) => {
    if (state.searchView === "files") {
      const filePath = r.doc_rel_path || r.path || "";
      const projectPath = r.project_path || "";
      return `
      <div style="padding:6px 0;border-bottom:1px solid #223252;">
        <div><strong>${String(i + 1).padStart(2, "0")}</strong> ${esc(shortLabel(filePath, 42))}</div>
        <div class="small">score=${Number(r.score || 0).toFixed(3)} project=${esc(shortLabel(basenamePath(projectPath), 20))}</div>
        ${projectPath ? `<div class="small"><a href="#" onclick="selectProject(decodeURIComponent('${encodeURIComponent(projectPath)}')); return false;">open project</a></div>` : ""}
      </div>`;
    }
    const path = r.path || "";
    return `
    <div style="padding:6px 0;border-bottom:1px solid #223252;">
      <div><strong>${String(i + 1).padStart(2, "0")}</strong> ${esc(shortLabel(basenamePath(path), 34))}</div>
      <div class="small">score=${Number(r.score || 0).toFixed(3)} sem=${Number(r.semantic || 0).toFixed(3)} lex=${Number(r.lexical || 0).toFixed(3)} gscore=${Number(r.graph || 0).toFixed(3)}</div>
      <div class="small"><a href="#" onclick="selectProject(decodeURIComponent('${encodeURIComponent(path)}')); return false;">open project</a></div>
    </div>`;
  }).join("");
}
function relationReason(rel) {
  switch (String(rel || "")) {
    case "same_file": return "Connected because both chunks are from the same file.";
    case "same_project": return "Connected because both chunks are in the same project.";
    case "project_edge": return "Connected through a project-level RELATED graph edge.";
    case "related_project": return "Connected through graph expansion from a related project.";
    case "seed": return "Top seed chunk directly retrieved for this context.";
    default: return "Connected by hybrid semantic/lexical ranking and graph context.";
  }
}
function relationQualityBadge(v) {
  const qv = String(v || "unspecified");
  if (qv === "good") return "good";
  if (qv === "weak") return "weak";
  if (qv === "wrong") return "wrong";
  return "unspecified";
}
function renderExplain() {
  if (!state.relatedRows.length || state.selectedRelationIdx < 0 || !state.relatedSource) {
    q("explain").innerHTML = `<div class="small">select a related chunk</div>`;
    return;
  }
  const src = state.relatedSource;
  const r = state.relatedRows[state.selectedRelationIdx];
  if (!r) {
    q("explain").innerHTML = `<div class="small">select a related chunk</div>`;
    return;
  }
  q("explain").innerHTML = `
    <div><span class="chip">${esc(r.relation)}</span> #${esc(r.chunk_id)}:${esc(r.chunk_index)} ${esc(r.doc_rel_path || "")}</div>
    <div class="small">source: #${esc(src.chunk_id)}:${esc(src.chunk_index)} ${esc(src.doc_rel_path || "")}</div>
    <div class="small">score=${Number(r.score || 0).toFixed(3)} relation_weight=${Number(r.relation_weight || 0).toFixed(3)}</div>
    <div class="small">semantic=${Number(r.semantic || 0).toFixed(3)} lexical=${Number(r.lexical || 0).toFixed(3)} quality=${Number(r.quality || 0).toFixed(3)}</div>
    <div class="small">relation_quality=${esc(relationQualityBadge(r.relation_quality))} multiplier=${Number(r.relation_quality_multiplier || 1).toFixed(3)}</div>
    <div class="small">${esc(relationReason(r.relation))}</div>
  `;
}
function setExplain(idx) {
  state.selectedRelationIdx = Number(idx || 0);
  renderExplain();
}
function drillChunk(chunkId) {
  if (!chunkId) return;
  loadRelated(Number(chunkId)).catch((e) => alert(e.message));
}
async function loadSuppressed(chunkId) {
  if (!chunkId) {
    state.feedbackRows = [];
    q("suppressed").innerHTML = `<div class="small">select a chunk</div>`;
    return;
  }
  const data = await jget(`/chunks/feedback?chunk_id=${encodeURIComponent(chunkId)}&decision=suppressed&limit=120`);
  const rows = data.results || [];
  state.feedbackRows = rows;
  if (!rows.length) {
    q("suppressed").innerHTML = `<div class="small">none</div>`;
    return;
  }
  q("suppressed").innerHTML = rows.map((r, i) => `
    <div style="padding:8px 0;border-bottom:1px solid #223252;">
      <div><span class="chip">${esc(r.relation)}</span> #${esc(r.target_chunk_id)}:${esc(r.target_chunk_index)} ${esc(r.target_doc_rel_path || "")}</div>
      <div class="small">quality=${esc(relationQualityBadge(r.quality_label))}</div>
      <div class="small">updated=${new Date((Number(r.updated_at || 0) * 1000)).toLocaleString()} source=${esc(r.source || "-")}</div>
      ${r.note ? `<div class="small mono">${esc(r.note)}</div>` : ""}
      <div class="chunk-actions">
        <button onclick="restoreSuppressed(${Number(i)})">Restore</button>
      </div>
    </div>
  `).join("");
}
async function suppressRelated(idx) {
  const i = Number(idx || 0);
  const row = state.relatedRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/suppress", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.chunk_id || 0),
    relation: String(row.relation || ""),
    note: "suppressed via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function restoreSuppressed(idx) {
  const i = Number(idx || 0);
  const row = state.feedbackRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/restore", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.target_chunk_id || 0),
    relation: String(row.relation || ""),
    note: "restored via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function setRelatedQuality(idx, qualityLabel) {
  const i = Number(idx || 0);
  const row = state.relatedRows[i];
  if (!row || !state.selectedChunk) return;
  await jpost("/chunks/feedback/quality", {
    source_chunk_id: Number(state.selectedChunk),
    target_chunk_id: Number(row.chunk_id || 0),
    relation: String(row.relation || ""),
    quality_label: String(qualityLabel || "unspecified"),
    note: "quality set via graph viewer",
  });
  await loadRelated(Number(state.selectedChunk));
}
async function loadGraph() {
  const requestId = nextRequestId();
  const search = q("search").value.trim();
  const searchView = q("searchView").value === "files" ? "files" : "projects";
  state.searchQuery = search;
  state.searchView = searchView;
  if (search) {
    await loadSearchGraph(search, searchView, requestId);
    if (isStaleRequest(requestId)) return;
    renderSearchMatches();
    return;
  }
  const focus = q("focus").value.trim();
  const limit = q("limit").value;
  const url = `/graph/view/data?limit=${encodeURIComponent(limit)}${focus ? `&focus=${encodeURIComponent(focus)}` : ""}`;
  const data = await jget(url);
  if (isStaleRequest(requestId)) return;
  state.nodes = data.nodes || [];
  state.edges = data.edges || [];
  state.searchResults = [];
  renderSearchMatches();
  drawGraph();
}
async function loadSearchGraph(query, view, requestId) {
  const limit = Number(q("limit").value || 120);
  const searchLimit = Math.min(view === "files" ? 70 : 36, Math.max(12, Math.floor(limit / 2)));
  const data = await jget(`/search?q=${encodeURIComponent(query)}&view=${encodeURIComponent(view)}&limit=${encodeURIComponent(searchLimit)}`);
  if (isStaleRequest(requestId)) return;
  const results = Array.isArray(data.results) ? data.results : [];
  state.searchResults = results;
  if (!results.length) {
    state.nodes = [];
    state.edges = [];
    drawGraph();
    return;
  }
  const nodeMap = new Map();
  const edgeMap = new Map();
  const queryId = `query:${query}`;
  nodeMap.set(queryId, { id: queryId, label: `? ${query}`, path: "", kind: "query" });
  const pushEdge = (source, target, kind, weight) => {
    const k = `${source}|${target}|${kind}`;
    if (!edgeMap.has(k)) edgeMap.set(k, { source, target, kind, weight: Number(weight || 0) });
  };

  if (view === "projects") {
    const top = results.slice(0, 24);
    for (const r of top) {
      const path = String(r.path || "");
      if (!path) continue;
      nodeMap.set(path, { id: path, label: basenamePath(path), path, kind: "match", score: Number(r.score || 0) });
      pushEdge(queryId, path, "match", Number(r.score || 0));
    }
    const neighborCalls = top.slice(0, 8).map((r) => jget(`/graph/neighbors?path=${encodeURIComponent(String(r.path || ""))}&limit=6`).catch(() => null));
    const neighbors = await Promise.all(neighborCalls);
    if (isStaleRequest(requestId)) return;
    for (const payload of neighbors) {
      if (!payload || !Array.isArray(payload.neighbors)) continue;
      const srcPath = String(payload.path || "");
      for (const n of payload.neighbors.slice(0, 6)) {
        const dst = String(n.dst || "");
        if (!dst) continue;
        if (!nodeMap.has(dst)) {
          nodeMap.set(dst, { id: dst, label: basenamePath(dst), path: dst, kind: "related" });
        }
        pushEdge(srcPath, dst, String(n.kind || "related"), Number(n.weight || 0));
      }
    }
  } else {
    const top = results.slice(0, 32);
    const projectCandidates = new Set();
    for (const r of top) {
      const filePath = String(r.path || "");
      const projectPath = String(r.project_path || "");
      const chunkId = Number(r.chunk_id || 0);
      const fileId = `file:${chunkId || filePath}`;
      const fileLabel = String(r.doc_rel_path || basenamePath(filePath) || "file");
      nodeMap.set(fileId, {
        id: fileId,
        label: shortLabel(fileLabel, 30),
        path: filePath,
        project_path: projectPath,
        kind: "file",
        score: Number(r.score || 0),
      });
      pushEdge(queryId, fileId, "match", Number(r.score || 0));
      if (projectPath) {
        if (!nodeMap.has(projectPath)) {
          nodeMap.set(projectPath, { id: projectPath, label: basenamePath(projectPath), path: projectPath, kind: "project" });
        }
        pushEdge(fileId, projectPath, "in_project", 1.0);
        projectCandidates.add(projectPath);
      }
    }
    const projectList = Array.from(projectCandidates).slice(0, 6);
    const neighborCalls = projectList.map((p) => jget(`/graph/neighbors?path=${encodeURIComponent(p)}&limit=4`).catch(() => null));
    const neighbors = await Promise.all(neighborCalls);
    if (isStaleRequest(requestId)) return;
    for (const payload of neighbors) {
      if (!payload || !Array.isArray(payload.neighbors)) continue;
      const srcPath = String(payload.path || "");
      for (const n of payload.neighbors.slice(0, 4)) {
        const dst = String(n.dst || "");
        if (!dst) continue;
        if (!nodeMap.has(dst)) {
          nodeMap.set(dst, { id: dst, label: basenamePath(dst), path: dst, kind: "related" });
        }
        pushEdge(srcPath, dst, String(n.kind || "related"), Number(n.weight || 0));
      }
    }
  }
  state.nodes = Array.from(nodeMap.values());
  state.edges = Array.from(edgeMap.values());
  drawGraph();
}
function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
function fitGraphToViewport(force) {
  const svg = q("graph");
  if (!svg || !state.view.bounds) return;
  if (!force && state.view.hasInteracted) return;
  const rect = svg.getBoundingClientRect();
  const bounds = state.view.bounds;
  const graphW = Math.max(120, (bounds.maxX - bounds.minX) + 130);
  const graphH = Math.max(120, (bounds.maxY - bounds.minY) + 90);
  const availW = Math.max(120, rect.width - 60);
  const availH = Math.max(120, rect.height - 60);
  const scale = clamp(Math.min(availW / graphW, availH / graphH), 0.45, 1.65);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  state.view.zoom = scale;
  state.view.panX = -centerX * scale;
  state.view.panY = -centerY * scale;
  applyGraphTransform();
}
function applyGraphTransform() {
  const svg = q("graph");
  const viewport = q("graphViewport");
  if (!svg || !viewport) return;
  const rect = svg.getBoundingClientRect();
  const tx = (rect.width / 2) + state.view.panX;
  const ty = (rect.height / 2) + state.view.panY;
  viewport.setAttribute(
    "transform",
    `translate(${tx.toFixed(2)} ${ty.toFixed(2)}) scale(${state.view.zoom.toFixed(4)})`
  );
}
function initGraphInteractions() {
  if (state.view.initialized) return;
  state.view.initialized = true;
  const svg = q("graph");
  if (!svg) return;
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const oldZoom = state.view.zoom;
    const nextZoom = clamp(oldZoom * Math.exp(-e.deltaY * 0.0016), 0.35, 3.2);
    if (Math.abs(nextZoom - oldZoom) < 1e-5) return;
    const wx = (sx - (rect.width / 2) - state.view.panX) / oldZoom;
    const wy = (sy - (rect.height / 2) - state.view.panY) / oldZoom;
    state.view.zoom = nextZoom;
    state.view.panX = sx - (rect.width / 2) - (wx * nextZoom);
    state.view.panY = sy - (rect.height / 2) - (wy * nextZoom);
    state.view.hasInteracted = true;
    applyGraphTransform();
  }, { passive: false });
  svg.addEventListener("pointerdown", (e) => {
    state.view.drag.active = true;
    state.view.drag.moved = false;
    state.view.drag.startX = e.clientX;
    state.view.drag.startY = e.clientY;
    state.view.drag.pointerId = e.pointerId;
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", (e) => {
    if (!state.view.drag.active) return;
    const dx = e.clientX - state.view.drag.startX;
    const dy = e.clientY - state.view.drag.startY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
      state.view.drag.moved = true;
    }
    state.view.drag.startX = e.clientX;
    state.view.drag.startY = e.clientY;
    state.view.panX += dx;
    state.view.panY += dy;
    state.view.hasInteracted = true;
    applyGraphTransform();
  });
  const endDrag = () => {
    if (!state.view.drag.active) return;
    if (state.view.drag.moved) {
      state.view.ignoreClicksUntil = Date.now() + 180;
    }
    state.view.drag.active = false;
    state.view.drag.moved = false;
    state.view.drag.pointerId = null;
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("dblclick", (e) => {
    e.preventDefault();
    state.view.hasInteracted = false;
    fitGraphToViewport(true);
  });
  window.addEventListener("resize", () => {
    drawGraph();
  });
}
function drawGraph() {
  initGraphInteractions();
  const svg = q("graph");
  svg.innerHTML = "";
  const box = svg.getBoundingClientRect();
  const w = Math.max(300, box.width);
  const h = Math.max(300, box.height);
  const n = state.nodes.length;
  if (!n) {
    const msg = state.searchQuery
      ? `No graph matches for "${state.searchQuery}".`
      : "No graph data yet. Run retrivio index first.";
    svg.innerHTML = `<text x="24" y="40" class="label">${esc(msg)}</text>`;
    return;
  }

  const pos = new Map();
  const queryNode = state.nodes.find((node) => String(node.kind || "") === "query");
  if (queryNode) {
    pos.set(queryNode.id, { x: 0, y: 0, node: queryNode });
    const matchNodes = state.nodes.filter((node) => node.id !== queryNode.id && (node.kind === "match" || node.kind === "file"));
    const otherNodes = state.nodes.filter((node) => node.id !== queryNode.id && node.kind !== "match" && node.kind !== "file");
    const r1 = Math.max(70, Math.min(w, h) * 0.23);
    const r2 = Math.max(130, Math.min(w, h) * 0.38);
    matchNodes.forEach((node, i) => {
      const a = (Math.PI * 2 * i / Math.max(1, matchNodes.length)) - (Math.PI / 2);
      pos.set(node.id, { x: Math.cos(a) * r1, y: Math.sin(a) * r1, node });
    });
    otherNodes.forEach((node, i) => {
      const a = (Math.PI * 2 * i / Math.max(1, otherNodes.length)) - (Math.PI / 2);
      pos.set(node.id, { x: Math.cos(a) * r2, y: Math.sin(a) * r2, node });
    });
  } else {
    const radius = Math.max(70, Math.min(w, h) * 0.40);
    state.nodes.forEach((node, i) => {
      const a = (Math.PI * 2 * i / n) - (Math.PI / 2);
      const x = Math.cos(a) * radius;
      const y = Math.sin(a) * radius;
      pos.set(node.id, { x, y, node });
    });
  }
  const viewport = document.createElementNS("http://www.w3.org/2000/svg", "g");
  viewport.setAttribute("id", "graphViewport");
  svg.appendChild(viewport);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const markBounds = (x, y) => {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  };

  state.edges.forEach((e) => {
    const s = pos.get(e.source);
    const t = pos.get(e.target);
    if (!s || !t) return;
    markBounds(s.x, s.y);
    markBounds(t.x, t.y);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", s.x); line.setAttribute("y1", s.y);
    line.setAttribute("x2", t.x); line.setAttribute("y2", t.y);
    const w = Math.max(0.6, Math.min(4.0, Number(e.weight || 0.5) * 2.5));
    line.setAttribute("stroke-width", w);
    line.setAttribute("stroke", "rgba(126,181,255,0.38)");
    viewport.appendChild(line);
  });
  state.nodes.forEach((node) => {
    const p = pos.get(node.id);
    if (!p) return;
    markBounds(p.x, p.y);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", p.x); c.setAttribute("cy", p.y);
    const clickPath = String(node.project_path || node.path || "");
    const isSelected = clickPath && state.selectedProject === clickPath;
    c.setAttribute("r", isSelected ? nodeRadius(node) + 2 : nodeRadius(node));
    c.setAttribute("fill", isSelected ? "#65d9a7" : nodeColor(node));
    c.style.cursor = clickPath ? "pointer" : "default";
    if (clickPath) {
      c.onclick = () => {
        if (Date.now() < state.view.ignoreClicksUntil) return;
        withBusy("Loading project chunks...", () => selectProject(clickPath))
          .catch((e) => alert(e.message));
      };
    }
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", p.x + 9); t.setAttribute("y", p.y + 4);
    t.setAttribute("class", "label");
    t.textContent = shortLabel(node.label || node.path || node.id, state.searchQuery ? 22 : 28);
    g.appendChild(c); g.appendChild(t);
    viewport.appendChild(g);
  });
  if (!Number.isFinite(minX)) {
    minX = -100; minY = -100; maxX = 100; maxY = 100;
  }
  state.view.bounds = { minX, minY, maxX, maxY };
  fitGraphToViewport(false);
  applyGraphTransform();
}
async function selectProject(path) {
  state.selectedProject = path;
  state.selectedChunk = 0;
  state.relatedRows = [];
  state.relatedSource = null;
  state.selectedRelationIdx = -1;
  state.feedbackRows = [];
  q("selectedProject").textContent = path;
  q("selectedChunk").textContent = "none";
  q("related").innerHTML = `<div class="small">select a chunk</div>`;
  q("suppressed").innerHTML = `<div class="small">select a chunk</div>`;
  renderExplain();
  drawGraph();
  const data = await jget(`/graph/view/chunks?path=${encodeURIComponent(path)}&limit=40`);
  const rows = data.chunks || [];
  if (!rows.length) {
    q("chunks").innerHTML = `<div class="small">no chunks</div>`;
    return;
  }
  q("chunks").innerHTML = rows.map((r) => `
    <div style="padding:8px 0;border-bottom:1px solid #223252;">
      <div><span class="chip">#${esc(r.chunk_id)}:${esc(r.chunk_index)}</span> ${esc(r.doc_rel_path)}</div>
      <div class="small">tokens=${Number(r.token_count || 0)}</div>
      <div class="small mono">${esc(r.excerpt)}</div>
      <div class="chunk-actions">
        <button onclick="withBusy('Loading related chunks...', () => loadRelated(${Number(r.chunk_id || 0)})).catch(e => alert(e.message))">Related</button>
      </div>
    </div>
  `).join("");
}
async function loadRelated(chunkId) {
  if (!chunkId) return;
  state.selectedChunk = chunkId;
  const data = await jget(`/graph/view/related?chunk_id=${encodeURIComponent(chunkId)}&limit=20`);
  const src = data.source || {};
  const rows = data.results || [];
  state.relatedSource = src;
  state.relatedRows = rows;
  state.selectedRelationIdx = rows.length ? 0 : -1;
  q("selectedChunk").textContent = src.chunk_id
    ? `#${src.chunk_id}:${src.chunk_index} ${src.doc_rel_path || ""}`
    : `#${chunkId}`;
  if (!rows.length) {
    q("related").innerHTML = `<div class="small">no related chunks for ${esc(chunkId)}</div>`;
    renderExplain();
    await loadSuppressed(Number(chunkId));
    return;
  }
  q("related").innerHTML = `
    <div class="small">source: #${esc(src.chunk_id)}:${esc(src.chunk_index)} ${esc(src.doc_rel_path || "")}</div>
    ${rows.map((r, i) => `
      <div style="padding:8px 0;border-bottom:1px solid #223252;">
        <div><span class="chip">${esc(r.relation)}</span> #${esc(r.chunk_id)}:${esc(r.chunk_index)} ${esc(r.doc_rel_path)}</div>
        <div class="small">score=${Number(r.score || 0).toFixed(3)} relation_weight=${Number(r.relation_weight || 0).toFixed(3)} sem=${Number(r.semantic || 0).toFixed(2)} lex=${Number(r.lexical || 0).toFixed(2)} q=${Number(r.quality || 0).toFixed(2)}</div>
        <div class="small">relation_quality=${esc(relationQualityBadge(r.relation_quality))} multiplier=${Number(r.relation_quality_multiplier || 1).toFixed(2)}</div>
        <div class="small mono">${esc(r.excerpt)}</div>
        <div class="chunk-actions">
          <button onclick="setExplain(${Number(i)})">Why</button>
          <button onclick="drillChunk(${Number(r.chunk_id || 0)})">Drill</button>
          <button onclick="suppressRelated(${Number(i)}).catch(e => alert(e.message))">Suppress</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'good').catch(e => alert(e.message))">Good</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'weak').catch(e => alert(e.message))">Weak</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'wrong').catch(e => alert(e.message))">Wrong</button>
          <button onclick="setRelatedQuality(${Number(i)}, 'unspecified').catch(e => alert(e.message))">Clear</button>
        </div>
      </div>
    `).join("")}
  `;
  renderExplain();
  await loadSuppressed(Number(chunkId));
}
q("reload").onclick = () =>
  withBusy("Loading graph...", loadGraph).catch((e) => alert(e.message));
q("searchBtn").onclick = () =>
  withBusy("Searching graph...", loadGraph).catch((e) => alert(e.message));
q("clearSearch").onclick = () => {
  q("search").value = "";
  withBusy("Loading graph...", loadGraph).catch((e) => alert(e.message));
};
q("search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    withBusy("Searching graph...", loadGraph).catch((err) => alert(err.message));
  }
});
q("focus").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    withBusy("Loading graph...", loadGraph).catch((err) => alert(err.message));
  }
});
q("searchView").addEventListener("change", () =>
  withBusy("Searching graph...", loadGraph).catch((e) => alert(e.message)));
q("limit").addEventListener("change", () =>
  withBusy("Loading graph...", loadGraph).catch((e) => alert(e.message)));
withBusy("Loading graph...", loadGraph).catch((e) => alert(e.message));
</script>
</body>
</html>"##
}

fn send_http_text(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> Result<(), String> {
    let payload = body.as_bytes();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nConnection: close\r\n\r\n",
        status,
        reason,
        content_type,
        payload.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(payload))
        .map_err(|e| format!("failed writing HTTP text response: {}", e))
}

fn serve_graph_viewer(host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("failed to bind graph viewer listener: {}", e))?;
    println!(
        "retrivio graph viewer listening on http://{}:{}/",
        host, port
    );
    println!(
        "endpoints: GET /, GET /health, GET /search, GET /context/pack, GET /graph/neighbors, GET /graph/view/state, GET /graph/view/data, GET /graph/view/chunks, GET /graph/view/related, GET /chunks/feedback, POST /context/pack, POST /chunks/feedback/suppress, POST /chunks/feedback/restore, POST /chunks/feedback/quality"
    );
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            continue;
        };
        thread::spawn(move || {
            let req = match parse_http_request(&mut stream) {
                Ok(Some(v)) => v,
                Ok(None) => return,
                Err(err) => {
                    eprintln!("graph-viewer: {}", err);
                    return;
                }
            };
            if req.method == "GET" && (req.path == "/" || req.path == "/index.html") {
                if let Err(err) = send_http_text(
                    &mut stream,
                    200,
                    "text/html; charset=utf-8",
                    graph_viewer_html(),
                ) {
                    eprintln!("graph-viewer: {}", err);
                }
                return;
            }
            let (status, payload) = handle_api_request(req);
            if let Err(err) = send_http_json(&mut stream, status, &payload) {
                eprintln!("graph-viewer: {}", err);
            }
        });
    }
    Ok(())
}

fn run_api_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio api [--host <addr>] [--port <n>]");
        return;
    }
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8765;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--host" => {
                i += 1;
                host = arg_value(args, i, "--host");
            }
            "--port" => {
                i += 1;
                let raw = arg_value(args, i, "--port");
                port = raw.parse::<u16>().unwrap_or_else(|_| {
                    eprintln!("error: --port must be an integer between 1 and 65535");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }
    serve_api_native(&host, port).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

#[derive(Debug)]
struct ApiRequest {
    method: String,
    path: String,
    query: HashMap<String, String>,
    body: Vec<u8>,
}

fn serve_api_native(host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("failed to bind api listener: {}", e))?;
    println!("retrivio api listening on http://{}:{}", host, port);
    println!(
        "endpoints: GET /health, GET /search, GET /search/pick, GET /context/pack, GET /chunks/search, GET /chunks/related, GET /chunks/get, GET /docs/read, GET /chunks/feedback, GET /tracked, GET /graph/neighbors, GET /graph/view/state, GET /graph/view/data, GET /graph/view/chunks, GET /graph/view/related, POST /context/pack, POST /chunks/feedback/suppress, POST /chunks/feedback/restore, POST /chunks/feedback/quality, POST /refresh, POST /select, POST /tracked/add, POST /tracked/del"
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        if let Err(err) = handle_api_connection(stream) {
            eprintln!("api: {}", err);
        }
    }
    Ok(())
}

fn handle_api_connection(mut stream: TcpStream) -> Result<(), String> {
    let req_started = Instant::now();
    let req = match parse_http_request(&mut stream)? {
        Some(v) => v,
        None => return Ok(()),
    };
    let method = req.method.clone();
    let path = req.path.clone();
    if method == "GET" && path == "/search/pick" {
        let (status, body) = handle_search_pick_request(&req);
        let send = send_http_response(
            &mut stream,
            status,
            "text/plain; charset=utf-8",
            body.as_bytes(),
        );
        if api_trace_enabled() {
            let elapsed_ms = req_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "api trace: {} {} status={} elapsed_ms={:.2}",
                method, path, status, elapsed_ms
            );
        }
        return send;
    }
    let (status, payload) = handle_api_request(req);
    let send = send_http_json(&mut stream, status, &payload);
    if api_trace_enabled() {
        let elapsed_ms = req_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "api trace: {} {} status={} elapsed_ms={:.2}",
            method, path, status, elapsed_ms
        );
    }
    send
}

fn parse_http_request(stream: &mut TcpStream) -> Result<Option<ApiRequest>, String> {
    let cloned = stream
        .try_clone()
        .map_err(|e| format!("failed to clone stream: {}", e))?;
    let mut reader = BufReader::new(cloned);
    let mut request_line = String::new();
    if reader
        .read_line(&mut request_line)
        .map_err(|e| format!("failed reading request line: {}", e))?
        == 0
    {
        return Ok(None);
    }
    let request_line = request_line.trim_end_matches(['\r', '\n']).to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();
    if method.is_empty() || target.is_empty() {
        return Ok(None);
    }

    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("failed reading header line: {}", e))?;
        if n == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']).to_string();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let content_len = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("failed reading request body: {}", e))?;
    }

    let (path_raw, query_raw) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    Ok(Some(ApiRequest {
        method,
        path: path_raw,
        query: parse_query_params(&query_raw),
        body,
    }))
}

fn parse_query_params(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for part in raw.split('&') {
        if part.is_empty() {
            continue;
        }
        let (k, v) = match part.split_once('=') {
            Some((k, v)) => (k, v),
            None => (part, ""),
        };
        out.insert(url_decode_component(k), url_decode_component(v));
    }
    out
}

fn hex_val(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some((c as u8) - b'0'),
        'a'..='f' => Some((c as u8) - b'a' + 10),
        'A'..='F' => Some((c as u8) - b'A' + 10),
        _ => None,
    }
}

fn url_decode_component(raw: &str) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        if ch == '%' && i + 2 < chars.len() {
            if let (Some(h1), Some(h2)) = (hex_val(chars[i + 1]), hex_val(chars[i + 2])) {
                out.push((h1 << 4) | h2);
                i += 3;
                continue;
            }
        }
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        out.extend_from_slice(encoded.as_bytes());
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn parse_limit(raw: Option<&String>, default: usize, max_limit: usize) -> usize {
    let parsed = raw.and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
    parsed.max(1).min(max_limit)
}

fn parse_bool_flag(raw: Option<&String>) -> bool {
    let Some(v) = raw else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

static API_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

fn api_trace_enabled() -> bool {
    *API_TRACE_ENABLED.get_or_init(|| {
        let raw = env::var("RETRIVIO_API_TRACE").unwrap_or_default();
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn hybrid_search_lance(
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

    // 2. SQLite FTS5 BM25 search (existing function)
    let lexical_signals = search_lexical_chunks_sqlite(conn, query, lexical_limit)?;
    let lexical_scores: HashMap<i64, f64> = lexical_signals
        .iter()
        .map(|(id, sig)| (*id, sig.lexical))
        .collect();

    // 3. Join with metadata via existing chunk_signals_for_ids()
    chunk_signals_for_ids(conn, &semantic_scores, &lexical_scores)
}

fn pick_num3(value: f64) -> String {
    format!("{:.3}", value)
}

fn pick_trunc(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let take = max_chars.saturating_sub(3);
    let mut out: String = text.chars().take(take).collect();
    out.push_str("...");
    out
}

fn pick_pad_right(text: &str, width: usize) -> String {
    let out = pick_trunc(text, width);
    let len = out.chars().count();
    if len >= width {
        return out;
    }
    format!("{}{}", out, " ".repeat(width - len))
}

fn pick_pad_left(text: &str, width: usize) -> String {
    let out = pick_trunc(text, width);
    let len = out.chars().count();
    if len >= width {
        return out;
    }
    format!("{}{}", " ".repeat(width - len), out)
}

fn pick_home_short(path: &str) -> String {
    let home = env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        let prefix = format!("{}/", home);
        if path.starts_with(&prefix) {
            return format!("~/{}", &path[prefix.len()..]);
        }
    }
    path.to_string()
}

fn pick_preview_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('\r', "")
        .replace('\t', "    ")
        .replace('\n', "\\n")
}

fn render_project_pick_line(item: &RankedResult, verbose_metrics: bool) -> String {
    let score = pick_num3(item.score);
    let sem = pick_num3(item.semantic);
    let lex = pick_num3(item.lexical);
    let fr = pick_num3(item.frecency);
    let gscore = pick_num3(item.graph);
    let short_path = pick_home_short(&item.path);
    let metrics = if verbose_metrics {
        format!("s:{} l:{} f:{} g:{}", sem, lex, fr, gscore)
    } else {
        format!("sem:{} fr:{}", sem, fr)
    };
    let row = format!(
        "{} | {} | {} | {}",
        pick_pad_right(&short_path, 46),
        pick_pad_left(&score, 6),
        pick_pad_right("dir", 3),
        metrics
    );
    let mut preview = format!(
        "directory: {}\nscore: {} sem={} lex={} fr={} gscore={}",
        short_path, score, sem, lex, fr, gscore
    );
    if item.evidence.is_empty() {
        preview.push_str("\n\nno chunk evidence available");
    } else {
        preview.push_str("\n\ntop chunks:\n");
        for (idx, ev) in item.evidence.iter().take(4).enumerate() {
            let excerpt = pick_trunc(&collapse_whitespace(&ev.excerpt), 180);
            preview.push_str(&format!(
                "\n  {}. {}#{} score={} rel={}\n     {}",
                idx + 1,
                ev.doc_rel_path,
                ev.chunk_index,
                pick_num3(ev.score),
                ev.relation,
                excerpt
            ));
        }
    }
    format!(
        "D:{}\t{}\t\t\t\t{}",
        item.path,
        row,
        pick_preview_escape(&preview)
    )
}

fn render_file_pick_line(item: &RankedFileResult, verbose_metrics: bool) -> String {
    let score = pick_num3(item.score);
    let sem = pick_num3(item.semantic);
    let lex = pick_num3(item.lexical);
    let gscore = pick_num3(item.graph);
    let q = pick_num3(item.quality);
    let short_path = pick_home_short(&item.path);
    let short_project = pick_home_short(&item.project_path);
    let display = if item.doc_rel_path.trim().is_empty() {
        path_basename(&item.path)
    } else {
        item.doc_rel_path.clone()
    };
    let project_label = format!(
        "proj:{}",
        pick_trunc(&path_basename(&item.project_path), 16)
    );
    let metrics = if verbose_metrics {
        format!("s:{} l:{} g:{} q:{}", sem, lex, gscore, q)
    } else {
        format!("sem:{} q:{}", sem, q)
    };
    let row = format!(
        "{} | {} | {} | {}",
        pick_pad_right(&display, 46),
        pick_pad_left(&score, 6),
        pick_pad_right(&project_label, 21),
        metrics
    );
    let mut preview = format!(
        "file: {}\nproject: {}\nscore: {} sem={} lex={} gscore={} q={}\n\nchunk excerpt:\n{}",
        short_path,
        short_project,
        score,
        sem,
        lex,
        gscore,
        q,
        pick_trunc(&collapse_whitespace(&item.excerpt), 220)
    );
    if !item.evidence.is_empty() {
        preview.push_str("\n\nrelated chunks:\n");
        for (idx, ev) in item.evidence.iter().take(4).enumerate() {
            let excerpt = pick_trunc(&collapse_whitespace(&ev.excerpt), 180);
            preview.push_str(&format!(
                "\n  {}. {}#{} score={} rel={}\n     {}",
                idx + 1,
                ev.doc_rel_path,
                ev.chunk_index,
                pick_num3(ev.score),
                ev.relation,
                excerpt
            ));
        }
    }
    format!(
        "F:{}\t{}\t\t\t\t{}",
        item.path,
        row,
        pick_preview_escape(&preview)
    )
}

fn handle_search_pick_request(req: &ApiRequest) -> (u16, String) {
    let query = req.query.get("q").cloned().unwrap_or_default();
    let q = query.trim().to_string();
    if q.is_empty() {
        return (400, "missing query parameter q".to_string());
    }
    let view = req
        .query
        .get("view")
        .cloned()
        .unwrap_or_else(|| "projects".to_string())
        .trim()
        .to_ascii_lowercase();
    if view != "projects" && view != "files" {
        return (400, "invalid view parameter".to_string());
    }
    let verbose_metrics = parse_bool_flag(req.query.get("verbose_metrics"));
    let limit = if view == "files" {
        parse_limit(req.query.get("limit"), 120, 240)
    } else {
        parse_limit(req.query.get("limit"), 40, 120)
    };
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    if let Err(e) = ensure_native_embed_backend(&cfg, "api search/pick") {
        return (503, e);
    }
    if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api search/pick") {
        return (503, e);
    }
    let conn = match open_db_rw(&db_path(&cwd)) {
        Ok(v) => v,
        Err(e) => return (500, e),
    };
    if view == "files" {
        let rows = match rank_files_native(&conn, &cfg, &q, limit) {
            Ok(v) => v,
            Err(e) => return (503, e),
        };
        let lines: Vec<String> = rows
            .iter()
            .map(|item| render_file_pick_line(item, verbose_metrics))
            .collect();
        return (200, lines.join("\n"));
    }
    let rows = match rank_projects_native(&conn, &cfg, &q, limit) {
        Ok(v) => v,
        Err(e) => return (503, e),
    };
    let lines: Vec<String> = rows
        .iter()
        .map(|item| render_project_pick_line(item, verbose_metrics))
        .collect();
    (200, lines.join("\n"))
}

fn ensure_native_embed_backend(cfg: &ConfigValues, context: &str) -> Result<(), String> {
    if matches!(cfg.embed_backend.as_str(), "ollama" | "bedrock") {
        return Ok(());
    }
    Err(format!(
        "{} requires native embed_backend in [ollama, bedrock] (current='{}')",
        context, cfg.embed_backend
    ))
}

fn list_neighbors_by_path(
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

fn payload_paths(body: &Value) -> Vec<String> {
    if let Some(paths) = body.get("paths").and_then(|v| v.as_array()) {
        return paths
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
    }
    if let Some(path) = body.get("path").and_then(|v| v.as_str()) {
        return vec![path.to_string()];
    }
    Vec::new()
}

fn payload_i64(body: &Value, key: &str) -> i64 {
    if let Some(v) = body.get(key) {
        if let Some(n) = v.as_i64() {
            return n;
        }
        if let Some(s) = v.as_str() {
            if let Ok(n) = s.trim().parse::<i64>() {
                return n;
            }
        }
    }
    0
}

fn payload_string(body: &Value, key: &str) -> String {
    body.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn unique_valid_dirs(paths: &[String]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw in paths {
        let p = normalize_path(raw);
        if !p.exists() || !p.is_dir() {
            continue;
        }
        let key = p.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(p);
        }
    }
    out
}

fn stats_payload_json(stats: &IndexStats) -> Value {
    serde_json::json!({
        "total_projects": stats.total_projects,
        "updated_projects": stats.updated_projects,
        "skipped_projects": stats.skipped_projects,
        "removed_projects": stats.removed_projects,
        "vectorized_projects": stats.vectorized_projects,
        "vector_failures": stats.vector_failures,
        "tracked_roots": stats.tracked_roots,
        "graph_edges": stats.graph_edges,
        "chunk_rows": stats.chunk_rows,
        "chunk_vectors": stats.chunk_vectors,
        "retrieval_backend": stats.retrieval_backend,
        "retrieval_synced_chunks": stats.retrieval_synced_chunks,
        "retrieval_error": stats.retrieval_error,
    })
}

fn path_basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

fn path_in_tracked_roots(path: &str, tracked_roots: &[PathBuf]) -> bool {
    let target = normalize_path(path);
    tracked_roots
        .iter()
        .any(|root| target == *root || target.starts_with(root))
}

fn graph_view_data_json(
    conn: &Connection,
    focus: Option<&str>,
    limit: usize,
) -> Result<Value, String> {
    let use_limit = limit.max(10).min(600);
    let tracked_roots: Vec<PathBuf> = list_tracked_roots_conn(conn)?
        .into_iter()
        .map(|p| normalize_path(&p.to_string_lossy()))
        .collect();
    if tracked_roots.is_empty() {
        return Ok(serde_json::json!({"focus": focus.unwrap_or(""), "nodes": [], "edges": []}));
    }

    let mut selected: Vec<String> = Vec::new();
    if let Some(raw_focus) = focus {
        let focus_path = normalize_path(raw_focus).to_string_lossy().to_string();
        if !path_in_tracked_roots(&focus_path, &tracked_roots) {
            return Ok(serde_json::json!({"focus": focus.unwrap_or(""), "nodes": [], "edges": []}));
        }
        selected.push(focus_path.clone());
        let outgoing = list_neighbors_by_path(conn, &focus_path, use_limit)?;
        for (dst, _, _) in outgoing {
            if path_in_tracked_roots(&dst, &tracked_roots) {
                selected.push(dst);
            }
        }
        let mut incoming_stmt = conn
            .prepare(
                r#"
SELECT src.path, pe.kind, pe.weight
FROM project_edges pe
JOIN projects src ON src.id = pe.src_project_id
WHERE pe.dst = ?1
ORDER BY pe.weight DESC
LIMIT ?2
"#,
            )
            .map_err(|e| format!("failed preparing incoming neighbor query: {}", e))?;
        let incoming = incoming_stmt
            .query_map(params![focus_path, use_limit as i64], |row| {
                let path: String = row.get(0)?;
                let kind: String = row.get(1)?;
                let weight: f64 = row.get(2)?;
                Ok((path, kind, weight))
            })
            .map_err(|e| format!("failed querying incoming neighbor rows: {}", e))?;
        for row in incoming {
            let (path, _, _) =
                row.map_err(|e| format!("failed reading incoming neighbor row: {}", e))?;
            if path_in_tracked_roots(&path, &tracked_roots) {
                selected.push(path);
            }
        }
    } else {
        let mut stmt = conn
            .prepare(
                r#"
SELECT p.path, (
    COALESCE(out_deg.cnt, 0) + COALESCE(in_deg.cnt, 0)
) AS degree
FROM projects p
LEFT JOIN (
    SELECT src_project_id, COUNT(*) AS cnt
    FROM project_edges
    GROUP BY src_project_id
) out_deg ON out_deg.src_project_id = p.id
LEFT JOIN (
    SELECT dst, COUNT(*) AS cnt
    FROM project_edges
    GROUP BY dst
) in_deg ON in_deg.dst = p.path
ORDER BY degree DESC, p.path ASC
LIMIT ?1
"#,
            )
            .map_err(|e| format!("failed preparing project degree query: {}", e))?;
        let rows = stmt
            .query_map(params![use_limit as i64], |row| row.get::<_, String>(0))
            .map_err(|e| format!("failed querying project degree rows: {}", e))?;
        for row in rows {
            let path = row.map_err(|e| format!("failed reading project degree row: {}", e))?;
            if path_in_tracked_roots(&path, &tracked_roots) {
                selected.push(path);
            }
        }
    }

    selected.sort();
    selected.dedup();
    if selected.is_empty() {
        return Ok(serde_json::json!({"focus": focus, "nodes": [], "edges": []}));
    }
    if selected.len() > use_limit {
        selected.truncate(use_limit);
    }
    let selected_set: HashSet<String> = selected.iter().cloned().collect();

    let nodes: Vec<Value> = selected
        .iter()
        .map(|path| {
            serde_json::json!({
                "id": path,
                "label": path_basename(path),
                "path": path
            })
        })
        .collect();

    let mut edge_stmt = conn
        .prepare(
            r#"
SELECT src.path, pe.dst, pe.kind, pe.weight
FROM project_edges pe
JOIN projects src ON src.id = pe.src_project_id
ORDER BY pe.weight DESC, src.path ASC, pe.dst ASC
"#,
        )
        .map_err(|e| format!("failed preparing graph edge query: {}", e))?;
    let edge_rows = edge_stmt
        .query_map([], |row| {
            let src: String = row.get(0)?;
            let dst: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let weight: f64 = row.get(3)?;
            Ok((src, dst, kind, weight))
        })
        .map_err(|e| format!("failed querying graph edge rows: {}", e))?;

    let mut edges: Vec<Value> = Vec::new();
    for row in edge_rows {
        let (src, dst, kind, weight) =
            row.map_err(|e| format!("failed reading graph edge row: {}", e))?;
        if selected_set.contains(&src) && selected_set.contains(&dst) {
            edges.push(serde_json::json!({
                "source": src,
                "target": dst,
                "kind": kind,
                "weight": weight
            }));
            if edges.len() >= use_limit.saturating_mul(8) {
                break;
            }
        }
    }

    Ok(serde_json::json!({
        "focus": focus.unwrap_or(""),
        "nodes": nodes,
        "edges": edges
    }))
}

fn project_chunks_preview_json(
    conn: &Connection,
    project_path: &str,
    limit: usize,
) -> Result<Value, String> {
    let target = normalize_path(project_path).to_string_lossy().to_string();
    let mut stmt = conn
        .prepare(
            r#"
SELECT pc.id, pc.doc_path, pc.doc_rel_path, pc.chunk_index, pc.token_count, pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE p.path = ?1
ORDER BY pc.updated_at DESC, pc.id DESC
LIMIT ?2
"#,
        )
        .map_err(|e| format!("failed preparing project chunk preview query: {}", e))?;
    let rows = stmt
        .query_map(params![target, limit.max(1).min(400) as i64], |row| {
            let chunk_id: i64 = row.get(0)?;
            let doc_path: String = row.get(1)?;
            let doc_rel_path: String = row.get(2)?;
            let chunk_index: i64 = row.get(3)?;
            let token_count: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            Ok((
                chunk_id,
                doc_path,
                doc_rel_path,
                chunk_index,
                token_count,
                text,
            ))
        })
        .map_err(|e| format!("failed querying project chunk preview rows: {}", e))?;
    let mut chunks: Vec<Value> = Vec::new();
    for row in rows {
        let (chunk_id, doc_path, doc_rel_path, chunk_index, token_count, text) =
            row.map_err(|e| format!("failed reading project chunk preview row: {}", e))?;
        chunks.push(serde_json::json!({
            "chunk_id": chunk_id,
            "path": doc_path,
            "doc_rel_path": doc_rel_path,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "excerpt": clip_text(&text, 260),
        }));
    }
    Ok(serde_json::json!({"path": project_path, "count": chunks.len(), "chunks": chunks}))
}

fn send_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nConnection: close\r\n\r\n",
        status,
        reason,
        content_type,
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|e| format!("failed writing HTTP response: {}", e))
}

fn send_http_json(stream: &mut TcpStream, status: u16, payload: &Value) -> Result<(), String> {
    let body =
        serde_json::to_vec(payload).map_err(|e| format!("failed serializing JSON: {}", e))?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\nConnection: close\r\n\r\n",
        status,
        reason,
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|e| format!("failed writing HTTP response: {}", e))
}

fn parse_json_body(body: &[u8]) -> Value {
    if body.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_slice::<Value>(body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()))
}

fn handle_api_request(req: ApiRequest) -> (u16, Value) {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            return (200, serde_json::json!({"ok": true, "time": now_ts()}));
        }
        ("GET", "/tracked") => {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (200, serde_json::json!({"tracked_roots": values}));
        }
        ("GET", "/search") => {
            let query = req.query.get("q").cloned().unwrap_or_default();
            let q = query.trim().to_string();
            if q.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'q'."}),
                );
            }
            let search_started = Instant::now();
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let view = req
                .query
                .get("view")
                .cloned()
                .unwrap_or_else(|| "projects".to_string())
                .trim()
                .to_lowercase();
            if view != "projects" && view != "files" {
                return (
                    400,
                    serde_json::json!({"error": "Invalid view. Use 'projects' or 'files'."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api search") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api search") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if view == "files" {
                let rows = match rank_files_native(&conn, &cfg, &q, limit) {
                    Ok(v) => v,
                    Err(e) => return (503, serde_json::json!({"error": e})),
                };
                let results: Vec<Value> = rows
                    .into_iter()
                    .map(|item| {
                        let evidence: Vec<Value> = item
                            .evidence
                            .into_iter()
                            .map(|ev| {
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
                                })
                            })
                            .collect();
                        serde_json::json!({
                            "path": item.path,
                            "project_path": item.project_path,
                            "doc_rel_path": item.doc_rel_path,
                            "chunk_id": item.chunk_id,
                            "chunk_index": item.chunk_index,
                            "score": item.score,
                            "semantic": item.semantic,
                            "lexical": item.lexical,
                            "graph": item.graph,
                            "relation": item.relation,
                            "quality": item.quality,
                            "excerpt": item.excerpt,
                            "evidence": evidence,
                        })
                    })
                    .collect();
                return (
                    200,
                    serde_json::json!({
                        "query": q,
                        "view": "files",
                        "results": results,
                        "timing_ms": search_started.elapsed().as_secs_f64() * 1000.0
                    }),
                );
            }
            let rows = match rank_projects_native(&conn, &cfg, &q, limit) {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
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
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "frecency": item.frecency,
                        "graph": item.graph,
                        "evidence": evidence,
                    })
                })
                .collect();
            return (
                200,
                serde_json::json!({
                    "query": q,
                    "view": "projects",
                    "results": results,
                    "timing_ms": search_started.elapsed().as_secs_f64() * 1000.0
                }),
            );
        }
        ("GET", "/chunks/search") => {
            let query = req.query.get("q").cloned().unwrap_or_default();
            let q = query.trim().to_string();
            if q.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'q'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 30, 300);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match rank_chunks_native(&conn, &cfg, &q, limit) {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(ranked_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_search_schema(),
                    "query": q,
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("GET", "/chunks/related") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api chunks/related") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api chunks/related") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (source, rows) = match related_chunks_native(&conn, &cfg, chunk_id, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_related_schema(),
                    "source": source_chunk_json(&source),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("GET", "/chunks/get") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let max_chars = parse_limit(req.query.get("max_chars"), 8000, 500_000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let chunk = match indexed_chunk_by_id(&conn, chunk_id) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    return (
                        404,
                        serde_json::json!({"error": format!("chunk {} not found", chunk_id)}),
                    );
                }
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (text, truncated, text_chars) = truncate_text_chars(&chunk.text, max_chars);
            let returned_chars = text.chars().count();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_get_schema(),
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "path": chunk.doc_path,
                        "project_path": chunk.project_path,
                        "doc_rel_path": chunk.doc_rel_path,
                        "doc_mtime": chunk.doc_mtime,
                        "token_count": chunk.token_count,
                        "text_chars": text_chars,
                        "returned_chars": returned_chars,
                        "truncated": truncated,
                        "text": text
                    }
                }),
            );
        }
        ("GET", "/docs/read") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let raw_path = path.trim().to_string();
            if raw_path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let max_chars = parse_limit(req.query.get("max_chars"), 120_000, 2_000_000);
            let normalized_path = normalize_path(&raw_path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let mut chunks = match indexed_doc_chunks_by_path(&conn, &normalized_path) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if chunks.is_empty() && raw_path != normalized_path {
                chunks = match indexed_doc_chunks_by_path(&conn, &raw_path) {
                    Ok(v) => v,
                    Err(e) => return (500, serde_json::json!({"error": e})),
                };
            }
            if chunks.is_empty() {
                return (
                    404,
                    serde_json::json!({"error": "document path is not indexed"}),
                );
            }
            let first = &chunks[0];
            let mut full_text = String::new();
            let mut token_total: i64 = 0;
            let mut chunk_refs: Vec<Value> = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                full_text.push_str(&chunk.text);
                token_total += chunk.token_count;
                chunk_refs.push(serde_json::json!({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }));
            }
            let (text, truncated, text_chars) = truncate_text_chars(&full_text, max_chars);
            let returned_chars = text.chars().count();
            return (
                200,
                serde_json::json!({
                    "schema": doc_read_schema(),
                    "path": first.doc_path.clone(),
                    "project_path": first.project_path.clone(),
                    "doc_rel_path": first.doc_rel_path.clone(),
                    "doc_mtime": first.doc_mtime,
                    "chunk_count": chunks.len(),
                    "token_count": token_total,
                    "text_chars": text_chars,
                    "returned_chars": returned_chars,
                    "truncated": truncated,
                    "chunks": chunk_refs,
                    "text": text
                }),
            );
        }
        ("GET", "/context/pack") => {
            let query = req.query.get("q").cloned().unwrap_or_default();
            let q = query.trim().to_string();
            if q.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'q'."}),
                );
            }
            let budget_chars = parse_limit(req.query.get("budget_chars"), 12_000, 400_000);
            let seed_limit = parse_limit(req.query.get("seed_limit"), 8, 40);
            let related_per_seed = parse_limit(req.query.get("related_per_seed"), 3, 12);
            let include_docs = parse_bool_flag(req.query.get("include_docs"));
            let doc_max_chars = parse_limit(req.query.get("doc_max_chars"), 12_000, 500_000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api context/pack") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api context/pack") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let options = ContextPackOptions {
                budget_chars,
                seed_limit,
                related_per_seed,
                include_docs,
                doc_max_chars,
            };
            match build_context_pack_native(&conn, &cfg, &q, options) {
                Ok(payload) => return (200, payload),
                Err(e) => return (500, serde_json::json!({"error": e})),
            }
        }
        ("POST", "/context/pack") => {
            let body = parse_json_body(&req.body);
            let query = body
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return (400, serde_json::json!({"error": "query must be non-empty"}));
            }
            let budget_chars = body
                .get("budget_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(12_000)
                .clamp(1_000, 400_000);
            let seed_limit = body
                .get("seed_limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(8)
                .clamp(1, 40);
            let related_per_seed = body
                .get("related_per_seed")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(3)
                .clamp(0, 12);
            let include_docs = body
                .get("include_docs")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let doc_max_chars = body
                .get("doc_max_chars")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(12_000)
                .clamp(1_000, 500_000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api context/pack") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api context/pack") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let options = ContextPackOptions {
                budget_chars,
                seed_limit,
                related_per_seed,
                include_docs,
                doc_max_chars,
            };
            match build_context_pack_native(&conn, &cfg, &query, options) {
                Ok(payload) => return (200, payload),
                Err(e) => return (500, serde_json::json!({"error": e})),
            }
        }
        ("GET", "/chunks/feedback") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let decision = req
                .query
                .get("decision")
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref d) = decision {
                if d != "suppressed" && d != "active" {
                    return (
                        400,
                        serde_json::json!({"error": "Invalid decision. Use 'suppressed' or 'active'."}),
                    );
                }
            }
            let quality = req
                .query
                .get("quality")
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref qv) = quality {
                if normalize_relation_quality_label(qv).is_none() {
                    return (
                        400,
                        serde_json::json!({"error": "Invalid quality. Use 'good', 'weak', 'wrong', or 'unspecified'."}),
                    );
                }
            }
            let limit = parse_limit(req.query.get("limit"), 100, 2000);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match list_chunk_relation_feedback(
                &conn,
                chunk_id,
                decision.as_deref(),
                quality.as_deref(),
                limit,
            ) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(relation_feedback_row_json).collect();
            return (
                200,
                serde_json::json!({
                    "source_chunk_id": chunk_id,
                    "decision": decision.unwrap_or_else(|| "all".to_string()),
                    "quality": quality.unwrap_or_else(|| "all".to_string()),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("POST", "/chunks/feedback/suppress") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "suppressed",
                &note,
                "api",
                now_ts(),
            ) {
                return (500, serde_json::json!({"error": e}));
            }
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "decision": "suppressed",
                    "note": note
                }),
            );
        }
        ("POST", "/chunks/feedback/restore") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "active",
                &note,
                "api",
                now_ts(),
            ) {
                return (500, serde_json::json!({"error": e}));
            }
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "decision": "active",
                    "note": note
                }),
            );
        }
        ("POST", "/chunks/feedback/quality") => {
            let body = parse_json_body(&req.body);
            let source_chunk_id = payload_i64(&body, "source_chunk_id");
            let target_chunk_id = payload_i64(&body, "target_chunk_id");
            let relation = payload_string(&body, "relation");
            let quality_label = {
                let q = payload_string(&body, "quality_label");
                if q.is_empty() {
                    payload_string(&body, "quality")
                } else {
                    q
                }
            };
            let note = payload_string(&body, "note");
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "source_chunk_id, target_chunk_id, and relation are required."}),
                );
            }
            let normalized = match normalize_relation_quality_label(&quality_label) {
                Some(v) => v.to_string(),
                None => {
                    return (
                        400,
                        serde_json::json!({"error": "quality_label must be one of: good, weak, wrong, unspecified"}),
                    );
                }
            };
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let persisted = match set_chunk_relation_quality(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                &normalized,
                &note,
                "api",
                now_ts(),
            ) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (
                200,
                serde_json::json!({
                    "ok": true,
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": target_chunk_id,
                    "relation": relation,
                    "quality_label": persisted,
                    "note": note
                }),
            );
        }
        ("GET", "/graph/neighbors") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let path = path.trim().to_string();
            if path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 500);
            let target = normalize_path(&path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match list_neighbors_by_path(&conn, &target, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let neighbors: Vec<Value> = rows
                .into_iter()
                .map(|(dst, kind, weight)| {
                    serde_json::json!({
                        "dst": dst,
                        "kind": kind,
                        "weight": weight,
                    })
                })
                .collect();
            return (
                200,
                serde_json::json!({"path": target, "neighbors": neighbors}),
            );
        }
        ("GET", "/graph/view/state") => {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let payload = match graph_view_state_json(&conn, &cwd) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (200, payload);
        }
        ("GET", "/graph/view/data") => {
            let focus = req
                .query
                .get("focus")
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty());
            let limit = parse_limit(req.query.get("limit"), 120, 600);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let payload = match graph_view_data_json(&conn, focus.as_deref(), limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (200, payload);
        }
        ("GET", "/graph/view/chunks") => {
            let path = req.query.get("path").cloned().unwrap_or_default();
            let target = path.trim().to_string();
            if target.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing query parameter 'path'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 40, 400);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let payload = match project_chunks_preview_json(&conn, &target, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            return (200, payload);
        }
        ("GET", "/graph/view/related") => {
            let chunk_id = req
                .query
                .get("chunk_id")
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0);
            if chunk_id <= 0 {
                return (
                    400,
                    serde_json::json!({"error": "Missing or invalid query parameter 'chunk_id'."}),
                );
            }
            let limit = parse_limit(req.query.get("limit"), 20, 200);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api graph/view/related") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api graph/view/related") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let (source, rows) = match related_chunks_native(&conn, &cfg, chunk_id, limit) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            return (
                200,
                serde_json::json!({
                    "schema": chunk_related_schema(),
                    "source": source_chunk_json(&source),
                    "count": results.len(),
                    "results": results
                }),
            );
        }
        ("POST", "/refresh") => {
            let body = parse_json_body(&req.body);
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api refresh") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api refresh") {
                return (503, serde_json::json!({"error": e}));
            }
            let raw_paths = payload_paths(&body);
            let result = if raw_paths.is_empty() {
                run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    true,
                    HashSet::new(),
                    true,
                    false,
                    "api refresh",
                )
            } else {
                let roots = unique_valid_dirs(&raw_paths);
                if roots.is_empty() {
                    return (
                        400,
                        serde_json::json!({"error": "No valid directory paths provided."}),
                    );
                }
                let force_paths: HashSet<PathBuf> = roots.iter().cloned().collect();
                run_native_index(
                    &cwd,
                    &cfg,
                    Some(roots),
                    true,
                    force_paths,
                    false,
                    false,
                    "api refresh",
                )
            };
            let stats = match result {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            return (
                200,
                serde_json::json!({"stats": stats_payload_json(&stats)}),
            );
        }
        ("POST", "/select") => {
            let body = parse_json_body(&req.body);
            let path = body
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing path in request body."}),
                );
            }
            let query = body
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let target = normalize_path(&path).to_string_lossy().to_string();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let conn = match open_db_rw(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            if let Err(e) = record_selection_event(&conn, &query, &target, now_ts()) {
                return (500, serde_json::json!({"error": e}));
            }
            return (200, serde_json::json!({"ok": true}));
        }
        ("POST", "/tracked/add") => {
            let body = parse_json_body(&req.body);
            let paths = payload_paths(&body);
            let roots = unique_valid_dirs(&paths);
            if roots.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "No valid directory paths provided."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            for root in roots {
                if let Err(e) = ensure_tracked_root(&dbp, &root, now_ts()) {
                    return (500, serde_json::json!({"error": e}));
                }
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (200, serde_json::json!({"tracked_roots": values}));
        }
        ("POST", "/tracked/del") => {
            let body = parse_json_body(&req.body);
            let paths = payload_paths(&body);
            if paths.is_empty() {
                return (
                    400,
                    serde_json::json!({"error": "Missing path(s) in request body."}),
                );
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let dbp = db_path(&cwd);
            if let Err(e) = ensure_db_schema(&dbp) {
                return (500, serde_json::json!({"error": e}));
            }
            let mut removed: i64 = 0;
            for raw in paths {
                removed += remove_tracked_root(&dbp, &normalize_path(&raw)).unwrap_or(0);
            }
            let rows = match list_tracked_roots(&dbp) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let values: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            return (
                200,
                serde_json::json!({"removed": removed, "tracked_roots": values}),
            );
        }
        _ => {}
    }
    (404, serde_json::json!({"error": "Not found."}))
}

fn run_mcp_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio mcp [serve|doctor|register|unregister] [--yes]");
        println!("hint: run `retrivio mcp serve` to start the MCP stdio server");
        println!("hint: run `retrivio mcp doctor` to verify local MCP integration setup");
        return;
    }
    let Some(action_raw) = args.first() else {
        eprintln!("error: missing mcp action");
        eprintln!("usage: retrivio mcp [serve|doctor|register|unregister] [--yes]");
        eprintln!("hint: run `retrivio mcp serve` to start the MCP stdio server");
        process::exit(2);
    };
    let action = action_raw.to_string_lossy().trim().to_lowercase();
    match action.as_str() {
        "serve" => serve_mcp_native().unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        }),
        "doctor" => run_mcp_doctor(),
        "register" => run_mcp_register(&args[1..]),
        "unregister" => run_mcp_unregister(&args[1..]),
        "help" => {
            println!("usage: retrivio mcp [serve|doctor|register|unregister] [--yes]");
        }
        other => {
            eprintln!("error: unknown mcp action '{}'", other);
            process::exit(2);
        }
    }
}

fn resolve_retrivio_command_path_native() -> Option<PathBuf> {
    if let Ok(path) = env::current_exe() {
        if is_executable_file(&path) {
            return Some(path);
        }
    }
    let cwd = env::current_dir().ok()?;
    let repo_candidate = cwd.join("retrivio");
    if is_executable_file(&repo_candidate) {
        return Some(repo_candidate);
    }
    if let Some(path) = resolve_command_path("retrivio") {
        return Some(path);
    }
    if let Ok(raw) = env::var("RETRIVIO_APP_VENV") {
        let v = raw.trim();
        if !v.is_empty() {
            let path = expand_tilde(v).join("bin").join("retrivio");
            if is_executable_file(&path) {
                return Some(path);
            }
        }
    }
    let data_candidate = data_dir(&cwd).join("venv").join("bin").join("retrivio");
    if is_executable_file(&data_candidate) {
        return Some(data_candidate);
    }
    None
}

#[derive(Clone, Copy, PartialEq)]
enum McpConfigFormat {
    Json,
    Toml,
}

struct McpToolTarget {
    name: &'static str,
    format: McpConfigFormat,
    detect_dir: &'static str,
    config_file: &'static str,
    legacy_key: Option<&'static str>,
}

const MCP_TOOL_TARGETS: &[McpToolTarget] = &[
    McpToolTarget {
        name: "Claude Code",
        format: McpConfigFormat::Json,
        detect_dir: ".claude",
        config_file: ".claude.json",
        legacy_key: Some("cypress"),
    },
    McpToolTarget {
        name: "Kiro",
        format: McpConfigFormat::Json,
        detect_dir: ".kiro",
        config_file: ".kiro/settings/mcp.json",
        legacy_key: Some("cypress"),
    },
    McpToolTarget {
        name: "Codex",
        format: McpConfigFormat::Toml,
        detect_dir: ".codex",
        config_file: ".codex/config.toml",
        legacy_key: Some("cypress"),
    },
    McpToolTarget {
        name: "Gemini CLI",
        format: McpConfigFormat::Json,
        detect_dir: ".gemini",
        config_file: ".gemini/settings.json",
        legacy_key: Some("cypress"),
    },
];

// ── JSON config helpers ──────────────────────────────────────────────

fn mcp_register_json(path: &Path, key: &str, command: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir: {}", e))?;
    }
    let mut root: serde_json::Map<String, Value> = if path.exists() {
        let raw = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| format!("json parse: {}", e))?;
        match v {
            Value::Object(m) => m,
            _ => return Err("config is not a JSON object".to_string()),
        }
    } else {
        serde_json::Map::new()
    };
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let servers_map = servers
        .as_object_mut()
        .ok_or_else(|| "mcpServers is not an object".to_string())?;
    let entry = serde_json::json!({
        "type": "stdio",
        "command": command,
        "args": ["mcp", "serve"],
        "env": {}
    });
    servers_map.insert(key.to_string(), entry);
    let out = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| format!("json serialize: {}", e))?;
    fs::write(path, out.as_bytes()).map_err(|e| format!("write: {}", e))
}

fn mcp_unregister_json(path: &Path, key: &str) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let raw = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
    let mut root: Value = serde_json::from_str(&raw).map_err(|e| format!("json parse: {}", e))?;
    if let Some(servers) = root.get_mut("mcpServers").and_then(|v| v.as_object_mut()) {
        servers.remove(key);
    }
    let out = serde_json::to_string_pretty(&root).map_err(|e| format!("json serialize: {}", e))?;
    fs::write(path, out.as_bytes()).map_err(|e| format!("write: {}", e))
}

fn mcp_tool_has_entry_json(path: &Path, key: &str) -> bool {
    let raw = match fs::read_to_string(path) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let root: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return false,
    };
    root.get("mcpServers").and_then(|v| v.get(key)).is_some()
}

fn mcp_tool_current_command_json(path: &Path, key: &str) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let root: Value = serde_json::from_str(&raw).ok()?;
    root.get("mcpServers")?
        .get(key)?
        .get("command")?
        .as_str()
        .map(|s| s.to_string())
}

// ── TOML config helpers (string manipulation) ────────────────────────

fn find_next_toml_section(raw: &str, from: usize) -> Option<usize> {
    for (i, line) in raw[from..].lines().enumerate() {
        if i == 0 {
            continue; // skip the current section header
        }
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            // find byte offset
            let byte_offset = raw[from..].find(line).map(|pos| from + pos);
            return byte_offset;
        }
    }
    None
}

fn mcp_register_toml(path: &Path, key: &str, command: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir: {}", e))?;
    }
    let section_header = format!("[mcp_servers.{}]", key);
    let section_body = format!(
        "{}\ncommand = \"{}\"\nargs = [\"mcp\", \"serve\"]\n",
        section_header,
        command.replace('\\', "\\\\").replace('"', "\\\"")
    );

    if !path.exists() {
        return fs::write(path, section_body.as_bytes()).map_err(|e| format!("write: {}", e));
    }

    let raw = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;

    if let Some(start) = raw.find(&section_header) {
        let end = find_next_toml_section(&raw, start).unwrap_or(raw.len());
        let mut out = String::with_capacity(raw.len());
        out.push_str(&raw[..start]);
        out.push_str(&section_body);
        if end < raw.len() {
            out.push_str(&raw[end..]);
        }
        fs::write(path, out.as_bytes()).map_err(|e| format!("write: {}", e))
    } else {
        let mut out = raw;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&section_body);
        fs::write(path, out.as_bytes()).map_err(|e| format!("write: {}", e))
    }
}

fn mcp_unregister_toml(path: &Path, key: &str) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let raw = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
    let section_header = format!("[mcp_servers.{}]", key);
    if let Some(start) = raw.find(&section_header) {
        let end = find_next_toml_section(&raw, start).unwrap_or(raw.len());
        let mut out = String::with_capacity(raw.len());
        out.push_str(&raw[..start]);
        let rest = &raw[end..];
        // avoid double blank lines
        let rest = rest.trim_start_matches('\n');
        if !rest.is_empty() {
            out.push_str(rest);
        }
        if !out.ends_with('\n') && !out.is_empty() {
            out.push('\n');
        }
        fs::write(path, out.as_bytes()).map_err(|e| format!("write: {}", e))
    } else {
        Ok(())
    }
}

fn mcp_tool_has_entry_toml(path: &Path, key: &str) -> bool {
    let raw = match fs::read_to_string(path) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let header = format!("[mcp_servers.{}]", key);
    raw.contains(&header)
}

fn mcp_tool_current_command_toml(path: &Path, key: &str) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let header = format!("[mcp_servers.{}]", key);
    let start = raw.find(&header)?;
    let end = find_next_toml_section(&raw, start).unwrap_or(raw.len());
    let section = &raw[start..end];
    for line in section.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("command") {
            if let Some(eq_pos) = trimmed.find('=') {
                let val = trimmed[eq_pos + 1..].trim().trim_matches('"');
                return Some(val.to_string());
            }
        }
    }
    None
}

// ── Dispatcher helpers ───────────────────────────────────────────────

fn mcp_tool_is_installed(home: &Path, target: &McpToolTarget) -> bool {
    home.join(target.detect_dir).is_dir()
}

fn mcp_tool_has_entry(home: &Path, target: &McpToolTarget) -> bool {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_tool_has_entry_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_tool_has_entry_toml(&path, "retrivio"),
    }
}

fn mcp_tool_current_command(home: &Path, target: &McpToolTarget) -> Option<String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_tool_current_command_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_tool_current_command_toml(&path, "retrivio"),
    }
}

fn mcp_tool_has_legacy_entry(home: &Path, target: &McpToolTarget) -> bool {
    let legacy_key = match target.legacy_key {
        Some(k) => k,
        None => return false,
    };
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_tool_has_entry_json(&path, legacy_key),
        McpConfigFormat::Toml => mcp_tool_has_entry_toml(&path, legacy_key),
    }
}

fn mcp_tool_register(home: &Path, target: &McpToolTarget, command: &str) -> Result<(), String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_register_json(&path, "retrivio", command),
        McpConfigFormat::Toml => mcp_register_toml(&path, "retrivio", command),
    }
}

fn mcp_tool_unregister(home: &Path, target: &McpToolTarget) -> Result<(), String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_unregister_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_unregister_toml(&path, "retrivio"),
    }
}

fn mcp_tool_remove_legacy(home: &Path, target: &McpToolTarget) -> Result<(), String> {
    let legacy_key = match target.legacy_key {
        Some(k) => k,
        None => return Ok(()),
    };
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_unregister_json(&path, legacy_key),
        McpConfigFormat::Toml => mcp_unregister_toml(&path, legacy_key),
    }
}

// ── Register / Unregister commands ───────────────────────────────────

fn run_mcp_register(args: &[OsString]) {
    let auto_yes = args.iter().any(|a| a == "-y" || a == "--yes");
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();

    if !interactive && !auto_yes {
        eprintln!("hint: not a terminal — use --yes to skip prompts");
        process::exit(2);
    }

    let home = match home_dir() {
        Some(h) => h,
        None => {
            eprintln!("error: cannot determine home directory");
            process::exit(1);
        }
    };

    let cmd_path = resolve_retrivio_command_path_native();
    let command = match &cmd_path {
        Some(p) => p.to_string_lossy().to_string(),
        None => {
            eprintln!("error: cannot resolve retrivio command path");
            process::exit(1);
        }
    };

    println!("  command: {}", command);

    let mut registered: Vec<&str> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();

    for target in MCP_TOOL_TARGETS {
        if !mcp_tool_is_installed(&home, target) {
            println!("  {} - not detected", target.name);
            continue;
        }

        // handle legacy entry
        if mcp_tool_has_legacy_entry(&home, target) {
            let legacy_key = target.legacy_key.unwrap();
            let do_replace = if auto_yes {
                true
            } else {
                match prompt_yes_no(
                    &format!(
                        "  replace '{}' with 'retrivio' in {}?",
                        legacy_key, target.name
                    ),
                    true,
                ) {
                    Ok(v) => v,
                    Err(_) => false,
                }
            };
            if do_replace {
                if let Err(e) = mcp_tool_remove_legacy(&home, target) {
                    eprintln!("  error removing legacy entry from {}: {}", target.name, e);
                }
                if let Err(e) = mcp_tool_register(&home, target, &command) {
                    eprintln!("  error registering with {}: {}", target.name, e);
                    skipped.push(target.name);
                } else {
                    println!("  {} - replaced {}, registered", target.name, legacy_key);
                    registered.push(target.name);
                }
                continue;
            } else {
                skipped.push(target.name);
                println!("  {} - skipped", target.name);
                continue;
            }
        }

        // already registered?
        if mcp_tool_has_entry(&home, target) {
            let current_cmd = mcp_tool_current_command(&home, target);
            if current_cmd.as_deref() == Some(command.as_str()) {
                println!("  {} - already registered", target.name);
                registered.push(target.name);
                continue;
            }
            // different command — prompt to update
            let old_cmd = current_cmd.unwrap_or_else(|| "unknown".to_string());
            let do_update = if auto_yes {
                true
            } else {
                match prompt_yes_no(
                    &format!(
                        "  update {} to use {}? (currently {})",
                        target.name, command, old_cmd
                    ),
                    true,
                ) {
                    Ok(v) => v,
                    Err(_) => false,
                }
            };
            if do_update {
                if let Err(e) = mcp_tool_register(&home, target, &command) {
                    eprintln!("  error updating {}: {}", target.name, e);
                    skipped.push(target.name);
                } else {
                    println!("  {} - updated", target.name);
                    registered.push(target.name);
                }
            } else {
                skipped.push(target.name);
                println!("  {} - skipped", target.name);
            }
            continue;
        }

        // not registered — prompt
        let do_register = if auto_yes {
            true
        } else {
            match prompt_yes_no(&format!("  register with {}?", target.name), true) {
                Ok(v) => v,
                Err(_) => false,
            }
        };
        if do_register {
            if let Err(e) = mcp_tool_register(&home, target, &command) {
                eprintln!("  error registering with {}: {}", target.name, e);
                skipped.push(target.name);
            } else {
                println!("  {} - registered", target.name);
                registered.push(target.name);
            }
        } else {
            skipped.push(target.name);
            println!("  {} - skipped", target.name);
        }
    }

    // summary
    println!();
    if !registered.is_empty() {
        println!("registered: {}", registered.join(", "));
    }
    if !skipped.is_empty() {
        println!("skipped: {}", skipped.join(", "));
    }
}

fn run_mcp_unregister(args: &[OsString]) {
    let auto_yes = args.iter().any(|a| a == "-y" || a == "--yes");
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();

    if !interactive && !auto_yes {
        eprintln!("hint: not a terminal — use --yes to skip prompts");
        process::exit(2);
    }

    let home = match home_dir() {
        Some(h) => h,
        None => {
            eprintln!("error: cannot determine home directory");
            process::exit(1);
        }
    };

    let mut unregistered: Vec<&str> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();

    for target in MCP_TOOL_TARGETS {
        if !mcp_tool_has_entry(&home, target) {
            println!("  {} - not registered", target.name);
            continue;
        }

        let do_unregister = if auto_yes {
            true
        } else {
            match prompt_yes_no(&format!("  unregister from {}?", target.name), true) {
                Ok(v) => v,
                Err(_) => false,
            }
        };
        if do_unregister {
            if let Err(e) = mcp_tool_unregister(&home, target) {
                eprintln!("  error unregistering from {}: {}", target.name, e);
                skipped.push(target.name);
            } else {
                println!("  {} - unregistered", target.name);
                unregistered.push(target.name);
            }
        } else {
            skipped.push(target.name);
            println!("  {} - skipped", target.name);
        }
    }

    println!();
    if !unregistered.is_empty() {
        println!("unregistered: {}", unregistered.join(", "));
    }
    if !skipped.is_empty() {
        println!("skipped: {}", skipped.join(", "));
    }
}

fn run_mcp_doctor() {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let dbfile = db_path(&cwd);
    let cfg_ok = cfg_path.exists();
    let db_ok = dbfile.exists();

    let cmd_path = resolve_retrivio_command_path_native();
    let cmd_ok = cmd_path.is_some();
    let cmd_display = cmd_path
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "-".to_string());

    let ready = cfg_ok && db_ok && cmd_ok;
    println!("mcp doctor");
    println!("  status:   {}", if ready { "ready" } else { "not ready" });
    println!("  command:  {}", cmd_display);
    println!("  config:   {}", yes_no(cfg_ok));
    println!("  db:       {}", yes_no(db_ok));
    if !ready {
        if !cmd_ok {
            println!("  hint: ensure a stable retrivio command path is available");
        }
        if !cfg_ok || !db_ok {
            println!(
                "  hint: run `retrivio init` (and optionally `retrivio index`) in this workspace"
            );
        }
    }
    // Check for incomplete chunk vectors
    if db_ok {
        if let Ok(conn) = open_db_read_only(&dbfile) {
            let model_key = model_key_for_cfg(&cfg);
            match count_incomplete_vector_projects(&conn, &model_key) {
                Ok(gaps) if !gaps.is_empty() => {
                    println!(
                        "  warning: {} project(s) have incomplete embeddings",
                        gaps.len()
                    );
                    for (name, have, total) in &gaps {
                        println!("    {} ({}/{} chunks have embeddings)", name, have, total);
                    }
                    println!("    run `retrivio index` to repair");
                }
                _ => {}
            }
        }
    }

    println!();
    if let Some(home) = home_dir() {
        for target in MCP_TOOL_TARGETS {
            let status = if !mcp_tool_is_installed(&home, target) {
                "not detected"
            } else if mcp_tool_has_entry(&home, target) {
                "registered"
            } else {
                "not registered"
            };
            println!("  {:<12} {}", target.name, status);
        }
    }
}

fn mcp_success_result(data: Value) -> Value {
    let text = serde_json::to_string_pretty(&data).unwrap_or_else(|_| "{}".to_string());
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
        "isError": false
    })
}

fn mcp_error_response(id: Value, code: i64, message: &str) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

fn mcp_response(id: Value, result: Value) -> Value {
    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn mcp_tool_specs() -> Vec<Value> {
    vec![
        serde_json::json!({
            "name": "search_projects",
            "description": "Semantic search across tracked projects with ranked evidence docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 12}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "search_files",
            "description": "Semantic search across indexed files (with project context).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "search_chunks",
            "description": "Semantic+keyword search across indexed chunks/segments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 30}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "search_symbols",
            "description": "Search for function, class, method, struct, trait, and other symbol definitions by name. Uses AST-extracted symbol index with FTS5 for fast prefix matching.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name or partial name to search for (e.g., 'validateToken', 'AuthMiddleware')"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "get_related_chunks",
            "description": "Retrieve chunks related to a source chunk using graph lineage and semantic similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["chunk_id"]
            }
        }),
        serde_json::json!({
            "name": "read_chunk",
            "description": "Read full indexed text for a chunk id (token-budgeted by max_chars).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "max_chars": {"type": "integer", "default": 8000}
                },
                "required": ["chunk_id"]
            }
        }),
        serde_json::json!({
            "name": "read_document",
            "description": "Read an indexed document by path, reconstructed from chunk sequence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 120000}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "pack_context",
            "description": "Build a ranked context package with top chunks, related chunks, and optional full docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "budget_chars": {"type": "integer", "default": 12000},
                    "seed_limit": {"type": "integer", "default": 8},
                    "related_per_seed": {"type": "integer", "default": 3},
                    "include_docs": {"type": "boolean", "default": false},
                    "doc_max_chars": {"type": "integer", "default": 12000}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "list_relation_feedback",
            "description": "List relation curation feedback for a source chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "decision": {"type": "string", "description": "Optional: suppressed|active"},
                    "quality": {"type": "string", "description": "Optional: good|weak|wrong|unspecified"},
                    "limit": {"type": "integer", "default": 120}
                },
                "required": ["chunk_id"]
            }
        }),
        serde_json::json!({
            "name": "suppress_relation",
            "description": "Suppress a relation from source chunk to target chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation"]
            }
        }),
        serde_json::json!({
            "name": "restore_relation",
            "description": "Restore a previously suppressed relation from source chunk to target chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation"]
            }
        }),
        serde_json::json!({
            "name": "set_relation_quality",
            "description": "Set relation quality label for a source->target relation (good/weak/wrong/unspecified).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_chunk_id": {"type": "integer"},
                    "target_chunk_id": {"type": "integer"},
                    "relation": {"type": "string"},
                    "quality_label": {"type": "string"},
                    "note": {"type": "string"}
                },
                "required": ["source_chunk_id", "target_chunk_id", "relation", "quality_label"]
            }
        }),
        serde_json::json!({
            "name": "get_project_neighbors",
            "description": "Return relationship graph neighbors for a project path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "list_tracked_roots",
            "description": "List tracked root directories that feed indexing/search.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        serde_json::json!({
            "name": "add_tracked_root",
            "description": "Track a new root directory and optionally index it immediately.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "refresh": {"type": "boolean", "default": true}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "remove_tracked_root",
            "description": "Stop tracking a root directory and optionally refresh index.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "refresh": {"type": "boolean", "default": true}
                },
                "required": ["path"]
            }
        }),
        serde_json::json!({
            "name": "run_incremental_index",
            "description": "Run incremental index across all tracked roots.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        serde_json::json!({
            "name": "run_forced_refresh",
            "description": "Force-refresh all tracked roots or a supplied subset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }),
    ]
}

fn mcp_status_resource() -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp)?;
    let roots = list_tracked_roots_conn(&conn)?;
    let projects: i64 = conn
        .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
        .unwrap_or(0);
    let reembed_required = reembed_requirement_reason(&conn, &cfg)?;
    Ok(format!(
        "root={}\nembed_backend={}\nembed_model={}\nretrieval_backend={}\nreembed_required={}\ntracked_roots={}\nprojects={}\ndb={}\n",
        cfg.root.to_string_lossy(),
        cfg.embed_backend,
        cfg.embed_model,
        cfg.retrieval_backend,
        if let Some(reason) = reembed_required {
            reason
        } else {
            "no".to_string()
        },
        roots.len(),
        projects,
        dbp.display()
    ))
}

fn mcp_tool_call(name: &str, args: &Value) -> Result<Value, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_rw(&dbp)?;

    match name {
        "search_projects" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(12)
                .clamp(1, 50) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_projects")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_projects")?;
            let rows = rank_projects_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
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
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "frecency": item.frecency,
                        "graph": item.graph,
                        "evidence": evidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({"query": query, "count": results.len(), "results": results}))
        }
        "search_files" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_files")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_files")?;
            let rows = rank_files_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows
                .into_iter()
                .map(|item| {
                    let evidence: Vec<Value> = item
                        .evidence
                        .into_iter()
                        .map(|ev| {
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
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "path": item.path,
                        "project_path": item.project_path,
                        "doc_rel_path": item.doc_rel_path,
                        "chunk_id": item.chunk_id,
                        "chunk_index": item.chunk_index,
                        "score": item.score,
                        "semantic": item.semantic,
                        "lexical": item.lexical,
                        "graph": item.graph,
                        "relation": item.relation,
                        "quality": item.quality,
                        "excerpt": item.excerpt,
                        "evidence": evidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({"query": query, "count": results.len(), "results": results}))
        }
        "search_chunks" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(30)
                .clamp(1, 200) as usize;
            ensure_native_embed_backend(&cfg, "mcp search_chunks")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_chunks")?;
            let rows = rank_chunks_native(&conn, &cfg, &query, limit)?;
            let results: Vec<Value> = rows.iter().map(ranked_chunk_result_json).collect();
            Ok(serde_json::json!({
                "schema": chunk_search_schema(),
                "query": query,
                "count": results.len(),
                "results": results
            }))
        }
        "search_symbols" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let results = search_symbols_fts(&conn, &query, limit)?;
            Ok(serde_json::json!({
                "query": query,
                "count": results.len(),
                "results": results
            }))
        }
        "get_related_chunks" => {
            let chunk_id = args.get("chunk_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if chunk_id <= 0 {
                return Err("chunk_id must be a positive integer".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 200) as usize;
            ensure_native_embed_backend(&cfg, "mcp get_related_chunks")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp get_related_chunks")?;
            let (source, rows) = related_chunks_native(&conn, &cfg, chunk_id, limit)?;
            let results: Vec<Value> = rows.iter().map(related_chunk_result_json).collect();
            Ok(serde_json::json!({
                "schema": chunk_related_schema(),
                "source": source_chunk_json(&source),
                "count": results.len(),
                "results": results
            }))
        }
        "read_chunk" => {
            let chunk_id = args.get("chunk_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if chunk_id <= 0 {
                return Err("chunk_id must be a positive integer".to_string());
            }
            let max_chars = args
                .get("max_chars")
                .and_then(|v| v.as_i64())
                .unwrap_or(8_000)
                .clamp(200, 500_000) as usize;
            let chunk = indexed_chunk_by_id(&conn, chunk_id)?
                .ok_or_else(|| format!("chunk {} not found", chunk_id))?;
            let (text, truncated, text_chars) = truncate_text_chars(&chunk.text, max_chars);
            let returned_chars = text.chars().count();
            Ok(serde_json::json!({
                "schema": chunk_get_schema(),
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "path": chunk.doc_path,
                    "project_path": chunk.project_path,
                    "doc_rel_path": chunk.doc_rel_path,
                    "doc_mtime": chunk.doc_mtime,
                    "token_count": chunk.token_count,
                    "text_chars": text_chars,
                    "returned_chars": returned_chars,
                    "truncated": truncated,
                    "text": text
                }
            }))
        }
        "read_document" => {
            let raw_path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if raw_path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let max_chars = args
                .get("max_chars")
                .and_then(|v| v.as_i64())
                .unwrap_or(120_000)
                .clamp(500, 2_000_000) as usize;
            let normalized_path = normalize_path(&raw_path).to_string_lossy().to_string();
            let mut chunks = indexed_doc_chunks_by_path(&conn, &normalized_path)?;
            if chunks.is_empty() && raw_path != normalized_path {
                chunks = indexed_doc_chunks_by_path(&conn, &raw_path)?;
            }
            if chunks.is_empty() {
                return Err(format!("document path is not indexed: {}", raw_path));
            }
            let first = &chunks[0];
            let mut full_text = String::new();
            let mut token_total: i64 = 0;
            let mut chunk_refs: Vec<Value> = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                full_text.push_str(&chunk.text);
                token_total += chunk.token_count;
                chunk_refs.push(serde_json::json!({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }));
            }
            let (text, truncated, text_chars) = truncate_text_chars(&full_text, max_chars);
            let returned_chars = text.chars().count();
            Ok(serde_json::json!({
                "schema": doc_read_schema(),
                "path": first.doc_path.clone(),
                "project_path": first.project_path.clone(),
                "doc_rel_path": first.doc_rel_path.clone(),
                "doc_mtime": first.doc_mtime,
                "chunk_count": chunks.len(),
                "token_count": token_total,
                "text_chars": text_chars,
                "returned_chars": returned_chars,
                "truncated": truncated,
                "chunks": chunk_refs,
                "text": text
            }))
        }
        "pack_context" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if query.is_empty() {
                return Err("query must be non-empty".to_string());
            }
            ensure_native_embed_backend(&cfg, "mcp pack_context")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp pack_context")?;
            let options = ContextPackOptions {
                budget_chars: args
                    .get("budget_chars")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(12_000)
                    .clamp(1_000, 400_000) as usize,
                seed_limit: args
                    .get("seed_limit")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(8)
                    .clamp(1, 40) as usize,
                related_per_seed: args
                    .get("related_per_seed")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(3)
                    .clamp(0, 12) as usize,
                include_docs: args
                    .get("include_docs")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                doc_max_chars: args
                    .get("doc_max_chars")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(12_000)
                    .clamp(1_000, 500_000) as usize,
            };
            build_context_pack_native(&conn, &cfg, &query, options)
        }
        "list_relation_feedback" => {
            let chunk_id = args.get("chunk_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if chunk_id <= 0 {
                return Err("chunk_id must be a positive integer".to_string());
            }
            let decision = args
                .get("decision")
                .and_then(|v| v.as_str())
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref d) = decision {
                if d != "suppressed" && d != "active" {
                    return Err("decision must be 'suppressed' or 'active'".to_string());
                }
            }
            let quality = args
                .get("quality")
                .and_then(|v| v.as_str())
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty());
            if let Some(ref qv) = quality {
                if normalize_relation_quality_label(qv).is_none() {
                    return Err(
                        "quality must be one of: good, weak, wrong, unspecified".to_string()
                    );
                }
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(120)
                .clamp(1, 2000) as usize;
            let rows = list_chunk_relation_feedback(
                &conn,
                chunk_id,
                decision.as_deref(),
                quality.as_deref(),
                limit,
            )?;
            let results: Vec<Value> = rows.iter().map(relation_feedback_row_json).collect();
            Ok(serde_json::json!({
                "source_chunk_id": chunk_id,
                "decision": decision.unwrap_or_else(|| "all".to_string()),
                "quality": quality.unwrap_or_else(|| "all".to_string()),
                "count": results.len(),
                "results": results
            }))
        }
        "suppress_relation" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "suppressed",
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "decision": "suppressed",
                "note": note
            }))
        }
        "restore_relation" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            apply_chunk_relation_decision(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                "active",
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "decision": "active",
                "note": note
            }))
        }
        "set_relation_quality" => {
            let source_chunk_id = args
                .get("source_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let target_chunk_id = args
                .get("target_chunk_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let relation = args
                .get("relation")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let quality_label = args
                .get("quality_label")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("quality").and_then(|v| v.as_str()))
                .unwrap_or_default()
                .trim()
                .to_string();
            if source_chunk_id <= 0 || target_chunk_id <= 0 || relation.is_empty() {
                return Err(
                    "source_chunk_id, target_chunk_id, and relation are required".to_string(),
                );
            }
            let normalized = normalize_relation_quality_label(&quality_label)
                .ok_or_else(|| {
                    "quality_label must be one of: good, weak, wrong, unspecified".to_string()
                })?
                .to_string();
            let note = args
                .get("note")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let persisted = set_chunk_relation_quality(
                &conn,
                source_chunk_id,
                target_chunk_id,
                &relation,
                &normalized,
                &note,
                "mcp",
                now_ts(),
            )?;
            Ok(serde_json::json!({
                "ok": true,
                "source_chunk_id": source_chunk_id,
                "target_chunk_id": target_chunk_id,
                "relation": relation,
                "quality_label": persisted,
                "note": note
            }))
        }
        "get_project_neighbors" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(20)
                .clamp(1, 200) as usize;
            let target = normalize_path(&path).to_string_lossy().to_string();
            let rows = list_neighbors_by_path(&conn, &target, limit)?;
            let neighbors: Vec<Value> = rows
                .into_iter()
                .map(|(dst, kind, weight)| serde_json::json!({"dst": dst, "kind": kind, "weight": weight}))
                .collect();
            Ok(
                serde_json::json!({"path": target, "count": neighbors.len(), "neighbors": neighbors}),
            )
        }
        "list_tracked_roots" => {
            let rows = list_tracked_roots_conn(&conn)?;
            let roots: Vec<String> = rows
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            Ok(serde_json::json!({"count": roots.len(), "roots": roots}))
        }
        "add_tracked_root" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let refresh = args
                .get("refresh")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let root = normalize_path(&path);
            if !root.exists() || !root.is_dir() {
                return Err(format!("path is not a directory: {}", root.display()));
            }
            ensure_tracked_root_conn(&conn, &root, now_ts())?;
            let mut out = serde_json::json!({"added": root.to_string_lossy(), "refreshed": false});
            if refresh {
                ensure_native_embed_backend(&cfg, "mcp add_tracked_root refresh")?;
                ensure_retrieval_backend_ready(&cfg, true, "mcp add_tracked_root refresh")?;
                let mut force_paths = HashSet::new();
                force_paths.insert(root.clone());
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    Some(vec![root.clone()]),
                    true,
                    force_paths,
                    false,
                    false,
                    "mcp add root refresh",
                )?;
                out["refreshed"] = Value::Bool(true);
                out["stats"] = stats_payload_json(&stats);
            }
            Ok(out)
        }
        "remove_tracked_root" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if path.is_empty() {
                return Err("path must be non-empty".to_string());
            }
            let refresh = args
                .get("refresh")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let root = normalize_path(&path).to_string_lossy().to_string();
            let removed = remove_tracked_root(&dbp, &normalize_path(&root))?;
            let mut out = serde_json::json!({"removed": removed, "path": root, "refreshed": false});
            if refresh {
                ensure_native_embed_backend(&cfg, "mcp remove_tracked_root refresh")?;
                ensure_retrieval_backend_ready(&cfg, true, "mcp remove_tracked_root refresh")?;
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    false,
                    HashSet::new(),
                    true,
                    false,
                    "mcp remove root refresh",
                )?;
                out["refreshed"] = Value::Bool(true);
                out["stats"] = stats_payload_json(&stats);
            }
            Ok(out)
        }
        "run_incremental_index" => {
            ensure_native_embed_backend(&cfg, "mcp incremental index")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp incremental index")?;
            let stats = run_native_index(
                &cwd,
                &cfg,
                None,
                false,
                HashSet::new(),
                true,
                false,
                "mcp incremental index",
            )?;
            Ok(serde_json::json!({"mode": "incremental", "stats": stats_payload_json(&stats)}))
        }
        "run_forced_refresh" => {
            ensure_native_embed_backend(&cfg, "mcp forced refresh")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp forced refresh")?;
            let paths: Vec<String> = args
                .get("paths")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if paths.is_empty() {
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    None,
                    true,
                    HashSet::new(),
                    true,
                    false,
                    "mcp forced refresh",
                )?;
                return Ok(
                    serde_json::json!({"mode": "forced_all", "stats": stats_payload_json(&stats)}),
                );
            }
            let roots = unique_valid_dirs(&paths);
            if roots.is_empty() {
                return Err("no valid directory paths provided".to_string());
            }
            let force_paths: HashSet<PathBuf> = roots.iter().cloned().collect();
            let stats = run_native_index(
                &cwd,
                &cfg,
                Some(roots.clone()),
                true,
                force_paths,
                false,
                false,
                "mcp forced refresh",
            )?;
            let paths_out: Vec<String> = roots
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            Ok(
                serde_json::json!({"mode": "forced_scoped", "paths": paths_out, "stats": stats_payload_json(&stats)}),
            )
        }
        _ => Err(format!("unknown tool '{}'", name)),
    }
}

/// Auto-detecting MCP frame reader: supports both Content-Length (LSP) framing
/// and newline-delimited JSON (NDJSON) used by Claude Code v2.x.
fn read_mcp_frame<R: BufRead + Read>(
    reader: &mut R,
    use_ndjson: &mut Option<bool>,
) -> Result<Option<Value>, String> {
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("failed reading MCP input: {}", e))?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Auto-detect: if the line starts with '{', it's NDJSON.
        if trimmed.starts_with('{') {
            if use_ndjson.is_none() {
                *use_ndjson = Some(true);
            }
            let value: Value =
                serde_json::from_str(trimmed).map_err(|e| format!("invalid MCP JSON: {}", e))?;
            return Ok(Some(value));
        }
        // Otherwise, it's a Content-Length header line.
        if use_ndjson.is_none() {
            *use_ndjson = Some(false);
        }
        // Parse headers until blank line.
        let mut headers = HashMap::new();
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
        loop {
            let mut hline = String::new();
            let hn = reader
                .read_line(&mut hline)
                .map_err(|e| format!("failed reading MCP header: {}", e))?;
            if hn == 0 {
                return Ok(None);
            }
            let ht = hline.trim();
            if ht.is_empty() {
                break;
            }
            if let Some((k, v)) = ht.split_once(':') {
                headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        let len = headers
            .get("content-length")
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| "MCP frame missing Content-Length".to_string())?;
        let mut body = vec![0u8; len];
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("failed reading MCP frame body: {}", e))?;
        let value = serde_json::from_slice::<Value>(&body)
            .map_err(|e| format!("invalid MCP JSON: {}", e))?;
        return Ok(Some(value));
    }
}

fn write_mcp_frame<W: Write>(writer: &mut W, value: &Value, ndjson: bool) -> Result<(), String> {
    if ndjson {
        let body = serde_json::to_string(value)
            .map_err(|e| format!("failed serializing MCP JSON: {}", e))?;
        writer
            .write_all(body.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .and_then(|_| writer.flush())
            .map_err(|e| format!("failed writing MCP frame: {}", e))
    } else {
        let body =
            serde_json::to_vec(value).map_err(|e| format!("failed serializing MCP JSON: {}", e))?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        writer
            .write_all(header.as_bytes())
            .and_then(|_| writer.write_all(&body))
            .and_then(|_| writer.flush())
            .map_err(|e| format!("failed writing MCP frame: {}", e))
    }
}

fn serve_mcp_native() -> Result<(), String> {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        eprintln!(
            "retrivio mcp serve: waiting for MCP client messages on stdio (Ctrl+C to exit)"
        );
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    let mut use_ndjson: Option<bool> = None;

    while let Some(msg) = read_mcp_frame(&mut reader, &mut use_ndjson)? {
        let ndjson = use_ndjson.unwrap_or(false);
        let id = msg.get("id").cloned();
        let method = msg
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if method.starts_with("notifications/") {
            continue;
        }
        let Some(id) = id else {
            continue;
        };
        let params = msg.get("params").cloned().unwrap_or_else(|| Value::Null);
        let response = match method.as_str() {
            "initialize" => {
                let protocol = params
                    .get("protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("2025-03-26");
                mcp_response(
                    id,
                    serde_json::json!({
                        "protocolVersion": protocol,
                        "capabilities": {
                            "tools": {"listChanged": false},
                            "resources": {"listChanged": false}
                        },
                        "serverInfo": {
                            "name": "retrivio",
                            "version": env!("CARGO_PKG_VERSION")
                        },
                        "instructions": "Semantic project memory server for local files. Use search to find relevant projects by meaning and evidence docs."
                    }),
                )
            }
            "ping" => mcp_response(id, serde_json::json!({})),
            "tools/list" => mcp_response(id, serde_json::json!({"tools": mcp_tool_specs()})),
            "tools/call" => {
                let tool_name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if tool_name.is_empty() {
                    mcp_error_response(id, -32602, "tools/call requires tool name")
                } else {
                    let args = params
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    match mcp_tool_call(&tool_name, &args) {
                        Ok(data) => mcp_response(id, mcp_success_result(data)),
                        Err(err) => mcp_error_response(id, -32000, &err),
                    }
                }
            }
            "resources/list" => mcp_response(
                id,
                serde_json::json!({
                    "resources": [{
                        "uri": "retrivio://status",
                        "name": "retrivio status",
                        "description": "Quick status snapshot for agent context hydration.",
                        "mimeType": "text/plain"
                    }]
                }),
            ),
            "resources/read" => {
                let uri = params
                    .get("uri")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if uri != "retrivio://status" {
                    mcp_error_response(id, -32602, "unknown resource uri")
                } else {
                    match mcp_status_resource() {
                        Ok(text) => mcp_response(
                            id,
                            serde_json::json!({
                                "contents": [{
                                    "uri": "retrivio://status",
                                    "mimeType": "text/plain",
                                    "text": text
                                }]
                            }),
                        ),
                        Err(err) => mcp_error_response(id, -32000, &err),
                    }
                }
            }
            "prompts/list" => mcp_response(id, serde_json::json!({"prompts": []})),
            _ => mcp_error_response(id, -32601, "method not found"),
        };
        write_mcp_frame(&mut writer, &response, ndjson)?;
    }
    Ok(())
}

fn run_self_test_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio self-test [--query <text>] [--lifecycle] [--timeout <seconds>]");
        println!("notes:");
        println!("  --lifecycle runs daemon lifecycle probes on temporary ports");
        return;
    }
    let mut query = "storage".to_string();
    let mut lifecycle = false;
    let mut timeout_s = 8u64;
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--query" => {
                i += 1;
                query = arg_value(args, i, "--query");
            }
            "--lifecycle" => {
                lifecycle = true;
            }
            "--timeout" => {
                i += 1;
                let raw = arg_value(args, i, "--timeout");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(2);
            }
            x if x.starts_with("--timeout=") => {
                let raw = x.trim_start_matches("--timeout=");
                timeout_s = raw.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("error: --timeout must be an integer number of seconds");
                    process::exit(2);
                });
                timeout_s = timeout_s.max(2);
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let mut ok = true;

    println!("self-test: config={}", config_path(&cwd).display());
    println!("self-test: db={}", dbp.display());
    if !config_path(&cwd).exists() {
        println!("self-test: config file missing");
        ok = false;
    }
    if !dbp.exists() {
        println!("self-test: database missing");
        ok = false;
    }

    println!("self-test: retrieval backend=lancedb (embedded)");

    if ensure_native_embed_backend(&cfg, "self-test search").is_ok() {
        if ensure_retrieval_backend_ready(&cfg, false, "self-test search").is_ok() {
            match open_db_rw(&dbp).and_then(|conn| rank_projects_native(&conn, &cfg, &query, 3)) {
                Ok(rows) => println!("self-test: search results={}", rows.len()),
                Err(err) => {
                    println!("self-test: search failed ({})", err);
                    ok = false;
                }
            }
        }
    } else {
        println!("self-test: search skipped (non-native embed backend)");
    }

    let mcp_cmd_ok = resolve_retrivio_command_path_native().is_some();
    println!("self-test: mcp command path={}", yes_no(mcp_cmd_ok));
    if !mcp_cmd_ok {
        ok = false;
    }

    println!("self-test: api command available=yes");
    println!("self-test: mcp command available=yes");

    if lifecycle {
        println!("self-test: lifecycle enabled=yes");
        match run_self_test_lifecycle_probe(&cwd, &cfg, &query, timeout_s) {
            Ok(lines) => {
                for line in lines {
                    println!("self-test:lifecycle: {}", line);
                }
            }
            Err(err) => {
                println!("self-test: lifecycle failed ({})", err);
                ok = false;
            }
        }
    }

    if !ok {
        process::exit(1);
    }
}

fn run_self_test_lifecycle_probe(
    cwd: &Path,
    cfg: &ConfigValues,
    query: &str,
    timeout_s: u64,
) -> Result<Vec<String>, String> {
    let mut lines: Vec<String> = Vec::new();
    let (daemon_port, daemon_pid) = run_self_test_daemon_probe(cwd, timeout_s)?;
    lines.push(format!(
        "daemon probe ok (host={} port={} pid={})",
        daemon_default_host(),
        daemon_port,
        daemon_pid
    ));

    lines.push("retrieval backend: lancedb (embedded, no probe needed)".to_string());
    Ok(lines)
}

fn run_self_test_daemon_probe(cwd: &Path, timeout_s: u64) -> Result<(u16, u32), String> {
    let host = daemon_default_host();
    let start_port = daemon_default_port().saturating_add(20);
    let port = find_free_port(&host, start_port, 300)
        .ok_or_else(|| "no free port available for daemon probe".to_string())?;
    let pid = spawn_api_daemon(cwd, &host, port)?;
    let deadline = Instant::now() + Duration::from_secs(timeout_s.max(2));
    let mut up = false;
    while Instant::now() < deadline {
        if api_health_host_port(&host, port) {
            up = true;
            break;
        }
        if !pid_is_alive(pid) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if !up {
        let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
        return Err(format!(
            "daemon probe failed to become healthy on {}:{} (pid={})",
            host, port, pid
        ));
    }
    let _ = run_shell_capture(&format!("kill {} >/dev/null 2>&1 || true", pid));
    let stop_deadline = Instant::now() + Duration::from_secs(timeout_s.max(2));
    let mut down = false;
    while Instant::now() < stop_deadline {
        if !api_health_host_port(&host, port) {
            down = true;
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    if !down {
        let _ = run_shell_capture(&format!("kill -9 {} >/dev/null 2>&1 || true", pid));
        let hard_deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < hard_deadline {
            if !api_health_host_port(&host, port) {
                down = true;
                break;
            }
            thread::sleep(Duration::from_millis(120));
        }
    }
    if !down {
        return Err(format!(
            "daemon probe endpoint {}:{} did not stop cleanly (pid={})",
            host, port, pid
        ));
    }
    Ok((port, pid))
}

fn run_index_with_strategy(
    cwd: &Path,
    cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
    force_all: bool,
    force_paths: Option<HashSet<PathBuf>>,
    remove_missing: bool,
    reason: &str,
) -> Result<(), String> {
    ensure_retrieval_backend_ready(cfg, true, reason)?;
    ensure_native_embed_backend(cfg, reason)?;

    let stats = run_native_index(
        cwd,
        cfg,
        scope_roots,
        force_all,
        force_paths.unwrap_or_default(),
        remove_missing,
        true,
        reason,
    )?;
    print_index_stats(&stats, cfg);
    Ok(())
}

#[derive(Default)]
struct IndexStats {
    total_projects: i64,
    updated_projects: i64,
    skipped_projects: i64,
    removed_projects: i64,
    vectorized_projects: i64,
    vector_failures: i64,
    tracked_roots: i64,
    graph_edges: i64,
    chunk_rows: i64,
    chunk_vectors: i64,
    retrieval_backend: String,
    retrieval_synced_chunks: i64,
    retrieval_error: String,
    elapsed_index_ms: u64,
    elapsed_sync_ms: u64,
    elapsed_total_ms: u64,
}

#[derive(Clone)]
struct ProjectDoc {
    path: PathBuf,
    title: String,
    summary: String,
    mtime: f64,
}

#[derive(Clone)]
struct ProjectChunk {
    doc_path: String,
    doc_rel_path: String,
    doc_mtime: f64,
    chunk_index: i64,
    token_count: i64,
    text_hash: String,
    text: String,
    // Code intelligence metadata (populated by AST chunker, empty for fallback chunks)
    chunk_kind: String,
    symbol_name: String,
    parent_context: String,
    line_start: i64,
    line_end: i64,
    context_header: String,
}

struct ProjectCorpus {
    doc: ProjectDoc,
    chunks: Vec<ProjectChunk>,
}

struct ExistingProject {
    id: i64,
    path: String,
    title: String,
    summary: String,
    project_mtime: f64,
}

#[derive(Default)]
struct LiveIndexProgress {
    completed_projects: AtomicUsize,
    current_project_idx: AtomicUsize,
    current_project_chunks_done: AtomicU64,
    current_project_chunks_total: AtomicU64,
    current_project_tokens_done: AtomicU64,
    current_project_tokens_total: AtomicU64,
    current_project_name: Mutex<String>,
    current_phase: Mutex<String>,
}

impl LiveIndexProgress {
    fn mark_project_done(&self) {
        self.completed_projects.fetch_add(1, Ordering::Relaxed);
    }

    fn start_project(
        &self,
        project_idx_1based: usize,
        project_name: &str,
        total_chunks: usize,
        total_tokens: u64,
    ) {
        self.set_phase("embed");
        self.current_project_idx
            .store(project_idx_1based, Ordering::Relaxed);
        self.current_project_chunks_done.store(0, Ordering::Relaxed);
        self.current_project_chunks_total
            .store(total_chunks as u64, Ordering::Relaxed);
        self.current_project_tokens_done.store(0, Ordering::Relaxed);
        self.current_project_tokens_total
            .store(total_tokens, Ordering::Relaxed);
        if let Ok(mut name) = self.current_project_name.lock() {
            *name = project_name.to_string();
        }
    }

    fn add_chunks_done(&self, n: usize) {
        self.current_project_chunks_done
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    fn add_tokens_done(&self, n: u64) {
        self.current_project_tokens_done
            .fetch_add(n, Ordering::Relaxed);
    }

    fn set_phase(&self, phase: &str) {
        if let Ok(mut cur) = self.current_phase.lock() {
            *cur = phase.to_string();
        }
    }
}

fn progress_spinner(elapsed: Duration) -> char {
    match ((elapsed.as_millis() / 250) % 4) as u8 {
        0 => '|',
        1 => '/',
        2 => '-',
        _ => '\\',
    }
}

fn format_ascii_bar(width: usize, ratio: f64) -> String {
    let w = width.max(8);
    let r = ratio.clamp(0.0, 1.0);
    let filled = ((w as f64) * r).round() as usize;
    let filled = filled.min(w);
    let mut out = String::with_capacity(w + 2);
    out.push('[');
    out.push_str(&"#".repeat(filled));
    out.push_str(&"-".repeat(w - filled));
    out.push(']');
    out
}

fn compact_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", (n as f64) / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", (n as f64) / 1_000.0)
    } else {
        n.to_string()
    }
}

fn render_live_progress_line(
    reason: &str,
    total_projects: usize,
    started_at: Instant,
    progress: &LiveIndexProgress,
) {
    if !progress_use_single_line() {
        return;
    }
    let completed = progress.completed_projects.load(Ordering::Relaxed);
    let current_idx = progress.current_project_idx.load(Ordering::Relaxed);
    let chunks_done = progress.current_project_chunks_done.load(Ordering::Relaxed);
    let chunks_total = progress
        .current_project_chunks_total
        .load(Ordering::Relaxed);
    let tokens_done = progress.current_project_tokens_done.load(Ordering::Relaxed);
    let tokens_total = progress
        .current_project_tokens_total
        .load(Ordering::Relaxed);
    let phase = progress
        .current_phase
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| "index".to_string());
    let m = embed_runtime_snapshot();
    let avg_ms = if m.latency_samples > 0 {
        m.latency_sum_ms / m.latency_samples
    } else {
        0
    };
    let elapsed = started_at.elapsed();
    let elapsed_s = elapsed.as_secs_f64().max(0.001);
    let req_rate = m.requests_succeeded as f64 / elapsed_s;
    let chunk_ratio = if chunks_total > 0 {
        chunks_done as f64 / (chunks_total as f64)
    } else {
        0.0
    };
    let project_ratio = if total_projects > 0 {
        ((completed as f64) + chunk_ratio) / (total_projects as f64)
    } else {
        1.0
    };
    let done_projects = completed.min(total_projects);
    let done_all = total_projects == 0 || done_projects >= total_projects;
    let bar_ratio = if done_all {
        project_ratio
    } else {
        project_ratio.min(0.999)
    };
    let mut pct = project_ratio.clamp(0.0, 1.0) * 100.0;
    if !done_all {
        pct = pct.min(99.9);
    }
    let bar = format_ascii_bar(18, bar_ratio);
    let spin = progress_spinner(elapsed);
    let line = format!(
        "{spin} {reason} {bar} {pct:5.1}% | phase {phase} | proj {done}/{total} act#{cur}/{total} | active ch {ch_done}/{ch_total} tok {tok_done}/{tok_total} emb {emb} | req {ok}/{fail} if {inflight} rt {retry} th {thr} | {rate:.1}/s avg {avg}ms max {max}ms",
        spin = spin,
        reason = reason,
        bar = bar,
        pct = pct,
        phase = phase,
        done = done_projects,
        total = total_projects,
        cur = current_idx.min(total_projects),
        ch_done = compact_count(chunks_done),
        ch_total = compact_count(chunks_total),
        tok_done = compact_count(tokens_done),
        tok_total = compact_count(tokens_total),
        emb = compact_count(m.texts_embedded),
        ok = compact_count(m.requests_succeeded),
        fail = compact_count(m.requests_failed),
        inflight = m.in_flight.max(0),
        retry = m.request_retries,
        thr = m.throttles,
        rate = req_rate,
        avg = avg_ms,
        max = m.latency_max_ms
    );
    let (cols, _) = terminal_size_stty();
    let display = clipped(&line, cols.saturating_sub(1));
    if let Ok(_g) = progress_io_lock().lock() {
        eprint!("\r\x1b[2K{}", display);
        let _ = std::io::stderr().flush();
    }
}

struct LiveProgressReporter {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LiveProgressReporter {
    fn start(
        reason: &str,
        total_projects: usize,
        started_at: Instant,
        progress: Arc<LiveIndexProgress>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let reason = reason.to_string();
        let interval = progress_heartbeat_interval();
        let handle = thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                thread::sleep(interval);
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                render_live_progress_line(&reason, total_projects, started_at, &progress);
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for LiveProgressReporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        progress_clear_line();
    }
}

fn run_native_index(
    cwd: &Path,
    cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
    force_all: bool,
    force_paths: HashSet<PathBuf>,
    remove_missing: bool,
    emit_progress: bool,
    reason: &str,
) -> Result<IndexStats, String> {
    let t_start = Instant::now();
    reset_embed_runtime_metrics();
    let dbp = db_path(cwd);
    let conn = open_db_rw(&dbp)?;
    if reason != "reembed" {
        ensure_reembed_ready(&conn, cfg, &format!("{} indexing", reason))?;
    }
    let roots = resolve_roots(&conn, cfg, scope_roots)?;
    if roots.is_empty() {
        return Err("No tracked roots configured. Add one with `retrivio add <path>`.".to_string());
    }
    let projects = discover_projects(&roots);
    let embedder = build_embedder(cfg)?;
    let model_key = embedder.model_key();
    let mode = if force_all {
        "forced refresh"
    } else {
        "incremental index"
    };
    if emit_progress {
        progress_clear_line();
        println!(
            "{}: {} started (backend={}, model={})",
            reason, mode, cfg.embed_backend, cfg.embed_model
        );
    }

    let mut stats = IndexStats {
        tracked_roots: roots.len() as i64,
        retrieval_backend: cfg.retrieval_backend.clone(),
        ..Default::default()
    };
    stats.total_projects = projects.len() as i64;
    let live_progress = Arc::new(LiveIndexProgress::default());
    live_progress.set_phase("scan");
    let live_started_at = Instant::now();
    let _live_reporter = if emit_progress {
        Some(LiveProgressReporter::start(
            reason,
            projects.len(),
            live_started_at,
            Arc::clone(&live_progress),
        ))
    } else {
        None
    };
    if emit_progress {
        render_live_progress_line(reason, projects.len(), live_started_at, &live_progress);
    }

    // If this model already has vectors, pre-open LanceDB at that known dimension.
    // Otherwise defer opening until we have real embeddings from the current run.
    if let Some(lance_dim) = vector_dim_from_sqlite(&conn, &model_key) {
        if let Err(e) = get_or_open_lance(cwd, lance_dim) {
            eprintln!(
                "warning: LanceDB open failed (vectors still written to sqlite): {}",
                e
            );
        }
    }

    let mut keep_paths: Vec<String> = Vec::new();
    let mut docs_by_id: HashMap<i64, ProjectDoc> = HashMap::new();
    let now = now_ts();

    // ── Phase 1: Determine which projects need updating (sequential, needs conn) ──
    struct ProjectWork {
        idx: usize,
        dir: PathBuf,
        latest_mtime: f64,
        exclude_abs: HashSet<PathBuf>,
    }
    let mut needs_update: Vec<ProjectWork> = Vec::new();

    for (idx, project_dir) in projects.iter().enumerate() {
        let project_path = project_dir.to_string_lossy().to_string();
        keep_paths.push(project_path.clone());
        let forced = force_all || is_under_any(project_dir, &force_paths);
        let project_excludes = project_excludes_for_path(project_dir, &roots);
        let latest_mtime = project_latest_mtime(project_dir, &project_excludes)?;
        let existing = get_project_by_path(&conn, &project_path)?;
        if let Some(row) = existing {
            if !forced && row.project_mtime >= latest_mtime {
                let project_vec_ready = has_project_vector(&conn, row.id, &model_key)?;
                let chunk_count = count_project_chunks(&conn, row.id)?;
                let chunk_vec_count = count_project_chunk_vectors(&conn, row.id, &model_key)?;
                let chunk_vec_ready = chunk_count == 0 || chunk_vec_count >= chunk_count;
                if project_vec_ready && chunk_vec_ready {
                    stats.skipped_projects += 1;
                    live_progress.mark_project_done();
                    docs_by_id.insert(
                        row.id,
                        ProjectDoc {
                            path: PathBuf::from(row.path),
                            title: row.title,
                            summary: row.summary,
                            mtime: row.project_mtime,
                        },
                    );
                    if emit_progress {
                        progress_clear_line();
                        println!(
                            "[{}/{}] skip {}",
                            idx + 1,
                            projects.len(),
                            project_dir
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("project")
                        );
                        render_live_progress_line(
                            reason,
                            projects.len(),
                            live_started_at,
                            &live_progress,
                        );
                    }
                    continue;
                }
                // Log when re-indexing due to incomplete vectors
                if emit_progress && !chunk_vec_ready && chunk_count > 0 {
                    progress_clear_line();
                    println!(
                        "[{}/{}] re-indexing {} (incomplete vectors: {}/{} chunks have embeddings)",
                        idx + 1,
                        projects.len(),
                        project_dir
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("project"),
                        chunk_vec_count,
                        chunk_count,
                    );
                    render_live_progress_line(
                        reason,
                        projects.len(),
                        live_started_at,
                        &live_progress,
                    );
                }
            }
        }
        needs_update.push(ProjectWork {
            idx,
            dir: project_dir.clone(),
            latest_mtime,
            exclude_abs: project_excludes,
        });
    }

    // ── Phase 2+3: Collect corpus + embed/store in bounded batches ──
    // Keep peak memory bounded by processing only INDEX_PARALLELISM corpora at once.
    live_progress.set_phase("collect");
    let max_chars = cfg.max_chars_per_project as usize;
    const INDEX_PARALLELISM: usize = 4;
    for work_batch in needs_update.chunks(INDEX_PARALLELISM) {
        let results =
            std::sync::Mutex::new(
                Vec::<(usize, PathBuf, f64, Result<ProjectCorpus, String>)>::new(),
            );
        std::thread::scope(|s| {
            let handles: Vec<_> = work_batch
                .iter()
                .map(|pw| {
                    let results = &results;
                    s.spawn(move || {
                        let result = collect_project_corpus(
                            &pw.dir,
                            max_chars,
                            pw.latest_mtime,
                            &pw.exclude_abs,
                        );
                        if let Ok(mut vec) = results.lock() {
                            vec.push((pw.idx, pw.dir.clone(), pw.latest_mtime, result));
                        }
                    })
                })
                .collect();
            for h in handles {
                let _ = h.join();
            }
        });
        let mut corpus_results = results.into_inner().unwrap_or_default();
        corpus_results.sort_by_key(|(idx, _, _, _)| *idx);

        for (idx, project_dir, _latest_mtime, corpus_result) in corpus_results {
            let corpus = corpus_result?;
            let project_name = project_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("project")
                .to_string();
            let project_total_tokens: u64 = corpus
                .chunks
                .iter()
                .map(|c| c.token_count.max(0) as u64)
                .sum();
            live_progress.start_project(
                idx + 1,
                &project_name,
                corpus.chunks.len(),
                project_total_tokens,
            );
            let project_id = upsert_project(
                &conn,
                &corpus.doc.path.to_string_lossy(),
                &corpus.doc.title,
                &corpus.doc.summary,
                corpus.doc.mtime,
                now,
            )?;
            stats.updated_projects += 1;
            docs_by_id.insert(project_id, corpus.doc.clone());

            {
                let summary_text = format!("{}\n{}", corpus.doc.title, corpus.doc.summary);
                let mut last_err = String::new();
                let mut succeeded = false;
                for attempt in 0..=3usize {
                    if attempt > 0 {
                        thread::sleep(Duration::from_millis(500 * 2u64.pow((attempt - 1) as u32)));
                    }
                    match embedder.embed_one(&summary_text) {
                        Ok(pvec) => {
                            set_project_vector(&conn, project_id, &model_key, &pvec)?;
                            stats.vectorized_projects += 1;
                            succeeded = true;
                            break;
                        }
                        Err(e) => {
                            last_err = e;
                        }
                    }
                }
                if !succeeded {
                    return Err(format!(
                        "failed embedding project summary for '{}' after retries: {}",
                        project_dir.display(),
                        last_err
                    ));
                }
            }

            let (rows, vecs, failures) = reindex_project_chunks(
                cwd,
                &conn,
                project_id,
                &model_key,
                embedder.as_ref(),
                &corpus.chunks,
                now,
                Some(&live_progress),
            )?;
            stats.chunk_rows += rows;
            stats.chunk_vectors += vecs;
            stats.vector_failures += failures;

            // Extract symbols and imports from code files via AST.
            // Iterates over unique doc_paths in the chunks to avoid re-parsing.
            {
                live_progress.set_phase("code-intel");
                let mut seen_docs: HashSet<String> = HashSet::new();
                let mut total_symbols = 0usize;
                let mut total_imports = 0usize;
                let project_file_list: Vec<String> = corpus
                    .chunks
                    .iter()
                    .map(|c| c.doc_rel_path.clone())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                let _ = conn.execute_batch("BEGIN TRANSACTION;");
                for chunk in &corpus.chunks {
                    if !seen_docs.insert(chunk.doc_path.clone()) {
                        continue;
                    }
                    let path = Path::new(&chunk.doc_path);
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    let lang = code_intel::language_for_extension(&ext);
                    if lang.is_none() {
                        continue;
                    }
                    let lang = lang.unwrap();
                    if let Some(source) = read_for_index(path) {
                        let symbols = code_intel::extract_symbols(path, &source);
                        if !symbols.is_empty() {
                            let stored = match store_project_symbols(
                                &conn,
                                project_id,
                                &symbols,
                                &chunk.doc_path,
                                &chunk.doc_rel_path,
                            ) {
                                Ok(n) => n,
                                Err(e) => {
                                    let _ = conn.execute_batch("ROLLBACK;");
                                    return Err(e);
                                }
                            };
                            total_symbols += stored;
                        }
                        let raw_imports = code_intel::extract_imports(path, &source);
                        if !raw_imports.is_empty() {
                            let stored = match store_file_imports(
                                &conn,
                                project_id,
                                &chunk.doc_path,
                                lang,
                                path,
                                &project_dir,
                                &raw_imports,
                                &project_file_list,
                            ) {
                                Ok(n) => n,
                                Err(e) => {
                                    let _ = conn.execute_batch("ROLLBACK;");
                                    return Err(e);
                                }
                            };
                            total_imports += stored;
                        }
                    }
                }
                let _ = conn.execute_batch("COMMIT;");
                let _ = total_symbols;
                let _ = total_imports;
            }

            if emit_progress {
                progress_clear_line();
                println!(
                    "[{}/{}] index {} chunks={} chunk_vecs={} vector_failures={}",
                    idx + 1,
                    projects.len(),
                    project_name,
                    rows,
                    vecs,
                    failures
                );
                render_live_progress_line(reason, projects.len(), live_started_at, &live_progress);
            }
            live_progress.mark_project_done();
            if emit_progress {
                render_live_progress_line(reason, projects.len(), live_started_at, &live_progress);
            }
        }
    }

    if remove_missing {
        live_progress.set_phase("cleanup");
        stats.removed_projects = remove_projects_not_in(&conn, &keep_paths)?;
    }
    live_progress.set_phase("graph-edges");
    stats.graph_edges = rebuild_relationship_edges(&conn, &docs_by_id)?;
    live_progress.set_phase("finalize");
    let t_before_sync = Instant::now();
    stats.elapsed_index_ms = t_before_sync.duration_since(t_start).as_millis() as u64;
    // LanceDB vectors are written inline during reindex_project_chunks;
    // report count for diagnostics.
    if let Ok(n) = with_lance_store(|store| lance_store::count(store)) {
        stats.retrieval_synced_chunks = n as i64;
    }
    stats.elapsed_sync_ms = t_before_sync.elapsed().as_millis() as u64;
    if !app_state_bool(&conn, APP_STATE_REEMBED_REQUIRED)? {
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &model_key)?;
    }

    conn.execute_batch("PRAGMA optimize;")
        .map_err(|e| format!("database optimize failed: {}", e))?;
    stats.elapsed_total_ms = t_start.elapsed().as_millis() as u64;
    Ok(stats)
}

fn format_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

fn print_index_stats(stats: &IndexStats, cfg: &ConfigValues) {
    println!("config root: {}", cfg.root.display());
    println!("tracked roots indexed: {}", stats.tracked_roots);
    println!("projects found: {}", stats.total_projects);
    println!("projects updated: {}", stats.updated_projects);
    println!("projects skipped (unchanged): {}", stats.skipped_projects);
    println!("projects removed: {}", stats.removed_projects);
    println!("vectors refreshed: {}", stats.vectorized_projects);
    println!("chunks indexed: {}", stats.chunk_rows);
    println!("chunk vectors refreshed: {}", stats.chunk_vectors);
    println!("graph edges refreshed: {}", stats.graph_edges);
    println!("retrieval backend: {}", stats.retrieval_backend);
    println!("retrieval chunks synced: {}", stats.retrieval_synced_chunks);
    if !stats.retrieval_error.is_empty() {
        println!("retrieval sync warning: {}", stats.retrieval_error);
    }
    if stats.vector_failures > 0 {
        println!("vector failures: {}", stats.vector_failures);
    }
    println!(
        "timing: index={}, sync={}, total={}",
        format_duration_ms(stats.elapsed_index_ms),
        format_duration_ms(stats.elapsed_sync_ms),
        format_duration_ms(stats.elapsed_total_ms),
    );
}

#[derive(Clone)]
struct EvidenceHit {
    chunk_id: i64,
    chunk_index: i64,
    doc_path: String,
    doc_rel_path: String,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

#[derive(Clone)]
struct RankedResult {
    path: String,
    score: f64,
    lexical: f64,
    semantic: f64,
    frecency: f64,
    graph: f64,
    evidence: Vec<EvidenceHit>,
}

#[derive(Clone)]
struct RankedFileResult {
    path: String,
    project_path: String,
    doc_rel_path: String,
    chunk_id: i64,
    chunk_index: i64,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
    evidence: Vec<EvidenceHit>,
}

#[derive(Clone)]
struct RankedChunkResult {
    chunk_id: i64,
    chunk_index: i64,
    path: String,
    project_path: String,
    doc_rel_path: String,
    score: f64,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

#[derive(Clone)]
struct RelatedChunkResult {
    chunk_id: i64,
    chunk_index: i64,
    path: String,
    project_path: String,
    doc_rel_path: String,
    relation: String,
    relation_weight: f64,
    relation_quality: String,
    relation_quality_multiplier: f64,
    score: f64,
    semantic: f64,
    lexical: f64,
    quality: f64,
    excerpt: String,
}

fn chunk_search_schema() -> &'static str {
    "chunk-search-v1"
}

fn chunk_related_schema() -> &'static str {
    "chunk-related-v1"
}

fn chunk_get_schema() -> &'static str {
    "chunk-get-v1"
}

fn doc_read_schema() -> &'static str {
    "doc-read-v1"
}

fn ranked_chunk_result_json(item: &RankedChunkResult) -> Value {
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
    })
}

fn related_chunk_result_json(item: &RelatedChunkResult) -> Value {
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
    })
}

#[derive(Clone)]
struct ChunkSignal {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    semantic: f64,
    lexical: f64,
    graph: f64,
    relation: String,
    quality: f64,
    excerpt: String,
}

fn print_project_results(results: &[RankedResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    score={:.3} semantic={:.3} lexical={:.3} frecency={:.3} graph={:.3}",
            idx + 1,
            item.path,
            item.score,
            item.semantic,
            item.lexical,
            item.frecency,
            item.graph
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

fn print_file_results(results: &[RankedFileResult]) {
    for (idx, item) in results.iter().enumerate() {
        println!(
            "{:>2}. {}\n    project={}\n    chunk_id={} chunk_index={}\n    score={:.3} semantic={:.3} lexical={:.3} graph={:.3} relation={} quality={:.2}\n    {}",
            idx + 1,
            item.path,
            item.project_path,
            item.chunk_id,
            item.chunk_index,
            item.score,
            item.semantic,
            item.lexical,
            item.graph,
            item.relation,
            item.quality,
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

#[derive(Clone)]
struct QueryEmbedCacheEntry {
    cached_at: Instant,
    vector: Vec<f32>,
}

static QUERY_EMBED_CACHE: OnceLock<Mutex<HashMap<String, QueryEmbedCacheEntry>>> = OnceLock::new();

fn query_embed_cache() -> &'static Mutex<HashMap<String, QueryEmbedCacheEntry>> {
    QUERY_EMBED_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn query_embed_cache_limit() -> usize {
    env::var("RETRIVIO_QUERY_EMBED_CACHE_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(4096)
        .clamp(64, 100_000)
}

fn query_embed_cache_ttl() -> Duration {
    let sec = env::var("RETRIVIO_QUERY_EMBED_CACHE_TTL_SEC")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(3600); // 1 hour (was 15 minutes)
    Duration::from_secs(sec.clamp(15, 86_400))
}

fn model_key_for_cfg(cfg: &ConfigValues) -> String {
    match cfg.embed_backend.as_str() {
        "ollama" => {
            let model = if cfg.embed_model.trim().is_empty() {
                "qwen3-embedding".to_string()
            } else {
                cfg.embed_model.trim().to_string()
            };
            format!("ollama:{}", model)
        }
        "bedrock" => {
            let model = if cfg.embed_model.trim().is_empty() {
                default_embed_model_for_backend("bedrock").to_string()
            } else {
                cfg.embed_model.trim().to_string()
            };
            format!("bedrock:{}:{}", bedrock_region_for_cfg(Some(cfg)), model)
        }
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

fn embed_query_cached(cfg: &ConfigValues, query: &str) -> Result<(String, Vec<f32>), String> {
    let q = query.trim();
    if q.is_empty() {
        return Err("query embedding requested for empty query".to_string());
    }
    let model_key_guess = model_key_for_cfg(cfg);
    let normalized_query = q.to_ascii_lowercase();
    let cache_key_guess = format!("{}::{}", model_key_guess, normalized_query);
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

    // Check persistent disk cache (survives process restarts)
    let db_p = db_path(&cfg.root);
    if let Ok(conn) = open_db_rw(&db_p) {
        if let Ok(cached_vec) = disk_cache_lookup(&conn, &normalized_query, &model_key_guess) {
            // Found in disk cache — populate in-memory cache and return
            let cache_key = format!("{}::{}", model_key_guess, normalized_query);
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
    let vector = embedder.embed_one(q)?;
    let cache_key = format!("{}::{}", model_key, normalized_query);
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

    // Write to persistent disk cache (best-effort, don't fail the query if this errors)
    if let Ok(conn) = open_db_rw(&db_p) {
        let _ = disk_cache_store(&conn, &normalized_query, &model_key, &vector);
    }

    Ok((model_key, vector))
}

/// Lookup a query embedding from the persistent SQLite cache.
fn disk_cache_lookup(
    conn: &Connection,
    query_normalized: &str,
    model_key: &str,
) -> Result<Vec<f32>, String> {
    let (blob, cached_at): (Vec<u8>, f64) = conn
        .query_row(
            "SELECT vector, cached_at FROM query_embed_cache WHERE query_normalized = ?1 AND model_key = ?2",
            params![query_normalized, model_key],
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

/// Store a query embedding in the persistent SQLite cache.
fn disk_cache_store(
    conn: &Connection,
    query_normalized: &str,
    model_key: &str,
    vector: &[f32],
) -> Result<(), String> {
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO query_embed_cache(query_normalized, model_key, vector, cached_at)
VALUES (?1, ?2, ?3, ?4)
ON CONFLICT(query_normalized, model_key) DO UPDATE SET
    vector = excluded.vector,
    cached_at = excluded.cached_at
"#,
        params![query_normalized, model_key, blob, now_ts()],
    )
    .map_err(|e| format!("failed writing disk cache: {}", e))?;
    Ok(())
}

fn rank_projects_native(
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
    let keyword_focused = is_keyword_focused_query(q);
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
    let project_evidence = project_evidence(&fused, cfg);
    let project_content = project_content_scores(&project_evidence);
    let frecency = frecency_scores(conn)?;
    let graph = graph_scores(conn)?;
    let path_keywords = path_keyword_scores(&existing_paths, q);

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
        let w = query_type.project_weights();
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
        out.push(RankedResult {
            path,
            score,
            lexical,
            semantic,
            frecency: fr,
            graph: gscore,
            evidence: evidence.into_iter().take(4).collect(),
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok(out)
}

fn rank_files_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
) -> Result<Vec<RankedFileResult>, String> {
    ensure_reembed_ready(conn, cfg, "search")?;
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let keyword_focused = is_keyword_focused_query(q);
    let query_type = QueryType::classify(q);
    let (sem_limit, lex_limit) = query_type.retrieval_limits(
        cfg.vector_candidates.max(1) as usize,
        cfg.lexical_candidates.max(1) as usize,
    );
    let (model_key, query_vector) = embed_query_cached(cfg, q)?;

    let project_semantic = project_semantic_scores(
        conn,
        &model_key,
        &query_vector,
        std::cmp::max(120, sem_limit * 2),
    )?;
    let mut fused = hybrid_search_lance(
        conn,
        &model_key,
        q,
        &query_vector,
        std::cmp::max(160, sem_limit * 4),
        std::cmp::max(120, lex_limit * 2),
    )?;
    if fused.is_empty() {
        return Ok(Vec::new());
    }
    // For Symbol and PathQuery types, boost path-based matching signals
    if query_type == QueryType::Symbol || query_type == QueryType::PathQuery {
        let path_signals = keyword_path_chunk_scores(conn, q, std::cmp::max(220, lex_limit * 3))?;
        if !path_signals.is_empty() {
            fused = fuse_chunk_signals(&path_signals, &fused);
        }
    }
    apply_graph_chunk_expansion(conn, &mut fused, cfg)?;

    let frecency = frecency_scores(conn)?;
    let project_paths = list_project_paths(conn)?;
    let project_path_keywords = path_keyword_scores(&project_paths, q);
    let mut by_file: HashMap<String, RankedFileResult> = HashMap::new();
    for row in fused.values() {
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
        score *= path_noise_penalty(&row.doc_rel_path);
        let candidate = RankedFileResult {
            path: row.doc_path.clone(),
            project_path: row.project_path.clone(),
            doc_rel_path: row.doc_rel_path.clone(),
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            graph: row.graph,
            relation: row.relation.clone(),
            quality: row.quality,
            excerpt: row.excerpt.clone(),
            evidence: Vec::new(),
        };
        let prev = by_file.get(&row.doc_path);
        if prev.is_none() || candidate.score > prev.map(|p| p.score).unwrap_or(0.0) {
            by_file.insert(row.doc_path.clone(), candidate);
        }
    }
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
            support.push(evidence_hit_from_chunk(row, chunk_base_score(row, cfg)));
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
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok(out)
}

/// Generate a hypothetical code snippet that would answer the query (HyDE technique).
/// Returns the generated text, or None on failure.
fn generate_hyde_snippet(cfg: &ConfigValues, query: &str) -> Option<String> {
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
fn rerank_with_ollama(
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
                    let timeout = timeout;
                    let model = model;
                    let query = query;
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

fn rank_chunks_native(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
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
    if use_tiered {
        let top_projects: HashSet<&String> = project_semantic.keys().collect();
        fused.retain(|_id, row| top_projects.contains(&row.project_path));
        if fused.is_empty() {
            return Ok(Vec::new());
        }
    }

    // HyDE: for natural language queries, generate a hypothetical code snippet,
    // embed it, run a second vector search, and merge new results into fused.
    if cfg.hyde_enabled && query_type == QueryType::NaturalLanguage {
        if let Some(hyde_text) = generate_hyde_snippet(cfg, q) {
            if let Ok((_hyde_model, hyde_vector)) = embed_query_cached(cfg, &hyde_text) {
                // Run vector-only search with the hypothetical embedding
                if let Ok(hyde_scores) = with_lance_store(|store| {
                    lance_store::search_vectors(store, &hyde_vector, sem_limit)
                }) {
                    // Merge HyDE results: add new chunks or boost existing ones
                    let hyde_signals = chunk_signals_for_ids(conn, &hyde_scores, &HashMap::new());
                    if let Ok(new_signals) = hyde_signals {
                        for (chunk_id, mut signal) in new_signals {
                            if !fused.contains_key(&chunk_id) {
                                // Scale down HyDE-only results to avoid dominating
                                signal.semantic *= 0.7;
                                fused.insert(chunk_id, signal);
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
    let mut out: Vec<RankedChunkResult> = Vec::new();
    for row in fused.values() {
        let content = chunk_base_score(row, cfg);
        let project_sem = *project_semantic.get(&row.project_path).unwrap_or(&0.0);
        let fr = *frecency.get(&row.project_path).unwrap_or(&0.0);
        let kw = doc_keyword_score(&row.doc_rel_path, q);
        // Chunk-level scoring: blend content quality, project semantic, frecency,
        // keyword match, and graph signals. Weights adjust based on query type.
        let score = match query_type {
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
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Cross-encoder re-ranking: take top pool_size candidates, score with LLM,
    // blend re-rank score with original score, then re-sort.
    if cfg.reranker_enabled && !out.is_empty() {
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
            out[..pool].sort_by(|a, b| b.score.total_cmp(&a.score));
        }
    }

    out.truncate(limit.max(1));
    Ok(out)
}

#[derive(Clone)]
struct SourceChunk {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    text: String,
}

#[derive(Clone)]
struct IndexedChunkRow {
    chunk_id: i64,
    chunk_index: i64,
    project_path: String,
    doc_path: String,
    doc_rel_path: String,
    doc_mtime: f64,
    token_count: i64,
    text: String,
}

#[derive(Clone)]
struct ChunkRelationFeedbackRow {
    src_chunk_id: i64,
    dst_chunk_id: i64,
    relation: String,
    decision: String,
    quality_label: String,
    note: String,
    source: String,
    created_at: f64,
    updated_at: f64,
    dst_chunk_index: i64,
    dst_doc_path: String,
    dst_doc_rel_path: String,
    dst_project_path: String,
}

fn apply_chunk_relation_decision(
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

fn normalize_relation_quality_label(raw: &str) -> Option<&'static str> {
    let v = raw.trim().to_lowercase();
    match v.as_str() {
        "" | "unspecified" | "clear" | "none" => Some("unspecified"),
        "good" => Some("good"),
        "weak" => Some("weak"),
        "wrong" => Some("wrong"),
        _ => None,
    }
}

fn relation_quality_multiplier(cfg: &ConfigValues, quality_label: &str) -> f64 {
    match quality_label {
        "good" => 1.0 + cfg.rank_relation_quality_good_boost.clamp(0.0, 1.0),
        "weak" => (1.0 - cfg.rank_relation_quality_weak_penalty.clamp(0.0, 1.0)).max(0.0),
        "wrong" => (1.0 - cfg.rank_relation_quality_wrong_penalty.clamp(0.0, 1.0)).max(0.0),
        _ => 1.0,
    }
}

fn set_chunk_relation_quality(
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

fn list_chunk_relation_feedback(
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
                limit.max(1).min(2000) as i64
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

fn suppressed_relation_set(
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

fn active_relation_quality_map(
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

fn relation_feedback_row_json(row: &ChunkRelationFeedbackRow) -> Value {
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

fn source_chunk_json(source: &SourceChunk) -> Value {
    serde_json::json!({
        "chunk_id": source.chunk_id,
        "chunk_index": source.chunk_index,
        "path": source.doc_path,
        "project_path": source.project_path,
        "doc_rel_path": source.doc_rel_path,
    })
}

#[cfg(test)]
mod chunk_contract_tests {
    use super::*;

    #[test]
    fn terminal_escape_sequences_are_stripped_from_prompt_input() {
        assert_eq!(strip_terminal_control_sequences("\u{1b}[A\u{1b}[A1"), "1");
        assert_eq!(strip_terminal_control_sequences("\u{1b}[B"), "");
        assert_eq!(strip_terminal_control_sequences("1\u{1b}[D"), "1");
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
        assert_eq!(chunk_search_schema(), "chunk-search-v1");
        assert_eq!(chunk_related_schema(), "chunk-related-v1");
        assert_eq!(chunk_get_schema(), "chunk-get-v1");
        assert_eq!(doc_read_schema(), "doc-read-v1");
        assert_eq!(context_pack_schema(), "context-pack-v1");
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
        };
        let json = ranked_chunk_result_json(&item);
        let obj = json.as_object().expect("expected object");
        assert_eq!(obj.len(), 12);
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
        ] {
            assert!(obj.contains_key(key), "missing key: {}", key);
        }
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
        };
        let json = related_chunk_result_json(&item);
        let obj = json.as_object().expect("expected object");
        assert_eq!(obj.len(), 14);
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

        let cleared = set_chunk_relation_quality(
            &conn,
            1,
            2,
            "same_project",
            "unspecified",
            "",
            "test",
            12.0,
        )
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
    fn bedrock_defaults_and_payload_contract() {
        assert_eq!(
            default_embed_model_for_backend("bedrock"),
            "amazon.titan-embed-text-v2:0"
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
    }

    #[test]
    fn bedrock_response_parsing_contract() {
        let direct = serde_json::json!({"embedding": [0.1, 0.2, 0.3]});
        let parsed = BedrockEmbedder::parse_vector(&direct).expect("parse direct embedding");
        assert_eq!(parsed.len(), 3);

        let batched = serde_json::json!({"embeddings": [[0.4, 0.5]]});
        let parsed_batch =
            BedrockEmbedder::parse_vector(&batched).expect("parse batched embedding");
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
}

fn source_chunk_by_id(conn: &Connection, chunk_id: i64) -> Result<Option<SourceChunk>, String> {
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

fn indexed_chunk_by_id(
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

fn indexed_doc_chunks_by_path(
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

fn truncate_text_chars(text: &str, max_chars: usize) -> (String, bool, usize) {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return (text.to_string(), false, total_chars);
    }
    let clipped: String = text.chars().take(max_chars).collect();
    (clipped, true, total_chars)
}

fn project_neighbor_weights(
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
fn multi_hop_project_neighbors(
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

fn related_chunks_native(
    conn: &Connection,
    cfg: &ConfigValues,
    chunk_id: i64,
    limit: usize,
) -> Result<(SourceChunk, Vec<RelatedChunkResult>), String> {
    let source = source_chunk_by_id(conn, chunk_id)?
        .ok_or_else(|| format!("chunk {} was not found", chunk_id))?;
    let query_text: String = source.text.chars().take(2400).collect();
    let candidate_limit = std::cmp::max(40, limit.max(1) * 8).min(400);
    let ranked = rank_chunks_native(conn, cfg, &query_text, candidate_limit)?;
    let neighbor_weights = project_neighbor_weights(conn, &source.project_path, 120)?;
    let suppressed = suppressed_relation_set(conn, source.chunk_id)?;
    let quality_feedback = active_relation_quality_map(conn, source.chunk_id)?;

    let mut out: Vec<RelatedChunkResult> = Vec::new();
    for row in ranked {
        if row.chunk_id == source.chunk_id {
            continue;
        }
        let (relation, relation_weight) = if row.path == source.doc_path {
            ("same_file".to_string(), 1.0)
        } else if row.project_path == source.project_path {
            ("same_project".to_string(), 0.82)
        } else if let Some(weight) = neighbor_weights.get(&row.project_path) {
            (
                "project_edge".to_string(),
                (0.55 + (0.45 * *weight)).clamp(0.0, 1.0),
            )
        } else {
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
        out.push(RelatedChunkResult {
            chunk_id: row.chunk_id,
            chunk_index: row.chunk_index,
            path: row.path,
            project_path: row.project_path,
            doc_rel_path: row.doc_rel_path,
            relation,
            relation_weight,
            relation_quality,
            relation_quality_multiplier: relation_quality_weight,
            score,
            semantic: row.semantic,
            lexical: row.lexical,
            quality: row.quality,
            excerpt: row.excerpt,
        });
    }
    if out.is_empty() {
        return Ok((source, Vec::new()));
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(limit.max(1));
    Ok((source, out))
}

fn context_pack_schema() -> &'static str {
    "context-pack-v1"
}

#[derive(Clone, Copy)]
struct ContextPackOptions {
    budget_chars: usize,
    seed_limit: usize,
    related_per_seed: usize,
    include_docs: bool,
    doc_max_chars: usize,
}

fn build_context_pack_native(
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

    let seeds = rank_chunks_native(conn, cfg, q, opts.seed_limit.max(1).min(80))?;
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

    for seed in seeds.iter().take(opts.seed_limit.max(1) * 2) {
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

        let mut related_rows: Vec<Value> = Vec::new();
        if opts.related_per_seed > 0 {
            let (_, related) =
                related_chunks_native(conn, cfg, chunk.chunk_id, opts.related_per_seed)?;
            for rel in related.into_iter().take(opts.related_per_seed) {
                related_rows.push(related_chunk_result_json(&rel));
            }
        }

        staged.push(StagedChunk {
            seed: seed.clone(),
            chunk,
            related: related_rows,
        });

        if staged.len() >= opts.seed_limit.max(1) {
            break;
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

fn list_project_paths(conn: &Connection) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing project list query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed querying project paths: {}", e))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| format!("failed reading project path row: {}", e))?);
    }
    Ok(out)
}

fn project_semantic_scores(
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

fn search_lexical_chunks_sqlite(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let fts = fts_query(query);
    if fts.is_empty() {
        return Ok(HashMap::new());
    }
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
    bm25(chunk_fts) AS lexical_bm25
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
            Ok((
                chunk_id,
                project_path,
                doc_path,
                doc_rel_path,
                chunk_index,
                text,
                bm25,
            ))
        })
        .map_err(|e| format!("failed querying lexical chunks: {}", e))?;

    let mut raw: Vec<(i64, String, String, String, i64, String, f64)> = Vec::new();
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
    for (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, bm25) in raw {
        let lexical = if span.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ((bm25 - lo) / span)
        };
        out.insert(
            chunk_id,
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path,
                doc_path,
                doc_rel_path: doc_rel_path.clone(),
                semantic: 0.0,
                lexical: lexical.clamp(0.0, 1.0),
                graph: 0.0,
                relation: "direct".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            },
        );
    }
    Ok(out)
}

fn keyword_path_chunk_scores(
    conn: &Connection,
    query: &str,
    keep_top: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return Ok(HashMap::new());
    }
    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    let per_token_limit = (keep_top.max(1) * 3).clamp(50, 2500);
    let mut stmt = conn
        .prepare(
            r#"
SELECT
    pc.id AS chunk_id,
    p.path AS project_path,
    pc.doc_path,
    pc.doc_rel_path,
    pc.chunk_index,
    pc.text
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
WHERE lower(pc.doc_path) LIKE ?1
   OR lower(pc.doc_rel_path) LIKE ?2
   OR lower(p.path) LIKE ?3
LIMIT ?4
"#,
        )
        .map_err(|e| format!("failed preparing keyword path chunk query: {}", e))?;

    let q_n = q_tokens.len() as f64;
    for token in &q_tokens {
        let pattern = format!("%{}%", token.to_lowercase());
        let rows = stmt
            .query_map(
                params![pattern, pattern, pattern, per_token_limit as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .map_err(|e| format!("failed querying keyword path chunks: {}", e))?;
        for row in rows {
            let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) =
                row.map_err(|e| format!("failed reading keyword path chunk row: {}", e))?;
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
                continue;
            }
            let entry = out.entry(chunk_id).or_insert_with(|| ChunkSignal {
                chunk_id,
                chunk_index,
                project_path: project_path.clone(),
                doc_path: doc_path.clone(),
                doc_rel_path: doc_rel_path.clone(),
                semantic: 0.0,
                lexical,
                graph: 0.0,
                relation: "path_keyword".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            });
            if lexical > entry.lexical {
                entry.lexical = lexical;
                entry.relation = "path_keyword".to_string();
            }
        }
    }

    if out.len() <= keep_top.max(1) {
        return Ok(out);
    }
    let mut pairs: Vec<(i64, ChunkSignal)> = out.into_iter().collect();
    pairs.sort_by(|a, b| b.1.lexical.total_cmp(&a.1.lexical));
    pairs.truncate(keep_top.max(1));
    Ok(pairs.into_iter().collect())
}

fn semantic_chunk_scores(
    conn: &Connection,
    model: &str,
    query_vector: &[f32],
    keep_top: usize,
) -> Result<HashMap<i64, ChunkSignal>, String> {
    let qnorm = vector_norm(query_vector);
    if qnorm == 0.0 {
        return Ok(HashMap::new());
    }
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
    pcv.norm,
    pcv.vector
FROM project_chunks pc
JOIN projects p ON p.id = pc.project_id
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pcv.model = ?1
"#,
        )
        .map_err(|e| format!("failed preparing semantic chunk query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let chunk_id: i64 = row.get(0)?;
            let project_path: String = row.get(1)?;
            let doc_path: String = row.get(2)?;
            let doc_rel_path: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let text: String = row.get(5)?;
            let norm: f64 = row.get(6)?;
            let blob: Vec<u8> = row.get(7)?;
            Ok((
                chunk_id,
                project_path,
                doc_path,
                doc_rel_path,
                chunk_index,
                text,
                norm,
                blob,
            ))
        })
        .map_err(|e| format!("failed querying semantic chunks: {}", e))?;

    let mut scored: Vec<(f64, i64, String, String, String, i64, String)> = Vec::new();
    for row in rows {
        let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text, vnorm, blob) =
            row.map_err(|e| format!("failed reading semantic chunk row: {}", e))?;
        if vnorm == 0.0 {
            continue;
        }
        let vec = blob_to_f32_vec(&blob);
        let sim = ((cosine_raw(query_vector, &vec, qnorm, vnorm) + 1.0) / 2.0).clamp(0.0, 1.0);
        scored.push((
            sim,
            chunk_id,
            project_path,
            doc_path,
            doc_rel_path,
            chunk_index,
            text,
        ));
    }
    if scored.is_empty() {
        return Ok(HashMap::new());
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(keep_top.max(1));
    let lo = scored.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let hi = scored.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let mut out = HashMap::new();
    for (score, chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) in scored {
        let semantic = if span.abs() < f64::EPSILON {
            1.0
        } else {
            (score - lo) / span
        };
        out.insert(
            chunk_id,
            ChunkSignal {
                chunk_id,
                chunk_index,
                project_path,
                doc_path,
                doc_rel_path: doc_rel_path.clone(),
                semantic: semantic.clamp(0.0, 1.0),
                lexical: 0.0,
                graph: 0.0,
                relation: "direct".to_string(),
                quality: content_quality(&doc_rel_path, &text),
                excerpt: clip_text(&text, 190),
            },
        );
    }
    Ok(out)
}

fn fuse_chunk_signals(
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

fn chunk_base_score(row: &ChunkSignal, cfg: &ConfigValues) -> f64 {
    // Graph is treated as contextual support with hop-penalty baked into row.graph.
    let direct = (cfg.rank_chunk_semantic_weight * row.semantic)
        + (cfg.rank_chunk_lexical_weight * row.lexical)
        + (cfg.rank_chunk_graph_weight * row.graph);
    let quality_mix = cfg.rank_quality_mix.clamp(0.0, 1.0);
    direct * ((1.0 - quality_mix) + (quality_mix * row.quality))
}

fn apply_graph_chunk_expansion(
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
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1));
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

fn evidence_hit_from_chunk(row: &ChunkSignal, score: f64) -> EvidenceHit {
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
    }
}

fn chunk_signals_for_ids(
    conn: &Connection,
    semantic_scores: &HashMap<i64, f64>,
    lexical_scores: &HashMap<i64, f64>,
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

    let mut out: HashMap<i64, ChunkSignal> = HashMap::new();
    for batch in all_ids.chunks(300) {
        let placeholders = std::iter::repeat("?")
            .take(batch.len())
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
    pc.text
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
                Ok((
                    chunk_id,
                    project_path,
                    doc_path,
                    doc_rel_path,
                    chunk_index,
                    text,
                ))
            })
            .map_err(|e| format!("failed querying chunk metadata lookup: {}", e))?;
        for row in rows {
            let (chunk_id, project_path, doc_path, doc_rel_path, chunk_index, text) =
                row.map_err(|e| format!("failed reading chunk metadata lookup row: {}", e))?;
            out.insert(
                chunk_id,
                ChunkSignal {
                    chunk_id,
                    chunk_index,
                    project_path,
                    doc_path,
                    doc_rel_path: doc_rel_path.clone(),
                    semantic: *semantic_scores.get(&chunk_id).unwrap_or(&0.0),
                    lexical: *lexical_scores.get(&chunk_id).unwrap_or(&0.0),
                    graph: 0.0,
                    relation: "direct".to_string(),
                    quality: content_quality(&doc_rel_path, &text),
                    excerpt: clip_text(&text, 190),
                },
            );
        }
    }
    Ok(out)
}

fn project_evidence(
    fused_chunks: &HashMap<i64, ChunkSignal>,
    cfg: &ConfigValues,
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
            hits.push(evidence_hit_from_chunk(row, chunk_base_score(row, cfg)));
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

fn project_content_scores(
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

fn fts_query(query: &str) -> String {
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

fn frecency_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
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

fn rank_by_frecency_only(conn: &Connection, limit: usize) -> Result<Vec<RankedResult>, String> {
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

fn graph_scores(conn: &Connection) -> Result<HashMap<String, f64>, String> {
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

fn clip_text(text: &str, max_chars: usize) -> String {
    let value = collapse_whitespace(text);
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut clipped: String = value.chars().take(max_chars).collect();
    clipped = clipped.trim_end().to_string();
    clipped.push_str("...");
    clipped
}

fn content_quality(doc_rel_path: &str, text: &str) -> f64 {
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
    if name.contains("session_") || format!("/{}/", name).contains("/sessions/") {
        score *= 0.35;
    }
    if name.ends_with(".min.js") {
        score *= 0.55;
    }
    score.clamp(0.08, 1.0)
}

fn is_hex_token(token: &str) -> bool {
    if token.len() < 16 {
        return false;
    }
    token.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn path_keyword_scores(paths: &[String], query: &str) -> HashMap<String, f64> {
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

fn doc_keyword_score(doc_rel_path: &str, query: &str) -> f64 {
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

fn is_keyword_focused_query(query: &str) -> bool {
    let q_tokens: Vec<String> = all_word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    if q_tokens.is_empty() {
        return false;
    }
    q_tokens.len() <= 2 && q_tokens.iter().all(|t| t.len() <= 32)
}

// ── Query type classification ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryType {
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
struct QueryWeights {
    semantic: f64,
    lexical: f64,
    path_kw: f64,
    graph: f64,
    frecency: f64,
}

impl QueryType {
    /// Classify a query string into a QueryType using heuristics.
    fn classify(query: &str) -> QueryType {
        let q = query.trim();

        // PathQuery: contains path separators or file extensions
        if q.contains('/') || q.contains('\\') {
            return QueryType::PathQuery;
        }
        if q.starts_with("*.")
            || q.contains(".rs")
            || q.contains(".py")
            || q.contains(".ts")
            || q.contains(".js")
            || q.contains(".go")
            || q.contains(".java")
            || q.contains(".cpp")
            || q.contains(".c")
        {
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
        if code_keywords
            .iter()
            .any(|kw| lower.starts_with(kw) || lower.contains(&format!(" {}", kw.trim())))
        {
            return QueryType::CodePattern;
        }

        // Symbol: looks like an identifier (camelCase, snake_case, PascalCase, UPPER_CASE)
        // Heuristic: no spaces, or 1-2 tokens that look like identifiers
        let tokens: Vec<&str> = q.split_whitespace().collect();
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

    /// Get the ranking weights for project-level scoring.
    fn project_weights(&self) -> QueryWeights {
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
                semantic: 0.58,
                lexical: 0.14,
                path_kw: 0.10,
                graph: 0.10,
                frecency: 0.08,
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

fn path_noise_penalty(doc_rel_path: &str) -> f64 {
    let p = doc_rel_path.to_lowercase();
    let mut penalty = 1.0f64;
    if p.starts_with("tmp/") || p.contains("/tmp/") {
        penalty *= 0.55;
    }
    if p.starts_with("state/") || p.contains("/state/") {
        penalty *= 0.72;
    }
    if p.starts_with("archived/") || p.contains("/archived/") {
        penalty *= 0.82;
    }
    if p.ends_with(".json") {
        penalty *= 0.92;
    }
    penalty.clamp(0.25, 1.0)
}

fn is_generic_container(path: &str) -> bool {
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

fn cosine_raw(a: &[f32], b: &[f32], anorm: f64, bnorm: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for i in 0..n {
        dot += (a[i] as f64) * (b[i] as f64);
    }
    let denom = anorm * bnorm;
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn all_word_tokens(text: &str) -> Vec<String> {
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

fn run_legacy_cmd(args: &[OsString]) -> ! {
    let wants_help = args.len() == 1 && (args[0] == "-h" || args[0] == "--help");
    if args.is_empty() || wants_help {
        eprintln!("usage: retrivio legacy <args...>");
        eprintln!("example: retrivio legacy install --bench");
        process::exit(2);
    }
    process::exit(run_legacy_bridge_status(args));
}

fn run_legacy_bridge_status(args: &[OsString]) -> i32 {
    let repo = find_repo_root()
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let python = env::var("RETRIVIO_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let mut cmd = Command::new(&python);
    cmd.arg("-m").arg("retrivio.cli");
    cmd.args(args);
    cmd.stdin(process::Stdio::inherit());
    cmd.stdout(process::Stdio::inherit());
    cmd.stderr(process::Stdio::inherit());

    let src_dir = repo.join("src");
    let mut py_path = src_dir.to_string_lossy().to_string();
    if let Ok(existing) = env::var("PYTHONPATH") {
        if !existing.trim().is_empty() {
            py_path.push(':');
            py_path.push_str(&existing);
        }
    }
    cmd.env("PYTHONPATH", py_path);
    // Keep legacy bridge on the same config resolution contract as native:
    // ~/.retrivio by default, runtime CLI override only.
    cmd.env(
        "RETRIVIO_CONFIG",
        config_path(&cwd).to_string_lossy().to_string(),
    );

    let status = cmd.status().unwrap_or_else(|err| {
        eprintln!(
            "error: failed to run legacy bridge via '{}': {}",
            python, err
        );
        process::exit(1);
    });
    status.code().unwrap_or(1)
}

fn vector_dim_from_sqlite(conn: &Connection, model_key: &str) -> Option<usize> {
    conn.query_row(
        "SELECT dim FROM project_chunk_vectors WHERE model = ?1 LIMIT 1",
        params![model_key],
        |row| row.get::<_, i64>(0),
    )
    .ok()
    .map(|d| d.max(1) as usize)
}

#[derive(Clone, Debug)]
struct ConfigValues {
    root: PathBuf,
    embed_backend: String,
    embed_model: String,
    aws_profile: String,
    aws_region: String,
    aws_refresh_cmd: String,
    bedrock_concurrency: i64,
    bedrock_max_retries: i64,
    bedrock_retry_base_ms: i64,
    retrieval_backend: String,
    local_embed_dim: i64,
    max_chars_per_project: i64,
    lexical_candidates: i64,
    vector_candidates: i64,
    rank_chunk_semantic_weight: f64,
    rank_chunk_lexical_weight: f64,
    rank_chunk_graph_weight: f64,
    rank_quality_mix: f64,
    rank_relation_quality_good_boost: f64,
    rank_relation_quality_weak_penalty: f64,
    rank_relation_quality_wrong_penalty: f64,
    rank_project_content_weight: f64,
    rank_project_semantic_weight: f64,
    rank_project_path_weight: f64,
    rank_project_graph_weight: f64,
    rank_project_frecency_weight: f64,
    graph_seed_limit: i64,
    graph_neighbor_limit: i64,
    graph_same_project_high: f64,
    graph_same_project_low: f64,
    graph_related_base: f64,
    graph_related_scale: f64,
    graph_related_cap: f64,
    // HyDE: Hypothetical Document Embedding (opt-in, adds ~800ms)
    hyde_enabled: bool,
    // Cross-encoder re-ranking via Ollama
    reranker_enabled: bool,
    reranker_model: String,
    reranker_pool_size: usize,
    reranker_batch_size: usize,
    reranker_timeout_ms: u64,
}

fn default_config_root() -> PathBuf {
    env::current_dir()
        .ok()
        .map(|p| normalize_path(&p.to_string_lossy()))
        .unwrap_or_else(|| expand_tilde("~"))
}

impl ConfigValues {
    fn from_map(map: std::collections::HashMap<String, String>) -> Self {
        let mut embed_backend = map
            .get("embed_backend")
            .cloned()
            .unwrap_or_else(|| "ollama".to_string())
            .trim()
            .to_lowercase();
        if !matches!(embed_backend.as_str(), "ollama" | "bedrock") {
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
        let aws_refresh_cmd = map
            .get("aws_refresh_cmd")
            .cloned()
            .unwrap_or_default()
            .trim()
            .to_string();
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
        let lexical_candidates = map
            .get("lexical_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120);
        let vector_candidates = map
            .get("vector_candidates")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(120);
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
            bedrock_concurrency,
            bedrock_max_retries,
            bedrock_retry_base_ms,
            retrieval_backend,
            local_embed_dim,
            max_chars_per_project,
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
        }
    }
}

fn arg_value(args: &[OsString], index: usize, flag: &str) -> String {
    let Some(v) = args.get(index) else {
        eprintln!("error: {} expects a value", flag);
        process::exit(2);
    };
    v.to_string_lossy().to_string()
}

fn normalize_path(raw: &str) -> PathBuf {
    let mut path = expand_tilde(raw);
    if !path.is_absolute() {
        if let Ok(cwd) = env::current_dir() {
            path = cwd.join(path);
        }
    }
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    normalize_lexical(&path)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn write_config_file(path: &Path, cfg: &ConfigValues) -> Result<(), String> {
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
        format!("bedrock_concurrency = {}", cfg.bedrock_concurrency),
        format!("bedrock_max_retries = {}", cfg.bedrock_max_retries),
        format!("bedrock_retry_base_ms = {}", cfg.bedrock_retry_base_ms),
        format!(
            "retrieval_backend = \"{}\"",
            toml_escape(&cfg.retrieval_backend)
        ),
        format!("local_embed_dim = {}", cfg.local_embed_dim),
        format!("max_chars_per_project = {}", cfg.max_chars_per_project),
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
        String::new(),
    ];
    fs::write(path, lines.join("\n")).map_err(|e| format!("failed writing config: {}", e))
}

fn toml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn ensure_db_schema(db_path: &Path) -> Result<(), String> {
    let _ = open_db_rw(db_path)?;
    Ok(())
}

fn ensure_tracked_root(db_path: &Path, path: &Path, added_at: f64) -> Result<(), String> {
    let conn = open_db_rw(db_path)?;
    ensure_tracked_root_conn(&conn, path, added_at)
}

fn remove_tracked_root(db_path: &Path, path: &Path) -> Result<i64, String> {
    let conn = open_db_rw(db_path)?;
    conn.execute(
        "DELETE FROM tracked_roots WHERE path = ?1",
        params![path.to_string_lossy().to_string()],
    )
    .map(|n| n as i64)
    .map_err(|e| format!("failed to remove tracked root: {}", e))
}

fn list_tracked_roots(db_path: &Path) -> Result<Vec<PathBuf>, String> {
    let conn = open_db_rw(db_path)?;
    list_tracked_roots_conn(&conn)
}

fn get_exclude_patterns_conn(conn: &Connection, root_path: &Path) -> Result<Vec<String>, String> {
    let raw: String = conn
        .query_row(
            "SELECT exclude_patterns FROM tracked_roots WHERE path = ?1 AND enabled = 1",
            params![root_path.to_string_lossy().to_string()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed reading exclude patterns: {}", e))?;
    Ok(raw
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn add_exclude_patterns_conn(
    conn: &Connection,
    root_path: &Path,
    patterns: &[String],
) -> Result<(), String> {
    let existing = get_exclude_patterns_conn(conn, root_path).unwrap_or_default();
    let mut merged: Vec<String> = existing;
    for p in patterns {
        let trimmed = p.trim().to_string();
        if !trimmed.is_empty() && !merged.contains(&trimmed) {
            merged.push(trimmed);
        }
    }
    merged.sort();
    let joined = merged.join("\n");
    conn.execute(
        "UPDATE tracked_roots SET exclude_patterns = ?1 WHERE path = ?2",
        params![joined, root_path.to_string_lossy().to_string()],
    )
    .map_err(|e| format!("failed updating exclude patterns: {}", e))?;
    Ok(())
}

fn remove_exclude_patterns_conn(
    conn: &Connection,
    root_path: &Path,
    patterns: &[String],
) -> Result<usize, String> {
    let existing = get_exclude_patterns_conn(conn, root_path).unwrap_or_default();
    let remove_set: HashSet<&str> = patterns.iter().map(|s| s.trim()).collect();
    let filtered: Vec<String> = existing
        .into_iter()
        .filter(|p| !remove_set.contains(p.as_str()))
        .collect();
    let removed = patterns.len().saturating_sub(filtered.len());
    let joined = filtered.join("\n");
    conn.execute(
        "UPDATE tracked_roots SET exclude_patterns = ?1 WHERE path = ?2",
        params![joined, root_path.to_string_lossy().to_string()],
    )
    .map_err(|e| format!("failed updating exclude patterns: {}", e))?;
    Ok(removed)
}

fn app_state_get(conn: &Connection, key: &str) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT value FROM app_state WHERE key = ?1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|e| format!("failed reading app state '{}': {}", key, e))
}

fn app_state_set(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO app_state(key, value, updated_at)
VALUES (?1, ?2, ?3)
ON CONFLICT(key) DO UPDATE SET
    value = excluded.value,
    updated_at = excluded.updated_at
"#,
        params![key, value, now_ts()],
    )
    .map_err(|e| format!("failed writing app state '{}': {}", key, e))?;
    Ok(())
}

fn app_state_bool(conn: &Connection, key: &str) -> Result<bool, String> {
    let raw = app_state_get(conn, key)?.unwrap_or_default();
    let v = raw.trim().to_ascii_lowercase();
    Ok(matches!(v.as_str(), "1" | "true" | "yes" | "on"))
}

fn has_any_indexed_vectors(conn: &Connection) -> Result<bool, String> {
    let chunk_any: Option<i64> = conn
        .query_row("SELECT 1 FROM project_chunk_vectors LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|e| format!("failed checking chunk vectors: {}", e))?;
    if chunk_any.is_some() {
        return Ok(true);
    }
    let project_any: Option<i64> = conn
        .query_row("SELECT 1 FROM project_vectors LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()
        .map_err(|e| format!("failed checking project vectors: {}", e))?;
    Ok(project_any.is_some())
}

fn model_change_reembed_reason(old_model_key: &str, new_model_key: &str) -> String {
    format!(
        "embedding model changed from '{}' to '{}'; run `retrivio reembed` before searching.",
        old_model_key, new_model_key
    )
}

fn dominant_chunk_vector_model(conn: &Connection) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT model FROM project_chunk_vectors GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1",
        [],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|e| format!("failed reading dominant chunk vector model: {}", e))
}

fn chunk_vector_compatibility_reason(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    let any_chunk_vectors: i64 = conn
        .query_row("SELECT COUNT(*) FROM project_chunk_vectors", [], |row| {
            row.get(0)
        })
        .map_err(|e| format!("failed counting chunk vectors: {}", e))?;
    if any_chunk_vectors <= 0 {
        return Ok(None);
    }

    let current_model_key = model_key_for_cfg(cfg);
    let total_chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM project_chunks", [], |row| row.get(0))
        .map_err(|e| format!("failed counting project chunks: {}", e))?;
    let current_model_vectors: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM project_chunk_vectors WHERE model = ?1",
            params![current_model_key.clone()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed counting vectors for current model: {}", e))?;

    if current_model_vectors == 0 {
        let existing_model =
            dominant_chunk_vector_model(conn)?.unwrap_or_else(|| "<unknown>".to_string());
        return Ok(Some(format!(
            "existing chunk embeddings are for '{}' but current config expects '{}'; run `retrivio reembed` before searching.",
            existing_model, current_model_key
        )));
    }

    if total_chunks > 0 && current_model_vectors < total_chunks {
        return Ok(Some(format!(
            "chunk embeddings for '{}' are incomplete ({} of {} chunks); run `retrivio reembed` before searching.",
            current_model_key, current_model_vectors, total_chunks
        )));
    }

    let dim_variants: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT dim) FROM project_chunk_vectors WHERE model = ?1",
            params![current_model_key.clone()],
            |row| row.get(0),
        )
        .map_err(|e| format!("failed checking vector dimensions for current model: {}", e))?;
    if dim_variants > 1 {
        return Ok(Some(format!(
            "chunk embeddings for '{}' have inconsistent dimensions ({} variants); run `retrivio reembed` before searching.",
            current_model_key, dim_variants
        )));
    }

    Ok(None)
}

fn sync_reembed_requirement_state(
    cwd: &Path,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    let conn = open_db_rw(&db_path(cwd))?;
    let current_model_key = model_key_for_cfg(cfg);

    if let Some(reason) = chunk_vector_compatibility_reason(&conn, cfg)? {
        if let Some(existing_model) = dominant_chunk_vector_model(&conn)? {
            if !existing_model.trim().is_empty() && existing_model != current_model_key {
                app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &existing_model)?;
            }
        }
        app_state_set(&conn, APP_STATE_REEMBED_REQUIRED, "1")?;
        app_state_set(&conn, APP_STATE_REEMBED_REASON, &reason)?;
        return Ok(Some(reason));
    }

    if has_any_indexed_vectors(&conn)? {
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &current_model_key)?;
    }
    app_state_set(&conn, APP_STATE_REEMBED_REQUIRED, "0")?;
    app_state_set(&conn, APP_STATE_REEMBED_REASON, "")?;
    Ok(None)
}

fn refresh_reembed_requirement_for_config_change(
    cwd: &Path,
    before: &ConfigValues,
    after: &ConfigValues,
) -> Result<Option<String>, String> {
    // Preserve existing behavior that marks migration required on explicit model changes.
    let _ = mark_reembed_required_if_model_changed(cwd, before, after)?;
    // Then run a quick compatibility check so prompts/warnings reflect current DB reality.
    sync_reembed_requirement_state(cwd, after)
}

fn mark_reembed_required_if_model_changed(
    cwd: &Path,
    before: &ConfigValues,
    after: &ConfigValues,
) -> Result<Option<String>, String> {
    let old_model_key = model_key_for_cfg(before);
    let new_model_key = model_key_for_cfg(after);
    if old_model_key == new_model_key {
        return Ok(None);
    }
    let conn = open_db_rw(&db_path(cwd))?;
    let has_vectors = has_any_indexed_vectors(&conn)?;
    let reason = model_change_reembed_reason(&old_model_key, &new_model_key);
    if has_vectors {
        // Keep old model key so reembed knows what to migrate from.
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &old_model_key)?;
    } else {
        // No vectors yet — just record the new model for future embeds.
        app_state_set(&conn, APP_STATE_ACTIVE_MODEL_KEY, &new_model_key)?;
    }
    app_state_set(
        &conn,
        APP_STATE_REEMBED_REQUIRED,
        if has_vectors { "1" } else { "0" },
    )?;
    app_state_set(
        &conn,
        APP_STATE_REEMBED_REASON,
        if has_vectors { &reason } else { "" },
    )?;
    Ok(Some(reason))
}

fn reembed_requirement_reason(
    conn: &Connection,
    cfg: &ConfigValues,
) -> Result<Option<String>, String> {
    let current_model_key = model_key_for_cfg(cfg);
    let mut required = app_state_bool(conn, APP_STATE_REEMBED_REQUIRED)?;
    let last_model_key = app_state_get(conn, APP_STATE_ACTIVE_MODEL_KEY)?.unwrap_or_default();
    let mut reason = app_state_get(conn, APP_STATE_REEMBED_REASON)?
        .unwrap_or_default()
        .trim()
        .to_string();

    if !required && !last_model_key.trim().is_empty() && last_model_key.trim() != current_model_key
    {
        required = true;
        reason = model_change_reembed_reason(last_model_key.trim(), &current_model_key);
        app_state_set(conn, APP_STATE_REEMBED_REQUIRED, "1")?;
        app_state_set(conn, APP_STATE_REEMBED_REASON, &reason)?;
    }
    if !required {
        return Ok(None);
    }
    if reason.is_empty() {
        reason = if !last_model_key.trim().is_empty() && last_model_key.trim() != current_model_key
        {
            model_change_reembed_reason(last_model_key.trim(), &current_model_key)
        } else {
            format!(
                "embedding model migration required for '{}'; run `retrivio reembed` before searching.",
                current_model_key
            )
        };
        app_state_set(conn, APP_STATE_REEMBED_REASON, &reason)?;
    }
    Ok(Some(reason))
}

fn ensure_reembed_ready(
    conn: &Connection,
    cfg: &ConfigValues,
    context: &str,
) -> Result<(), String> {
    if let Some(reason) = reembed_requirement_reason(conn, cfg)? {
        return Err(format!("{} blocked: {}", context, reason));
    }
    Ok(())
}

fn mark_reembed_completed(conn: &Connection, model_key: &str) -> Result<(), String> {
    app_state_set(conn, APP_STATE_ACTIVE_MODEL_KEY, model_key)?;
    app_state_set(conn, APP_STATE_REEMBED_REQUIRED, "0")?;
    app_state_set(conn, APP_STATE_REEMBED_REASON, "")?;
    Ok(())
}

fn record_selection_event(
    conn: &Connection,
    query: &str,
    path: &str,
    selected_at: f64,
) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO selection_events(query, path, selected_at)
VALUES (?1, ?2, ?3)
"#,
        params![query, path, selected_at],
    )
    .map_err(|e| format!("failed recording selection event: {}", e))?;
    Ok(())
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn open_db_rw(db_path: &Path) -> Result<Connection, String> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("failed creating db dir: {}", e))?;
    }
    let conn = Connection::open(db_path).map_err(|e| format!("failed opening database: {}", e))?;
    conn.execute_batch(
        r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
"#,
    )
    .map_err(|e| format!("failed setting db pragmas: {}", e))?;
    init_schema(&conn)?;
    Ok(conn)
}

fn open_db_read_only(db_path: &Path) -> Result<Connection, String> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("failed opening database readonly: {}", e))?;
    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    project_mtime REAL NOT NULL,
    last_indexed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS project_vectors (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS project_chunks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_path TEXT NOT NULL,
    doc_rel_path TEXT NOT NULL,
    doc_mtime REAL NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    updated_at REAL NOT NULL,
    chunk_kind TEXT NOT NULL DEFAULT 'text_window',
    symbol_name TEXT NOT NULL DEFAULT '',
    parent_context TEXT NOT NULL DEFAULT '',
    line_start INTEGER NOT NULL DEFAULT 0,
    line_end INTEGER NOT NULL DEFAULT 0,
    context_header TEXT NOT NULL DEFAULT '',
    UNIQUE(project_id, doc_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_project_chunks_project
    ON project_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_project_chunks_doc
    ON project_chunks(doc_path);

CREATE TABLE IF NOT EXISTS project_chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES project_chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    norm REAL NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS selection_events (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    path TEXT NOT NULL,
    selected_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at REAL NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    exclude_patterns TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracked_roots_enabled_path
    ON tracked_roots(enabled, path);

CREATE TABLE IF NOT EXISTS project_edges (
    src_project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(src_project_id, dst, kind)
);

CREATE INDEX IF NOT EXISTS idx_project_edges_dst
    ON project_edges(dst);
CREATE INDEX IF NOT EXISTS idx_project_edges_src_kind
    ON project_edges(src_project_id, kind);

CREATE TABLE IF NOT EXISTS chunk_relation_feedback (
    id INTEGER PRIMARY KEY,
    src_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    dst_chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('active', 'suppressed')),
    quality_label TEXT NOT NULL DEFAULT 'unspecified' CHECK(quality_label IN ('unspecified', 'good', 'weak', 'wrong')),
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(src_chunk_id, dst_chunk_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_decision_time
    ON chunk_relation_feedback(src_chunk_id, decision, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_dst
    ON chunk_relation_feedback(dst_chunk_id);

CREATE INDEX IF NOT EXISTS idx_selection_events_path_time
    ON selection_events(path, selected_at DESC);
CREATE INDEX IF NOT EXISTS idx_selection_events_query_time
    ON selection_events(query, selected_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS project_fts USING fts5(
    path,
    title,
    summary,
    content='projects',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS projects_ai AFTER INSERT ON projects BEGIN
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_ad AFTER DELETE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS projects_au AFTER UPDATE ON projects BEGIN
    INSERT INTO project_fts(project_fts, rowid, path, title, summary)
    VALUES ('delete', old.id, old.path, old.title, old.summary);
    INSERT INTO project_fts(rowid, path, title, summary)
    VALUES (new.id, new.path, new.title, new.summary);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    doc_path,
    doc_rel_path,
    text,
    content='project_chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS project_chunks_ai AFTER INSERT ON project_chunks BEGIN
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_ad AFTER DELETE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
END;

CREATE TRIGGER IF NOT EXISTS project_chunks_au AFTER UPDATE ON project_chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, doc_path, doc_rel_path, text)
    VALUES ('delete', old.id, old.doc_path, old.doc_rel_path, old.text);
    INSERT INTO chunk_fts(rowid, doc_path, doc_rel_path, text)
    VALUES (new.id, new.doc_path, new.doc_rel_path, new.text);
END;
"#,
    )
    .map_err(|e| {
        format!(
            "failed initializing db schema (ensure sqlite build includes FTS5): {}",
            e
        )
    })?;
    // Persistent query embedding cache (survives process restarts)
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS query_embed_cache (
    query_normalized TEXT NOT NULL,
    model_key TEXT NOT NULL,
    vector BLOB NOT NULL,
    cached_at REAL NOT NULL,
    PRIMARY KEY(query_normalized, model_key)
);
"#,
    )
    .map_err(|e| format!("failed creating query_embed_cache table: {}", e))?;

    // File-level change detection for incremental indexing
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS project_files (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_mtime REAL NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    last_indexed REAL NOT NULL,
    UNIQUE(project_id, rel_path)
);

CREATE INDEX IF NOT EXISTS idx_project_files_project ON project_files(project_id);
CREATE INDEX IF NOT EXISTS idx_project_files_hash ON project_files(content_hash);
"#,
    )
    .map_err(|e| format!("failed creating project_files table: {}", e))?;

    // Code intelligence: symbol index tables
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_path TEXT NOT NULL,
    doc_rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    parent_symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    signature TEXT NOT NULL DEFAULT '',
    doc_comment TEXT NOT NULL DEFAULT '',
    visibility TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_project ON symbols(project_id);
CREATE INDEX IF NOT EXISTS idx_symbols_doc ON symbols(doc_path);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

CREATE TABLE IF NOT EXISTS symbol_chunk_map (
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES project_chunks(id) ON DELETE CASCADE,
    coverage TEXT NOT NULL DEFAULT 'full',
    PRIMARY KEY(symbol_id, chunk_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(
    name,
    qualified_name,
    signature,
    doc_comment,
    content='symbols',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbol_fts(rowid, name, qualified_name, signature, doc_comment)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbol_fts(symbol_fts, rowid, name, qualified_name, signature, doc_comment)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.doc_comment);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbol_fts(symbol_fts, rowid, name, qualified_name, signature, doc_comment)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.doc_comment);
    INSERT INTO symbol_fts(rowid, name, qualified_name, signature, doc_comment)
    VALUES (new.id, new.name, new.qualified_name, new.signature, new.doc_comment);
END;
"#,
    )
    .map_err(|e| format!("failed creating symbol index tables: {}", e))?;

    // Import graph tables
    conn.execute_batch(
        r#"
CREATE TABLE IF NOT EXISTS file_imports (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_doc_path TEXT NOT NULL,
    import_kind TEXT NOT NULL,
    raw_specifier TEXT NOT NULL,
    resolved_path TEXT NOT NULL DEFAULT '',
    imported_names TEXT NOT NULL DEFAULT '',
    line_number INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_imports_project ON file_imports(project_id);
CREATE INDEX IF NOT EXISTS idx_file_imports_source ON file_imports(source_doc_path);

CREATE TABLE IF NOT EXISTS file_dependency_edges (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_doc_path TEXT NOT NULL,
    target_doc_path TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    imported_symbol_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    UNIQUE(project_id, source_doc_path, target_doc_path, edge_kind)
);

CREATE INDEX IF NOT EXISTS idx_file_dep_edges_project ON file_dependency_edges(project_id);
CREATE INDEX IF NOT EXISTS idx_file_dep_edges_source ON file_dependency_edges(source_doc_path);
CREATE INDEX IF NOT EXISTS idx_file_dep_edges_target ON file_dependency_edges(target_doc_path);
"#,
    )
    .map_err(|e| format!("failed creating import graph tables: {}", e))?;

    ensure_vector_model_column(conn)?;
    ensure_relation_feedback_quality_column(conn)?;
    ensure_tracked_roots_exclude_column(conn)?;
    ensure_chunk_code_intel_columns(conn)?;
    Ok(())
}

fn ensure_vector_model_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(project_vectors)")
        .map_err(|e| format!("failed inspecting project_vectors schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading project_vectors schema: {}", e))?;
    let mut has_model = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "model" {
            has_model = true;
            break;
        }
    }
    if !has_model {
        conn.execute(
            "ALTER TABLE project_vectors ADD COLUMN model TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating project_vectors.model: {}", e))?;
        conn.execute("UPDATE project_vectors SET model = '' WHERE model = ''", [])
            .map_err(|e| format!("failed finalizing project_vectors.model migration: {}", e))?;
    }
    Ok(())
}

fn ensure_relation_feedback_quality_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(chunk_relation_feedback)")
        .map_err(|e| format!("failed inspecting chunk_relation_feedback schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading chunk_relation_feedback schema: {}", e))?;
    let mut has_quality = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "quality_label" {
            has_quality = true;
            break;
        }
    }
    if !has_quality {
        conn.execute(
            "ALTER TABLE chunk_relation_feedback ADD COLUMN quality_label TEXT NOT NULL DEFAULT 'unspecified'",
            [],
        )
        .map_err(|e| format!("failed migrating chunk_relation_feedback.quality_label: {}", e))?;
        conn.execute(
            "UPDATE chunk_relation_feedback SET quality_label = 'unspecified' WHERE quality_label = '' OR quality_label IS NULL",
            [],
        )
        .map_err(|e| {
            format!(
                "failed finalizing chunk_relation_feedback.quality_label migration: {}",
                e
            )
        })?;
    }
    conn.execute_batch(
        r#"
CREATE INDEX IF NOT EXISTS idx_chunk_relation_feedback_src_quality
    ON chunk_relation_feedback(src_chunk_id, quality_label, updated_at DESC);
"#,
    )
    .map_err(|e| {
        format!(
            "failed ensuring chunk_relation_feedback quality index: {}",
            e
        )
    })?;
    Ok(())
}

fn ensure_tracked_roots_exclude_column(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(tracked_roots)")
        .map_err(|e| format!("failed inspecting tracked_roots schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading tracked_roots schema: {}", e))?;
    let mut has_exclude = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "exclude_patterns" {
            has_exclude = true;
            break;
        }
    }
    if !has_exclude {
        conn.execute(
            "ALTER TABLE tracked_roots ADD COLUMN exclude_patterns TEXT NOT NULL DEFAULT ''",
            [],
        )
        .map_err(|e| format!("failed migrating tracked_roots.exclude_patterns: {}", e))?;
    }
    Ok(())
}

/// Migrate project_chunks table to include code intelligence columns.
/// Checks for `chunk_kind` as the sentinel — if missing, adds all 6 new columns.
fn ensure_chunk_code_intel_columns(conn: &Connection) -> Result<(), String> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(project_chunks)")
        .map_err(|e| format!("failed inspecting project_chunks schema: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed reading project_chunks schema: {}", e))?;
    let mut has_chunk_kind = false;
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating schema rows: {}", e))?
    {
        let name: String = row
            .get(1)
            .map_err(|e| format!("failed reading schema column name: {}", e))?;
        if name == "chunk_kind" {
            has_chunk_kind = true;
            break;
        }
    }
    if !has_chunk_kind {
        let alters = [
            "ALTER TABLE project_chunks ADD COLUMN chunk_kind TEXT NOT NULL DEFAULT 'text_window'",
            "ALTER TABLE project_chunks ADD COLUMN symbol_name TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE project_chunks ADD COLUMN parent_context TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE project_chunks ADD COLUMN line_start INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE project_chunks ADD COLUMN line_end INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE project_chunks ADD COLUMN context_header TEXT NOT NULL DEFAULT ''",
        ];
        for sql in &alters {
            conn.execute(sql, []).map_err(|e| {
                format!("failed migrating project_chunks code-intel columns: {}", e)
            })?;
        }
    }
    Ok(())
}

struct TrackedRoot {
    path: PathBuf,
    exclude_patterns: Vec<String>,
}

impl TrackedRoot {
    /// Returns absolute paths that should be excluded during traversal.
    fn absolute_excludes(&self) -> std::collections::HashSet<PathBuf> {
        self.exclude_patterns
            .iter()
            .map(|p| normalize_path(&self.path.join(p).to_string_lossy()))
            .collect()
    }
}

fn ensure_tracked_root_conn(conn: &Connection, path: &Path, added_at: f64) -> Result<(), String> {
    conn.execute(
        r#"
INSERT INTO tracked_roots(path, added_at, enabled)
VALUES (?1, ?2, 1)
ON CONFLICT(path) DO UPDATE SET enabled = 1
"#,
        params![path.to_string_lossy().to_string(), added_at],
    )
    .map_err(|e| format!("failed ensuring tracked root: {}", e))?;
    Ok(())
}

fn list_tracked_roots_conn(conn: &Connection) -> Result<Vec<PathBuf>, String> {
    Ok(list_tracked_roots_full_conn(conn)?
        .into_iter()
        .map(|r| r.path)
        .collect())
}

fn list_tracked_roots_full_conn(conn: &Connection) -> Result<Vec<TrackedRoot>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT path, exclude_patterns
FROM tracked_roots
WHERE enabled = 1
ORDER BY path
"#,
        )
        .map_err(|e| format!("failed preparing tracked roots query: {}", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| format!("failed querying tracked roots: {}", e))?;

    let mut out = Vec::new();
    for row in rows {
        let (path_str, excl_str) =
            row.map_err(|e| format!("failed reading tracked root row: {}", e))?;
        let exclude_patterns: Vec<String> = excl_str
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        out.push(TrackedRoot {
            path: PathBuf::from(path_str),
            exclude_patterns,
        });
    }
    Ok(out)
}

fn resolve_roots(
    conn: &Connection,
    _cfg: &ConfigValues,
    scope_roots: Option<Vec<PathBuf>>,
) -> Result<Vec<TrackedRoot>, String> {
    if let Some(roots) = scope_roots {
        // Ad-hoc roots from CLI: look up excludes from DB if the root is tracked,
        // otherwise use empty excludes.
        let all_tracked = list_tracked_roots_full_conn(conn)?;
        let mut out = Vec::new();
        for root in roots {
            let p = normalize_path(&root.to_string_lossy());
            if p.is_dir() {
                let excludes = all_tracked
                    .iter()
                    .find(|t| t.path == p)
                    .map(|t| t.exclude_patterns.clone())
                    .unwrap_or_default();
                out.push(TrackedRoot {
                    path: p,
                    exclude_patterns: excludes,
                });
            }
        }
        return Ok(out);
    }

    let rows = list_tracked_roots_full_conn(conn)?;
    Ok(rows
        .into_iter()
        .map(|r| TrackedRoot {
            path: normalize_path(&r.path.to_string_lossy()),
            exclude_patterns: r.exclude_patterns,
        })
        .collect())
}

fn project_excludes_for_path(project_dir: &Path, roots: &[TrackedRoot]) -> HashSet<PathBuf> {
    let project = normalize_path(&project_dir.to_string_lossy());
    let mut out: HashSet<PathBuf> = HashSet::new();
    for root in roots {
        if !(project == root.path || project.starts_with(&root.path)) {
            continue;
        }
        for abs in root.absolute_excludes() {
            if abs == project || abs.starts_with(&project) {
                out.insert(abs);
            }
        }
    }
    out
}

fn discover_projects(roots: &[TrackedRoot]) -> Vec<PathBuf> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<PathBuf> = Vec::new();

    for root in roots {
        if !root.path.is_dir() {
            continue;
        }
        let exclude_abs = root.absolute_excludes();
        let candidates = discover_root_projects(&root.path, &exclude_abs);
        for candidate in candidates {
            let key = candidate.to_string_lossy().to_string();
            if seen.insert(key) {
                out.push(candidate);
            }
        }
    }

    out.sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
    out
}

fn list_project_child_dirs(root: &Path, exclude_abs: &HashSet<PathBuf>) -> Vec<PathBuf> {
    let mut children: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(root) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || is_skip_dir(&name) {
                continue;
            }
            if let Ok(ft) = entry.file_type() {
                if ft.is_dir() {
                    let p = normalize_path(&entry.path().to_string_lossy());
                    if exclude_abs.contains(&p) {
                        continue;
                    }
                    children.push(p);
                }
            }
        }
    }
    children.sort_by_key(|p| p.file_name().map(|s| s.to_string_lossy().to_lowercase()));
    children
}

fn discover_root_projects(root: &Path, exclude_abs: &HashSet<PathBuf>) -> Vec<PathBuf> {
    // Unwrap common single-container roots (e.g. demo-data/projects/*) so users
    // can track the parent and still get project-level indexing.
    let mut cursor = normalize_path(&root.to_string_lossy());
    let container_names: HashSet<&str> = ["projects", "repos", "repositories", "workspaces"]
        .into_iter()
        .collect();
    for _ in 0..4 {
        let children = list_project_child_dirs(&cursor, exclude_abs);
        if children.is_empty() {
            return vec![cursor];
        }
        let mut expanded_from_containers: Vec<PathBuf> = Vec::new();
        for child in &children {
            let child_name = child
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if container_names.contains(child_name.as_str()) {
                let mut grand = list_project_child_dirs(child, exclude_abs);
                if grand.is_empty() {
                    expanded_from_containers.push(child.clone());
                } else {
                    expanded_from_containers.append(&mut grand);
                }
            }
        }
        if !expanded_from_containers.is_empty() {
            expanded_from_containers.sort_by_key(|p| {
                p.file_name()
                    .map(|s| s.to_string_lossy().to_lowercase())
                    .unwrap_or_default()
            });
            expanded_from_containers.dedup_by(|a, b| a == b);
            return expanded_from_containers;
        }
        if looks_like_single_project_root(&cursor) {
            return vec![cursor];
        }
        if looks_like_workspace_root(&cursor, &children) {
            return children;
        }
        if children.len() == 1 {
            let child_name = children[0]
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if container_names.contains(child_name.as_str()) {
                cursor = children[0].clone();
                continue;
            }
            if has_indexable_files_in_root(&cursor)
                && is_common_single_project_child_dir(child_name.as_str())
            {
                return vec![cursor];
            }
        }
        if children.len() == 1 && !looks_like_single_project_root(&cursor) {
            cursor = children[0].clone();
            continue;
        }
        return vec![cursor];
    }
    vec![cursor]
}

fn looks_like_workspace_root(root: &Path, children: &[PathBuf]) -> bool {
    if children.len() < 2 {
        return false;
    }
    !looks_like_single_project_root(root)
}

fn looks_like_single_project_root(root: &Path) -> bool {
    let marker_files = [
        "cargo.toml",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "makefile",
        "justfile",
    ];
    let Ok(rd) = fs::read_dir(root) else {
        return false;
    };
    let mut names: HashSet<String> = HashSet::new();
    for entry in rd.flatten() {
        if let Ok(ft) = entry.file_type() {
            if ft.is_file() {
                names.insert(entry.file_name().to_string_lossy().to_lowercase());
            }
        }
    }
    marker_files.iter().any(|marker| names.contains(*marker))
}

fn is_common_single_project_child_dir(name: &str) -> bool {
    matches!(
        name,
        "src" | "app" | "lib" | "cmd" | "tests" | "test" | "docs" | "scripts"
    )
}

#[cfg(test)]
mod project_discovery_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    fn temp_dir(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        p.push(format!("retrivio-{}-{}-{}", prefix, std::process::id(), n));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("create temp dir");
        p
    }

    #[test]
    fn discovers_children_inside_projects_container_even_with_root_readme() {
        let root = temp_dir("container-root");
        fs::write(root.join("README.md"), "root").expect("write readme");
        let projects = root.join("projects");
        fs::create_dir_all(projects.join("proj-a")).expect("mk proj-a");
        fs::create_dir_all(projects.join("proj-b")).expect("mk proj-b");
        fs::write(projects.join("proj-a").join("notes.md"), "a").expect("write proj-a");
        fs::write(projects.join("proj-b").join("notes.md"), "b").expect("write proj-b");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes);
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("proj-a")));
        assert!(got.iter().any(|p| p.ends_with("proj-b")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn keeps_single_repo_root_when_project_markers_exist() {
        let root = temp_dir("single-root");
        fs::write(root.join("Cargo.toml"), "[package]\nname='x'\n").expect("write cargo");
        fs::create_dir_all(root.join("src")).expect("mk src");
        fs::create_dir_all(root.join("docs")).expect("mk docs");
        fs::write(root.join("src").join("main.rs"), "fn main() {}").expect("write main");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], normalize_path(&root.to_string_lossy()));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn expands_workspace_children_even_with_root_files() {
        let root = temp_dir("workspace-root");
        fs::write(root.join("README.md"), "workspace").expect("write readme");
        fs::create_dir_all(root.join("alpha")).expect("mk alpha");
        fs::create_dir_all(root.join("beta")).expect("mk beta");
        fs::write(root.join("alpha").join("notes.md"), "a").expect("write alpha");
        fs::write(root.join("beta").join("notes.md"), "b").expect("write beta");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes);
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("alpha")));
        assert!(got.iter().any(|p| p.ends_with("beta")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn excludes_skip_directories_during_discovery() {
        let root = temp_dir("exclude-test");
        fs::create_dir_all(root.join("alpha")).expect("mk alpha");
        fs::create_dir_all(root.join("beta")).expect("mk beta");
        fs::create_dir_all(root.join("gamma")).expect("mk gamma");
        fs::write(root.join("alpha").join("notes.md"), "a").expect("write alpha");
        fs::write(root.join("beta").join("notes.md"), "b").expect("write beta");
        fs::write(root.join("gamma").join("notes.md"), "c").expect("write gamma");

        // Without excludes: all 3 children
        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let got = discover_root_projects(&root, &no_excludes);
        assert_eq!(got.len(), 3);

        // With beta excluded
        let mut excludes: HashSet<PathBuf> = HashSet::new();
        excludes.insert(normalize_path(&root.join("beta").to_string_lossy()));
        let got = discover_root_projects(&root, &excludes);
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.ends_with("alpha")));
        assert!(got.iter().any(|p| p.ends_with("gamma")));
        assert!(!got.iter().any(|p| p.ends_with("beta")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn collect_project_corpus_respects_excluded_subdirs() {
        let root = temp_dir("corpus-exclude");
        fs::create_dir_all(root.join("src")).expect("mk src");
        fs::create_dir_all(root.join("tmp")).expect("mk tmp");
        fs::write(
            root.join("src").join("main.rs"),
            "fn keep_me() { println!(\"ok\"); }\n",
        )
        .expect("write src");
        fs::write(
            root.join("tmp").join("generated.rs"),
            "fn generated_artifact() { println!(\"x\"); }\n".repeat(120),
        )
        .expect("write tmp");

        let no_excludes: HashSet<PathBuf> = HashSet::new();
        let newest = project_latest_mtime(&root, &no_excludes).expect("mtime");
        let all = collect_project_corpus(&root, 100_000, newest, &no_excludes).expect("all corpus");
        assert!(all
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("tmp/")));

        let mut excludes: HashSet<PathBuf> = HashSet::new();
        excludes.insert(normalize_path(&root.join("tmp").to_string_lossy()));
        let newest_excluded = project_latest_mtime(&root, &excludes).expect("mtime excluded");
        let filtered =
            collect_project_corpus(&root, 100_000, newest_excluded, &excludes).expect("filtered");
        assert!(filtered
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("src/")));
        assert!(!filtered
            .chunks
            .iter()
            .any(|c| c.doc_rel_path.starts_with("tmp/")));

        let _ = fs::remove_dir_all(&root);
    }
}

fn has_indexable_files_in_root(root: &Path) -> bool {
    let Ok(rd) = fs::read_dir(root) else {
        return false;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        let lname = name.to_lowercase();
        if lname == "readme" || lname == "readme.md" || lname == "notes.txt" {
            return true;
        }
        let ext = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if is_indexable_suffix(&format!(".{}", ext)) {
            return true;
        }
    }
    false
}

fn project_latest_mtime(project_dir: &Path, exclude_abs: &HashSet<PathBuf>) -> Result<f64, String> {
    let mut newest = file_mtime(project_dir).unwrap_or(0.0);
    let mut stack = vec![project_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if is_under_any(&path, exclude_abs) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if name.starts_with('.') || is_skip_dir(&name) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !ft.is_file() || name.starts_with('.') {
                continue;
            }
            if let Ok(meta) = entry.metadata() {
                let mt = metadata_mtime(&meta);
                if mt > newest {
                    newest = mt;
                }
            }
        }
    }
    Ok(newest)
}

fn collect_project_corpus(
    project_dir: &Path,
    max_chars: usize,
    newest_mtime: f64,
    exclude_abs: &HashSet<PathBuf>,
) -> Result<ProjectCorpus, String> {
    const MAX_FILE_BYTES: u64 = 2_000_000;
    const MAX_FILES_PER_PROJECT: usize = 2000;
    const MAX_CHUNKS_PER_FILE: usize = 28;
    const MAX_CHUNKS_PER_PROJECT: usize = 6000;
    const CHUNK_SIZE_CHARS: usize = 1000;
    const CHUNK_OVERLAP_CHARS: usize = 180;
    const SUMMARY_SNIPPET_CHARS: usize = 900;
    const SUMMARY_SNIPPET_FILES: usize = 30;

    let mut file_names: Vec<String> = Vec::new();
    let mut candidate_files: Vec<(f64, PathBuf, String)> = Vec::new();

    let mut stack = vec![project_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if is_under_any(&path, exclude_abs) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if name.starts_with('.') || is_skip_dir(&name) {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !ft.is_file() || name.starts_with('.') {
                continue;
            }

            let rel = path
                .strip_prefix(project_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            file_names.push(rel.clone());
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !is_indexable_suffix(&format!(".{}", ext)) {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
                continue;
            }
            candidate_files.push((metadata_mtime(&meta), path, rel));
        }
    }

    candidate_files.sort_by(|a, b| {
        let a_doc = is_doc_extension(&a.1);
        let b_doc = is_doc_extension(&b.1);
        b_doc.cmp(&a_doc).then_with(|| b.0.total_cmp(&a.0))
    });
    let mut chunks: Vec<ProjectChunk> = Vec::new();
    let mut snippets: Vec<String> = Vec::new();
    let mut indexed_files = 0i64;

    const AST_CHUNK_SIZE: usize = 1500; // larger for AST chunks since they're semantic units

    for (doc_mtime, path, rel) in candidate_files.into_iter().take(MAX_FILES_PER_PROJECT) {
        let Some(text) = read_for_index(&path) else {
            continue;
        };
        indexed_files += 1;
        if snippets.len() < SUMMARY_SNIPPET_FILES {
            let snippet: String = text.chars().take(SUMMARY_SNIPPET_CHARS).collect();
            snippets.push(format!("{}\n{}", rel, snippet));
        }

        // Try AST-aware chunking for code files, fall back to text windows for prose
        let is_code = code_intel::language_for_extension(
            &path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase(),
        )
        .is_some();

        let doc_path_normalized = normalize_path(&path.to_string_lossy())
            .to_string_lossy()
            .to_string();

        if is_code {
            // AST-aware chunking: produces semantic chunks (functions, classes, imports)
            let semantic_chunks = code_intel::analyze_file(&path, &text, &rel, AST_CHUNK_SIZE);
            for (chunk_index, sc) in semantic_chunks.into_iter().enumerate() {
                if chunks.len() >= MAX_CHUNKS_PER_PROJECT {
                    break;
                }
                if chunk_index >= MAX_CHUNKS_PER_FILE {
                    break;
                }
                let token_count = word_tokens(&sc.text).len() as i64;
                if token_count == 0 {
                    continue;
                }
                let mut hasher = Sha1::new();
                hasher.update(sc.text.as_bytes());
                let text_hash = format!("{:x}", hasher.finalize());
                let header = code_intel::build_context_header(&sc, &rel);
                let kind_str = match sc.kind {
                    code_intel::ChunkKind::ImportBlock => "import_block",
                    code_intel::ChunkKind::Preamble => "preamble",
                    code_intel::ChunkKind::Function => "function",
                    code_intel::ChunkKind::TypeHeader => "type_header",
                    code_intel::ChunkKind::Method => "method",
                    code_intel::ChunkKind::Declaration => "declaration",
                    code_intel::ChunkKind::TextWindow => "text_window",
                };
                chunks.push(ProjectChunk {
                    doc_path: doc_path_normalized.clone(),
                    doc_rel_path: rel.clone(),
                    doc_mtime,
                    chunk_index: chunk_index as i64,
                    token_count,
                    text_hash,
                    text: sc.text,
                    chunk_kind: kind_str.to_string(),
                    symbol_name: sc.symbol_name,
                    parent_context: sc.parent_context,
                    line_start: sc.line_start as i64,
                    line_end: sc.line_end as i64,
                    context_header: header,
                });
            }
        } else {
            // Fallback: character-window chunking for prose/non-code files
            for (chunk_index, chunk_text) in chunk_text(
                &text,
                CHUNK_SIZE_CHARS,
                CHUNK_OVERLAP_CHARS,
                MAX_CHUNKS_PER_FILE,
            )
            .into_iter()
            .enumerate()
            {
                if chunks.len() >= MAX_CHUNKS_PER_PROJECT {
                    break;
                }
                let token_count = word_tokens(&chunk_text).len() as i64;
                if token_count == 0 {
                    continue;
                }
                let mut hasher = Sha1::new();
                hasher.update(chunk_text.as_bytes());
                let text_hash = format!("{:x}", hasher.finalize());
                chunks.push(ProjectChunk {
                    doc_path: doc_path_normalized.clone(),
                    doc_rel_path: rel.clone(),
                    doc_mtime,
                    chunk_index: chunk_index as i64,
                    token_count,
                    text_hash,
                    text: chunk_text,
                    chunk_kind: "text_window".to_string(),
                    symbol_name: String::new(),
                    parent_context: String::new(),
                    line_start: 0,
                    line_end: 0,
                    context_header: String::new(),
                });
            }
        }
        if chunks.len() >= MAX_CHUNKS_PER_PROJECT {
            break;
        }
    }

    let names_section = file_names
        .into_iter()
        .take(500)
        .collect::<Vec<_>>()
        .join(" ");
    let snippet_section = snippets.join("\n\n");
    let title = project_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("project")
        .replace('-', " ")
        .replace('_', " ");
    let mut summary = format!(
        "project {}\nindexed_files {}\nfiles {}\n\n{}",
        project_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("project"),
        indexed_files,
        names_section,
        snippet_section
    );
    if summary.chars().count() > max_chars {
        summary = summary.chars().take(max_chars).collect();
    }

    Ok(ProjectCorpus {
        doc: ProjectDoc {
            path: normalize_path(&project_dir.to_string_lossy()),
            title,
            summary,
            mtime: newest_mtime,
        },
        chunks,
    })
}

fn read_for_index(path: &Path) -> Option<String> {
    let raw = fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&raw);
    // Preserve whitespace for code files (indentation conveys scope structure).
    // Only collapse whitespace for prose files where it's noise.
    let is_code = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| code_intel::language_for_extension(&ext.to_lowercase()).is_some())
        .unwrap_or(false);
    let cleaned = if is_code {
        text.to_string()
    } else {
        collapse_whitespace(&text)
    };
    if cleaned.trim().is_empty() {
        return None;
    }
    Some(cleaned.chars().take(80_000).collect())
}

/// Compute xxhash64 of file contents. Very fast (~2GB/s).
fn content_hash_xxh64(content: &[u8]) -> String {
    format!("{:016x}", xxh64::xxh64(content, 0))
}

/// Load the existing file manifest for a project from SQLite.
/// Returns rel_path -> (file_size, file_mtime, content_hash).
fn load_file_manifest(
    conn: &Connection,
    project_id: i64,
) -> Result<HashMap<String, (i64, f64, String)>, String> {
    let mut stmt = conn
        .prepare("SELECT rel_path, file_size, file_mtime, content_hash FROM project_files WHERE project_id = ?1")
        .map_err(|e| format!("failed preparing file manifest query: {}", e))?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| format!("failed querying file manifest: {}", e))?;
    let mut out = HashMap::new();
    for row in rows {
        let (rel, size, mtime, hash) =
            row.map_err(|e| format!("failed reading file manifest row: {}", e))?;
        out.insert(rel, (size, mtime, hash));
    }
    Ok(out)
}

/// Update the file manifest entry after indexing a file.
fn upsert_file_manifest(
    conn: &Connection,
    project_id: i64,
    rel_path: &str,
    abs_path: &str,
    file_size: i64,
    file_mtime: f64,
    content_hash: &str,
    chunk_count: i64,
) -> Result<(), String> {
    let now = now_ts();
    conn.execute(
        r#"
INSERT INTO project_files(project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, last_indexed)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
ON CONFLICT(project_id, rel_path) DO UPDATE SET
    abs_path = excluded.abs_path,
    file_size = excluded.file_size,
    file_mtime = excluded.file_mtime,
    content_hash = excluded.content_hash,
    chunk_count = excluded.chunk_count,
    last_indexed = excluded.last_indexed
"#,
        params![project_id, rel_path, abs_path, file_size, file_mtime, content_hash, chunk_count, now],
    )
    .map_err(|e| format!("failed upserting file manifest: {}", e))?;
    Ok(())
}

/// Check if a file has changed by comparing mtime+size, then verifying with content hash.
/// Returns (changed: bool, content_hash: String).
fn file_has_changed(
    manifest: &HashMap<String, (i64, f64, String)>,
    rel_path: &str,
    file_size: i64,
    file_mtime: f64,
    content: &[u8],
) -> (bool, String) {
    let hash = content_hash_xxh64(content);
    match manifest.get(rel_path) {
        None => (true, hash), // New file
        Some((old_size, old_mtime, old_hash)) => {
            // Fast path: if mtime and size match, assume unchanged
            if *old_size == file_size && (*old_mtime - file_mtime).abs() < 0.001 {
                return (false, old_hash.clone());
            }
            // Verify with content hash (handles clock skew, touch without modification)
            if *old_hash == hash {
                return (false, hash);
            }
            (true, hash)
        }
    }
}

/// Remove manifest entries for files that no longer exist.
fn clean_file_manifest(
    conn: &Connection,
    project_id: i64,
    current_rel_paths: &HashSet<String>,
) -> Result<usize, String> {
    let existing = load_file_manifest(conn, project_id)?;
    let mut removed = 0;
    for rel_path in existing.keys() {
        if !current_rel_paths.contains(rel_path) {
            conn.execute(
                "DELETE FROM project_files WHERE project_id = ?1 AND rel_path = ?2",
                params![project_id, rel_path],
            )
            .map_err(|e| format!("failed cleaning file manifest: {}", e))?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn chunk_text(text: &str, size: usize, overlap: usize, max_chunks: usize) -> Vec<String> {
    let cleaned = collapse_whitespace(text);
    if cleaned.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = cleaned.chars().collect();
    if chars.len() <= size {
        return vec![cleaned];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let step = size.saturating_sub(overlap).max(1);
    let n = chars.len();
    while start < n && out.len() < max_chunks {
        let mut end = (start + size).min(n);
        let mut window: String = chars[start..end].iter().collect();
        if end < n {
            if let Some(split) = window.rfind(' ') {
                if split > ((size as f32) * 0.60) as usize {
                    window = window[..split].to_string();
                    end = start + window.chars().count();
                }
            }
        }
        let trimmed = window.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
        if end >= n {
            break;
        }
        start += step;
    }
    out
}

fn word_tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if cur.len() >= 2 {
            out.push(cur.clone());
            cur.clear();
        } else {
            cur.clear();
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn is_under_any(path: &Path, candidates: &HashSet<PathBuf>) -> bool {
    if candidates.is_empty() {
        return false;
    }
    let resolved = normalize_path(&path.to_string_lossy());
    for candidate in candidates {
        if resolved == *candidate || resolved.starts_with(candidate) {
            return true;
        }
    }
    false
}

fn get_project_by_path(conn: &Connection, path: &str) -> Result<Option<ExistingProject>, String> {
    conn.query_row(
        "SELECT id, path, title, summary, project_mtime FROM projects WHERE path = ?1",
        params![path],
        |row| {
            Ok(ExistingProject {
                id: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                summary: row.get(3)?,
                project_mtime: row.get(4)?,
            })
        },
    )
    .optional()
    .map_err(|e| format!("failed fetching project row: {}", e))
}

fn upsert_project(
    conn: &Connection,
    path: &str,
    title: &str,
    summary: &str,
    project_mtime: f64,
    last_indexed: f64,
) -> Result<i64, String> {
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM projects WHERE path = ?1",
            params![path],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking existing project: {}", e))?;
    if let Some(id) = existing {
        conn.execute(
            r#"
UPDATE projects
SET title = ?1, summary = ?2, project_mtime = ?3, last_indexed = ?4
WHERE id = ?5
"#,
            params![title, summary, project_mtime, last_indexed, id],
        )
        .map_err(|e| format!("failed updating project row: {}", e))?;
        Ok(id)
    } else {
        conn.execute(
            r#"
INSERT INTO projects(path, title, summary, project_mtime, last_indexed)
VALUES (?1, ?2, ?3, ?4, ?5)
"#,
            params![path, title, summary, project_mtime, last_indexed],
        )
        .map_err(|e| format!("failed inserting project row: {}", e))?;
        Ok(conn.last_insert_rowid())
    }
}

fn has_project_vector(conn: &Connection, project_id: i64, model: &str) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM project_vectors WHERE project_id = ?1 AND model = ?2 LIMIT 1",
            params![project_id, model],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project vector: {}", e))?;
    Ok(row.is_some())
}

fn has_project_chunks(conn: &Connection, project_id: i64) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM project_chunks WHERE project_id = ?1 LIMIT 1",
            params![project_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project chunks: {}", e))?;
    Ok(row.is_some())
}

fn has_project_chunk_vectors(
    conn: &Connection,
    project_id: i64,
    model: &str,
) -> Result<bool, String> {
    let row: Option<i64> = conn
        .query_row(
            r#"
SELECT 1
FROM project_chunks pc
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pc.project_id = ?1 AND pcv.model = ?2
LIMIT 1
"#,
            params![project_id, model],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking chunk vectors: {}", e))?;
    Ok(row.is_some())
}

fn count_project_chunks(conn: &Connection, project_id: i64) -> Result<i64, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM project_chunks WHERE project_id = ?1",
        params![project_id],
        |row| row.get(0),
    )
    .map_err(|e| format!("failed counting project chunks: {}", e))
}

fn count_project_chunk_vectors(
    conn: &Connection,
    project_id: i64,
    model: &str,
) -> Result<i64, String> {
    conn.query_row(
        r#"
SELECT COUNT(*)
FROM project_chunks pc
JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
WHERE pc.project_id = ?1 AND pcv.model = ?2
"#,
        params![project_id, model],
        |row| row.get(0),
    )
    .map_err(|e| format!("failed counting chunk vectors: {}", e))
}

/// Returns (project_name, vectors_present, total_chunks) for projects with incomplete vectors.
fn count_incomplete_vector_projects(
    conn: &Connection,
    model: &str,
) -> Result<Vec<(String, i64, i64)>, String> {
    let mut stmt = conn
        .prepare(
            r#"
SELECT p.path,
       (SELECT COUNT(*) FROM project_chunks pc WHERE pc.project_id = p.id) AS chunk_count,
       (SELECT COUNT(*) FROM project_chunks pc
        JOIN project_chunk_vectors pcv ON pcv.chunk_id = pc.id
        WHERE pc.project_id = p.id AND pcv.model = ?1) AS vec_count
FROM projects p
HAVING chunk_count > 0 AND vec_count < chunk_count
"#,
        )
        .map_err(|e| format!("failed preparing incomplete vector query: {}", e))?;
    let rows = stmt
        .query_map(params![model], |row| {
            let path: String = row.get(0)?;
            let chunk_count: i64 = row.get(1)?;
            let vec_count: i64 = row.get(2)?;
            Ok((path, vec_count, chunk_count))
        })
        .map_err(|e| format!("failed querying incomplete vectors: {}", e))?;
    let mut result = Vec::new();
    for row in rows {
        let (path, vec_count, chunk_count) =
            row.map_err(|e| format!("failed reading incomplete vector row: {}", e))?;
        let name = Path::new(&path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&path)
            .to_string();
        result.push((name, vec_count, chunk_count));
    }
    Ok(result)
}

fn set_project_vector(
    conn: &Connection,
    project_id: i64,
    model: &str,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_vectors(project_id, model, dim, norm, vector)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(project_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector
"#,
        params![project_id, model, vector.len() as i64, norm, blob],
    )
    .map_err(|e| format!("failed upserting project vector: {}", e))?;
    Ok(())
}

fn clear_project_chunks(conn: &Connection, project_id: i64) -> Result<(), String> {
    conn.execute(
        "DELETE FROM project_chunks WHERE project_id = ?1",
        params![project_id],
    )
    .map_err(|e| format!("failed clearing project chunks: {}", e))?;
    Ok(())
}

/// Store extracted symbols for a project. Clears existing symbols first.
fn store_project_symbols(
    conn: &Connection,
    project_id: i64,
    symbols: &[code_intel::ExtractedSymbol],
    doc_path: &str,
    doc_rel_path: &str,
) -> Result<usize, String> {
    let now = now_ts();
    // Clear existing symbols for this doc
    conn.execute(
        "DELETE FROM symbols WHERE project_id = ?1 AND doc_path = ?2",
        params![project_id, doc_path],
    )
    .map_err(|e| format!("failed clearing symbols: {}", e))?;

    let mut count = 0;
    for sym in symbols {
        conn.execute(
            r#"
INSERT INTO symbols(project_id, doc_path, doc_rel_path, name, qualified_name, kind,
    line_start, line_end, signature, doc_comment, visibility, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
"#,
            params![
                project_id,
                doc_path,
                doc_rel_path,
                sym.name,
                sym.qualified_name,
                sym.kind,
                sym.line_start as i64,
                sym.line_end as i64,
                sym.signature,
                sym.doc_comment,
                sym.visibility,
                now
            ],
        )
        .map_err(|e| format!("failed inserting symbol: {}", e))?;
        count += 1;
    }
    Ok(count)
}

/// Store extracted imports for a source file, resolve them to file paths,
/// and create file_dependency_edges.
fn store_file_imports(
    conn: &Connection,
    project_id: i64,
    source_doc_path: &str,
    lang: code_intel::LanguageId,
    source_file: &Path,
    project_root: &Path,
    raw_imports: &[code_intel::RawImport],
    project_files: &[String],
) -> Result<usize, String> {
    let now = now_ts();

    // Clear existing imports for this source file
    conn.execute(
        "DELETE FROM file_imports WHERE project_id = ?1 AND source_doc_path = ?2",
        params![project_id, source_doc_path],
    )
    .map_err(|e| format!("failed clearing file imports: {}", e))?;

    // Clear existing dependency edges from this source
    conn.execute(
        "DELETE FROM file_dependency_edges WHERE project_id = ?1 AND source_doc_path = ?2",
        params![project_id, source_doc_path],
    )
    .map_err(|e| format!("failed clearing file dep edges: {}", e))?;

    let mut count = 0;
    let mut resolved_targets: HashMap<String, usize> = HashMap::new(); // target_path -> import_count

    for imp in raw_imports {
        let resolved = code_intel::resolve_import_path(
            lang,
            &imp.raw_specifier,
            source_file,
            project_root,
            project_files,
        );

        let names_joined = imp.imported_names.join(", ");

        conn.execute(
            r#"
INSERT INTO file_imports(project_id, source_doc_path, import_kind, raw_specifier,
    resolved_path, imported_names, line_number, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
"#,
            params![
                project_id,
                source_doc_path,
                imp.import_kind,
                imp.raw_specifier,
                resolved,
                names_joined,
                imp.line_number as i64,
                now
            ],
        )
        .map_err(|e| format!("failed inserting file import: {}", e))?;
        count += 1;

        // Track resolved targets for dependency edge creation
        if !resolved.is_empty() {
            let entry = resolved_targets.entry(resolved).or_insert(0);
            *entry += imp.imported_names.len().max(1);
        }
    }

    // Create file_dependency_edges from resolved imports
    for (target_path, symbol_count) in &resolved_targets {
        let weight = match *symbol_count {
            1 => 0.5,
            2..=5 => 0.7,
            _ => 0.9,
        };
        conn.execute(
            r#"
INSERT OR REPLACE INTO file_dependency_edges(
    project_id, source_doc_path, target_doc_path, edge_kind, weight,
    imported_symbol_count, updated_at)
VALUES (?1, ?2, ?3, 'imports', ?4, ?5, ?6)
"#,
            params![
                project_id,
                source_doc_path,
                target_path,
                weight,
                *symbol_count as i64,
                now
            ],
        )
        .map_err(|e| format!("failed inserting file dep edge: {}", e))?;
    }

    Ok(count)
}

/// Search symbols using FTS5 full-text search.
fn search_symbols_fts(conn: &Connection, query: &str, limit: usize) -> Result<Vec<Value>, String> {
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

fn upsert_project_chunk(
    conn: &Connection,
    project_id: i64,
    chunk: &ProjectChunk,
    updated_at: f64,
) -> Result<i64, String> {
    let existing: Option<i64> = conn
        .query_row(
            r#"
SELECT id
FROM project_chunks
WHERE project_id = ?1 AND doc_path = ?2 AND chunk_index = ?3
"#,
            params![project_id, chunk.doc_path, chunk.chunk_index],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("failed checking project chunk row: {}", e))?;
    if let Some(chunk_id) = existing {
        conn.execute(
            r#"
UPDATE project_chunks
SET doc_rel_path = ?1, doc_mtime = ?2, token_count = ?3, text_hash = ?4, text = ?5, updated_at = ?6,
    chunk_kind = ?7, symbol_name = ?8, parent_context = ?9, line_start = ?10, line_end = ?11, context_header = ?12
WHERE id = ?13
"#,
            params![
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at,
                chunk.chunk_kind,
                chunk.symbol_name,
                chunk.parent_context,
                chunk.line_start,
                chunk.line_end,
                chunk.context_header,
                chunk_id
            ],
        )
        .map_err(|e| format!("failed updating project chunk row: {}", e))?;
        Ok(chunk_id)
    } else {
        conn.execute(
            r#"
INSERT INTO project_chunks(
    project_id, doc_path, doc_rel_path, doc_mtime, chunk_index, token_count, text_hash, text, updated_at,
    chunk_kind, symbol_name, parent_context, line_start, line_end, context_header
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
"#,
            params![
                project_id,
                chunk.doc_path,
                chunk.doc_rel_path,
                chunk.doc_mtime,
                chunk.chunk_index,
                chunk.token_count,
                chunk.text_hash,
                chunk.text,
                updated_at,
                chunk.chunk_kind,
                chunk.symbol_name,
                chunk.parent_context,
                chunk.line_start,
                chunk.line_end,
                chunk.context_header
            ],
        )
        .map_err(|e| format!("failed inserting project chunk row: {}", e))?;
        Ok(conn.last_insert_rowid())
    }
}

fn set_project_chunk_vector(
    conn: &Connection,
    chunk_id: i64,
    model: &str,
    vector: &[f32],
) -> Result<(), String> {
    let norm = vector_norm(vector);
    let blob = f32_blob(vector);
    conn.execute(
        r#"
INSERT INTO project_chunk_vectors(chunk_id, model, dim, norm, vector)
VALUES (?1, ?2, ?3, ?4, ?5)
ON CONFLICT(chunk_id) DO UPDATE SET
    model = excluded.model,
    dim = excluded.dim,
    norm = excluded.norm,
    vector = excluded.vector
"#,
        params![chunk_id, model, vector.len() as i64, norm, blob],
    )
    .map_err(|e| format!("failed upserting chunk vector: {}", e))?;
    Ok(())
}

fn embed_and_store_chunk_batch(
    cwd: &Path,
    conn: &Connection,
    model_key: &str,
    embedder: &dyn Embedder,
    batch_ids: &[i64],
    batch_texts: &[String],
) -> Result<i64, String> {
    if batch_ids.is_empty() {
        return Ok(0);
    }
    const BATCH_MAX_RETRIES: usize = 3;
    let mut last_err = String::new();
    for attempt in 0..=BATCH_MAX_RETRIES {
        if attempt > 0 {
            let backoff_ms = 500u64 * 2u64.pow((attempt - 1) as u32);
            thread::sleep(Duration::from_millis(backoff_ms));
        }
        match embedder.embed_many(batch_texts) {
            Ok(vectors) => {
                if vectors.len() != batch_ids.len() {
                    last_err = format!(
                        "embed_many returned {} vectors for {} chunks",
                        vectors.len(),
                        batch_ids.len()
                    );
                    continue;
                }
                let batch_dim = vectors.first().map(|v| v.len()).unwrap_or(0);
                if batch_dim == 0 {
                    last_err = "embed_many returned empty vectors".to_string();
                    continue;
                }
                if vectors.iter().any(|v| v.len() != batch_dim) {
                    return Err(format!(
                        "embed_many returned mixed vector dimensions in a single batch (expected {})",
                        batch_dim
                    ));
                }
                if let Err(e) = get_or_open_lance(cwd, batch_dim) {
                    eprintln!(
                        "warning: LanceDB open failed (vectors still written to sqlite): {}",
                        e
                    );
                }

                let mut lance_batch: Vec<(i64, Vec<f32>)> = Vec::with_capacity(batch_ids.len());
                conn.execute_batch("BEGIN TRANSACTION;")
                    .map_err(|e| format!("failed starting chunk vector transaction: {}", e))?;
                for (chunk_id, vector) in batch_ids.iter().zip(vectors.into_iter()) {
                    if let Err(e) = set_project_chunk_vector(conn, *chunk_id, model_key, &vector) {
                        let _ = conn.execute_batch("ROLLBACK;");
                        return Err(e);
                    }
                    lance_batch.push((*chunk_id, vector));
                }
                conn.execute_batch("COMMIT;")
                    .map_err(|e| format!("failed committing chunk vector transaction: {}", e))?;
                if let Err(e) =
                    with_lance_store(|store| lance_store::upsert_chunks(store, &lance_batch))
                {
                    eprintln!(
                        "warning: LanceDB upsert failed ({}); attempting rebuild from sqlite",
                        e
                    );
                    let lance_path = data_dir(cwd).join("lance");
                    match lance_store::rebuild_from_sqlite(conn, model_key, &lance_path) {
                        Ok(rebuilt_store) => {
                            let lock = LANCE_STORE.get_or_init(|| Mutex::new(None));
                            let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
                            *guard = Some(rebuilt_store);
                        }
                        Err(rebuild_err) => {
                            eprintln!(
                                "warning: LanceDB rebuild failed (sqlite vectors remain intact): {}",
                                rebuild_err
                            );
                        }
                    }
                }
                return Ok(batch_ids.len() as i64);
            }
            Err(e) => {
                last_err = e;
            }
        }
    }
    Err(format!(
        "embedding batch failed after {} retries: {}",
        BATCH_MAX_RETRIES, last_err
    ))
}

fn reindex_project_chunks(
    cwd: &Path,
    conn: &Connection,
    project_id: i64,
    model_key: &str,
    embedder: &dyn Embedder,
    chunks: &[ProjectChunk],
    now: f64,
    live_progress: Option<&Arc<LiveIndexProgress>>,
) -> Result<(i64, i64, i64), String> {
    const CHUNK_EMBED_BATCH: usize = 512;
    clear_project_chunks(conn, project_id)?;
    if chunks.is_empty() {
        return Ok((0, 0, 0));
    }
    let mut vectorized = 0i64;
    let mut batch_ids: Vec<i64> = Vec::with_capacity(CHUNK_EMBED_BATCH);
    let mut batch_texts: Vec<String> = Vec::with_capacity(CHUNK_EMBED_BATCH);
    let mut batch_token_total: u64 = 0;
    let mut tx_open = false;

    for chunk in chunks {
        if !tx_open {
            conn.execute_batch("BEGIN TRANSACTION;")
                .map_err(|e| format!("failed starting project chunk transaction: {}", e))?;
            tx_open = true;
        }

        let chunk_id = match upsert_project_chunk(conn, project_id, chunk, now) {
            Ok(id) => id,
            Err(e) => {
                if tx_open {
                    let _ = conn.execute_batch("ROLLBACK;");
                }
                return Err(e);
            }
        };
        batch_ids.push(chunk_id);

        // Use context_header + text for embedding when a header exists.
        // The header provides structural context (file path, parent type, symbol name)
        // that dramatically improves embedding quality for code chunks.
        let embed_text = if chunk.context_header.is_empty() {
            chunk.text.clone()
        } else {
            format!("{}\n{}", chunk.context_header, chunk.text)
        };
        batch_texts.push(embed_text);
        batch_token_total = batch_token_total.saturating_add(chunk.token_count.max(0) as u64);

        if batch_ids.len() >= CHUNK_EMBED_BATCH {
            conn.execute_batch("COMMIT;")
                .map_err(|e| format!("failed committing project chunk transaction: {}", e))?;
            tx_open = false;
            let stored = embed_and_store_chunk_batch(
                cwd,
                conn,
                model_key,
                embedder,
                &batch_ids,
                &batch_texts,
            )?;
            vectorized += stored;
            if let Some(lp) = live_progress {
                lp.add_chunks_done(stored as usize);
                lp.add_tokens_done(batch_token_total);
            }
            batch_ids.clear();
            batch_texts.clear();
            batch_token_total = 0;
        }
    }

    if tx_open {
        conn.execute_batch("COMMIT;")
            .map_err(|e| format!("failed committing project chunk transaction: {}", e))?;
    }
    if !batch_ids.is_empty() {
        let stored =
            embed_and_store_chunk_batch(cwd, conn, model_key, embedder, &batch_ids, &batch_texts)?;
        vectorized += stored;
        if let Some(lp) = live_progress {
            lp.add_chunks_done(stored as usize);
            lp.add_tokens_done(batch_token_total);
        }
    }

    Ok((chunks.len() as i64, vectorized, 0))
}

fn remove_projects_not_in(conn: &Connection, keep_paths: &[String]) -> Result<i64, String> {
    if keep_paths.is_empty() {
        let removed = conn
            .execute("DELETE FROM projects", [])
            .map_err(|e| format!("failed clearing projects table: {}", e))?;
        return Ok(removed as i64);
    }
    let keep_set: HashSet<String> = keep_paths.iter().cloned().collect();
    let mut stmt = conn
        .prepare("SELECT path FROM projects ORDER BY path")
        .map_err(|e| format!("failed preparing project list query: {}", e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("failed listing existing projects: {}", e))?;
    let mut delete_paths: Vec<String> = Vec::new();
    for row in rows {
        let path = row.map_err(|e| format!("failed reading existing project row: {}", e))?;
        if !keep_set.contains(&path) {
            delete_paths.push(path);
        }
    }
    let mut removed = 0i64;
    for path in delete_paths {
        removed += conn
            .execute("DELETE FROM projects WHERE path = ?1", params![path])
            .map_err(|e| format!("failed deleting stale project row: {}", e))?
            as i64;
    }
    Ok(removed)
}

fn rebuild_relationship_edges(
    conn: &Connection,
    docs_by_id: &HashMap<i64, ProjectDoc>,
) -> Result<i64, String> {
    if docs_by_id.is_empty() {
        return Ok(0);
    }
    let mut node_meta: HashMap<i64, (String, HashSet<String>, HashSet<String>)> = HashMap::new();
    for (project_id, doc) in docs_by_id {
        let signal_tokens: HashSet<String> = word_tokens(&doc.summary)
            .into_iter()
            .filter(|t| !is_graph_stopword(t))
            .collect();
        let mut name_tokens: HashSet<String> = word_tokens(&doc.title).into_iter().collect();
        if let Some(name) = doc.path.file_name().and_then(|s| s.to_str()) {
            for t in word_tokens(name) {
                name_tokens.insert(t);
            }
        }
        node_meta.insert(
            *project_id,
            (
                doc.path.to_string_lossy().to_string(),
                signal_tokens,
                name_tokens,
            ),
        );
    }

    // Load project centroid vectors for embedding-based similarity
    let project_vectors = load_project_vectors(conn)?;

    // Load import-based cross-project edges: for each project, find which other
    // projects its files depend on via resolved file_dependency_edges.
    let import_cross_edges = load_import_cross_project_edges(conn)?;

    let mut total_edges = 0i64;
    for (src_id, (_src_path, src_tokens, src_name_tokens)) in &node_meta {
        let mut edges: Vec<(String, String, f64)> = Vec::new();

        for (dst_id, (dst_path, _dst_tokens, dst_name_tokens)) in &node_meta {
            if src_id == dst_id {
                continue;
            }

            // Jaccard/mention-based edges (existing logic)
            let mention = src_tokens.intersection(dst_name_tokens).count();
            let mention_score = if mention >= 2 {
                0.8
            } else if mention == 1 && dst_name_tokens.len() <= 2 {
                0.4
            } else {
                0.0
            };
            let overlap = jaccard(src_name_tokens, dst_name_tokens);
            let overlap_score = if overlap >= 0.25 { overlap } else { 0.0 };
            let jaccard_weight = mention_score + overlap_score;
            if jaccard_weight >= 0.45 {
                let rounded = ((jaccard_weight.min(2.0) * 1000.0).round()) / 1000.0;
                edges.push((dst_path.clone(), "semantic_related".to_string(), rounded));
            }

            // Embedding-based similarity: cosine between project centroid vectors
            if let (Some((src_vec, src_norm)), Some((dst_vec, dst_norm))) =
                (project_vectors.get(src_id), project_vectors.get(dst_id))
            {
                let sim = cosine_raw(src_vec, dst_vec, *src_norm, *dst_norm);
                if sim >= 0.40 {
                    let rounded = ((sim.min(1.0) * 1000.0).round()) / 1000.0;
                    edges.push((
                        dst_path.clone(),
                        "embedding_similarity".to_string(),
                        rounded,
                    ));
                }
            }

            // Import-derived edges: file A in src_project imports file B in dst_project
            if let Some(import_count) = import_cross_edges
                .get(src_id)
                .and_then(|targets| targets.get(dst_id))
            {
                // Weight scales with number of import relationships
                let weight = match *import_count {
                    1 => 0.55,
                    2..=5 => 0.75,
                    _ => 0.90,
                };
                edges.push((dst_path.clone(), "imports_from".to_string(), weight));
            }
        }

        edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        if edges.len() > 60 {
            edges.truncate(60);
        }
        set_project_edges(conn, *src_id, &edges)?;
        total_edges += edges.len() as i64;
    }
    Ok(total_edges)
}

fn set_project_edges(
    conn: &Connection,
    project_id: i64,
    edges: &[(String, String, f64)],
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM project_edges WHERE src_project_id = ?1",
        params![project_id],
    )
    .map_err(|e| format!("failed clearing project edges: {}", e))?;
    for (dst, kind, weight) in edges {
        conn.execute(
            r#"
INSERT INTO project_edges(src_project_id, dst, kind, weight)
VALUES (?1, ?2, ?3, ?4)
"#,
            params![project_id, dst, kind, weight],
        )
        .map_err(|e| format!("failed inserting project edge: {}", e))?;
    }
    Ok(())
}

/// Load all project centroid vectors from SQLite.
/// Returns project_id -> (vector, norm) pairs.
fn load_project_vectors(conn: &Connection) -> Result<HashMap<i64, (Vec<f32>, f64)>, String> {
    let mut stmt = conn
        .prepare("SELECT project_id, vector, norm FROM project_vectors")
        .map_err(|e| format!("failed preparing project_vectors query: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed querying project_vectors: {}", e))?;
    let mut out: HashMap<i64, (Vec<f32>, f64)> = HashMap::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating project_vectors: {}", e))?
    {
        let project_id: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading project_id: {}", e))?;
        let blob: Vec<u8> = row
            .get(1)
            .map_err(|e| format!("failed reading vector blob: {}", e))?;
        let norm: f64 = row
            .get(2)
            .map_err(|e| format!("failed reading norm: {}", e))?;
        let vector = blob_to_f32_vec(&blob);
        out.insert(project_id, (vector, norm));
    }
    Ok(out)
}

/// Load cross-project import relationships from file_dependency_edges.
/// Returns src_project_id -> { dst_project_id -> import_count }.
/// This finds cases where files in one project import files in another project.
fn load_import_cross_project_edges(
    conn: &Connection,
) -> Result<HashMap<i64, HashMap<i64, usize>>, String> {
    // Join file_dependency_edges with project_chunks to find which project
    // each target file belongs to.
    let mut stmt = conn
        .prepare(
            r#"
SELECT DISTINCT
    e.project_id AS src_project_id,
    pc.project_id AS dst_project_id,
    COUNT(*) AS edge_count
FROM file_dependency_edges e
JOIN project_chunks pc ON pc.doc_rel_path = e.target_doc_path
    AND pc.project_id != e.project_id
GROUP BY e.project_id, pc.project_id
"#,
        )
        .map_err(|e| format!("failed preparing import cross-project query: {}", e))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("failed querying import cross-project edges: {}", e))?;
    let mut out: HashMap<i64, HashMap<i64, usize>> = HashMap::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("failed iterating import cross-project rows: {}", e))?
    {
        let src: i64 = row
            .get(0)
            .map_err(|e| format!("failed reading src_project_id: {}", e))?;
        let dst: i64 = row
            .get(1)
            .map_err(|e| format!("failed reading dst_project_id: {}", e))?;
        let count: i64 = row
            .get(2)
            .map_err(|e| format!("failed reading edge_count: {}", e))?;
        out.entry(src).or_default().insert(dst, count as usize);
    }
    Ok(out)
}

fn f32_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn vector_norm(vector: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for v in vector {
        let f = *v as f64;
        sum += f * f;
    }
    sum.sqrt()
}

fn metadata_mtime(meta: &fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn file_mtime(path: &Path) -> Option<f64> {
    let meta = fs::metadata(path).ok()?;
    Some(metadata_mtime(&meta))
}

fn is_skip_dir(name: &str) -> bool {
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
            | ".venv"
            | "venv"
            | ".idea"
            | "cdk.out"
            | ".next"
            | "dist"
            | "build"
            | "target"
    )
}

fn is_indexable_suffix(suffix: &str) -> bool {
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

fn is_doc_extension(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "md" | "markdown"
            | "txt"
            | "rst"
            | "adoc"
            | "html"
            | "htm"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "cfg"
            | "ini"
    )
}

fn is_graph_stopword(tok: &str) -> bool {
    matches!(
        tok,
        "the"
            | "and"
            | "for"
            | "with"
            | "from"
            | "into"
            | "this"
            | "that"
            | "project"
            | "file"
            | "files"
            | "readme"
            | "docs"
            | "notes"
            | "src"
            | "test"
            | "tests"
    )
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    if inter == 0 {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

trait Embedder {
    fn model_key(&self) -> String;
    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let rows = self.embed_many(&[text.to_string()])?;
        rows.into_iter()
            .next()
            .ok_or_else(|| "No embedding returned.".to_string())
    }
}

fn bedrock_profile_for_cfg(cfg: Option<&ConfigValues>) -> Option<String> {
    non_empty_env("RETRIVIO_AWS_PROFILE")
        .or_else(|| non_empty_env("AWS_PROFILE"))
        .or_else(|| cfg.and_then(|c| non_empty_string(c.aws_profile.trim())))
        .or_else(|| {
            let in_config =
                parse_ini_profile_names(&aws_config_file_path(), true).contains("default");
            let in_credentials =
                parse_ini_profile_names(&aws_credentials_file_path(), false).contains("default");
            if in_config || in_credentials {
                Some("default".to_string())
            } else {
                None
            }
        })
}

fn bedrock_region_for_cfg(cfg: Option<&ConfigValues>) -> String {
    if let Some(v) = non_empty_env("RETRIVIO_AWS_REGION")
        .or_else(|| non_empty_env("AWS_REGION"))
        .or_else(|| non_empty_env("AWS_DEFAULT_REGION"))
    {
        return v;
    }
    if let Some(v) = cfg.and_then(|c| non_empty_string(c.aws_region.trim())) {
        return v;
    }
    if let Some(profile) = bedrock_profile_for_cfg(cfg) {
        if let Some(region) = aws_region_for_profile_name(&profile) {
            return region;
        }
    }
    if let Some(region) = aws_region_for_profile_name("default") {
        return region;
    }
    "us-east-1".to_string()
}

fn bedrock_refresh_cmd_for_cfg(cfg: Option<&ConfigValues>) -> Option<String> {
    non_empty_env("RETRIVIO_AWS_REFRESH_CMD")
        .or_else(|| cfg.and_then(|c| non_empty_string(c.aws_refresh_cmd.trim())))
}

fn bedrock_concurrency_for_cfg(cfg: Option<&ConfigValues>) -> usize {
    non_empty_env("RETRIVIO_BEDROCK_CONCURRENCY")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_concurrency as usize))
        .unwrap_or(32)
        .clamp(1, 128)
}

fn bedrock_max_retries_for_cfg(cfg: Option<&ConfigValues>) -> usize {
    non_empty_env("RETRIVIO_BEDROCK_MAX_RETRIES")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_max_retries as usize))
        .unwrap_or(3)
        .clamp(0, 12)
}

fn bedrock_retry_base_ms_for_cfg(cfg: Option<&ConfigValues>) -> u64 {
    non_empty_env("RETRIVIO_BEDROCK_RETRY_BASE_MS")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_retry_base_ms as u64))
        .unwrap_or(250)
        .clamp(50, 10_000)
}

fn non_empty_string(v: &str) -> Option<String> {
    let t = v.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn bedrock_refresh_once_state() -> &'static Mutex<HashSet<String>> {
    BEDROCK_REFRESH_ONCE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn run_refresh_command_once(cmd: &str) -> Result<(), String> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let always = bool_env("RETRIVIO_AWS_REFRESH_ALWAYS", false);
    if !always {
        if let Ok(done) = bedrock_refresh_once_state().lock() {
            if done.contains(trimmed) {
                return Ok(());
            }
        }
    }

    let status = Command::new("bash")
        .arg("-lc")
        .arg(trimmed)
        .status()
        .map_err(|e| format!("failed executing aws refresh command: {}", e))?;
    if !status.success() {
        return Err(format!(
            "aws refresh command failed with status {}",
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ));
    }

    if !always {
        if let Ok(mut done) = bedrock_refresh_once_state().lock() {
            done.insert(trimmed.to_string());
        }
    }
    Ok(())
}

fn refresh_aws_credentials_if_configured(cfg: Option<&ConfigValues>) -> Result<(), String> {
    let Some(cmd) = bedrock_refresh_cmd_for_cfg(cfg) else {
        return Ok(());
    };
    run_refresh_command_once(&cmd)
}

fn bedrock_aws_cli_path() -> String {
    non_empty_env("RETRIVIO_AWS_CLI").unwrap_or_else(|| "aws".to_string())
}

fn build_embedder(cfg: &ConfigValues) -> Result<Box<dyn Embedder>, String> {
    match cfg.embed_backend.as_str() {
        "ollama" => Ok(Box::new(OllamaEmbedder::new(&cfg.embed_model))),
        "bedrock" => Ok(Box::new(BedrockEmbedder::new_with_config(
            &cfg.embed_model,
            Some(cfg),
        ))),
        other => Err(format!(
            "native index does not support backend '{}' yet; use ollama or bedrock",
            other
        )),
    }
}

/// Resolve the Ollama API host (respects OLLAMA_HOST env var).
fn ollama_host() -> String {
    env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Check if Ollama is reachable by hitting GET /api/tags with a short timeout.
/// Returns Ok(true) if any HTTP response, Ok(false) on transport error.
fn ollama_is_reachable() -> Result<bool, String> {
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

fn ollama_autostart_once_state() -> &'static Mutex<HashSet<String>> {
    OLLAMA_AUTOSTART_ONCE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn ollama_autostart_timeout() -> Duration {
    non_empty_env("RETRIVIO_OLLAMA_AUTOSTART_TIMEOUT_SEC")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|secs| Duration::from_secs(secs.clamp(2, 90)))
        .unwrap_or_else(|| Duration::from_secs(12))
}

fn maybe_autostart_ollama(host: &str) -> Result<bool, String> {
    if !bool_env("RETRIVIO_OLLAMA_AUTOSTART", true) {
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
fn ollama_list_local_models() -> Result<Vec<String>, String> {
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
fn ollama_pull_model(model: &str) -> Result<(), String> {
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

struct OllamaEmbedder {
    model: String,
    host: String,
    keep_alive: Option<String>,
    timeout_sec: u64,
    max_input_chars: usize,
}

impl OllamaEmbedder {
    fn new(model: &str) -> Self {
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

#[derive(Clone)]
struct AwsCredentials {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    expires_at: Option<u64>,
}

impl AwsCredentials {
    fn resolve(profile: Option<&str>, aws_cli: &str) -> Result<Self, String> {
        let mut cmd = Command::new(aws_cli);
        cmd.arg("configure").arg("export-credentials");
        if let Some(p) = profile {
            cmd.arg("--profile").arg(p);
        }
        let output = cmd
            .output()
            .map_err(|e| format!("aws configure export-credentials: {}", e))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "aws configure export-credentials failed: {}",
                stderr.trim()
            ));
        }
        let json: Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("failed parsing credentials JSON: {}", e))?;
        let access_key = json["AccessKeyId"]
            .as_str()
            .ok_or("missing AccessKeyId in exported credentials")?
            .to_string();
        let secret_key = json["SecretAccessKey"]
            .as_str()
            .ok_or("missing SecretAccessKey in exported credentials")?
            .to_string();
        let session_token = json["SessionToken"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let expires_at = json["Expiration"].as_str().and_then(parse_iso8601_to_unix);
        Ok(Self {
            access_key_id: access_key,
            secret_access_key: secret_key,
            session_token,
            expires_at,
        })
    }

    fn is_near_expiry(&self) -> bool {
        match self.expires_at {
            Some(exp) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                now + 300 >= exp
            }
            None => false,
        }
    }
}

fn parse_iso8601_to_unix(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let min: i64 = s.get(14..16)?.parse().ok()?;
    let sec: i64 = s.get(17..19)?.parse().ok()?;
    let (adj_m, adj_y) = if month <= 2 {
        (month + 9, year - 1)
    } else {
        (month - 3, year)
    };
    let era = if adj_y >= 0 { adj_y } else { adj_y - 399 } / 400;
    let yoe = (adj_y - era * 400) as u32;
    let doy = (153 * adj_m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i64 - 719468;
    Some((days * 86400 + hour * 3600 + min * 60 + sec) as u64)
}

fn unix_to_amz_date(secs: u64) -> (String, String) {
    let s = secs as i64;
    let day_secs = ((s % 86400) + 86400) % 86400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let sc = day_secs % 60;
    let mut days = s / 86400;
    if s < 0 && s % 86400 != 0 {
        days -= 1;
    }
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mon <= 2 { y + 1 } else { y };
    (
        format!("{:04}{:02}{:02}T{:02}{:02}{:02}Z", year, mon, d, h, m, sc),
        format!("{:04}{:02}{:02}", year, mon, d),
    )
}

fn hex_encode_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sigv4_sha256_hex(data: &[u8]) -> String {
    let hash = <sha2::Sha256 as Digest>::digest(data);
    hex_encode_bytes(&hash)
}

fn sigv4_hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<sha2::Sha256>;
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn uri_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

fn sigv4_canonical_uri(url_path: &str) -> String {
    // SigV4 requires URI-encoding each path segment of the already-encoded URL path.
    // This means %3A in the URL becomes %253A in the canonical request (double-encoding).
    url_path
        .split('/')
        .map(|seg| uri_encode_path_segment(seg))
        .collect::<Vec<_>>()
        .join("/")
}

fn sigv4_authorize(
    method: &str,
    host: &str,
    path: &str,
    body: &[u8],
    region: &str,
    service: &str,
    creds: &AwsCredentials,
) -> Vec<(String, String)> {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (timestamp, datestamp) = unix_to_amz_date(now_secs);
    let payload_hash = sigv4_sha256_hex(body);

    let mut header_pairs: Vec<(&str, String)> = vec![
        ("content-type", "application/json".to_string()),
        ("host", host.to_string()),
        ("x-amz-content-sha256", payload_hash.clone()),
        ("x-amz-date", timestamp.clone()),
    ];
    if let Some(token) = &creds.session_token {
        header_pairs.push(("x-amz-security-token", token.clone()));
    }
    header_pairs.sort_by_key(|(k, _)| k.to_string());

    let signed_headers: String = header_pairs
        .iter()
        .map(|(k, _)| *k)
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers: String = header_pairs
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v.trim()))
        .collect();

    let canonical_uri = sigv4_canonical_uri(path);
    let canonical_request = format!(
        "{}\n{}\n\n{}\n{}\n{}",
        method, canonical_uri, canonical_headers, signed_headers, payload_hash
    );

    let credential_scope = format!("{}/{}/{}/aws4_request", datestamp, region, service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        timestamp,
        credential_scope,
        sigv4_sha256_hex(canonical_request.as_bytes())
    );

    let k_date = sigv4_hmac_sha256(
        format!("AWS4{}", creds.secret_access_key).as_bytes(),
        datestamp.as_bytes(),
    );
    let k_region = sigv4_hmac_sha256(&k_date, region.as_bytes());
    let k_service = sigv4_hmac_sha256(&k_region, service.as_bytes());
    let k_signing = sigv4_hmac_sha256(&k_service, b"aws4_request");
    let signature = hex_encode_bytes(&sigv4_hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        creds.access_key_id, credential_scope, signed_headers, signature
    );

    let mut result = vec![
        ("Authorization".to_string(), authorization),
        ("Content-Type".to_string(), "application/json".to_string()),
        ("x-amz-content-sha256".to_string(), payload_hash),
        ("x-amz-date".to_string(), timestamp),
    ];
    if let Some(token) = &creds.session_token {
        result.push(("x-amz-security-token".to_string(), token.clone()));
    }
    result
}

// ---- BedrockEmbedder ----

#[derive(Clone)]
struct BedrockEmbedder {
    model: String,
    region: String,
    profile: Option<String>,
    refresh_cmd: Option<String>,
    normalize: bool,
    aws_cli: String,
    concurrency: usize,
    max_retries: usize,
    retry_base_ms: u64,
    credentials: Arc<Mutex<Option<AwsCredentials>>>,
    http_agent: ureq::Agent,
    /// Adaptive concurrency: decreases on throttle, recovers on success.
    active_concurrency: Arc<AtomicUsize>,
    /// Count of consecutive successful embed_many calls (no throttles).
    consecutive_ok: Arc<AtomicU64>,
}

impl BedrockEmbedder {
    fn new(model: &str) -> Self {
        Self::new_with_config(model, None)
    }

    fn new_with_config(model: &str, cfg: Option<&ConfigValues>) -> Self {
        let model_name = if model.trim().is_empty() {
            default_embed_model_for_backend("bedrock").to_string()
        } else {
            model.trim().to_string()
        };
        let region = bedrock_region_for_cfg(cfg);
        let profile = bedrock_profile_for_cfg(cfg);
        let refresh_cmd = bedrock_refresh_cmd_for_cfg(cfg);
        let normalize = bool_env("RETRIVIO_BEDROCK_NORMALIZE", true);
        let aws_cli = bedrock_aws_cli_path();
        let concurrency = bedrock_concurrency_for_cfg(cfg);
        let max_retries = bedrock_max_retries_for_cfg(cfg);
        let retry_base_ms = bedrock_retry_base_ms_for_cfg(cfg);
        let http_agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(60))
            .build();
        Self {
            model: model_name,
            region,
            profile,
            refresh_cmd,
            normalize,
            aws_cli,
            concurrency,
            max_retries,
            retry_base_ms,
            credentials: Arc::new(Mutex::new(None)),
            http_agent,
            active_concurrency: Arc::new(AtomicUsize::new(concurrency)),
            consecutive_ok: Arc::new(AtomicU64::new(0)),
        }
    }

    fn request_payload(&self, text: &str) -> Value {
        let model = self.model.to_ascii_lowercase();
        if model.contains("cohere.embed") {
            serde_json::json!({
                "texts": [text],
                "input_type": "search_document",
                "truncate": "END"
            })
        } else {
            serde_json::json!({
                "inputText": text,
                "normalize": self.normalize
            })
        }
    }

    fn parse_vector(data: &Value) -> Result<Vec<f32>, String> {
        if let Some(arr) = data.get("embedding").and_then(|v| v.as_array()) {
            let mut out = Vec::with_capacity(arr.len());
            for n in arr {
                if let Some(v) = n.as_f64() {
                    out.push(v as f32);
                }
            }
            if !out.is_empty() {
                return Ok(out);
            }
        }
        if let Some(rows) = data.get("embeddings").and_then(|v| v.as_array()) {
            if let Some(first) = rows.first().and_then(|v| v.as_array()) {
                let mut out = Vec::with_capacity(first.len());
                for n in first {
                    if let Some(v) = n.as_f64() {
                        out.push(v as f32);
                    }
                }
                if !out.is_empty() {
                    return Ok(out);
                }
            }
        }
        if let Some(rows) = data
            .pointer("/embeddingsByType/float")
            .and_then(|v| v.as_array())
        {
            if let Some(first) = rows.first().and_then(|v| v.as_array()) {
                let mut out = Vec::with_capacity(first.len());
                for n in first {
                    if let Some(v) = n.as_f64() {
                        out.push(v as f32);
                    }
                }
                if !out.is_empty() {
                    return Ok(out);
                }
            }
        }
        Err("Unexpected Bedrock embedding response format.".to_string())
    }

    fn ensure_credentials(&self) -> Option<AwsCredentials> {
        {
            if let Ok(guard) = self.credentials.lock() {
                if let Some(creds) = guard.as_ref() {
                    if !creds.is_near_expiry() {
                        return Some(creds.clone());
                    }
                }
            }
        }
        match AwsCredentials::resolve(self.profile.as_deref(), &self.aws_cli) {
            Ok(creds) => {
                if let Ok(mut guard) = self.credentials.lock() {
                    *guard = Some(creds.clone());
                }
                Some(creds)
            }
            Err(_) => None,
        }
    }

    fn invoke_model_http(&self, payload: &Value, creds: &AwsCredentials) -> Result<Value, String> {
        let encoded_model = uri_encode_path_segment(&self.model);
        let path = format!("/model/{}/invoke", encoded_model);
        let host = format!("bedrock-runtime.{}.amazonaws.com", self.region);
        let url = format!("https://{}{}", host, path);
        let body = serde_json::to_vec(payload)
            .map_err(|e| format!("failed serializing Bedrock request payload: {}", e))?;
        let headers = sigv4_authorize("POST", &host, &path, &body, &self.region, "bedrock", creds);
        let mut req = self.http_agent.post(&url);
        for (name, value) in &headers {
            req = req.set(name, value);
        }
        req = req.set("Accept", "application/json");
        let resp = req.send_bytes(&body).map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let body_text = resp.into_string().unwrap_or_default();
                format!(
                    "Bedrock invoke failed (model='{}', region='{}'): HTTP {} - {}",
                    self.model, self.region, code, body_text
                )
            }
            ureq::Error::Transport(t) => {
                format!(
                    "Bedrock invoke failed (model='{}', region='{}'): {}",
                    self.model, self.region, t
                )
            }
        })?;
        resp.into_json::<Value>()
            .map_err(|e| format!("failed parsing Bedrock response JSON: {}", e))
    }

    fn invoke_model_cli(&self, payload: &Value) -> Result<Value, String> {
        let temp = env::temp_dir();
        let nonce = format!(
            "{}-{}-{}",
            process::id(),
            now_ts(),
            BEDROCK_REQ_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let req_path = temp.join(format!("retrivio-bedrock-req-{}.json", nonce));
        let out_path = temp.join(format!("retrivio-bedrock-out-{}.json", nonce));
        let body = serde_json::to_vec(payload)
            .map_err(|e| format!("failed serializing Bedrock request payload: {}", e))?;
        fs::write(&req_path, &body).map_err(|e| {
            format!(
                "failed writing Bedrock request payload '{}': {}",
                req_path.display(),
                e
            )
        })?;

        let mut cmd = Command::new(&self.aws_cli);
        cmd.arg("bedrock-runtime")
            .arg("invoke-model")
            .arg("--model-id")
            .arg(&self.model)
            .arg("--content-type")
            .arg("application/json")
            .arg("--accept")
            .arg("application/json")
            .arg("--region")
            .arg(&self.region)
            .arg("--body")
            .arg(format!("fileb://{}", req_path.to_string_lossy()))
            .arg(out_path.to_string_lossy().to_string());
        if let Some(profile) = &self.profile {
            cmd.arg("--profile").arg(profile);
        }
        let output = cmd
            .output()
            .map_err(|e| format!("failed executing AWS CLI for Bedrock embeddings: {}", e))?;

        let _ = fs::remove_file(&req_path);
        if !output.status.success() {
            let _ = fs::remove_file(&out_path);
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let detail = if !stderr.is_empty() { stderr } else { stdout };
            return Err(format!(
                "Bedrock invoke failed (model='{}', region='{}'): {}",
                self.model, self.region, detail
            ));
        }
        let raw = fs::read_to_string(&out_path).map_err(|e| {
            format!(
                "failed reading Bedrock response body '{}': {}",
                out_path.display(),
                e
            )
        })?;
        let _ = fs::remove_file(&out_path);
        serde_json::from_str::<Value>(&raw)
            .map_err(|e| format!("failed parsing Bedrock response JSON: {}", e))
    }

    fn invoke_with_retry(&self, payload: &Value) -> Result<Value, String> {
        let mut attempt = 0usize;
        let max_attempts = self.max_retries + 1;
        loop {
            attempt += 1;
            embed_metric_request_start();
            let req_started = Instant::now();
            let creds = self.ensure_credentials();
            let result = match &creds {
                Some(c) => self.invoke_model_http(payload, c),
                None => self.invoke_model_cli(payload),
            };
            match result {
                Ok(parsed) => {
                    embed_metric_request_end(true, req_started.elapsed());
                    self.consecutive_ok.fetch_add(1, Ordering::Relaxed);
                    return Ok(parsed);
                }
                Err(msg) => {
                    embed_metric_request_end(false, req_started.elapsed());
                    if msg.contains("ExpiredToken") || msg.contains("expired") {
                        if let Ok(mut guard) = self.credentials.lock() {
                            *guard = None;
                        }
                    }
                    let is_throttle = msg.contains("Throttl")
                        || msg.contains("TooManyRequests")
                        || msg.contains("429");
                    let retryable = is_throttle
                        || msg.contains("timed out")
                        || msg.contains("ExpiredToken")
                        || msg.contains("expired");
                    if is_throttle {
                        embed_metric_throttle();
                        // Adaptive: halve active concurrency (min 1).
                        let prev = self.active_concurrency.load(Ordering::Relaxed);
                        let reduced = (prev / 2).max(1);
                        self.active_concurrency.store(reduced, Ordering::Relaxed);
                        self.consecutive_ok.store(0, Ordering::Relaxed);
                    }
                    if retryable && attempt < max_attempts {
                        embed_metric_retry();
                        let exp = ((attempt - 1).min(8)) as u32;
                        let backoff = self
                            .retry_base_ms
                            .saturating_mul(2u64.saturating_pow(exp))
                            .min(30_000);
                        thread::sleep(Duration::from_millis(backoff));
                        continue;
                    }
                    let profile = self.profile.as_deref().unwrap_or("<default>");
                    return Err(format!(
                        "Bedrock request failed (model='{}', region='{}', profile='{}', attempt={}/{}): {}",
                        self.model, self.region, profile, attempt, max_attempts, msg
                    ));
                }
            }
        }
    }

    fn is_cohere_model(&self) -> bool {
        self.model.to_ascii_lowercase().contains("cohere.embed")
    }

    fn request_payload_batch(&self, texts: &[String]) -> Value {
        serde_json::json!({
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END"
        })
    }

    fn parse_vectors(data: &Value) -> Result<Vec<Vec<f32>>, String> {
        if let Some(rows) = data.get("embeddings").and_then(|v| v.as_array()) {
            let mut result = Vec::with_capacity(rows.len());
            for row in rows {
                if let Some(arr) = row.as_array() {
                    let vec: Vec<f32> = arr
                        .iter()
                        .filter_map(|n| n.as_f64().map(|v| v as f32))
                        .collect();
                    if !vec.is_empty() {
                        result.push(vec);
                    }
                }
            }
            if !result.is_empty() {
                return Ok(result);
            }
        }
        if let Some(rows) = data
            .pointer("/embeddingsByType/float")
            .and_then(|v| v.as_array())
        {
            let mut result = Vec::with_capacity(rows.len());
            for row in rows {
                if let Some(arr) = row.as_array() {
                    let vec: Vec<f32> = arr
                        .iter()
                        .filter_map(|n| n.as_f64().map(|v| v as f32))
                        .collect();
                    if !vec.is_empty() {
                        result.push(vec);
                    }
                }
            }
            if !result.is_empty() {
                return Ok(result);
            }
        }
        if let Some(arr) = data.get("embedding").and_then(|v| v.as_array()) {
            let vec: Vec<f32> = arr
                .iter()
                .filter_map(|n| n.as_f64().map(|v| v as f32))
                .collect();
            if !vec.is_empty() {
                return Ok(vec![vec]);
            }
        }
        Err("Unexpected Bedrock embedding response format.".to_string())
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>, String> {
        let text = self.truncate_for_model(text);
        let payload = self.request_payload(&text);
        let data = self.invoke_with_retry(&payload)?;
        Self::parse_vector(&data)
    }

    /// Truncate text to stay within model input limits.
    ///
    /// Titan V2: 8,192 tokens max. At worst-case ~2 chars/token (code-heavy),
    /// 16,000 chars ≈ 8,000 tokens — safely under the limit.
    /// Cohere models handle truncation server-side via `"truncate": "END"`.
    fn truncate_for_model(&self, text: &str) -> String {
        if self.is_cohere_model() {
            return text.to_string();
        }
        const MAX_CHARS: usize = 16_000;
        if text.len() <= MAX_CHARS {
            return text.to_string();
        }
        // Find a clean char boundary at or before MAX_CHARS
        let truncated: String = text.chars().take(MAX_CHARS).collect();
        eprintln!(
            "  warning: truncated embedding input from {} to {} chars for {}",
            text.len(),
            truncated.len(),
            self.model
        );
        truncated
    }

    fn embed_batch_cohere(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let payload = self.request_payload_batch(texts);
        let data = self.invoke_with_retry(&payload)?;
        Self::parse_vectors(&data)
    }
}

impl Embedder for BedrockEmbedder {
    fn model_key(&self) -> String {
        format!("bedrock:{}:{}", self.region, self.model)
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(cmd) = &self.refresh_cmd {
            run_refresh_command_once(cmd)?;
        }

        // Adaptive concurrency: after 5 consecutive clean calls, ramp up by 1.
        let ok_streak = self.consecutive_ok.load(Ordering::Relaxed);
        let cur = self.active_concurrency.load(Ordering::Relaxed);
        if ok_streak >= 5 && cur < self.concurrency {
            self.active_concurrency
                .store((cur + 1).min(self.concurrency), Ordering::Relaxed);
            self.consecutive_ok.store(0, Ordering::Relaxed);
        }

        // Cohere models support batching up to 96 texts per API call
        if self.is_cohere_model() {
            const MAX_COHERE_BATCH: usize = 96;
            if texts.len() <= MAX_COHERE_BATCH {
                let out = self.embed_batch_cohere(texts)?;
                embed_metric_texts(out.len());
                return Ok(out);
            }
            // Split into batches and parallelize
            let batches: Vec<Vec<String>> =
                texts.chunks(MAX_COHERE_BATCH).map(|c| c.to_vec()).collect();
            let worker_count = self
                .active_concurrency
                .load(Ordering::Relaxed)
                .min(batches.len())
                .max(1);
            let (job_tx, job_rx) = mpsc::channel::<(usize, Vec<String>)>();
            for (idx, batch) in batches.iter().enumerate() {
                let _ = job_tx.send((idx, batch.clone()));
            }
            drop(job_tx);
            let shared_rx = Arc::new(Mutex::new(job_rx));
            let (result_tx, result_rx) = mpsc::channel::<(usize, Result<Vec<Vec<f32>>, String>)>();
            let mut workers = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                let embedder = self.clone();
                let rx = Arc::clone(&shared_rx);
                let tx = result_tx.clone();
                workers.push(thread::spawn(move || loop {
                    let next = {
                        let guard = match rx.lock() {
                            Ok(g) => g,
                            Err(_) => break,
                        };
                        guard.recv()
                    };
                    let Ok((idx, batch_texts)) = next else {
                        break;
                    };
                    let _ = tx.send((idx, embedder.embed_batch_cohere(&batch_texts)));
                }));
            }
            drop(result_tx);
            let mut ordered: Vec<Option<Vec<Vec<f32>>>> = vec![None; batches.len()];
            let mut first_error: Option<String> = None;
            for _ in 0..batches.len() {
                let (idx, result) = result_rx
                    .recv()
                    .map_err(|e| format!("failed receiving batch result: {}", e))?;
                match result {
                    Ok(vecs) => {
                        ordered[idx] = Some(vecs);
                    }
                    Err(err) => {
                        if first_error.is_none() {
                            first_error = Some(err);
                        }
                    }
                }
            }
            for w in workers {
                let _ = w.join();
            }
            if let Some(err) = first_error {
                return Err(err);
            }
            let mut out = Vec::with_capacity(texts.len());
            for row in ordered.into_iter().flatten() {
                out.extend(row);
            }
            embed_metric_texts(out.len());
            return Ok(out);
        }

        // Non-Cohere (e.g. Titan): one text per request, parallelized via thread pool
        let worker_count = self
            .active_concurrency
            .load(Ordering::Relaxed)
            .min(texts.len())
            .max(1);
        if worker_count <= 1 {
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                out.push(self.embed_single(text)?);
            }
            embed_metric_texts(out.len());
            return Ok(out);
        }

        let (job_tx, job_rx) = mpsc::channel::<(usize, String)>();
        for (idx, text) in texts.iter().enumerate() {
            job_tx
                .send((idx, text.clone()))
                .map_err(|e| format!("failed queueing Bedrock embedding job: {}", e))?;
        }
        drop(job_tx);

        let shared_rx = Arc::new(Mutex::new(job_rx));
        let (result_tx, result_rx) = mpsc::channel::<(usize, Result<Vec<f32>, String>)>();
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let embedder = self.clone();
            let rx = Arc::clone(&shared_rx);
            let tx = result_tx.clone();
            workers.push(thread::spawn(move || loop {
                let next = {
                    let guard = match rx.lock() {
                        Ok(g) => g,
                        Err(_) => break,
                    };
                    guard.recv()
                };
                let Ok((idx, text)) = next else {
                    break;
                };
                let result = embedder.embed_single(&text);
                let _ = tx.send((idx, result));
            }));
        }
        drop(result_tx);

        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut first_error: Option<String> = None;
        for _ in 0..texts.len() {
            let (idx, result) = result_rx
                .recv()
                .map_err(|e| format!("failed receiving Bedrock embedding result: {}", e))?;
            match result {
                Ok(vec) => {
                    if idx < ordered.len() {
                        ordered[idx] = Some(vec);
                    }
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        for worker in workers {
            let _ = worker.join();
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for (idx, row) in ordered.into_iter().enumerate() {
            let Some(vec) = row else {
                return Err(format!(
                    "Bedrock embedding worker did not return vector for item {}",
                    idx
                ));
            };
            out.push(vec);
        }
        embed_metric_texts(out.len());
        Ok(out)
    }
}

struct LocalHashEmbedder {
    dim: usize,
    synonym_map: HashMap<String, Vec<String>>,
}

impl LocalHashEmbedder {
    fn new(dim: usize) -> Self {
        let use_dim = dim.max(64);
        let groups: &[&[&str]] = &[
            &[
                "semantic", "meaning", "ontology", "taxonomy", "model", "schema", "layer",
            ],
            &["api", "service", "endpoint", "backend"],
            &["ui", "frontend", "interface", "ux"],
            &["storage", "database", "db", "persistence"],
            &["auth", "authentication", "login", "identity"],
            &["agent", "assistant", "automation"],
        ];
        let mut synonym_map: HashMap<String, Vec<String>> = HashMap::new();
        for group in groups {
            for token in *group {
                let mut list = Vec::new();
                for other in *group {
                    if other != token {
                        list.push((*other).to_string());
                    }
                }
                synonym_map.insert((*token).to_string(), list);
            }
        }
        Self {
            dim: use_dim,
            synonym_map,
        }
    }

    fn model_key_local(&self) -> String {
        format!("local-hash-v1:{}", self.dim)
    }

    fn embed_one_local(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let normalized = text.to_lowercase();
        let tokens = word_tokens(&normalized);
        if tokens.is_empty() {
            return vec;
        }

        for token in &tokens {
            self.add_feature(&mut vec, &format!("t:{}", token), 1.0);
            if let Some(expanded) = self.synonym_map.get(token) {
                for syn in expanded {
                    self.add_feature(&mut vec, &format!("s:{}", syn), 0.35);
                }
            }
        }
        for pair in tokens.windows(2) {
            self.add_feature(&mut vec, &format!("b:{}_{}", pair[0], pair[1]), 0.8);
        }

        let compact: String = normalized.chars().filter(|c| !c.is_whitespace()).collect();
        let compact_chars: Vec<char> = compact.chars().collect();
        if compact_chars.len() >= 3 {
            for tri in compact_chars.windows(3) {
                let trigram: String = tri.iter().collect();
                self.add_feature(&mut vec, &format!("c:{}", trigram), 0.15);
            }
        }

        let norm = vector_norm(&vec);
        if norm > 0.0 {
            for v in &mut vec {
                *v = (*v as f64 / norm) as f32;
            }
        }
        vec
    }

    fn add_feature(&self, vec: &mut [f32], feature: &str, weight: f32) {
        let mut hasher = Sha1::new();
        hasher.update(feature.as_bytes());
        let digest = hasher.finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&digest[..8]);
        let idx = (u64::from_le_bytes(first) as usize) % self.dim;
        let sign = if (digest[8] & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign * weight;
    }
}

impl Embedder for LocalHashEmbedder {
    fn model_key(&self) -> String {
        self.model_key_local()
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let out: Vec<Vec<f32>> = texts.iter().map(|t| self.embed_one_local(t)).collect();
        embed_metric_texts(out.len());
        Ok(out)
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        embed_metric_texts(1);
        Ok(self.embed_one_local(text))
    }
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {} >/dev/null 2>&1", shell_escape(name)))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn is_executable_file(path: &Path) -> bool {
    let meta = match fs::metadata(path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        return (meta.permissions().mode() & 0o111) != 0;
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn tracked_roots_count(db_path: &Path) -> Option<i64> {
    if !db_path.exists() {
        return Some(0);
    }
    let conn = open_db_read_only(db_path).ok()?;
    conn.query_row(
        "SELECT COUNT(*) FROM tracked_roots WHERE enabled = 1",
        [],
        |row| row.get::<_, i64>(0),
    )
    .ok()
}

fn database_ready(db_path: &Path) -> bool {
    if !db_path.exists() {
        return false;
    }
    match open_db_read_only(db_path) {
        Ok(conn) => conn
            .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
            .map(|_| true)
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn config_path(cwd: &Path) -> PathBuf {
    if let Some(path) = CLI_CONFIG_OVERRIDE.get() {
        return path.clone();
    }
    data_dir(cwd).join("config.toml")
}

fn db_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("retrivio.db")
}

fn data_dir(_cwd: &Path) -> PathBuf {
    if let Some(path) = CLI_DATA_DIR_OVERRIDE.get() {
        return path.clone();
    }
    expand_tilde("~/.retrivio")
}

fn find_repo_root() -> Option<PathBuf> {
    if let Ok(v) = env::var("RETRIVIO_REPO_DIR") {
        let p = expand_tilde(v);
        if p.is_dir() {
            return Some(p);
        }
    }
    let cwd = env::current_dir().ok()?;
    let mut cur = cwd;
    loop {
        if cur.join("Cargo.toml").exists() && cur.join("crates").join("retrivio").is_dir() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn load_config_values(path: &Path) -> std::collections::HashMap<String, String> {
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
    out
}

fn home_dir() -> Option<PathBuf> {
    env::var("HOME").ok().map(PathBuf::from)
}

fn expand_tilde<S: AsRef<str>>(s: S) -> PathBuf {
    let raw = s.as_ref();
    if raw == "~" {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{}'", escaped)
}
