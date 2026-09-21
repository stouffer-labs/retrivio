//! Command-line surface: help text, command typo hints, the stale local binary warning, shell hook detection and installation, and the simple command bodies (doctor, config, init, install, add, del, roots, exclude, index, refresh, reembed, prune, dossier, search, ui, stop, self-test).

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use crate::api::{
    api_health_host_port, daemon_default_host, daemon_default_port, find_free_port,
    spawn_api_daemon,
};
use crate::autotune::run_autotune_cmd;
use crate::config::{
    config_path, config_set_value, config_value_string, data_dir, db_path, load_config_values,
    print_config_values, write_config_file, ConfigValues,
};
use crate::config_tui::run_config_edit;
use crate::db::{
    add_exclude_patterns_conn, database_ready, ensure_db_schema, ensure_retrieval_backend_ready,
    ensure_tracked_root_conn, get_exclude_patterns_conn, list_tracked_roots_full_conn,
    mark_reembed_completed, open_db_read_only, open_db_rw, open_db_writer,
    reembed_requirement_reason, refresh_reembed_requirement_for_config_change,
    remove_exclude_patterns_conn, remove_tracked_root, tracked_roots_count, WriterLock,
    LANCE_STORE,
};
use crate::embed::{
    aws_cli_json, bedrock_aws_cli_path, bedrock_concurrency_for_cfg,
    bedrock_credential_cmd_for_cfg, bedrock_max_retries_for_cfg, bedrock_profile_for_cfg,
    bedrock_refresh_cmd_for_cfg, bedrock_region_for_cfg, bedrock_retry_base_ms_for_cfg,
    default_embed_model_for_backend, ensure_native_embed_backend,
    ensure_ollama_ready_for_add_refresh, format_bedrock_preflight_error, is_known_embed_backend,
    model_key_for_cfg, refresh_aws_credentials_if_configured, AwsCredentials, BedrockEmbedder,
    Embedder, LocalHashEmbedder, EMBED_BACKENDS,
};
use crate::graph_viewer::{graph_runtime_dir_path, run_graph_open_cmd};
use crate::index::{
    count_incomplete_vector_projects, index_run_verdict, print_index_stats, run_index_with_lock,
    run_index_with_strategy, run_native_index, IndexRunOptions,
};
use crate::mcp::resolve_retrivio_command_path_native;
use crate::prune::run_prune;
use crate::rank::{
    print_file_results, print_project_results, rank_files_native_with, rank_projects_native,
    search_files_response_json, search_projects_response_json, RankOptions,
};
use crate::scan::{plan_scoped_refresh, IndexScope};
use crate::util::{
    arg_value, bool_env, command_available, command_exists, expand_tilde, file_mtime,
    is_executable_file, non_empty_env, normalize_path, now_ts, pid_is_alive, prompt_yes_no,
    run_shell_capture, shell_escape, yes_no,
};
use crate::{dossier, lance_store};

pub(crate) const SHELL_HOOK_MARKER_START: &str = "# >>> retrivio shell >>>";
pub(crate) const SHELL_HOOK_MARKER_END: &str = "# <<< retrivio shell <<<";
pub(crate) const SHELL_WRAPPER_ENV: &str = "RETRIVIO_SHELL_WRAPPER";

pub(crate) const KNOWN_TOP_LEVEL_COMMANDS: &[&str] = &[
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
    "prune",
    "watch",
    "search",
    "pick",
    "jump",
    "jump-feed",
    "ui",
    "stop",
    "daemon",
    "bench",
    "api",
    "mcp",
    "self-test",
    "graph",
    "recall",
    "hook",
    "service",
];

pub(crate) fn damerau_levenshtein_ascii(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b_chars.len() + 1]; a_chars.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
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

pub(crate) fn likely_command_typo(input: &str) -> Option<&'static str> {
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

pub(crate) fn current_platform_exe_name() -> String {
    format!("retrivio{}", env::consts::EXE_SUFFIX)
}

pub(crate) fn repo_root_from_anchor_path(anchor: &Path) -> Option<PathBuf> {
    let mut cur = if anchor.is_file() {
        anchor.parent()?.to_path_buf()
    } else {
        anchor.to_path_buf()
    };
    loop {
        if cur.join("Cargo.toml").exists() && cur.join("crates").join("retrivio").is_dir() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

pub(crate) fn preferred_local_repo_binary(repo: &Path) -> Option<PathBuf> {
    let exe_name = current_platform_exe_name();
    let release = repo.join("target").join("release").join(&exe_name);
    let debug = repo.join("target").join("debug").join(&exe_name);
    let release_ok = is_executable_file(&release);
    let debug_ok = is_executable_file(&debug);
    match (release_ok, debug_ok) {
        (true, true) => {
            let release_mtime = file_mtime(&release).unwrap_or(0.0);
            let debug_mtime = file_mtime(&debug).unwrap_or(0.0);
            if debug_mtime > release_mtime {
                Some(debug)
            } else {
                Some(release)
            }
        }
        (true, false) => Some(release),
        (false, true) => Some(debug),
        (false, false) => None,
    }
}

pub(crate) fn stale_local_repo_binary_warning() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let current_exe = env::current_exe().ok()?;
    if !is_executable_file(&current_exe) {
        return None;
    }
    let repo = repo_root_from_anchor_path(&current_exe).or_else(find_repo_root)?;
    let preferred = preferred_local_repo_binary(&repo)?;
    let current_norm = normalize_path(&current_exe.to_string_lossy());
    let preferred_norm = normalize_path(&preferred.to_string_lossy());
    let repo_norm = normalize_path(&repo.to_string_lossy());
    if !current_norm.starts_with(&repo_norm) || current_norm == preferred_norm {
        return None;
    }
    Some((repo_norm, current_norm, preferred_norm))
}

pub(crate) fn is_mcp_serve_invocation(args: &[OsString]) -> bool {
    args.first()
        .and_then(|v| v.to_str())
        .map(|v| v == "mcp")
        .unwrap_or(false)
        && args
            .get(1)
            .and_then(|v| v.to_str())
            .map(|v| v == "serve")
            .unwrap_or(false)
}

pub(crate) fn should_warn_for_local_binary_mismatch(args: &[OsString]) -> bool {
    if env::var_os("RETRIVIO_SKIP_LOCAL_BINARY_WARNING").is_some() {
        return false;
    }
    if is_mcp_serve_invocation(args) {
        return true;
    }
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

pub(crate) fn maybe_warn_if_stale_local_binary(args: &[OsString]) {
    if !should_warn_for_local_binary_mismatch(args) {
        return;
    }
    let Some((repo, current, preferred)) = stale_local_repo_binary_warning() else {
        return;
    };
    eprintln!(
        "warning: running stale local retrivio binary '{}'; newer local build is '{}'",
        current.display(),
        preferred.display()
    );
    eprintln!(
        "hint: point local MCP configs at '{}' so they always use the newest local build",
        repo.join("retrivio").display()
    );
}

pub(crate) fn detect_active_shell_for_init() -> String {
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

pub(crate) fn shell_rc_path(shell: &str) -> Option<PathBuf> {
    let home = env::var("HOME").ok().map(PathBuf::from)?;
    match shell {
        "zsh" => Some(home.join(".zshrc")),
        "bash" => Some(home.join(".bashrc")),
        "fish" => Some(home.join(".config").join("fish").join("config.fish")),
        _ => None,
    }
}

pub(crate) fn shell_hook_block_present(rc_path: &Path) -> bool {
    fs::read_to_string(rc_path)
        .map(|raw| raw.contains(SHELL_HOOK_MARKER_START) && raw.contains(SHELL_HOOK_MARKER_END))
        .unwrap_or(false)
}

pub(crate) fn default_shell_hook_present_in_rc_files() -> bool {
    let Some(home) = env::var("HOME").ok().map(PathBuf::from) else {
        return false;
    };
    let bash = shell_hook_block_present(&home.join(".bashrc"));
    let zsh = shell_hook_block_present(&home.join(".zshrc"));
    bash || zsh
}

pub(crate) fn print_shell_activation_hint() {
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

pub(crate) fn maybe_prompt_shell_hook_setup_for_shorthand_query() {
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

pub(crate) fn print_help() {
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
    println!("  retrivio install [--no-system-install] [--no-shell-hook]");
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
    println!(
        "  retrivio prune [--dry-run] [path ...]  # drop index rows for files no longer in the corpus"
    );
    println!("  retrivio watch [--interval <seconds>] [--debounce-ms <ms>] [--once] [--quiet]");
    println!("  retrivio search [--view projects|files] [--limit <n>] <query...>");
    println!("  retrivio dossier [--limit <n>] [--json] <topic...>   # cross-folder topic dossier: top projects, entry files, related projects");
    println!(
        "  retrivio pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
    );
    println!("  retrivio jump [--files|--dirs] [--limit <n>] [query...]");
    println!("  retrivio ui [--host <addr>] [--port <n>]");
    println!("  retrivio stop [--wait <seconds>]");
    println!(
        "  retrivio daemon [start|stop|restart|status|logs [n]] [--host <addr>] [--port <n>] [--timeout <seconds>]"
    );
    println!("  retrivio bench [plan|doctor|export] [options]");
    println!("  retrivio graph [doctor|status|view|open|neighbors|lineage]  # advanced");
    println!("  retrivio api [--host <addr>] [--port <n>]");
    println!("  retrivio mcp [serve|doctor|register|unregister]");
    println!("  retrivio self-test");
    println!("  retrivio recall [--query <text>] [--cwd <dir>] [--session <id>] [--format json|text] [--reset-session]  # agent hook mode (reads hook JSON on stdin)");
    println!("  retrivio hook [install|uninstall|status] [--claude] [--codex] [--yes]   # Claude Code / Codex UserPromptSubmit hooks");
    println!("  retrivio service [install|uninstall|status]                            # background watcher (launchd)");
}

pub(crate) fn run_ui_cmd(args: &[OsString]) {
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

pub(crate) fn run_stop_cmd(args: &[OsString]) {
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

pub(crate) fn run_doctor(args: &[OsString]) {
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
    if embed_backend == "hash" {
        println!(
            "embed backend note: hash is offline feature hashing ({} dims, model key {}); for tests and smoke checks only, not semantic",
            LocalHashEmbedder::effective_dim(cfg.local_embed_dim.max(0) as usize),
            model_key_for_cfg(&cfg)
        );
    }
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

pub(crate) fn run_doctor_fix(cfg: &ConfigValues) -> Result<(), String> {
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
    let credential_cmd = bedrock_credential_cmd_for_cfg(Some(cfg));
    println!(
        "doctor --fix: bedrock preflight (model='{}', region='{}', profile='{}')",
        cfg.embed_model,
        region,
        profile.clone().unwrap_or_else(|| "<default>".to_string())
    );

    if credential_cmd.is_none() && !command_available(&aws_cli) {
        return Err(format!(
            "AWS CLI '{}' is not available/executable (set RETRIVIO_AWS_CLI or install aws cli, or set aws_credential_cmd)",
            aws_cli
        ));
    }
    if credential_cmd.is_none() {
        println!("doctor --fix: aws cli executable: ok ({})", aws_cli);
    }

    if bedrock_refresh_cmd_for_cfg(Some(cfg)).is_some() {
        println!("doctor --fix: running aws_refresh_cmd...");
        refresh_aws_credentials_if_configured(Some(cfg))?;
        println!("doctor --fix: credential refresh: ok");
    } else if credential_cmd.is_none() {
        println!(
            "doctor --fix: no credential hook set (configure via `retrivio setup` / `retrivio config set aws_credential_cmd ...`)"
        );
    }

    if let Some(cmd) = credential_cmd.as_deref() {
        println!("doctor --fix: running aws_credential_cmd...");
        let creds =
            AwsCredentials::resolve(profile.as_deref(), &aws_cli, Some(cmd)).map_err(|err| {
                format_bedrock_preflight_error(
                    "doctor --fix",
                    &region,
                    profile.as_deref(),
                    &format!("aws_credential_cmd failed: {}", err),
                )
            })?;
        if creds.is_near_expiry() {
            return Err(format_bedrock_preflight_error(
                "doctor --fix",
                &region,
                profile.as_deref(),
                "aws_credential_cmd returned credentials that are already expired or expire within 5 minutes",
            ));
        }
        println!("doctor --fix: aws_credential_cmd: ok (returned valid creds)");
    } else {
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
    }

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

pub(crate) fn run_config_cmd(args: &[OsString]) {
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

pub(crate) fn run_init(args: &[OsString]) {
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
                if !is_known_embed_backend(&v) {
                    eprintln!(
                        "error: invalid --embed-backend '{}' (one of {})",
                        v,
                        EMBED_BACKENDS.join(", ")
                    );
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

pub(crate) fn render_shell_init_script(shell: &str) -> String {
    let exec_path = env::current_exe()
        .ok()
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let exec_q = shell_escape(&exec_path);
    let wrapper_env = SHELL_WRAPPER_ENV;
    let passthrough = "init|install|setup|auth|add|del|roots|index|refresh|reembed|prune|watch|search|pick|jump|doctor|config|autotune|version|api|mcp|self-test|graph|bench|daemon|ui|stop|recall|hook|service|help|-h|--help";

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

pub(crate) fn run_install_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio install [--no-system-install] [--no-shell-hook]");
        println!("notes:");
        println!("  - uses global Retrivio state under ~/.retrivio by default");
        println!("  - ensures LanceDB directory exists (embedded, no external process)");
        println!("  - attempts to install fswatch for low-latency event-driven watch");
        println!("  - installs shell hook into ~/.bashrc and ~/.zshrc by default (eval \"$(retrivio init <shell>)\")");
        println!("  - native Rust api/mcp runtime is included by default");
        return;
    }

    let mut allow_system_install = true;
    let mut install_shell_hook = true;

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--no-system-install" => {
                allow_system_install = false;
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

    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
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
}

pub(crate) fn render_shell_hook_block(shell: &str, preferred_bin: &Path) -> String {
    let block = format!(
        "{start}\nif [ -x {bin} ]; then\n  eval \"$({bin} init {shell})\"\nelif command -v retrivio >/dev/null 2>&1; then\n  eval \"$(retrivio init {shell})\"\nfi\n{end}\n",
        start = SHELL_HOOK_MARKER_START,
        bin = shell_escape(&preferred_bin.to_string_lossy()),
        shell = shell,
        end = SHELL_HOOK_MARKER_END
    );
    block
}

pub(crate) fn ensure_shell_hook_in_rc_file(
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

pub(crate) fn install_default_shell_hooks(cwd: &Path) -> Result<Vec<String>, String> {
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

pub(crate) fn try_install_fswatch_with_homebrew() -> Result<bool, String> {
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

pub(crate) fn run_add(args: &[OsString]) {
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

    // Every tracked-root mutation happens under the writer lock, like every other write; the
    // refresh below runs under the same lock.
    let writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
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
                    eprintln!(
                        "warning: skipping initial index because Ollama is not ready: {}",
                        e
                    );
                    eprintln!(
                        "note: tracked roots were added. run `retrivio setup`, start Ollama, or change embed_backend; then run `retrivio index`."
                    );
                    return;
                }
            }
        }
        let force_paths: HashSet<PathBuf> = added.iter().cloned().collect();
        ensure_retrieval_backend_ready(&cfg, true, "add refresh")
            .and_then(|_| ensure_native_embed_backend(&cfg, "add refresh"))
            .and_then(|_| {
                run_index_with_lock(
                    &cwd,
                    &cfg,
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::roots(added.clone()),
                        force_all: true,
                        force_paths,
                        remove_missing: false,
                        reason: "add refresh",
                    },
                )
            })
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
    } else {
        println!("note: run `retrivio index` when ready.");
    }
}

pub(crate) fn run_del(args: &[OsString]) {
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

    // Under the writer lock: a running watcher or index must not see the root vanish
    // mid-run, and the refresh below runs under the same lock.
    let writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
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
        ensure_retrieval_backend_ready(&_cfg, true, "delete refresh")
            .and_then(|_| ensure_native_embed_backend(&_cfg, "delete refresh"))
            .and_then(|_| {
                run_index_with_lock(
                    &cwd,
                    &_cfg,
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::AllRoots,
                        force_all: false,
                        force_paths: HashSet::new(),
                        remove_missing: true,
                        reason: "delete refresh",
                    },
                )
            })
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
    } else if removed > 0 {
        println!("note: run `retrivio refresh` when convenient.");
    }
}

pub(crate) fn run_roots(args: &[OsString]) {
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

pub(crate) fn run_exclude_cmd(args: &[OsString]) {
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
    let _writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
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

pub(crate) fn run_include_cmd(args: &[OsString]) {
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
    let _writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
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

pub(crate) fn run_index_cmd(args: &[OsString]) {
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
    run_index_with_strategy(
        &cwd,
        &cfg,
        IndexRunOptions {
            scope: IndexScope::AllRoots,
            force_all: false,
            force_paths: HashSet::new(),
            remove_missing: true,
            reason: "index",
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

pub(crate) fn run_refresh_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio refresh [path ...]");
        println!("re-collects and re-embeds projects regardless of modification times.");
        println!("  no path     every project under every tracked root (also drops projects");
        println!("              whose directory is gone)");
        println!("  path        a tracked root: every project discovered under it");
        println!("              a project directory: exactly that project (its child directories");
        println!("              are never treated as projects of their own)");
        println!("              anything else is an error naming the project or root to use");
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
        scoped.push(normalize_path(&s));
    }

    ensure_retrieval_backend_ready(&cfg, true, "refresh").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "refresh").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    // Planning a scoped refresh reads project rows, so the writer lock is taken (and the
    // schema brought up to date) before the plan, not only before the index run.
    let writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let (scope, remove_missing) = if scoped.is_empty() {
        (IndexScope::AllRoots, true)
    } else {
        let conn = open_db_writer(&db_path(&cwd), &writer).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        let scope = plan_scoped_refresh(&conn, &cfg, &scoped).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(2);
        });
        (scope, false)
    };

    run_index_with_lock(
        &cwd,
        &cfg,
        &writer,
        IndexRunOptions {
            scope,
            force_all: true,
            force_paths: HashSet::new(),
            remove_missing,
            reason: "refresh",
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
}

pub(crate) fn run_reembed_cmd(args: &[OsString]) {
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

    let writer = WriterLock::try_acquire(&data_dir(&cwd)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let mut stats = run_native_index(
        &cwd,
        &cfg,
        &writer,
        IndexRunOptions {
            scope: IndexScope::AllRoots,
            force_all: true,
            force_paths: HashSet::new(),
            remove_missing: true,
            reason: "reembed",
        },
        true,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    if let Err(e) = index_run_verdict(&stats) {
        // Some project still holds vectors of the old model: LanceDB is not rebuilt and the
        // re-embed is not marked complete; the next `reembed` retries.
        print_index_stats(&stats, &cfg);
        eprintln!("error: {}", e);
        eprintln!(
            "hint: LanceDB was not rebuilt; rerun `retrivio reembed` once the failure is resolved"
        );
        process::exit(1);
    }

    let dbp = db_path(&cwd);
    let conn = open_db_writer(&dbp, &writer).unwrap_or_else(|e| {
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

pub(crate) fn run_prune_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio prune [--dry-run] [--no-compact] [path ...]");
        println!("removes index rows for files that are no longer part of a project's corpus:");
        println!(
            "deleted files, files under excluded or skip_dir_names directories, files past the"
        );
        println!(
            "per-project caps. Drops their chunks, LanceDB vectors, manifest, symbol and import"
        );
        println!("rows; removes projects whose directory is gone or excluded; and deletes LanceDB");
        println!("vectors that have no sqlite chunk. Re-reads and re-chunks files (no embedding).");
        println!("Real runs end by compacting LanceDB (deletes are tombstones until then) and");
        println!("dropping its old versions, which is what returns disk space.");
        println!("  --dry-run     report what would be removed without writing anything");
        println!("  --no-compact  skip the LanceDB compaction at the end");
        println!("  path ...      only projects at or under these directories");
        return;
    }
    let mut dry_run = false;
    let mut compact = true;
    let mut scoped: Vec<PathBuf> = Vec::new();
    for raw in args {
        let s = raw.to_string_lossy().to_string();
        if s == "--dry-run" || s == "-n" {
            dry_run = true;
            continue;
        }
        if s == "--no-compact" {
            compact = false;
            continue;
        }
        if s.starts_with('-') {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        }
        let p = normalize_path(&s);
        if !p.is_dir() {
            eprintln!(
                "note: {} is not a directory; only stale project rows under it can be pruned",
                p.display()
            );
        }
        scoped.push(p);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let scope = if scoped.is_empty() {
        None
    } else {
        Some(scoped)
    };
    if let Err(e) = run_prune(&cwd, &cfg, scope, dry_run, compact) {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

/// `retrivio dossier <topic> [--limit N] [--json]`: the explicit cross-folder dossier.
pub(crate) fn run_dossier_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio dossier [--limit <n>] [--json] <topic...>");
        println!(
            "  One fused retrieval pass grouped by project: the top {} (at most {}) projects that hold material about the topic, each with its best entry file (role, date, cosine), the newest evidence date, the number of distinct files and a one-line reason; then related projects from the project graph.",
            dossier::DEFAULT_LIMIT,
            dossier::MAX_LIMIT
        );
        println!("  --limit <n>  projects to list (1-{})", dossier::MAX_LIMIT);
        println!("  --json       emit the topic-dossier-v1 payload (the MCP tool topic_dossier returns the same)");
        return;
    }
    let mut limit = dossier::DEFAULT_LIMIT;
    let mut json_output = false;
    let mut topic_parts: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--json" {
            json_output = true;
        } else if s == "--limit" {
            i += 1;
            let v = arg_value(args, i, "--limit");
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
        } else if let Some(v) = s.strip_prefix("--limit=") {
            limit = v.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("error: --limit must be an integer");
                process::exit(2);
            });
        } else if s.starts_with('-') && topic_parts.is_empty() {
            eprintln!("error: unknown option '{}'", s);
            process::exit(2);
        } else {
            topic_parts.push(s);
        }
        i += 1;
    }
    if topic_parts.is_empty() {
        eprintln!("error: topic is empty");
        process::exit(2);
    }
    let topic = topic_parts.join(" ");
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    ensure_retrieval_backend_ready(&cfg, true, "dossier").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "dossier").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let dbp = db_path(&cwd);
    if !dbp.is_file() {
        eprintln!(
            "error: no index at {}; run `retrivio index` first",
            dbp.display()
        );
        process::exit(1);
    }
    let conn = open_db_read_only(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let d = dossier::build(&conn, &cfg, &topic, limit.clamp(1, dossier::MAX_LIMIT)).unwrap_or_else(
        |e| {
            eprintln!("error: {}", e);
            process::exit(1);
        },
    );
    if json_output {
        println!("{}", dossier::to_json(&d));
    } else {
        println!("{}", dossier::render_text(&d));
    }
}

pub(crate) fn run_search_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio search [--view projects|files] [--limit <n>] [--since <days>] [--include-superseded] [--json] <query...>"
        );
        println!("  --since <days>  files view only: drop results whose content date is older");
        println!("  --include-superseded  files view only: show older handoffs/status files of a series at full strength");
        println!("  --json          emit the same JSON payload as the API GET /search");
        return;
    }

    let mut limit: usize = 20;
    let mut view = "projects".to_string();
    let mut json_output = false;
    let mut since_days: Option<f64> = None;
    let mut include_superseded = false;
    let mut query_parts: Vec<String> = Vec::new();

    let parse_since = |v: &str| -> f64 {
        match v.trim().parse::<f64>() {
            Ok(d) if d.is_finite() && d > 0.0 => d,
            _ => {
                eprintln!("error: --since must be a positive number of days");
                process::exit(2);
            }
        }
    };

    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--json" {
            json_output = true;
            i += 1;
            continue;
        }
        if s == "--include-superseded" {
            include_superseded = true;
            i += 1;
            continue;
        }
        if s == "--since" {
            i += 1;
            since_days = Some(parse_since(&arg_value(args, i, "--since")));
            i += 1;
            continue;
        }
        if let Some(v) = s.strip_prefix("--since=") {
            since_days = Some(parse_since(v));
            i += 1;
            continue;
        }
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
    if since_days.is_some() && view != "files" {
        eprintln!("error: --since requires --view files");
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
    let search_started = Instant::now();
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    ensure_native_embed_backend(&cfg, "search").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!("hint: use `retrivio init --embed-backend <ollama|bedrock>`");
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    if !dbp.is_file() {
        eprintln!(
            "error: no index at {}; run `retrivio index` first",
            dbp.display()
        );
        process::exit(1);
    }
    // Search only reads; a read-only connection never contends with the watcher's writes.
    let conn = open_db_read_only(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    if view == "files" {
        let opts = RankOptions {
            since_days,
            include_superseded,
            min_raw_similarity: cfg.search_min_abs_score,
            ..RankOptions::default()
        };
        let rows = rank_files_native_with(&conn, &cfg, &query, limit, opts).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if json_output {
            println!(
                "{}",
                search_files_response_json(&query, &rows, search_started)
            );
            return;
        }
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
    if json_output {
        println!(
            "{}",
            search_projects_response_json(&query, &rows, search_started)
        );
        return;
    }
    if rows.is_empty() {
        println!("No results found.");
        return;
    }
    print_project_results(&rows);
}

pub(crate) fn run_self_test_cmd(args: &[OsString]) {
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
        match run_self_test_lifecycle_probe(&cwd, timeout_s) {
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

pub(crate) fn run_self_test_lifecycle_probe(
    cwd: &Path,
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

pub(crate) fn run_self_test_daemon_probe(cwd: &Path, timeout_s: u64) -> Result<(u16, u32), String> {
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

pub(crate) fn find_repo_root() -> Option<PathBuf> {
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
