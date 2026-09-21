//! The MCP server: client registration (JSON and TOML configs), the tool specs, tool dispatch, the status resource and the framed stdio loop.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::{env, fs, process};

use serde_json::Value;

use crate::api::{stats_payload_json, unique_valid_dirs};
use crate::config::{config_path, data_dir, db_path, load_config_values, ConfigValues};
use crate::db::{
    ensure_db_schema, ensure_retrieval_backend_ready, ensure_tracked_root_conn,
    list_tracked_roots_conn, open_db_read_only, open_db_rw, open_db_writer,
    reembed_requirement_reason, remove_tracked_root, WriterLock,
};
use crate::dossier;
use crate::embed::{ensure_native_embed_backend, model_key_for_cfg};
use crate::index::{count_incomplete_vector_projects, run_native_index, IndexRunOptions};
use crate::rank::{
    chunk_get_schema, chunk_related_schema, chunk_search_schema, doc_read_schema,
    list_neighbors_by_path, rank_chunks_native_with, rank_files_native_with, rank_projects_native,
    ranked_chunk_result_json, ranked_file_result_json, ranked_project_result_json,
    related_chunk_result_json, search_symbols_fts, RankOptions,
};
use crate::related::{
    apply_chunk_relation_decision, build_context_pack_native, indexed_chunk_by_id,
    indexed_doc_chunks_by_path, list_chunk_relation_feedback, normalize_relation_quality_label,
    related_chunks_native, relation_feedback_row_json, set_chunk_relation_quality,
    source_chunk_json, ContextPackOptions,
};
use crate::scan::{plan_scoped_refresh, IndexScope};
use crate::util::{
    home_dir, is_executable_file, normalize_path, now_ts, prompt_yes_no, resolve_command_path,
    truncate_text_chars, yes_no,
};

pub(crate) fn run_mcp_cmd(args: &[OsString]) {
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

pub(crate) fn resolve_retrivio_command_path_native() -> Option<PathBuf> {
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
    resolve_command_path("retrivio")
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum McpConfigFormat {
    Json,
    Toml,
}

pub(crate) struct McpToolTarget {
    name: &'static str,
    format: McpConfigFormat,
    detect_dir: &'static str,
    config_file: &'static str,
    legacy_key: Option<&'static str>,
}

pub(crate) const MCP_TOOL_TARGETS: &[McpToolTarget] = &[
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

pub(crate) fn mcp_register_json(path: &Path, key: &str, command: &str) -> Result<(), String> {
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

pub(crate) fn mcp_unregister_json(path: &Path, key: &str) -> Result<(), String> {
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

pub(crate) fn mcp_tool_has_entry_json(path: &Path, key: &str) -> bool {
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

pub(crate) fn mcp_tool_current_command_json(path: &Path, key: &str) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let root: Value = serde_json::from_str(&raw).ok()?;
    root.get("mcpServers")?
        .get(key)?
        .get("command")?
        .as_str()
        .map(|s| s.to_string())
}

// ── TOML config helpers (string manipulation) ────────────────────────

pub(crate) fn find_next_toml_section(raw: &str, from: usize) -> Option<usize> {
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

pub(crate) fn mcp_register_toml(path: &Path, key: &str, command: &str) -> Result<(), String> {
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

pub(crate) fn mcp_unregister_toml(path: &Path, key: &str) -> Result<(), String> {
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

pub(crate) fn mcp_tool_has_entry_toml(path: &Path, key: &str) -> bool {
    let raw = match fs::read_to_string(path) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let header = format!("[mcp_servers.{}]", key);
    raw.contains(&header)
}

pub(crate) fn mcp_tool_current_command_toml(path: &Path, key: &str) -> Option<String> {
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

pub(crate) fn mcp_tool_is_installed(home: &Path, target: &McpToolTarget) -> bool {
    home.join(target.detect_dir).is_dir()
}

pub(crate) fn mcp_tool_has_entry(home: &Path, target: &McpToolTarget) -> bool {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_tool_has_entry_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_tool_has_entry_toml(&path, "retrivio"),
    }
}

pub(crate) fn mcp_tool_current_command(home: &Path, target: &McpToolTarget) -> Option<String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_tool_current_command_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_tool_current_command_toml(&path, "retrivio"),
    }
}

pub(crate) fn mcp_tool_has_legacy_entry(home: &Path, target: &McpToolTarget) -> bool {
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

pub(crate) fn mcp_tool_register(
    home: &Path,
    target: &McpToolTarget,
    command: &str,
) -> Result<(), String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_register_json(&path, "retrivio", command),
        McpConfigFormat::Toml => mcp_register_toml(&path, "retrivio", command),
    }
}

pub(crate) fn mcp_tool_unregister(home: &Path, target: &McpToolTarget) -> Result<(), String> {
    let path = home.join(target.config_file);
    match target.format {
        McpConfigFormat::Json => mcp_unregister_json(&path, "retrivio"),
        McpConfigFormat::Toml => mcp_unregister_toml(&path, "retrivio"),
    }
}

pub(crate) fn mcp_tool_remove_legacy(home: &Path, target: &McpToolTarget) -> Result<(), String> {
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

pub(crate) fn run_mcp_register(args: &[OsString]) {
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
                prompt_yes_no(
                    &format!(
                        "  replace '{}' with 'retrivio' in {}?",
                        legacy_key, target.name
                    ),
                    true,
                )
                .unwrap_or_default()
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
                prompt_yes_no(
                    &format!(
                        "  update {} to use {}? (currently {})",
                        target.name, command, old_cmd
                    ),
                    true,
                )
                .unwrap_or_default()
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
            prompt_yes_no(&format!("  register with {}?", target.name), true).unwrap_or_default()
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

pub(crate) fn run_mcp_unregister(args: &[OsString]) {
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
            prompt_yes_no(&format!("  unregister from {}?", target.name), true).unwrap_or_default()
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

pub(crate) fn run_mcp_doctor() {
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

pub(crate) fn mcp_success_result(data: Value) -> Value {
    let text = serde_json::to_string_pretty(&data).unwrap_or_else(|_| "{}".to_string());
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
        "isError": false
    })
}

pub(crate) fn mcp_error_response(id: Value, code: i64, message: &str) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

pub(crate) fn mcp_response(id: Value, result: Value) -> Value {
    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub(crate) fn mcp_tool_specs() -> Vec<Value> {
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
            "description": "Semantic search across indexed files (with project context). Results carry content_date, age_days, freshness_tier, role (state/knowledge/record), verify, noise, raw_similarity (cosine) and superseded_by (newer file of the same handoff/status series).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "since_days": {"type": "number", "description": "Only return files whose content date is within this many days."},
                    "include_superseded": {"type": "boolean", "default": false, "description": "Rank superseded state files (older handoffs of a series) at full strength; implied when the query asks for history explicitly (history, previous/earlier version, what did ... say, originally, changelog, timeline, back in, a month name, a year or a date)."}
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "topic_dossier",
            "description": "Cross-folder topic dossier for broad questions (\"what do we know about X\", \"everything about Y\"): one retrieval pass grouped by project. Returns the top projects (default 6, max 8) that hold material about the topic, each with its best entry file (path, role, content_date, age_days, freshness_tier, verify, raw_similarity, why), the newest evidence date, the number of distinct files, a one-line reason and a weak flag when its best cosine sits under the recall floor; then related projects from the project graph and an instruction. Use search_files for a specific document and pack_context for depth on one project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The entity or topic, e.g. a customer, system or initiative name with a few words of context."},
                    "limit": {"type": "integer", "default": 6, "description": "Projects to return (1-8)."}
                },
                "required": ["topic"]
            }
        }),
        serde_json::json!({
            "name": "search_chunks",
            "description": "Semantic+keyword search across indexed chunks/segments. Results carry content_date, age_days, freshness_tier, role, verify, noise and raw_similarity (cosine).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 30},
                    "since_days": {"type": "number", "description": "Only return chunks whose content date is within this many days."}
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

pub(crate) fn mcp_status_resource() -> Result<String, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = open_db_read_only(&dbp)?;
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

pub(crate) fn mcp_tool_needs_rw(name: &str) -> bool {
    matches!(
        name,
        "suppress_relation"
            | "restore_relation"
            | "set_relation_quality"
            | "add_tracked_root"
            | "remove_tracked_root"
            | "run_incremental_index"
            | "run_forced_refresh"
    )
}

/// Optional `since_days` MCP argument: a positive number of days, otherwise no filter.
pub(crate) fn mcp_since_days(args: &Value) -> Option<f64> {
    args.get("since_days")
        .and_then(|v| v.as_f64())
        .filter(|d| d.is_finite() && *d > 0.0)
}

pub(crate) fn mcp_tool_call(name: &str, args: &Value) -> Result<Value, String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = if mcp_tool_needs_rw(name) {
        ensure_db_schema(&dbp)?;
        open_db_rw(&dbp)?
    } else {
        open_db_read_only(&dbp)?
    };

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
            let results: Vec<Value> = rows.iter().map(ranked_project_result_json).collect();
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
            let since_days = mcp_since_days(args);
            let include_superseded = args
                .get("include_superseded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            ensure_native_embed_backend(&cfg, "mcp search_files")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_files")?;
            let opts = RankOptions {
                since_days,
                include_superseded,
                min_raw_similarity: cfg.search_min_abs_score,
                ..RankOptions::default()
            };
            let rows = rank_files_native_with(&conn, &cfg, &query, limit, opts)?;
            let results: Vec<Value> = rows.iter().map(ranked_file_result_json).collect();
            Ok(serde_json::json!({"query": query, "count": results.len(), "results": results}))
        }
        "topic_dossier" => {
            let topic = args
                .get("topic")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("query").and_then(|v| v.as_str()))
                .unwrap_or_default()
                .trim()
                .to_string();
            if topic.is_empty() {
                return Err("topic must be non-empty".to_string());
            }
            let limit = args
                .get("limit")
                .and_then(|v| v.as_i64())
                .unwrap_or(dossier::DEFAULT_LIMIT as i64)
                .clamp(1, dossier::MAX_LIMIT as i64) as usize;
            ensure_native_embed_backend(&cfg, "mcp topic_dossier")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp topic_dossier")?;
            let d = dossier::build(&conn, &cfg, &topic, limit)?;
            Ok(dossier::to_json(&d))
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
            let since_days = mcp_since_days(args);
            ensure_native_embed_backend(&cfg, "mcp search_chunks")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp search_chunks")?;
            let rows = rank_chunks_native_with(&conn, &cfg, &query, limit, since_days)?;
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
            // The root mutation and the refresh that follows share one writer lock.
            let writer = WriterLock::try_acquire(&data_dir(&cwd))?;
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
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::roots(vec![root.clone()]),
                        force_all: true,
                        force_paths,
                        remove_missing: false,
                        reason: "mcp add root refresh",
                    },
                    false,
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
            let writer = WriterLock::try_acquire(&data_dir(&cwd))?;
            let removed = remove_tracked_root(&dbp, &normalize_path(&root))?;
            let mut out = serde_json::json!({"removed": removed, "path": root, "refreshed": false});
            if refresh {
                ensure_native_embed_backend(&cfg, "mcp remove_tracked_root refresh")?;
                ensure_retrieval_backend_ready(&cfg, true, "mcp remove_tracked_root refresh")?;
                let stats = run_native_index(
                    &cwd,
                    &cfg,
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::AllRoots,
                        force_all: false,
                        force_paths: HashSet::new(),
                        remove_missing: true,
                        reason: "mcp remove root refresh",
                    },
                    false,
                )?;
                out["refreshed"] = Value::Bool(true);
                out["stats"] = stats_payload_json(&stats);
            }
            Ok(out)
        }
        "run_incremental_index" => {
            ensure_native_embed_backend(&cfg, "mcp incremental index")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp incremental index")?;
            let writer = WriterLock::try_acquire(&data_dir(&cwd))?;
            let stats = run_native_index(
                &cwd,
                &cfg,
                &writer,
                IndexRunOptions {
                    scope: IndexScope::AllRoots,
                    force_all: false,
                    force_paths: HashSet::new(),
                    remove_missing: true,
                    reason: "mcp incremental index",
                },
                false,
            )?;
            Ok(serde_json::json!({"mode": "incremental", "stats": stats_payload_json(&stats)}))
        }
        "run_forced_refresh" => {
            ensure_native_embed_backend(&cfg, "mcp forced refresh")?;
            ensure_retrieval_backend_ready(&cfg, true, "mcp forced refresh")?;
            let writer = WriterLock::try_acquire(&data_dir(&cwd))?;
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
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::AllRoots,
                        force_all: true,
                        force_paths: HashSet::new(),
                        remove_missing: true,
                        reason: "mcp forced refresh",
                    },
                    false,
                )?;
                return Ok(
                    serde_json::json!({"mode": "forced_all", "stats": stats_payload_json(&stats)}),
                );
            }
            let dirs = unique_valid_dirs(&paths);
            if dirs.is_empty() {
                return Err("no valid directory paths provided".to_string());
            }
            let scope = {
                let writer_conn = open_db_writer(&dbp, &writer)?;
                plan_scoped_refresh(&writer_conn, &cfg, &dirs)?
            };
            let (scope_roots, scope_projects) = match &scope {
                IndexScope::Targets { roots, projects } => (roots.clone(), projects.clone()),
                IndexScope::AllRoots => (Vec::new(), Vec::new()),
            };
            let as_strings = |v: &[PathBuf]| -> Vec<String> {
                v.iter().map(|p| p.to_string_lossy().to_string()).collect()
            };
            let paths_out = as_strings(&scope.target_paths());
            let roots_out = as_strings(&scope_roots);
            let projects_out = as_strings(&scope_projects);
            let stats = run_native_index(
                &cwd,
                &cfg,
                &writer,
                IndexRunOptions {
                    scope,
                    force_all: true,
                    force_paths: HashSet::new(),
                    remove_missing: false,
                    reason: "mcp forced refresh",
                },
                false,
            )?;
            Ok(serde_json::json!({
                "mode": "forced_scoped",
                "paths": paths_out,
                "roots": roots_out,
                "projects": projects_out,
                "stats": stats_payload_json(&stats)
            }))
        }
        _ => Err(format!("unknown tool '{}'", name)),
    }
}

/// Auto-detecting MCP frame reader: supports both Content-Length (LSP) framing
/// and newline-delimited JSON (NDJSON) used by Claude Code v2.x.
pub(crate) fn read_mcp_frame<R: BufRead + Read>(
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

pub(crate) fn write_mcp_frame<W: Write>(
    writer: &mut W,
    value: &Value,
    ndjson: bool,
) -> Result<(), String> {
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

pub(crate) fn serve_mcp_native() -> Result<(), String> {
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        eprintln!("retrivio mcp serve: waiting for MCP client messages on stdio (Ctrl+C to exit)");
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
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
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
