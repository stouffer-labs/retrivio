//! The web graph viewer and the graph command: viewer lifecycle and runtime state, the embedded HTML, the view JSON, neighbours and lineage.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::api::{
    api_health_host_port, find_free_port, handle_api_request, parse_http_request, path_basename,
    path_in_tracked_roots, send_http_json, send_http_response,
};
use crate::config::{data_dir, db_path};
use crate::config_tui::clipped;
use crate::db::{ensure_db_schema, list_tracked_roots_conn, open_db_read_only, open_db_rw};
use crate::rank::{clip_text, list_neighbors_by_path};
use crate::util::{
    arg_value, command_exists, display_path_compact, normalize_path, now_ts,
    open_url_in_default_browser, pid_is_alive, run_shell_capture,
};

pub(crate) fn run_graph_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio graph [doctor|status|view|open|neighbors|lineage] [--path <project-or-child-path>] [--limit <n>] [--threshold <0..1>] [--depth <1..3>]");
        println!("quick examples:");
        println!("  retrivio ui");
        println!("  retrivio graph neighbors --path ~/projects/sample-project --limit 12");
        println!(
            "  retrivio graph lineage --path ~/projects/sample-project --depth 2 --threshold 0.60"
        );
        println!("note: retrieval uses embedded LanceDB + SQLite FTS5 (no external server needed)");
        return;
    }

    let mut action = "status".to_string();
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
            "doctor" | "status" | "view" | "open" | "neighbors" | "lineage" | "text" | "ui" => {
                action = s;
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

pub(crate) fn graph_viewer_has_nodes(host: &str, port: u16) -> Option<bool> {
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

pub(crate) fn graph_viewer_state(host: &str, port: u16) -> Option<Value> {
    let url = format!("http://{}:{}/graph/view/state", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    resp.into_json::<Value>().ok()
}

pub(crate) fn graph_view_state_json(conn: &Connection, cwd: &Path) -> Result<Value, String> {
    let mut roots: Vec<String> = list_tracked_roots_conn(conn)?
        .into_iter()
        .map(|p| {
            normalize_path(&p.to_string_lossy())
                .to_string_lossy()
                .to_string()
        })
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

pub(crate) fn local_graph_state(cwd: &Path) -> Option<Value> {
    let dbp = db_path(cwd);
    ensure_db_schema(&dbp).ok()?;
    let conn = open_db_rw(&dbp).ok()?;
    graph_view_state_json(&conn, cwd).ok()
}

pub(crate) fn graph_state_matches(remote: &Value, local: &Value) -> bool {
    remote.get("db_path") == local.get("db_path")
        && remote.get("tracked_roots_hash") == local.get("tracked_roots_hash")
}

pub(crate) fn graph_viewer_is_retrivio(host: &str, port: u16) -> Option<bool> {
    let url = format!("http://{}:{}/", host, port);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    let resp = agent.get(&url).call().ok()?;
    let body = resp.into_string().ok()?;
    Some(body.contains("Retrivio Graph Viewer"))
}

pub(crate) fn local_graph_has_nodes(cwd: &Path) -> bool {
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

pub(crate) fn stop_retrivio_graph_viewer_on_port(port: u16) -> bool {
    let Some(pid) = listener_pid_for_port(port) else {
        return false;
    };
    stop_pid_graceful(pid)
}

pub(crate) fn try_reuse_saved_graph_viewer(
    host: &str,
    requested_port: u16,
    cwd: &Path,
    local_state: Option<&Value>,
) -> Option<u16> {
    let saved = parse_graph_runtime_state(cwd)?;
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

pub(crate) fn ensure_graph_viewer_running(
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

pub(crate) fn resolve_graph_target_project(
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

pub(crate) fn run_graph_open_cmd(host: &str, port: u16) -> Result<(), String> {
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

pub(crate) fn run_graph_neighbors_cmd(
    raw_path: Option<&str>,
    limit: usize,
    min_weight: f64,
) -> Result<(), String> {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dbp = db_path(&cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_read_only(&dbp)?;
    let project_path = resolve_graph_target_project(&conn, raw_path)?;

    let limit = limit.clamp(1, 200);
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
    println!("{:<4} {:>7}  {:<18} target", "rank", "weight", "relation");
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

pub(crate) fn run_graph_lineage_cmd(
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

    let limit = limit.clamp(1, 200);
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
        "{:<4} {:<4} {:>7}  {:<18} path",
        "rank", "dir", "weight", "relation"
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
            "{:<4} {:>7}  {:<18} source -> target",
            "rank", "weight", "relation"
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

pub(crate) fn resolve_project_for_target(
    conn: &Connection,
    target: &str,
) -> Result<Option<String>, String> {
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

pub(crate) fn suggest_project_paths(
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
pub(crate) struct GraphTextEdge {
    source: String,
    target: String,
    kind: String,
    weight: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct GraphTextData {
    edges: Vec<GraphTextEdge>,
}

pub(crate) fn load_graph_text_data(
    cwd: &Path,
    focus_hint: Option<&str>,
    limit: usize,
) -> Result<GraphTextData, String> {
    let dbp = db_path(cwd);
    ensure_db_schema(&dbp)?;
    let conn = open_db_rw(&dbp)?;
    let limit = limit.clamp(12, 600);
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
    Ok(GraphTextData { edges })
}

#[derive(Clone, Debug)]
pub(crate) struct GraphTextSubgraph {
    nodes: Vec<String>,
    edges: Vec<GraphTextEdge>,
}

pub(crate) fn build_focus_subgraph(
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
            if selected.contains(&edge.source)
                && selected.insert(edge.target.clone())
                && selected.len() >= max_nodes
            {
                break;
            }
            if selected.contains(&edge.target)
                && selected.insert(edge.source.clone())
                && selected.len() >= max_nodes
            {
                break;
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

pub(crate) fn graph_runtime_dir_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("runtime").join("graph")
}

pub(crate) fn graph_runtime_state_path(cwd: &Path) -> PathBuf {
    graph_runtime_dir_path(cwd).join("runtime-state.json")
}

#[derive(Debug, Clone)]
pub(crate) struct GraphRuntimeState {
    pid: u32,
    host: String,
    port: u16,
}

pub(crate) fn load_graph_runtime_state(cwd: &Path) -> Option<Value> {
    let path = graph_runtime_state_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<Value>(&raw).ok()
}

pub(crate) fn parse_graph_runtime_state(cwd: &Path) -> Option<GraphRuntimeState> {
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

pub(crate) fn persist_graph_runtime_state(
    cwd: &Path,
    pid: u32,
    host: &str,
    port: u16,
) -> Result<(), String> {
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

pub(crate) fn clear_graph_runtime_state(cwd: &Path) {
    let _ = fs::remove_file(graph_runtime_state_path(cwd));
}

pub(crate) fn stop_pid_graceful(pid: u32) -> bool {
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

pub(crate) fn listener_pid_for_port(port: u16) -> Option<u32> {
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

pub(crate) fn graph_viewer_html() -> &'static str {
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

pub(crate) fn serve_graph_viewer(host: &str, port: u16) -> Result<(), String> {
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
                if let Err(err) = send_http_response(
                    &mut stream,
                    200,
                    "text/html; charset=utf-8",
                    graph_viewer_html().as_bytes(),
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

pub(crate) fn graph_view_data_json(
    conn: &Connection,
    focus: Option<&str>,
    limit: usize,
) -> Result<Value, String> {
    let use_limit = limit.clamp(10, 600);
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

pub(crate) fn project_chunks_preview_json(
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
        .query_map(params![target, limit.clamp(1, 400) as i64], |row| {
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
