//! The HTTP API: daemon lifecycle (pid and port files, spawn, stop), the request parser, the endpoint dispatcher and the response helpers.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use serde_json::Value;

use crate::config::{config_path, data_dir, db_path, load_config_values, ConfigValues};
use crate::db::{
    ensure_db_schema, ensure_retrieval_backend_ready, ensure_tracked_root, is_index_busy_error,
    list_tracked_roots, open_db_read_only, open_db_rw, open_db_writer, record_selection_event,
    remove_tracked_root, WriterLock,
};
use crate::embed::ensure_native_embed_backend;
use crate::graph_viewer::{
    graph_view_data_json, graph_view_state_json, project_chunks_preview_json,
};
use crate::index::{run_native_index, IndexRunOptions, IndexStats};
use crate::pick::{render_file_pick_line, render_project_pick_line};
use crate::rank::{
    chunk_get_schema, chunk_related_schema, chunk_search_schema, doc_read_schema,
    list_neighbors_by_path, rank_chunks_native_with, rank_files_native, rank_files_native_with,
    rank_projects_native, ranked_chunk_result_json, related_chunk_result_json,
    search_files_response_json, search_projects_response_json, RankOptions,
};
use crate::related::{
    apply_chunk_relation_decision, build_context_pack_native, indexed_chunk_by_id,
    indexed_doc_chunks_by_path, list_chunk_relation_feedback, normalize_relation_quality_label,
    related_chunks_native, relation_feedback_row_json, set_chunk_relation_quality,
    source_chunk_json, ContextPackOptions,
};
use crate::scan::{plan_scoped_refresh, IndexScope};
use crate::util::{
    arg_value, normalize_path, now_ts, pid_is_alive, run_shell_capture, tail_file_lines,
    truncate_text_chars,
};

pub(crate) fn api_pid_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.pid")
}

pub(crate) fn api_port_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.port")
}

pub(crate) fn api_log_path(cwd: &Path) -> PathBuf {
    data_dir(cwd).join("api.log")
}

pub(crate) fn daemon_default_host() -> String {
    env::var("RETRIVIO_API_HOST")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

pub(crate) fn daemon_default_port() -> u16 {
    env::var("RETRIVIO_API_PORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(8765)
}

pub(crate) fn daemon_default_timeout() -> u64 {
    env::var("RETRIVIO_API_START_TIMEOUT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(8)
        .max(1)
}

pub(crate) fn read_saved_api_port(cwd: &Path) -> Option<u16> {
    let path = api_port_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u16>().ok()
}

pub(crate) fn read_saved_api_pid(cwd: &Path) -> Option<u32> {
    let path = api_pid_path(cwd);
    let raw = fs::read_to_string(path).ok()?;
    raw.trim().parse::<u32>().ok()
}

pub(crate) fn persist_api_runtime_state(cwd: &Path, pid: u32, port: u16) -> Result<(), String> {
    let data = data_dir(cwd);
    fs::create_dir_all(&data)
        .map_err(|e| format!("failed creating data dir '{}': {}", data.display(), e))?;
    fs::write(api_pid_path(cwd), format!("{}\n", pid))
        .map_err(|e| format!("failed writing api pid file: {}", e))?;
    fs::write(api_port_path(cwd), format!("{}\n", port))
        .map_err(|e| format!("failed writing api port file: {}", e))?;
    Ok(())
}

pub(crate) fn clear_api_runtime_state(cwd: &Path) {
    let _ = fs::remove_file(api_pid_path(cwd));
    let _ = fs::remove_file(api_port_path(cwd));
}

pub(crate) fn api_health_host_port(host: &str, port: u16) -> bool {
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

pub(crate) fn port_available(host: &str, port: u16) -> bool {
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

pub(crate) fn find_free_port(host: &str, start: u16, span: usize) -> Option<u16> {
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

pub(crate) fn spawn_api_daemon(cwd: &Path, host: &str, port: u16) -> Result<u32, String> {
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

pub(crate) fn run_daemon_cmd(args: &[OsString]) {
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

pub(crate) fn run_api_cmd(args: &[OsString]) {
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
pub(crate) struct ApiRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    query: HashMap<String, String>,
    body: Vec<u8>,
}

pub(crate) fn serve_api_native(host: &str, port: u16) -> Result<(), String> {
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

pub(crate) fn handle_api_connection(mut stream: TcpStream) -> Result<(), String> {
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

pub(crate) fn parse_http_request(stream: &mut TcpStream) -> Result<Option<ApiRequest>, String> {
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

pub(crate) fn parse_query_params(raw: &str) -> HashMap<String, String> {
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

pub(crate) fn hex_val(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some((c as u8) - b'0'),
        'a'..='f' => Some((c as u8) - b'a' + 10),
        'A'..='F' => Some((c as u8) - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn url_decode_component(raw: &str) -> String {
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

pub(crate) fn parse_limit(raw: Option<&String>, default: usize, max_limit: usize) -> usize {
    let parsed = raw.and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
    parsed.max(1).min(max_limit)
}

/// `since_days` query parameter: a positive finite number of days, otherwise no filter.
pub(crate) fn parse_since_days(raw: Option<&String>) -> Option<f64> {
    raw.and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|d| d.is_finite() && *d > 0.0)
}

pub(crate) fn parse_bool_flag(raw: Option<&String>) -> bool {
    let Some(v) = raw else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(crate) static API_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn api_trace_enabled() -> bool {
    *API_TRACE_ENABLED.get_or_init(|| {
        let raw = env::var("RETRIVIO_API_TRACE").unwrap_or_default();
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(crate) fn handle_search_pick_request(req: &ApiRequest) -> (u16, String) {
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
    let conn = match open_db_read_only(&db_path(&cwd)) {
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

pub(crate) fn payload_paths(body: &Value) -> Vec<String> {
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

pub(crate) fn payload_i64(body: &Value, key: &str) -> i64 {
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

pub(crate) fn payload_string(body: &Value, key: &str) -> String {
    body.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

pub(crate) fn unique_valid_dirs(paths: &[String]) -> Vec<PathBuf> {
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

pub(crate) fn stats_payload_json(stats: &IndexStats) -> Value {
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
        "files_selected": stats.files_selected,
        "files_unchanged": stats.files_unchanged,
        "files_rechunked": stats.files_rechunked,
        "chunks_embedded": stats.chunks_embedded,
        "chunks_reused": stats.chunks_reused,
        "chunks_deleted": stats.chunks_deleted,
        "files_unreadable": stats.files_unreadable,
        "projects_incomplete": stats.projects_incomplete,
        "files_evicted_by_cap": stats.files_evicted_by_cap,
        "files_truncated_by_cap": stats.files_truncated_by_cap,
        "lance_repaired": stats.lance_repaired,
        "lance_orphans_removed": stats.lance_orphans_removed,
        "documents_extracted": stats.documents_extracted,
        "documents_failed": stats.documents_failed,
        "projects_failed": stats.projects_failed,
        "failures": stats.failures,
        "stopped": stats.stopped,
        "lance_error": stats.lance_error,
        "retrieval_backend": stats.retrieval_backend,
        "retrieval_synced_chunks": stats.retrieval_synced_chunks,
        "retrieval_error": stats.retrieval_error,
    })
}

pub(crate) fn path_basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

pub(crate) fn path_in_tracked_roots(path: &str, tracked_roots: &[PathBuf]) -> bool {
    let target = normalize_path(path);
    tracked_roots
        .iter()
        .any(|root| target == *root || target.starts_with(root))
}

pub(crate) fn send_http_response(
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
        .and_then(|_| stream.write_all(body))
        .map_err(|e| format!("failed writing HTTP response: {}", e))
}

pub(crate) fn send_http_json(
    stream: &mut TcpStream,
    status: u16,
    payload: &Value,
) -> Result<(), String> {
    let body =
        serde_json::to_vec(payload).map_err(|e| format!("failed serializing JSON: {}", e))?;
    send_http_response(stream, status, "application/json", &body)
}

pub(crate) fn parse_json_body(body: &[u8]) -> Value {
    if body.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_slice::<Value>(body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()))
}

pub(crate) fn handle_api_request(req: ApiRequest) -> (u16, Value) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let since_days = parse_since_days(req.query.get("since_days"));
            if view == "files" {
                let opts = RankOptions {
                    since_days,
                    include_superseded: parse_bool_flag(req.query.get("include_superseded")),
                    min_raw_similarity: cfg.search_min_abs_score,
                    ..RankOptions::default()
                };
                let rows = match rank_files_native_with(&conn, &cfg, &q, limit, opts) {
                    Ok(v) => v,
                    Err(e) => return (503, serde_json::json!({"error": e})),
                };
                return (200, search_files_response_json(&q, &rows, search_started));
            }
            let rows = match rank_projects_native(&conn, &cfg, &q, limit) {
                Ok(v) => v,
                Err(e) => return (503, serde_json::json!({"error": e})),
            };
            return (
                200,
                search_projects_response_json(&q, &rows, search_started),
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
            let since_days = parse_since_days(req.query.get("since_days"));
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            if let Err(e) = ensure_native_embed_backend(&cfg, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            if let Err(e) = ensure_retrieval_backend_ready(&cfg, true, "api chunks/search") {
                return (503, serde_json::json!({"error": e}));
            }
            let conn = match open_db_read_only(&db_path(&cwd)) {
                Ok(v) => v,
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let rows = match rank_chunks_native_with(&conn, &cfg, &q, limit, since_days) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let conn = match open_db_read_only(&db_path(&cwd)) {
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
            let writer = match WriterLock::try_acquire(&data_dir(&cwd)) {
                Ok(v) => v,
                Err(e) if is_index_busy_error(&e) => {
                    return (409, serde_json::json!({"error": e}));
                }
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let result = if raw_paths.is_empty() {
                run_native_index(
                    &cwd,
                    &cfg,
                    &writer,
                    IndexRunOptions {
                        scope: IndexScope::AllRoots,
                        force_all: true,
                        force_paths: HashSet::new(),
                        remove_missing: true,
                        reason: "api refresh",
                    },
                    false,
                )
            } else {
                let dirs = unique_valid_dirs(&raw_paths);
                if dirs.is_empty() {
                    return (
                        400,
                        serde_json::json!({"error": "No valid directory paths provided."}),
                    );
                }
                let conn = match open_db_writer(&db_path(&cwd), &writer) {
                    Ok(v) => v,
                    Err(e) => return (500, serde_json::json!({"error": e})),
                };
                let scope = match plan_scoped_refresh(&conn, &cfg, &dirs) {
                    Ok(v) => v,
                    Err(e) => return (400, serde_json::json!({"error": e})),
                };
                drop(conn);
                run_native_index(
                    &cwd,
                    &cfg,
                    &writer,
                    IndexRunOptions {
                        scope,
                        force_all: true,
                        force_paths: HashSet::new(),
                        remove_missing: false,
                        reason: "api refresh",
                    },
                    false,
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
            // A tracked-root mutation is a write: it takes the writer lock like every other.
            let writer = match WriterLock::try_acquire(&data_dir(&cwd)) {
                Ok(v) => v,
                Err(e) if is_index_busy_error(&e) => {
                    return (409, serde_json::json!({"error": e}));
                }
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            for root in roots {
                if let Err(e) = ensure_tracked_root(&dbp, &root, now_ts()) {
                    return (500, serde_json::json!({"error": e}));
                }
            }
            drop(writer);
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
            let writer = match WriterLock::try_acquire(&data_dir(&cwd)) {
                Ok(v) => v,
                Err(e) if is_index_busy_error(&e) => {
                    return (409, serde_json::json!({"error": e}));
                }
                Err(e) => return (500, serde_json::json!({"error": e})),
            };
            let mut removed: i64 = 0;
            for raw in paths {
                removed += remove_tracked_root(&dbp, &normalize_path(&raw)).unwrap_or(0);
            }
            drop(writer);
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
