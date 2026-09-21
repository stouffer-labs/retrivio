//! Small shared helpers: environment and prompt reading, path normalisation, shell escaping, time and byte formatting, vector and token primitives.

use std::collections::HashSet;
use std::ffi::OsString;
use std::io::{IsTerminal, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, process};

use crate::config_tui::run_stty_capture;

pub(crate) fn strip_terminal_control_sequences(raw: &str) -> String {
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
                for next in chars.by_ref() {
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

pub(crate) fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|v| {
        let t = v.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    })
}

pub(crate) fn bool_env(name: &str, default: bool) -> bool {
    non_empty_env(name)
        .map(|v| {
            let n = v.to_ascii_lowercase();
            matches!(n.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(default)
}

pub(crate) fn command_available(cmd: &str) -> bool {
    if cmd.contains('/') {
        return is_executable_file(Path::new(cmd));
    }
    command_exists(cmd)
}

pub(crate) fn prompt_line_to(prompt: &str, stderr_prompt: bool) -> Result<String, String> {
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

pub(crate) fn prompt_line(prompt: &str) -> Result<String, String> {
    prompt_line_to(prompt, false)
}

/// Prompt for yes/no with a default. Shows `[Y/n]` or `[y/N]`.
pub(crate) fn prompt_yes_no(prompt: &str, default_yes: bool) -> Result<bool, String> {
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

pub(crate) fn tty_ui_available() -> bool {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return false;
    }
    if !command_exists("stty") {
        return false;
    }
    run_stty_capture(["-g"]).is_ok()
}

pub(crate) fn display_path_compact(path: &str) -> String {
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

pub(crate) fn open_url_in_default_browser(url: &str) -> Result<(), String> {
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

#[derive(Debug)]
pub(crate) struct ShellCommandOutput {
    pub(crate) exit_code: i32,
    pub(crate) stdout: String,
}

pub(crate) fn run_shell_capture(command: &str) -> Result<ShellCommandOutput, String> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg(command)
        .output()
        .map_err(|e| format!("failed running shell command '{}': {}", command, e))?;
    Ok(ShellCommandOutput {
        exit_code: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
    })
}

pub(crate) fn pid_is_alive(pid: u32) -> bool {
    let out = run_shell_capture(&format!("kill -0 {} >/dev/null 2>&1", pid));
    matches!(out, Ok(out) if out.exit_code == 0)
}

pub(crate) fn resolve_command_path(name: &str) -> Option<PathBuf> {
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

/// Human-readable byte count: 512 B, 3.4 KB, 1.52 GB.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} B", bytes)
    } else if value >= 100.0 {
        format!("{:.0} {}", value, UNITS[unit])
    } else if value >= 10.0 {
        format!("{:.1} {}", value, UNITS[unit])
    } else {
        format!("{:.2} {}", value, UNITS[unit])
    }
}

pub(crate) fn chrono_like_now() -> String {
    let out = run_shell_capture("date '+%Y-%m-%d %H:%M:%S'");
    match out {
        Ok(v) if v.exit_code == 0 && !v.stdout.trim().is_empty() => v.stdout.trim().to_string(),
        _ => "now".to_string(),
    }
}

pub(crate) fn tail_file_lines(path: &Path, lines: usize) -> Result<(), String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed reading log '{}': {}", path.display(), e))?;
    let items: Vec<&str> = raw.lines().collect();
    let start = items.len().saturating_sub(lines.max(1));
    for line in &items[start..] {
        println!("{}", line);
    }
    Ok(())
}

pub(crate) fn format_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

pub(crate) fn truncate_text_chars(text: &str, max_chars: usize) -> (String, bool, usize) {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return (text.to_string(), false, total_chars);
    }
    let clipped: String = text.chars().take(max_chars).collect();
    (clipped, true, total_chars)
}

pub(crate) fn cosine_raw(a: &[f32], b: &[f32], anorm: f64, bnorm: f64) -> f64 {
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

pub(crate) fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

pub(crate) fn arg_value(args: &[OsString], index: usize, flag: &str) -> String {
    let Some(v) = args.get(index) else {
        eprintln!("error: {} expects a value", flag);
        process::exit(2);
    };
    v.to_string_lossy().to_string()
}

pub(crate) fn normalize_path(raw: &str) -> PathBuf {
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

pub(crate) fn normalize_lexical(path: &Path) -> PathBuf {
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

pub(crate) fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub(crate) fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn word_tokens(text: &str) -> Vec<String> {
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

pub(crate) fn is_under_any(path: &Path, candidates: &HashSet<PathBuf>) -> bool {
    if candidates.is_empty() {
        return false;
    }
    path_is_under_any(&normalize_path(&path.to_string_lossy()), candidates)
}

/// [`is_under_any`] for a path that is already absolute and normalised (no canonicalise call).
pub(crate) fn path_is_under_any(resolved: &Path, candidates: &HashSet<PathBuf>) -> bool {
    candidates
        .iter()
        .any(|candidate| resolved == candidate || resolved.starts_with(candidate))
}

pub(crate) fn f32_blob(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub(crate) fn vector_norm(vector: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for v in vector {
        let f = *v as f64;
        sum += f * f;
    }
    sum.sqrt()
}

pub(crate) fn metadata_mtime(meta: &fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Modification time as whole nanoseconds since the epoch (0 when unavailable), the exact
/// value the scan signature hashes.
pub(crate) fn metadata_mtime_ns(meta: &fs::Metadata) -> i128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
}

pub(crate) fn file_mtime(path: &Path) -> Option<f64> {
    let meta = fs::metadata(path).ok()?;
    Some(metadata_mtime(&meta))
}

/// Minimal POSIX-ish shell tokenizer: handles single/double quotes and backslash escapes.
/// Returns None on unbalanced quotes.
pub(crate) fn shell_split(input: &str) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaping = false;
    let mut has_token = false;
    for ch in input.chars() {
        if escaping {
            current.push(ch);
            escaping = false;
            has_token = true;
            continue;
        }
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else if q == '"' && ch == '\\' {
                escaping = true;
            } else {
                current.push(ch);
            }
            continue;
        }
        match ch {
            '\\' => {
                escaping = true;
                has_token = true;
            }
            '\'' | '"' => {
                quote = Some(ch);
                has_token = true;
            }
            c if c.is_whitespace() => {
                if has_token {
                    out.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            other => {
                current.push(other);
                has_token = true;
            }
        }
    }
    if quote.is_some() || escaping {
        return None;
    }
    if has_token {
        out.push(current);
    }
    Some(out)
}

pub(crate) fn non_empty_string(v: &str) -> Option<String> {
    let t = v.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub(crate) fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return lines.join("\n");
    }
    let start = lines.len() - max_lines;
    lines[start..].join("\n")
}

pub(crate) fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

pub(crate) fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-lc")
        .arg(format!("command -v {} >/dev/null 2>&1", shell_escape(name)))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub(crate) fn is_executable_file(path: &Path) -> bool {
    let meta = match fs::metadata(path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        (meta.permissions().mode() & 0o111) != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    env::var("HOME").ok().map(PathBuf::from)
}

pub(crate) fn expand_tilde<S: AsRef<str>>(s: S) -> PathBuf {
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

pub(crate) fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\"'\"'");
    format!("'{}'", escaped)
}
