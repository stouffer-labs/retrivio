//! The interactive config editor: raw-mode terminal handling, key reading, terminal size probes and the settings menu.

use std::env;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::autotune::{autotune_recommendation, AutotuneOutcome};
use crate::config::{
    config_enum_options, config_path, config_rows, config_set_value, config_value_string, db_path,
    load_config_values, write_config_file, ConfigValues,
};
use crate::db::{ensure_db_schema, open_db_rw, refresh_reembed_requirement_for_config_change};
use crate::util::{command_exists, prompt_line, tty_ui_available};

#[derive(Clone, Debug)]
pub(crate) enum ConfigTuiMode {
    Navigate,
    Edit { key: String, buffer: String },
}

#[derive(Clone, Debug)]
pub(crate) struct ConfigTuiState {
    selected: usize,
    scroll: usize,
    dirty: bool,
    discard_armed: bool,
    status: String,
    mode: ConfigTuiMode,
}

pub(crate) enum ConfigKey {
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

pub(crate) struct ConfigTuiGuard {
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

pub(crate) fn enter_config_tui_mode() -> Result<ConfigTuiGuard, String> {
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

pub(crate) fn args_slice<const N: usize>(arr: [&str; N]) -> [&str; N] {
    arr
}

pub(crate) fn run_stty<const N: usize>(args: [&str; N]) -> Result<(), String> {
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

pub(crate) fn run_stty_capture<const N: usize>(args: [&str; N]) -> Result<String, String> {
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

pub(crate) fn read_config_key() -> Result<Option<ConfigKey>, String> {
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

pub(crate) fn terminal_size_fallback() -> (usize, usize) {
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

pub(crate) fn terminal_size_ioctl() -> Option<(usize, usize)> {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        // Try stdout first, then /dev/tty
        let mut ret = libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws);
        if ret != 0 || ws.ws_col == 0 {
            let tty_fd = libc::open(c"/dev/tty".as_ptr(), libc::O_RDONLY);
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
pub(crate) fn terminal_size_cursor_probe() -> Option<(usize, usize)> {
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

pub(crate) fn terminal_size_stty() -> (usize, usize) {
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

pub(crate) fn clipped(s: &str, max_width: usize) -> String {
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

pub(crate) fn config_rows_count() -> usize {
    config_rows().len()
}

pub(crate) fn cycle_enum_setting(
    cfg: &mut ConfigValues,
    key: &str,
    direction: i32,
) -> Result<bool, String> {
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

pub(crate) fn draw_config_tui(
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

pub(crate) fn run_config_tui(
    cwd: &Path,
    cfg_path: &Path,
    working: &mut ConfigValues,
) -> Result<bool, String> {
    let _guard = enter_config_tui_mode()?;
    let mut stdout = std::io::stdout();

    // Probe terminal size — cursor probe is most reliable in raw mode
    let term_size = terminal_size_cursor_probe().unwrap_or_else(terminal_size_stty);

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

pub(crate) fn select_config_menu_key(cfg: &ConfigValues) -> Result<Option<String>, String> {
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

pub(crate) fn select_enum_option(key: &str, current: &str) -> Result<Option<String>, String> {
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

pub(crate) fn edit_config_key_interactive(cfg: &mut ConfigValues, key: &str) -> Result<(), String> {
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

pub(crate) fn run_config_edit_legacy(cwd: &Path) -> Result<(), String> {
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

pub(crate) fn run_config_edit(cwd: &Path) -> Result<(), String> {
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
