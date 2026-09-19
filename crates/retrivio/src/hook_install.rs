//! `retrivio hook` and `retrivio service`: install/uninstall/status for CLI hooks and the launchd watcher.
//! See docs/superpowers/specs/2026-09-19-proactive-recall-design.md §6.
//!
//! The JSON and plist work is done by pure functions over `serde_json::Value` / strings so it can
//! be unit tested; the thin I/O wrappers further down handle files, prompts and `launchctl`.
//!
//! Environment: `RETRIVIO_HOME`, when set, replaces the real home directory for every path this
//! module touches (`~/.claude`, `~/.codex`, `~/Library/LaunchAgents`, `~/.retrivio/*.log`). It
//! exists so the commands can be exercised against a scratch directory.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

const LAUNCHD_LABEL: &str = "com.stouffer-labs.retrivio.watch";
const HOOK_TIMEOUT_SECS: u64 = 5;
const SESSION_START_MATCHER: &str = "compact|clear";
const CODEX_DESCRIPTION: &str = "Retrivio proactive recall hooks";
const CODEX_TRUST_REMINDER: &str = "Codex trusts the exact hook definition (hash-based). Run /hooks in Codex and trust the retrivio entries after installation AND after any change to them (for example a new binary path).";
const WATCH_INTERVAL_SECS: &str = "300";
const LAUNCHD_THROTTLE_SECS: u32 = 60;
const BACKUP_SUFFIX: &str = ".bak-retrivio";
const LAUNCHD_PATH_ENV: &str = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin";

/// PATH for the background service: the standard directories plus every directory on the
/// installing shell's PATH (launchd starts agents with a minimal PATH, which hides credential
/// helpers such as isengardcli or aws-sso tooling and makes the watcher fail on startup).
fn service_path_env() -> String {
    let mut parts: Vec<String> = LAUNCHD_PATH_ENV.split(':').map(|s| s.to_string()).collect();
    if let Ok(cur) = std::env::var("PATH") {
        for seg in cur.split(':') {
            let seg = seg.trim();
            // Only stable, absolute, existing directories: no relative entries, no temp/shim dirs
            // that vanish, nothing that could hijack a bare helper name.
            if seg.starts_with('/')
                && !seg.starts_with("/tmp")
                && !seg.starts_with("/private/tmp")
                && !seg.starts_with("/var/folders")
                && Path::new(seg).is_dir()
                && !parts.iter().any(|p| p == seg)
            {
                parts.push(seg.to_string());
            }
        }
    }
    parts.join(":")
}
const SYSTEMD_UNIT_NAME: &str = "retrivio-watch.service";
const TAIL_LINES: usize = 5;

const HOOK_USAGE: &str = "usage: retrivio hook <install|uninstall|status> [--claude] [--codex] [--yes] [--bin <path>]

  install     add the retrivio recall hooks (UserPromptSubmit + SessionStart) to each detected CLI
  uninstall   remove only the retrivio entries, leaving every other hook untouched
  status      report detection, install state, referenced binary and the recall log tail

  --claude / --codex   limit to one CLI (default: every detected CLI)
  --yes, -y            do not prompt before writing
  --bin <path>         binary to reference from the hooks (default: this executable)

Files: ~/.claude/settings.json and ~/.codex/hooks.json (a .bak-retrivio copy is taken before writing).
RETRIVIO_HOME overrides the home directory used to locate them.
";

const SERVICE_USAGE: &str = "usage: retrivio service <install|uninstall|status|run> [--bin <path>]

  install     write ~/Library/LaunchAgents/com.stouffer-labs.retrivio.watch.plist and load it (launchctl bootstrap)
  uninstall   unload it (launchctl bootout) and remove the plist
  status      launchd state, pid, last exit status, plist path and the watch log tail

  --bin <path>   binary the agent runs (default: this executable)

The agent runs `<bin> service run`, which supervises `watch --quiet --interval 300` with exponential backoff, and logs to ~/.retrivio/watch.log.
On Linux the equivalent systemd user unit is printed instead of being installed.
";

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cli {
    Claude,
    Codex,
}

impl Cli {
    const ALL: [Cli; 2] = [Cli::Claude, Cli::Codex];

    fn display(self) -> &'static str {
        match self {
            Cli::Claude => "Claude Code",
            Cli::Codex => "Codex",
        }
    }

    fn dir(self, home: &Path) -> PathBuf {
        match self {
            Cli::Claude => home.join(".claude"),
            Cli::Codex => home.join(".codex"),
        }
    }

    fn hooks_file(self, home: &Path) -> PathBuf {
        match self {
            Cli::Claude => self.dir(home).join("settings.json"),
            Cli::Codex => self.dir(home).join("hooks.json"),
        }
    }

    fn detected(self, home: &Path) -> bool {
        self.dir(home).is_dir()
    }

    fn merge(self, doc: &mut Value, bin: &str) -> Result<ChangeKind, String> {
        match self {
            Cli::Claude => merge_claude_hooks(doc, bin),
            Cli::Codex => merge_codex_hooks(doc, bin),
        }
    }
}

/// How the command is stored. Claude Code supports an exec form (`command` + `args`, spawned
/// without a shell); Codex only has the shell-string form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EntryForm {
    Exec,
    Shell,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChangeKind {
    Installed,
    Updated,
    Unchanged,
}

impl ChangeKind {
    fn combine(self, other: ChangeKind) -> ChangeKind {
        match (self, other) {
            (ChangeKind::Unchanged, ChangeKind::Unchanged) => ChangeKind::Unchanged,
            (ChangeKind::Installed, ChangeKind::Installed) => ChangeKind::Installed,
            _ => ChangeKind::Updated,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ChangeKind::Installed => "installed",
            ChangeKind::Updated => "updated",
            ChangeKind::Unchanged => "already installed",
        }
    }
}

/// The two hooks we own. Both files (Claude `settings.json`, Codex `hooks.json`) use the same
/// `hooks.<Event>[] -> {matcher?, hooks[]}` layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HookKind {
    /// `UserPromptSubmit`: `<bin> recall`
    Prompt,
    /// `SessionStart` (matcher `compact|clear`): `<bin> recall --reset-session`
    Reset,
}

impl HookKind {
    const ALL: [HookKind; 2] = [HookKind::Prompt, HookKind::Reset];

    fn event(self) -> &'static str {
        match self {
            HookKind::Prompt => "UserPromptSubmit",
            HookKind::Reset => "SessionStart",
        }
    }

    fn matcher(self) -> Option<&'static str> {
        match self {
            HookKind::Prompt => None,
            HookKind::Reset => Some(SESSION_START_MATCHER),
        }
    }

    fn args(self) -> &'static [&'static str] {
        match self {
            HookKind::Prompt => &["recall"],
            HookKind::Reset => &["recall", "--reset-session"],
        }
    }

    /// Shell-string form used by Codex: `<bin> recall [--reset-session]`.
    fn shell_command(self, bin: &str) -> String {
        format!("{} {}", shell_quote(bin), self.args().join(" "))
    }

    fn entry(self, bin: &str, form: EntryForm) -> Value {
        let mut m = Map::new();
        m.insert("type".to_string(), json!("command"));
        match form {
            EntryForm::Exec => {
                m.insert("command".to_string(), json!(bin));
                m.insert("args".to_string(), json!(self.args()));
            }
            EntryForm::Shell => {
                m.insert("command".to_string(), json!(self.shell_command(bin)));
            }
        }
        m.insert("timeout".to_string(), json!(HOOK_TIMEOUT_SECS));
        if self == HookKind::Prompt {
            m.insert("statusMessage".to_string(), json!("retrivio recall"));
        }
        Value::Object(m)
    }
}

// ---------------------------------------------------------------------------
// Pure helpers over serde_json::Value
// ---------------------------------------------------------------------------

/// The binary we install is always named `retrivio` (bare, on PATH, or an absolute path).
fn is_retrivio_bin(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .map(|n| n == "retrivio")
        .unwrap_or(false)
}

/// Exactly the argument vectors [`HookKind::args`] generates.
fn is_recall_args(args: &[&str]) -> bool {
    HookKind::ALL.iter().any(|k| k.args() == args)
}

/// First token of a shell-form command (quotes removed) and the remaining text. Handles the
/// `'...'` form [`shell_quote`] writes (with `'\''` for an embedded quote), `"..."`, and bare
/// words. A quoted token must be followed by whitespace or the end.
fn split_first_token(cmd: &str) -> Option<(String, &str)> {
    let s = cmd.trim_start();
    let first = s.chars().next()?;
    match first {
        '\'' => {
            let mut out = String::new();
            let mut i = 1usize;
            loop {
                let close = s[i..].find('\'')? + i;
                out.push_str(&s[i..close]);
                if s[close..].starts_with("'\\''") {
                    out.push('\'');
                    i = close + 4;
                    continue;
                }
                let rest = &s[close + 1..];
                if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
                    return None;
                }
                return Some((out, rest));
            }
        }
        '"' => {
            let close = s[1..].find('"')? + 1;
            let rest = &s[close + 1..];
            if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
                return None;
            }
            Some((s[1..close].to_string(), rest))
        }
        _ => {
            let end = s.find(char::is_whitespace).unwrap_or(s.len());
            Some((s[..end].to_string(), &s[end..]))
        }
    }
}

/// Shell form of ours: `<bin> recall` or `<bin> recall --reset-session` where the first token's
/// basename is `retrivio`. Returns the binary. Nothing else (a `retrivio-recall-proxy`, a
/// wrapper script mentioning recall, extra arguments) is ours.
fn parse_shell_recall(cmd: &str) -> Option<String> {
    let (bin, rest) = split_first_token(cmd)?;
    if !is_retrivio_bin(&bin) {
        return None;
    }
    let args: Vec<&str> = rest.split_whitespace().collect();
    is_recall_args(&args).then_some(bin)
}

/// Ours = exactly one of our generated forms. Exec form: `command` basename is `retrivio` and
/// `args` is `["recall"]` or `["recall", "--reset-session"]`. Shell form (no `args`): the
/// command string parses per [`parse_shell_recall`]. Legacy shell-form Claude entries are thus
/// still recognised for the upgrade to exec form.
fn is_retrivio_entry(entry: &Value) -> bool {
    let Some(cmd) = entry.get("command").and_then(Value::as_str) else {
        return false;
    };
    match entry.get("args") {
        Some(args) => {
            let Some(arr) = args.as_array() else {
                return false;
            };
            let strs: Vec<&str> = arr.iter().filter_map(Value::as_str).collect();
            strs.len() == arr.len() && is_retrivio_bin(cmd) && is_recall_args(&strs)
        }
        None => parse_shell_recall(cmd).is_some(),
    }
}

/// Binary path referenced by one of our entries: the `command` itself in exec form, or the
/// first token of the shell-form command string.
fn bin_from_entry(entry: &Value) -> Option<String> {
    let cmd = entry.get("command").and_then(Value::as_str)?;
    if entry.get("args").is_some() {
        return is_retrivio_entry(entry).then(|| cmd.to_string());
    }
    parse_shell_recall(cmd)
}

/// Quote a path for `sh -c` only when it needs it, so the common case stays readable.
fn shell_quote(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+@:,=~".contains(c));
    if plain {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn new_group(matcher: Option<&str>, entry: Value) -> Value {
    let mut g = Map::new();
    if let Some(m) = matcher {
        g.insert("matcher".to_string(), json!(m));
    }
    g.insert("hooks".to_string(), Value::Array(vec![entry]));
    Value::Object(g)
}

fn group_hooks_empty(group: &Value) -> bool {
    group
        .get("hooks")
        .and_then(Value::as_array)
        .map(|a| a.is_empty())
        .unwrap_or(false)
}

/// Ensure exactly one of our entries for `kind` exists under `hooks.<event>`, pointing at `bin`.
fn ensure_event_hook(
    hooks: &mut Map<String, Value>,
    kind: HookKind,
    bin: &str,
    form: EntryForm,
) -> Result<ChangeKind, String> {
    let event = kind.event();
    let desired = kind.entry(bin, form);
    let slot = hooks
        .entry(event.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let groups = slot
        .as_array_mut()
        .ok_or_else(|| format!("hooks.{} is not a JSON array", event))?;

    let mut found: Vec<(usize, usize)> = Vec::new();
    for (gi, group) in groups.iter().enumerate() {
        if let Some(entries) = group.get("hooks").and_then(Value::as_array) {
            for (ei, entry) in entries.iter().enumerate() {
                if is_retrivio_entry(entry) {
                    found.push((gi, ei));
                }
            }
        }
    }

    if found.is_empty() {
        groups.push(new_group(kind.matcher(), desired));
        return Ok(ChangeKind::Installed);
    }

    let mut changed = false;
    let mut touched: Vec<usize> = Vec::new();

    // Duplicates beyond the first are dropped; walk backwards so indices stay valid.
    for &(gi, ei) in found.iter().skip(1).rev() {
        if let Some(entries) = groups[gi].get_mut("hooks").and_then(Value::as_array_mut) {
            entries.remove(ei);
            touched.push(gi);
            changed = true;
        }
    }

    let (gi, ei) = found[0];
    let matcher_ok = match kind.matcher() {
        None => true,
        Some(m) => groups[gi].get("matcher").and_then(Value::as_str) == Some(m),
    };
    let only_ours = groups[gi]
        .get("hooks")
        .and_then(Value::as_array)
        .map(|a| a.len() == 1)
        .unwrap_or(false);

    if matcher_ok || only_ours {
        if !matcher_ok {
            if let Some(obj) = groups[gi].as_object_mut() {
                obj.insert(
                    "matcher".to_string(),
                    json!(kind.matcher().unwrap_or_default()),
                );
                changed = true;
            }
        }
        if let Some(entries) = groups[gi].get_mut("hooks").and_then(Value::as_array_mut) {
            if entries[ei] != desired {
                entries[ei] = desired;
                changed = true;
            }
        }
    } else {
        // Ours sits in a group shared with foreign hooks under a different matcher: move it
        // out rather than changing the matcher under the other hooks.
        if let Some(entries) = groups[gi].get_mut("hooks").and_then(Value::as_array_mut) {
            entries.remove(ei);
        }
        touched.push(gi);
        groups.push(new_group(kind.matcher(), desired));
        changed = true;
    }

    touched.sort_unstable();
    touched.dedup();
    for gi in touched.into_iter().rev() {
        if group_hooks_empty(&groups[gi]) {
            groups.remove(gi);
        }
    }

    Ok(if changed {
        ChangeKind::Updated
    } else {
        ChangeKind::Unchanged
    })
}

fn merge_hooks_document(doc: &mut Value, bin: &str, form: EntryForm) -> Result<ChangeKind, String> {
    let root = doc
        .as_object_mut()
        .ok_or_else(|| "document root is not a JSON object".to_string())?;
    let hooks = root
        .entry("hooks".to_string())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| "\"hooks\" is not a JSON object".to_string())?;
    let a = ensure_event_hook(hooks, HookKind::Prompt, bin, form)?;
    let b = ensure_event_hook(hooks, HookKind::Reset, bin, form)?;
    Ok(a.combine(b))
}

/// Merge our two hooks into a Claude Code `settings.json` document (`Null` = file absent).
/// Entries use the exec form (`command` = binary, `args` = ["recall", ...]); an older
/// shell-form entry of ours is converted in place.
pub fn merge_claude_hooks(settings: &mut Value, bin: &str) -> Result<ChangeKind, String> {
    if settings.is_null() {
        *settings = Value::Object(Map::new());
    }
    merge_hooks_document(settings, bin, EntryForm::Exec)
}

/// Merge our two hooks into a Codex `hooks.json` document (`Null` = file absent). Codex has no
/// exec form, so the entries are shell strings (`<bin> recall`).
pub fn merge_codex_hooks(doc: &mut Value, bin: &str) -> Result<ChangeKind, String> {
    if doc.is_null() {
        *doc = json!({ "description": CODEX_DESCRIPTION, "hooks": {} });
    }
    merge_hooks_document(doc, bin, EntryForm::Shell)
}

/// Remove every retrivio recall entry. Groups and event arrays we empty are dropped; the `hooks`
/// object itself and every foreign entry are kept. Returns whether anything changed.
pub fn remove_retrivio_hooks(doc: &mut Value) -> bool {
    let Some(hooks) = doc.get_mut("hooks").and_then(Value::as_object_mut) else {
        return false;
    };
    let mut changed = false;
    let mut emptied: Vec<String> = Vec::new();
    for (event, groups_value) in hooks.iter_mut() {
        let Some(groups) = groups_value.as_array_mut() else {
            continue;
        };
        let mut event_changed = false;
        let mut i = 0;
        while i < groups.len() {
            let mut removed_here = false;
            if let Some(entries) = groups[i].get_mut("hooks").and_then(Value::as_array_mut) {
                let before = entries.len();
                entries.retain(|e| !is_retrivio_entry(e));
                removed_here = entries.len() != before;
            }
            if removed_here {
                event_changed = true;
            }
            if removed_here && group_hooks_empty(&groups[i]) {
                groups.remove(i);
            } else {
                i += 1;
            }
        }
        if event_changed {
            changed = true;
            if groups.is_empty() {
                emptied.push(event.clone());
            }
        }
    }
    for event in emptied {
        hooks.remove(&event);
    }
    changed
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HookStatus {
    pub prompt_command: Option<String>,
    pub reset_command: Option<String>,
    pub prompt_bin: Option<String>,
    pub reset_bin: Option<String>,
}

impl HookStatus {
    pub fn installed(&self) -> bool {
        self.prompt_command.is_some() && self.reset_command.is_some()
    }

    pub fn partial(&self) -> bool {
        !self.installed() && (self.prompt_command.is_some() || self.reset_command.is_some())
    }

    /// Distinct binary paths referenced by our entries.
    pub fn bins(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for bin in [&self.prompt_bin, &self.reset_bin].into_iter().flatten() {
            if !out.contains(bin) {
                out.push(bin.clone());
            }
        }
        out
    }
}

/// Which of our hooks a document contains, with the stored command strings.
pub fn retrivio_hook_status(doc: &Value) -> HookStatus {
    let mut status = HookStatus::default();
    for kind in HookKind::ALL {
        let entry = doc
            .get("hooks")
            .and_then(|h| h.get(kind.event()))
            .and_then(Value::as_array)
            .and_then(|groups| {
                groups
                    .iter()
                    .filter_map(|g| g.get("hooks").and_then(Value::as_array))
                    .flatten()
                    .find(|e| is_retrivio_entry(e))
            });
        let command = entry
            .and_then(|e| e.get("command"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let bin = entry.and_then(bin_from_entry);
        match kind {
            HookKind::Prompt => {
                status.prompt_command = command;
                status.prompt_bin = bin;
            }
            HookKind::Reset => {
                status.reset_command = command;
                status.reset_bin = bin;
            }
        }
    }
    status
}

// ---------------------------------------------------------------------------
// Pure helpers: launchd / systemd rendering and launchctl parsing
// ---------------------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn watch_log_path(home: &Path) -> PathBuf {
    home.join(".retrivio").join("watch.log")
}

fn recall_log_path(home: &Path) -> PathBuf {
    home.join(".retrivio").join("recall.log")
}

fn plist_path(home: &Path) -> PathBuf {
    home.join("Library")
        .join("LaunchAgents")
        .join(format!("{}.plist", LAUNCHD_LABEL))
}

pub fn render_launchd_plist(bin: &str, home: &Path) -> String {
    let log = watch_log_path(home);
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>service</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>{throttle}</integer>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path}</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>
</dict>
</plist>
"#,
        label = LAUNCHD_LABEL,
        bin = xml_escape(bin),
        throttle = LAUNCHD_THROTTLE_SECS,
        log = xml_escape(&log.to_string_lossy()),
        path = xml_escape(&service_path_env()),
        home = xml_escape(&home.to_string_lossy()),
    )
}

pub fn render_systemd_unit(bin: &str, home: &Path) -> String {
    let log = watch_log_path(home);
    format!(
        "[Unit]\nDescription=Retrivio filesystem watcher\n\n[Service]\nExecStart={bin} service run\nRestart=always\nRestartSec=60\nEnvironment=PATH={path}\nStandardOutput=append:{log}\nStandardError=append:{log}\n\n[Install]\nWantedBy=default.target\n",
        bin = shell_quote(bin),
        path = service_path_env(),
        log = log.to_string_lossy(),
    )
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LaunchdSummary {
    pub state: Option<String>,
    pub pid: Option<String>,
    pub last_exit: Option<String>,
}

/// Pull `state`, `pid` and `last exit code` out of `launchctl print` output (first occurrence).
pub fn summarize_launchctl_print(out: &str) -> LaunchdSummary {
    let mut s = LaunchdSummary::default();
    for line in out.lines() {
        if let Some((k, v)) = line.trim().split_once('=') {
            let (k, v) = (k.trim(), v.trim());
            match k {
                "state" if s.state.is_none() => s.state = Some(v.to_string()),
                "pid" if s.pid.is_none() => s.pid = Some(v.to_string()),
                "last exit code" if s.last_exit.is_none() => s.last_exit = Some(v.to_string()),
                _ => {}
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

fn base_home() -> Result<PathBuf, String> {
    if let Ok(raw) = env::var("RETRIVIO_HOME") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }
    super::home_dir().ok_or_else(|| "cannot determine home directory (HOME unset)".to_string())
}

fn resolve_bin(explicit: Option<&str>) -> Result<PathBuf, String> {
    if let Some(raw) = explicit {
        let path = super::expand_tilde(raw);
        let canon = fs::canonicalize(&path).map_err(|e| format!("--bin {}: {}", raw, e))?;
        if !super::is_executable_file(&canon) {
            return Err(format!("--bin {}: not an executable file", canon.display()));
        }
        return Ok(canon);
    }
    if let Ok(exe) = env::current_exe() {
        if let Ok(canon) = fs::canonicalize(&exe) {
            if super::is_executable_file(&canon) {
                return Ok(canon);
            }
        }
    }
    if let Some(path) = super::resolve_retrivio_command_path_native() {
        return Ok(fs::canonicalize(&path).unwrap_or(path));
    }
    Err("cannot resolve the retrivio binary path; pass --bin <path>".to_string())
}

/// `Ok(None)` when the file does not exist; an empty file reads as `Null` (treated as absent).
fn read_json_file(path: &Path) -> Result<Option<Value>, String> {
    match fs::read_to_string(path) {
        Ok(text) => {
            if text.trim().is_empty() {
                return Ok(Some(Value::Null));
            }
            serde_json::from_str::<Value>(&text)
                .map(Some)
                .map_err(|e| format!("{}: invalid JSON: {}", path.display(), e))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {}", path.display(), e)),
    }
}

fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(BACKUP_SUFFIX);
    path.with_file_name(name)
}

/// Where a write to `path` really lands: the path itself when it is missing or a regular file;
/// the resolved target when `path` is a symlink to a regular file (written through, so the
/// link stays a link). A symlink to anything else, a dangling link, or a non-regular file is
/// refused.
fn resolve_write_target(path: &Path) -> Result<PathBuf, String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(path.to_path_buf()),
        Err(e) => Err(format!("{}: {}", path.display(), e)),
        Ok(md) if md.file_type().is_symlink() => {
            let target = fs::canonicalize(path)
                .map_err(|e| format!("{}: symlink target: {}", path.display(), e))?;
            let tmd = fs::metadata(&target).map_err(|e| format!("{}: {}", target.display(), e))?;
            if !tmd.is_file() {
                return Err(format!(
                    "{}: refusing to write through a symlink to a non-regular file ({})",
                    path.display(),
                    target.display()
                ));
            }
            Ok(target)
        }
        Ok(md) if md.is_file() => Ok(path.to_path_buf()),
        Ok(_) => Err(format!("{}: not a regular file", path.display())),
    }
}

/// Unpredictable hex suffix for temp files (time, pid and a counter through a 64-bit mixer).
fn random_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut z = nanos
        ^ (process::id() as u64).rotate_left(32)
        ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    format!("{:016x}", z)
}

/// Mode for a brand-new hooks file when there is nothing to preserve.
#[cfg(unix)]
const NEW_FILE_MODE: u32 = 0o600;

/// `create_new` open options with the given permission bits applied at creation time.
fn create_new_with_mode(path: &Path, mode: u32) -> std::io::Result<fs::File> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(mode);
    }
    #[cfg(not(unix))]
    let _ = mode;
    opts.open(path)
}

/// Create a fresh temp file next to `target` (`create_new`, random suffix, final permission
/// bits from the start; retried on a clash).
fn create_temp_next_to(target: &Path, parent: &Path, mode: u32) -> std::io::Result<(fs::File, PathBuf)> {
    let file_name = target
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "hooks.json".to_string());
    let mut last_err = None;
    for _ in 0..16 {
        let tmp = parent.join(format!(".{}.tmp-retrivio-{}", file_name, random_suffix()));
        match create_new_with_mode(&tmp, mode) {
            Ok(f) => return Ok((f, tmp)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last_err = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("could not create a temp file")))
}

/// Copy `target` to `bak` through a `create_new` handle. An existing backup that is a regular
/// file is replaced; a symlink or anything else at the backup path is refused (never followed).
fn write_backup(target: &Path, bak: &Path, mode: u32) -> Result<(), String> {
    match fs::symlink_metadata(bak) {
        Ok(md) if md.file_type().is_symlink() || !md.is_file() => {
            return Err(format!(
                "backup {}: exists and is not a regular file; refusing to overwrite",
                bak.display()
            ));
        }
        Ok(_) => fs::remove_file(bak).map_err(|e| format!("backup {}: {}", bak.display(), e))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("backup {}: {}", bak.display(), e)),
    }
    let copy = || -> std::io::Result<()> {
        let mut src = fs::File::open(target)?;
        let mut out = create_new_with_mode(bak, mode)?;
        std::io::copy(&mut src, &mut out)?;
        out.sync_all()
    };
    copy().map_err(|e| format!("backup {}: {}", bak.display(), e))
}

/// Pretty-print `value` (2-space indent, trailing newline) to `path` through a `create_new` temp
/// file in the target's directory, fsync of the file and of the directory, and a rename. A
/// symlinked `path` is written through to its target only when that is a regular file. The
/// first write to an existing file in a run copies it to `<file>.bak-retrivio` (returned); the
/// original permissions are preserved (so 0600 stays 0600).
fn write_json_atomic(
    path: &Path,
    value: &Value,
    backed_up: &mut bool,
) -> Result<Option<PathBuf>, String> {
    let target = resolve_write_target(path)?;
    let parent = target
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", target.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;

    let existing = fs::metadata(&target).ok();
    #[cfg(unix)]
    let mode = existing
        .as_ref()
        .map(|m| m.permissions().mode() & 0o777)
        .unwrap_or(NEW_FILE_MODE);
    #[cfg(not(unix))]
    let mode = 0o600;
    let mut backup = None;
    if existing.is_some() && !*backed_up {
        let bak = backup_path(&target);
        write_backup(&target, &bak, mode)?;
        *backed_up = true;
        backup = Some(bak);
    }

    let mut text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    text.push('\n');

    let (mut f, tmp) = create_temp_next_to(&target, parent, mode)
        .map_err(|e| format!("write {}: temp file: {}", target.display(), e))?;
    let write = || -> std::io::Result<()> {
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, &target)?;
        // Make the rename durable; best effort where the filesystem does not support it.
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    };
    write()
        .map_err(|e| {
            let _ = fs::remove_file(&tmp);
            format!("write {}: {}", target.display(), e)
        })
        .map(|()| backup)
}

/// Last `n` lines of a file, reading at most the final 64 KiB.
fn tail_lines(path: &Path, n: usize) -> std::io::Result<Vec<String>> {
    const MAX_BYTES: u64 = 64 * 1024;
    let mut f = fs::File::open(path)?;
    let len = f.metadata()?.len();
    let start = len.saturating_sub(MAX_BYTES);
    f.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<&str> = text.lines().collect();
    if start > 0 && !lines.is_empty() {
        lines.remove(0); // partial first line
    }
    let skip = lines.len().saturating_sub(n);
    Ok(lines[skip..].iter().map(|s| s.to_string()).collect())
}

fn print_log_tail(label: &str, path: &Path) {
    if !path.is_file() {
        println!("{} ({}): not present", label, path.display());
        return;
    }
    match tail_lines(path, TAIL_LINES) {
        Ok(lines) if lines.is_empty() => println!("{} ({}): empty", label, path.display()),
        Ok(lines) => {
            println!("{} ({}), last {} lines:", label, path.display(), lines.len());
            for l in lines {
                println!("  {}", l);
            }
        }
        Err(e) => println!("{} ({}): unreadable ({})", label, path.display(), e),
    }
}

fn interactive_terminal() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn confirm(auto_yes: bool, question: &str) -> bool {
    if auto_yes {
        return true;
    }
    super::prompt_yes_no(question, true).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// `retrivio hook`
// ---------------------------------------------------------------------------

struct HookOpts {
    sub: String,
    claude: bool,
    codex: bool,
    yes: bool,
    bin: Option<String>,
    help: bool,
}

fn parse_common(
    args: &[OsString],
    subs: &[&str],
    default_sub: &str,
    allow_cli_flags: bool,
) -> Result<HookOpts, String> {
    let mut opts = HookOpts {
        sub: String::new(),
        claude: false,
        codex: false,
        yes: false,
        bin: None,
        help: false,
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].to_string_lossy().to_string();
        match arg.as_str() {
            "-h" | "--help" => opts.help = true,
            "-y" | "--yes" => opts.yes = true,
            "--claude" if allow_cli_flags => opts.claude = true,
            "--codex" if allow_cli_flags => opts.codex = true,
            "--bin" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "--bin requires a path".to_string())?;
                opts.bin = Some(v.to_string_lossy().to_string());
            }
            other if other.starts_with("--bin=") => {
                opts.bin = Some(other["--bin=".len()..].to_string());
            }
            other if other.starts_with('-') => return Err(format!("unknown option '{}'", other)),
            other => {
                if !opts.sub.is_empty() {
                    return Err(format!("unexpected argument '{}'", other));
                }
                if !subs.contains(&other) {
                    return Err(format!(
                        "unknown subcommand '{}' (expected {})",
                        other,
                        subs.join("|")
                    ));
                }
                opts.sub = other.to_string();
            }
        }
        i += 1;
    }
    if opts.sub.is_empty() {
        opts.sub = default_sub.to_string();
    }
    Ok(opts)
}

fn selected_clis(opts: &HookOpts) -> Vec<Cli> {
    if !opts.claude && !opts.codex {
        return Cli::ALL.to_vec();
    }
    let mut out = Vec::new();
    if opts.claude {
        out.push(Cli::Claude);
    }
    if opts.codex {
        out.push(Cli::Codex);
    }
    out
}

pub fn run_hook_cmd(args: &[OsString]) {
    let opts = match parse_common(args, &["install", "uninstall", "status"], "status", true) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!();
            eprint!("{}", HOOK_USAGE);
            process::exit(2);
        }
    };
    if opts.help {
        print!("{}", HOOK_USAGE);
        return;
    }
    let result = match opts.sub.as_str() {
        "install" => hook_install(&opts),
        "uninstall" => hook_uninstall(&opts),
        _ => hook_status(&opts),
    };
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn require_yes_or_tty(opts: &HookOpts) {
    if !opts.yes && !interactive_terminal() {
        eprintln!("hint: not a terminal - use --yes to skip prompts");
        process::exit(2);
    }
}

fn hook_install(opts: &HookOpts) -> Result<(), String> {
    require_yes_or_tty(opts);
    let home = base_home()?;
    let bin = resolve_bin(opts.bin.as_deref())?;
    let bin_str = bin.to_string_lossy().to_string();
    println!("  bin: {}", bin_str);

    let mut failures = 0usize;
    let mut saw_codex = false;
    for cli in selected_clis(opts) {
        if !cli.detected(&home) {
            println!(
                "  {}: skipped (not detected: {} missing)",
                cli.display(),
                cli.dir(&home).display()
            );
            continue;
        }
        if cli == Cli::Codex {
            saw_codex = true;
        }
        let file = cli.hooks_file(&home);
        let outcome = (|| -> Result<(), String> {
            let mut doc = read_json_file(&file)?.unwrap_or(Value::Null);
            let change = cli.merge(&mut doc, &bin_str)?;
            if change == ChangeKind::Unchanged {
                println!("  {}: already installed ({})", cli.display(), file.display());
                return Ok(());
            }
            let question = match change {
                ChangeKind::Installed => {
                    format!("  install retrivio hooks into {}?", file.display())
                }
                _ => format!("  update retrivio hooks in {}?", file.display()),
            };
            if !confirm(opts.yes, &question) {
                println!("  {}: skipped", cli.display());
                return Ok(());
            }
            let mut backed_up = false;
            let bak = write_json_atomic(&file, &doc, &mut backed_up)?
                .map(|p| format!(", backup {}", p.display()))
                .unwrap_or_default();
            println!(
                "  {}: {} ({}{})",
                cli.display(),
                change.label(),
                file.display(),
                bak
            );
            Ok(())
        })();
        if let Err(e) = outcome {
            eprintln!("  error: {}: {}", cli.display(), e);
            failures += 1;
        }
    }
    if saw_codex {
        println!("  {}", CODEX_TRUST_REMINDER);
    }
    if failures > 0 {
        return Err(format!("{} CLI(s) failed", failures));
    }
    Ok(())
}

fn hook_uninstall(opts: &HookOpts) -> Result<(), String> {
    require_yes_or_tty(opts);
    let home = base_home()?;
    let mut failures = 0usize;
    for cli in selected_clis(opts) {
        if !cli.detected(&home) {
            println!(
                "  {}: skipped (not detected: {} missing)",
                cli.display(),
                cli.dir(&home).display()
            );
            continue;
        }
        let file = cli.hooks_file(&home);
        let outcome = (|| -> Result<(), String> {
            let mut doc = match read_json_file(&file)? {
                Some(v) if !v.is_null() => v,
                _ => {
                    println!(
                        "  {}: not installed ({} missing)",
                        cli.display(),
                        file.display()
                    );
                    return Ok(());
                }
            };
            if !remove_retrivio_hooks(&mut doc) {
                println!("  {}: not installed ({})", cli.display(), file.display());
                return Ok(());
            }
            let question = format!("  remove retrivio hooks from {}?", file.display());
            if !confirm(opts.yes, &question) {
                println!("  {}: skipped", cli.display());
                return Ok(());
            }
            let mut backed_up = false;
            let bak = write_json_atomic(&file, &doc, &mut backed_up)?
                .map(|p| format!(", backup {}", p.display()))
                .unwrap_or_default();
            println!("  {}: removed ({}{})", cli.display(), file.display(), bak);
            Ok(())
        })();
        if let Err(e) = outcome {
            eprintln!("  error: {}: {}", cli.display(), e);
            failures += 1;
        }
    }
    if failures > 0 {
        return Err(format!("{} CLI(s) failed", failures));
    }
    Ok(())
}

fn hook_status(opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    let current_bin = resolve_bin(opts.bin.as_deref()).ok();
    if let Some(b) = &current_bin {
        println!("current binary: {}", b.display());
    }
    for cli in selected_clis(opts) {
        println!("{}:", cli.display());
        let dir = cli.dir(&home);
        println!(
            "  detected: {} ({})",
            if cli.detected(&home) { "yes" } else { "no" },
            dir.display()
        );
        let file = cli.hooks_file(&home);
        println!(
            "  file: {} ({})",
            file.display(),
            if file.is_file() { "present" } else { "missing" }
        );
        match read_json_file(&file) {
            Ok(Some(doc)) if !doc.is_null() => {
                let status = retrivio_hook_status(&doc);
                let state = if status.installed() {
                    "yes (UserPromptSubmit + SessionStart)".to_string()
                } else if status.partial() {
                    format!(
                        "partial ({} present, {} missing)",
                        if status.prompt_command.is_some() { "UserPromptSubmit" } else { "SessionStart" },
                        if status.prompt_command.is_some() { "SessionStart" } else { "UserPromptSubmit" },
                    )
                } else {
                    "no".to_string()
                };
                println!("  installed: {}", state);
                for bin in status.bins() {
                    let path = Path::new(&bin);
                    let exists = path.is_file();
                    let executable = super::is_executable_file(path);
                    let matches = match (&current_bin, fs::canonicalize(path)) {
                        (Some(cur), Ok(canon)) => canon == *cur,
                        _ => false,
                    };
                    println!(
                        "  bin: {} ({}, {}, {})",
                        bin,
                        if exists { "exists" } else { "MISSING" },
                        if executable { "executable" } else { "not executable" },
                        if matches {
                            "matches current binary"
                        } else {
                            "differs from current binary"
                        }
                    );
                }
            }
            Ok(_) => println!("  installed: no"),
            Err(e) => println!("  installed: unknown ({})", e),
        }
        if cli == Cli::Codex {
            println!("  note: {}", CODEX_TRUST_REMINDER);
        }
    }
    print_log_tail("recall log", &recall_log_path(&home));
    Ok(())
}

// ---------------------------------------------------------------------------
// `retrivio service`
// ---------------------------------------------------------------------------

pub fn run_service_cmd(args: &[OsString]) {
    let opts = match parse_common(args, &["install", "uninstall", "status", "run"], "status", false) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!();
            eprint!("{}", SERVICE_USAGE);
            process::exit(2);
        }
    };
    if opts.help {
        print!("{}", SERVICE_USAGE);
        return;
    }
    let result = if opts.sub == "run" {
        service_run()
    } else if cfg!(target_os = "macos") {
        match opts.sub.as_str() {
            "install" => service_install(&opts),
            "uninstall" => service_uninstall(&opts),
            _ => service_status(&opts),
        }
    } else {
        service_non_macos(&opts)
    };
    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

fn current_uid() -> u32 {
    // SAFETY: getuid(2) has no preconditions and cannot fail.
    unsafe { libc::getuid() }
}

fn launchctl(args: &[&str]) -> Result<String, String> {
    let output = Command::new("launchctl")
        .args(args)
        .output()
        .map_err(|e| format!("launchctl {}: {}", args.join(" "), e))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(stdout)
    } else {
        let detail = if stderr.trim().is_empty() { stdout } else { stderr };
        Err(format!(
            "launchctl {} failed ({}): {}",
            args.join(" "),
            output.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string()),
            detail.trim()
        ))
    }
}

/// Foreground supervisor used by the launchd agent / systemd unit: runs `watch --quiet` and,
/// when it exits (expired credentials, backend down, fswatch crash), waits with exponential
/// backoff (60 s .. 15 min) and starts it again instead of letting launchd restart-storm.
static SERVICE_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn service_on_signal(_sig: libc::c_int) {
    SERVICE_STOP.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn service_run() -> Result<(), String> {
    use std::sync::atomic::Ordering;
    let bin = std::env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    // SAFETY: installing a minimal async-signal-safe handler that only stores a flag.
    unsafe {
        libc::signal(libc::SIGTERM, service_on_signal as usize);
        libc::signal(libc::SIGINT, service_on_signal as usize);
    }
    let mut backoff: u64 = 60;
    while !SERVICE_STOP.load(Ordering::SeqCst) {
        let started = std::time::Instant::now();
        eprintln!("[{}] retrivio service: starting watch", unix_now());
        let mut child = match std::process::Command::new(&bin)
            .args(["watch", "--quiet", "--interval", WATCH_INTERVAL_SECS])
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[{}] retrivio service: failed to start watch: {}; retry in {}s", unix_now(), e, backoff);
                service_sleep(backoff);
                backoff = (backoff * 2).min(900);
                continue;
            }
        };
        // Poll the child so a TERM/INT to the supervisor is forwarded and the child is reaped.
        let status = loop {
            if SERVICE_STOP.load(Ordering::SeqCst) {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!("[{}] retrivio service: stopped", unix_now());
                return Ok(());
            }
            match child.try_wait() {
                Ok(Some(st)) => break st,
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(500)),
                Err(e) => {
                    eprintln!("[{}] retrivio service: wait failed: {}", unix_now(), e);
                    let _ = child.kill();
                    let _ = child.wait();
                    break std::process::ExitStatus::default();
                }
            }
        };
        eprintln!("[{}] retrivio service: watch exited ({}); retry in {}s", unix_now(), status, backoff);
        if started.elapsed() > std::time::Duration::from_secs(600) {
            backoff = 60;
        }
        service_sleep(backoff);
        backoff = (backoff * 2).min(900);
    }
    Ok(())
}

/// Sleep in one-second slices so a stop signal interrupts the backoff.
fn service_sleep(secs: u64) {
    for _ in 0..secs {
        if SERVICE_STOP.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn service_install(opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    let bin = resolve_bin(opts.bin.as_deref())?;
    let plist = plist_path(&home);
    let log_dir = home.join(".retrivio");
    fs::create_dir_all(&log_dir).map_err(|e| format!("{}: {}", log_dir.display(), e))?;
    if let Some(parent) = plist.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {}", parent.display(), e))?;
    }
    let rendered = render_launchd_plist(&bin.to_string_lossy(), &home);
    fs::write(&plist, rendered).map_err(|e| format!("write {}: {}", plist.display(), e))?;
    println!("  wrote {}", plist.display());
    println!("  program: {} service run  (supervises `watch --quiet --interval {}` with backoff)", bin.display(), WATCH_INTERVAL_SECS);
    println!("  log: {}", watch_log_path(&home).display());

    let domain = format!("gui/{}", current_uid());
    let target = format!("{}/{}", domain, LAUNCHD_LABEL);
    let plist_str = plist.to_string_lossy().to_string();
    match launchctl(&["bootstrap", &domain, &plist_str]) {
        Ok(_) => {
            println!("  loaded {}", target);
            return Ok(());
        }
        Err(first) => {
            // Already loaded (EEXIST / "already bootstrapped" / I/O error 5): reload so the new
            // plist takes effect, then fall back to a kickstart if the reload is refused.
            let _ = launchctl(&["bootout", &target]);
            if launchctl(&["bootstrap", &domain, &plist_str]).is_ok() {
                println!("  reloaded {} (was already loaded)", target);
                return Ok(());
            }
            if launchctl(&["kickstart", "-k", &target]).is_ok() {
                println!("  restarted {} (already loaded; kickstart -k)", target);
                return Ok(());
            }
            Err(first)
        }
    }
}

fn service_uninstall(_opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    let target = format!("gui/{}/{}", current_uid(), LAUNCHD_LABEL);
    match launchctl(&["bootout", &target]) {
        Ok(_) => println!("  unloaded {}", target),
        Err(e) => {
            let lower = e.to_lowercase();
            if lower.contains("no such process")
                || lower.contains("could not find")
                || lower.contains("not find")
                || lower.contains("(3)")
                || lower.contains("(113)")
            {
                println!("  not loaded ({})", target);
            } else {
                eprintln!("  warning: {}", e);
            }
        }
    }
    let plist = plist_path(&home);
    if plist.is_file() {
        fs::remove_file(&plist).map_err(|e| format!("remove {}: {}", plist.display(), e))?;
        println!("  removed {}", plist.display());
    } else {
        println!("  plist not present ({})", plist.display());
    }
    Ok(())
}

fn service_status(_opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    let target = format!("gui/{}/{}", current_uid(), LAUNCHD_LABEL);
    println!("service {}:", LAUNCHD_LABEL);
    match launchctl(&["print", &target]) {
        Ok(out) => {
            let s = summarize_launchctl_print(&out);
            println!("  loaded: yes ({})", target);
            println!("  state: {}", s.state.as_deref().unwrap_or("unknown"));
            println!("  pid: {}", s.pid.as_deref().unwrap_or("-"));
            println!("  last exit status: {}", s.last_exit.as_deref().unwrap_or("-"));
        }
        Err(_) => println!("  loaded: no ({})", target),
    }
    let plist = plist_path(&home);
    println!(
        "  plist: {} ({})",
        plist.display(),
        if plist.is_file() { "present" } else { "missing" }
    );
    match find_on_path("fswatch", LAUNCHD_PATH_ENV) {
        Some(p) => println!("  fswatch: {} (on the agent PATH)", p.display()),
        None => println!(
            "  fswatch: not found on the agent PATH ({}); the watcher falls back to polling",
            LAUNCHD_PATH_ENV
        ),
    }
    let log = watch_log_path(&home);
    match fs::metadata(&log) {
        Ok(meta) => {
            println!(
                "  watch log: {} ({} bytes, modified {})",
                log.display(),
                meta.len(),
                meta.modified()
                    .ok()
                    .map(describe_mtime)
                    .unwrap_or_else(|| "unknown".to_string())
            );
            match tail_lines(&log, TAIL_LINES) {
                Ok(lines) if lines.is_empty() => println!("    (empty)"),
                Ok(lines) => {
                    println!("    last {} lines:", lines.len());
                    for l in lines {
                        println!("    {}", l);
                    }
                }
                Err(e) => println!("    unreadable ({})", e),
            }
        }
        Err(_) => println!("  watch log: {} (not present)", log.display()),
    }
    Ok(())
}

/// First executable named `name` in the colon-separated `path` (the launchd agent's PATH, not
/// the caller's, so the answer reflects what the service will actually see).
fn find_on_path(name: &str, path: &str) -> Option<PathBuf> {
    path.split(':')
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join(name))
        .find(|p| super::is_executable_file(p))
}

/// "N seconds/minutes/hours/days ago" plus the raw unix timestamp.
fn describe_mtime(t: std::time::SystemTime) -> String {
    let unix = t
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ago = std::time::SystemTime::now()
        .duration_since(t)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let human = if ago < 60 {
        format!("{}s ago", ago)
    } else if ago < 3600 {
        format!("{}m ago", ago / 60)
    } else if ago < 86_400 {
        format!("{}h ago", ago / 3600)
    } else {
        format!("{}d ago", ago / 86_400)
    };
    format!("{} (unix {})", human, unix)
}

fn service_non_macos(opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    let unit_path = home
        .join(".config")
        .join("systemd")
        .join("user")
        .join(SYSTEMD_UNIT_NAME);
    match opts.sub.as_str() {
        "install" => {
            let bin = resolve_bin(opts.bin.as_deref())?;
            println!(
                "launchd is macOS-only; save the following as {} then run:",
                unit_path.display()
            );
            println!("  systemctl --user daemon-reload && systemctl --user enable --now {}", SYSTEMD_UNIT_NAME);
            println!();
            print!("{}", render_systemd_unit(&bin.to_string_lossy(), &home));
        }
        "uninstall" => {
            println!("launchd is macOS-only; to remove the systemd user unit run:");
            println!("  systemctl --user disable --now {}", SYSTEMD_UNIT_NAME);
            println!("  rm {}", unit_path.display());
        }
        _ => {
            println!("launchd is macOS-only; check the systemd user unit with:");
            println!("  systemctl --user status {}", SYSTEMD_UNIT_NAME);
            println!(
                "  unit file: {} ({})",
                unit_path.display(),
                if unit_path.is_file() { "present" } else { "missing" }
            );
            print_log_tail("  watch log", &watch_log_path(&home));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BIN: &str = "/Users/me/.local/bin/retrivio";
    const BIN2: &str = "/opt/homebrew/bin/retrivio";

    fn is_retrivio_recall_command(cmd: &str) -> bool {
        parse_shell_recall(cmd).is_some()
    }

    fn bin_from_command(cmd: &str) -> Option<String> {
        parse_shell_recall(cmd)
    }

    /// Shaped like a real ~/.claude/settings.json with several unrelated hooks.
    fn fixture_settings() -> Value {
        json!({
            "model": "claude-fable-5-1",
            "effortLevel": "xhigh",
            "env": { "CLAUDE_CODE_SUBAGENT_MODEL": "claude-fable-5-1" },
            "permissions": { "allow": ["Bash(ls:*)"], "deny": ["mcp__slack__send"] },
            "statusLine": { "type": "command", "command": "~/.claude/hooks/statusline-wrapper.sh" },
            "hooks": {
                "Stop": [
                    { "hooks": [ { "type": "command", "command": "printf '\\a'" } ] }
                ],
                "PostToolUse": [
                    { "matcher": "*", "hooks": [
                        { "type": "command", "command": "/Users/me/.claude/hooks/context-budget.sh", "timeout": 15 }
                    ] }
                ],
                "PostCompact": [
                    { "hooks": [
                        { "type": "command", "command": "/Users/me/.claude/hooks/postcompact-generation.sh", "timeout": 15 }
                    ] }
                ],
                "PreCompact": [
                    { "hooks": [
                        { "type": "command", "command": "/Users/me/.claude/hooks/precompact-log.sh", "timeout": 10 }
                    ] }
                ],
                "SessionStart": [
                    { "matcher": "compact", "hooks": [
                        { "type": "command", "command": "/Users/me/.claude/hooks/sessionstart-resume.sh", "timeout": 20 }
                    ] }
                ]
            }
        })
    }

    fn groups<'a>(doc: &'a Value, event: &str) -> &'a Vec<Value> {
        doc["hooks"][event].as_array().expect("event array")
    }

    fn commands(doc: &Value, event: &str) -> Vec<String> {
        groups(doc, event)
            .iter()
            .flat_map(|g| g["hooks"].as_array().cloned().unwrap_or_default())
            .map(|e| e["command"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn merge_into_empty_object_installs_both_hooks() {
        let mut doc = json!({});
        let change = merge_claude_hooks(&mut doc, BIN).unwrap();
        assert_eq!(change, ChangeKind::Installed);
        let ups = groups(&doc, "UserPromptSubmit");
        assert_eq!(ups.len(), 1);
        assert!(ups[0].get("matcher").is_none());
        assert_eq!(
            ups[0]["hooks"][0],
            json!({ "type": "command", "command": BIN, "args": ["recall"], "timeout": 5, "statusMessage": "retrivio recall" })
        );
        let ss = groups(&doc, "SessionStart");
        assert_eq!(ss.len(), 1);
        assert_eq!(ss[0]["matcher"], json!("compact|clear"));
        assert_eq!(
            ss[0]["hooks"][0],
            json!({ "type": "command", "command": BIN, "args": ["recall", "--reset-session"], "timeout": 5 })
        );
        let st = retrivio_hook_status(&doc);
        assert!(st.installed());
        assert_eq!(st.bins(), vec![BIN.to_string()]);
    }

    #[test]
    fn merge_converts_old_shell_form_claude_entry_to_exec_form() {
        let mut doc = json!({
            "hooks": {
                "UserPromptSubmit": [
                    { "hooks": [ { "type": "command", "command": format!("{} recall", BIN), "timeout": 5, "statusMessage": "retrivio recall" } ] }
                ],
                "SessionStart": [
                    { "matcher": "compact|clear", "hooks": [ { "type": "command", "command": format!("{} recall --reset-session", BIN), "timeout": 5 } ] }
                ]
            }
        });
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Updated);
        assert_eq!(groups(&doc, "UserPromptSubmit").len(), 1);
        assert_eq!(groups(&doc, "SessionStart").len(), 1);
        assert_eq!(doc["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"], json!(BIN));
        assert_eq!(doc["hooks"]["UserPromptSubmit"][0]["hooks"][0]["args"], json!(["recall"]));
        assert_eq!(doc["hooks"]["SessionStart"][0]["hooks"][0]["args"], json!(["recall", "--reset-session"]));
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Unchanged);
    }

    #[test]
    fn exec_form_entries_are_recognised_and_yield_bin() {
        let exec = json!({ "type": "command", "command": BIN, "args": ["recall"], "timeout": 5 });
        assert!(is_retrivio_entry(&exec));
        assert_eq!(bin_from_entry(&exec).as_deref(), Some(BIN));
        let shell = json!({ "type": "command", "command": format!("{} recall", BIN) });
        assert!(is_retrivio_entry(&shell));
        assert_eq!(bin_from_entry(&shell).as_deref(), Some(BIN));
        // Same binary, unrelated subcommand: not ours.
        let other = json!({ "type": "command", "command": BIN, "args": ["search", "x"] });
        assert!(!is_retrivio_entry(&other));
        let other_shell = json!({ "type": "command", "command": format!("{} watch", BIN) });
        assert!(!is_retrivio_entry(&other_shell));
        let foreign = json!({ "type": "command", "command": "/x/hooks/recall.sh" });
        assert!(!is_retrivio_entry(&foreign));
    }

    #[test]
    fn merge_from_null_document_works() {
        let mut doc = Value::Null;
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Installed);
        assert!(doc.is_object());
        assert!(retrivio_hook_status(&doc).installed());
    }

    #[test]
    fn merge_preserves_unrelated_hooks_and_keys() {
        let original = fixture_settings();
        let mut doc = original.clone();
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Installed);

        // Every non-hook key is untouched.
        for (k, v) in original.as_object().unwrap() {
            if k != "hooks" {
                assert_eq!(&doc[k], v, "key {} changed", k);
            }
        }
        // Every pre-existing event keeps its groups as a prefix, in order.
        for (event, orig_groups) in original["hooks"].as_object().unwrap() {
            let orig = orig_groups.as_array().unwrap();
            let now = groups(&doc, event);
            assert!(now.len() >= orig.len(), "{} lost groups", event);
            assert_eq!(&now[..orig.len()], &orig[..], "{} groups changed", event);
        }
        // SessionStart: foreign "compact" group first, ours appended with its own matcher.
        let ss = groups(&doc, "SessionStart");
        assert_eq!(ss.len(), 2);
        assert_eq!(ss[0]["matcher"], json!("compact"));
        assert_eq!(ss[1]["matcher"], json!("compact|clear"));
        assert_eq!(commands(&doc, "UserPromptSubmit"), vec![BIN.to_string()]);
        // Unrelated events untouched in count.
        assert_eq!(doc["hooks"].as_object().unwrap().len(), 6);
    }

    #[test]
    fn merge_twice_is_a_noop() {
        let mut doc = fixture_settings();
        merge_claude_hooks(&mut doc, BIN).unwrap();
        let snapshot = doc.clone();
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Unchanged);
        assert_eq!(doc, snapshot);
    }

    #[test]
    fn merge_updates_bin_path_in_place() {
        let mut doc = fixture_settings();
        merge_claude_hooks(&mut doc, BIN).unwrap();
        let ss_before = groups(&doc, "SessionStart").len();
        assert_eq!(merge_claude_hooks(&mut doc, BIN2).unwrap(), ChangeKind::Updated);
        assert_eq!(groups(&doc, "SessionStart").len(), ss_before);
        assert_eq!(groups(&doc, "UserPromptSubmit").len(), 1);
        let all: Vec<String> = commands(&doc, "UserPromptSubmit")
            .into_iter()
            .chain(commands(&doc, "SessionStart"))
            .collect();
        assert_eq!(all.iter().filter(|c| c.as_str() == BIN2).count(), 2);
        assert!(!all.iter().any(|c| c.contains(BIN)), "old bin still referenced: {:?}", all);
        assert_eq!(retrivio_hook_status(&doc).bins(), vec![BIN2.to_string()]);
    }

    #[test]
    fn merge_reports_updated_when_only_one_hook_was_missing() {
        let mut doc = json!({});
        merge_claude_hooks(&mut doc, BIN).unwrap();
        doc["hooks"].as_object_mut().unwrap().remove("SessionStart");
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Updated);
        assert!(retrivio_hook_status(&doc).installed());
    }

    #[test]
    fn merge_moves_ours_out_of_shared_group_with_wrong_matcher() {
        let mut doc = json!({
            "hooks": {
                "SessionStart": [
                    { "matcher": "compact", "hooks": [
                        { "type": "command", "command": "/x/foreign.sh" },
                        { "type": "command", "command": "/old/retrivio recall --reset-session", "timeout": 5 }
                    ] }
                ]
            }
        });
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Updated);
        let ss = groups(&doc, "SessionStart");
        assert_eq!(ss.len(), 2);
        assert_eq!(ss[0]["matcher"], json!("compact"));
        assert_eq!(ss[0]["hooks"].as_array().unwrap().len(), 1);
        assert_eq!(ss[0]["hooks"][0]["command"], json!("/x/foreign.sh"));
        assert_eq!(ss[1]["matcher"], json!("compact|clear"));
        assert_eq!(ss[1]["hooks"][0]["command"], json!(BIN));
        assert_eq!(ss[1]["hooks"][0]["args"], json!(["recall", "--reset-session"]));
    }

    #[test]
    fn merge_drops_duplicate_retrivio_entries() {
        let mut doc = json!({});
        merge_claude_hooks(&mut doc, BIN).unwrap();
        let dup = HookKind::Prompt.entry("/dup/retrivio", EntryForm::Exec);
        doc["hooks"]["UserPromptSubmit"]
            .as_array_mut()
            .unwrap()
            .push(new_group(None, dup));
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Updated);
        assert_eq!(commands(&doc, "UserPromptSubmit"), vec![BIN.to_string()]);
    }

    #[test]
    fn merge_rejects_non_object_hooks() {
        let mut doc = json!({ "hooks": [] });
        assert!(merge_claude_hooks(&mut doc, BIN).is_err());
        let mut doc = json!({ "hooks": { "UserPromptSubmit": {} } });
        assert!(merge_claude_hooks(&mut doc, BIN).is_err());
    }

    #[test]
    fn uninstall_removes_only_ours() {
        let original = fixture_settings();
        let mut doc = original.clone();
        merge_claude_hooks(&mut doc, BIN).unwrap();
        assert!(remove_retrivio_hooks(&mut doc));
        assert_eq!(doc, original);
        assert!(doc["hooks"].is_object());
        // Nothing left to remove.
        assert!(!remove_retrivio_hooks(&mut doc));
        assert_eq!(doc, original);
        // A file without ours is untouched and reports false.
        let mut untouched = fixture_settings();
        assert!(!remove_retrivio_hooks(&mut untouched));
        assert_eq!(untouched, fixture_settings());
    }

    #[test]
    fn uninstall_keeps_preexisting_empty_event_arrays() {
        let mut doc = json!({ "hooks": { "Stop": [] } });
        merge_claude_hooks(&mut doc, BIN).unwrap();
        assert!(remove_retrivio_hooks(&mut doc));
        assert_eq!(doc, json!({ "hooks": { "Stop": [] } }));
    }

    #[test]
    fn codex_file_creation_shape() {
        let mut doc = Value::Null;
        assert_eq!(merge_codex_hooks(&mut doc, BIN).unwrap(), ChangeKind::Installed);
        let expected = json!({
            "description": "Retrivio proactive recall hooks",
            "hooks": {
                "UserPromptSubmit": [
                    { "hooks": [
                        { "type": "command", "command": format!("{} recall", BIN), "timeout": 5, "statusMessage": "retrivio recall" }
                    ] }
                ],
                "SessionStart": [
                    { "matcher": "compact|clear", "hooks": [
                        { "type": "command", "command": format!("{} recall --reset-session", BIN), "timeout": 5 }
                    ] }
                ]
            }
        });
        assert_eq!(doc, expected);
        // Merging into an existing Codex file keeps its description and is idempotent.
        let mut existing = json!({ "description": "mine", "hooks": { "Stop": [ { "hooks": [ { "type": "command", "command": "say hi" } ] } ] } });
        assert_eq!(merge_codex_hooks(&mut existing, BIN).unwrap(), ChangeKind::Installed);
        assert_eq!(existing["description"], json!("mine"));
        assert_eq!(existing["hooks"]["Stop"][0]["hooks"][0]["command"], json!("say hi"));
        assert_eq!(merge_codex_hooks(&mut existing, BIN).unwrap(), ChangeKind::Unchanged);
    }

    #[test]
    fn uninstall_from_codex_only_file_leaves_empty_hooks_object() {
        let mut doc = Value::Null;
        merge_codex_hooks(&mut doc, BIN).unwrap();
        assert!(remove_retrivio_hooks(&mut doc));
        assert_eq!(doc, json!({ "description": "Retrivio proactive recall hooks", "hooks": {} }));
    }

    #[test]
    fn status_reports_partial_and_absent() {
        let fixture = fixture_settings();
        let st = retrivio_hook_status(&fixture);
        assert!(!st.installed() && !st.partial());
        assert!(st.bins().is_empty());
        let mut doc = json!({});
        merge_claude_hooks(&mut doc, BIN).unwrap();
        doc["hooks"].as_object_mut().unwrap().remove("UserPromptSubmit");
        let st = retrivio_hook_status(&doc);
        assert!(st.partial());
        assert_eq!(st.bins(), vec![BIN.to_string()]);
    }

    #[test]
    fn command_roundtrip_with_spaces_and_quotes() {
        let spaced = "/Users/me/My Apps/retrivio";
        let cmd = HookKind::Prompt.shell_command(spaced);
        assert_eq!(cmd, "'/Users/me/My Apps/retrivio' recall");
        assert!(is_retrivio_recall_command(&cmd));
        assert_eq!(bin_from_command(&cmd).as_deref(), Some(spaced));
        let reset = HookKind::Reset.shell_command(BIN);
        assert_eq!(reset, format!("{} recall --reset-session", BIN));
        assert_eq!(bin_from_command(&reset).as_deref(), Some(BIN));
        assert_eq!(bin_from_command("\"/a b/retrivio\" recall").as_deref(), Some("/a b/retrivio"));
        assert_eq!(bin_from_command("something else"), None);
        assert!(!is_retrivio_recall_command("/Users/me/.claude/hooks/context-budget.sh"));
    }

    #[test]
    fn plist_contains_label_and_args() {
        let home = Path::new("/Users/me");
        let plist = render_launchd_plist("/Users/me/bin/retrivio", home);
        assert!(plist.contains("<string>com.stouffer-labs.retrivio.watch</string>"));
        assert!(plist.contains("<string>/Users/me/bin/retrivio</string>"));
        for arg in ["service", "run"] {
            assert!(plist.contains(&format!("<string>{}</string>", arg)), "missing {}", arg);
        }
        assert!(!plist.contains("<string>60</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>\n    <true/>"));
        assert!(plist.contains("<key>KeepAlive</key>\n    <true/>"));
        assert!(plist.contains("<key>ThrottleInterval</key>\n    <integer>60</integer>"));
        assert!(plist.contains("<key>ProcessType</key>\n    <string>Background</string>"));
        assert_eq!(plist.matches("<string>/Users/me/.retrivio/watch.log</string>").count(), 2);
        assert!(plist.contains("<string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"));
        assert!(plist.contains("<key>HOME</key>\n        <string>/Users/me</string>"));
        // Special characters are escaped.
        let odd = render_launchd_plist("/tmp/a&b/retrivio", home);
        assert!(odd.contains("/tmp/a&amp;b/retrivio"));
        let unit = render_systemd_unit("/usr/local/bin/retrivio", home);
        assert!(unit.contains("ExecStart=/usr/local/bin/retrivio service run"));
        assert!(unit.contains("/Users/me/.retrivio/watch.log"));
    }

    #[test]
    fn launchctl_print_summary_parses_fields() {
        let out = "gui/501/com.stouffer-labs.retrivio.watch = {\n\tactive count = 1\n\tpath = /Users/me/Library/LaunchAgents/x.plist\n\tstate = running\n\tprogram = /usr/local/bin/retrivio\n\targuments = {\n\t\t/usr/local/bin/retrivio\n\t\twatch\n\t}\n\tpid = 4242\n\tlast exit code = 0\n\tevent channels = {\n\t\tstate = 1\n\t}\n}\n";
        let s = summarize_launchctl_print(out);
        assert_eq!(s.state.as_deref(), Some("running"));
        assert_eq!(s.pid.as_deref(), Some("4242"));
        assert_eq!(s.last_exit.as_deref(), Some("0"));
        let empty = summarize_launchctl_print("");
        assert_eq!(empty, LaunchdSummary::default());
    }

    #[test]
    fn backup_path_appends_suffix() {
        let p = backup_path(Path::new("/h/.claude/settings.json"));
        assert_eq!(p, PathBuf::from("/h/.claude/settings.json.bak-retrivio"));
    }

    #[test]
    fn unrelated_hooks_mentioning_retrivio_and_recall_are_untouched() {
        let proxy_shell = json!({ "type": "command", "command": "/x/bin/retrivio-recall-proxy recall", "timeout": 5 });
        let proxy_exec = json!({ "type": "command", "command": "/x/bin/retrivio-recall-proxy", "args": ["recall"] });
        let wrapper = json!({ "type": "command", "command": "/x/hooks/recall.sh retrivio recall" });
        let extra_args = json!({ "type": "command", "command": BIN, "args": ["recall", "--verbose"] });
        let extra_shell = json!({ "type": "command", "command": format!("{} recall --reset-session --now", BIN) });
        let piped = json!({ "type": "command", "command": format!("{} recall | tee log", BIN) });
        let foreign: Vec<Value> = vec![proxy_shell, proxy_exec, wrapper, extra_args, extra_shell, piped];
        for e in &foreign {
            assert!(!is_retrivio_entry(e), "{}", e);
            assert!(bin_from_entry(e).is_none(), "{}", e);
        }
        let mut doc = json!({
            "hooks": {
                "UserPromptSubmit": [ { "hooks": foreign.clone() } ],
                "SessionStart": [ { "matcher": "compact|clear", "hooks": [ foreign[0].clone() ] } ]
            }
        });
        let before = doc.clone();
        assert_eq!(merge_claude_hooks(&mut doc, BIN).unwrap(), ChangeKind::Installed);
        // The foreign groups are untouched; ours were appended as new groups.
        assert_eq!(doc["hooks"]["UserPromptSubmit"][0], before["hooks"]["UserPromptSubmit"][0]);
        assert_eq!(doc["hooks"]["SessionStart"][0], before["hooks"]["SessionStart"][0]);
        assert_eq!(groups(&doc, "UserPromptSubmit").len(), 2);
        assert_eq!(groups(&doc, "SessionStart").len(), 2);
        assert_eq!(retrivio_hook_status(&doc).bins(), vec![BIN.to_string()]);
        assert!(remove_retrivio_hooks(&mut doc));
        assert_eq!(doc, before);
        assert!(!remove_retrivio_hooks(&mut doc));
        assert!(!retrivio_hook_status(&before).partial());
    }

    #[test]
    fn ownership_matches_exact_generated_forms_only() {
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": "retrivio", "args": ["recall"] })));
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": BIN, "args": ["recall", "--reset-session"] })));
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": "retrivio recall" })));
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": format!("  {}   recall   --reset-session  ", BIN) })));
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": "'/Users/me/My Apps/retrivio' recall" })));
        assert!(is_retrivio_entry(&json!({ "type": "command", "command": "\"/a b/retrivio\" recall" })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": BIN, "args": ["recall", 1] })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": BIN, "args": "recall" })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": BIN, "args": [] })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": BIN })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": "/x/Retrivio recall" })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": "/x/retrivio.sh recall" })));
        assert!(!is_retrivio_entry(&json!({ "type": "command", "command": "'/x/retrivio'recall" })));
        assert!(!is_retrivio_entry(&json!({ "type": "command" })));
        assert_eq!(bin_from_command("'/a'\\''b/retrivio' recall").as_deref(), Some("/a'b/retrivio"));
        assert_eq!(bin_from_command("'/unterminated/retrivio recall"), None);
        assert_eq!(bin_from_command("retrivio recall"), Some("retrivio".to_string()));
        assert_eq!(bin_from_command("retrivio recall now"), None);
        assert_eq!(split_first_token("  a b").map(|(t, r)| (t, r.to_string())), Some(("a".to_string(), " b".to_string())));
        assert_eq!(split_first_token(""), None);
    }

    fn scratch(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/test-hooks")
            .join(format!("{}-{}-{}", name, process::id(), nanos));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn atomic_write_creates_backs_up_and_leaves_no_temp_files() {
        let dir = scratch("atomic");
        let file = dir.join("settings.json");
        let mut backed_up = false;
        assert_eq!(write_json_atomic(&file, &json!({ "a": 1 }), &mut backed_up).unwrap(), None);
        assert!(!backed_up);
        assert_eq!(fs::read_to_string(&file).unwrap(), "{\n  \"a\": 1\n}\n");
        #[cfg(unix)]
        assert_eq!(fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600, "new files are private");
        #[cfg(unix)]
        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();
        let bak = write_json_atomic(&file, &json!({ "a": 2 }), &mut backed_up).unwrap().unwrap();
        assert!(backed_up);
        assert_eq!(bak, backup_path(&file));
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\n  \"a\": 1\n}\n");
        assert_eq!(fs::read_to_string(&file).unwrap(), "{\n  \"a\": 2\n}\n");
        #[cfg(unix)]
        {
            assert_eq!(fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o644, "existing mode preserved");
            assert_eq!(fs::metadata(&bak).unwrap().permissions().mode() & 0o777, 0o644);
        }
        // A second write in the same run does not overwrite the backup.
        assert_eq!(write_json_atomic(&file, &json!({ "a": 3 }), &mut backed_up).unwrap(), None);
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\n  \"a\": 1\n}\n");
        let names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(names.iter().all(|n| !n.contains("tmp-retrivio")), "{:?}", names);
        assert_ne!(random_suffix(), random_suffix());
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_follows_symlinks_only_to_regular_files() {
        use std::os::unix::fs::symlink;
        let dir = scratch("symlink");
        let real = dir.join("real.json");
        fs::write(&real, "{}\n").unwrap();
        let link = dir.join("settings.json");
        symlink(&real, &link).unwrap();
        let mut backed_up = false;
        let bak = write_json_atomic(&link, &json!({ "via": "link" }), &mut backed_up).unwrap().unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink(), "link stays a link");
        assert_eq!(fs::read_to_string(&real).unwrap(), "{\n  \"via\": \"link\"\n}\n");
        assert_eq!(bak, backup_path(&fs::canonicalize(&real).unwrap()));
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{}\n");
        // Symlink to a directory: refused, nothing written.
        let to_dir = dir.join("to-dir.json");
        symlink(&dir, &to_dir).unwrap();
        let err = write_json_atomic(&to_dir, &json!({}), &mut false).unwrap_err();
        assert!(err.contains("refusing"), "{}", err);
        // Dangling symlink: refused.
        let dangling = dir.join("dangling.json");
        symlink(dir.join("missing.json"), &dangling).unwrap();
        assert!(write_json_atomic(&dangling, &json!({}), &mut false).is_err());
        assert!(!dir.join("missing.json").exists());
        // A directory at the path itself: refused.
        let as_dir = dir.join("dir.json");
        fs::create_dir(&as_dir).unwrap();
        let err = write_json_atomic(&as_dir, &json!({}), &mut false).unwrap_err();
        assert!(err.contains("not a regular file"), "{}", err);
        // A planted symlink at the backup path is never followed: the write is refused before
        // anything is touched.
        let cfg = dir.join("hooks.json");
        fs::write(&cfg, "{\"keep\":true}\n").unwrap();
        let victim = dir.join("victim.txt");
        fs::write(&victim, "victim\n").unwrap();
        symlink(&victim, backup_path(&cfg)).unwrap();
        let err = write_json_atomic(&cfg, &json!({ "keep": false }), &mut false).unwrap_err();
        assert!(err.contains("refusing"), "{}", err);
        assert_eq!(fs::read_to_string(&victim).unwrap(), "victim\n");
        assert_eq!(fs::read_to_string(&cfg).unwrap(), "{\"keep\":true}\n");
        // An existing regular backup is replaced.
        fs::remove_file(backup_path(&cfg)).unwrap();
        fs::write(backup_path(&cfg), "old backup\n").unwrap();
        let bak = write_json_atomic(&cfg, &json!({ "keep": false }), &mut false).unwrap().unwrap();
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\"keep\":true}\n");
        let _ = fs::remove_dir_all(&dir);
    }
}
