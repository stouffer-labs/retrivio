//! `retrivio hook` and `retrivio service`: install/uninstall/status for CLI hooks and the launchd watcher.
//! See docs/superpowers/specs/2026-09-19-proactive-recall-design.md §6.
//!
//! The JSON and plist work is done by pure functions over `serde_json::Value` / strings so it can
//! be unit tested; the thin I/O wrappers further down handle files, prompts and `launchctl`.
//!
//! Environment: `RETRIVIO_HOME`, when set, replaces the real home directory for every path this
//! module touches (`~/.claude`, `~/.codex`, `~/Library/LaunchAgents`, `~/.retrivio/*.log`). It
//! exists so the commands can be exercised against a scratch directory. Without it, a set and
//! non-empty `CODEX_HOME` relocates the Codex directory the way Codex itself does (see
//! [`codex_dir`]).

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

const LAUNCHD_LABEL: &str = "com.stouffer-labs.retrivio.watch";
const HOOK_TIMEOUT_SECS: u64 = 5;
const SESSION_START_MATCHER: &str = "compact|clear";
const CODEX_DESCRIPTION: &str = "Retrivio proactive recall hooks";
const CODEX_TRUST_REMINDER: &str = "Codex trusts the exact hook definition (hash-based). Run `retrivio hook trust --codex` (or /hooks inside Codex) after installation AND after any change to the hooks (for example a new binary path).";
const WATCH_INTERVAL_SECS: &str = "300";
const LAUNCHD_THROTTLE_SECS: u32 = 60;
const BACKUP_SUFFIX: &str = ".bak-retrivio";
const LAUNCHD_PATH_ENV: &str = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin";

/// PATH for the background service: the standard directories plus every directory on the
/// installing shell's PATH (launchd starts agents with a minimal PATH, which hides credential
/// helpers such as isengardcli or aws-sso tooling and makes the watcher fail on startup).
fn service_path_env() -> String {
    service_path_env_from(&std::env::var("PATH").unwrap_or_default())
}

/// [`service_path_env`] over an explicit PATH string: [`LAUNCHD_PATH_ENV`] first, then every
/// entry of `current_path` that passes [`service_path_dir_ok`], once each.
fn service_path_env_from(current_path: &str) -> String {
    let mut parts: Vec<String> = LAUNCHD_PATH_ENV.split(':').map(|s| s.to_string()).collect();
    for seg in current_path.split(':') {
        let seg = seg.trim();
        if service_path_dir_ok(seg) && !parts.iter().any(|p| p == seg) {
            parts.push(seg.to_string());
        }
    }
    parts.join(":")
}

/// Whether one shell PATH entry may be baked into the agent's PATH. Only stable, absolute,
/// existing directories that nobody else can write to qualify: no relative entries, no temp or
/// shim directories that vanish (`/tmp`, `/var/folders`), nothing under `node_modules` or in a
/// `.bin` directory (per-project tool shims), and nothing writable by group or others (a bare
/// helper name such as `isengardcli` could be hijacked there).
fn service_path_dir_ok(seg: &str) -> bool {
    let trimmed = if seg.len() > 1 {
        seg.trim_end_matches('/')
    } else {
        seg
    };
    if !trimmed.starts_with('/')
        || trimmed.starts_with("/tmp")
        || trimmed.starts_with("/private/tmp")
        || trimmed.starts_with("/var/folders")
        || trimmed.contains("/node_modules/")
        || trimmed.ends_with("/node_modules")
        || trimmed.ends_with("/.bin")
    {
        return false;
    }
    let Ok(meta) = fs::metadata(trimmed) else {
        return false;
    };
    if !meta.is_dir() {
        return false;
    }
    #[cfg(unix)]
    if meta.permissions().mode() & 0o022 != 0 {
        return false;
    }
    true
}
const SYSTEMD_UNIT_NAME: &str = "retrivio-watch.service";
const TAIL_LINES: usize = 5;

const HOOK_USAGE: &str = "usage: retrivio hook <install|uninstall|status|trust> [--claude] [--codex] [--yes] [--bin <path>]

  install     add the retrivio recall hooks (UserPromptSubmit + SessionStart) to each detected CLI
  uninstall   remove only the retrivio entries, leaving every other hook untouched
  status      report detection, install state, referenced binary and the recall log tail
  trust       mark the installed Codex hooks as trusted (what /hooks does), via `codex app-server`

  --claude / --codex   limit to one CLI (default: every detected CLI)
  --yes, -y            do not prompt before writing
  --bin <path>         binary to reference from the hooks (default: this executable)

Files: ~/.claude/settings.json and ~/.codex/hooks.json (a .bak-retrivio copy is taken before writing).
Codex trust state lives in ~/.codex/config.toml ([hooks.state]) and is written by Codex itself.
RETRIVIO_HOME overrides the home directory used to locate them; otherwise CODEX_HOME, when set, relocates ~/.codex.
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
            Cli::Codex => codex_dir(
                home,
                env::var("CODEX_HOME").ok().as_deref(),
                env::var("RETRIVIO_HOME").ok().as_deref(),
            ),
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

/// The Codex directory (`hooks.json`, `config.toml`). A set, non-empty `RETRIVIO_HOME` wins so
/// scratch runs stay inside the scratch home (`home` is already that directory); otherwise a set,
/// non-empty `CODEX_HOME` is used as given, the way Codex itself resolves it; otherwise
/// `~/.codex`.
fn codex_dir(home: &Path, env_codex_home: Option<&str>, retrivio_home: Option<&str>) -> PathBuf {
    fn non_empty(v: Option<&str>) -> Option<&str> {
        v.map(str::trim).filter(|s| !s.is_empty())
    }
    if non_empty(retrivio_home).is_some() {
        return home.join(".codex");
    }
    match non_empty(env_codex_home) {
        Some(dir) => PathBuf::from(dir),
        None => home.join(".codex"),
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

    /// The same event as the Codex app-server reports it (camelCase `eventName`).
    fn codex_event(self) -> &'static str {
        match self {
            HookKind::Prompt => "userPromptSubmit",
            HookKind::Reset => "sessionStart",
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
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
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
    let mut z =
        nanos ^ (process::id() as u64).rotate_left(32) ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
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
fn create_temp_next_to(
    target: &Path,
    parent: &Path,
    mode: u32,
) -> std::io::Result<(fs::File, PathBuf)> {
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
            println!(
                "{} ({}), last {} lines:",
                label,
                path.display(),
                lines.len()
            );
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
    let opts = match parse_common(
        args,
        &["install", "uninstall", "status", "trust"],
        "status",
        true,
    ) {
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
        "trust" => hook_trust(&opts),
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
    // True once the Codex hooks.json holds our hooks (freshly written or already there), i.e.
    // the moment Codex could be asked to trust them.
    let mut codex_ready = false;
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
        // Ok(true) = the file now holds our hooks; Ok(false) = the user declined.
        let outcome = (|| -> Result<bool, String> {
            let mut doc = read_json_file(&file)?.unwrap_or(Value::Null);
            let change = cli.merge(&mut doc, &bin_str)?;
            if change == ChangeKind::Unchanged {
                println!(
                    "  {}: already installed ({})",
                    cli.display(),
                    file.display()
                );
                return Ok(true);
            }
            let question = match change {
                ChangeKind::Installed => {
                    format!("  install retrivio hooks into {}?", file.display())
                }
                _ => format!("  update retrivio hooks in {}?", file.display()),
            };
            if !confirm(opts.yes, &question) {
                println!("  {}: skipped", cli.display());
                return Ok(false);
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
            Ok(true)
        })();
        match outcome {
            Ok(ready) => {
                if cli == Cli::Codex {
                    codex_ready = ready;
                }
            }
            Err(e) => {
                eprintln!("  error: {}: {}", cli.display(), e);
                failures += 1;
            }
        }
    }
    if codex_ready {
        // Installing is not enough for Codex: it ignores hooks until their exact definition is
        // trusted. Do what /hooks does through the app-server; a failure here never changes the
        // exit code because the install itself succeeded.
        match codex_trust_hooks(&home, &bin_str, true) {
            Ok(TrustOutcome::Trusted(_)) => {
                println!("  Codex: hooks trusted automatically (via codex app-server)");
            }
            Ok(TrustOutcome::AlreadyTrusted(_)) => println!("  Codex: hooks already trusted"),
            Ok(TrustOutcome::NotFound { codex_home }) => {
                println!(
                    "  Codex: automatic trust skipped: codex app-server lists no retrivio hooks for {} from {}{}",
                    bin_str,
                    Cli::Codex.hooks_file(&home).display(),
                    describe_codex_home_mismatch(&home, codex_home.as_deref())
                );
                println!("  {}", CODEX_TRUST_REMINDER);
            }
            Ok(TrustOutcome::Declined) => println!("  {}", CODEX_TRUST_REMINDER),
            Err(e) => {
                println!("  Codex: automatic trust failed: {}", e);
                println!("  {}", CODEX_TRUST_REMINDER);
            }
        }
    } else if saw_codex {
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
        // Binaries our entries in the file reference; Codex is asked about exactly these below.
        let mut installed_bins: Vec<String> = Vec::new();
        match read_json_file(&file) {
            Ok(Some(doc)) if !doc.is_null() => {
                let status = retrivio_hook_status(&doc);
                let state = if status.installed() {
                    "yes (UserPromptSubmit + SessionStart)".to_string()
                } else if status.partial() {
                    format!(
                        "partial ({} present, {} missing)",
                        if status.prompt_command.is_some() {
                            "UserPromptSubmit"
                        } else {
                            "SessionStart"
                        },
                        if status.prompt_command.is_some() {
                            "SessionStart"
                        } else {
                            "UserPromptSubmit"
                        },
                    )
                } else {
                    "no".to_string()
                };
                println!("  installed: {}", state);
                installed_bins = status.bins();
                for bin in &installed_bins {
                    let path = Path::new(bin);
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
                        if executable {
                            "executable"
                        } else {
                            "not executable"
                        },
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
            let mut need_note = true;
            if cli.detected(&home) {
                if installed_bins.is_empty() {
                    println!("  trust: none (no retrivio hooks in {})", file.display());
                } else {
                    match codex_hook_trust_report(&home, &installed_bins) {
                        Ok((report, codex_home)) => {
                            let mut any = false;
                            let mut all_trusted = true;
                            for (bin, hooks) in &report {
                                if hooks.is_empty() {
                                    println!(
                                        "  trust: none for {} (codex app-server lists no exact retrivio hooks from {}{})",
                                        bin,
                                        file.display(),
                                        describe_codex_home_mismatch(&home, codex_home.as_deref())
                                    );
                                    continue;
                                }
                                let which = if report.len() > 1 {
                                    format!(" ({})", bin)
                                } else {
                                    String::new()
                                };
                                for h in hooks {
                                    any = true;
                                    all_trusted &= h.trust_status == "trusted";
                                    println!(
                                        "  trust: {} {}{}",
                                        h.event_name, h.trust_status, which
                                    );
                                }
                            }
                            need_note = !any || !all_trusted;
                        }
                        Err(e) => {
                            println!("  trust: unknown (codex app-server unavailable: {})", e)
                        }
                    }
                }
            }
            if need_note {
                println!("  note: {}", CODEX_TRUST_REMINDER);
            }
        }
    }
    print_log_tail("recall log", &recall_log_path(&home));
    Ok(())
}

fn hook_trust(opts: &HookOpts) -> Result<(), String> {
    let home = base_home()?;
    if opts.claude {
        println!("  Claude Code: nothing to do (Claude Code hooks need no trust step)");
        if !opts.codex {
            return Ok(());
        }
    }
    if !Cli::Codex.detected(&home) {
        println!(
            "  Codex: skipped (not detected: {} missing)",
            Cli::Codex.dir(&home).display()
        );
        return Ok(());
    }
    // Only hooks that are byte-exactly what `hook install` writes for this binary are trusted.
    let bin = resolve_bin(opts.bin.as_deref())?;
    let bin_str = bin.to_string_lossy().to_string();
    println!("  bin: {}", bin_str);
    match codex_trust_hooks(&home, &bin_str, opts.yes)? {
        TrustOutcome::NotFound { codex_home } => {
            println!(
                "  Codex: no retrivio hooks for {} found (codex app-server lists none from {}{})",
                bin_str,
                Cli::Codex.hooks_file(&home).display(),
                describe_codex_home_mismatch(&home, codex_home.as_deref())
            );
            println!("  hint: run `retrivio hook install --codex` first, or pass --bin <path> naming the binary the installed hooks reference");
        }
        TrustOutcome::AlreadyTrusted(n) => {
            println!("  Codex: already trusted ({} retrivio hook(s))", n);
        }
        TrustOutcome::Declined => println!("  Codex: skipped"),
        TrustOutcome::Trusted(n) => {
            println!("  Codex: {} hook(s) trusted (via codex app-server)", n);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Codex hook trust via `codex app-server` (JSON-RPC 2.0, one JSON object per line over stdio)
// ---------------------------------------------------------------------------
//
// Codex only runs hooks whose exact definition the user has trusted. `/hooks` in the Codex TUI
// stores a per-hook `trusted_hash` under `[hooks.state."<key>"]` in ~/.codex/config.toml; both
// the key and the hash are computed by Codex, so we never write that file ourselves. The same
// data comes from `codex app-server` (`hooks/list`) and the same write is `config/batchWrite`,
// which is exactly what the `/hooks` UI does. The message builders and response readers are pure
// functions (unit tested against fixtures shaped like the real replies); `AppServer` is the
// stdio transport.

const CODEX_APP_SERVER_TIMEOUT_SECS: u64 = 30;
/// Characters that chain, redirect, substitute or escape in `sh`. We never write one into a
/// hook command (an unusual binary path is single-quoted by [`shell_quote`], and even then such a
/// path is not auto-trusted), so a command containing any of them is never ours.
const SHELL_METACHARACTERS: &[char] = &[
    ';', '&', '|', '<', '>', '$', '`', '(', ')', '\\', '\n', '\r',
];
const CODEX_APP_SERVER_STDERR_CAP: usize = 4096;

pub fn build_initialize_request(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "clientInfo": {
                "name": "retrivio",
                "version": env!("CARGO_PKG_VERSION"),
                "title": "retrivio hook install"
            },
            "capabilities": { "experimentalApi": true }
        }
    })
}

pub fn build_initialized_notification() -> Value {
    json!({ "jsonrpc": "2.0", "method": "initialized", "params": {} })
}

pub fn build_hooks_list_request(id: u64, cwd: &Path) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "hooks/list",
        "params": { "cwds": [cwd.to_string_lossy()] }
    })
}

/// One of our hooks as the Codex app-server sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexHookTrust {
    pub key: String,
    pub event_name: String,
    pub source_path: String,
    pub current_hash: String,
    /// `trusted`, `untrusted`, `modified` or `managed`.
    pub trust_status: String,
}

/// Every hook entry in a `hooks/list` result (`result.data[].hooks[]`), across all cwds.
fn hooks_in_list_result(list_result: &Value) -> impl Iterator<Item = &Value> {
    list_result
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("hooks").and_then(Value::as_array))
        .flatten()
}

fn is_null_or_absent(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null))
}

/// Whether one `hooks/list` entry is byte-exactly a hook `hook install` writes for `bin`:
/// loaded from the hooks.json we manage (`sourcePath`), a plain user hook (`source == "user"`,
/// no `pluginId`, handler `command`), for one of our two events, with exactly the command string
/// [`HookKind::shell_command`] produces for `bin` (same binary path, same arguments, no shell
/// metacharacters), our matcher (none for userPromptSubmit, `compact|clear` for sessionStart),
/// our timeout and not async. Anything else - another binary named retrivio, extra arguments,
/// a chained command, a wrapper, a different matcher - is somebody's own hook and never ours.
fn is_exact_retrivio_codex_hook(hook: &Value, hooks_json_path: &Path, bin: &str) -> bool {
    let field = |name: &str| hook.get(name).and_then(Value::as_str);
    let (Some(event), Some(source_path), Some(command)) =
        (field("eventName"), field("sourcePath"), field("command"))
    else {
        return false;
    };
    let Some(kind) = HookKind::ALL
        .iter()
        .copied()
        .find(|k| k.codex_event() == event)
    else {
        return false;
    };
    if Path::new(source_path) != hooks_json_path {
        return false;
    }
    if field("source") != Some("user") || !is_null_or_absent(hook.get("pluginId")) {
        return false;
    }
    if let Some(handler) = hook.get("handlerType") {
        if handler.as_str() != Some("command") {
            return false;
        }
    }
    if command != kind.shell_command(bin) || command.contains(SHELL_METACHARACTERS) {
        return false;
    }
    let matcher_ok = match kind.matcher() {
        None => is_null_or_absent(hook.get("matcher")),
        Some(expected) => field("matcher") == Some(expected),
    };
    if !matcher_ok {
        return false;
    }
    if hook.get("timeoutSec").and_then(Value::as_f64) != Some(HOOK_TIMEOUT_SECS as f64) {
        return false;
    }
    matches!(
        hook.get("async"),
        None | Some(Value::Null) | Some(Value::Bool(false))
    )
}

/// Our hooks in a `hooks/list` result: every entry that passes
/// [`is_exact_retrivio_codex_hook`] for `bin`, deduplicated by key (one hook is listed once per
/// cwd). A user's own hook in the same file, or ours pointing at another binary, is not included.
pub fn retrivio_codex_hooks(
    list_result: &Value,
    hooks_json_path: &Path,
    bin: &str,
) -> Vec<CodexHookTrust> {
    let mut out: Vec<CodexHookTrust> = Vec::new();
    for hook in hooks_in_list_result(list_result) {
        if !is_exact_retrivio_codex_hook(hook, hooks_json_path, bin) {
            continue;
        }
        let field = |name: &str| hook.get(name).and_then(Value::as_str);
        let (Some(key), Some(event), Some(source_path), Some(hash), Some(status)) = (
            field("key"),
            field("eventName"),
            field("sourcePath"),
            field("currentHash"),
            field("trustStatus"),
        ) else {
            continue;
        };
        if out.iter().any(|h| h.key == key) {
            continue;
        }
        out.push(CodexHookTrust {
            key: key.to_string(),
            event_name: event.to_string(),
            source_path: source_path.to_string(),
            current_hash: hash.to_string(),
            trust_status: status.to_string(),
        });
    }
    out
}

/// What [`select_untrusted_retrivio_hooks`] decided.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TrustSelection {
    /// Hooks to write a `trusted_hash` for (`untrusted`, or `modified` after a definition
    /// change): at most one per event.
    pub pending: Vec<CodexHookTrust>,
    /// Events skipped because more than one exact match was listed for them, naming the keys.
    pub warnings: Vec<String>,
}

/// Our hooks that Codex will not run yet. `managed` and `trusted` hooks need nothing. An event
/// with two or more exact matches (our merge never writes two, so somebody edited the file) gets
/// nothing selected and a warning instead.
pub fn select_untrusted_retrivio_hooks(
    list_result: &Value,
    hooks_json_path: &Path,
    bin: &str,
) -> TrustSelection {
    let ours = retrivio_codex_hooks(list_result, hooks_json_path, bin);
    let mut selection = TrustSelection::default();
    for kind in HookKind::ALL {
        let event = kind.codex_event();
        let matches: Vec<&CodexHookTrust> = ours.iter().filter(|h| h.event_name == event).collect();
        match matches.as_slice() {
            [] => {}
            [one] => {
                if matches!(one.trust_status.as_str(), "untrusted" | "modified") {
                    selection.pending.push((*one).clone());
                }
            }
            many => {
                let keys: Vec<&str> = many.iter().map(|h| h.key.as_str()).collect();
                selection.warnings.push(format!(
                    "{}: {} retrivio hooks match where one is expected; none trusted: {}",
                    event,
                    many.len(),
                    keys.join(", ")
                ));
            }
        }
    }
    selection
}

/// The `config/batchWrite` the `/hooks` UI sends: upsert `hooks.state.<key>.trusted_hash` for
/// every hook and reload the user config so the running server sees it.
pub fn build_trust_write_request(id: u64, hooks: &[CodexHookTrust]) -> Value {
    let mut state = Map::new();
    for hook in hooks {
        state.insert(
            hook.key.clone(),
            json!({ "trusted_hash": hook.current_hash }),
        );
    }
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "config/batchWrite",
        "params": {
            "edits": [{
                "keyPath": "hooks.state",
                "value": Value::Object(state),
                "mergeStrategy": "upsert"
            }],
            "reloadUserConfig": true
        }
    })
}

/// Outcome of re-checking one written hook against a fresh `hooks/list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustVerification {
    /// Listed with the same key, hash, source path and event, and `trusted`.
    Confirmed,
    /// Same identity, but Codex still reports this other status.
    NotTrusted(String),
    /// The key is listed, but its hash, source path or event differs from what we selected:
    /// the definition changed between list and re-list, so the trusted hash may belong to
    /// something we never saw.
    DefinitionChanged,
    /// The key is no longer listed at all.
    NotListed,
}

/// Confirm each of `expected` in a `hooks/list` result by identity (key and hash and source
/// path and event), not by key alone.
pub fn verify_trusted_hooks(
    list_result: &Value,
    expected: &[CodexHookTrust],
) -> Vec<(CodexHookTrust, TrustVerification)> {
    fn field<'a>(hook: &'a Value, name: &str) -> Option<&'a str> {
        hook.get(name).and_then(Value::as_str)
    }
    expected
        .iter()
        .map(|want| {
            let same_key: Vec<&Value> = hooks_in_list_result(list_result)
                .filter(|h| field(h, "key") == Some(want.key.as_str()))
                .collect();
            let identical = same_key.iter().find(|h| {
                field(h, "currentHash") == Some(want.current_hash.as_str())
                    && field(h, "sourcePath") == Some(want.source_path.as_str())
                    && field(h, "eventName") == Some(want.event_name.as_str())
            });
            let outcome = match identical {
                Some(h) => match field(h, "trustStatus") {
                    Some("trusted") => TrustVerification::Confirmed,
                    other => TrustVerification::NotTrusted(other.unwrap_or("?").to_string()),
                },
                None if same_key.is_empty() => TrustVerification::NotListed,
                None => TrustVerification::DefinitionChanged,
            };
            (want.clone(), outcome)
        })
        .collect()
}

/// `Some(Ok(result))` or `Some(Err(message))` when `msg` is the JSON-RPC response to request
/// `id`; `None` for everything else (notifications, other ids, and server-to-client requests,
/// which carry both `id` and `method`).
pub fn match_response(msg: &Value, id: u64) -> Option<Result<Value, String>> {
    if msg.get("method").is_some() || msg.get("id").and_then(Value::as_u64) != Some(id) {
        return None;
    }
    if let Some(err) = msg.get("error") {
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| err.to_string());
        return Some(Err(match err.get("code").and_then(Value::as_i64) {
            Some(code) => format!("{} (code {})", message, code),
            None => message,
        }));
    }
    Some(Ok(msg.get("result").cloned().unwrap_or(Value::Null)))
}

/// Kills and reaps a freshly spawned child unless [`ChildGuard::into_inner`] hands it over
/// first, so nothing between `spawn()` and the construction of [`AppServer`] (a failed pipe, a
/// thread that could not start, a panic) leaks the process.
struct ChildGuard(Option<process::Child>);

impl ChildGuard {
    fn into_inner(mut self) -> process::Child {
        self.0.take().expect("child already taken")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// A running `codex app-server` child. Stdout is drained by a thread into a channel so every
/// wait can time out; the child is killed on drop (every error path included).
struct AppServer {
    child: process::Child,
    stdin: process::ChildStdin,
    lines: mpsc::Receiver<std::io::Result<String>>,
    stderr: Arc<Mutex<String>>,
    next_id: u64,
}

impl AppServer {
    fn spawn(codex_home: Option<&Path>) -> Result<AppServer, String> {
        let path_env = env::var("PATH").unwrap_or_default();
        let codex = find_on_path("codex", &path_env)
            .ok_or_else(|| "codex not found on PATH".to_string())?;
        let mut cmd = Command::new(&codex);
        cmd.arg("app-server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = codex_home {
            cmd.env("CODEX_HOME", dir);
        }
        // Until the AppServer owns it, the guard kills and reaps the child on every early return.
        let mut guard =
            ChildGuard(Some(cmd.spawn().map_err(|e| {
                format!("cannot start `{} app-server`: {}", codex.display(), e)
            })?));
        let child = guard.0.as_mut().expect("child just spawned");
        let stdin = child.stdin.take().ok_or("codex app-server: no stdin")?;
        let stdout = child.stdout.take().ok_or("codex app-server: no stdout")?;
        let stderr_pipe = child.stderr.take().ok_or("codex app-server: no stderr")?;

        let (tx, lines) = mpsc::channel();
        thread::Builder::new()
            .name("codex-app-server-stdout".to_string())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {
                            if tx.send(Ok(line)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            break;
                        }
                    }
                }
            })
            .map_err(|e| format!("codex app-server: cannot start stdout reader: {}", e))?;
        let stderr = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&stderr);
        thread::Builder::new()
            .name("codex-app-server-stderr".to_string())
            .spawn(move || {
                let mut reader = BufReader::new(stderr_pipe);
                let mut line = String::new();
                while let Ok(n) = reader.read_line(&mut line) {
                    if n == 0 {
                        break;
                    }
                    if let Ok(mut buf) = sink.lock() {
                        if buf.len() < CODEX_APP_SERVER_STDERR_CAP {
                            buf.push_str(&line);
                        }
                    }
                    line.clear();
                }
            })
            .map_err(|e| format!("codex app-server: cannot start stderr reader: {}", e))?;
        Ok(AppServer {
            child: guard.into_inner(),
            stdin,
            lines,
            stderr,
            next_id: 1,
        })
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn stderr_tail(&self) -> String {
        let text = self
            .stderr
            .lock()
            .map(|b| b.trim().to_string())
            .unwrap_or_default();
        if text.is_empty() {
            String::new()
        } else {
            format!(" (stderr: {})", text.lines().last().unwrap_or(""))
        }
    }

    fn notify(&mut self, msg: &Value) -> Result<(), String> {
        let mut text = msg.to_string();
        text.push('\n');
        self.stdin
            .write_all(text.as_bytes())
            .and_then(|_| self.stdin.flush())
            .map_err(|e| {
                format!(
                    "codex app-server: write failed: {}{}",
                    e,
                    self.stderr_tail()
                )
            })
    }

    /// Send `request` (which must carry an `id`) and wait up to
    /// [`CODEX_APP_SERVER_TIMEOUT_SECS`] for its response, skipping notifications and anything
    /// else on the way. Returns the `result`.
    fn call(&mut self, request: &Value) -> Result<Value, String> {
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let id = request
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("codex app-server: request {} has no id", method))?;
        self.notify(request)?;
        let deadline = Instant::now() + Duration::from_secs(CODEX_APP_SERVER_TIMEOUT_SECS);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timed_out = || {
                format!(
                    "codex app-server: no response to {} within {} s",
                    method, CODEX_APP_SERVER_TIMEOUT_SECS
                )
            };
            if remaining.is_zero() {
                return Err(timed_out());
            }
            let line = match self.lines.recv_timeout(remaining) {
                Ok(Ok(line)) => line,
                Ok(Err(e)) => {
                    return Err(format!(
                        "codex app-server: read failed: {}{}",
                        e,
                        self.stderr_tail()
                    ))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => return Err(timed_out()),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(format!(
                        "codex app-server exited before answering {}{}",
                        method,
                        self.stderr_tail()
                    ))
                }
            };
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if let Some(outcome) = match_response(&msg, id) {
                return outcome.map_err(|e| format!("codex app-server {}: {}", method, e));
            }
        }
    }
}

impl Drop for AppServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// When `RETRIVIO_HOME` redirects this module to a scratch home, point the app-server at that
/// home's `.codex` too (`CODEX_HOME`), so the whole flow stays inside the scratch directory.
/// Otherwise the child inherits the environment, including any `CODEX_HOME` the user set, which
/// is the same directory [`codex_dir`] resolves for this module.
fn codex_home_override(home: &Path) -> Option<PathBuf> {
    match env::var("RETRIVIO_HOME") {
        Ok(v) if !v.trim().is_empty() => Some(Cli::Codex.dir(home)),
        _ => None,
    }
}

/// Explains an empty `hooks/list` when the app-server reads a different CODEX_HOME than the
/// directory this module manages.
fn describe_codex_home_mismatch(home: &Path, codex_home: Option<&str>) -> String {
    match codex_home {
        Some(reported) if Path::new(reported) != Cli::Codex.dir(home) => {
            format!("; codex app-server uses CODEX_HOME={}", reported)
        }
        _ => String::new(),
    }
}

/// Spawn, `initialize`, `initialized`. Returns the server and the `codexHome` it reported.
fn codex_app_server_connect(home: &Path) -> Result<(AppServer, Option<String>), String> {
    let override_home = codex_home_override(home);
    let mut server = AppServer::spawn(override_home.as_deref())?;
    let id = server.next_id();
    let init = server.call(&build_initialize_request(id))?;
    server.notify(&build_initialized_notification())?;
    let codex_home = init
        .get("codexHome")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok((server, codex_home))
}

fn codex_list_hooks(server: &mut AppServer, home: &Path) -> Result<Value, String> {
    let id = server.next_id();
    server.call(&build_hooks_list_request(id, home))
}

/// Read-only: for each binary in `bins`, our exact hooks and their trust status as Codex sees
/// them, plus the `codexHome` the app-server reported. `Err` = app-server unavailable (codex
/// missing, too old, or not answering).
fn codex_hook_trust_report(
    home: &Path,
    bins: &[String],
) -> Result<(Vec<(String, Vec<CodexHookTrust>)>, Option<String>), String> {
    let (mut server, codex_home) = codex_app_server_connect(home)?;
    let listed = codex_list_hooks(&mut server, home)?;
    let hooks_json = Cli::Codex.hooks_file(home);
    let report = bins
        .iter()
        .map(|bin| (bin.clone(), retrivio_codex_hooks(&listed, &hooks_json, bin)))
        .collect();
    Ok((report, codex_home))
}

enum TrustOutcome {
    /// Codex lists none of our hooks for this binary from the file we manage.
    NotFound {
        codex_home: Option<String>,
    },
    AlreadyTrusted(usize),
    Declined,
    Trusted(usize),
}

/// The `/hooks` trust flow for the hooks `hook install` writes for `bin`: list, select our exact
/// untrusted hooks, `config/batchWrite` their hashes, re-list and confirm each one by identity
/// (key, hash, source path, event), printing one `Codex trust: <event> <status>` line per hook.
/// `Err` when anything selected is not confirmed trusted, so `hook trust` exits non-zero.
/// `auto_yes` skips the confirmation (used by `hook install`, where the user already confirmed
/// the install).
fn codex_trust_hooks(home: &Path, bin: &str, auto_yes: bool) -> Result<TrustOutcome, String> {
    let hooks_json = Cli::Codex.hooks_file(home);
    let (mut server, codex_home) = codex_app_server_connect(home)?;
    let listed = codex_list_hooks(&mut server, home)?;
    let ours = retrivio_codex_hooks(&listed, &hooks_json, bin);
    if ours.is_empty() {
        return Ok(TrustOutcome::NotFound { codex_home });
    }
    let TrustSelection { pending, warnings } =
        select_untrusted_retrivio_hooks(&listed, &hooks_json, bin);
    for warning in &warnings {
        println!("  Codex trust: warning: {}", warning);
    }
    if pending.is_empty() {
        if warnings.is_empty() {
            return Ok(TrustOutcome::AlreadyTrusted(ours.len()));
        }
        return Err(format!(
            "{} event(s) list more than one retrivio hook in {}; nothing trusted (remove the duplicates, then rerun)",
            warnings.len(),
            hooks_json.display()
        ));
    }
    if !auto_yes && !interactive_terminal() {
        return Err(format!(
            "{} hook(s) need trusting but this is not a terminal - rerun with --yes",
            pending.len()
        ));
    }
    let question = format!(
        "  trust {} retrivio hook(s) in Codex (Codex writes [hooks.state] to {})?",
        pending.len(),
        Cli::Codex.dir(home).join("config.toml").display()
    );
    if !confirm(auto_yes, &question) {
        return Ok(TrustOutcome::Declined);
    }
    let id = server.next_id();
    server.call(&build_trust_write_request(id, &pending))?;
    let relisted = codex_list_hooks(&mut server, home)?;
    let mut failures = warnings.len();
    for (hook, outcome) in verify_trusted_hooks(&relisted, &pending) {
        let event = hook.event_name.as_str();
        match outcome {
            TrustVerification::Confirmed => println!("  Codex trust: {} trusted", event),
            TrustVerification::NotTrusted(status) => {
                println!("  Codex trust: {} {} (expected trusted)", event, status);
                failures += 1;
            }
            TrustVerification::DefinitionChanged => {
                println!(
                    "  Codex trust: {} not confirmed (definition changed between list and re-list; rerun `retrivio hook trust --codex`)",
                    event
                );
                failures += 1;
            }
            TrustVerification::NotListed => {
                println!("  Codex trust: {} not listed after the write", event);
                failures += 1;
            }
        }
    }
    if failures > 0 {
        return Err(format!(
            "{} retrivio hook(s) not confirmed trusted after config/batchWrite",
            failures
        ));
    }
    Ok(TrustOutcome::Trusted(pending.len()))
}

// ---------------------------------------------------------------------------
// `retrivio service`
// ---------------------------------------------------------------------------

pub fn run_service_cmd(args: &[OsString]) {
    let opts = match parse_common(
        args,
        &["install", "uninstall", "status", "run"],
        "status",
        false,
    ) {
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
        let detail = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        Err(format!(
            "launchctl {} failed ({}): {}",
            args.join(" "),
            output
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string()),
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
    let bin = env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    // SAFETY: installing a minimal async-signal-safe handler that only stores a flag.
    unsafe {
        libc::signal(libc::SIGTERM, service_on_signal as usize);
        libc::signal(libc::SIGINT, service_on_signal as usize);
    }
    let mut backoff: u64 = 60;
    while !SERVICE_STOP.load(Ordering::SeqCst) {
        let started = Instant::now();
        eprintln!("[{}] retrivio service: starting watch", unix_now());
        let mut cmd = Command::new(&bin);
        cmd.args(["watch", "--quiet", "--interval", WATCH_INTERVAL_SECS]);
        // The watch leads its own process group so a stop reaches everything it spawned
        // (fswatch) in one signal instead of orphaning it under launchd.
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[{}] retrivio service: failed to start watch: {}; retry in {}s",
                    unix_now(),
                    e,
                    backoff
                );
                service_sleep(backoff);
                backoff = (backoff * 2).min(900);
                continue;
            }
        };
        // Poll the child so a TERM/INT to the supervisor stops the whole watch process group
        // (TERM, then KILL after 3 s) and the child is always reaped.
        let outcome: Result<process::ExitStatus, String> = loop {
            if SERVICE_STOP.load(Ordering::SeqCst) {
                terminate_process_group(&mut child);
                eprintln!("[{}] retrivio service: stopped", unix_now());
                return Ok(());
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(500)),
                Err(e) => {
                    // The child's state is unknown: stop its group and treat this as a failed
                    // run, so the backoff applies rather than an immediate restart.
                    terminate_process_group(&mut child);
                    break Err(format!("watch wait failed: {}", e));
                }
            }
        };
        match outcome {
            Ok(status) => eprintln!(
                "[{}] retrivio service: watch exited ({}); retry in {}s",
                unix_now(),
                status,
                backoff
            ),
            Err(e) => eprintln!(
                "[{}] retrivio service: {}; retry in {}s",
                unix_now(),
                e,
                backoff
            ),
        }
        if started.elapsed() > Duration::from_secs(600) {
            backoff = 60;
        }
        service_sleep(backoff);
        backoff = (backoff * 2).min(900);
    }
    Ok(())
}

/// Stop `child` and everything in its process group: SIGTERM the group, wait up to 3 s for the
/// leader to exit, then SIGKILL the group and reap the leader. `child` must have been spawned
/// with `process_group(0)`, so its pid is the group id.
fn terminate_process_group(child: &mut process::Child) {
    let pgid = child.id() as libc::pid_t;
    // SAFETY: kill(2) with a negative pid signals that process group; no memory is involved.
    unsafe {
        libc::kill(-pgid, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
            _ => break,
        }
    }
    // SAFETY: as above.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
    let _ = child.wait();
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
    println!(
        "  program: {} service run  (supervises `watch --quiet --interval {}` with backoff)",
        bin.display(),
        WATCH_INTERVAL_SECS
    );
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
            println!(
                "  last exit status: {}",
                s.last_exit.as_deref().unwrap_or("-")
            );
        }
        Err(_) => println!("  loaded: no ({})", target),
    }
    let plist = plist_path(&home);
    println!(
        "  plist: {} ({})",
        plist.display(),
        if plist.is_file() {
            "present"
        } else {
            "missing"
        }
    );
    // Resolve fswatch against the PATH the agent really gets: the one in the installed plist,
    // or, without a plist, the one `service install` would write now.
    let (agent_path, path_source) = match fs::read_to_string(&plist)
        .ok()
        .and_then(|t| plist_path_env(&t))
    {
        Some(p) => (p, "the plist's PATH"),
        None => (service_path_env(), "the PATH `service install` would write"),
    };
    match find_on_path("fswatch", &agent_path) {
        Some(p) => println!("  fswatch: {} (on {})", p.display(), path_source),
        None => println!(
            "  fswatch: not found on {} ({}); the watcher falls back to polling",
            path_source, agent_path
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

/// The `PATH` value inside a rendered launchd plist (`EnvironmentVariables` -> `PATH`), XML
/// entities decoded; `None` when the plist carries none.
pub fn plist_path_env(plist: &str) -> Option<String> {
    const KEY: &str = "<key>PATH</key>";
    let after = &plist[plist.find(KEY)? + KEY.len()..];
    let start = after.find("<string>")? + "<string>".len();
    let end = after[start..].find("</string>")? + start;
    Some(xml_unescape(&after[start..end]))
}

/// Inverse of [`xml_escape`] (`&amp;` last, so `&amp;lt;` decodes to the literal `&lt;`).
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
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
            println!(
                "  systemctl --user daemon-reload && systemctl --user enable --now {}",
                SYSTEMD_UNIT_NAME
            );
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
                if unit_path.is_file() {
                    "present"
                } else {
                    "missing"
                }
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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Updated
        );
        assert_eq!(groups(&doc, "UserPromptSubmit").len(), 1);
        assert_eq!(groups(&doc, "SessionStart").len(), 1);
        assert_eq!(
            doc["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"],
            json!(BIN)
        );
        assert_eq!(
            doc["hooks"]["UserPromptSubmit"][0]["hooks"][0]["args"],
            json!(["recall"])
        );
        assert_eq!(
            doc["hooks"]["SessionStart"][0]["hooks"][0]["args"],
            json!(["recall", "--reset-session"])
        );
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Unchanged
        );
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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Installed
        );
        assert!(doc.is_object());
        assert!(retrivio_hook_status(&doc).installed());
    }

    #[test]
    fn merge_preserves_unrelated_hooks_and_keys() {
        let original = fixture_settings();
        let mut doc = original.clone();
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Installed
        );

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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Unchanged
        );
        assert_eq!(doc, snapshot);
    }

    #[test]
    fn merge_updates_bin_path_in_place() {
        let mut doc = fixture_settings();
        merge_claude_hooks(&mut doc, BIN).unwrap();
        let ss_before = groups(&doc, "SessionStart").len();
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN2).unwrap(),
            ChangeKind::Updated
        );
        assert_eq!(groups(&doc, "SessionStart").len(), ss_before);
        assert_eq!(groups(&doc, "UserPromptSubmit").len(), 1);
        let all: Vec<String> = commands(&doc, "UserPromptSubmit")
            .into_iter()
            .chain(commands(&doc, "SessionStart"))
            .collect();
        assert_eq!(all.iter().filter(|c| c.as_str() == BIN2).count(), 2);
        assert!(
            !all.iter().any(|c| c.contains(BIN)),
            "old bin still referenced: {:?}",
            all
        );
        assert_eq!(retrivio_hook_status(&doc).bins(), vec![BIN2.to_string()]);
    }

    #[test]
    fn merge_reports_updated_when_only_one_hook_was_missing() {
        let mut doc = json!({});
        merge_claude_hooks(&mut doc, BIN).unwrap();
        doc["hooks"].as_object_mut().unwrap().remove("SessionStart");
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Updated
        );
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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Updated
        );
        let ss = groups(&doc, "SessionStart");
        assert_eq!(ss.len(), 2);
        assert_eq!(ss[0]["matcher"], json!("compact"));
        assert_eq!(ss[0]["hooks"].as_array().unwrap().len(), 1);
        assert_eq!(ss[0]["hooks"][0]["command"], json!("/x/foreign.sh"));
        assert_eq!(ss[1]["matcher"], json!("compact|clear"));
        assert_eq!(ss[1]["hooks"][0]["command"], json!(BIN));
        assert_eq!(
            ss[1]["hooks"][0]["args"],
            json!(["recall", "--reset-session"])
        );
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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Updated
        );
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
        assert_eq!(
            merge_codex_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Installed
        );
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
        assert_eq!(
            merge_codex_hooks(&mut existing, BIN).unwrap(),
            ChangeKind::Installed
        );
        assert_eq!(existing["description"], json!("mine"));
        assert_eq!(
            existing["hooks"]["Stop"][0]["hooks"][0]["command"],
            json!("say hi")
        );
        assert_eq!(
            merge_codex_hooks(&mut existing, BIN).unwrap(),
            ChangeKind::Unchanged
        );
    }

    #[test]
    fn uninstall_from_codex_only_file_leaves_empty_hooks_object() {
        let mut doc = Value::Null;
        merge_codex_hooks(&mut doc, BIN).unwrap();
        assert!(remove_retrivio_hooks(&mut doc));
        assert_eq!(
            doc,
            json!({ "description": "Retrivio proactive recall hooks", "hooks": {} })
        );
    }

    #[test]
    fn status_reports_partial_and_absent() {
        let fixture = fixture_settings();
        let st = retrivio_hook_status(&fixture);
        assert!(!st.installed() && !st.partial());
        assert!(st.bins().is_empty());
        let mut doc = json!({});
        merge_claude_hooks(&mut doc, BIN).unwrap();
        doc["hooks"]
            .as_object_mut()
            .unwrap()
            .remove("UserPromptSubmit");
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
        assert_eq!(
            bin_from_command("\"/a b/retrivio\" recall").as_deref(),
            Some("/a b/retrivio")
        );
        assert_eq!(bin_from_command("something else"), None);
        assert!(!is_retrivio_recall_command(
            "/Users/me/.claude/hooks/context-budget.sh"
        ));
    }

    #[test]
    fn plist_contains_label_and_args() {
        let home = Path::new("/Users/me");
        let plist = render_launchd_plist("/Users/me/bin/retrivio", home);
        assert!(plist.contains("<string>com.stouffer-labs.retrivio.watch</string>"));
        assert!(plist.contains("<string>/Users/me/bin/retrivio</string>"));
        for arg in ["service", "run"] {
            assert!(
                plist.contains(&format!("<string>{}</string>", arg)),
                "missing {}",
                arg
            );
        }
        assert!(!plist.contains("<string>60</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>\n    <true/>"));
        assert!(plist.contains("<key>KeepAlive</key>\n    <true/>"));
        assert!(plist.contains("<key>ThrottleInterval</key>\n    <integer>60</integer>"));
        assert!(plist.contains("<key>ProcessType</key>\n    <string>Background</string>"));
        assert_eq!(
            plist
                .matches("<string>/Users/me/.retrivio/watch.log</string>")
                .count(),
            2
        );
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
        let extra_args =
            json!({ "type": "command", "command": BIN, "args": ["recall", "--verbose"] });
        let extra_shell = json!({ "type": "command", "command": format!("{} recall --reset-session --now", BIN) });
        let piped = json!({ "type": "command", "command": format!("{} recall | tee log", BIN) });
        let foreign: Vec<Value> = vec![
            proxy_shell,
            proxy_exec,
            wrapper,
            extra_args,
            extra_shell,
            piped,
        ];
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
        assert_eq!(
            merge_claude_hooks(&mut doc, BIN).unwrap(),
            ChangeKind::Installed
        );
        // The foreign groups are untouched; ours were appended as new groups.
        assert_eq!(
            doc["hooks"]["UserPromptSubmit"][0],
            before["hooks"]["UserPromptSubmit"][0]
        );
        assert_eq!(
            doc["hooks"]["SessionStart"][0],
            before["hooks"]["SessionStart"][0]
        );
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
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": "retrivio", "args": ["recall"] })
        ));
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": BIN, "args": ["recall", "--reset-session"] })
        ));
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": "retrivio recall" })
        ));
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": format!("  {}   recall   --reset-session  ", BIN) })
        ));
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": "'/Users/me/My Apps/retrivio' recall" })
        ));
        assert!(is_retrivio_entry(
            &json!({ "type": "command", "command": "\"/a b/retrivio\" recall" })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": BIN, "args": ["recall", 1] })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": BIN, "args": "recall" })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": BIN, "args": [] })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": BIN })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": "/x/Retrivio recall" })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": "/x/retrivio.sh recall" })
        ));
        assert!(!is_retrivio_entry(
            &json!({ "type": "command", "command": "'/x/retrivio'recall" })
        ));
        assert!(!is_retrivio_entry(&json!({ "type": "command" })));
        assert_eq!(
            bin_from_command("'/a'\\''b/retrivio' recall").as_deref(),
            Some("/a'b/retrivio")
        );
        assert_eq!(bin_from_command("'/unterminated/retrivio recall"), None);
        assert_eq!(
            bin_from_command("retrivio recall"),
            Some("retrivio".to_string())
        );
        assert_eq!(bin_from_command("retrivio recall now"), None);
        assert_eq!(
            split_first_token("  a b").map(|(t, r)| (t, r.to_string())),
            Some(("a".to_string(), " b".to_string()))
        );
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
        assert_eq!(
            write_json_atomic(&file, &json!({ "a": 1 }), &mut backed_up).unwrap(),
            None
        );
        assert!(!backed_up);
        assert_eq!(fs::read_to_string(&file).unwrap(), "{\n  \"a\": 1\n}\n");
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600,
            "new files are private"
        );
        #[cfg(unix)]
        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();
        let bak = write_json_atomic(&file, &json!({ "a": 2 }), &mut backed_up)
            .unwrap()
            .unwrap();
        assert!(backed_up);
        assert_eq!(bak, backup_path(&file));
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\n  \"a\": 1\n}\n");
        assert_eq!(fs::read_to_string(&file).unwrap(), "{\n  \"a\": 2\n}\n");
        #[cfg(unix)]
        {
            assert_eq!(
                fs::metadata(&file).unwrap().permissions().mode() & 0o777,
                0o644,
                "existing mode preserved"
            );
            assert_eq!(
                fs::metadata(&bak).unwrap().permissions().mode() & 0o777,
                0o644
            );
        }
        // A second write in the same run does not overwrite the backup.
        assert_eq!(
            write_json_atomic(&file, &json!({ "a": 3 }), &mut backed_up).unwrap(),
            None
        );
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\n  \"a\": 1\n}\n");
        let names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            names.iter().all(|n| !n.contains("tmp-retrivio")),
            "{:?}",
            names
        );
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
        let bak = write_json_atomic(&link, &json!({ "via": "link" }), &mut backed_up)
            .unwrap()
            .unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "link stays a link"
        );
        assert_eq!(
            fs::read_to_string(&real).unwrap(),
            "{\n  \"via\": \"link\"\n}\n"
        );
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
        let bak = write_json_atomic(&cfg, &json!({ "keep": false }), &mut false)
            .unwrap()
            .unwrap();
        assert_eq!(fs::read_to_string(&bak).unwrap(), "{\"keep\":true}\n");
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Codex trust via app-server --------------------------------------------------------

    const CODEX_HOOKS_JSON: &str = "/Users/me/.codex/hooks.json";
    const K_PROMPT: &str = "/Users/me/.codex/hooks.json:user_prompt_submit:0:0";
    const K_RESET: &str = "/Users/me/.codex/hooks.json:session_start:0:0";

    /// One `hooks/list` entry shaped like the real reply; `matcher` is what we write for the
    /// event, `command` is taken verbatim.
    fn codex_hook(
        key: &str,
        event: &str,
        command: &str,
        source_path: &str,
        status: &str,
        hash: &str,
    ) -> Value {
        json!({
            "key": key,
            "eventName": event,
            "handlerType": "command",
            "command": command,
            "async": false,
            "matcher": if event == "sessionStart" { json!("compact|clear") } else { Value::Null },
            "timeoutSec": 5,
            "statusMessage": if event == "userPromptSubmit" { json!("retrivio recall") } else { Value::Null },
            "additionalContextLimit": Value::Null,
            "sourcePath": source_path,
            "source": if status == "managed" { "managed" } else { "user" },
            "pluginId": Value::Null,
            "displayOrder": 0,
            "enabled": true,
            "isManaged": status == "managed",
            "currentHash": hash,
            "trustStatus": status
        })
    }

    fn hooks_list(hooks: Vec<Value>) -> Value {
        json!({ "data": [{ "cwd": "/Users/me", "hooks": hooks, "warnings": [], "errors": [] }] })
    }

    /// Exactly what `hook install` writes for BIN, as the app-server lists it.
    fn our_prompt(status: &str, hash: &str) -> Value {
        codex_hook(
            K_PROMPT,
            "userPromptSubmit",
            &format!("{} recall", BIN),
            CODEX_HOOKS_JSON,
            status,
            hash,
        )
    }

    fn our_reset(status: &str, hash: &str) -> Value {
        codex_hook(
            K_RESET,
            "sessionStart",
            &format!("{} recall --reset-session", BIN),
            CODEX_HOOKS_JSON,
            status,
            hash,
        )
    }

    /// Shaped like a real `hooks/list` result: ours (one untrusted, one modified), a managed
    /// hook, a user's own hook in our file, a retrivio-looking hook from another file and one
    /// for an event we never install.
    fn fixture_hooks_list() -> Value {
        hooks_list(vec![
            our_prompt("untrusted", "sha256:aaa"),
            our_reset("modified", "sha256:bbb"),
            codex_hook(
                "/etc/codex/managed_hooks.json:user_prompt_submit:0:0",
                "userPromptSubmit",
                "/opt/corp/audit --prompt",
                "/etc/codex/managed_hooks.json",
                "managed",
                "sha256:ccc",
            ),
            codex_hook(
                "/Users/me/.codex/hooks.json:user_prompt_submit:1:0",
                "userPromptSubmit",
                "/Users/me/bin/my-own-hook.sh",
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:ddd",
            ),
            codex_hook(
                "/Users/me/proj/.codex/hooks.json:user_prompt_submit:0:0",
                "userPromptSubmit",
                &format!("{} recall", BIN),
                "/Users/me/proj/.codex/hooks.json",
                "untrusted",
                "sha256:eee",
            ),
            codex_hook(
                "/Users/me/.codex/hooks.json:stop:0:0",
                "stop",
                &format!("{} recall", BIN),
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:fff",
            ),
        ])
    }

    fn selected_keys(sel: &TrustSelection) -> Vec<&str> {
        sel.pending.iter().map(|h| h.key.as_str()).collect()
    }

    fn select_one(hook: Value, bin: &str) -> TrustSelection {
        select_untrusted_retrivio_hooks(&hooks_list(vec![hook]), Path::new(CODEX_HOOKS_JSON), bin)
    }

    #[test]
    fn initialize_request_matches_app_server_handshake() {
        let req = build_initialize_request(1);
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], 1);
        assert_eq!(req["method"], "initialize");
        assert_eq!(req["params"]["clientInfo"]["name"], "retrivio");
        assert_eq!(
            req["params"]["clientInfo"]["version"],
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(
            req["params"]["clientInfo"]["title"],
            "retrivio hook install"
        );
        assert_eq!(req["params"]["capabilities"]["experimentalApi"], true);
        let note = build_initialized_notification();
        assert_eq!(note["method"], "initialized");
        assert!(note.get("id").is_none(), "a notification has no id");
        assert_eq!(note["params"], json!({}));
    }

    #[test]
    fn hooks_list_request_carries_home_as_cwd() {
        let req = build_hooks_list_request(2, Path::new("/Users/me"));
        assert_eq!(req["id"], 2);
        assert_eq!(req["method"], "hooks/list");
        assert_eq!(req["params"], json!({ "cwds": ["/Users/me"] }));
    }

    #[test]
    fn select_untrusted_picks_only_our_exact_untrusted_or_modified_hooks() {
        let hooks_json = Path::new(CODEX_HOOKS_JSON);
        let sel = select_untrusted_retrivio_hooks(&fixture_hooks_list(), hooks_json, BIN);
        assert!(sel.warnings.is_empty(), "{:?}", sel.warnings);
        assert_eq!(selected_keys(&sel), vec![K_PROMPT, K_RESET]);
        assert_eq!(sel.pending[0].current_hash, "sha256:aaa");
        assert_eq!(sel.pending[0].source_path, CODEX_HOOKS_JSON);
        assert_eq!(sel.pending[0].event_name, "userPromptSubmit");
        assert_eq!(sel.pending[1].current_hash, "sha256:bbb");
        // Path comparison is by components, so a trailing slash in the home does not matter.
        let via_join = PathBuf::from("/Users/me/")
            .join(".codex")
            .join("hooks.json");
        assert_eq!(
            select_untrusted_retrivio_hooks(&fixture_hooks_list(), &via_join, BIN)
                .pending
                .len(),
            2
        );
        // Another user's hooks.json, or hooks for a binary other than the one we install: none.
        assert!(select_untrusted_retrivio_hooks(
            &fixture_hooks_list(),
            Path::new("/Users/other/.codex/hooks.json"),
            BIN
        )
        .pending
        .is_empty());
        assert!(
            select_untrusted_retrivio_hooks(&fixture_hooks_list(), hooks_json, BIN2)
                .pending
                .is_empty()
        );
        assert!(retrivio_codex_hooks(&fixture_hooks_list(), hooks_json, BIN2).is_empty());
        // Already trusted: nothing pending and no warning, but still reported as ours.
        let trusted = hooks_list(vec![
            our_prompt("trusted", "sha256:aaa"),
            our_reset("trusted", "sha256:bbb"),
        ]);
        let sel = select_untrusted_retrivio_hooks(&trusted, hooks_json, BIN);
        assert!(sel.pending.is_empty() && sel.warnings.is_empty());
        assert_eq!(retrivio_codex_hooks(&trusted, hooks_json, BIN).len(), 2);
        // Malformed replies select nothing rather than panicking.
        assert_eq!(
            select_untrusted_retrivio_hooks(&json!({}), hooks_json, BIN),
            TrustSelection::default()
        );
        assert_eq!(
            select_untrusted_retrivio_hooks(
                &json!({ "data": [{ "hooks": [{ "key": 1 }] }] }),
                hooks_json,
                BIN
            ),
            TrustSelection::default()
        );
    }

    /// Every deviation from the exact generated form is somebody else's hook: never trusted.
    #[test]
    fn select_untrusted_rejects_everything_but_the_exact_generated_forms() {
        let prompt = |command: &str| {
            codex_hook(
                K_PROMPT,
                "userPromptSubmit",
                command,
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:aaa",
            )
        };
        let reset = |command: &str| {
            codex_hook(
                K_RESET,
                "sessionStart",
                command,
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:bbb",
            )
        };
        let exact_prompt = format!("{} recall", BIN);
        let exact_reset = format!("{} recall --reset-session", BIN);
        // Baseline: the two exact forms are selected.
        assert_eq!(select_one(prompt(&exact_prompt), BIN).pending.len(), 1);
        assert_eq!(select_one(reset(&exact_reset), BIN).pending.len(), 1);
        // Another binary named retrivio.
        assert!(select_one(prompt("/tmp/retrivio recall"), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt("retrivio recall"), BIN)
            .pending
            .is_empty());
        // A chained command whose first token is our binary, in either position.
        assert!(select_one(
            prompt("/known/retrivio;/tmp/retrivio recall"),
            "/known/retrivio"
        )
        .pending
        .is_empty());
        assert!(select_one(
            prompt("/known/retrivio recall; /tmp/retrivio recall"),
            "/known/retrivio"
        )
        .pending
        .is_empty());
        assert!(select_one(
            prompt("/known/retrivio recall && rm -rf ~"),
            "/known/retrivio"
        )
        .pending
        .is_empty());
        assert!(
            select_one(prompt("/known/retrivio recall $(id)"), "/known/retrivio")
                .pending
                .is_empty()
        );
        // The reset command on userPromptSubmit, the plain one on sessionStart, extra or
        // repeated arguments, and quoting or spacing we would not write.
        assert!(select_one(prompt(&exact_reset), BIN).pending.is_empty());
        assert!(select_one(reset(&exact_prompt), BIN).pending.is_empty());
        assert!(
            select_one(prompt(&format!("{} recall --verbose", BIN)), BIN)
                .pending
                .is_empty()
        );
        assert!(select_one(prompt(&format!("{} recall recall", BIN)), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt(&format!("'{}' recall", BIN)), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt(&format!("\"{}\" recall", BIN)), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt(&format!("{}  recall", BIN)), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt(&format!("{} recall ", BIN)), BIN)
            .pending
            .is_empty());
        assert!(select_one(prompt(&format!(" {} recall", BIN)), BIN)
            .pending
            .is_empty());
        // Matcher mismatch in either direction.
        let mut h = prompt(&exact_prompt);
        h["matcher"] = json!("compact|clear");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["matcher"] = json!("");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = reset(&exact_reset);
        h["matcher"] = Value::Null;
        assert!(select_one(h.clone(), BIN).pending.is_empty());
        h["matcher"] = json!("compact");
        assert!(select_one(h.clone(), BIN).pending.is_empty());
        h.as_object_mut().unwrap().remove("matcher");
        assert!(select_one(h, BIN).pending.is_empty());
        // Provenance and options: a plugin, a non-user source, another timeout, async, another
        // handler type, a missing source path.
        let mut h = prompt(&exact_prompt);
        h["pluginId"] = json!("some.plugin");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["source"] = json!("project");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["source"] = json!("managed");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["timeoutSec"] = json!(10);
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["timeoutSec"] = Value::Null;
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["async"] = json!(true);
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h["handlerType"] = json!("prompt");
        assert!(select_one(h, BIN).pending.is_empty());
        let mut h = prompt(&exact_prompt);
        h.as_object_mut().unwrap().remove("sourcePath");
        assert!(select_one(h, BIN).pending.is_empty());
        // Fields the app-server may omit are fine when they are absent rather than wrong.
        let mut h = prompt(&exact_prompt);
        for optional in [
            "async",
            "pluginId",
            "handlerType",
            "matcher",
            "statusMessage",
        ] {
            h.as_object_mut().unwrap().remove(optional);
        }
        assert_eq!(select_one(h, BIN).pending.len(), 1);
        // Our own quoting of an unusual path is byte-exact and accepted; the unquoted form is not.
        let spaced = "/Users/me/My Tools/retrivio";
        let quoted = codex_hook(
            K_PROMPT,
            "userPromptSubmit",
            &format!("'{}' recall", spaced),
            CODEX_HOOKS_JSON,
            "untrusted",
            "sha256:aaa",
        );
        assert_eq!(select_one(quoted, spaced).pending.len(), 1);
        let unquoted = codex_hook(
            K_PROMPT,
            "userPromptSubmit",
            &format!("{} recall", spaced),
            CODEX_HOOKS_JSON,
            "untrusted",
            "sha256:aaa",
        );
        assert!(select_one(unquoted, spaced).pending.is_empty());
        // A binary path that itself carries a shell metacharacter is quoted by shell_quote when
        // installed, but never auto-trusted.
        let odd = "/Users/me/a;b/retrivio";
        let odd_hook = codex_hook(
            K_PROMPT,
            "userPromptSubmit",
            &HookKind::Prompt.shell_command(odd),
            CODEX_HOOKS_JSON,
            "untrusted",
            "sha256:aaa",
        );
        assert!(select_one(odd_hook, odd).pending.is_empty());
    }

    #[test]
    fn select_untrusted_skips_an_event_with_two_exact_matches_and_names_both() {
        let hooks_json = Path::new(CODEX_HOOKS_JSON);
        let dup_key = "/Users/me/.codex/hooks.json:session_start:1:0";
        let list = hooks_list(vec![
            our_prompt("untrusted", "sha256:aaa"),
            our_reset("modified", "sha256:bbb"),
            codex_hook(
                dup_key,
                "sessionStart",
                &format!("{} recall --reset-session", BIN),
                CODEX_HOOKS_JSON,
                "trusted",
                "sha256:ggg",
            ),
        ]);
        let sel = select_untrusted_retrivio_hooks(&list, hooks_json, BIN);
        assert_eq!(
            selected_keys(&sel),
            vec![K_PROMPT],
            "the unambiguous event is still selected"
        );
        assert_eq!(sel.warnings.len(), 1);
        let w = &sel.warnings[0];
        assert!(w.starts_with("sessionStart:"), "{}", w);
        assert!(w.contains(K_RESET) && w.contains(dup_key), "{}", w);
        // Both events duplicated: nothing selected at all, two warnings.
        let list = hooks_list(vec![
            our_prompt("untrusted", "sha256:aaa"),
            codex_hook(
                "/Users/me/.codex/hooks.json:user_prompt_submit:2:0",
                "userPromptSubmit",
                &format!("{} recall", BIN),
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:hhh",
            ),
            our_reset("untrusted", "sha256:bbb"),
            codex_hook(
                dup_key,
                "sessionStart",
                &format!("{} recall --reset-session", BIN),
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:ggg",
            ),
        ]);
        let sel = select_untrusted_retrivio_hooks(&list, hooks_json, BIN);
        assert!(sel.pending.is_empty());
        assert_eq!(sel.warnings.len(), 2);
        assert!(
            sel.warnings[0].starts_with("userPromptSubmit:"),
            "{}",
            sel.warnings[0]
        );
        // A second hook for the same event that is NOT ours (other binary) is not a duplicate.
        let list = hooks_list(vec![
            our_reset("untrusted", "sha256:bbb"),
            codex_hook(
                dup_key,
                "sessionStart",
                &format!("{} recall --reset-session", BIN2),
                CODEX_HOOKS_JSON,
                "untrusted",
                "sha256:ggg",
            ),
        ]);
        let sel = select_untrusted_retrivio_hooks(&list, hooks_json, BIN);
        assert_eq!(selected_keys(&sel), vec![K_RESET]);
        assert!(sel.warnings.is_empty());
        // The same hook listed under two cwds is one match, not a duplicate.
        let mut doubled = fixture_hooks_list();
        let entry = doubled["data"][0].clone();
        doubled["data"].as_array_mut().unwrap().push(entry);
        let sel = select_untrusted_retrivio_hooks(&doubled, hooks_json, BIN);
        assert!(sel.warnings.is_empty());
        assert_eq!(sel.pending.len(), 2);
    }

    #[test]
    fn retrivio_codex_hooks_reports_status_and_dedupes_across_cwds() {
        let ours = retrivio_codex_hooks(&fixture_hooks_list(), Path::new(CODEX_HOOKS_JSON), BIN);
        let summary: Vec<(&str, &str)> = ours
            .iter()
            .map(|h| (h.event_name.as_str(), h.trust_status.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("userPromptSubmit", "untrusted"),
                ("sessionStart", "modified")
            ]
        );
        // The same hook listed under two cwds appears once.
        let mut doubled = fixture_hooks_list();
        let entry = doubled["data"][0].clone();
        doubled["data"].as_array_mut().unwrap().push(entry);
        assert_eq!(
            retrivio_codex_hooks(&doubled, Path::new(CODEX_HOOKS_JSON), BIN).len(),
            2
        );
    }

    #[test]
    fn trust_write_request_upserts_hooks_state() {
        let sel = select_untrusted_retrivio_hooks(
            &fixture_hooks_list(),
            Path::new(CODEX_HOOKS_JSON),
            BIN,
        );
        let req = build_trust_write_request(3, &sel.pending);
        assert_eq!(req["id"], 3);
        assert_eq!(req["method"], "config/batchWrite");
        assert_eq!(req["params"]["reloadUserConfig"], true);
        let edits = req["params"]["edits"].as_array().unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0]["keyPath"], "hooks.state");
        assert_eq!(edits[0]["mergeStrategy"], "upsert");
        assert_eq!(
            edits[0]["value"],
            json!({
                K_PROMPT: { "trusted_hash": "sha256:aaa" },
                K_RESET: { "trusted_hash": "sha256:bbb" }
            })
        );
    }

    #[test]
    fn verify_trusted_hooks_confirms_by_identity_not_by_key() {
        use TrustVerification::*;
        let pending = select_untrusted_retrivio_hooks(
            &fixture_hooks_list(),
            Path::new(CODEX_HOOKS_JSON),
            BIN,
        )
        .pending;
        assert_eq!(pending.len(), 2);
        let outcomes = |relisted: &Value| -> Vec<TrustVerification> {
            verify_trusted_hooks(relisted, &pending)
                .into_iter()
                .map(|(_, o)| o)
                .collect()
        };
        // Same key, hash, source path and event, now trusted: confirmed.
        let relisted = hooks_list(vec![
            our_prompt("trusted", "sha256:aaa"),
            our_reset("trusted", "sha256:bbb"),
        ]);
        assert_eq!(outcomes(&relisted), vec![Confirmed, Confirmed]);
        // Identity intact but Codex still reports another status.
        let relisted = hooks_list(vec![
            our_prompt("trusted", "sha256:aaa"),
            our_reset("modified", "sha256:bbb"),
        ]);
        assert_eq!(
            outcomes(&relisted),
            vec![Confirmed, NotTrusted("modified".to_string())]
        );
        // The hash changed between list and re-list (hooks.json edited meanwhile): the key alone
        // is not enough, even though Codex now says trusted.
        let relisted = hooks_list(vec![
            our_prompt("trusted", "sha256:aaa"),
            our_reset("trusted", "sha256:zzz"),
        ]);
        assert_eq!(outcomes(&relisted), vec![Confirmed, DefinitionChanged]);
        // Same key and hash but another source path or event: also a changed definition.
        let mut moved = our_reset("trusted", "sha256:bbb");
        moved["sourcePath"] = json!("/Users/me/proj/.codex/hooks.json");
        assert_eq!(
            outcomes(&hooks_list(vec![
                our_prompt("trusted", "sha256:aaa"),
                moved
            ])),
            vec![Confirmed, DefinitionChanged]
        );
        let mut renamed = our_reset("trusted", "sha256:bbb");
        renamed["eventName"] = json!("stop");
        assert_eq!(
            outcomes(&hooks_list(vec![
                our_prompt("trusted", "sha256:aaa"),
                renamed
            ])),
            vec![Confirmed, DefinitionChanged]
        );
        // Listed twice (two cwds), one of them identical: confirmed.
        let relisted = hooks_list(vec![
            our_prompt("trusted", "sha256:aaa"),
            our_reset("trusted", "sha256:bbb"),
            our_reset("trusted", "sha256:bbb"),
        ]);
        assert_eq!(outcomes(&relisted), vec![Confirmed, Confirmed]);
        // Gone entirely.
        let relisted = hooks_list(vec![our_prompt("trusted", "sha256:aaa")]);
        assert_eq!(outcomes(&relisted), vec![Confirmed, NotListed]);
        assert_eq!(outcomes(&json!({})), vec![NotListed, NotListed]);
        // The verified hooks are handed back so the caller can name the event.
        assert_eq!(
            verify_trusted_hooks(&relisted, &pending)[1].0.event_name,
            "sessionStart"
        );
    }

    #[test]
    fn match_response_skips_notifications_other_ids_and_server_requests() {
        assert!(match_response(
            &json!({ "method": "remoteControl/status/changed", "params": {} }),
            1
        )
        .is_none());
        assert!(match_response(&json!({ "id": 2, "result": {} }), 1).is_none());
        // A server-to-client request carries both id and method: not our reply even if the id collides.
        assert!(match_response(
            &json!({ "id": 1, "method": "item/commandExecution/requestApproval", "params": {} }),
            1
        )
        .is_none());
        assert!(match_response(&json!("not an object"), 1).is_none());
        assert_eq!(
            match_response(
                &json!({ "id": 1, "result": { "codexHome": "/Users/me/.codex" } }),
                1
            ),
            Some(Ok(json!({ "codexHome": "/Users/me/.codex" })))
        );
        assert_eq!(
            match_response(
                &json!({ "id": 1, "error": { "code": -32601, "message": "method not found" } }),
                1
            ),
            Some(Err("method not found (code -32601)".to_string()))
        );
        assert_eq!(
            match_response(&json!({ "id": 1, "error": "boom" }), 1),
            Some(Err("\"boom\"".to_string()))
        );
    }

    #[test]
    fn codex_dir_honours_codex_home_unless_retrivio_home_is_set() {
        let home = Path::new("/Users/me");
        assert_eq!(
            codex_dir(home, None, None),
            PathBuf::from("/Users/me/.codex")
        );
        assert_eq!(
            codex_dir(home, Some(""), None),
            PathBuf::from("/Users/me/.codex")
        );
        assert_eq!(
            codex_dir(home, Some("   "), None),
            PathBuf::from("/Users/me/.codex")
        );
        assert_eq!(
            codex_dir(home, Some("/Volumes/work/codex-home"), None),
            PathBuf::from("/Volumes/work/codex-home")
        );
        assert_eq!(
            codex_dir(home, Some(" /Volumes/work/codex-home "), None),
            PathBuf::from("/Volumes/work/codex-home")
        );
        // A scratch RETRIVIO_HOME wins (home is already the scratch directory then).
        let scratch = Path::new("/Users/me/proj/tmp/scratch-home2");
        let rh = scratch.to_str();
        assert_eq!(
            codex_dir(scratch, Some("/Volumes/work/codex-home"), rh),
            scratch.join(".codex")
        );
        assert_eq!(codex_dir(scratch, None, rh), scratch.join(".codex"));
        // An empty RETRIVIO_HOME is the same as unset.
        assert_eq!(
            codex_dir(home, Some("/Volumes/work/codex-home"), Some("")),
            PathBuf::from("/Volumes/work/codex-home")
        );
        assert_eq!(
            codex_dir(home, None, Some(" ")),
            PathBuf::from("/Users/me/.codex")
        );
    }

    // -- service: PATH policy, plist PATH, process group -----------------------------------

    #[test]
    fn service_path_skips_writable_shim_and_node_modules_dirs() {
        let dir = scratch("path");
        let mk = |name: &str, mode: u32| -> String {
            let p = dir.join(name);
            fs::create_dir_all(&p).unwrap();
            fs::set_permissions(&p, fs::Permissions::from_mode(mode)).unwrap();
            p.to_string_lossy().to_string()
        };
        let good = mk("bin", 0o755);
        let good2 = mk("tools", 0o700);
        let group_w = mk("group-writable", 0o775);
        let world_w = mk("world-writable", 0o777);
        let sticky = mk("sticky", 0o1777);
        let node_bin = mk("proj/node_modules/.bin", 0o755);
        let node_dir = mk("proj/node_modules", 0o755);
        let dot_bin = mk("proj/.bin", 0o755);
        let file = dir.join("not-a-dir");
        fs::write(&file, "").unwrap();

        assert!(service_path_dir_ok(&good));
        assert!(service_path_dir_ok(&good2));
        assert!(
            service_path_dir_ok(&format!("{}/", good)),
            "trailing slash is tolerated"
        );
        assert!(!service_path_dir_ok(&group_w), "group-writable");
        assert!(!service_path_dir_ok(&world_w), "world-writable");
        assert!(
            !service_path_dir_ok(&sticky),
            "sticky but still world-writable"
        );
        assert!(!service_path_dir_ok(&node_bin), "node_modules/.bin");
        assert!(!service_path_dir_ok(&node_dir), "node_modules itself");
        assert!(!service_path_dir_ok(&dot_bin), ".bin");
        assert!(!service_path_dir_ok(&file.to_string_lossy()), "a file");
        assert!(
            !service_path_dir_ok(&dir.join("missing").to_string_lossy()),
            "missing"
        );
        assert!(!service_path_dir_ok("relative/bin"));
        assert!(!service_path_dir_ok(""));
        assert!(!service_path_dir_ok("/tmp/x"));
        assert!(!service_path_dir_ok("/private/tmp/x"));
        assert!(!service_path_dir_ok("/var/folders/zz/x"));

        let rendered = service_path_env_from(&format!(
            "{}:{}:{}:{}:{}:/usr/bin::{}:{}",
            good, group_w, node_bin, dot_bin, world_w, good, good2
        ));
        let parts: Vec<&str> = rendered.split(':').collect();
        assert_eq!(
            &parts[..4],
            &["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        );
        assert_eq!(
            &parts[4..],
            &[good.as_str(), good2.as_str()],
            "kept once each, nothing else"
        );
        assert_eq!(service_path_env_from(""), LAUNCHD_PATH_ENV);

        // Make the scratch tree removable again.
        for p in [&group_w, &world_w, &sticky] {
            let _ = fs::set_permissions(p, fs::Permissions::from_mode(0o755));
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn plist_path_env_reads_back_the_rendered_path() {
        let plist = render_launchd_plist("/Users/me/bin/retrivio", Path::new("/Users/me"));
        assert_eq!(
            plist_path_env(&plist).as_deref(),
            Some(service_path_env().as_str())
        );
        let snippet = "<dict><key>HOME</key><string>/Users/me</string>\n<key>PATH</key>\n<string>/a&amp;b/bin:/x&lt;y/bin:/usr/bin</string></dict>";
        assert_eq!(
            plist_path_env(snippet).as_deref(),
            Some("/a&b/bin:/x<y/bin:/usr/bin")
        );
        assert_eq!(
            plist_path_env("<dict><key>HOME</key><string>/x</string></dict>"),
            None
        );
        assert_eq!(plist_path_env("<key>PATH</key><integer>3</integer>"), None);
        assert_eq!(plist_path_env(""), None);
        assert_eq!(xml_unescape("&amp;lt;"), "&lt;");
    }

    #[cfg(unix)]
    #[test]
    fn terminate_process_group_terms_then_kills_the_group() {
        use std::os::unix::process::ExitStatusExt;
        // A group that honours TERM is gone well inside the 3 s grace period.
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "sleep 30"]).process_group(0);
        let mut child = cmd.spawn().unwrap();
        let t = Instant::now();
        terminate_process_group(&mut child);
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        let status = child.try_wait().unwrap().expect("reaped");
        assert!(!status.success());
        // A leader that ignores TERM (its sleep children die, it loops on) is killed after 3 s.
        // It reports readiness on stdout so the signal cannot race the trap being installed.
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "trap '' TERM; echo ready; while :; do sleep 1; done"])
            .stdout(Stdio::piped())
            .process_group(0);
        let mut child = cmd.spawn().unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready.trim(), "ready");
        let t = Instant::now();
        terminate_process_group(&mut child);
        let elapsed = t.elapsed();
        assert!(
            elapsed >= Duration::from_secs(3) && elapsed < Duration::from_secs(10),
            "{:?}",
            elapsed
        );
        let status = child.try_wait().unwrap().expect("reaped");
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }

    #[test]
    fn hook_usage_and_dispatch_accept_trust() {
        assert!(HOOK_USAGE.contains("trust"));
        let opts = parse_common(
            &[
                OsString::from("trust"),
                OsString::from("--codex"),
                OsString::from("--yes"),
            ],
            &["install", "uninstall", "status", "trust"],
            "status",
            true,
        )
        .unwrap();
        assert_eq!(opts.sub, "trust");
        assert!(opts.codex && opts.yes && !opts.claude);
    }
}
