//! Amazon Bedrock backend: credential resolution and refresh, SigV4 signing, the preflight check and the embedder.

use std::collections::HashSet;
use std::io::{IsTerminal, Read};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs, process, thread};

use serde_json::Value;
use sha1::Digest;

use crate::config::ConfigValues;
use crate::embed::{
    bedrock_embedding_space_key, default_embed_model_for_backend, embed_metric_request_end,
    embed_metric_request_start, embed_metric_retry, embed_metric_texts, embed_metric_throttle,
    hook_mode_active, Embedder,
};
use crate::index::progress_clear_line;
use crate::setup::{
    aws_config_file_path, aws_credentials_file_path, aws_region_for_profile_name,
    parse_ini_profile_names,
};
use crate::util::{
    bool_env, command_available, non_empty_env, non_empty_string, now_ts, shell_escape,
    shell_split, tail_lines,
};

pub(crate) static BEDROCK_REQ_SEQ: AtomicU64 = AtomicU64::new(1);
pub(crate) static BEDROCK_REFRESH_ONCE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
pub(crate) static BEDROCK_REFRESH_FAILED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
pub(crate) static BEDROCK_PREFLIGHT_STATE: OnceLock<Mutex<Option<Result<(), String>>>> =
    OnceLock::new();
/// True after the first verbose Bedrock 5xx diagnostic has been emitted, so
/// repeat 5xx errors during a long index don't spam the terminal.
pub(crate) static BEDROCK_5XX_DIAG_LOGGED: AtomicBool = AtomicBool::new(false);

/// Known-good bedrock embedding models: (model_id, description).
pub(crate) const KNOWN_BEDROCK_EMBEDDING_MODELS: &[(&str, &str)] = &[
    (
        "amazon.titan-embed-text-v2:0",
        "default, 1024-dim, AWS native",
    ),
    ("cohere.embed-english-v3", "1024-dim, English-optimized"),
    ("cohere.embed-multilingual-v3", "1024-dim, multilingual"),
];

pub(crate) fn aws_cli_json(
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

/// Verify that Bedrock credentials are usable (run refresh cmd if configured,
/// then `sts get-caller-identity`). Caches the result per process so repeated
/// calls (MCP requests, indexing, etc.) don't re-run the refresh command on
/// every invocation.
///
/// On failure, returns a single actionable error message. Callers should
/// surface it verbatim and abort the current operation rather than falling
/// through to Bedrock calls that would fail in noisy ways.
pub(crate) fn bedrock_preflight_credentials(
    cfg: &ConfigValues,
    context: &str,
) -> Result<(), String> {
    if bool_env("RETRIVIO_BEDROCK_PREFLIGHT_SKIP", false) {
        return Ok(());
    }
    let lock = BEDROCK_PREFLIGHT_STATE.get_or_init(|| Mutex::new(None));
    // Hold the lock across the whole preflight so concurrent callers (e.g.
    // parallel MCP requests) don't stampede the refresh command. Only cache
    // successful preflights for the life of the process; failures are re-checked
    // on subsequent calls so the user can recover (e.g. after re-running mwinit)
    // without restarting Retrivio.
    let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(Ok(())) = guard.as_ref() {
        return Ok(());
    }
    let result = bedrock_preflight_credentials_inner(cfg, context);
    if result.is_ok() {
        *guard = Some(Ok(()));
    }
    result
}

pub(crate) fn bedrock_preflight_credentials_inner(
    cfg: &ConfigValues,
    context: &str,
) -> Result<(), String> {
    let region = bedrock_region_for_cfg(Some(cfg));
    let profile = bedrock_profile_for_cfg(Some(cfg));
    let credential_cmd = bedrock_credential_cmd_for_cfg(Some(cfg));

    // Allow the user to retry after fixing their environment (e.g. re-running
    // `mwinit`) without restarting Retrivio: clear the per-process failure latch
    // before each preflight attempt so the refresh command actually runs again.
    if let Ok(mut failed) = bedrock_refresh_failed_state().lock() {
        failed.clear();
    }

    if let Some(refresh_cmd) = bedrock_refresh_cmd_for_cfg(Some(cfg)) {
        // Also clear the success latch so a retry of the refresh command runs fresh.
        if let Ok(mut done) = bedrock_refresh_once_state().lock() {
            done.clear();
        }
        if let Err(err) = run_refresh_command_once(&refresh_cmd) {
            return Err(format_bedrock_preflight_error(
                context,
                &region,
                profile.as_deref(),
                &format!("credential refresh failed: {}", err),
            ));
        }
    }

    if let Some(cmd) = credential_cmd.as_deref() {
        // On-demand credential command: exec it, validate the JSON, and confirm
        // Expiration is in the future. Skip the `aws sts` probe because the AWS CLI
        // doesn't share Retrivio's in-memory creds.
        match AwsCredentials::resolve(profile.as_deref(), &bedrock_aws_cli_path(), Some(cmd)) {
            Ok(creds) => {
                if creds.is_near_expiry() {
                    return Err(format_bedrock_preflight_error(
                        context,
                        &region,
                        profile.as_deref(),
                        "aws_credential_cmd returned credentials that are already expired or expire within 5 minutes",
                    ));
                }
                Ok(())
            }
            Err(err) => Err(format_bedrock_preflight_error(
                context,
                &region,
                profile.as_deref(),
                &format!("aws_credential_cmd failed: {}", err),
            )),
        }
    } else {
        let aws_cli = bedrock_aws_cli_path();
        if !command_available(&aws_cli) {
            return Err(format!(
                "{}: AWS CLI '{}' not available/executable; install aws cli or set RETRIVIO_AWS_CLI",
                context, aws_cli
            ));
        }
        match aws_cli_json(
            &aws_cli,
            &region,
            profile.as_deref(),
            &["sts", "get-caller-identity"],
        ) {
            Ok(_) => Ok(()),
            Err(err) => Err(format_bedrock_preflight_error(
                context,
                &region,
                profile.as_deref(),
                &err,
            )),
        }
    }
}

pub(crate) fn format_bedrock_preflight_error(
    context: &str,
    region: &str,
    profile: Option<&str>,
    detail: &str,
) -> String {
    let profile_label = profile.unwrap_or("<default>");
    let lower = detail.to_ascii_lowercase();
    let looks_midway_stale = lower.contains("midway")
        || lower.contains("mwinit")
        || lower.contains("cookie")
        || lower.contains("unable to resolve midway");
    let primary_fix = if looks_midway_stale {
        "- midway cookie looks stale: `mwinit --ssh-public-key ~/.ssh/id_ecdsa.pub && ssh-add -K -t 72000`"
    } else {
        "- refresh AWS credentials (e.g. run your isengard/ada/aws-sso login)"
    };
    format!(
        "{}: Bedrock credentials are not usable (region={}, profile={}).\n\
         \n\
         detail: {}\n\
         \n\
         fixes:\n\
         {}\n\
         - then run `retrivio doctor --fix` to validate\n\
         - or switch to local embeddings with `retrivio config set embed_backend ollama`",
        context, region, profile_label, detail, primary_fix
    )
}

pub(crate) fn bedrock_profile_for_cfg(cfg: Option<&ConfigValues>) -> Option<String> {
    non_empty_env("RETRIVIO_AWS_PROFILE")
        .or_else(|| cfg.and_then(|c| non_empty_string(c.aws_profile.trim())))
        .or_else(|| non_empty_env("AWS_PROFILE"))
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

pub(crate) fn bedrock_region_for_cfg(cfg: Option<&ConfigValues>) -> String {
    if let Some(v) = non_empty_env("RETRIVIO_AWS_REGION") {
        return v;
    }
    if let Some(v) = cfg.and_then(|c| non_empty_string(c.aws_region.trim())) {
        return v;
    }
    if let Some(v) = non_empty_env("AWS_REGION").or_else(|| non_empty_env("AWS_DEFAULT_REGION")) {
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

pub(crate) fn bedrock_refresh_cmd_for_cfg(cfg: Option<&ConfigValues>) -> Option<String> {
    non_empty_env("RETRIVIO_AWS_REFRESH_CMD")
        .or_else(|| cfg.and_then(|c| non_empty_string(c.aws_refresh_cmd.trim())))
}

pub(crate) fn bedrock_credential_cmd_for_cfg(cfg: Option<&ConfigValues>) -> Option<String> {
    non_empty_env("RETRIVIO_AWS_CREDENTIAL_CMD")
        .or_else(|| cfg.and_then(|c| non_empty_string(c.aws_credential_cmd.trim())))
}

/// If `cmd` is the legacy `isengardcli add-profile EMAIL --role ROLE` shape,
/// return the equivalent on-demand `isengardcli credentials --awscli EMAIL --role ROLE`.
/// Returns None if `cmd` doesn't match the legacy shape.
pub(crate) fn migrate_isengard_add_profile_to_credential_cmd(cmd: &str) -> Option<String> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return None;
    }
    let tokens = shell_split(trimmed)?;
    let mut iter = tokens.iter();
    let bin = iter.next()?;
    let bin_basename = std::path::Path::new(bin)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| bin.to_string());
    if !bin_basename.contains("isengardcli") {
        return None;
    }
    let sub = iter.next()?;
    if sub != "add-profile" {
        return None;
    }
    let account = iter.next()?;
    let mut role = "Admin".to_string();
    while let Some(arg) = iter.next() {
        if arg == "--role" {
            if let Some(v) = iter.next() {
                role = v.clone();
            }
        }
    }
    Some(format!(
        "{} credentials --awscli {} --role {}",
        shell_escape(bin),
        shell_escape(account),
        shell_escape(&role)
    ))
}

pub(crate) fn bedrock_concurrency_for_cfg(cfg: Option<&ConfigValues>) -> usize {
    non_empty_env("RETRIVIO_BEDROCK_CONCURRENCY")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_concurrency as usize))
        .unwrap_or(32)
        .clamp(1, 128)
}

pub(crate) fn bedrock_max_retries_for_cfg(cfg: Option<&ConfigValues>) -> usize {
    non_empty_env("RETRIVIO_BEDROCK_MAX_RETRIES")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_max_retries as usize))
        .unwrap_or(3)
        .clamp(0, 12)
}

pub(crate) fn bedrock_retry_base_ms_for_cfg(cfg: Option<&ConfigValues>) -> u64 {
    non_empty_env("RETRIVIO_BEDROCK_RETRY_BASE_MS")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| cfg.map(|c| c.bedrock_retry_base_ms as u64))
        .unwrap_or(250)
        .clamp(50, 10_000)
}

pub(crate) fn bedrock_refresh_once_state() -> &'static Mutex<HashSet<String>> {
    BEDROCK_REFRESH_ONCE.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn bedrock_refresh_failed_state() -> &'static Mutex<HashSet<String>> {
    BEDROCK_REFRESH_FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn run_refresh_command_once(cmd: &str) -> Result<(), String> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    // `retrivio recall` runs as an editor hook: it must never spawn an (often interactive)
    // credential refresh. The semantic path then fails fast and the lexical fallback runs.
    if hook_mode_active() {
        return Err("refresh disabled in hook mode".to_string());
    }

    let always = bool_env("RETRIVIO_AWS_REFRESH_ALWAYS", false);
    if !always {
        if let Ok(done) = bedrock_refresh_once_state().lock() {
            if done.contains(trimmed) {
                return Ok(());
            }
        }
        // If we already failed this exact command once in this process, short-circuit silently
        // with a cached error rather than re-running it and re-spamming the terminal.
        if let Ok(failed) = bedrock_refresh_failed_state().lock() {
            if failed.contains(trimmed) {
                return Err("aws refresh command previously failed in this process; not retrying (run `retrivio doctor --fix` after resolving)".to_string());
            }
        }
    }

    let verbose = bool_env("RETRIVIO_AWS_REFRESH_VERBOSE", false);
    // If stdin is an interactive terminal, let the refresh command inherit
    // stdio so prompts (Midway PIN, SSO codes) still work for humans running
    // `retrivio index` directly. When headless (MCP server, piped, CI), capture
    // output and surface a compact summary on failure instead of spamming.
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let force_capture = bool_env("RETRIVIO_AWS_REFRESH_CAPTURE", false);
    let capture = force_capture || !interactive;

    let mut builder = Command::new("bash");
    builder.arg("-lc").arg(trimmed);
    if capture {
        builder
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
    }
    // Clear the progress line so any direct output (interactive mode) or our
    // follow-up diagnostics aren't interleaved with the spinner.
    progress_clear_line();

    if capture {
        let output = builder
            .output()
            .map_err(|e| format!("failed executing aws refresh command: {}", e))?;
        if !output.status.success() {
            if !always {
                if let Ok(mut failed) = bedrock_refresh_failed_state().lock() {
                    failed.insert(trimmed.to_string());
                }
            }
            let code = output
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = if !stderr.trim().is_empty() {
                stderr.to_string()
            } else {
                stdout.to_string()
            };
            let tail = tail_lines(combined.trim(), 6);
            let hint = if tail.is_empty() {
                String::new()
            } else {
                format!("\n  last output:\n    {}", tail.replace('\n', "\n    "))
            };
            return Err(format!(
                "aws credential refresh failed (exit {}){}",
                code, hint
            ));
        }
        if verbose {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stdout.trim().is_empty() {
                eprintln!("aws refresh stdout:\n{}", stdout.trim());
            }
            if !stderr.trim().is_empty() {
                eprintln!("aws refresh stderr:\n{}", stderr.trim());
            }
        }
    } else {
        let status = builder
            .status()
            .map_err(|e| format!("failed executing aws refresh command: {}", e))?;
        if !status.success() {
            if !always {
                if let Ok(mut failed) = bedrock_refresh_failed_state().lock() {
                    failed.insert(trimmed.to_string());
                }
            }
            return Err(format!(
                "aws credential refresh failed (exit {})",
                status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            ));
        }
    }

    if !always {
        if let Ok(mut done) = bedrock_refresh_once_state().lock() {
            done.insert(trimmed.to_string());
        }
    }
    Ok(())
}

pub(crate) fn refresh_aws_credentials_if_configured(
    cfg: Option<&ConfigValues>,
) -> Result<(), String> {
    let Some(cmd) = bedrock_refresh_cmd_for_cfg(cfg) else {
        return Ok(());
    };
    run_refresh_command_once(&cmd)
}

pub(crate) fn bedrock_aws_cli_path() -> String {
    non_empty_env("RETRIVIO_AWS_CLI").unwrap_or_else(|| "aws".to_string())
}

#[derive(Clone)]
pub(crate) struct AwsCredentials {
    pub(crate) access_key_id: String,
    pub(crate) secret_access_key: String,
    pub(crate) session_token: Option<String>,
    pub(crate) expires_at: Option<u64>,
}

/// Longest a non-interactive credential export may run inside `retrivio recall` (hook mode).
pub(crate) const HOOK_CREDENTIAL_CMD_TIMEOUT: Duration = Duration::from_millis(2500);

/// Run a credential-export command and capture its output. Stdin is always `/dev/null` (the
/// command must never prompt). In hook mode the wait is bounded by
/// [`HOOK_CREDENTIAL_CMD_TIMEOUT`]: the child is polled with `try_wait` and killed on timeout.
pub(crate) fn credential_command_output(
    cmd: &mut Command,
) -> std::io::Result<std::process::Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !hook_mode_active() {
        return cmd.output();
    }
    bounded_command_output(cmd, HOOK_CREDENTIAL_CMD_TIMEOUT)
}

/// Drain a child's pipe on a helper thread so a descendant that inherited the descriptor can
/// never block the caller; the bytes arrive on the returned channel when the pipe closes.
pub(crate) fn spawn_pipe_reader<R: Read + Send + 'static>(pipe: Option<R>) -> Receiver<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    match pipe {
        Some(mut reader) => {
            let _ = std::thread::Builder::new()
                .name("cred-pipe".to_string())
                .spawn(move || {
                    let mut buf = Vec::new();
                    let _ = reader.read_to_end(&mut buf);
                    let _ = tx.send(buf);
                });
        }
        None => {
            let _ = tx.send(Vec::new());
        }
    }
    rx
}

pub(crate) fn bounded_timeout_error(what: &str, timeout: Duration) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!(
            "credential command {} exceeded {} ms in hook mode",
            what,
            timeout.as_millis()
        ),
    )
}

/// `Command::output()` with a wall-clock bound: `try_wait` polling, kill + reap on timeout, and
/// pipe reads that also stop at the deadline (a grandchild holding the pipe cannot stall us).
/// The caller has already configured stdio (stdin null, stdout/stderr piped).
pub(crate) fn bounded_command_output(
    cmd: &mut Command,
    timeout: Duration,
) -> std::io::Result<std::process::Output> {
    let mut child = cmd.spawn()?;
    let started = Instant::now();
    let stdout_rx = spawn_pipe_reader(child.stdout.take());
    let stderr_rx = spawn_pipe_reader(child.stderr.take());
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(bounded_timeout_error("(killed)", timeout));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_rx
        .recv_timeout(timeout.saturating_sub(started.elapsed()))
        .map_err(|_| bounded_timeout_error("stdout", timeout))?;
    let stderr = stderr_rx
        .recv_timeout(timeout.saturating_sub(started.elapsed()))
        .map_err(|_| bounded_timeout_error("stderr", timeout))?;
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

impl AwsCredentials {
    pub(crate) fn resolve(
        profile: Option<&str>,
        aws_cli: &str,
        credential_cmd: Option<&str>,
    ) -> Result<Self, String> {
        let (stdout, source_label) = if let Some(cmd_str) =
            credential_cmd.map(|s| s.trim()).filter(|s| !s.is_empty())
        {
            let tokens = shell_split(cmd_str)
                .ok_or_else(|| format!("aws_credential_cmd has unbalanced quotes: {}", cmd_str))?;
            let (program, args) = tokens
                .split_first()
                .ok_or_else(|| "aws_credential_cmd is empty".to_string())?;
            let mut cmd = Command::new(program);
            cmd.args(args);
            let output = credential_command_output(&mut cmd)
                .map_err(|e| format!("aws_credential_cmd '{}': {}", program, e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let detail = if !stderr.trim().is_empty() {
                    stderr.trim().to_string()
                } else {
                    stdout.trim().to_string()
                };
                return Err(format!("aws_credential_cmd failed: {}", detail));
            }
            (output.stdout, "aws_credential_cmd")
        } else {
            let mut cmd = Command::new(aws_cli);
            cmd.arg("configure").arg("export-credentials");
            if let Some(p) = profile {
                cmd.arg("--profile").arg(p);
            }
            let output = credential_command_output(&mut cmd)
                .map_err(|e| format!("aws configure export-credentials: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "aws configure export-credentials failed: {}",
                    stderr.trim()
                ));
            }
            (output.stdout, "aws configure export-credentials")
        };
        let json: Value = serde_json::from_slice(&stdout)
            .map_err(|e| format!("{}: failed parsing credentials JSON: {}", source_label, e))?;
        let access_key = json["AccessKeyId"]
            .as_str()
            .ok_or_else(|| format!("{}: missing AccessKeyId", source_label))?
            .to_string();
        let secret_key = json["SecretAccessKey"]
            .as_str()
            .ok_or_else(|| format!("{}: missing SecretAccessKey", source_label))?
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

    pub(crate) fn is_near_expiry(&self) -> bool {
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

pub(crate) fn parse_iso8601_to_unix(s: &str) -> Option<u64> {
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

pub(crate) fn unix_to_amz_date(secs: u64) -> (String, String) {
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

pub(crate) fn hex_encode_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub(crate) fn sigv4_sha256_hex(data: &[u8]) -> String {
    let hash = <sha2::Sha256 as Digest>::digest(data);
    hex_encode_bytes(&hash)
}

pub(crate) fn sigv4_hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<sha2::Sha256>;
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

pub(crate) fn uri_encode_path_segment(s: &str) -> String {
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

pub(crate) fn sigv4_canonical_uri(url_path: &str) -> String {
    // SigV4 requires URI-encoding each path segment of the already-encoded URL path.
    // This means %3A in the URL becomes %253A in the canonical request (double-encoding).
    url_path
        .split('/')
        .map(uri_encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

pub(crate) fn sigv4_authorize(
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
pub(crate) struct BedrockEmbedder {
    model: String,
    region: String,
    profile: Option<String>,
    refresh_cmd: Option<String>,
    credential_cmd: Option<String>,
    normalize: bool,
    aws_cli: String,
    concurrency: usize,
    max_retries: usize,
    retry_base_ms: u64,
    credentials: Arc<Mutex<Option<AwsCredentials>>>,
    /// Single-flight gate so concurrent embed threads don't all spawn
    /// the credential command at once on cold start / refresh.
    credential_resolve_lock: Arc<Mutex<()>>,
    http_agent: ureq::Agent,
    /// Adaptive concurrency: decreases on throttle, recovers on success.
    active_concurrency: Arc<AtomicUsize>,
    /// Count of consecutive successful embed_many calls (no throttles).
    consecutive_ok: Arc<AtomicU64>,
}

impl BedrockEmbedder {
    #[cfg(test)]
    pub(crate) fn new(model: &str) -> Self {
        Self::new_with_config(model, None)
    }

    pub(crate) fn new_with_config(model: &str, cfg: Option<&ConfigValues>) -> Self {
        let model_name = if model.trim().is_empty() {
            default_embed_model_for_backend("bedrock").to_string()
        } else {
            model.trim().to_string()
        };
        let region = bedrock_region_for_cfg(cfg);
        let profile = bedrock_profile_for_cfg(cfg);
        let refresh_cmd = bedrock_refresh_cmd_for_cfg(cfg);
        let credential_cmd = bedrock_credential_cmd_for_cfg(cfg);
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
            credential_cmd,
            normalize,
            aws_cli,
            concurrency,
            max_retries,
            retry_base_ms,
            credentials: Arc::new(Mutex::new(None)),
            credential_resolve_lock: Arc::new(Mutex::new(())),
            http_agent,
            active_concurrency: Arc::new(AtomicUsize::new(concurrency)),
            consecutive_ok: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn request_payload(&self, text: &str) -> Value {
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

    pub(crate) fn parse_vector(data: &Value) -> Result<Vec<f32>, String> {
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
        // Fast path: cached creds still valid.
        if let Ok(guard) = self.credentials.lock() {
            if let Some(creds) = guard.as_ref() {
                if !creds.is_near_expiry() {
                    return Some(creds.clone());
                }
            }
        }
        // Slow path: serialize concurrent resolvers so cold-start with N
        // worker threads only spawns the credential command once.
        let _resolve_guard = self
            .credential_resolve_lock
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // Re-check inside the resolve gate: another thread may have populated.
        if let Ok(guard) = self.credentials.lock() {
            if let Some(creds) = guard.as_ref() {
                if !creds.is_near_expiry() {
                    return Some(creds.clone());
                }
            }
        }
        match AwsCredentials::resolve(
            self.profile.as_deref(),
            &self.aws_cli,
            self.credential_cmd.as_deref(),
        ) {
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
                // Capture diagnostic fields BEFORE consuming the body. Bedrock
                // surfaces the actual error in headers (x-amzn-RequestId,
                // x-amzn-ErrorType) — empty 5xx bodies often still carry these.
                let request_id = resp.header("x-amzn-RequestId").unwrap_or("").to_string();
                let error_type = resp.header("x-amzn-ErrorType").unwrap_or("").to_string();
                let body_text = resp.into_string().unwrap_or_default();
                // Never in hook mode: the body can echo the request (the derived query) and
                // the hook's stderr is the CLI's hook log.
                if code >= 500
                    && !hook_mode_active()
                    && !BEDROCK_5XX_DIAG_LOGGED.swap(true, Ordering::Relaxed)
                {
                    progress_clear_line();
                    eprintln!(
                        "bedrock 5xx diagnostic (first occurrence in this process):\n  model={}\n  region={}\n  http_status={}\n  x-amzn-RequestId={}\n  x-amzn-ErrorType={}\n  body={:?}",
                        self.model,
                        self.region,
                        code,
                        if request_id.is_empty() { "<missing>" } else { &request_id },
                        if error_type.is_empty() { "<missing>" } else { &error_type },
                        body_text.chars().take(500).collect::<String>(),
                    );
                }
                let mut detail = format!("HTTP {}", code);
                if !error_type.is_empty() {
                    detail.push_str(&format!(" {}", error_type));
                }
                if !body_text.trim().is_empty() {
                    detail.push_str(&format!(" - {}", body_text.trim()));
                }
                if !request_id.is_empty() {
                    detail.push_str(&format!(" (RequestId={})", request_id));
                }
                format!(
                    "Bedrock invoke failed (model='{}', region='{}'): {}",
                    self.model, self.region, detail
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

    pub(crate) fn invoke_model_cli(&self, payload: &Value) -> Result<Value, String> {
        // The AWS CLI may block on an interactive login and is not deadline-aware.
        if hook_mode_active() {
            return Err("aws cli fallback disabled in hook mode".to_string());
        }
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
                None => {
                    if self.credential_cmd.is_some() {
                        // With an explicit credential_cmd configured, the AWS CLI
                        // fallback won't see those creds — surface the resolution
                        // failure rather than producing a misleading SignatureV4 error.
                        Err("aws_credential_cmd did not return usable credentials".to_string())
                    } else {
                        self.invoke_model_cli(payload)
                    }
                }
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
                    // Bedrock occasionally returns empty-body 5xx responses
                    // mid-stream during long bulk index runs (e.g. HTTP 500/502/503).
                    // Treat these as retryable — they're transient backend hiccups,
                    // not auth or input failures.
                    let is_server_5xx = msg.contains("HTTP 500")
                        || msg.contains("HTTP 502")
                        || msg.contains("HTTP 503")
                        || msg.contains("HTTP 504");
                    let retryable = is_throttle
                        || is_server_5xx
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

    pub(crate) fn is_cohere_model(&self) -> bool {
        self.model.to_ascii_lowercase().contains("cohere.embed")
    }

    pub(crate) fn request_payload_batch(&self, texts: &[String]) -> Value {
        self.cohere_payload(texts, "search_document")
    }

    /// Cohere embed request; `input_type` is `search_document` for indexed text and
    /// `search_query` for queries (the model embeds the two sides differently).
    pub(crate) fn cohere_payload(&self, texts: &[String], input_type: &str) -> Value {
        serde_json::json!({
            "texts": texts,
            "input_type": input_type,
            "truncate": "END"
        })
    }

    pub(crate) fn parse_vectors(data: &Value) -> Result<Vec<Vec<f32>>, String> {
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
        bedrock_embedding_space_key(&self.model)
    }

    fn normalizes_output(&self) -> bool {
        self.normalize
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, String> {
        if !self.is_cohere_model() {
            return self.embed_one(text);
        }
        if let Some(cmd) = &self.refresh_cmd {
            run_refresh_command_once(cmd)?;
        }
        let payload = self.cohere_payload(&[text.to_string()], "search_query");
        let data = self.invoke_with_retry(&payload)?;
        let rows = Self::parse_vectors(&data)?;
        embed_metric_texts(1);
        rows.into_iter()
            .next()
            .ok_or_else(|| "No embedding returned.".to_string())
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
