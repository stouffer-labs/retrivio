//! End-to-end through the installed entry path: the real `retrivio` binary (`main`, argument
//! dispatch, SIGPIPE setup) in hook mode against a small offline store built through the
//! CLI itself (`embed_backend = "hash"`), fed the inputs a hook can receive. Each run must
//! exit within the hook deadline with a valid (possibly empty) output; the unit test
//! `recall::tests::hook_runs_finish_under_the_deadline_on_hostile_input` covers the same
//! inputs at the function level with more cases.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const HARD_DEADLINE: Duration = Duration::from_millis(4000);

fn work_dir() -> PathBuf {
    // Never the system temp dir: the workspace `tmp/` folder, as every other test.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tmp")
        .join(format!("hook-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("work dir");
    dir
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    wall: Duration,
}

/// `retrivio --data-dir <data> --config <config> <args>` with `stdin` piped in; `close_stdout`
/// drops the read end of the child's stdout before it can write (a CLI that went away).
fn run(
    data: &Path,
    config: &Path,
    cwd: &Path,
    args: &[&str],
    stdin: Option<&[u8]>,
    close_stdout: bool,
) -> Run {
    let started = Instant::now();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_retrivio"));
    cmd.arg("--data-dir")
        .arg(data)
        .arg("--config")
        .arg(config)
        .args(args)
        .current_dir(cwd)
        .env_remove("RETRIVIO_HOOK")
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn retrivio");
    if close_stdout {
        drop(child.stdout.take());
    }
    if let Some(bytes) = stdin {
        let mut sin = child.stdin.take().expect("stdin");
        let _ = sin.write_all(bytes);
    }
    let out = child.wait_with_output().expect("wait");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        wall: started.elapsed(),
    }
}

#[test]
fn the_installed_entry_path_finishes_hostile_hook_input_under_the_deadline() {
    let dir = work_dir();
    let data = dir.join("data");
    let config = dir.join("config.toml");
    let root = dir.join("corpus");
    let write = |rel: &str, text: &str| {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, text).unwrap();
    };
    write(
        "alpha/storage.md",
        "Alpha design notes: storage layers, the api endpoints and the retry budget. Decision 2026-09-01: keep the cache warm.",
    );
    write(
        "alpha/api.md",
        "Alpha api endpoints and their storage layers; pagination and the retry budget.",
    );
    write(
        "beta/auth.md",
        "Beta notes on authentication tokens and the login flow for the widget service.",
    );
    fs::write(
        &config,
        format!(
            "root = \"{}\"\nembed_backend = \"hash\"\nlocal_embed_dim = 64\nretrieval_backend = \"lancedb\"\nrecall_min_abs_score = 0.2\n",
            root.to_string_lossy()
        ),
    )
    .unwrap();
    let cwd = &root;
    let add = run(
        &data,
        &config,
        cwd,
        &["add", root.to_str().unwrap(), "--no-refresh"],
        None,
        false,
    );
    assert_eq!(add.code, Some(0), "add: {}\n{}", add.stdout, add.stderr);
    let index = run(&data, &config, cwd, &["index"], None, false);
    assert_eq!(
        index.code,
        Some(0),
        "index: {}\n{}",
        index.stdout,
        index.stderr
    );
    assert!(
        index.stdout.contains("chunks embedded: 3"),
        "index output:\n{}",
        index.stdout
    );

    let log_path = data.join("recall.log");
    let last_log = || -> String {
        fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .last()
            .unwrap_or("")
            .to_string()
    };
    let envelope = |prompt: &str, session: &str| -> Vec<u8> {
        format!(
            "{{\"prompt\": {}, \"session_id\": \"{}\", \"cwd\": {}, \"hook_event_name\": \"UserPromptSubmit\"}}",
            serde_json_string(prompt),
            session,
            serde_json_string(&root.to_string_lossy())
        )
        .into_bytes()
    };
    let mut timings: Vec<(String, u128)> = Vec::new();
    let mut check = |name: &str, r: &Run, expect_log: &str| {
        timings.push((name.to_string(), r.wall.as_millis()));
        assert!(
            r.wall < HARD_DEADLINE,
            "{}: {} ms; stderr:\n{}",
            name,
            r.wall.as_millis(),
            r.stderr
        );
        let json_lines: Vec<&str> = r.stdout.lines().filter(|l| l.starts_with('{')).collect();
        assert!(json_lines.len() <= 1, "{}: {:?}", name, json_lines);
        for line in &json_lines {
            assert!(
                line.contains("\"hookSpecificOutput\"") && line.contains("\"additionalContext\""),
                "{}: {}",
                name,
                line
            );
        }
        let log = last_log();
        assert!(
            log.contains(expect_log),
            "{}: log {:?} lacks {:?}; stderr:\n{}",
            name,
            log,
            expect_log,
            r.stderr
        );
    };

    // A content prompt through the real binary: exit 0 and a block naming the file.
    let content = run(
        &data,
        &config,
        cwd,
        &["recall"],
        Some(&envelope(
            "what did we decide about the storage layers and the cache",
            "e2e-content",
        )),
        false,
    );
    check("content", &content, " leads=");
    assert_eq!(content.code, Some(0), "{}", content.stderr);
    assert!(content.stdout.contains("storage.md"), "{}", content.stdout);

    // A malformed envelope: skipped as bad input, exit 0, nothing printed.
    let malformed = run(
        &data,
        &config,
        cwd,
        &["recall"],
        Some(br#"{"prompt": "tell me about storage layers", "session_id": "e2e-bad"#),
        false,
    );
    check("malformed-json", &malformed, "skipped:bad-input");
    assert_eq!(malformed.code, Some(0));
    assert!(malformed.stdout.trim().is_empty(), "{}", malformed.stdout);

    // An oversized envelope: truncated at 64 KiB, hence bad input, logged as truncated.
    let huge = envelope(&"storage layers ".repeat(6000), "e2e-huge");
    assert!(huge.len() > 64 * 1024);
    let oversize = run(&data, &config, cwd, &["recall"], Some(&huge), false);
    check("oversize-stdin", &oversize, "skipped:bad-input");
    assert!(last_log().ends_with("stdin:truncated"), "{}", last_log());

    // A 20 KB prompt: runs, bounded.
    let big_prompt = format!(
        "what did we decide about the storage layers {} and the retry budget",
        "pasted transcript line about api endpoints and widgets ".repeat(380)
    );
    let big = run(
        &data,
        &config,
        cwd,
        &["recall"],
        Some(&envelope(&big_prompt, "e2e-big")),
        false,
    );
    check("20kb-prompt", &big, " cand=");
    assert_eq!(big.code, Some(0), "{}", big.stderr);

    // The CLI went away (closed stdout): the process still ends within the deadline.
    let closed = run(
        &data,
        &config,
        cwd,
        &["recall"],
        Some(&envelope(
            "what did we decide about the storage layers and the cache",
            "e2e-closed",
        )),
        true,
    );
    let closed_ms = closed.wall.as_millis();
    assert!(
        closed.wall < HARD_DEADLINE,
        "closed stdout: {} ms",
        closed_ms
    );

    // A dry run in text format from the same binary.
    let dry = run(
        &data,
        &config,
        cwd,
        &[
            "recall",
            "--format",
            "text",
            "--query",
            "what did we decide about the storage layers and the cache",
            "--session",
            "e2e-dry",
        ],
        None,
        false,
    );
    check("dry-run", &dry, " leads=");
    assert!(dry.stdout.contains("<retrivio_leads>"), "{}", dry.stdout);
    assert!(dry.stdout.contains("storage.md"), "{}", dry.stdout);

    drop(check);
    timings.push(("closed-stdout".to_string(), closed_ms));
    println!("recall binary e2e timings (ms): {:?}", timings);
    let _ = fs::remove_dir_all(&dir);
}

/// Minimal JSON string encoding for the envelope (no serde in the integration test).
fn serde_json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
