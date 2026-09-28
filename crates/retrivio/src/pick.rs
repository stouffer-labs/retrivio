//! The jump and pick flows: the fzf-driven picker, jump-feed lines, candidate rendering for projects and files, and the emit-path handshake with the shell wrapper.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::{env, fs, process};

use rusqlite::{params_from_iter, Connection};

use crate::api::path_basename;
use crate::config::{config_path, db_path, load_config_values, ConfigValues};
use crate::db::{ensure_retrieval_backend_ready, open_db_rw, record_selection_event};
use crate::describe::{classify_doc_kind, extract_title};
use crate::describe_store::{
    describe_tables_present, load_file_meta, load_project_meta, StoredFileMeta, StoredProjectMeta,
};
use crate::embed::ensure_native_embed_backend;
use crate::freshness::format_ymd;
use crate::pick_view::{
    header_lines, layout_for_width, match_label, pick_preview_escape, render_dir_preview,
    render_dir_row, render_file_preview, render_file_row, snippet_around, why_phrases, DirPreview,
    DirRow, FilePreview, FileRow, RelatedDoc, RowLayout,
};
use crate::rank::{rank_files_native, rank_projects_native, RankedFileResult, RankedResult};
use crate::util::{arg_value, collapse_whitespace, command_exists, expand_tilde, now_ts};

pub(crate) fn run_jump_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio jump [--files|--dirs] [--view projects|files] [--limit <n>] [--emit-path-file <path>] [query...]"
        );
        println!("prints selected path to stdout (intended for shell wrappers to cd/open).");
        println!(
            "picker keys: Enter=select/requery, Tab=toggle dir/file, Ctrl-D=dirs, Ctrl-F=files, Ctrl-U=clear query"
        );
        println!(
            "rows: type | title | where | when | match. match words: words=exact terms, meaning=semantic, both, path, weak; seed / project / linked = graph relation (project = same project as a seed)."
        );
        println!("preview: title; kind · role · date · project · path; why; query-centred snippet; related document.");
        return;
    }

    let mut view = "projects".to_string();
    let mut limit: usize = 40;
    let mut emit_path_file = String::new();
    let mut query_parts: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--files" | "-f" => {
                view = "files".to_string();
            }
            "--dirs" | "-d" | "--directories" => {
                view = "projects".to_string();
            }
            "--view" => {
                i += 1;
                view = arg_value(args, i, "--view").to_lowercase();
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--emit-path-file" => {
                i += 1;
                emit_path_file = arg_value(args, i, "--emit-path-file");
            }
            x if x.starts_with("--view=") => {
                view = x.trim_start_matches("--view=").to_lowercase();
            }
            x if x.starts_with("--limit=") => {
                let raw = x.trim_start_matches("--limit=");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            x if x.starts_with("--emit-path-file=") => {
                emit_path_file = x.trim_start_matches("--emit-path-file=").to_string();
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
            other => {
                query_parts.push(other.to_string());
            }
        }
        i += 1;
    }

    if limit == 0 {
        limit = 1;
    }
    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }

    let query = query_parts.join(" ");

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "jump").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "jump").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!("hint: use `retrivio init --embed-backend <ollama|bedrock>`");
        process::exit(1);
    });

    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });

    // Non-interactive fallback: need a query and use old path.
    // Check stderr (not stdout) because shell wrappers like `r() { cd "$(retrivio "$@")" }`
    // capture stdout, making stdout().is_terminal() false even in interactive use.
    // fzf renders via /dev/tty, so it works fine with captured stdout.
    let cols = terminal_columns().unwrap_or(120);
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        if query.trim().is_empty() {
            eprintln!("usage: retrivio jump [--files|--dirs] [query...]");
            process::exit(2);
        }
        let layout = layout_for_width(usable_columns(cols));
        let candidates = if view == "files" {
            build_file_candidates(&conn, &cfg, &query, limit, layout)
        } else {
            build_project_candidates(&conn, &cfg, &query, limit, layout)
        }
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        if let Some(first) = candidates.first() {
            record_selection_event(&conn, &query, &first.path, now_ts()).ok();
            maybe_write_emit_path(&cwd, &emit_path_file, &first.path).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if emit_path_file.trim().is_empty() {
                println!("{}", first.path);
            }
        }
        return;
    }

    if !command_exists("fzf") {
        eprintln!(
            "error: `fzf` is required for interactive mode (fzf 0.35 or newer; live reflow on resize needs 0.46 or newer). install it or provide a query: retrivio [query]"
        );
        process::exit(2);
    }

    // Interactive live-search mode: fzf calls `jump-feed` on each keystroke
    let mut active_view = view;
    let mut active_query = query;
    loop {
        let action = pick_interactive_live(&active_view, &active_query, cols).unwrap_or_else(|e| {
            eprintln!("error: picker failed: {}", e);
            process::exit(1);
        });
        match action {
            PickAction::Cancel => process::exit(130),
            PickAction::Toggle { view, query } => {
                active_view = view;
                active_query = query;
            }
            PickAction::Refresh { query } => {
                active_query = query;
            }
            PickAction::Selected {
                path: selected_path,
                query,
            } => {
                let record_query = if query.trim().is_empty() {
                    active_query.clone()
                } else {
                    query
                };
                record_selection_event(&conn, &record_query, &selected_path, now_ts())
                    .unwrap_or_else(|e| {
                        eprintln!("warning: failed to record selection event: {}", e);
                    });
                maybe_write_emit_path(&cwd, &emit_path_file, &selected_path).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
                if emit_path_file.trim().is_empty() {
                    println!("{}", selected_path);
                }
                return;
            }
        }
    }
}

pub(crate) fn run_jump_feed_cmd(args: &[OsString]) {
    let mut view = "projects".to_string();
    let mut query_parts: Vec<String> = Vec::new();
    let mut limit: usize = 40;
    let mut width_flag: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].to_string_lossy().to_string();
        match a.as_str() {
            "--dirs" | "--projects" => view = "projects".to_string(),
            "--files" => view = "files".to_string(),
            "--limit" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    limit = v.to_string_lossy().parse().unwrap_or(40);
                }
            }
            "--width" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    width_flag = v.to_string_lossy().parse::<usize>().ok();
                }
            }
            other if !other.starts_with('-') => {
                query_parts.push(other.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    let query = query_parts.join(" ");
    let query = query.trim();
    if query.is_empty() {
        return;
    }
    let cols = feed_columns(env::var("FZF_COLUMNS").ok().as_deref(), width_flag);
    let layout = layout_for_width(usable_columns(cols));
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = match open_db_rw(&dbp) {
        Ok(c) => c,
        Err(_) => return,
    };
    // fzf's reload command: a ranking error must not print anything into the list.
    let candidates = if view == "files" {
        build_file_candidates(&conn, &cfg, query, limit, layout)
    } else {
        build_project_candidates(&conn, &cfg, query, limit, layout)
    }
    .unwrap_or_default();
    for item in &candidates {
        println!("{}", format_pick_candidate_line(item));
    }
}

pub(crate) fn maybe_write_emit_path(
    cwd: &Path,
    emit_path_file: &str,
    selected_path: &str,
) -> Result<(), String> {
    if emit_path_file.trim().is_empty() {
        return Ok(());
    }
    let mut out_path = expand_tilde(emit_path_file);
    if !out_path.is_absolute() {
        out_path = cwd.join(out_path);
    }
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create emit path parent '{}': {}",
                parent.display(),
                e
            )
        })?;
    }
    fs::write(&out_path, format!("{}\n", selected_path)).map_err(|e| {
        format!(
            "failed writing emit path file '{}': {}",
            out_path.display(),
            e
        )
    })?;
    Ok(())
}

pub(crate) fn run_pick_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "usage: retrivio pick [--query <text>] [--view projects|files] [--limit <n>] [--emit-path-file <path>]"
        );
        return;
    }

    let mut query = String::new();
    let mut view = "projects".to_string();
    let mut limit: usize = 30;
    let mut emit_path_file = String::new();
    let mut i = 0usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        match s.as_str() {
            "--query" => {
                i += 1;
                query = arg_value(args, i, "--query");
            }
            "--view" => {
                i += 1;
                view = arg_value(args, i, "--view").to_lowercase();
            }
            "--limit" => {
                i += 1;
                let raw = arg_value(args, i, "--limit");
                limit = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --limit must be an integer");
                    process::exit(2);
                });
            }
            "--emit-path-file" => {
                i += 1;
                emit_path_file = arg_value(args, i, "--emit-path-file");
            }
            other => {
                eprintln!("error: unknown option '{}'", other);
                process::exit(2);
            }
        }
        i += 1;
    }

    if limit == 0 {
        limit = 1;
    }
    if view != "projects" && view != "files" {
        eprintln!("error: --view must be one of: projects, files");
        process::exit(2);
    }

    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    let cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    ensure_retrieval_backend_ready(&cfg, true, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    ensure_native_embed_backend(&cfg, "pick").unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!("hint: use `retrivio init --embed-backend <ollama|bedrock>`");
        process::exit(1);
    });
    let dbp = db_path(&cwd);
    let conn = open_db_rw(&dbp).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    });
    let layout = layout_for_width(usable_columns(terminal_columns().unwrap_or(120)));
    let mut active_view = view;
    let mut active_query = query;
    let selected_path = loop {
        let candidates = if active_view == "files" {
            let rows = build_file_candidates(&conn, &cfg, &active_query, limit, layout)
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
            if rows.is_empty() {
                eprintln!("error: no file matches found.");
                process::exit(1);
            }
            rows
        } else {
            let rows = build_project_candidates(&conn, &cfg, &active_query, limit, layout)
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
            if rows.is_empty() {
                eprintln!("error: no indexed projects found. run `retrivio index` first.");
                process::exit(1);
            }
            rows
        };

        let selected = pick_candidate_path(&candidates, &active_query, &active_view)
            .unwrap_or_else(|e| {
                eprintln!("error: picker failed: {}", e);
                process::exit(1);
            });
        match selected {
            PickAction::Selected { path, .. } => break path,
            PickAction::Toggle { view, query } => {
                active_view = view;
                active_query = query;
            }
            PickAction::Refresh { query } => {
                active_query = query;
            }
            PickAction::Cancel => process::exit(130),
        }
    };
    if selected_path.is_empty() {
        process::exit(1);
    }

    if !emit_path_file.trim().is_empty() {
        let mut out_path = expand_tilde(&emit_path_file);
        if !out_path.is_absolute() {
            out_path = cwd.join(out_path);
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed to create emit path parent '{}': {}",
                    parent.display(),
                    e
                );
                process::exit(1);
            });
        }
        fs::write(&out_path, format!("{}\n", selected_path)).unwrap_or_else(|e| {
            eprintln!(
                "error: failed writing emit path file '{}': {}",
                out_path.display(),
                e
            );
            process::exit(1);
        });
    }

    record_selection_event(&conn, &active_query, &selected_path, now_ts()).unwrap_or_else(|e| {
        eprintln!("warning: failed to record selection event: {}", e);
    });
    println!("{}", selected_path);
}

/// One fzf line: the path fzf returns on Enter (hidden field 1), the rendered row (field 2)
/// and the escaped preview card (field 3).
pub(crate) struct PickCandidate {
    path: String,
    row: String,
    preview: String,
}

pub(crate) fn format_pick_candidate_line(item: &PickCandidate) -> String {
    format!("{}\t{}\t{}", item.path, item.row, item.preview)
}

/// Everything a candidate needs besides the ranked result, fetched once per `jump-feed` run
/// (a constant number of queries, whatever the result count).
pub(crate) struct PickContext<'a> {
    pub(crate) layout: RowLayout,
    pub(crate) query: &'a str,
    pub(crate) color: bool,
    pub(crate) now: f64,
    /// `doc_path` → stored title/kind. Empty when the store predates the tables.
    pub(crate) meta: HashMap<String, StoredFileMeta>,
    /// project path → synopsis, mtime, file count.
    pub(crate) project_meta: HashMap<String, StoredProjectMeta>,
    /// chunk id → text, document chunk count and code symbol, for snippets and code titles.
    pub(crate) texts: HashMap<i64, ChunkText>,
}

/// One chunk as the picker needs it: its text, how many chunks its document has, and the
/// symbol the AST chunker recorded (empty for prose).
pub(crate) struct ChunkText {
    pub(crate) text: String,
    pub(crate) total: i64,
    pub(crate) symbol_name: String,
    pub(crate) parent_context: String,
}

/// Columns for the rows: `FZF_COLUMNS` when fzf set it to a positive number (it is 0 inside
/// the `start` reload), else the `--width` the parent measured, else 120.
fn feed_columns(fzf_columns: Option<&str>, width_flag: Option<usize>) -> usize {
    match fzf_columns.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n,
        _ => width_flag.unwrap_or(120),
    }
}

/// Width of the controlling terminal, from the tty (stderr, stdout, stdin in turn), else
/// `COLUMNS`.
pub(crate) fn terminal_columns() -> Option<usize> {
    #[cfg(unix)]
    {
        for fd in [libc::STDERR_FILENO, libc::STDOUT_FILENO, libc::STDIN_FILENO] {
            // SAFETY: `winsize` is a plain C struct of four `u16` fields; all-zero is a valid
            // value.
            let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
            // SAFETY: TIOCGWINSZ writes a `winsize` into the pointer we pass; `ws` is a valid,
            // properly aligned `winsize` that outlives the call.
            let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws as *mut libc::winsize) };
            if rc == 0 && ws.ws_col > 0 {
                return Some(ws.ws_col as usize);
            }
        }
    }
    env::var("COLUMNS").ok().and_then(|v| v.trim().parse().ok())
}

/// The list area fzf leaves for a row: its columns minus the border and the pointer gutter.
fn usable_columns(cols: usize) -> usize {
    cols.saturating_sub(4)
}

/// fzf's `(major, minor)` from `fzf --version` (`0.74.1 (Homebrew)`), probed once per
/// process; `None` when fzf is missing or prints something else.
fn fzf_version() -> Option<(u32, u32)> {
    static VERSION: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *VERSION.get_or_init(|| {
        let out = Command::new("fzf").arg("--version").output().ok()?;
        parse_fzf_version(&String::from_utf8_lossy(&out.stdout))
    })
}

/// The first two numbers of a `fzf --version` line.
fn parse_fzf_version(output: &str) -> Option<(u32, u32)> {
    let mut parts = output.split_whitespace().next()?.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// The `resize` event exists from fzf 0.46; an older fzf rejects the bind (`unsupported key`,
/// exit 2), which the picker would read as a cancel. An unknown version gets no bind.
fn supports_resize_event(version: Option<(u32, u32)>) -> bool {
    matches!(version, Some(v) if v >= (0, 46))
}

/// Field 1 of a picker line is the raw path fzf hands back on Enter. A tab, newline or
/// carriage return would break the three-field record and `--ansi` would rewrite an escape,
/// so a candidate with such a path is not rendered at all (a reversible encoding of the field
/// is a follow-up).
pub(crate) fn path_is_fzf_safe(path: &str) -> bool {
    !path
        .chars()
        .any(|c| matches!(c, '\t' | '\n' | '\r' | '\u{1b}'))
}

/// The items whose path is fzf-safe, and how many were dropped. The count is not reported
/// anywhere yet: `jump-feed` is fzf's reload command and must print nothing but rows.
fn retain_fzf_safe<T>(items: Vec<T>, path_of: impl Fn(&T) -> &str) -> (Vec<T>, usize) {
    let before = items.len();
    let kept: Vec<T> = items
        .into_iter()
        .filter(|item| path_is_fzf_safe(path_of(item)))
        .collect();
    let dropped = before - kept.len();
    (kept, dropped)
}

fn chunk_texts(conn: &Connection, ids: &[i64]) -> HashMap<i64, ChunkText> {
    let mut out = HashMap::new();
    if ids.is_empty() {
        return out;
    }
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT pc.id, pc.text, (SELECT COUNT(*) FROM project_chunks c2 WHERE c2.doc_path = pc.doc_path), pc.symbol_name, pc.parent_context FROM project_chunks pc WHERE pc.id IN ({})",
        placeholders
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return out;
    };
    let Ok(rows) = stmt.query_map(params_from_iter(ids.iter()), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            ChunkText {
                text: row.get::<_, String>(1)?,
                total: row.get::<_, i64>(2)?,
                symbol_name: row.get::<_, String>(3)?,
                parent_context: row.get::<_, String>(4)?,
            },
        ))
    }) else {
        return out;
    };
    for (id, chunk) in rows.flatten() {
        out.insert(id, chunk);
    }
    out
}

fn raw_signals(pairs: &[(&str, f64)]) -> String {
    pairs
        .iter()
        .filter(|(_, v)| *v > 0.0005)
        .map(|(k, v)| format!("{} {:.2}", k, v))
        .collect::<Vec<_>>()
        .join("  ")
}

/// Title and kind for a document: the stored row, else what the describe pass would store
/// for an empty head (the humanised stem for prose, the full file name for config, data and
/// code) and the kind its extension implies.
fn title_and_kind<'a>(
    ctx: &'a PickContext<'_>,
    doc_path: &str,
    rel_path: &str,
) -> (String, &'a str) {
    match ctx.meta.get(doc_path) {
        Some(m) if !m.title.is_empty() => (m.title.clone(), m.doc_kind.as_str()),
        _ => (
            extract_title(rel_path, "", None).0,
            classify_doc_kind(rel_path, ""),
        ),
    }
}

pub(crate) fn make_project_pick_candidate(
    item: &RankedResult,
    ctx: &PickContext<'_>,
) -> PickCandidate {
    let name = path_basename(&item.path);
    let pm = ctx.project_meta.get(&item.path);
    let file_count = pm.map(|m| m.file_count).unwrap_or(0);
    let age_days = pm
        .map(|m| ((ctx.now - m.project_mtime) / 86_400.0).max(0.0))
        .unwrap_or(f64::NAN);
    let evidence: Vec<RelatedDoc> = item
        .evidence
        .iter()
        .take(3)
        .map(|ev| {
            let (title, _) = title_and_kind(ctx, &ev.doc_path, &ev.doc_rel_path);
            RelatedDoc {
                title,
                rel_path: ev.doc_rel_path.clone(),
                score: ev.score,
            }
        })
        .collect();
    let synopsis = pm.map(|m| m.synopsis.as_str()).unwrap_or("");
    let about = if !synopsis.is_empty() {
        synopsis.to_string()
    } else if let Some(top) = evidence.first() {
        format!("{} files · top: {}", file_count, top.title)
    } else {
        format!("{} files", file_count)
    };
    let relation = item
        .evidence
        .first()
        .map(|e| e.relation.as_str())
        .unwrap_or("");
    let label = match_label(item.semantic, item.lexical, relation, "");
    let row = render_dir_row(
        &DirRow {
            name: &name,
            about: &about,
            age_days,
            match_label: &label,
        },
        &ctx.layout,
        ctx.color,
    );
    let why = why_phrases(
        item.semantic,
        item.lexical,
        relation,
        "",
        1.0,
        item.frecency,
        "",
    );
    let raw = raw_signals(&[
        ("s", item.semantic),
        ("l", item.lexical),
        ("f", item.frecency),
        ("g", item.graph),
    ]);
    let snippet = match item.evidence.first() {
        Some(ev) => {
            let text = ctx
                .texts
                .get(&ev.chunk_id)
                .map(|c| c.text.as_str())
                .unwrap_or(ev.excerpt.as_str());
            snippet_around(text, ctx.query, 110)
        }
        None => Vec::new(),
    };
    let preview = render_dir_preview(&DirPreview {
        name: &name,
        synopsis,
        file_count,
        age_days,
        why: &why,
        raw_signals: &raw,
        evidence: &evidence,
        snippet: &snippet,
    });
    PickCandidate {
        path: item.path.clone(),
        row,
        preview,
    }
}

pub(crate) fn make_file_pick_candidate(
    item: &RankedFileResult,
    ctx: &PickContext<'_>,
) -> PickCandidate {
    let rel = if item.doc_rel_path.trim().is_empty() {
        path_basename(&item.path)
    } else {
        item.doc_rel_path.clone()
    };
    let (mut title, kind) = title_and_kind(ctx, &item.path, &rel);
    let chunk = ctx.texts.get(&item.chunk_id);
    // Code: the matched symbol is the title (`parent::symbol · file.rs`), not a stored one.
    if kind == "code" {
        if let Some(c) = chunk {
            if !c.symbol_name.is_empty() {
                let file = path_basename(&item.path);
                title = if c.parent_context.is_empty() {
                    format!("{} · {}", c.symbol_name, file)
                } else {
                    format!("{}::{} · {}", c.parent_context, c.symbol_name, file)
                };
            }
        }
    }
    let project = path_basename(&item.project_path);
    let label = match_label(item.semantic, item.lexical, &item.relation, &item.why);
    let row = render_file_row(
        &FileRow {
            kind,
            title: &title,
            project: &project,
            rel_path: &rel,
            age_days: item.age_days,
            match_label: &label,
        },
        &ctx.layout,
        ctx.color,
    );
    let why = why_phrases(
        item.semantic,
        item.lexical,
        &item.relation,
        &item.freshness_tier,
        item.quality,
        0.0,
        &item.why,
    );
    let raw = raw_signals(&[
        ("s", item.semantic),
        ("l", item.lexical),
        ("g", item.graph),
        ("q", item.quality),
    ]);
    let (text, total) = chunk
        .map(|c| (c.text.as_str(), c.total))
        .unwrap_or((item.excerpt.as_str(), item.chunk_index + 1));
    let snippet = snippet_around(text, ctx.query, 110);
    let related = item
        .evidence
        .iter()
        .find(|ev| ev.doc_path != item.path)
        .map(|ev| {
            let (t, _) = title_and_kind(ctx, &ev.doc_path, &ev.doc_rel_path);
            RelatedDoc {
                title: t,
                rel_path: ev.doc_rel_path.clone(),
                score: ev.score,
            }
        });
    let date = if item.content_date > 0.0 {
        format_ymd(item.content_date)
    } else {
        String::new()
    };
    let preview = render_file_preview(&FilePreview {
        title: &title,
        kind,
        role: item.role,
        date_ymd: &date,
        age_days: item.age_days,
        project: &project,
        rel_path: &rel,
        why: &why,
        raw_signals: &raw,
        snippet: &snippet,
        chunk_index: item.chunk_index,
        chunk_total: total,
        related: related.as_ref(),
    });
    PickCandidate {
        path: item.path.clone(),
        row,
        preview,
    }
}

/// The project an evidence hit belongs to: its `doc_path` minus its `doc_rel_path` (file
/// evidence may come from a linked project, not the result's own), else `fallback`.
fn evidence_project(doc_path: &str, doc_rel_path: &str, fallback: &str) -> String {
    if doc_rel_path.is_empty() {
        return fallback.to_string();
    }
    doc_path
        .strip_suffix(doc_rel_path)
        .map(|p| p.trim_end_matches('/'))
        .filter(|p| !p.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// File candidates for `query`, ready for fzf. A constant number of queries whatever the
/// result count: the ranking, project ids + `file_meta` rows, and the chunk texts. Results
/// whose path is not fzf-safe ([`path_is_fzf_safe`]) are left out. The error is the ranking's
/// (a blocked re-embed, an embedding failure); callers decide whether to show it.
fn build_file_candidates(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
    layout: RowLayout,
) -> Result<Vec<PickCandidate>, String> {
    let (results, _unsafe_dropped) =
        retain_fzf_safe(rank_files_native(conn, cfg, query, limit)?, |r| {
            r.path.as_str()
        });
    let mut keys: Vec<(String, String)> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();
    for r in &results {
        keys.push((r.project_path.clone(), r.doc_rel_path.clone()));
        ids.push(r.chunk_id);
        for ev in &r.evidence {
            keys.push((
                evidence_project(&ev.doc_path, &ev.doc_rel_path, &r.project_path),
                ev.doc_rel_path.clone(),
            ));
        }
    }
    let meta = if describe_tables_present(conn) {
        load_file_meta(conn, &keys).unwrap_or_default()
    } else {
        HashMap::new()
    };
    let ctx = PickContext {
        layout,
        query,
        color: true,
        now: now_ts(),
        meta,
        project_meta: HashMap::new(),
        texts: chunk_texts(conn, &ids),
    };
    Ok(results
        .iter()
        .map(|r| make_file_pick_candidate(r, &ctx))
        .collect())
}

/// Project candidates for `query`, ready for fzf. A constant number of queries whatever the
/// result count: the ranking, project ids + `file_meta` rows for the evidence titles, the
/// chunk texts, and the `project_meta` rows. Results whose path is not fzf-safe
/// ([`path_is_fzf_safe`]) are left out. The error is the ranking's.
fn build_project_candidates(
    conn: &Connection,
    cfg: &ConfigValues,
    query: &str,
    limit: usize,
    layout: RowLayout,
) -> Result<Vec<PickCandidate>, String> {
    let (results, _unsafe_dropped) =
        retain_fzf_safe(rank_projects_native(conn, cfg, query, limit)?, |r| {
            r.path.as_str()
        });
    let paths: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
    let mut keys: Vec<(String, String)> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();
    for r in &results {
        for ev in r.evidence.iter().take(3) {
            keys.push((r.path.clone(), ev.doc_rel_path.clone()));
        }
        if let Some(ev) = r.evidence.first() {
            ids.push(ev.chunk_id);
        }
    }
    let tables = describe_tables_present(conn);
    let meta = if tables {
        load_file_meta(conn, &keys).unwrap_or_default()
    } else {
        HashMap::new()
    };
    let ctx = PickContext {
        layout,
        query,
        color: true,
        now: now_ts(),
        meta,
        project_meta: load_project_meta(conn, &paths, tables).unwrap_or_default(),
        texts: chunk_texts(conn, &ids),
    };
    Ok(results
        .iter()
        .map(|r| make_project_pick_candidate(r, &ctx))
        .collect())
}

pub(crate) enum PickAction {
    Selected { path: String, query: String },
    Toggle { view: String, query: String },
    Refresh { query: String },
    Cancel,
}

pub(crate) fn normalize_pick_query(query: &str) -> String {
    collapse_whitespace(query).to_ascii_lowercase()
}

pub(crate) fn pick_query_changed(original: &str, updated: &str) -> bool {
    normalize_pick_query(original) != normalize_pick_query(updated)
}

/// The live picker: fzf calls `jump-feed` on every keystroke (and, from fzf 0.46, on resize)
/// with the width the parent measured, so rows are laid out for the terminal even inside the
/// `start` reload, where `FZF_COLUMNS` is 0.
pub(crate) fn pick_interactive_live(
    view: &str,
    initial_query: &str,
    cols: usize,
) -> Result<PickAction, String> {
    let bin = env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "retrivio".to_string());
    let view_flag = if view == "files" { "--files" } else { "--dirs" };
    // Shell-escape the binary path (handle spaces/quotes)
    let escaped_bin = bin.replace('\'', "'\\''");
    let feed_cmd = format!(
        "'{}' jump-feed {} --width {} {{q}}",
        escaped_bin, view_flag, cols
    );
    let layout = layout_for_width(usable_columns(cols));
    let header = header_lines(&layout, view);
    let prompt = if view == "files" {
        "retrivio[file]> "
    } else {
        "retrivio[dir]> "
    };
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--ansi")
        .arg("--delimiter=\t")
        .arg("--with-nth=2")
        .arg("--no-sort")
        .arg("--disabled")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg(header)
        .arg("--preview")
        .arg("printf '%b' {3}")
        .arg("--preview-window=down,8,wrap")
        .arg("--bind")
        .arg(format!("start:reload({})", feed_cmd))
        .arg("--bind")
        .arg(format!("change:reload({})", feed_cmd));
    if supports_resize_event(fzf_version()) {
        cmd.arg("--bind")
            .arg(format!("resize:reload({})", feed_cmd));
    }
    cmd.arg("--print-query")
        .arg("--expect=tab,ctrl-d,ctrl-f")
        .arg("--bind")
        .arg("ctrl-u:unix-line-discard");
    if !initial_query.is_empty() {
        cmd.arg("--query").arg(initial_query);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());
    let out = cmd
        .output()
        .map_err(|e| format!("failed to launch fzf: {}", e))?;
    let output = String::from_utf8_lossy(&out.stdout);
    let mut lines = output.lines();
    let query_line = lines.next().unwrap_or("").trim().to_string();
    let effective_query = if query_line.trim().is_empty() {
        initial_query.to_string()
    } else {
        query_line
    };
    let second_line = lines.next().unwrap_or("").trim().to_string();
    let mut key_line = second_line.clone();
    let mut selected_line = String::new();
    if second_line.contains('\t') {
        key_line.clear();
        selected_line = second_line;
    }
    if key_line == "tab" {
        let toggled = if view == "files" { "projects" } else { "files" };
        return Ok(PickAction::Toggle {
            view: toggled.to_string(),
            query: effective_query,
        });
    }
    if key_line == "ctrl-d" {
        return Ok(PickAction::Toggle {
            view: "projects".to_string(),
            query: effective_query,
        });
    }
    if key_line == "ctrl-f" {
        return Ok(PickAction::Toggle {
            view: "files".to_string(),
            query: effective_query,
        });
    }
    if !out.status.success() {
        return Ok(PickAction::Cancel);
    }
    if selected_line.is_empty() {
        selected_line = lines
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .to_string();
    }
    if selected_line.trim().is_empty() {
        return Ok(PickAction::Cancel);
    }
    let path = selected_line
        .split('\t')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if path.is_empty() {
        return Ok(PickAction::Cancel);
    }
    Ok(PickAction::Selected {
        path,
        query: effective_query,
    })
}

pub(crate) fn pick_candidate_path(
    candidates: &[PickCandidate],
    query: &str,
    view: &str,
) -> Result<PickAction, String> {
    if candidates.is_empty() {
        return Ok(PickAction::Cancel);
    }
    if !std::io::stdin().is_terminal() {
        return Ok(PickAction::Selected {
            path: candidates[0].path.clone(),
            query: query.to_string(),
        });
    }
    if !command_exists("fzf") {
        eprintln!(
            "warning: `fzf` is not installed; interactive picker UI is unavailable. selecting the top-ranked {} match automatically.",
            if view == "files" { "file" } else { "directory" }
        );
        eprintln!(
            "hint: install `fzf` to enable interactive dir/file switching, previews, and manual selection."
        );
        return Ok(PickAction::Selected {
            path: candidates[0].path.clone(),
            query: query.to_string(),
        });
    }

    let mut payload = String::new();
    for item in candidates {
        payload.push_str(&format!("{}\t{}\t{}\n", item.path, item.row, item.preview));
    }
    let prompt = if view == "files" {
        "retrivio[file]> ".to_string()
    } else {
        "retrivio[dir]> ".to_string()
    };
    let header = header_lines(
        &layout_for_width(usable_columns(terminal_columns().unwrap_or(120))),
        view,
    );
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--ansi")
        .arg("--delimiter=\t")
        // Show and match on the rendered row (field 2) only; field 3 is the escaped preview
        // card, whose `\033[..m` sequences must not be searchable.
        .arg("--with-nth=2")
        .arg("--nth=2")
        .arg("--print-query")
        .arg("--expect=tab,ctrl-d,ctrl-f")
        .arg("--preview")
        .arg("printf '%b' {3}")
        .arg("--preview-window=down,8,wrap")
        .arg("--bind")
        .arg("ctrl-u:unix-line-discard")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg(header)
        .arg("--no-sort")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch fzf: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| format!("failed writing picker input: {}", e))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("failed waiting for fzf: {}", e))?;
    let output = String::from_utf8_lossy(&out.stdout);
    let mut lines = output.lines();
    let query_line = lines.next().unwrap_or("").trim().to_string();
    let effective_query = if query_line.trim().is_empty() {
        query.to_string()
    } else {
        query_line.clone()
    };
    let second_line = lines.next().unwrap_or("").trim().to_string();
    let mut key_line = second_line.clone();
    let mut selected_line = String::new();
    if second_line.contains('\t') {
        // Some fzf versions omit an empty expect-key line for Enter.
        key_line.clear();
        selected_line = second_line;
    }
    // Check expected toggle keys BEFORE exit status — fzf may exit 1 (no match)
    // when the user presses an --expect key with no visible matches, but still
    // prints the key to stdout.
    if key_line == "tab" {
        let toggled = if view == "files" { "projects" } else { "files" };
        return Ok(PickAction::Toggle {
            view: toggled.to_string(),
            query: effective_query.clone(),
        });
    }
    if key_line == "ctrl-d" {
        return Ok(PickAction::Toggle {
            view: "projects".to_string(),
            query: effective_query.clone(),
        });
    }
    if key_line == "ctrl-f" {
        return Ok(PickAction::Toggle {
            view: "files".to_string(),
            query: effective_query.clone(),
        });
    }
    if pick_query_changed(query, &effective_query) {
        return Ok(PickAction::Refresh {
            query: effective_query,
        });
    }
    if !out.status.success() {
        return Ok(PickAction::Cancel);
    }
    if selected_line.is_empty() {
        selected_line = lines
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .to_string();
    }
    let line = selected_line;
    if line.trim().is_empty() {
        return Ok(PickAction::Cancel);
    }
    let path = line.split('\t').next().unwrap_or("").trim().to_string();
    if path.is_empty() {
        return Ok(PickAction::Cancel);
    }
    Ok(PickAction::Selected {
        path,
        query: effective_query,
    })
}

pub(crate) fn pick_num3(value: f64) -> String {
    format!("{:.3}", value)
}

pub(crate) fn pick_trunc(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let take = max_chars.saturating_sub(3);
    let mut out: String = text.chars().take(take).collect();
    out.push_str("...");
    out
}

pub(crate) fn pick_pad_right(text: &str, width: usize) -> String {
    let out = pick_trunc(text, width);
    let len = out.chars().count();
    if len >= width {
        return out;
    }
    format!("{}{}", out, " ".repeat(width - len))
}

pub(crate) fn pick_pad_left(text: &str, width: usize) -> String {
    let out = pick_trunc(text, width);
    let len = out.chars().count();
    if len >= width {
        return out;
    }
    format!("{}{}", " ".repeat(width - len), out)
}

pub(crate) fn pick_home_short(path: &str) -> String {
    let home = env::var("HOME").unwrap_or_default();
    if !home.is_empty() {
        let prefix = format!("{}/", home);
        if path.starts_with(&prefix) {
            return format!("~/{}", &path[prefix.len()..]);
        }
    }
    path.to_string()
}

pub(crate) fn render_project_pick_line(item: &RankedResult, verbose_metrics: bool) -> String {
    let score = pick_num3(item.score);
    let sem = pick_num3(item.semantic);
    let lex = pick_num3(item.lexical);
    let fr = pick_num3(item.frecency);
    let gscore = pick_num3(item.graph);
    let short_path = pick_home_short(&item.path);
    let metrics = if verbose_metrics {
        format!("s:{} l:{} f:{} g:{}", sem, lex, fr, gscore)
    } else {
        format!("sem:{} fr:{}", sem, fr)
    };
    let row = format!(
        "{} | {} | {} | {}",
        pick_pad_right(&short_path, 46),
        pick_pad_left(&score, 6),
        pick_pad_right("dir", 3),
        metrics
    );
    let mut preview = format!(
        "directory: {}\nscore: {} sem={} lex={} fr={} gscore={}",
        short_path, score, sem, lex, fr, gscore
    );
    if item.evidence.is_empty() {
        preview.push_str("\n\nno chunk evidence available");
    } else {
        preview.push_str("\n\ntop chunks:\n");
        for (idx, ev) in item.evidence.iter().take(4).enumerate() {
            let excerpt = pick_trunc(&collapse_whitespace(&ev.excerpt), 180);
            preview.push_str(&format!(
                "\n  {}. {}#{} score={} rel={}\n     {}",
                idx + 1,
                ev.doc_rel_path,
                ev.chunk_index,
                pick_num3(ev.score),
                ev.relation,
                excerpt
            ));
        }
    }
    format!(
        "D:{}\t{}\t\t\t\t{}",
        item.path,
        row,
        pick_preview_escape(&preview)
    )
}

pub(crate) fn render_file_pick_line(item: &RankedFileResult, verbose_metrics: bool) -> String {
    let score = pick_num3(item.score);
    let sem = pick_num3(item.semantic);
    let lex = pick_num3(item.lexical);
    let gscore = pick_num3(item.graph);
    let q = pick_num3(item.quality);
    let short_path = pick_home_short(&item.path);
    let short_project = pick_home_short(&item.project_path);
    let display = if item.doc_rel_path.trim().is_empty() {
        path_basename(&item.path)
    } else {
        item.doc_rel_path.clone()
    };
    let project_label = format!(
        "proj:{}",
        pick_trunc(&path_basename(&item.project_path), 16)
    );
    let metrics = if verbose_metrics {
        format!("s:{} l:{} g:{} q:{}", sem, lex, gscore, q)
    } else {
        format!("sem:{} q:{}", sem, q)
    };
    let row = format!(
        "{} | {} | {} | {}",
        pick_pad_right(&display, 46),
        pick_pad_left(&score, 6),
        pick_pad_right(&project_label, 21),
        metrics
    );
    let mut preview = format!(
        "file: {}\nproject: {}\nscore: {} sem={} lex={} gscore={} q={}\n\nchunk excerpt:\n{}",
        short_path,
        short_project,
        score,
        sem,
        lex,
        gscore,
        q,
        pick_trunc(&collapse_whitespace(&item.excerpt), 220)
    );
    if !item.evidence.is_empty() {
        preview.push_str("\n\nrelated chunks:\n");
        for (idx, ev) in item.evidence.iter().take(4).enumerate() {
            let excerpt = pick_trunc(&collapse_whitespace(&ev.excerpt), 180);
            preview.push_str(&format!(
                "\n  {}. {}#{} score={} rel={}\n     {}",
                idx + 1,
                ev.doc_rel_path,
                ev.chunk_index,
                pick_num3(ev.score),
                ev.relation,
                excerpt
            ));
        }
    }
    format!(
        "F:{}\t{}\t\t\t\t{}",
        item.path,
        row,
        pick_preview_escape(&preview)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::describe_store::StoredFileMeta;
    use crate::pick_view::layout_for_width;
    use crate::rank::EvidenceHit;
    use std::collections::HashMap;

    fn file_result(path: &str, project: &str, rel: &str) -> RankedFileResult {
        RankedFileResult {
            path: path.to_string(),
            project_path: project.to_string(),
            doc_rel_path: rel.to_string(),
            chunk_id: 7,
            chunk_index: 2,
            score: 0.739,
            base_score: 0.7,
            semantic: 0.527,
            lexical: 1.0,
            graph: 1.0,
            relation: "seed".to_string(),
            quality: 1.0,
            excerpt: "the Acme semantic layer".to_string(),
            evidence: Vec::new(),
            doc_mtime: 0.0,
            content_date: 1_789_000_000.0,
            date_source: "path",
            age_days: 10.0,
            freshness_tier: "fresh".to_string(),
            is_record: true,
            role: "record",
            verify: false,
            noise: false,
            raw_similarity: Some(0.61),
            superseded_by: None,
            why: "semantic:0.61+lexical:1.00+graph:seed+recency:fresh".to_string(),
        }
    }

    fn ctx<'a>(
        query: &'a str,
        meta: HashMap<String, StoredFileMeta>,
        texts: HashMap<i64, ChunkText>,
    ) -> PickContext<'a> {
        PickContext {
            layout: layout_for_width(160),
            query,
            color: false,
            now: 1_790_000_000.0,
            meta,
            project_meta: HashMap::new(),
            texts,
        }
    }

    /// Evidence from another project (a linked relation): its `doc_path` is not under the
    /// result's project.
    fn linked_evidence(doc_path: &str, rel: &str) -> EvidenceHit {
        EvidenceHit {
            chunk_id: 99,
            chunk_index: 0,
            doc_path: doc_path.to_string(),
            doc_rel_path: rel.to_string(),
            score: 0.852,
            semantic: 0.8,
            lexical: 0.0,
            graph: 0.5,
            relation: "related_project".to_string(),
            quality: 1.0,
            excerpt: "context thread".to_string(),
            content_date: 0.0,
            date_source: "mtime",
            age_days: 3.0,
            freshness_tier: "fresh".to_string(),
            is_record: false,
            role: "knowledge",
            verify: false,
            noise: false,
            raw_similarity: Some(0.8),
            recency: 0.9,
            why: "semantic:0.80+graph:related_project".to_string(),
        }
    }

    fn chunk(text: String, total: i64) -> ChunkText {
        ChunkText {
            text,
            total,
            symbol_name: String::new(),
            parent_context: String::new(),
        }
    }

    #[test]
    fn file_candidate_uses_the_stored_title_and_centred_snippet() {
        let mut item = file_result(
            "/p/alpha/customer-signals/Acme/20260918-review.md",
            "/p/alpha",
            "customer-signals/Acme/20260918-review.md",
        );
        item.evidence.push(linked_evidence(
            "/p/beta/notes/context-thread.md",
            "notes/context-thread.md",
        ));
        let mut meta = HashMap::new();
        meta.insert(
            item.path.clone(),
            StoredFileMeta {
                title: "Acme design review".to_string(),
                title_source: "h1".to_string(),
                doc_kind: "slides".to_string(),
            },
        );
        // Keyed by the evidence document's own `doc_path`, as `build_file_candidates` loads it.
        meta.insert(
            "/p/beta/notes/context-thread.md".to_string(),
            StoredFileMeta {
                title: "AWS Context POC use case thread".to_string(),
                title_source: "h1".to_string(),
                doc_kind: "md".to_string(),
            },
        );
        let mut texts = HashMap::new();
        texts.insert(
            7,
            chunk(
                format!(
                    "{} the Acme semantic layer maps it {}",
                    "x ".repeat(200),
                    "y ".repeat(200)
                ),
                7,
            ),
        );
        let c = make_file_pick_candidate(&item, &ctx("acme", meta, texts));
        let line = format_pick_candidate_line(&c);
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "{}", line);
        assert_eq!(fields[0], item.path);
        assert!(fields[1].starts_with("slides "), "{}", fields[1]);
        assert!(fields[1].contains("Acme design review"), "{}", fields[1]);
        // semantic 0.527 is below the 0.7 "meaning" bar; lexical 1.0 is exact words.
        assert!(fields[1].contains("words · seed"), "{}", fields[1]);
        assert!(
            !fields[1].contains("s:0."),
            "no raw signals in the row: {}",
            fields[1]
        );
        assert!(
            fields[2].contains("\\033[1;4mAcme\\033[0m"),
            "{}",
            fields[2]
        );
        assert!(fields[2].contains("(chunk 3 of 7)"), "{}", fields[2]);
        assert!(fields[2].contains("exact words"), "{}", fields[2]);
        assert!(!fields[2].contains('\n'));
        // The related line names the linked project's document by its stored title, not by
        // a humanised file name.
        assert!(
            fields[2].contains("related:") && fields[2].contains("AWS Context POC use case thread"),
            "{}",
            fields[2]
        );
        assert!(!fields[2].contains("context thread —"), "{}", fields[2]);
    }

    #[test]
    fn evidence_project_is_derived_from_the_document_path() {
        assert_eq!(
            evidence_project("/p/beta/notes/x.md", "notes/x.md", "/p/alpha"),
            "/p/beta"
        );
        assert_eq!(
            evidence_project("/p/alpha/docs/a.md", "docs/a.md", "/p/alpha"),
            "/p/alpha"
        );
        // A path that does not end in the relative path falls back to the result's project.
        assert_eq!(
            evidence_project("/elsewhere/other.md", "docs/a.md", "/p/alpha"),
            "/p/alpha"
        );
        assert_eq!(evidence_project("/p/beta/a.md", "", "/p/alpha"), "/p/alpha");
    }

    #[test]
    fn builders_surface_ranking_errors() {
        // A bare connection has no `app_state` table, so the ranking's readiness check fails
        // before any embedding call; the builders must hand that error back, not swallow it.
        let conn = Connection::open_in_memory().unwrap();
        let cfg = ConfigValues::from_map(HashMap::new());
        let layout = layout_for_width(120);
        let err = build_file_candidates(&conn, &cfg, "acme", 10, layout)
            .err()
            .expect("the ranking error is handed back");
        assert!(!err.is_empty(), "{}", err);
        let err = build_project_candidates(&conn, &cfg, "acme", 10, layout)
            .err()
            .expect("the ranking error is handed back");
        assert!(!err.is_empty(), "{}", err);
    }

    #[test]
    fn file_candidate_without_metadata_falls_back_to_the_file_name() {
        let item = file_result(
            "/p/alpha/docs/2026-09-plan_v2.md",
            "/p/alpha",
            "docs/2026-09-plan_v2.md",
        );
        let c = make_file_pick_candidate(&item, &ctx("plan", HashMap::new(), HashMap::new()));
        assert!(c.row.contains("2026 09 plan v2"), "{}", c.row);
        assert!(
            c.row.starts_with("md "),
            "kind from the extension when no row: {}",
            c.row
        );
        assert!(
            c.preview.contains("the Acme semantic layer"),
            "excerpt is the snippet fallback: {}",
            c.preview
        );
        // Config, data and code files keep their full file name, as the describe pass would
        // store it; nothing is humanised for them.
        let item = file_result(
            "/p/alpha/deck/app-settings.toml",
            "/p/alpha",
            "deck/app-settings.toml",
        );
        let c = make_file_pick_candidate(&item, &ctx("plan", HashMap::new(), HashMap::new()));
        assert!(c.row.starts_with("config "), "{}", c.row);
        assert!(c.row.contains("app-settings.toml"), "{}", c.row);
        assert!(!c.row.contains("app settings"), "not humanised: {}", c.row);
    }

    #[test]
    fn code_rows_show_the_matched_symbol() {
        let item = file_result("/p/alpha/src/pick.rs", "/p/alpha", "src/pick.rs");
        let mut texts = HashMap::new();
        texts.insert(
            7,
            ChunkText {
                text: "fn render_row() {}".to_string(),
                total: 3,
                symbol_name: "render_row".to_string(),
                parent_context: String::new(),
            },
        );
        let c = make_file_pick_candidate(&item, &ctx("render", HashMap::new(), texts));
        assert!(c.row.starts_with("code "), "{}", c.row);
        assert!(c.row.contains("render_row · pick.rs"), "{}", c.row);
    }

    #[test]
    fn jump_feed_width_flag_and_env_precedence() {
        assert_eq!(feed_columns(Some("0"), Some(150)), 150);
        assert_eq!(feed_columns(Some("200"), Some(150)), 200);
        assert_eq!(feed_columns(None, None), 120);
        assert_eq!(feed_columns(Some("abc"), None), 120);
    }

    #[test]
    fn resize_bind_needs_fzf_0_46() {
        assert!(!supports_resize_event(Some((0, 44))));
        assert!(supports_resize_event(Some((0, 46))));
        assert!(supports_resize_event(Some((0, 74))));
        assert!(supports_resize_event(Some((1, 0))));
        assert!(!supports_resize_event(None));
        assert_eq!(parse_fzf_version("0.74.1 (Homebrew)\n"), Some((0, 74)));
        assert_eq!(parse_fzf_version("0.44.1 (d7a36f6)"), Some((0, 44)));
        assert_eq!(parse_fzf_version(""), None);
        assert_eq!(parse_fzf_version("fzf"), None);
    }

    #[test]
    fn unsafe_paths_are_dropped_before_rendering() {
        assert!(path_is_fzf_safe("/p/alpha/docs/a b.md"));
        assert!(!path_is_fzf_safe("/p/alpha/a\tb.md"));
        assert!(!path_is_fzf_safe("/p/alpha/a\nb.md"));
        assert!(!path_is_fzf_safe("/p/alpha/a\rb.md"));
        assert!(!path_is_fzf_safe("/p/alpha/a\u{1b}[31mb.md"));
        let (kept, dropped) = retain_fzf_safe(
            vec!["/p/a.md", "/p/b\tc.md", "/p/d\u{1b}.md", "/p/e.md"],
            |s| *s,
        );
        assert_eq!((kept, dropped), (vec!["/p/a.md", "/p/e.md"], 2));
    }
}
