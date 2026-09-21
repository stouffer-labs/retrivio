//! The jump and pick flows: the fzf-driven picker, jump-feed lines, candidate rendering for projects and files, and the emit-path handshake with the shell wrapper.

use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, fs, process};

use crate::api::path_basename;
use crate::config::{config_path, db_path, load_config_values, ConfigValues};
use crate::db::{ensure_retrieval_backend_ready, open_db_rw, record_selection_event};
use crate::embed::ensure_native_embed_backend;
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
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        if query.trim().is_empty() {
            eprintln!("usage: retrivio jump [--files|--dirs] [query...]");
            process::exit(2);
        }
        let candidates: Vec<PickCandidate> = if view == "files" {
            rank_files_native(&conn, &cfg, &query, limit)
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                })
                .iter()
                .map(make_file_pick_candidate)
                .collect()
        } else {
            rank_projects_native(&conn, &cfg, &query, limit)
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                })
                .iter()
                .map(make_project_pick_candidate)
                .collect()
        };
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
        eprintln!("error: `fzf` is required for interactive mode. install it or provide a query: retrivio [query]");
        process::exit(2);
    }

    // Interactive live-search mode: fzf calls `jump-feed` on each keystroke
    let mut active_view = view;
    let mut active_query = query;
    loop {
        let action = pick_interactive_live(&active_view, &active_query).unwrap_or_else(|e| {
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

pub(crate) fn format_pick_candidate_line(item: &PickCandidate) -> String {
    let display = format!(
        "{:<60} {:>5}  {:<4} {}",
        item.display_path, item.score, item.kind, item.signals
    );
    format!("{}\t{}\t{}", item.path, display, item.preview)
}

pub(crate) fn run_jump_feed_cmd(args: &[OsString]) {
    let mut view = "projects".to_string();
    let mut query_parts: Vec<String> = Vec::new();
    let mut limit: usize = 40;
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
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
    let dbp = db_path(&cwd);
    let conn = match open_db_rw(&dbp) {
        Ok(c) => c,
        Err(_) => return,
    };
    let candidates: Vec<PickCandidate> = if view == "files" {
        rank_files_native(&conn, &cfg, query, limit)
            .unwrap_or_default()
            .iter()
            .map(make_file_pick_candidate)
            .collect()
    } else {
        rank_projects_native(&conn, &cfg, query, limit)
            .unwrap_or_default()
            .iter()
            .map(make_project_pick_candidate)
            .collect()
    };
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
    let mut active_view = view;
    let mut active_query = query;
    let selected_path = loop {
        let candidates = if active_view == "files" {
            let rows = rank_files_native(&conn, &cfg, &active_query, limit).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            if rows.is_empty() {
                eprintln!("error: no file matches found.");
                process::exit(1);
            }
            rows.into_iter()
                .map(|item| make_file_pick_candidate(&item))
                .collect::<Vec<_>>()
        } else {
            let rows =
                rank_projects_native(&conn, &cfg, &active_query, limit).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    process::exit(1);
                });
            if rows.is_empty() {
                eprintln!("error: no indexed projects found. run `retrivio index` first.");
                process::exit(1);
            }
            rows.into_iter()
                .map(|item| make_project_pick_candidate(&item))
                .collect::<Vec<_>>()
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

pub(crate) struct PickCandidate {
    path: String,
    display_path: String,
    score: String,
    kind: String,
    signals: String,
    preview: String,
}

pub(crate) fn pick_one_line(text: &str, max_len: usize) -> String {
    pick_trunc(&collapse_whitespace(text), max_len).replace('\t', " ")
}

pub(crate) fn pick_signals(pairs: &[(&str, f64)]) -> String {
    let parts: Vec<String> = pairs
        .iter()
        .filter(|(_, v)| *v > 0.0005)
        .map(|(label, v)| format!("{}:{}", label, pick_num3(*v)))
        .collect();
    if parts.is_empty() {
        "-".to_string()
    } else {
        parts.join(" ")
    }
}

pub(crate) fn make_project_pick_candidate(item: &RankedResult) -> PickCandidate {
    let raw = pick_trunc(&pick_home_short(&item.path), 58);
    let display_path = format!("{:<60}", raw);
    let score = pick_num3(item.score);
    let signals = pick_signals(&[
        ("s", item.semantic),
        ("l", item.lexical),
        ("f", item.frecency),
        ("g", item.graph),
    ]);
    let preview = if let Some(ev) = item.evidence.first() {
        format!(
            "top chunk: {}#{} score={} rel={} :: {}",
            ev.doc_rel_path,
            ev.chunk_index,
            pick_num3(ev.score),
            ev.relation,
            pick_one_line(&ev.excerpt, 240)
        )
    } else {
        "no chunk evidence available".to_string()
    };
    PickCandidate {
        path: item.path.clone(),
        display_path,
        score,
        kind: "dir".to_string(),
        signals,
        preview,
    }
}

pub(crate) fn make_file_pick_candidate(item: &RankedFileResult) -> PickCandidate {
    let file_label = if item.doc_rel_path.trim().is_empty() {
        path_basename(&item.path)
    } else {
        item.doc_rel_path.clone()
    };
    let raw = format!(
        "{} [{}]",
        pick_trunc(&file_label, 40),
        pick_trunc(&path_basename(&item.project_path), 16)
    );
    let display_path = format!("{:<60}", raw);
    let score = pick_num3(item.score);
    let sig_pairs: Vec<(&str, f64)> = vec![
        ("s", item.semantic),
        ("l", item.lexical),
        ("g", item.graph),
        ("q", item.quality),
    ];
    let rel_display = pick_trunc(&item.relation, 12);
    let signals = {
        let base = pick_signals(&sig_pairs);
        if item.relation.is_empty() || item.relation == "none" {
            base
        } else {
            format!("{} r:{}", base, rel_display)
        }
    };
    let mut preview = format!(
        "chunk #{} score={} :: {}",
        item.chunk_index,
        pick_num3(item.score),
        pick_one_line(&item.excerpt, 220)
    );
    if let Some(ev) = item.evidence.first() {
        preview.push_str(&format!(
            " | related {}#{} {} {}",
            pick_trunc(&ev.doc_rel_path, 32),
            ev.chunk_index,
            pick_num3(ev.score),
            pick_one_line(&ev.excerpt, 120)
        ));
    }
    PickCandidate {
        path: item.path.clone(),
        display_path,
        score,
        kind: "file".to_string(),
        signals,
        preview,
    }
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

pub(crate) fn pick_interactive_live(view: &str, initial_query: &str) -> Result<PickAction, String> {
    let bin = env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "retrivio".to_string());
    let view_flag = if view == "files" { "--files" } else { "--dirs" };
    // Shell-escape the binary path (handle spaces/quotes)
    let escaped_bin = bin.replace('\'', "'\\''");
    let feed_cmd = format!("'{}' jump-feed {} {{q}}", escaped_bin, view_flag);
    let prompt = if view == "files" {
        "retrivio[file]> "
    } else {
        "retrivio[dir]> "
    };
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--delimiter=\t")
        .arg("--with-nth=2")
        .arg("--no-sort")
        .arg("--disabled")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg("type to search | Enter=select Tab=toggle Ctrl-D=dirs Ctrl-F=files")
        .arg("--preview")
        .arg("echo {3}")
        .arg("--preview-window=down,6,wrap")
        .arg("--bind")
        .arg(format!("start:reload({})", feed_cmd))
        .arg("--bind")
        .arg(format!("change:reload({})", feed_cmd))
        .arg("--print-query")
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
        let line = format!(
            "{:<60} {:>5}  {:<4} {}",
            item.display_path, item.score, item.kind, item.signals
        );
        payload.push_str(&format!("{}\t{}\t{}\n", item.path, line, item.preview));
    }
    let prompt = if view == "files" {
        "retrivio[file]> ".to_string()
    } else {
        "retrivio[dir]> ".to_string()
    };
    let mut cmd = Command::new("fzf");
    cmd.arg("--height=70%")
        .arg("--layout=reverse")
        .arg("--border")
        .arg("--delimiter=\t")
        // Keep display compact (field 2), but match query against both
        // display line + preview text (fields 2 and 3).
        .arg("--with-nth=2")
        .arg("--nth=2,3")
        .arg("--print-query")
        .arg("--expect=tab,ctrl-d,ctrl-f")
        .arg("--preview")
        .arg("echo {3}")
        .arg("--preview-window=down,6,wrap")
        .arg("--bind")
        .arg("ctrl-u:unix-line-discard")
        .arg("--prompt")
        .arg(prompt)
        .arg("--header")
        .arg(if query.trim().is_empty() {
            "path | score | type | s=sem l=lex g=graph q=qual r=rel | Enter=select/requery Tab=toggle Ctrl-D=dirs Ctrl-F=files".to_string()
        } else {
            format!(
                "ranked for query='{}' | path | score | type | s=sem l=lex g=graph q=qual r=rel | Enter=select/requery Tab=toggle Ctrl-D=dirs Ctrl-F=files",
                query
            )
        })
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

pub(crate) fn pick_preview_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('\r', "")
        .replace('\t', "    ")
        .replace('\n', "\\n")
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
