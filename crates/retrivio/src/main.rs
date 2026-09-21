mod api;
mod autotune;
mod bench;
#[cfg(test)]
mod chunk_contract_tests;
mod cli;
mod code_intel;
mod config;
mod config_tui;
mod db;
mod documents;
mod dossier;
mod embed;
mod freshness;
mod graph_viewer;
mod hook_install;
mod index;
#[cfg(test)]
mod index_run_tests;
mod lance_store;
mod mcp;
mod pick;
mod prune;
mod rank;
mod recall;
mod related;
mod roles;
mod scan;
mod setup;
#[cfg(test)]
mod test_support;
mod util;
mod watch;

use std::ffi::OsString;
use std::io::IsTerminal;
use std::{env, process};

use crate::api::{run_api_cmd, run_daemon_cmd};
use crate::autotune::run_autotune_cmd;
use crate::bench::run_bench_cmd;
use crate::cli::{
    likely_command_typo, maybe_prompt_shell_hook_setup_for_shorthand_query,
    maybe_warn_if_stale_local_binary, print_help, run_add, run_config_cmd, run_del, run_doctor,
    run_dossier_cmd, run_exclude_cmd, run_include_cmd, run_index_cmd, run_init, run_install_cmd,
    run_prune_cmd, run_reembed_cmd, run_refresh_cmd, run_roots, run_search_cmd, run_self_test_cmd,
    run_stop_cmd, run_ui_cmd,
};
use crate::config::consume_global_path_overrides;
use crate::graph_viewer::run_graph_cmd;
use crate::mcp::run_mcp_cmd;
use crate::pick::{run_jump_cmd, run_jump_feed_cmd, run_pick_cmd};
use crate::setup::{run_auth_cmd, run_setup_cmd};
use crate::watch::run_watch_cmd;

fn main() {
    configure_sigpipe();
    let raw_args: Vec<OsString> = env::args_os().skip(1).collect();
    let args = consume_global_path_overrides(raw_args).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(2);
    });
    if args.is_empty() {
        maybe_warn_if_stale_local_binary(&args);
        // Default to interactive query flow when invoked without a subcommand.
        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            maybe_prompt_shell_hook_setup_for_shorthand_query();
        }
        run_jump_cmd(&[]);
        return;
    }

    maybe_warn_if_stale_local_binary(&args);
    let first = args[0].to_string_lossy().to_string();
    match first.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" | "version" => {
            println!("retrivio {}", env!("CARGO_PKG_VERSION"));
        }
        "doctor" => run_doctor(&args[1..]),
        "setup" => run_setup_cmd(&args[1..]),
        "auth" => run_auth_cmd(&args[1..]),
        "config" => run_config_cmd(&args[1..]),
        "autotune" => run_autotune_cmd(&args[1..]),
        "init" => run_init(&args[1..]),
        "install" => run_install_cmd(&args[1..]),
        "add" => run_add(&args[1..]),
        "del" => run_del(&args[1..]),
        "roots" => run_roots(&args[1..]),
        "exclude" => run_exclude_cmd(&args[1..]),
        "include" => run_include_cmd(&args[1..]),
        "index" => run_index_cmd(&args[1..]),
        "refresh" => run_refresh_cmd(&args[1..]),
        "reembed" => run_reembed_cmd(&args[1..]),
        "prune" => run_prune_cmd(&args[1..]),
        "watch" => run_watch_cmd(&args[1..]),
        "search" => run_search_cmd(&args[1..]),
        "dossier" => run_dossier_cmd(&args[1..]),
        "recall" => recall::run_recall_cmd(&args[1..]),
        "hook" => hook_install::run_hook_cmd(&args[1..]),
        "service" => hook_install::run_service_cmd(&args[1..]),
        "pick" => run_pick_cmd(&args[1..]),
        "jump" => run_jump_cmd(&args[1..]),
        "jump-feed" => run_jump_feed_cmd(&args[1..]),
        "ui" => run_ui_cmd(&args[1..]),
        "stop" => run_stop_cmd(&args[1..]),
        "daemon" => run_daemon_cmd(&args[1..]),
        "bench" => run_bench_cmd(&args[1..]),
        "api" => run_api_cmd(&args[1..]),
        "mcp" => run_mcp_cmd(&args[1..]),
        "self-test" => run_self_test_cmd(&args[1..]),
        "graph" => run_graph_cmd(&args[1..]),
        // Hidden: the PDF extraction child the indexer spawns (documents.rs).
        "documents" => process::exit(documents::run_documents_cmd(&args[1..])),
        _ => {
            if first.starts_with('-') {
                eprintln!("error: unknown option '{}'", first);
                eprintln!();
                print_help();
                process::exit(2);
            }
            if args.len() == 1 {
                if let Some(candidate) = likely_command_typo(&first) {
                    eprintln!("error: unknown command '{}'", first);
                    eprintln!("hint: did you mean `retrivio {}`?", candidate);
                    process::exit(2);
                }
            }
            // Shorthand query mode:
            // - interactive terminals: open jump/picker flow
            // - non-interactive contexts: print search results
            if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                maybe_prompt_shell_hook_setup_for_shorthand_query();
                run_jump_cmd(&args);
            } else {
                run_search_cmd(&args);
            }
        }
    }
}

#[cfg(unix)]
fn configure_sigpipe() {
    // Allow shell pipelines like `retrivio doctor | head -n 5` to exit cleanly
    // when the downstream reader closes early.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn configure_sigpipe() {}
