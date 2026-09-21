//! The setup and auth commands: the interactive Bedrock and Ollama configuration, AWS profile and region discovery, and the credential helper detection.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{env, fs, process};

use serde_json::Value;

use crate::cli::{run_doctor_fix, run_reembed_cmd};
use crate::config::{config_path, load_config_values, write_config_file, ConfigValues};
use crate::db::refresh_reembed_requirement_for_config_change;
use crate::embed::{
    bedrock_concurrency_for_cfg, bedrock_credential_cmd_for_cfg, bedrock_max_retries_for_cfg,
    bedrock_profile_for_cfg, bedrock_refresh_cmd_for_cfg, bedrock_region_for_cfg,
    bedrock_retry_base_ms_for_cfg, default_embed_model_for_backend, ollama_host,
    ollama_is_reachable, ollama_list_local_models, ollama_pull_model, run_ollama_preflight,
    KNOWN_BEDROCK_EMBEDDING_MODELS, KNOWN_OLLAMA_EMBEDDING_MODELS,
};
use crate::util::{
    command_available, command_exists, expand_tilde, is_executable_file, non_empty_env,
    prompt_line, prompt_yes_no, shell_escape, strip_terminal_control_sequences,
};

#[derive(Clone, Debug)]
pub(crate) struct AwsProfileChoice {
    name: String,
    region: Option<String>,
    source: String,
}

#[derive(Clone, Debug)]
pub(crate) struct IsengardAccountChoice {
    account_ref: String,
    label: String,
    favorite: bool,
}

pub(crate) fn run_setup_cmd(args: &[OsString]) {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio setup");
        println!("guided first-time setup for backend/model/auth/performance.");
        return;
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!("error: `retrivio setup` requires an interactive terminal");
        process::exit(2);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cfg_path = config_path(&cwd);
    eprintln!("setup: using config -> {}", cfg_path.display());
    let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
    let old_cfg = cfg.clone();

    // 1. Select backend
    let backend_options = vec![
        "ollama (local model runtime)".to_string(),
        "bedrock (aws credentials + model API)".to_string(),
    ];
    let default_backend_idx = if cfg.embed_backend == "bedrock" { 1 } else { 0 };
    let selected_backend = select_option(
        "choose embedding backend",
        &backend_options,
        Some(default_backend_idx),
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        process::exit(1);
    })
    .unwrap_or(default_backend_idx);
    cfg.embed_backend = if selected_backend == 1 {
        "bedrock".to_string()
    } else {
        "ollama".to_string()
    };

    // 2. Select model (interactive)
    if cfg.embed_backend == "ollama" {
        configure_ollama_model_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        configure_bedrock_model_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    }

    // 3. Backend-specific auth/performance
    if cfg.embed_backend == "bedrock" {
        configure_bedrock_auth_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
        configure_bedrock_performance_interactive(&mut cfg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            process::exit(1);
        });
    } else {
        cfg.aws_profile.clear();
        cfg.aws_region.clear();
        cfg.aws_refresh_cmd.clear();
        cfg.aws_credential_cmd.clear();
    }

    // 4. Write config
    write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
        eprintln!("error: failed writing config: {}", e);
        process::exit(1);
    });
    println!("setup: config updated -> {}", cfg_path.display());
    println!("embed_backend: {}", cfg.embed_backend);
    println!("embed_model: {}", cfg.embed_model);
    if cfg.embed_backend == "bedrock" {
        println!(
            "aws_profile: {}",
            if cfg.aws_profile.trim().is_empty() {
                "<default chain>"
            } else {
                cfg.aws_profile.trim()
            }
        );
        println!("aws_region: {}", bedrock_region_for_cfg(Some(&cfg)));
        println!(
            "aws_refresh_cmd: {}",
            if cfg.aws_refresh_cmd.trim().is_empty() {
                "<none>"
            } else {
                cfg.aws_refresh_cmd.trim()
            }
        );
        println!(
            "aws_credential_cmd: {}",
            if cfg.aws_credential_cmd.trim().is_empty() {
                "<none>"
            } else {
                cfg.aws_credential_cmd.trim()
            }
        );
        println!("bedrock_concurrency: {}", cfg.bedrock_concurrency);
        println!("bedrock_max_retries: {}", cfg.bedrock_max_retries);
        println!("bedrock_retry_base_ms: {}", cfg.bedrock_retry_base_ms);
    }

    // 5. Preflight — both backends
    if cfg.embed_backend == "bedrock" {
        match run_doctor_fix(&cfg) {
            Ok(_) => println!("setup: bedrock preflight ok"),
            Err(e) => eprintln!("warning: setup preflight failed: {}", e),
        }
    } else {
        match run_ollama_preflight(&cfg) {
            Ok(_) => println!("setup: ollama preflight ok"),
            Err(e) => {
                eprintln!("warning: setup preflight failed: {}", e);
                if !command_exists("ollama") {
                    eprintln!(
                        "note: `ollama` is not installed on this system. install Ollama or rerun `retrivio setup` and choose `bedrock`."
                    );
                } else {
                    eprintln!(
                        "note: start Ollama with `ollama serve`, or rerun `retrivio setup` and choose `bedrock`."
                    );
                }
            }
        }
    }

    // 6. Reembed detection (config change + quick compatibility check)
    match refresh_reembed_requirement_for_config_change(&cwd, &old_cfg, &cfg) {
        Ok(Some(reason)) => {
            eprintln!();
            eprintln!("warning: {}", reason);
            match prompt_yes_no("run `retrivio reembed` now?", false) {
                Ok(true) => {
                    run_reembed_cmd(&[]);
                }
                Ok(false) => {
                    println!("run `retrivio reembed` before searching.");
                }
                Err(e) => {
                    eprintln!("warning: prompt failed: {}", e);
                    println!("run `retrivio reembed` before searching.");
                }
            }
        }
        Ok(None) => { /* embeddings compatible with current config */ }
        Err(e) => {
            // DB may not exist yet on first setup — that's fine
            if !e.contains("no such table") && !e.contains("unable to open") {
                eprintln!("warning: reembed check failed: {}", e);
            }
        }
    }
}

pub(crate) fn run_auth_cmd(args: &[OsString]) {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: retrivio auth [select|status]");
        println!("  select  interactive Bedrock auth/profile/region selection");
        println!("  status  show resolved Bedrock auth/profile/region");
        return;
    }
    let sub = args[0].to_string_lossy().to_ascii_lowercase();
    match sub.as_str() {
        "status" => {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg = ConfigValues::from_map(load_config_values(&config_path(&cwd)));
            println!("embed_backend: {}", cfg.embed_backend);
            println!(
                "aws_profile (configured): {}",
                if cfg.aws_profile.trim().is_empty() {
                    "<none>"
                } else {
                    cfg.aws_profile.trim()
                }
            );
            println!(
                "aws_profile (resolved): {}",
                bedrock_profile_for_cfg(Some(&cfg))
                    .unwrap_or_else(|| "<default chain>".to_string())
            );
            println!(
                "aws_region (configured): {}",
                if cfg.aws_region.trim().is_empty() {
                    "<none>"
                } else {
                    cfg.aws_region.trim()
                }
            );
            println!(
                "aws_region (resolved): {}",
                bedrock_region_for_cfg(Some(&cfg))
            );
            println!(
                "aws_refresh_cmd: {}",
                bedrock_refresh_cmd_for_cfg(Some(&cfg)).unwrap_or_else(|| "<none>".to_string())
            );
            println!(
                "aws_credential_cmd: {}",
                bedrock_credential_cmd_for_cfg(Some(&cfg)).unwrap_or_else(|| "<none>".to_string())
            );
            println!(
                "bedrock_concurrency: {}",
                bedrock_concurrency_for_cfg(Some(&cfg))
            );
            println!(
                "bedrock_max_retries: {}",
                bedrock_max_retries_for_cfg(Some(&cfg))
            );
            println!(
                "bedrock_retry_base_ms: {}",
                bedrock_retry_base_ms_for_cfg(Some(&cfg))
            );
        }
        "select" => {
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                eprintln!("error: `retrivio auth select` requires an interactive terminal");
                process::exit(2);
            }
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cfg_path = config_path(&cwd);
            let mut cfg = ConfigValues::from_map(load_config_values(&cfg_path));
            cfg.embed_backend = "bedrock".to_string();
            configure_bedrock_auth_interactive(&mut cfg).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                process::exit(1);
            });
            write_config_file(&cfg_path, &cfg).unwrap_or_else(|e| {
                eprintln!("error: failed writing config: {}", e);
                process::exit(1);
            });
            println!("auth: config updated -> {}", cfg_path.display());
            println!(
                "aws_profile: {}",
                if cfg.aws_profile.trim().is_empty() {
                    "<default chain>"
                } else {
                    cfg.aws_profile.trim()
                }
            );
            println!("aws_region: {}", bedrock_region_for_cfg(Some(&cfg)));
            println!("bedrock_concurrency: {}", cfg.bedrock_concurrency);
            println!("bedrock_max_retries: {}", cfg.bedrock_max_retries);
            println!("bedrock_retry_base_ms: {}", cfg.bedrock_retry_base_ms);
        }
        other => {
            eprintln!("error: unknown auth subcommand '{}'", other);
            process::exit(2);
        }
    }
}

pub(crate) fn configure_bedrock_auth_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let aws_profiles = list_aws_profiles();
    let isengard_cli = find_isengardcli_binary();
    let isengard_accounts = if let Some(cli) = &isengard_cli {
        list_isengard_accounts(cli).unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut use_isengard = false;
    if !isengard_accounts.is_empty() {
        let source_options = vec![
            "aws profile (from local aws config/credentials)".to_string(),
            "isengard account (on-demand credentials)".to_string(),
        ];
        let source_idx = select_option(
            "choose bedrock auth source",
            &source_options,
            Some(if cfg.aws_credential_cmd.trim().is_empty() {
                0
            } else {
                1
            }),
        )?
        .unwrap_or(0);
        use_isengard = source_idx == 1;
    }

    if use_isengard {
        let labels: Vec<String> = isengard_accounts.iter().map(|a| a.label.clone()).collect();
        if let Some(idx) =
            select_option("choose isengard account for credentials", &labels, Some(0))?
        {
            let choice = &isengard_accounts[idx];
            if let Some(cli) = &isengard_cli {
                let role_options = vec![
                    "Admin".to_string(),
                    "ReadOnly".to_string(),
                    "PowerUser".to_string(),
                ];
                let existing_role = extract_role_from_refresh_cmd(&cfg.aws_credential_cmd)
                    .or_else(|| extract_role_from_refresh_cmd(&cfg.aws_refresh_cmd));
                let default_role_idx = existing_role
                    .as_ref()
                    .and_then(|r| role_options.iter().position(|o| o.eq_ignore_ascii_case(r)))
                    .unwrap_or(0);
                let role = if let Some(role_idx) = select_option(
                    "choose isengard role",
                    &role_options,
                    Some(default_role_idx),
                )? {
                    role_options[role_idx].clone()
                } else {
                    "Admin".to_string()
                };
                cfg.aws_credential_cmd = format!(
                    "{} credentials --awscli {} --role {}",
                    shell_escape(cli),
                    shell_escape(&choice.account_ref),
                    shell_escape(&role)
                );
                // Stop running the legacy refresh-shaped command alongside the new
                // on-demand one — it's redundant at best, and broken in tenants that
                // no longer issue static keys.
                cfg.aws_refresh_cmd.clear();
            }
        }
    } else {
        cfg.aws_credential_cmd.clear();
        cfg.aws_refresh_cmd.clear();
    }

    if !aws_profiles.is_empty() {
        let mut profile_options = vec!["<default chain>".to_string()];
        for profile in &aws_profiles {
            let region = profile.region.clone().unwrap_or_else(|| "-".to_string());
            profile_options.push(format!(
                "{}  (region={} source={})",
                profile.name, region, profile.source
            ));
        }
        let default_idx = if cfg.aws_profile.trim().is_empty() {
            0
        } else {
            aws_profiles
                .iter()
                .position(|p| p.name == cfg.aws_profile.trim())
                .map(|i| i + 1)
                .unwrap_or(0)
        };
        let selected = select_option("choose aws profile", &profile_options, Some(default_idx))?
            .unwrap_or(default_idx);
        if selected == 0 {
            cfg.aws_profile.clear();
        } else if let Some(p) = aws_profiles.get(selected - 1) {
            cfg.aws_profile = p.name.clone();
            if cfg.aws_region.trim().is_empty() {
                if let Some(region) = &p.region {
                    cfg.aws_region = region.clone();
                }
            }
        }
    } else {
        let manual = prompt_line("aws profile name (empty for default chain): ")?;
        cfg.aws_profile = manual.trim().to_string();
    }

    let mut region_candidates = vec![
        "us-east-1".to_string(),
        "us-east-2".to_string(),
        "us-west-1".to_string(),
        "us-west-2".to_string(),
        "eu-west-1".to_string(),
        "eu-central-1".to_string(),
        "ap-southeast-1".to_string(),
        "ap-southeast-2".to_string(),
        "ap-northeast-1".to_string(),
    ];
    if let Some(region) = aws_region_for_profile_name(if cfg.aws_profile.trim().is_empty() {
        "default"
    } else {
        cfg.aws_profile.trim()
    }) {
        region_candidates.push(region);
    }
    if !cfg.aws_region.trim().is_empty() {
        region_candidates.push(cfg.aws_region.trim().to_string());
    }
    let resolved_region = bedrock_region_for_cfg(Some(cfg));
    region_candidates.push(resolved_region.clone());
    region_candidates.sort();
    region_candidates.dedup();
    let default_region_idx = region_candidates
        .iter()
        .position(|r| *r == resolved_region)
        .unwrap_or(0);
    if let Some(idx) = select_option(
        "choose aws region for bedrock",
        &region_candidates,
        Some(default_region_idx),
    )? {
        cfg.aws_region = region_candidates[idx].clone();
    }
    Ok(())
}

pub(crate) fn configure_bedrock_performance_interactive(
    cfg: &mut ConfigValues,
) -> Result<(), String> {
    let current_concurrency = cfg.bedrock_concurrency.clamp(1, 32);
    let current_retries = cfg.bedrock_max_retries.clamp(0, 12);
    let current_base_ms = cfg.bedrock_retry_base_ms.clamp(50, 10_000);

    let profiles = vec![
        "balanced (recommended): concurrency=4 retries=3 base_ms=250".to_string(),
        "fast: concurrency=8 retries=2 base_ms=150".to_string(),
        "conservative: concurrency=2 retries=4 base_ms=350".to_string(),
        "aggressive: concurrency=12 retries=1 base_ms=120".to_string(),
        format!(
            "keep current: concurrency={} retries={} base_ms={}",
            current_concurrency, current_retries, current_base_ms
        ),
        "custom...".to_string(),
    ];

    let selected =
        select_option("choose bedrock performance profile", &profiles, Some(4))?.unwrap_or(4);
    match selected {
        0 => {
            cfg.bedrock_concurrency = 4;
            cfg.bedrock_max_retries = 3;
            cfg.bedrock_retry_base_ms = 250;
        }
        1 => {
            cfg.bedrock_concurrency = 8;
            cfg.bedrock_max_retries = 2;
            cfg.bedrock_retry_base_ms = 150;
        }
        2 => {
            cfg.bedrock_concurrency = 2;
            cfg.bedrock_max_retries = 4;
            cfg.bedrock_retry_base_ms = 350;
        }
        3 => {
            cfg.bedrock_concurrency = 12;
            cfg.bedrock_max_retries = 1;
            cfg.bedrock_retry_base_ms = 120;
        }
        4 => {
            cfg.bedrock_concurrency = current_concurrency;
            cfg.bedrock_max_retries = current_retries;
            cfg.bedrock_retry_base_ms = current_base_ms;
        }
        _ => {
            let c = prompt_line(&format!(
                "bedrock_concurrency (1..32) [{}]: ",
                current_concurrency
            ))?;
            let r = prompt_line(&format!(
                "bedrock_max_retries (0..12) [{}]: ",
                current_retries
            ))?;
            let b = prompt_line(&format!(
                "bedrock_retry_base_ms (50..10000) [{}]: ",
                current_base_ms
            ))?;

            let parsed_c = if c.trim().is_empty() {
                current_concurrency
            } else {
                c.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_concurrency must be an integer".to_string())?
            };
            let parsed_r = if r.trim().is_empty() {
                current_retries
            } else {
                r.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_max_retries must be an integer".to_string())?
            };
            let parsed_b = if b.trim().is_empty() {
                current_base_ms
            } else {
                b.trim()
                    .parse::<i64>()
                    .map_err(|_| "bedrock_retry_base_ms must be an integer".to_string())?
            };
            cfg.bedrock_concurrency = parsed_c.clamp(1, 32);
            cfg.bedrock_max_retries = parsed_r.clamp(0, 12);
            cfg.bedrock_retry_base_ms = parsed_b.clamp(50, 10_000);
        }
    }
    Ok(())
}

/// Interactive ollama model selection with availability checks.
pub(crate) fn configure_ollama_model_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let reachable = ollama_is_reachable().unwrap_or(false);
    let local_models: Vec<String> = if reachable {
        ollama_list_local_models().unwrap_or_default()
    } else {
        eprintln!(
            "warning: ollama not reachable at '{}'; install status unknown",
            ollama_host()
        );
        Vec::new()
    };

    // Build the selection list from known-good models
    let mut options: Vec<String> = Vec::new();
    let mut option_model_names: Vec<String> = Vec::new();

    for &(name, desc) in KNOWN_OLLAMA_EMBEDDING_MODELS {
        let installed = local_models
            .iter()
            .any(|m| m == name || m.starts_with(&format!("{}:", name)));
        let status = if !reachable {
            String::new()
        } else if installed {
            ", installed".to_string()
        } else {
            ", not installed".to_string()
        };
        options.push(format!("{} ({}{})", name, desc, status));
        option_model_names.push(name.to_string());
    }

    // Add any locally-installed models not already in the known list
    if reachable {
        for local in &local_models {
            let already = KNOWN_OLLAMA_EMBEDDING_MODELS
                .iter()
                .any(|&(n, _)| local == n || local.starts_with(&format!("{}:", n)));
            if !already {
                options.push(format!("{} (installed, unknown dims)", local));
                option_model_names.push(local.clone());
            }
        }
    }

    // Add custom option
    options.push("custom (enter model name)...".to_string());
    option_model_names.push(String::new()); // sentinel for custom

    // Find default index: current model, or first entry
    let current_model = if cfg.embed_model.trim().is_empty() {
        default_embed_model_for_backend("ollama").to_string()
    } else {
        cfg.embed_model.trim().to_string()
    };
    let default_idx = option_model_names
        .iter()
        .position(|n| {
            !n.is_empty() && (n == &current_model || current_model.starts_with(&format!("{}:", n)))
        })
        .unwrap_or(0);

    let selected = select_option("choose ollama embedding model", &options, Some(default_idx))?
        .unwrap_or(default_idx);

    let chosen_model =
        if selected < option_model_names.len() && !option_model_names[selected].is_empty() {
            option_model_names[selected].clone()
        } else {
            // Custom entry
            let custom = prompt_line("enter ollama model name: ")?;
            let trimmed = custom.trim().to_string();
            if trimmed.is_empty() {
                eprintln!(
                    "warning: empty model name, keeping current: {}",
                    current_model
                );
                return Ok(());
            }
            trimmed
        };

    cfg.embed_model = chosen_model.clone();

    // Offer to pull if not installed and ollama CLI is available
    if reachable {
        let is_installed = local_models
            .iter()
            .any(|m| m == &chosen_model || m.starts_with(&format!("{}:", chosen_model)));
        if !is_installed && command_available("ollama") {
            match prompt_yes_no(
                &format!("'{}' is not installed locally. pull now?", chosen_model),
                true,
            ) {
                Ok(true) => {
                    if let Err(e) = ollama_pull_model(&chosen_model) {
                        eprintln!("warning: pull failed: {}", e);
                    }
                }
                Ok(false) => {
                    println!(
                        "skipped. run `ollama pull {}` before using this model.",
                        chosen_model
                    );
                }
                Err(e) => {
                    eprintln!("warning: prompt failed: {}", e);
                }
            }
        } else if !is_installed {
            println!(
                "note: '{}' is not installed. run `ollama pull {}` before using this model.",
                chosen_model, chosen_model
            );
        }
    }

    Ok(())
}

/// Interactive bedrock model selection.
pub(crate) fn configure_bedrock_model_interactive(cfg: &mut ConfigValues) -> Result<(), String> {
    let mut options: Vec<String> = Vec::new();
    let mut option_model_ids: Vec<String> = Vec::new();

    for &(model_id, desc) in KNOWN_BEDROCK_EMBEDDING_MODELS {
        options.push(format!("{} ({})", model_id, desc));
        option_model_ids.push(model_id.to_string());
    }

    options.push("custom (enter model ID or ARN)...".to_string());
    option_model_ids.push(String::new()); // sentinel

    let current_model = if cfg.embed_model.trim().is_empty() {
        default_embed_model_for_backend("bedrock").to_string()
    } else {
        cfg.embed_model.trim().to_string()
    };
    let default_idx = option_model_ids
        .iter()
        .position(|n| !n.is_empty() && n == &current_model)
        .unwrap_or(0);

    let selected = select_option(
        "choose bedrock embedding model",
        &options,
        Some(default_idx),
    )?
    .unwrap_or(default_idx);

    let chosen_model =
        if selected < option_model_ids.len() && !option_model_ids[selected].is_empty() {
            option_model_ids[selected].clone()
        } else {
            let custom = prompt_line("enter bedrock model ID or ARN: ")?;
            let trimmed = custom.trim().to_string();
            if trimmed.is_empty() {
                eprintln!(
                    "warning: empty model ID, keeping current: {}",
                    current_model
                );
                return Ok(());
            }
            trimmed
        };

    cfg.embed_model = chosen_model;
    Ok(())
}

pub(crate) fn select_option(
    prompt: &str,
    options: &[String],
    default: Option<usize>,
) -> Result<Option<usize>, String> {
    if options.is_empty() {
        return Ok(None);
    }
    let default_idx = default.unwrap_or(0).min(options.len().saturating_sub(1));
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() && command_exists("fzf") {
        let mut ordered_indices: Vec<usize> = (0..options.len()).collect();
        ordered_indices.rotate_left(default_idx);

        let mut cmd = Command::new("fzf");
        cmd.arg("--ansi")
            .arg("--prompt")
            .arg(format!("{} > ", prompt))
            .arg("--height")
            .arg("50%")
            .arg("--layout=reverse")
            .arg("--border")
            .arg("--cycle")
            .arg("--delimiter")
            .arg("\t")
            .arg("--with-nth")
            .arg("2")
            .arg("--no-sort")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed launching fzf selector: {}", e))?;
        if let Some(mut stdin) = child.stdin.take() {
            for idx in ordered_indices {
                let marker = if idx == default_idx { "[*]" } else { "[ ]" };
                writeln!(stdin, "{}\t{} {}", idx, marker, options[idx])
                    .map_err(|e| format!("failed writing selector input: {}", e))?;
            }
        }
        let output = child
            .wait_with_output()
            .map_err(|e| format!("failed waiting for fzf selector: {}", e))?;
        if !output.status.success() {
            return Ok(None);
        }
        let line = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if line.is_empty() {
            return Ok(None);
        }
        if let Some((left, _)) = line.split_once('\t') {
            if let Ok(n) = left.trim().parse::<usize>() {
                if n < options.len() {
                    return Ok(Some(n));
                }
            }
        }
    }

    println!("{}", prompt);
    for (idx, option) in options.iter().enumerate() {
        let marker = if idx == default_idx { "*" } else { " " };
        println!("  {:>2}. [{}] {}", idx + 1, marker, option);
    }
    let raw = prompt_line(&format!("select number (Enter for {}): ", default_idx + 1))?;
    let normalized = strip_terminal_control_sequences(raw.trim());
    if normalized.trim().is_empty() {
        return Ok(Some(default_idx));
    }
    let parsed = normalized
        .trim()
        .parse::<usize>()
        .map_err(|_| "invalid selection: expected a number".to_string())?;
    if parsed < 1 || parsed > options.len() {
        return Err(format!(
            "invalid selection: {} (must be between 1 and {})",
            parsed,
            options.len()
        ));
    }
    Ok(Some(parsed - 1))
}

pub(crate) fn aws_config_file_path() -> PathBuf {
    non_empty_env("AWS_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_tilde("~/.aws/config"))
}

pub(crate) fn aws_credentials_file_path() -> PathBuf {
    non_empty_env("AWS_SHARED_CREDENTIALS_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| expand_tilde("~/.aws/credentials"))
}

pub(crate) fn parse_aws_profile_regions() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let path = aws_config_file_path();
    let Ok(raw) = fs::read_to_string(path) else {
        return out;
    };
    let mut current: Option<String> = None;
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') {
            let mut section = t.trim_start_matches('[').trim_end_matches(']').trim();
            if let Some(rest) = section.strip_prefix("profile ") {
                section = rest.trim();
            }
            if section.is_empty() {
                current = None;
            } else {
                current = Some(section.to_string());
            }
            continue;
        }
        if let Some(profile) = &current {
            if let Some((k, v)) = t.split_once('=') {
                if k.trim().eq_ignore_ascii_case("region") {
                    let region = v.trim().to_string();
                    if !region.is_empty() {
                        out.insert(profile.clone(), region);
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn parse_ini_profile_names(path: &Path, treat_profile_prefix: bool) -> HashSet<String> {
    let mut out = HashSet::new();
    let Ok(raw) = fs::read_to_string(path) else {
        return out;
    };
    for line in raw.lines() {
        let t = line.trim();
        if !(t.starts_with('[') && t.ends_with(']')) {
            continue;
        }
        let mut section = t
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim()
            .to_string();
        if treat_profile_prefix {
            if let Some(rest) = section.strip_prefix("profile ") {
                section = rest.trim().to_string();
            }
        }
        if !section.is_empty() {
            out.insert(section);
        }
    }
    out
}

pub(crate) fn aws_cli_list_profiles() -> Vec<String> {
    if !command_available("aws") {
        return Vec::new();
    }
    let output = match Command::new("aws")
        .arg("configure")
        .arg("list-profiles")
        .output()
    {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

pub(crate) fn list_aws_profiles() -> Vec<AwsProfileChoice> {
    let mut names: HashSet<String> = HashSet::new();
    let mut source_map: HashMap<String, HashSet<String>> = HashMap::new();
    for p in aws_cli_list_profiles() {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-cli".to_string());
    }
    for p in parse_ini_profile_names(&aws_config_file_path(), true) {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-config".to_string());
    }
    for p in parse_ini_profile_names(&aws_credentials_file_path(), false) {
        names.insert(p.clone());
        source_map
            .entry(p)
            .or_default()
            .insert("aws-credentials".to_string());
    }
    let region_map = parse_aws_profile_regions();
    let mut out: Vec<AwsProfileChoice> = names
        .into_iter()
        .map(|name| {
            let source = source_map
                .remove(&name)
                .map(|set| {
                    let mut v: Vec<String> = set.into_iter().collect();
                    v.sort();
                    v.join("+")
                })
                .unwrap_or_else(|| "local".to_string());
            AwsProfileChoice {
                region: region_map.get(&name).cloned(),
                name,
                source,
            }
        })
        .collect();
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

pub(crate) fn aws_region_for_profile_name(profile: &str) -> Option<String> {
    let map = parse_aws_profile_regions();
    map.get(profile).cloned()
}

pub(crate) fn find_isengardcli_binary() -> Option<String> {
    if command_available("isengardcli") {
        return Some("isengardcli".to_string());
    }
    let fallback = expand_tilde("~/Scripts/isengardcli/isengardcli");
    if is_executable_file(&fallback) {
        return Some(fallback.to_string_lossy().to_string());
    }
    None
}

pub(crate) fn extract_role_from_refresh_cmd(cmd: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "--role" {
            if let Some(role) = parts.get(i + 1) {
                return Some(role.to_string());
            }
        }
    }
    None
}

pub(crate) fn list_isengard_accounts(cli: &str) -> Result<Vec<IsengardAccountChoice>, String> {
    let output = Command::new(cli)
        .arg("ls")
        .arg("--output")
        .arg("json")
        .arg("--all")
        .output()
        .map_err(|e| format!("failed running isengardcli ls: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("isengardcli ls failed: {}", detail));
    }
    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    let parsed: Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid isengard json output: {}", e))?;
    let arr = parsed
        .as_array()
        .ok_or_else(|| "unexpected isengard output format".to_string())?;
    let mut out = Vec::new();
    for row in arr {
        let status = row.get("Status").and_then(|v| v.as_str()).unwrap_or("");
        if !status.is_empty() && !status.eq_ignore_ascii_case("ACTIVE") {
            continue;
        }
        let email = row
            .get("Email")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let name = row
            .get("Name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let alias = row
            .get("Alias")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let account_id = row
            .get("AWSAccountID")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let favorite = row
            .get("Favorite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let account_ref = if !email.is_empty() {
            email.to_string()
        } else if !alias.is_empty() {
            alias.to_string()
        } else if !name.is_empty() {
            name.to_string()
        } else {
            continue;
        };
        let display_name = if !alias.is_empty() {
            alias.to_string()
        } else if !name.is_empty() {
            name.to_string()
        } else {
            account_ref.clone()
        };
        let mut label = display_name;
        if !email.is_empty() {
            label.push_str(&format!(" <{}>", email));
        }
        if !account_id.is_empty() {
            label.push_str(&format!(" [{}]", account_id));
        }
        out.push(IsengardAccountChoice {
            account_ref,
            label,
            favorite,
        });
    }
    out.sort_by(|a, b| {
        b.favorite
            .cmp(&a.favorite)
            .then_with(|| a.label.to_lowercase().cmp(&b.label.to_lowercase()))
    });
    out.dedup_by(|a, b| a.account_ref == b.account_ref);
    Ok(out)
}
