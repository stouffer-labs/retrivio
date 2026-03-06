// Import/dependency graph parsing and resolution.
//
// Extracts import statements from source files using tree-sitter,
// then resolves them to file paths within the project.

use super::lang_support::{self, LanguageId};
use std::path::{Path, PathBuf};

/// A raw import extracted from source code.
#[derive(Debug, Clone)]
pub struct RawImport {
    pub import_kind: String, // module, named, wildcard, default, side_effect, include, reexport
    pub raw_specifier: String, // the raw module path (e.g., "os.path", "./utils", "crate::auth")
    pub imported_names: Vec<String>, // specific names imported (e.g., ["join", "dirname"])
    pub line_number: usize,
}

/// Extract all import statements from a source file.
pub fn extract_imports(path: &Path, source: &str) -> Vec<RawImport> {
    let lang = match lang_support::language_for_path(path) {
        Some(l) => l,
        None => return Vec::new(),
    };
    let tree = match lang_support::parse_source(lang, source) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let root = tree.root_node();
    let import_types = lang_support::import_node_types(lang);
    let mut imports = Vec::new();

    for i in 0..root.child_count() {
        let child = match root.child(i) {
            Some(c) => c,
            None => continue,
        };
        let kind = child.kind();

        if !import_types.contains(&kind) {
            // For JS/TS, imports can be inside export_statement
            if (lang == LanguageId::JavaScript
                || lang == LanguageId::TypeScript
                || lang == LanguageId::Tsx)
                && kind == "export_statement"
            {
                // Look for re-exports: export { foo } from './bar'
                for j in 0..child.child_count() {
                    if let Some(inner) = child.child(j) {
                        if inner.kind() == "import_statement" || inner.kind() == "export_clause" {
                            if let Some(src) = child.child_by_field_name("source") {
                                let specifier = node_text(&src, source)
                                    .trim_matches(|c| c == '\'' || c == '"')
                                    .to_string();
                                if !specifier.is_empty() {
                                    let names = extract_js_import_names(&child, source);
                                    imports.push(RawImport {
                                        import_kind: "reexport".to_string(),
                                        raw_specifier: specifier,
                                        imported_names: names,
                                        line_number: child.start_position().row + 1,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            continue;
        }

        match lang {
            LanguageId::Python => extract_python_import(&child, source, &mut imports),
            LanguageId::JavaScript | LanguageId::TypeScript | LanguageId::Tsx => {
                extract_js_import(&child, source, &mut imports)
            }
            LanguageId::Rust => extract_rust_import(&child, source, &mut imports),
            LanguageId::Go => extract_go_import(&child, source, &mut imports),
            LanguageId::Java => extract_java_import(&child, source, &mut imports),
            LanguageId::C | LanguageId::Cpp => extract_c_include(&child, source, &mut imports),
        }
    }

    imports
}

/// Attempt to resolve a raw import specifier to a file path within the project.
/// Returns the resolved relative path if found, or empty string if unresolvable.
pub fn resolve_import_path(
    lang: LanguageId,
    raw_specifier: &str,
    source_file: &Path,
    project_root: &Path,
    project_files: &[String], // relative paths of all files in the project
) -> String {
    match lang {
        LanguageId::Python => resolve_python_import(raw_specifier, project_root, project_files),
        LanguageId::JavaScript | LanguageId::TypeScript | LanguageId::Tsx => {
            resolve_js_import(raw_specifier, source_file, project_root, project_files)
        }
        LanguageId::Rust => resolve_rust_import(raw_specifier, project_root, project_files),
        LanguageId::Go => resolve_go_import(raw_specifier, project_root, project_files),
        LanguageId::Java => resolve_java_import(raw_specifier, project_root, project_files),
        LanguageId::C | LanguageId::Cpp => {
            resolve_c_include(raw_specifier, source_file, project_root, project_files)
        }
    }
}

// ─── Helper: get node text ──────────────────────────────────────────────────

fn node_text<'a>(node: &tree_sitter::Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

// ─── Python ─────────────────────────────────────────────────────────────────

fn extract_python_import(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;
    let kind = node.kind();

    if kind == "import_statement" {
        // `import os.path` or `import foo, bar`
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "dotted_name" {
                    imports.push(RawImport {
                        import_kind: "module".to_string(),
                        raw_specifier: node_text(&child, source).to_string(),
                        imported_names: Vec::new(),
                        line_number: line,
                    });
                }
            }
        }
    } else if kind == "import_from_statement" {
        // `from os.path import join, dirname` or `from . import foo`
        let mut module = String::new();
        let mut names = Vec::new();

        if let Some(mod_node) = node.child_by_field_name("module_name") {
            module = node_text(&mod_node, source).to_string();
        }
        // Handle relative imports: count leading dots
        let text = node_text(node, source);
        let relative_dots: String = text
            .strip_prefix("from")
            .unwrap_or("")
            .trim_start()
            .chars()
            .take_while(|c| *c == '.')
            .collect();
        if !relative_dots.is_empty() {
            module = format!("{}{}", relative_dots, module);
        }

        // Extract imported names
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "dotted_name" || child.kind() == "aliased_import" {
                    let name = if child.kind() == "aliased_import" {
                        child
                            .child_by_field_name("name")
                            .map(|n| node_text(&n, source).to_string())
                            .unwrap_or_default()
                    } else {
                        node_text(&child, source).to_string()
                    };
                    if name != module {
                        names.push(name);
                    }
                }
                if child.kind() == "wildcard_import" {
                    imports.push(RawImport {
                        import_kind: "wildcard".to_string(),
                        raw_specifier: module.clone(),
                        imported_names: Vec::new(),
                        line_number: line,
                    });
                    return;
                }
            }
        }

        let ik = if names.is_empty() { "module" } else { "named" };
        if !module.is_empty() {
            imports.push(RawImport {
                import_kind: ik.to_string(),
                raw_specifier: module,
                imported_names: names,
                line_number: line,
            });
        }
    }
}

fn resolve_python_import(specifier: &str, _root: &Path, files: &[String]) -> String {
    // Convert dotted module path to file path candidates
    // "os.path" -> "os/path.py" or "os/path/__init__.py"
    let spec = specifier.trim_start_matches('.');
    if spec.is_empty() {
        return String::new();
    }
    let parts: String = spec.replace('.', "/");
    let candidates = [
        format!("{}.py", parts),
        format!("{}/__init__.py", parts),
        format!("src/{}.py", parts),
        format!("src/{}/__init__.py", parts),
    ];
    for c in &candidates {
        if files
            .iter()
            .any(|f| f == c || f.ends_with(&format!("/{}", c)))
        {
            return c.clone();
        }
    }
    String::new()
}

// ─── JavaScript / TypeScript ────────────────────────────────────────────────

fn extract_js_import(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;

    // Get the source/module specifier
    let source_node = node.child_by_field_name("source");
    let specifier = match source_node {
        Some(s) => node_text(&s, source)
            .trim_matches(|c| c == '\'' || c == '"' || c == '`')
            .to_string(),
        None => return,
    };

    if specifier.is_empty() {
        return;
    }

    let names = extract_js_import_names(node, source);

    let ik = if names.is_empty() {
        // `import './styles.css'` -> side_effect
        // Check for default import: `import React from 'react'`
        let text = node_text(node, source);
        if text.contains(" from ") {
            "default"
        } else {
            "side_effect"
        }
    } else {
        "named"
    };

    imports.push(RawImport {
        import_kind: ik.to_string(),
        raw_specifier: specifier,
        imported_names: names,
        line_number: line,
    });
}

fn extract_js_import_names(node: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            match child.kind() {
                "import_clause" | "named_imports" => {
                    // Recurse into the clause
                    for j in 0..child.child_count() {
                        if let Some(inner) = child.child(j) {
                            if inner.kind() == "import_specifier" {
                                if let Some(name) = inner.child_by_field_name("name") {
                                    names.push(node_text(&name, source).to_string());
                                }
                            } else if inner.kind() == "named_imports" {
                                for k in 0..inner.child_count() {
                                    if let Some(spec) = inner.child(k) {
                                        if spec.kind() == "import_specifier" {
                                            if let Some(name) = spec.child_by_field_name("name") {
                                                names.push(node_text(&name, source).to_string());
                                            }
                                        }
                                    }
                                }
                            } else if inner.kind() == "identifier" {
                                names.push(node_text(&inner, source).to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    names
}

fn resolve_js_import(
    specifier: &str,
    source_file: &Path,
    _root: &Path,
    files: &[String],
) -> String {
    // Only resolve relative imports (./foo, ../bar)
    if !specifier.starts_with('.') {
        return String::new();
    }

    let source_dir = source_file.parent().unwrap_or(Path::new(""));
    let resolved = source_dir.join(specifier);
    let normalized = normalize_relative_path(&resolved);

    // Try with various extensions
    let extensions = [
        "",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        "/index.ts",
        "/index.tsx",
        "/index.js",
    ];
    for ext in &extensions {
        let candidate = format!("{}{}", normalized, ext);
        if files
            .iter()
            .any(|f| f == &candidate || f.ends_with(&format!("/{}", candidate)))
        {
            return candidate;
        }
    }
    String::new()
}

// ─── Rust ───────────────────────────────────────────────────────────────────

fn extract_rust_import(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;
    let text = node_text(node, source).to_string();

    if node.kind() == "use_declaration" {
        // `use crate::auth::middleware;` or `use std::collections::HashMap;`
        let path = text
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim()
            .to_string();

        let (ik, names) = if path.contains('{') {
            // `use crate::auth::{foo, bar};`
            let names: Vec<String> = path
                .split('{')
                .nth(1)
                .unwrap_or("")
                .trim_end_matches('}')
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            ("named", names)
        } else if path.ends_with('*') {
            ("wildcard", Vec::new())
        } else {
            ("module", Vec::new())
        };

        imports.push(RawImport {
            import_kind: ik.to_string(),
            raw_specifier: path
                .split('{')
                .next()
                .unwrap_or(&path)
                .trim_end_matches("::")
                .to_string(),
            imported_names: names,
            line_number: line,
        });
    } else if node.kind() == "extern_crate_declaration" {
        let crate_name = text
            .trim_start_matches("extern crate ")
            .trim_end_matches(';')
            .trim()
            .to_string();
        imports.push(RawImport {
            import_kind: "module".to_string(),
            raw_specifier: crate_name,
            imported_names: Vec::new(),
            line_number: line,
        });
    }
}

fn resolve_rust_import(specifier: &str, _root: &Path, files: &[String]) -> String {
    // Resolve crate:: paths to src/ files
    // "crate::auth::middleware" -> "src/auth/middleware.rs" or "src/auth/middleware/mod.rs"
    let spec = if let Some(stripped) = specifier.strip_prefix("crate::") {
        stripped
    } else if specifier.starts_with("super::") || specifier.starts_with("self::") {
        // Can't reliably resolve without source file context
        return String::new();
    } else {
        // External crate, can't resolve
        return String::new();
    };

    let parts: String = spec.replace("::", "/");
    let candidates = [format!("src/{}.rs", parts), format!("src/{}/mod.rs", parts)];
    for c in &candidates {
        if files.iter().any(|f| f == c) {
            return c.clone();
        }
    }
    String::new()
}

// ─── Go ─────────────────────────────────────────────────────────────────────

fn extract_go_import(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;

    // Go import can be single or grouped:
    // import "fmt"
    // import ( "fmt"; "os" )
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == "import_spec" || child.kind() == "interpreted_string_literal" {
                let text = node_text(&child, source).trim_matches('"').to_string();
                if !text.is_empty() {
                    imports.push(RawImport {
                        import_kind: "module".to_string(),
                        raw_specifier: text,
                        imported_names: Vec::new(),
                        line_number: line,
                    });
                }
            } else if child.kind() == "import_spec_list" {
                for j in 0..child.child_count() {
                    if let Some(spec) = child.child(j) {
                        if spec.kind() == "import_spec" {
                            // The path is in the "path" field
                            if let Some(path_node) = spec.child_by_field_name("path") {
                                let text =
                                    node_text(&path_node, source).trim_matches('"').to_string();
                                if !text.is_empty() {
                                    imports.push(RawImport {
                                        import_kind: "module".to_string(),
                                        raw_specifier: text,
                                        imported_names: Vec::new(),
                                        line_number: line,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn resolve_go_import(specifier: &str, _root: &Path, files: &[String]) -> String {
    // Go imports are typically package paths. For local packages, check if
    // any file path contains the specifier as a directory component.
    let last_segment = specifier.rsplit('/').next().unwrap_or(specifier);
    for f in files {
        if f.contains(&format!("/{}/", last_segment))
            || f.starts_with(&format!("{}/", last_segment))
        {
            return f.clone();
        }
    }
    String::new()
}

// ─── Java ───────────────────────────────────────────────────────────────────

fn extract_java_import(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;
    let text = node_text(node, source).to_string();

    let specifier = text
        .trim_start_matches("import ")
        .trim_start_matches("static ")
        .trim_end_matches(';')
        .trim()
        .to_string();

    let ik = if specifier.ends_with(".*") {
        "wildcard"
    } else {
        "named"
    };

    imports.push(RawImport {
        import_kind: ik.to_string(),
        raw_specifier: specifier,
        imported_names: Vec::new(),
        line_number: line,
    });
}

fn resolve_java_import(specifier: &str, _root: &Path, files: &[String]) -> String {
    // Convert Java package path to file path
    // "com.example.auth.Middleware" -> find file containing "com/example/auth/Middleware.java"
    let path = specifier.replace('.', "/");
    let candidate = format!("{}.java", path);
    for f in files {
        if f.ends_with(&candidate) || f == &candidate {
            return f.clone();
        }
    }
    String::new()
}

// ─── C/C++ ──────────────────────────────────────────────────────────────────

fn extract_c_include(node: &tree_sitter::Node, source: &str, imports: &mut Vec<RawImport>) {
    let line = node.start_position().row + 1;

    // #include "foo.h" (local) or #include <stdio.h> (system)
    if let Some(path_node) = node.child_by_field_name("path") {
        let text = node_text(&path_node, source);
        let (ik, specifier) = if text.starts_with('"') {
            ("include", text.trim_matches('"').to_string())
        } else if text.starts_with('<') {
            // System include, typically not resolvable within project
            (
                "include",
                text.trim_matches(|c| c == '<' || c == '>').to_string(),
            )
        } else {
            ("include", text.to_string())
        };

        imports.push(RawImport {
            import_kind: ik.to_string(),
            raw_specifier: specifier,
            imported_names: Vec::new(),
            line_number: line,
        });
    } else {
        // Fallback: parse the text directly
        let text = node_text(node, source);
        let include_path = text
            .trim_start_matches("#include")
            .trim()
            .trim_matches(|c: char| c == '"' || c == '<' || c == '>')
            .to_string();
        if !include_path.is_empty() {
            imports.push(RawImport {
                import_kind: "include".to_string(),
                raw_specifier: include_path,
                imported_names: Vec::new(),
                line_number: line,
            });
        }
    }
}

fn resolve_c_include(
    specifier: &str,
    source_file: &Path,
    _root: &Path,
    files: &[String],
) -> String {
    // For local includes, check relative to source file and project root
    let source_dir = source_file.parent().unwrap_or(Path::new(""));

    // Try relative to source file
    let relative = source_dir.join(specifier);
    let normalized = normalize_relative_path(&relative);
    if files.iter().any(|f| f == &normalized) {
        return normalized;
    }

    // Try from project root
    if files
        .iter()
        .any(|f| f == specifier || f.ends_with(&format!("/{}", specifier)))
    {
        return specifier.to_string();
    }

    String::new()
}

// ─── Utility ────────────────────────────────────────────────────────────────

/// Normalize a relative path by resolving `.` and `..` components.
fn normalize_relative_path(path: &Path) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::Normal(s) => {
                if let Some(s) = s.to_str() {
                    parts.push(s);
                }
            }
            _ => {}
        }
    }
    parts.join("/")
}
