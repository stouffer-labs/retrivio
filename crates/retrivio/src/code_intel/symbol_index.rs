use super::lang_support::{self, LanguageId};
use std::path::Path;

/// An extracted symbol from source code.
#[derive(Debug, Clone)]
pub struct ExtractedSymbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: String, // "function", "method", "class", "struct", "trait", "interface", "enum", "type_alias", "constant", "impl_block"
    pub line_start: usize,
    pub line_end: usize,
    pub signature: String,   // first line or function signature
    pub doc_comment: String, // preceding doc comment
    pub visibility: String,  // "pub", "private", "protected", ""
    pub parent_name: String, // empty for top-level, parent symbol name for nested
}

/// Extract symbols from a source file.
/// Returns a list of symbols found in the file.
pub fn extract_symbols(path: &Path, source: &str) -> Vec<ExtractedSymbol> {
    let lang = match lang_support::language_for_path(path) {
        Some(l) => l,
        None => return Vec::new(),
    };
    let tree = match lang_support::parse_source(lang, source) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let root = tree.root_node();
    let def_types = lang_support::definition_node_types(lang);
    let mut symbols = Vec::new();

    extract_from_node(lang, &root, source, def_types, &mut symbols, "");
    symbols
}

fn extract_from_node(
    lang: LanguageId,
    node: &tree_sitter::Node,
    source: &str,
    def_types: &[&str],
    symbols: &mut Vec<ExtractedSymbol>,
    parent_name: &str,
) {
    for i in 0..node.child_count() {
        let child = match node.child(i) {
            Some(c) => c,
            None => continue,
        };
        let kind = child.kind();

        if !def_types.contains(&kind) {
            continue;
        }

        let line_start = child.start_position().row + 1;
        let line_end = child.end_position().row + 1;

        // Extract the name
        let name = extract_name(lang, &child, source);
        if name.is_empty() {
            // For decorated definitions, look inside
            if kind == "decorated_definition" {
                if let Some(inner) = child.child_by_field_name("definition") {
                    let inner_name = extract_name(lang, &inner, source);
                    if !inner_name.is_empty() {
                        let sym_kind = classify_symbol_kind(lang, inner.kind());
                        let sig = extract_signature(&child, source);
                        let doc = extract_doc_comment(node, i, source);
                        let vis = extract_visibility(lang, &child, source);
                        let qname = if parent_name.is_empty() {
                            inner_name.clone()
                        } else {
                            format!("{}.{}", parent_name, inner_name)
                        };
                        symbols.push(ExtractedSymbol {
                            name: inner_name,
                            qualified_name: qname,
                            kind: sym_kind,
                            line_start,
                            line_end,
                            signature: sig,
                            doc_comment: doc,
                            visibility: vis,
                            parent_name: parent_name.to_string(),
                        });
                    }
                }
            }
            continue;
        }

        let sym_kind = classify_symbol_kind(lang, kind);
        let sig = extract_signature(&child, source);
        let doc = extract_doc_comment(node, i, source);
        let vis = extract_visibility(lang, &child, source);
        let qname = if parent_name.is_empty() {
            name.clone()
        } else {
            format!("{}.{}", parent_name, name)
        };

        symbols.push(ExtractedSymbol {
            name: name.clone(),
            qualified_name: qname,
            kind: sym_kind.clone(),
            line_start,
            line_end,
            signature: sig,
            doc_comment: doc,
            visibility: vis,
            parent_name: parent_name.to_string(),
        });

        // Recurse into bodies of classes, structs, impl blocks to find methods
        if sym_kind == "class"
            || sym_kind == "struct"
            || sym_kind == "impl_block"
            || sym_kind == "trait"
            || sym_kind == "interface"
        {
            if let Some(body) = find_body_node(&child) {
                extract_from_node(lang, &body, source, def_types, symbols, &name);
            }
        }
    }
}

fn extract_name(lang: LanguageId, node: &tree_sitter::Node, source: &str) -> String {
    // Most languages use "name" field
    if let Some(name_node) = node.child_by_field_name("name") {
        return source[name_node.start_byte()..name_node.end_byte()].to_string();
    }
    // Rust impl blocks use "type" field
    if lang == LanguageId::Rust && node.kind() == "impl_item" {
        if let Some(type_node) = node.child_by_field_name("type") {
            return source[type_node.start_byte()..type_node.end_byte()].to_string();
        }
    }
    String::new()
}

fn classify_symbol_kind(lang: LanguageId, node_kind: &str) -> String {
    match lang {
        LanguageId::Python => match node_kind {
            "function_definition" => "function".to_string(),
            "class_definition" => "class".to_string(),
            _ => "declaration".to_string(),
        },
        LanguageId::JavaScript | LanguageId::TypeScript | LanguageId::Tsx => match node_kind {
            "function_declaration" => "function".to_string(),
            "class_declaration" => "class".to_string(),
            "interface_declaration" => "interface".to_string(),
            "type_alias_declaration" => "type_alias".to_string(),
            "enum_declaration" => "enum".to_string(),
            _ => "declaration".to_string(),
        },
        LanguageId::Rust => match node_kind {
            "function_item" => "function".to_string(),
            "struct_item" => "struct".to_string(),
            "enum_item" => "enum".to_string(),
            "trait_item" => "trait".to_string(),
            "impl_item" => "impl_block".to_string(),
            "type_item" => "type_alias".to_string(),
            "mod_item" => "module".to_string(),
            _ => "declaration".to_string(),
        },
        LanguageId::Go => match node_kind {
            "function_declaration" => "function".to_string(),
            "method_declaration" => "method".to_string(),
            "type_declaration" => "type_alias".to_string(),
            _ => "declaration".to_string(),
        },
        LanguageId::Java => match node_kind {
            "method_declaration" => "method".to_string(),
            "class_declaration" => "class".to_string(),
            "interface_declaration" => "interface".to_string(),
            _ => "declaration".to_string(),
        },
        LanguageId::C | LanguageId::Cpp => match node_kind {
            "function_definition" => "function".to_string(),
            "struct_specifier" => "struct".to_string(),
            "class_specifier" => "class".to_string(),
            _ => "declaration".to_string(),
        },
    }
}

/// Extract the first line or function signature from a node.
fn extract_signature(node: &tree_sitter::Node, source: &str) -> String {
    let text = &source[node.start_byte()..node.end_byte()];
    // Take up to the first opening brace or colon (for Python)
    if let Some(pos) = text.find('{') {
        return text[..pos].trim().to_string();
    }
    if let Some(pos) = text.find(':') {
        // For Python functions: `def foo(x, y):`
        let sig = &text[..=pos];
        return sig.trim().to_string();
    }
    // Fall back to first line
    text.lines().next().unwrap_or("").trim().to_string()
}

/// Look for a doc comment immediately preceding a node.
fn extract_doc_comment(parent: &tree_sitter::Node, child_index: usize, source: &str) -> String {
    if child_index == 0 {
        return String::new();
    }
    // Check the sibling immediately before this node
    let prev = match parent.child(child_index - 1) {
        Some(n) => n,
        None => return String::new(),
    };
    let prev_kind = prev.kind();
    if prev_kind == "comment" || prev_kind == "line_comment" || prev_kind == "block_comment" {
        let text = &source[prev.start_byte()..prev.end_byte()];
        // Limit to 500 chars
        return text.chars().take(500).collect();
    }
    String::new()
}

/// Extract visibility from a node (pub, private, etc.).
fn extract_visibility(lang: LanguageId, node: &tree_sitter::Node, source: &str) -> String {
    match lang {
        LanguageId::Rust => {
            // Check for visibility_modifier child
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    if child.kind() == "visibility_modifier" {
                        return source[child.start_byte()..child.end_byte()].to_string();
                    }
                }
            }
            String::new()
        }
        LanguageId::Java => {
            // Check for modifiers
            if let Some(mods) = node.child_by_field_name("modifiers") {
                let text = source[mods.start_byte()..mods.end_byte()].to_string();
                if text.contains("public") {
                    return "public".to_string();
                }
                if text.contains("private") {
                    return "private".to_string();
                }
                if text.contains("protected") {
                    return "protected".to_string();
                }
            }
            String::new()
        }
        LanguageId::TypeScript | LanguageId::Tsx => {
            let text = &source[node.start_byte()..node.end_byte()];
            let first_line = text.lines().next().unwrap_or("");
            if first_line.starts_with("export ") {
                return "export".to_string();
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Find the body/block node within a definition.
fn find_body_node<'a>(node: &tree_sitter::Node<'a>) -> Option<tree_sitter::Node<'a>> {
    node.child_by_field_name("body").or_else(|| {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                let k = child.kind();
                if k == "declaration_list"
                    || k == "field_declaration_list"
                    || k == "class_body"
                    || k == "block"
                {
                    return Some(child);
                }
            }
        }
        None
    })
}
