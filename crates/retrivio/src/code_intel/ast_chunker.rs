use super::lang_support::{self, LanguageId};
use std::path::Path;

/// What kind of semantic unit a chunk represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkKind {
    /// Import/use/include block at top of file.
    ImportBlock,
    /// Leading comments, module docstrings, file preamble.
    Preamble,
    /// A complete function/method definition.
    Function,
    /// A class/struct/enum/trait/interface header (without method bodies).
    TypeHeader,
    /// A method within a class/struct/impl block.
    Method,
    /// A standalone constant, type alias, or other declaration.
    Declaration,
    /// Fallback: a text window chunk (used for unsupported languages or oversized code).
    TextWindow,
}

/// A semantically meaningful chunk of source code produced by AST analysis.
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// The source text of this chunk.
    pub text: String,
    /// What kind of code unit this chunk represents.
    pub kind: ChunkKind,
    /// The primary symbol name (e.g., function name, class name). Empty for imports/preamble.
    pub symbol_name: String,
    /// Parent context (e.g., "impl AuthMiddleware" for a method inside that impl block).
    pub parent_context: String,
    /// 1-based line number where this chunk starts in the original file.
    pub line_start: usize,
    /// 1-based line number where this chunk ends in the original file.
    pub line_end: usize,
    /// Contextual header for embedding (e.g., "// File: src/auth.rs\n// Method: validate()").
    pub context_header: String,
}

/// Analyze a source file and produce semantic chunks.
///
/// This is the main entry point. It:
/// 1. Detects the language from the file extension
/// 2. Parses with tree-sitter if the language is supported
/// 3. Walks the AST to produce semantic chunks
/// 4. Falls back to character-window chunking for unsupported languages or parse failures
///
/// `max_chunk_chars`: maximum characters per chunk (default 1500 for code files).
/// `rel_path`: relative path for context header generation.
pub fn analyze_file(
    path: &Path,
    source: &str,
    rel_path: &str,
    max_chunk_chars: usize,
) -> Vec<SemanticChunk> {
    // Try AST-based chunking for supported languages
    if let Some(lang) = lang_support::language_for_path(path) {
        if let Some(tree) = lang_support::parse_source(lang, source) {
            let chunks = chunk_from_ast(lang, source, &tree, rel_path, max_chunk_chars);
            if !chunks.is_empty() {
                return chunks;
            }
        }
    }

    // Fallback: character-window chunking (same as existing chunk_text behavior,
    // but preserving whitespace and producing SemanticChunk structs)
    fallback_chunk(source, rel_path, max_chunk_chars)
}

/// Walk the AST and produce semantic chunks.
fn chunk_from_ast(
    lang: LanguageId,
    source: &str,
    tree: &tree_sitter::Tree,
    rel_path: &str,
    max_chunk_chars: usize,
) -> Vec<SemanticChunk> {
    let root = tree.root_node();
    let def_types = lang_support::definition_node_types(lang);
    let import_types = lang_support::import_node_types(lang);

    let mut chunks: Vec<SemanticChunk> = Vec::new();
    let mut import_lines: Vec<(usize, usize)> = Vec::new(); // (start_byte, end_byte)
    let mut preamble_end: usize = 0;
    let mut past_header = false; // true once we hit a non-import, non-comment node

    // First pass: identify imports and preamble
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        let kind = child.kind();

        // Only collect imports from the top-of-file header region.
        // Scoped `use` inside function bodies can surface as root-level nodes
        // in tree-sitter for very large files — ignore those.
        if import_types.contains(&kind) {
            if !past_header {
                import_lines.push((child.start_byte(), child.end_byte()));
            }
            continue;
        }

        // Leading comments before any definitions count as preamble
        if kind == "comment" || kind == "line_comment" || kind == "block_comment" {
            if chunks.is_empty() && import_lines.is_empty() {
                preamble_end = child.end_byte();
            }
            continue;
        }

        // Preamble also includes module docstrings in Python
        if lang == LanguageId::Python && kind == "expression_statement" {
            if let Some(first_child) = child.child(0) {
                if first_child.kind() == "string" && chunks.is_empty() && import_lines.is_empty() {
                    preamble_end = child.end_byte();
                    continue;
                }
            }
        }

        // Once we hit a definition, we're past the header region
        past_header = true;

        // If this is a definition node, extract it
        if def_types.contains(&kind) {
            let text = &source[child.start_byte()..child.end_byte()];
            let line_start = child.start_position().row + 1;
            let line_end = child.end_position().row + 1;

            let (symbol_name, parent_context) = extract_symbol_info(lang, &child, source);
            let chunk_kind = classify_node(lang, &child);

            // Size check: if the definition fits in max_chunk_chars, emit as one chunk.
            // If it's too large (e.g., a class with many methods), split it.
            if text.len() <= max_chunk_chars {
                chunks.push(SemanticChunk {
                    text: text.to_string(),
                    kind: chunk_kind,
                    symbol_name,
                    parent_context,
                    line_start,
                    line_end,
                    context_header: String::new(), // filled in later by chunk_header module
                });
            } else {
                // Large definition: try to split at method/function boundaries within it
                let sub_chunks =
                    split_large_definition(lang, &child, source, rel_path, max_chunk_chars);
                if sub_chunks.is_empty() {
                    // Can't split further — fall back to line-based splitting
                    let line_chunks = split_at_blank_lines(text, max_chunk_chars, line_start);
                    for lc in line_chunks {
                        chunks.push(SemanticChunk {
                            text: lc.0,
                            kind: ChunkKind::TextWindow,
                            symbol_name: symbol_name.clone(),
                            parent_context: parent_context.clone(),
                            line_start: lc.1,
                            line_end: lc.2,
                            context_header: String::new(),
                        });
                    }
                } else {
                    chunks.extend(sub_chunks);
                }
            }
        }
    }

    // Emit import block — split at blank lines if oversized
    if !import_lines.is_empty() {
        let first_start = import_lines.first().unwrap().0;
        let last_end = import_lines.last().unwrap().1;
        let text = &source[first_start..last_end];
        let line_start = byte_to_line(source, first_start);
        if text.len() <= max_chunk_chars {
            let line_end = byte_to_line(source, last_end);
            chunks.insert(
                0,
                SemanticChunk {
                    text: text.to_string(),
                    kind: ChunkKind::ImportBlock,
                    symbol_name: String::new(),
                    parent_context: String::new(),
                    line_start,
                    line_end,
                    context_header: String::new(),
                },
            );
        } else {
            let parts = split_at_blank_lines(text, max_chunk_chars, line_start);
            for (i, lc) in parts.into_iter().enumerate() {
                chunks.insert(
                    i,
                    SemanticChunk {
                        text: lc.0,
                        kind: ChunkKind::ImportBlock,
                        symbol_name: String::new(),
                        parent_context: String::new(),
                        line_start: lc.1,
                        line_end: lc.2,
                        context_header: String::new(),
                    },
                );
            }
        }
    }

    // Emit preamble — split at blank lines if oversized
    if preamble_end > 0 {
        let text = source[..preamble_end].trim();
        if !text.is_empty() {
            if text.len() <= max_chunk_chars {
                let line_end = byte_to_line(source, preamble_end);
                chunks.insert(
                    0,
                    SemanticChunk {
                        text: text.to_string(),
                        kind: ChunkKind::Preamble,
                        symbol_name: String::new(),
                        parent_context: String::new(),
                        line_start: 1,
                        line_end,
                        context_header: String::new(),
                    },
                );
            } else {
                let parts = split_at_blank_lines(text, max_chunk_chars, 1);
                for (i, lc) in parts.into_iter().enumerate() {
                    chunks.insert(
                        i,
                        SemanticChunk {
                            text: lc.0,
                            kind: ChunkKind::Preamble,
                            symbol_name: String::new(),
                            parent_context: String::new(),
                            line_start: lc.1,
                            line_end: lc.2,
                            context_header: String::new(),
                        },
                    );
                }
            }
        }
    }

    chunks
}

/// Extract the symbol name and parent context from an AST node.
fn extract_symbol_info(
    lang: LanguageId,
    node: &tree_sitter::Node,
    source: &str,
) -> (String, String) {
    let mut symbol_name = String::new();
    let parent_context = String::new();

    // Find the "name" child node — most languages use a child named "name"
    if let Some(name_node) = node.child_by_field_name("name") {
        symbol_name = source[name_node.start_byte()..name_node.end_byte()].to_string();
    }

    // For decorated definitions (Python), look inside the inner definition
    if lang == LanguageId::Python && node.kind() == "decorated_definition" {
        if let Some(def_node) = node.child_by_field_name("definition") {
            if let Some(name_node) = def_node.child_by_field_name("name") {
                symbol_name = source[name_node.start_byte()..name_node.end_byte()].to_string();
            }
        }
    }

    // For Rust impl blocks, extract the type being implemented
    if lang == LanguageId::Rust && node.kind() == "impl_item" {
        if let Some(type_node) = node.child_by_field_name("type") {
            symbol_name = source[type_node.start_byte()..type_node.end_byte()].to_string();
        }
    }

    (symbol_name, parent_context)
}

/// Classify a definition node into a ChunkKind.
fn classify_node(lang: LanguageId, node: &tree_sitter::Node) -> ChunkKind {
    let kind = node.kind();
    match lang {
        LanguageId::Python => match kind {
            "function_definition" => ChunkKind::Function,
            "class_definition" => ChunkKind::TypeHeader,
            "decorated_definition" => {
                // Check if it decorates a function or class
                if let Some(inner) = node.child_by_field_name("definition") {
                    match inner.kind() {
                        "function_definition" => ChunkKind::Function,
                        "class_definition" => ChunkKind::TypeHeader,
                        _ => ChunkKind::Declaration,
                    }
                } else {
                    ChunkKind::Declaration
                }
            }
            _ => ChunkKind::Declaration,
        },
        LanguageId::JavaScript | LanguageId::TypeScript | LanguageId::Tsx => match kind {
            "function_declaration" => ChunkKind::Function,
            "class_declaration" => ChunkKind::TypeHeader,
            "interface_declaration" | "type_alias_declaration" | "enum_declaration" => {
                ChunkKind::TypeHeader
            }
            _ => ChunkKind::Declaration,
        },
        LanguageId::Rust => match kind {
            "function_item" => ChunkKind::Function,
            "struct_item" | "enum_item" | "trait_item" | "type_item" => ChunkKind::TypeHeader,
            "impl_item" => ChunkKind::TypeHeader,
            "mod_item" => ChunkKind::Declaration,
            _ => ChunkKind::Declaration,
        },
        LanguageId::Go => match kind {
            "function_declaration" | "method_declaration" => ChunkKind::Function,
            "type_declaration" => ChunkKind::TypeHeader,
            _ => ChunkKind::Declaration,
        },
        LanguageId::Java => match kind {
            "method_declaration" => ChunkKind::Function,
            "class_declaration" | "interface_declaration" => ChunkKind::TypeHeader,
            _ => ChunkKind::Declaration,
        },
        LanguageId::C | LanguageId::Cpp => match kind {
            "function_definition" => ChunkKind::Function,
            "struct_specifier" | "class_specifier" => ChunkKind::TypeHeader,
            _ => ChunkKind::Declaration,
        },
    }
}

/// Split a large definition (e.g., a class with many methods) into sub-chunks.
/// Emits each method/inner function as its own chunk.
fn split_large_definition(
    lang: LanguageId,
    node: &tree_sitter::Node,
    source: &str,
    _rel_path: &str,
    max_chunk_chars: usize,
) -> Vec<SemanticChunk> {
    let mut sub_chunks = Vec::new();
    let parent_name = {
        if let Some(name_node) = node.child_by_field_name("name") {
            source[name_node.start_byte()..name_node.end_byte()].to_string()
        } else if let Some(type_node) = node.child_by_field_name("type") {
            // Rust impl blocks
            source[type_node.start_byte()..type_node.end_byte()].to_string()
        } else {
            String::new()
        }
    };

    let parent_context = format!("{} {}", node.kind(), parent_name)
        .trim()
        .to_string();

    // Find the body node — different languages use different field names.
    // We check named field "body" first, then scan children for block-like nodes
    // (Rust impl/trait blocks use "declaration_list" instead of "body").
    let body_node = node.child_by_field_name("body").or_else(|| {
        find_child_by_kinds(
            node,
            &["declaration_list", "field_declaration_list", "block"],
        )
    });

    let body = match body_node {
        Some(b) => b,
        None => return sub_chunks,
    };

    // Extract header (everything before the body)
    let header_text = &source[node.start_byte()..body.start_byte()];
    if !header_text.trim().is_empty() {
        sub_chunks.push(SemanticChunk {
            text: header_text.trim_end().to_string(),
            kind: ChunkKind::TypeHeader,
            symbol_name: parent_name.clone(),
            parent_context: String::new(),
            line_start: node.start_position().row + 1,
            line_end: body.start_position().row + 1,
            context_header: String::new(),
        });
    }

    // Extract each child definition inside the body as its own chunk
    let inner_def_types = lang_support::definition_node_types(lang);
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        let child_kind = child.kind();
        // Check for inner definitions (methods, nested functions, etc.)
        let is_inner_def = inner_def_types.contains(&child_kind)
            || child_kind == "function_definition"
            || child_kind == "function_item"
            || child_kind == "method_declaration"
            || child_kind == "method_definition";

        if is_inner_def {
            let text = &source[child.start_byte()..child.end_byte()];
            let method_name = if let Some(n) = child.child_by_field_name("name") {
                source[n.start_byte()..n.end_byte()].to_string()
            } else {
                String::new()
            };

            if text.len() <= max_chunk_chars {
                sub_chunks.push(SemanticChunk {
                    text: text.to_string(),
                    kind: ChunkKind::Method,
                    symbol_name: method_name,
                    parent_context: parent_context.clone(),
                    line_start: child.start_position().row + 1,
                    line_end: child.end_position().row + 1,
                    context_header: String::new(),
                });
            } else {
                // Even the method is too large — split at blank lines
                let line_start = child.start_position().row + 1;
                for lc in split_at_blank_lines(text, max_chunk_chars, line_start) {
                    sub_chunks.push(SemanticChunk {
                        text: lc.0,
                        kind: ChunkKind::TextWindow,
                        symbol_name: method_name.clone(),
                        parent_context: parent_context.clone(),
                        line_start: lc.1,
                        line_end: lc.2,
                        context_header: String::new(),
                    });
                }
            }
        }
    }

    sub_chunks
}

/// Split text at blank line boundaries, respecting max_chunk_chars.
/// Returns (text, line_start, line_end) tuples.
fn split_at_blank_lines(
    text: &str,
    max_chars: usize,
    base_line: usize,
) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut chunk_start_line = base_line;
    let mut current_line = base_line;

    for line in &lines {
        let would_be = current.len() + line.len() + 1; // +1 for newline
        if would_be > max_chars && !current.is_empty() {
            let end_line = current_line.saturating_sub(1).max(chunk_start_line);
            chunks.push((current.trim_end().to_string(), chunk_start_line, end_line));
            current = String::new();
            chunk_start_line = current_line;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
        current_line += 1;
    }

    if !current.trim().is_empty() {
        chunks.push((
            current.trim_end().to_string(),
            chunk_start_line,
            current_line.saturating_sub(1).max(chunk_start_line),
        ));
    }

    chunks
}

/// Fallback chunker for unsupported languages. Preserves whitespace (unlike the
/// original chunk_text which collapsed all whitespace). Produces TextWindow chunks.
fn fallback_chunk(source: &str, _rel_path: &str, max_chars: usize) -> Vec<SemanticChunk> {
    let overlap = max_chars / 6; // ~16% overlap
    let step = max_chars.saturating_sub(overlap).max(1);
    let chars: Vec<char> = source.chars().collect();
    let n = chars.len();

    if n == 0 {
        return Vec::new();
    }

    if n <= max_chars {
        let line_end = source.lines().count().max(1);
        return vec![SemanticChunk {
            text: source.to_string(),
            kind: ChunkKind::TextWindow,
            symbol_name: String::new(),
            parent_context: String::new(),
            line_start: 1,
            line_end,
            context_header: String::new(),
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let max_chunks = 28; // same limit as existing code

    while start < n && chunks.len() < max_chunks {
        let end = (start + max_chars).min(n);
        let window: String = chars[start..end].iter().collect();

        // Try to break at a newline boundary
        let split_text = if end < n {
            if let Some(pos) = window.rfind('\n') {
                if pos > max_chars / 2 {
                    window[..pos].to_string()
                } else {
                    window.clone()
                }
            } else {
                window.clone()
            }
        } else {
            window.clone()
        };

        // Count lines up to this point for line_start
        let byte_start = chars[..start].iter().collect::<String>().len();
        let line_start = source[..byte_start].matches('\n').count() + 1;
        let line_end = line_start + split_text.matches('\n').count();

        if !split_text.trim().is_empty() {
            chunks.push(SemanticChunk {
                text: split_text,
                kind: ChunkKind::TextWindow,
                symbol_name: String::new(),
                parent_context: String::new(),
                line_start,
                line_end,
                context_header: String::new(),
            });
        }

        if end >= n {
            break;
        }
        start += step;
    }

    chunks
}

/// Find a child node whose kind matches one of the given kinds.
/// Uses index-based iteration to avoid cursor lifetime issues.
fn find_child_by_kinds<'a>(
    node: &tree_sitter::Node<'a>,
    kinds: &[&str],
) -> Option<tree_sitter::Node<'a>> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if kinds.contains(&child.kind()) {
                return Some(child);
            }
        }
    }
    None
}

/// Convert a byte offset to a 1-based line number.
fn byte_to_line(source: &str, byte_offset: usize) -> usize {
    let clamped = byte_offset.min(source.len());
    source[..clamped].matches('\n').count() + 1
}
