use std::cell::RefCell;
use std::path::Path;

/// Supported language identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageId {
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Rust,
    Go,
    Java,
    C,
    Cpp,
}

/// Detect language from a file extension (lowercase, without dot).
pub fn language_for_extension(ext: &str) -> Option<LanguageId> {
    match ext {
        "py" | "pyi" | "pyw" => Some(LanguageId::Python),
        "js" | "mjs" | "cjs" | "jsx" => Some(LanguageId::JavaScript),
        "ts" | "mts" | "cts" => Some(LanguageId::TypeScript),
        "tsx" => Some(LanguageId::Tsx),
        "rs" => Some(LanguageId::Rust),
        "go" => Some(LanguageId::Go),
        "java" => Some(LanguageId::Java),
        "c" | "h" => Some(LanguageId::C),
        "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "hh" => Some(LanguageId::Cpp),
        _ => None,
    }
}

/// Detect language from a full file path.
pub fn language_for_path(path: &Path) -> Option<LanguageId> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    language_for_extension(&ext)
}

/// Get the tree-sitter Language for a given LanguageId.
pub fn tree_sitter_language(lang: LanguageId) -> tree_sitter::Language {
    match lang {
        LanguageId::Python => tree_sitter_python::LANGUAGE.into(),
        LanguageId::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        LanguageId::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        LanguageId::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
        LanguageId::Rust => tree_sitter_rust::LANGUAGE.into(),
        LanguageId::Go => tree_sitter_go::LANGUAGE.into(),
        LanguageId::Java => tree_sitter_java::LANGUAGE.into(),
        LanguageId::C => tree_sitter_c::LANGUAGE.into(),
        LanguageId::Cpp => tree_sitter_cpp::LANGUAGE.into(),
    }
}

/// AST node types that represent top-level definitions (functions, classes, structs, etc.).
/// These are the nodes we want to extract as individual semantic chunks.
pub fn definition_node_types(lang: LanguageId) -> &'static [&'static str] {
    match lang {
        LanguageId::Python => &[
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],
        LanguageId::JavaScript => &[
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
        ],
        LanguageId::TypeScript | LanguageId::Tsx => &[
            "function_declaration",
            "class_declaration",
            "export_statement",
            "lexical_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        ],
        LanguageId::Rust => &[
            "function_item",
            "struct_item",
            "enum_item",
            "impl_item",
            "trait_item",
            "mod_item",
            "type_item",
        ],
        LanguageId::Go => &[
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ],
        LanguageId::Java => &[
            "method_declaration",
            "class_declaration",
            "interface_declaration",
        ],
        LanguageId::C => &["function_definition", "struct_specifier"],
        LanguageId::Cpp => &["function_definition", "struct_specifier", "class_specifier"],
    }
}

/// AST node types that represent import/include statements.
pub fn import_node_types(lang: LanguageId) -> &'static [&'static str] {
    match lang {
        LanguageId::Python => &["import_statement", "import_from_statement"],
        LanguageId::JavaScript => &["import_statement"],
        LanguageId::TypeScript | LanguageId::Tsx => &["import_statement"],
        LanguageId::Rust => &["use_declaration", "extern_crate_declaration"],
        LanguageId::Go => &["import_declaration"],
        LanguageId::Java => &["import_declaration"],
        LanguageId::C | LanguageId::Cpp => &["preproc_include"],
    }
}

/// Thread-local parser pool. tree_sitter::Parser is not Send, so each thread
/// gets its own instance. We cache it to avoid re-allocation on every file.
thread_local! {
    static PARSER: RefCell<tree_sitter::Parser> = RefCell::new(tree_sitter::Parser::new());
}

/// Parse source code with the appropriate tree-sitter grammar.
/// Returns None if parsing fails.
pub fn parse_source(lang: LanguageId, source: &str) -> Option<tree_sitter::Tree> {
    PARSER.with(|parser_cell| {
        let mut parser = parser_cell.borrow_mut();
        let ts_lang = tree_sitter_language(lang);
        parser.set_language(&ts_lang).ok()?;
        parser.parse(source, None)
    })
}
