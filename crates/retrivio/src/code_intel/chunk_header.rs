use super::ast_chunker::{ChunkKind, SemanticChunk};

/// Build a contextual header string for a chunk, used to improve embedding quality.
///
/// Example output:
/// ```text
/// // File: src/auth/middleware.rs
/// // Impl: AuthMiddleware
/// // Method: validate_token(&self, token: &str) -> Result<Claims, Error>
/// ```
///
/// This header is prepended to the chunk text ONLY when generating embeddings —
/// it is not stored in the chunk's `text` field.
pub fn build_context_header(chunk: &SemanticChunk, rel_path: &str) -> String {
    let mut parts = Vec::new();
    parts.push(format!("// File: {}", rel_path));

    if !chunk.parent_context.is_empty() {
        parts.push(format!("// Context: {}", chunk.parent_context));
    }

    match chunk.kind {
        ChunkKind::Function => {
            if !chunk.symbol_name.is_empty() {
                parts.push(format!("// Function: {}", chunk.symbol_name));
            }
        }
        ChunkKind::Method => {
            if !chunk.symbol_name.is_empty() {
                parts.push(format!("// Method: {}", chunk.symbol_name));
            }
        }
        ChunkKind::TypeHeader => {
            if !chunk.symbol_name.is_empty() {
                parts.push(format!("// Type: {}", chunk.symbol_name));
            }
        }
        ChunkKind::ImportBlock => {
            parts.push("// Imports".to_string());
        }
        ChunkKind::Preamble => {
            parts.push("// File preamble".to_string());
        }
        ChunkKind::Declaration => {
            if !chunk.symbol_name.is_empty() {
                parts.push(format!("// Declaration: {}", chunk.symbol_name));
            }
        }
        ChunkKind::TextWindow => {}
    }

    parts.join("\n")
}
