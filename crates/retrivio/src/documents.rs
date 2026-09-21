//! Plain-text extraction from document formats the indexer cannot read as text.
//!
//! Supported: `.docx` (Word), `.pptx` (PowerPoint, slides in order with their speaker notes),
//! `.odt`/`.odp` (OpenDocument text and presentation), `.xlsx` (Excel, one line per row per
//! sheet), `.html`/`.htm` (markup converted to text) and `.pdf` (text-based PDFs only, through
//! the `pdf-extract` crate; no OCR). Legacy binary `.doc`/`.ppt`/`.xls` and `.rtf` are not
//! handled.
//!
//! The Office and OpenDocument formats are zip archives of XML. They are read with the `zip`
//! crate (deflate only) and walked with a small, forgiving tag/text scanner in this file; no
//! XML crate is used. Every in-process extractor runs inside `catch_unwind`, so a panic in a
//! parser is reported as an error, never propagated into the indexer.
//!
//! Bounds ([`ExtractLimits`]): an archive is rejected before any entry is decompressed when
//! it lists more than [`MAX_ZIP_ENTRIES`] entries or declares more uncompressed bytes than
//! `max_document_uncompressed_bytes`; an entry over [`MAX_ZIP_ENTRY_BYTES`] (declared or
//! actual) rejects the document rather than being truncated; the output text is capped while
//! it is appended, so a single huge text node cannot exceed the cap. PDF parsing runs in a
//! child process (`retrivio documents extract-pdf`) that the parent kills at
//! `document_extract_timeout_ms` or above [`PDF_CHILD_MEMORY_BYTES`] of resident memory, so
//! a hanging or exploding parser costs one failed document, never the indexer.
//!
//! The output is plain text with paragraph breaks (one paragraph per line, a blank line
//! between slides) so the collector chunks it as text windows like any prose file.

use std::ffi::OsString;
use std::io::{Cursor, Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Text extracted from one document.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtractedDocument {
    /// Plain text; paragraphs separated by `\n`, slides and sheets by a blank line.
    pub text: String,
    /// The document's own title when the format records one (`docProps/core.xml`,
    /// `<title>`); `None` otherwise.
    pub title: Option<String>,
    /// True when the text was cut at the caller's byte limit.
    pub truncated: bool,
}

/// Default for `max_document_uncompressed_bytes`: the declared total uncompressed size an
/// Office or OpenDocument archive may have before it is refused unread.
pub const DEFAULT_MAX_DOCUMENT_UNCOMPRESSED_BYTES: u64 = 200_000_000;
/// Default for `document_extract_timeout_ms`: how long the PDF child process may run.
pub const DEFAULT_DOCUMENT_EXTRACT_TIMEOUT_MS: u64 = 20_000;
/// Most bytes one zip entry may hold, declared or actual; a larger entry rejects the document
/// (never a silent truncation). Real Office parts stay far below it.
pub const MAX_ZIP_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Most entries an archive may list before it is refused unread.
pub const MAX_ZIP_ENTRIES: usize = 20_000;
/// Resident memory the PDF child process may reach before the parent kills it. Applied as
/// `RLIMIT_AS` in the child where the kernel honours that (Linux) and enforced by the parent
/// from the child's resident size everywhere (macOS returns `EINVAL` for `RLIMIT_AS` and
/// `RLIMIT_DATA` at any value; measured on Darwin 25.6).
pub const PDF_CHILD_MEMORY_BYTES: u64 = 1024 * 1024 * 1024;

/// The bounds one extraction honours.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractLimits {
    /// Output bytes; extraction stops when the text reaches this and the text is cut there.
    pub text_bytes: usize,
    /// Declared total uncompressed size an archive may have (`max_document_uncompressed_bytes`).
    pub archive_uncompressed_bytes: u64,
    /// Most bytes one zip entry may hold; over it, the document is rejected.
    pub entry_bytes: u64,
    /// Most entries an archive may list.
    pub archive_entries: usize,
    /// Deadline for the PDF child process (`document_extract_timeout_ms`).
    pub pdf_timeout: Duration,
}

impl Default for ExtractLimits {
    fn default() -> Self {
        ExtractLimits {
            text_bytes: usize::MAX,
            archive_uncompressed_bytes: DEFAULT_MAX_DOCUMENT_UNCOMPRESSED_BYTES,
            entry_bytes: MAX_ZIP_ENTRY_BYTES,
            archive_entries: MAX_ZIP_ENTRIES,
            pdf_timeout: Duration::from_millis(DEFAULT_DOCUMENT_EXTRACT_TIMEOUT_MS),
        }
    }
}

impl ExtractLimits {
    /// The defaults with the output capped at `text_bytes`.
    pub fn with_text_bytes(text_bytes: usize) -> Self {
        ExtractLimits {
            text_bytes,
            ..ExtractLimits::default()
        }
    }
}

/// File suffixes (lower-case, with the dot) this module extracts.
pub const DOCUMENT_SUFFIXES: &[&str] = &[
    ".docx", ".pptx", ".odt", ".odp", ".xlsx", ".pdf", ".html", ".htm",
];

/// Suffixes that are only indexable at all because this module reads them (HTML was always
/// indexed, as raw markup before, as text now).
pub const DOCUMENT_ONLY_SUFFIXES: &[&str] = &[".docx", ".pptx", ".odt", ".odp", ".xlsx", ".pdf"];

/// True for a lower-case suffix with its dot (`.docx`) that this module extracts.
pub fn is_document_suffix(suffix: &str) -> bool {
    DOCUMENT_SUFFIXES.contains(&suffix)
}

/// True for a suffix that is indexable only through this module (not `.html`).
pub fn is_document_only_suffix(suffix: &str) -> bool {
    DOCUMENT_ONLY_SUFFIXES.contains(&suffix)
}

/// Extract the text of `path`, refusing files larger than `max_bytes` (the
/// `max_document_bytes` config key). `Ok(None)` means the suffix is not a document format;
/// `Err` means the file was refused or could not be parsed (size, corrupt archive, missing
/// part, a bound exceeded, PDF without a text layer that the parser rejects, a parser panic,
/// or the PDF child timing out).
///
/// The indexer itself sizes, reads and hashes the file first and calls
/// [`extract_from_bytes`]; this whole-file entry point is the module's standalone API.
#[allow(dead_code)]
pub fn extract_document_text(
    path: &Path,
    max_bytes: u64,
    limits: &ExtractLimits,
) -> Result<Option<ExtractedDocument>, String> {
    let suffix = suffix_of(path);
    if !is_document_suffix(&suffix) {
        return Ok(None);
    }
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat: {}", e))?
        .len();
    if len > max_bytes {
        return Err(format!(
            "{} bytes exceeds max_document_bytes={}",
            len, max_bytes
        ));
    }
    let raw = std::fs::read(path).map_err(|e| format!("cannot read: {}", e))?;
    extract_from_bytes(path, &raw, limits)
}

/// [`extract_document_text`] on bytes already in memory (the collector has read the file for
/// its content hash). `limits.text_bytes` bounds how much text is collected: extraction stops
/// once the output holds that many bytes and the text is cut there, so a caller that will cut
/// the text at N characters anyway passes about `4 * N` and never materialises a whole 25 MB
/// document. PDFs are parsed by a child process reading `path` (see [`run_pdf_child`]); the
/// other formats are parsed in this process inside `catch_unwind`.
pub fn extract_from_bytes(
    path: &Path,
    raw: &[u8],
    limits: &ExtractLimits,
) -> Result<Option<ExtractedDocument>, String> {
    let suffix = suffix_of(path);
    if !is_document_suffix(&suffix) {
        return Ok(None);
    }
    let limits = ExtractLimits {
        text_bytes: limits.text_bytes.max(1),
        ..limits.clone()
    };
    if suffix == ".pdf" {
        return run_pdf_child(path, &limits).map(Some);
    }
    let result = catch_unwind(AssertUnwindSafe(|| match suffix.as_str() {
        ".docx" => extract_docx(raw, &limits),
        ".pptx" => extract_pptx(raw, &limits),
        ".odt" | ".odp" => extract_odf(raw, &limits),
        ".xlsx" => extract_xlsx(raw, &limits),
        ".html" | ".htm" => Ok(extract_html(raw, limits.text_bytes)),
        _ => unreachable!("suffix checked above"),
    }));
    match result {
        Ok(Ok(doc)) => Ok(Some(doc)),
        Ok(Err(e)) => Err(e),
        Err(panic) => Err(format!(
            "parser panicked: {}",
            panic_message(panic.as_ref())
        )),
    }
}

/// Convert HTML markup to plain text: `script`, `style`, `head`, `nav`, `noscript`,
/// `template` and `svg` subtrees and comments are dropped; block elements and headings become
/// line breaks, table cells tabs; character and numeric entity references are decoded;
/// whitespace is collapsed within a line and runs of blank lines to one.
#[cfg(test)]
pub fn html_to_text(html: &str) -> String {
    html_to_text_limited(html, usize::MAX).text
}

fn suffix_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

// ── zip access ────────────────────────────────────────────────────────────────

/// The archive reads the caller's bytes in place (no copy of the file).
type Archive<'a> = zip::ZipArchive<Cursor<&'a [u8]>>;

/// Open an archive and check its declared shape before anything is decompressed: the entry
/// count against `limits.archive_entries` and the sum of the entries' declared uncompressed
/// sizes against `limits.archive_uncompressed_bytes`. The central directory's sizes are what
/// is summed (a lying entry is caught by [`read_entry`] when it is actually read).
fn open_zip<'a>(raw: &'a [u8], limits: &ExtractLimits) -> Result<Archive<'a>, String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(raw)).map_err(|e| format!("not a zip archive: {}", e))?;
    if archive.len() > limits.archive_entries {
        return Err(format!(
            "archive lists {} entries; at most {} are accepted",
            archive.len(),
            limits.archive_entries
        ));
    }
    let mut declared: u128 = 0;
    for i in 0..archive.len() {
        let entry = archive
            .by_index_raw(i)
            .map_err(|e| format!("cannot read zip directory entry {}: {}", i, e))?;
        declared = declared.saturating_add(u128::from(entry.size()));
    }
    if declared > u128::from(limits.archive_uncompressed_bytes) {
        return Err(format!(
            "archive declares {} uncompressed bytes; max_document_uncompressed_bytes={}",
            declared, limits.archive_uncompressed_bytes
        ));
    }
    Ok(archive)
}

/// Read one entry as (lossy) UTF-8, `None` when the entry does not exist. An entry over
/// `limits.entry_bytes`, by its declared size or by what it actually decompresses to, is an
/// error: the document is rejected, never silently cut.
fn read_entry(
    archive: &mut Archive<'_>,
    name: &str,
    limits: &ExtractLimits,
) -> Result<Option<String>, String> {
    if archive.index_for_name(name).is_none() {
        return Ok(None);
    }
    let mut file = archive
        .by_name(name)
        .map_err(|e| format!("cannot open zip entry {}: {}", name, e))?;
    if file.size() > limits.entry_bytes {
        return Err(format!(
            "zip entry {} declares {} bytes; at most {} are accepted",
            name,
            file.size(),
            limits.entry_bytes
        ));
    }
    // Capacity from the declared size, but never more than 1 MiB up front: the declaration
    // may lie, and the read below is what bounds the real size.
    let mut buf = Vec::with_capacity(file.size().min(1024 * 1024) as usize);
    file.by_ref()
        .take(limits.entry_bytes.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("cannot read zip entry {}: {}", name, e))?;
    if buf.len() as u64 > limits.entry_bytes {
        return Err(format!(
            "zip entry {} decompresses to more than {} bytes (declared {})",
            name,
            limits.entry_bytes,
            file.size()
        ));
    }
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

fn require_entry(
    archive: &mut Archive<'_>,
    name: &str,
    limits: &ExtractLimits,
) -> Result<String, String> {
    read_entry(archive, name, limits)?.ok_or_else(|| format!("missing zip entry {}", name))
}

/// Entries matching `<dir><stem><N>.xml`, sorted by N. Slide and sheet numbering.
fn numbered_entries(archive: &Archive<'_>, dir: &str, stem: &str) -> Vec<(u32, String)> {
    let mut out: Vec<(u32, String)> = archive
        .file_names()
        .filter_map(|name| {
            let rest = name.strip_prefix(dir)?.strip_prefix(stem)?;
            let digits = rest.strip_suffix(".xml")?;
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            Some((digits.parse::<u32>().ok()?, name.to_string()))
        })
        .collect();
    out.sort();
    out
}

/// `dc:title` from `docProps/core.xml` when present and non-empty.
fn office_title(archive: &mut Archive<'_>, limits: &ExtractLimits) -> Option<String> {
    let core = read_entry(archive, "docProps/core.xml", limits)
        .ok()
        .flatten()?;
    let mut in_title = false;
    let mut title = String::new();
    for ev in XmlScanner::new(&core) {
        match ev {
            XmlEvent::Start { name, .. } if local_name(&name) == "title" => in_title = true,
            XmlEvent::End(name) if local_name(&name) == "title" => break,
            XmlEvent::Text(t) if in_title => title.push_str(&t),
            _ => {}
        }
    }
    let title = collapse_inline_whitespace(&title);
    (!title.is_empty()).then_some(title)
}

// ── a forgiving XML/HTML scanner ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum XmlEvent {
    Start {
        name: String,
        /// Raw attribute text between the name and `>` (or `/>`), unparsed.
        attrs: String,
        self_closing: bool,
    },
    End(String),
    /// Decoded text between tags (entities resolved; CDATA verbatim).
    Text(String),
}

/// Walks tags and text without building a tree. Unknown constructs are skipped, unbalanced
/// tags are reported as they come, and malformed input never panics: at worst the rest of
/// the document is one text run.
struct XmlScanner<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> XmlScanner<'a> {
    fn new(src: &'a str) -> Self {
        XmlScanner { src, pos: 0 }
    }

    /// Skip everything up to and including `end`, or to the end of input.
    fn skip_past(&mut self, end: &str) {
        match self.src[self.pos..].find(end) {
            Some(i) => self.pos += i + end.len(),
            None => self.pos = self.src.len(),
        }
    }

    /// Everything up to (not including) `end`, advancing past it.
    fn take_until(&mut self, end: &str) -> &'a str {
        let rest = &self.src[self.pos..];
        match rest.find(end) {
            Some(i) => {
                self.pos += i + end.len();
                &rest[..i]
            }
            None => {
                self.pos = self.src.len();
                rest
            }
        }
    }

    /// The end of a tag opened at `self.pos` (which is past `<`): the first `>` outside
    /// quotes. Returns the index relative to `self.pos` and whether it was found.
    fn tag_end(&self) -> Option<usize> {
        let bytes = self.src[self.pos..].as_bytes();
        let mut quote: Option<u8> = None;
        for (i, &b) in bytes.iter().enumerate() {
            match (quote, b) {
                (Some(q), _) if b == q => quote = None,
                (Some(_), _) => {}
                (None, b'"') | (None, b'\'') => quote = Some(b),
                (None, b'>') => return Some(i),
                _ => {}
            }
        }
        None
    }
}

impl Iterator for XmlScanner<'_> {
    type Item = XmlEvent;

    fn next(&mut self) -> Option<XmlEvent> {
        loop {
            if self.pos >= self.src.len() {
                return None;
            }
            let rest = &self.src[self.pos..];
            if let Some(stripped) = rest.strip_prefix('<') {
                if stripped.starts_with("!--") {
                    self.pos += 1;
                    self.skip_past("-->");
                    continue;
                }
                if stripped.starts_with("![CDATA[") {
                    self.pos += 1 + "![CDATA[".len();
                    let text = self.take_until("]]>");
                    if text.is_empty() {
                        continue;
                    }
                    return Some(XmlEvent::Text(text.to_string()));
                }
                if stripped.starts_with('?') || stripped.starts_with('!') {
                    self.pos += 1;
                    self.skip_past(">");
                    continue;
                }
                // A tag. Without a closing `>` the rest of the input is taken as the tag.
                self.pos += 1;
                let end = self.tag_end().unwrap_or(self.src.len() - self.pos);
                let body = &self.src[self.pos..self.pos + end];
                self.pos = (self.pos + end + 1).min(self.src.len());
                let body = body.trim();
                if body.is_empty() {
                    continue;
                }
                if let Some(name) = body.strip_prefix('/') {
                    return Some(XmlEvent::End(name.trim().to_ascii_lowercase()));
                }
                let (body, self_closing) = match body.strip_suffix('/') {
                    Some(b) => (b.trim_end(), true),
                    None => (body, false),
                };
                let name_end = body.find(|c: char| c.is_whitespace()).unwrap_or(body.len());
                let name = body[..name_end].to_ascii_lowercase();
                let attrs = body[name_end..].trim().to_string();
                if name.is_empty() {
                    continue;
                }
                return Some(XmlEvent::Start {
                    name,
                    attrs,
                    self_closing,
                });
            }
            // Text up to the next tag (the `<` itself is scanned on the next call).
            let advance = rest.find('<').unwrap_or(rest.len());
            let text = &rest[..advance];
            self.pos += advance;
            if text.is_empty() {
                continue;
            }
            return Some(XmlEvent::Text(decode_entities(text)));
        }
    }
}

/// `w:t` -> `t`; the part after the last namespace prefix.
fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// The value of attribute `key` (case-insensitive name, exact match) in a raw attribute
/// string: `name="value"`, `name='value'` or a bare `name=value`. `None` when absent.
fn attr_value(attrs: &str, key: &str) -> Option<String> {
    let mut rest = attrs.trim_start();
    while !rest.is_empty() {
        let name_end = rest
            .find(|c: char| c.is_whitespace() || c == '=')
            .unwrap_or(rest.len());
        let name = &rest[..name_end];
        rest = rest[name_end..].trim_start();
        let Some(after_eq) = rest.strip_prefix('=') else {
            // A valueless attribute; move on.
            if name.is_empty() {
                rest = &rest[rest.chars().next().map_or(0, char::len_utf8)..];
            }
            continue;
        };
        let after_eq = after_eq.trim_start();
        let (value, tail) = match after_eq.chars().next() {
            Some(q @ ('"' | '\'')) => {
                let inner = &after_eq[1..];
                let close = inner.find(q).unwrap_or(inner.len());
                (&inner[..close], &inner[(close + 1).min(inner.len())..])
            }
            _ => {
                let close = after_eq
                    .find(|c: char| c.is_whitespace())
                    .unwrap_or(after_eq.len());
                (&after_eq[..close], &after_eq[close..])
            }
        };
        if name.eq_ignore_ascii_case(key) {
            return Some(decode_entities(value));
        }
        rest = tail.trim_start();
    }
    None
}

/// Decode `&amp;`-style references: the XML five, `&nbsp;`, the common HTML names, and
/// decimal or hexadecimal numeric references. Unknown names are left as written.
fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        // An entity name is short: look for the `;` within the next 12 bytes, on a char
        // boundary (the text after `&` may hold multi-byte characters).
        let mut window = rest.len().min(12);
        while !rest.is_char_boundary(window) {
            window -= 1;
        }
        let Some(semi) = rest[..window].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let name = &rest[1..semi];
        let decoded: Option<String> = if let Some(num) = name.strip_prefix('#') {
            let code = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
                u32::from_str_radix(hex, 16).ok()
            } else {
                num.parse::<u32>().ok()
            };
            code.and_then(char::from_u32).map(|c| c.to_string())
        } else {
            named_entity(name).map(|c| c.to_string())
        };
        match decoded {
            Some(s) => {
                out.push_str(&s);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn named_entity(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        // A non-breaking space is a space for indexing purposes.
        "nbsp" => ' ',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "hellip" => '\u{2026}',
        "copy" => '\u{a9}',
        "reg" => '\u{ae}',
        "trade" => '\u{2122}',
        "laquo" => '\u{ab}',
        "raquo" => '\u{bb}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "bull" => '\u{2022}',
        "middot" => '\u{b7}',
        "deg" => '\u{b0}',
        "euro" => '\u{20ac}',
        "pound" => '\u{a3}',
        "yen" => '\u{a5}',
        "cent" => '\u{a2}',
        "times" => '\u{d7}',
        "divide" => '\u{f7}',
        "shy" => '\u{ad}',
        "iexcl" => '\u{a1}',
        "iquest" => '\u{bf}',
        "sect" => '\u{a7}',
        "para" => '\u{b6}',
        "plusmn" => '\u{b1}',
        "frac12" => '\u{bd}',
        "frac14" => '\u{bc}',
        "frac34" => '\u{be}',
        "larr" => '\u{2190}',
        "rarr" => '\u{2192}',
        "uarr" => '\u{2191}',
        "darr" => '\u{2193}',
        "hearts" => '\u{2665}',
        "check" => '\u{2713}',
        _ => return None,
    })
}

// ── output assembly ──────────────────────────────────────────────────────────

/// Collects paragraphs into the output text, never holding more than `limit` bytes: text is
/// cut while it is appended (a single huge node cannot push the buffer past the cap), and
/// once the cap is reached the sink is `full` and the scanners stop.
struct TextSink {
    out: String,
    limit: usize,
    /// The paragraph being assembled.
    current: String,
    full: bool,
}

impl TextSink {
    fn new(limit: usize) -> Self {
        TextSink {
            out: String::new(),
            limit,
            current: String::new(),
            full: false,
        }
    }

    fn is_full(&self) -> bool {
        self.full
    }

    /// Bytes held so far (flushed paragraphs plus the one being assembled).
    fn len(&self) -> usize {
        self.out.len() + self.current.len()
    }

    fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.len())
    }

    /// Append text, cut at the remaining budget (on a char boundary); reaching the budget
    /// marks the sink full.
    fn push_text(&mut self, text: &str) {
        if self.full {
            return;
        }
        let room = self.remaining();
        if text.len() <= room {
            self.current.push_str(text);
            return;
        }
        let mut cut = room;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        self.current.push_str(&text[..cut]);
        self.full = true;
    }

    fn push_char(&mut self, ch: char) {
        if self.full {
            return;
        }
        if ch.len_utf8() > self.remaining() {
            self.full = true;
            return;
        }
        self.current.push(ch);
    }

    /// Close the current paragraph: written as one line (inline whitespace collapsed, the
    /// `\n` of explicit breaks kept), empty paragraphs dropped. A paragraph cut by the cap is
    /// still written (up to the cap); nothing after it is.
    fn end_paragraph(&mut self) {
        let line = normalize_paragraph(&self.current);
        self.current.clear();
        if line.is_empty() {
            return;
        }
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
        self.out.push_str(&line);
        self.out.push('\n');
        if self.out.len() >= self.limit {
            self.full = true;
        }
    }

    /// A blank line (section, slide or sheet boundary).
    fn blank_line(&mut self) {
        self.end_paragraph();
        if !self.full && !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }

    /// A heading line such as `Slide 3` or `Notes:`.
    fn heading(&mut self, text: &str) {
        self.end_paragraph();
        self.push_text(text);
        self.end_paragraph();
    }

    /// The text (trailing newlines removed, never longer than the limit) and whether the cap
    /// cut it.
    fn finish(mut self) -> (String, bool) {
        self.end_paragraph();
        let mut out = std::mem::take(&mut self.out);
        let trimmed = out.trim_end_matches('\n').len();
        out.truncate(trimmed);
        if out.len() > self.limit {
            let mut cut = self.limit;
            while cut > 0 && !out.is_char_boundary(cut) {
                cut -= 1;
            }
            out.truncate(cut);
            self.full = true;
        }
        (out, self.full)
    }
}

/// Collapse spaces and tabs inside a paragraph; keep explicit line breaks and tabs that
/// separate cells (a tab surrounded by text stays a tab).
fn normalize_paragraph(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for (i, line) in s.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let mut pending_space = false;
        let mut pending_tab = false;
        let mut started = false;
        for ch in line.chars() {
            match ch {
                '\t' => pending_tab = true,
                c if c.is_whitespace() => pending_space = true,
                c => {
                    if started {
                        if pending_tab {
                            out.push('\t');
                        } else if pending_space {
                            out.push(' ');
                        }
                    }
                    pending_space = false;
                    pending_tab = false;
                    started = true;
                    out.push(c);
                }
            }
        }
    }
    let lines: Vec<&str> = out
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    lines.join("\n")
}

fn collapse_inline_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── DOCX ─────────────────────────────────────────────────────────────────────

fn extract_docx(raw: &[u8], limits: &ExtractLimits) -> Result<ExtractedDocument, String> {
    let mut archive = open_zip(raw, limits)?;
    let document = require_entry(&mut archive, "word/document.xml", limits)?;
    let mut sink = TextSink::new(limits.text_bytes);
    wordprocessing_text(&document, &mut sink);
    let title = office_title(&mut archive, limits);
    let (text, truncated) = sink.finish();
    Ok(ExtractedDocument {
        text,
        title,
        truncated,
    })
}

/// WordprocessingML: `w:p` paragraphs of `w:t` runs; `w:tab` and `w:br`; field
/// instructions (`w:instrText`) and tracked deletions (`w:delText`) are skipped.
fn wordprocessing_text(xml: &str, sink: &mut TextSink) {
    let mut skip_depth = 0usize;
    let mut in_text = false;
    for ev in XmlScanner::new(xml) {
        if sink.is_full() {
            break;
        }
        match ev {
            XmlEvent::Start {
                name, self_closing, ..
            } => match local_name(&name) {
                "instrtext" | "deltext" | "fldsimple" if !self_closing => skip_depth += 1,
                "p" if !self_closing => sink.end_paragraph(),
                "tab" => sink.push_char('\t'),
                "br" | "cr" => sink.push_char('\n'),
                "t" if !self_closing && skip_depth == 0 => in_text = true,
                _ => {}
            },
            XmlEvent::End(name) => match local_name(&name) {
                "instrtext" | "deltext" | "fldsimple" => skip_depth = skip_depth.saturating_sub(1),
                "p" => sink.end_paragraph(),
                "t" => in_text = false,
                // A table row ends a line even when its cells held no paragraph end.
                "tr" => sink.end_paragraph(),
                _ => {}
            },
            XmlEvent::Text(text) => {
                if in_text && skip_depth == 0 {
                    sink.push_text(&text);
                }
            }
        }
    }
    sink.end_paragraph();
}

// ── PPTX ─────────────────────────────────────────────────────────────────────

fn extract_pptx(raw: &[u8], limits: &ExtractLimits) -> Result<ExtractedDocument, String> {
    let mut archive = open_zip(raw, limits)?;
    let slides = numbered_entries(&archive, "ppt/slides/", "slide");
    if slides.is_empty() {
        return Err("no slides (ppt/slides/slideN.xml) in archive".to_string());
    }
    let mut sink = TextSink::new(limits.text_bytes);
    for (n, entry) in &slides {
        if sink.is_full() {
            break;
        }
        let xml = require_entry(&mut archive, entry, limits)?;
        if *n > 1 || !sink.out.is_empty() {
            sink.blank_line();
        }
        sink.heading(&format!("Slide {}", n));
        drawingml_text(&xml, &mut sink);
        if sink.is_full() {
            break;
        }
        if let Some(notes_xml) = notes_for_slide(&mut archive, *n, limits)? {
            // The notes get whatever budget the slides have left.
            let mut notes = TextSink::new(sink.remaining().max(1));
            drawingml_text(&notes_xml, &mut notes);
            let (notes_text, _) = notes.finish();
            if !notes_text.is_empty() {
                sink.heading("Notes:");
                for line in notes_text.lines() {
                    sink.push_text(line);
                    sink.end_paragraph();
                }
            }
        }
    }
    let title = office_title(&mut archive, limits);
    let (text, truncated) = sink.finish();
    Ok(ExtractedDocument {
        text,
        title,
        truncated,
    })
}

/// The notes slide of slide `n`: through the slide's relationships when present, else the
/// conventional `ppt/notesSlides/notesSlideN.xml`.
fn notes_for_slide(
    archive: &mut Archive<'_>,
    n: u32,
    limits: &ExtractLimits,
) -> Result<Option<String>, String> {
    let rels_name = format!("ppt/slides/_rels/slide{}.xml.rels", n);
    if let Some(rels) = read_entry(archive, &rels_name, limits)? {
        for ev in XmlScanner::new(&rels) {
            if let XmlEvent::Start { name, attrs, .. } = ev {
                if local_name(&name) != "relationship" {
                    continue;
                }
                let kind = attr_value(&attrs, "Type").unwrap_or_default();
                if !kind.ends_with("/notesSlide") {
                    continue;
                }
                if let Some(target) = attr_value(&attrs, "Target") {
                    // Targets are relative to ppt/slides/ (`../notesSlides/notesSlide1.xml`).
                    let resolved = if let Some(rest) = target.strip_prefix("../") {
                        format!("ppt/{}", rest)
                    } else if let Some(rest) = target.strip_prefix('/') {
                        rest.to_string()
                    } else {
                        format!("ppt/slides/{}", target)
                    };
                    return read_entry(archive, &resolved, limits);
                }
            }
        }
    }
    read_entry(
        archive,
        &format!("ppt/notesSlides/notesSlide{}.xml", n),
        limits,
    )
}

/// DrawingML: `a:p` paragraphs of `a:t` runs, `a:br` line breaks, `a:tab`. Slide numbers in
/// notes placeholders (`a:fld type="slidenum"`) are skipped.
fn drawingml_text(xml: &str, sink: &mut TextSink) {
    let mut in_text = false;
    let mut skip_depth = 0usize;
    for ev in XmlScanner::new(xml) {
        if sink.is_full() {
            break;
        }
        match ev {
            XmlEvent::Start {
                name,
                attrs,
                self_closing,
            } => match local_name(&name) {
                "fld"
                    if !self_closing
                        && attr_value(&attrs, "type")
                            .map(|t| t.eq_ignore_ascii_case("slidenum"))
                            .unwrap_or(false) =>
                {
                    skip_depth += 1
                }
                "p" if !self_closing => sink.end_paragraph(),
                "br" => sink.push_char('\n'),
                "tab" => sink.push_char('\t'),
                "t" if !self_closing && skip_depth == 0 => in_text = true,
                _ => {}
            },
            XmlEvent::End(name) => match local_name(&name) {
                "fld" => skip_depth = skip_depth.saturating_sub(1),
                "p" => sink.end_paragraph(),
                "t" => in_text = false,
                "tr" => sink.end_paragraph(),
                _ => {}
            },
            XmlEvent::Text(text) => {
                if in_text && skip_depth == 0 {
                    sink.push_text(&text);
                }
            }
        }
    }
    sink.end_paragraph();
}

// ── ODT / ODP ────────────────────────────────────────────────────────────────

fn extract_odf(raw: &[u8], limits: &ExtractLimits) -> Result<ExtractedDocument, String> {
    let mut archive = open_zip(raw, limits)?;
    let content = require_entry(&mut archive, "content.xml", limits)?;
    let mut sink = TextSink::new(limits.text_bytes);
    let mut page_no = 0u32;
    let mut text_depth = 0usize;
    for ev in XmlScanner::new(&content) {
        if sink.is_full() {
            break;
        }
        match ev {
            XmlEvent::Start {
                name,
                attrs,
                self_closing,
            } => match (name.as_str(), local_name(&name)) {
                ("draw:page", _) if !self_closing => {
                    page_no += 1;
                    if page_no > 1 {
                        sink.blank_line();
                    }
                    sink.heading(&format!("Slide {}", page_no));
                }
                ("presentation:notes", _) if !self_closing => sink.heading("Notes:"),
                (_, "p") | (_, "h") if !self_closing => {
                    sink.end_paragraph();
                    text_depth += 1;
                }
                (_, "tab") => sink.push_char('\t'),
                (_, "line-break") => sink.push_char('\n'),
                (_, "s") => {
                    let count = attr_value(&attrs, "text:c")
                        .and_then(|c| c.parse::<usize>().ok())
                        .unwrap_or(1)
                        .min(64);
                    for _ in 0..count {
                        sink.push_char(' ');
                    }
                }
                _ => {}
            },
            XmlEvent::End(name) => match local_name(&name) {
                "p" | "h" => {
                    sink.end_paragraph();
                    text_depth = text_depth.saturating_sub(1);
                }
                "table-row" => sink.end_paragraph(),
                _ => {}
            },
            XmlEvent::Text(text) => {
                if text_depth > 0 {
                    sink.push_text(&text);
                }
            }
        }
    }
    let title = odf_title(&mut archive, limits);
    let (text, truncated) = sink.finish();
    Ok(ExtractedDocument {
        text,
        title,
        truncated,
    })
}

fn odf_title(archive: &mut Archive<'_>, limits: &ExtractLimits) -> Option<String> {
    let meta = read_entry(archive, "meta.xml", limits).ok().flatten()?;
    let mut in_title = false;
    let mut title = String::new();
    for ev in XmlScanner::new(&meta) {
        match ev {
            XmlEvent::Start { name, .. } if name == "dc:title" => in_title = true,
            XmlEvent::End(name) if name == "dc:title" => break,
            XmlEvent::Text(t) if in_title => title.push_str(&t),
            _ => {}
        }
    }
    let title = collapse_inline_whitespace(&title);
    (!title.is_empty()).then_some(title)
}

// ── XLSX ─────────────────────────────────────────────────────────────────────

fn extract_xlsx(raw: &[u8], limits: &ExtractLimits) -> Result<ExtractedDocument, String> {
    let mut archive = open_zip(raw, limits)?;
    let shared = match read_entry(&mut archive, "xl/sharedStrings.xml", limits)? {
        Some(xml) => shared_strings(&xml, limits.entry_bytes as usize),
        None => Vec::new(),
    };
    let sheets = workbook_sheets(&mut archive, limits)?;
    if sheets.is_empty() {
        return Err("no worksheets in archive".to_string());
    }
    let mut sink = TextSink::new(limits.text_bytes);
    for (i, (name, entry)) in sheets.iter().enumerate() {
        if sink.is_full() {
            break;
        }
        let Some(xml) = read_entry(&mut archive, entry, limits)? else {
            continue;
        };
        if i > 0 {
            sink.blank_line();
        }
        sink.heading(&format!("Sheet: {}", name));
        worksheet_rows(&xml, &shared, &mut sink);
    }
    let title = office_title(&mut archive, limits);
    let (text, truncated) = sink.finish();
    Ok(ExtractedDocument {
        text,
        title,
        truncated,
    })
}

/// `si` items of the shared string table, rich-text runs concatenated. The table's total
/// text is bounded by `max_total_bytes` (the per-entry bound: the strings come out of one
/// entry, so they can never exceed it; the bound is explicit so a change to `read_entry`
/// cannot silently lift it). Strings past the bound are absent, so the cells that reference
/// them come out empty.
fn shared_strings(xml: &str, max_total_bytes: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut current: Option<String> = None;
    let mut in_t = false;
    let mut skip_depth = 0usize;
    for ev in XmlScanner::new(xml) {
        if total >= max_total_bytes {
            break;
        }
        match ev {
            XmlEvent::Start {
                name, self_closing, ..
            } => match local_name(&name) {
                "si" if !self_closing => current = Some(String::new()),
                // Phonetic runs are annotations, not cell text.
                "rph" if !self_closing => skip_depth += 1,
                "t" if !self_closing && skip_depth == 0 => in_t = true,
                _ => {}
            },
            XmlEvent::End(name) => match local_name(&name) {
                "si" => {
                    if let Some(s) = current.take() {
                        total += s.len();
                        out.push(s);
                    }
                }
                "rph" => skip_depth = skip_depth.saturating_sub(1),
                "t" => in_t = false,
                _ => {}
            },
            XmlEvent::Text(text) => {
                if in_t {
                    if let Some(s) = current.as_mut() {
                        s.push_str(&text);
                    }
                }
            }
        }
    }
    out
}

/// (sheet name, zip entry) in workbook order, from `xl/workbook.xml` and its relationships;
/// falls back to `xl/worksheets/sheetN.xml` in numeric order.
fn workbook_sheets(
    archive: &mut Archive<'_>,
    limits: &ExtractLimits,
) -> Result<Vec<(String, String)>, String> {
    let mut rels: Vec<(String, String)> = Vec::new(); // rId -> target
    if let Some(xml) = read_entry(archive, "xl/_rels/workbook.xml.rels", limits)? {
        for ev in XmlScanner::new(&xml) {
            if let XmlEvent::Start { name, attrs, .. } = ev {
                if local_name(&name) == "relationship" {
                    if let (Some(id), Some(target)) =
                        (attr_value(&attrs, "Id"), attr_value(&attrs, "Target"))
                    {
                        rels.push((id, target));
                    }
                }
            }
        }
    }
    let mut sheets: Vec<(String, String)> = Vec::new();
    if let Some(xml) = read_entry(archive, "xl/workbook.xml", limits)? {
        for ev in XmlScanner::new(&xml) {
            if let XmlEvent::Start { name, attrs, .. } = ev {
                if local_name(&name) != "sheet" {
                    continue;
                }
                let sheet_name = attr_value(&attrs, "name").unwrap_or_default();
                let rid = attr_value(&attrs, "r:id")
                    .or_else(|| attr_value(&attrs, "id"))
                    .unwrap_or_default();
                let Some((_, target)) = rels.iter().find(|(id, _)| *id == rid) else {
                    continue;
                };
                let entry = if let Some(rest) = target.strip_prefix('/') {
                    rest.to_string()
                } else {
                    format!("xl/{}", target)
                };
                sheets.push((sheet_name, entry));
            }
        }
    }
    if sheets.is_empty() {
        for (n, entry) in numbered_entries(archive, "xl/worksheets/", "sheet") {
            sheets.push((format!("Sheet {}", n), entry));
        }
    }
    Ok(sheets)
}

/// One line per `row`, cells tab-separated: shared strings (`t="s"`), inline strings
/// (`t="inlineStr"`), formula string results (`t="str"`), booleans and raw numbers.
fn worksheet_rows(xml: &str, shared: &[String], sink: &mut TextSink) {
    let mut in_row = false;
    let mut cell_type = String::new();
    let mut in_v = false;
    let mut in_is_t = false;
    let mut in_is = false;
    let mut cell_text = String::new();
    let mut cells_in_row = 0usize;
    let mut in_cell = false;
    for ev in XmlScanner::new(xml) {
        if sink.is_full() {
            break;
        }
        match ev {
            XmlEvent::Start {
                name,
                attrs,
                self_closing,
            } => match local_name(&name) {
                "row" if !self_closing => {
                    in_row = true;
                    cells_in_row = 0;
                }
                "c" if in_row => {
                    cell_type = attr_value(&attrs, "t").unwrap_or_default();
                    cell_text.clear();
                    in_cell = !self_closing;
                }
                "v" if in_cell && !self_closing => in_v = true,
                "is" if in_cell && !self_closing => in_is = true,
                "t" if in_is && !self_closing => in_is_t = true,
                _ => {}
            },
            XmlEvent::End(name) => match local_name(&name) {
                "row" => {
                    in_row = false;
                    sink.end_paragraph();
                }
                "c" if in_cell => {
                    in_cell = false;
                    let value = match cell_type.as_str() {
                        "s" => cell_text
                            .trim()
                            .parse::<usize>()
                            .ok()
                            .and_then(|i| shared.get(i).cloned())
                            .unwrap_or_default(),
                        "b" => {
                            if cell_text.trim() == "1" {
                                "TRUE".to_string()
                            } else {
                                "FALSE".to_string()
                            }
                        }
                        _ => cell_text.trim().to_string(),
                    };
                    if cells_in_row > 0 {
                        sink.push_char('\t');
                    }
                    sink.push_text(&value.replace(['\t', '\n'], " "));
                    cells_in_row += 1;
                }
                "v" => in_v = false,
                "is" => in_is = false,
                "t" => in_is_t = false,
                _ => {}
            },
            XmlEvent::Text(text) => {
                if in_v || in_is_t {
                    cell_text.push_str(&text);
                }
            }
        }
    }
    sink.end_paragraph();
}

// ── HTML ─────────────────────────────────────────────────────────────────────

fn extract_html(raw: &[u8], limit: usize) -> ExtractedDocument {
    let html = String::from_utf8_lossy(raw);
    html_to_text_limited(&html, limit)
}

fn is_block_element(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "li"
            | "ul"
            | "ol"
            | "dl"
            | "dt"
            | "dd"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "tr"
            | "table"
            | "thead"
            | "tbody"
            | "tfoot"
            | "caption"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "aside"
            | "main"
            | "blockquote"
            | "pre"
            | "hr"
            | "figure"
            | "figcaption"
            | "address"
            | "form"
            | "fieldset"
            | "legend"
            | "option"
            | "details"
            | "summary"
            | "body"
            | "html"
    )
}

fn is_dropped_element(name: &str) -> bool {
    matches!(
        name,
        "script" | "style" | "head" | "nav" | "noscript" | "template" | "svg" | "iframe" | "object"
    )
}

/// Elements HTML lets you leave unclosed; a matching end tag is never required.
fn is_void_element(name: &str) -> bool {
    matches!(
        name,
        "br" | "hr"
            | "img"
            | "input"
            | "meta"
            | "link"
            | "area"
            | "base"
            | "col"
            | "embed"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

fn html_to_text_limited(html: &str, limit: usize) -> ExtractedDocument {
    let mut sink = TextSink::new(limit);
    let mut drop_stack: Vec<String> = Vec::new();
    let mut title = String::new();
    let mut in_title = false;
    let mut in_pre = 0usize;
    for ev in XmlScanner::new(html) {
        if sink.is_full() {
            break;
        }
        match ev {
            XmlEvent::Start {
                name, self_closing, ..
            } => {
                if name == "title" && !self_closing {
                    in_title = true;
                    continue;
                }
                if !drop_stack.is_empty() {
                    if is_dropped_element(&name) && !self_closing && !is_void_element(&name) {
                        drop_stack.push(name);
                    }
                    continue;
                }
                if is_dropped_element(&name) {
                    if !self_closing && !is_void_element(&name) {
                        drop_stack.push(name);
                    }
                    continue;
                }
                match name.as_str() {
                    "br" => sink.push_char('\n'),
                    "td" | "th" => sink.push_char('\t'),
                    "pre" => {
                        in_pre += 1;
                        sink.end_paragraph();
                    }
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => sink.blank_line(),
                    n if is_block_element(n) => sink.end_paragraph(),
                    _ => {}
                }
            }
            XmlEvent::End(name) => {
                if name == "title" {
                    in_title = false;
                    continue;
                }
                if let Some(pos) = drop_stack.iter().rposition(|n| *n == name) {
                    drop_stack.truncate(pos);
                    continue;
                }
                if !drop_stack.is_empty() {
                    continue;
                }
                match name.as_str() {
                    "pre" => {
                        in_pre = in_pre.saturating_sub(1);
                        sink.end_paragraph();
                    }
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => sink.blank_line(),
                    n if is_block_element(n) => sink.end_paragraph(),
                    _ => {}
                }
            }
            XmlEvent::Text(text) => {
                if in_title {
                    title.push_str(&text);
                    continue;
                }
                if !drop_stack.is_empty() {
                    continue;
                }
                if in_pre > 0 {
                    // Preformatted text keeps its line structure.
                    for (i, line) in text.split('\n').enumerate() {
                        if i > 0 {
                            sink.push_char('\n');
                        }
                        sink.push_text(line);
                    }
                } else {
                    sink.push_text(&text);
                }
            }
        }
    }
    let title = collapse_inline_whitespace(&title);
    let (text, truncated) = sink.finish();
    ExtractedDocument {
        text,
        title: (!title.is_empty()).then_some(title),
        truncated,
    }
}

// ── PDF ──────────────────────────────────────────────────────────────────────

/// Text-based PDFs through `pdf-extract`, in this process. Pages are separated by a blank
/// line; a document without a text layer comes back as empty text (the caller records zero
/// chunks). The indexer never calls this directly: it runs in the child process started by
/// [`run_pdf_child`], because `pdf-extract` can loop or allocate without bound on a hostile
/// file and `catch_unwind` stops neither.
fn extract_pdf_bytes(raw: &[u8], text_bytes: usize) -> Result<ExtractedDocument, String> {
    let pages =
        pdf_extract::extract_text_from_mem_by_pages(raw).map_err(|e| format!("pdf: {}", e))?;
    let mut sink = TextSink::new(text_bytes.max(1));
    for (i, page) in pages.iter().enumerate() {
        if sink.is_full() {
            break;
        }
        if i > 0 {
            sink.blank_line();
        }
        for line in page.lines() {
            sink.push_text(line);
            sink.end_paragraph();
        }
    }
    let (text, truncated) = sink.finish();
    Ok(ExtractedDocument {
        text,
        title: None,
        truncated,
    })
}

/// First line of the child's stdout: `retrivio-pdf-text 1 <byte length>`, followed by exactly
/// that many bytes of text. The parent looks for the header line anywhere in the output, so
/// whatever the child's runtime prints before it (the test harness does) is ignored.
const PDF_CHILD_HEADER: &str = "retrivio-pdf-text 1 ";

/// Parse the PDF child's stdout (see [`PDF_CHILD_HEADER`]).
fn parse_pdf_child_output(out: &[u8]) -> Result<ExtractedDocument, String> {
    let mut start = 0usize;
    loop {
        let rest = &out[start..];
        let line_end = rest.iter().position(|b| *b == b'\n').unwrap_or(rest.len());
        let line = String::from_utf8_lossy(&rest[..line_end]);
        if let Some(fields) = line.strip_prefix(PDF_CHILD_HEADER) {
            let mut fields = fields.split_whitespace();
            let len: usize = fields
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| "pdf: malformed child header".to_string())?;
            let truncated = fields.next() == Some("truncated");
            let body_start = (start + line_end + 1).min(out.len());
            let body = &out[body_start..out.len().min(body_start + len)];
            if body.len() != len {
                return Err(format!(
                    "pdf: child announced {} text bytes but wrote {}",
                    len,
                    body.len()
                ));
            }
            return Ok(ExtractedDocument {
                text: String::from_utf8_lossy(body).into_owned(),
                title: None,
                truncated,
            });
        }
        if line_end >= rest.len() {
            return Err("pdf: extraction child produced no result".to_string());
        }
        start += line_end + 1;
    }
}

/// The child's side: `retrivio documents extract-pdf <path> [--text-limit <bytes>]`. Applies
/// the memory limit, reads the file, extracts under `catch_unwind` and writes the framed text
/// to stdout. Exit code 0 on success, 1 with the reason on stderr otherwise. In debug builds
/// `RETRIVIO_PDF_EXTRACT_HANG_MS` makes the child sleep first (the timeout test).
pub(crate) fn pdf_child_main(path: &Path, text_bytes: usize) -> i32 {
    apply_child_memory_limit(PDF_CHILD_MEMORY_BYTES);
    // The parent reads the first stderr line as the failure reason: keep the default panic
    // hook's `thread 'main' panicked at ...` line out of the way and report the panic below.
    std::panic::set_hook(Box::new(|_| {}));
    if cfg!(debug_assertions) {
        if let Some(ms) = std::env::var("RETRIVIO_PDF_EXTRACT_HANG_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            std::thread::sleep(Duration::from_millis(ms));
        }
    }
    let raw = match std::fs::read(path) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("cannot read {}: {}", path.display(), e);
            return 1;
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| extract_pdf_bytes(&raw, text_bytes)));
    let doc = match result {
        Ok(Ok(doc)) => doc,
        Ok(Err(e)) => {
            eprintln!("{}", e);
            return 1;
        }
        Err(panic) => {
            eprintln!("parser panicked: {}", panic_message(panic.as_ref()));
            return 1;
        }
    };
    let mut stdout = std::io::stdout().lock();
    let header = format!(
        "{}{}{}\n",
        PDF_CHILD_HEADER,
        doc.text.len(),
        if doc.truncated { " truncated" } else { "" }
    );
    if stdout
        .write_all(header.as_bytes())
        .and_then(|_| stdout.write_all(doc.text.as_bytes()))
        .and_then(|_| stdout.flush())
        .is_err()
    {
        return 1;
    }
    0
}

/// `retrivio documents <subcommand>`: the hidden entry point the indexer's PDF child uses.
/// Returns the process exit code.
pub fn run_documents_cmd(args: &[OsString]) -> i32 {
    let usage = "usage: retrivio documents extract-pdf <path> [--text-limit <bytes>]";
    let Some(sub) = args.first().map(|a| a.to_string_lossy().to_string()) else {
        eprintln!("{}", usage);
        return 2;
    };
    if sub != "extract-pdf" {
        eprintln!("{}", usage);
        return 2;
    }
    let mut path: Option<PathBuf> = None;
    let mut text_bytes = usize::MAX;
    let mut i = 1usize;
    while i < args.len() {
        let s = args[i].to_string_lossy().to_string();
        if s == "--text-limit" {
            i += 1;
            text_bytes = match args.get(i).and_then(|v| v.to_string_lossy().parse().ok()) {
                Some(n) => n,
                None => {
                    eprintln!("{}", usage);
                    return 2;
                }
            };
        } else if path.is_none() {
            path = Some(PathBuf::from(&args[i]));
        } else {
            eprintln!("{}", usage);
            return 2;
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("{}", usage);
        return 2;
    };
    pdf_child_main(&path, text_bytes)
}

/// Bound the child's address space where the kernel honours `RLIMIT_AS`. Best effort: macOS
/// rejects the call (`EINVAL`) and relies on the parent's resident-size watchdog instead.
fn apply_child_memory_limit(bytes: u64) {
    // SAFETY: getrlimit/setrlimit with a valid, initialised rlimit struct.
    unsafe {
        let mut rl = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_AS, &mut rl) != 0 {
            return;
        }
        let wanted = bytes as libc::rlim_t;
        if rl.rlim_max != libc::RLIM_INFINITY && rl.rlim_max < wanted {
            return;
        }
        rl.rlim_cur = wanted;
        let _ = libc::setrlimit(libc::RLIMIT_AS, &rl);
    }
}

/// Resident set size of a process, when the platform tells us.
#[cfg(target_os = "macos")]
fn process_resident_bytes(pid: u32) -> Option<u64> {
    // SAFETY: proc_pid_rusage fills the struct for RUSAGE_INFO_V0; a non-zero return means
    // the pid is gone or not ours, and the struct is left untouched.
    unsafe {
        let mut info: libc::rusage_info_v0 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V0,
            &mut info as *mut libc::rusage_info_v0 as *mut libc::rusage_info_t,
        );
        (rc == 0).then_some(info.ri_resident_size)
    }
}

#[cfg(target_os = "linux")]
fn process_resident_bytes(pid: u32) -> Option<u64> {
    let statm = std::fs::read_to_string(format!("/proc/{}/statm", pid)).ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    Some(pages.saturating_mul(page.max(1) as u64))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_resident_bytes(_pid: u32) -> Option<u64> {
    None
}

/// Drain a pipe to its end, keeping at most `cap` bytes.
fn read_capped(mut reader: impl Read, cap: usize) -> Vec<u8> {
    let mut kept: Vec<u8> = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let room = cap.saturating_sub(kept.len());
                kept.extend_from_slice(&buf[..n.min(room)]);
            }
        }
    }
    kept
}

/// The command that runs the PDF extraction: this very binary, as
/// `retrivio documents extract-pdf <path> --text-limit <bytes>`.
#[cfg(not(test))]
fn pdf_child_command(path: &Path, text_bytes: usize) -> Command {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("retrivio"));
    let mut cmd = Command::new(exe);
    cmd.arg("documents")
        .arg("extract-pdf")
        .arg(path)
        .arg("--text-limit")
        .arg(text_bytes.to_string());
    cmd
}

/// Under `cargo test` the only retrivio binary a unit test can rely on is the test binary,
/// whose entry point is libtest. The child is therefore the test binary running the named
/// helper test, which reads its arguments from the environment and runs [`pdf_child_main`];
/// the framed protocol lets the parent skip libtest's own output lines.
#[cfg(test)]
fn pdf_child_command(path: &Path, text_bytes: usize) -> Command {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = Command::new(exe);
    cmd.args([
        "documents::tests::pdf_child_helper",
        "--exact",
        "--nocapture",
    ])
    .env("RETRIVIO_PDF_CHILD_PATH", path)
    .env("RETRIVIO_PDF_CHILD_TEXT_LIMIT", text_bytes.to_string());
    if let Some(ms) = tests::child_hang_ms() {
        cmd.env("RETRIVIO_PDF_EXTRACT_HANG_MS", ms.to_string());
    }
    cmd
}

/// Run the PDF extraction in a child process and enforce the bounds from outside it: the
/// child is killed when it runs past `limits.pdf_timeout` or grows past
/// [`PDF_CHILD_MEMORY_BYTES`] resident; either is one failed document. Its stdout is read
/// concurrently (bounded), its stderr's first line is the failure reason.
fn run_pdf_child(path: &Path, limits: &ExtractLimits) -> Result<ExtractedDocument, String> {
    let mut cmd = pdf_child_command(path, limits.text_bytes);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("pdf: cannot start the extraction child: {}", e))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "pdf: extraction child has no stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "pdf: extraction child has no stderr".to_string())?;
    let out_cap = limits.text_bytes.saturating_add(64 * 1024);
    let out_reader = std::thread::spawn(move || read_capped(stdout, out_cap));
    let err_reader = std::thread::spawn(move || read_capped(stderr, 16 * 1024));
    let deadline = Instant::now() + limits.pdf_timeout;
    let mut peak_rss: u64 = 0;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(e) => {
                break Err(format!(
                    "pdf: waiting for the extraction child failed: {}",
                    e
                ))
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break Err(format!(
                "pdf: extraction exceeded document_extract_timeout_ms={}; child killed",
                limits.pdf_timeout.as_millis()
            ));
        }
        if let Some(rss) = process_resident_bytes(child.id()) {
            peak_rss = peak_rss.max(rss);
            if rss > PDF_CHILD_MEMORY_BYTES {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!(
                    "pdf: extraction used {} MiB of memory (limit {} MiB); child killed",
                    rss / (1024 * 1024),
                    PDF_CHILD_MEMORY_BYTES / (1024 * 1024)
                ));
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let out = out_reader.join().unwrap_or_default();
    let err = err_reader.join().unwrap_or_default();
    let status = status?;
    if !status.success() {
        let reason = String::from_utf8_lossy(&err)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("extraction child exited with {}", status));
        return Err(if reason.starts_with("pdf:") {
            reason
        } else {
            format!("pdf: {}", reason)
        });
    }
    parse_pdf_child_output(&out)
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod fixtures {
    //! Synthetic documents built in memory for tests (never copies of real files).
    use std::io::{Cursor, Write};
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    /// A deflate-compressed zip with the given (name, content) entries.
    pub(crate) fn zip_of(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut w = ZipWriter::new(Cursor::new(Vec::new()));
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, content) in entries {
            w.start_file(*name, opts).expect("start zip entry");
            w.write_all(content.as_bytes()).expect("write zip entry");
        }
        w.finish().expect("finish zip").into_inner()
    }

    pub(crate) fn docx(paragraphs: &[&str], title: Option<&str>) -> Vec<u8> {
        let body: String = paragraphs
            .iter()
            .map(|p| {
                format!(
                    "<w:p><w:r><w:t xml:space=\"preserve\">{}</w:t></w:r></w:p>",
                    p
                )
            })
            .collect();
        let document = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>{}\
<w:sectPr/></w:body></w:document>",
            body
        );
        let mut entries: Vec<(String, String)> = vec![
            (
                "[Content_Types].xml".to_string(),
                "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>"
                    .to_string(),
            ),
            ("word/document.xml".to_string(), document),
        ];
        if let Some(t) = title {
            entries.push((
                "docProps/core.xml".to_string(),
                format!(
                    "<cp:coreProperties xmlns:cp=\"x\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{}</dc:title></cp:coreProperties>",
                    t
                ),
            ));
        }
        let refs: Vec<(&str, &str)> = entries
            .iter()
            .map(|(n, c)| (n.as_str(), c.as_str()))
            .collect();
        zip_of(&refs)
    }

    /// Slides as lists of paragraphs; `notes[i]` (when `Some`) is the notes text of slide i+1,
    /// linked through the slide's relationships file.
    pub(crate) fn pptx(slides: &[&[&str]], notes: &[Option<&str>]) -> Vec<u8> {
        let mut entries: Vec<(String, String)> =
            vec![("[Content_Types].xml".to_string(), "<Types/>".to_string())];
        for (i, paragraphs) in slides.iter().enumerate() {
            let n = i + 1;
            let body: String = paragraphs
                .iter()
                .map(|p| {
                    format!(
                        "<a:p><a:r><a:rPr lang=\"en-US\"/><a:t>{}</a:t></a:r></a:p>",
                        p
                    )
                })
                .collect();
            entries.push((
                format!("ppt/slides/slide{}.xml", n),
                format!(
                    "<p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"><p:cSld><p:spTree><p:sp><p:txBody>{}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
                    body
                ),
            ));
            if let Some(Some(note)) = notes.get(i) {
                // Deliberately number the notes part differently from the slide so the test
                // proves the relationship lookup, not the naming convention.
                let notes_no = n + 10;
                entries.push((
                    format!("ppt/slides/_rels/slide{}.xml.rels", n),
                    format!(
                        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide\" Target=\"../notesSlides/notesSlide{}.xml\"/><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/></Relationships>",
                        notes_no
                    ),
                ));
                entries.push((
                    format!("ppt/notesSlides/notesSlide{}.xml", notes_no),
                    format!(
                        "<p:notes xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:fld type=\"slidenum\"><a:t>{}</a:t></a:fld></a:p><a:p><a:r><a:t>{}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>",
                        n, note
                    ),
                ));
            }
        }
        let refs: Vec<(&str, &str)> = entries
            .iter()
            .map(|(n, c)| (n.as_str(), c.as_str()))
            .collect();
        zip_of(&refs)
    }

    /// One workbook with the given sheets: (name, rows of cells). Cells starting with `=`
    /// are stored as formula string results (`t="str"`), cells that parse as numbers as
    /// numbers, `TRUE`/`FALSE` as booleans, every other odd cell as a shared string and
    /// even cell as an inline string, so every cell kind is exercised.
    pub(crate) fn xlsx(sheets: &[(&str, &[&[&str]])]) -> Vec<u8> {
        let mut shared: Vec<String> = Vec::new();
        let mut sheet_xml: Vec<String> = Vec::new();
        for (_, rows) in sheets {
            let mut rows_xml = String::new();
            for (r, cells) in rows.iter().enumerate() {
                rows_xml.push_str(&format!("<row r=\"{}\">", r + 1));
                for (c, cell) in cells.iter().enumerate() {
                    let col = (b'A' + (c as u8 % 26)) as char;
                    let reference = format!("{}{}", col, r + 1);
                    let xml = if let Some(formula_result) = cell.strip_prefix('=') {
                        format!(
                            "<c r=\"{}\" t=\"str\"><f>SUM(A1)</f><v>{}</v></c>",
                            reference, formula_result
                        )
                    } else if cell.parse::<f64>().is_ok() {
                        format!("<c r=\"{}\"><v>{}</v></c>", reference, cell)
                    } else if *cell == "TRUE" || *cell == "FALSE" {
                        format!(
                            "<c r=\"{}\" t=\"b\"><v>{}</v></c>",
                            reference,
                            if *cell == "TRUE" { 1 } else { 0 }
                        )
                    } else if c % 2 == 0 {
                        shared.push((*cell).to_string());
                        format!(
                            "<c r=\"{}\" t=\"s\"><v>{}</v></c>",
                            reference,
                            shared.len() - 1
                        )
                    } else {
                        format!(
                            "<c r=\"{}\" t=\"inlineStr\"><is><t>{}</t></is></c>",
                            reference, cell
                        )
                    };
                    rows_xml.push_str(&xml);
                }
                rows_xml.push_str("</row>");
            }
            sheet_xml.push(format!(
                "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>{}</sheetData></worksheet>",
                rows_xml
            ));
        }
        let shared_xml = format!(
            "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"{}\" uniqueCount=\"{}\">{}</sst>",
            shared.len(),
            shared.len(),
            shared
                .iter()
                .map(|s| format!("<si><t>{}</t></si>", s))
                .collect::<String>()
        );
        let workbook = format!(
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets>{}</sheets></workbook>",
            sheets
                .iter()
                .enumerate()
                .map(|(i, (name, _))| format!(
                    "<sheet name=\"{}\" sheetId=\"{}\" r:id=\"rId{}\"/>",
                    name,
                    i + 1,
                    i + 1
                ))
                .collect::<String>()
        );
        let rels = format!(
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">{}</Relationships>",
            sheets
                .iter()
                .enumerate()
                .map(|(i, _)| format!(
                    "<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{}.xml\"/>",
                    i + 1,
                    i + 1
                ))
                .collect::<String>()
        );
        let mut entries: Vec<(String, String)> = vec![
            ("[Content_Types].xml".to_string(), "<Types/>".to_string()),
            ("xl/workbook.xml".to_string(), workbook),
            ("xl/_rels/workbook.xml.rels".to_string(), rels),
            ("xl/sharedStrings.xml".to_string(), shared_xml),
        ];
        for (i, xml) in sheet_xml.into_iter().enumerate() {
            entries.push((format!("xl/worksheets/sheet{}.xml", i + 1), xml));
        }
        let refs: Vec<(&str, &str)> = entries
            .iter()
            .map(|(n, c)| (n.as_str(), c.as_str()))
            .collect();
        zip_of(&refs)
    }

    pub(crate) fn odt(paragraphs: &[&str]) -> Vec<u8> {
        let body: String = paragraphs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    format!("<text:h text:outline-level=\"1\">{}</text:h>", p)
                } else {
                    format!(
                        "<text:p text:style-name=\"P1\"><text:span>{}</text:span></text:p>",
                        p
                    )
                }
            })
            .collect();
        let content = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><office:text>{}</office:text></office:body></office:document-content>",
            body
        );
        zip_of(&[
            ("mimetype", "application/vnd.oasis.opendocument.text"),
            ("content.xml", content.as_str()),
        ])
    }

    /// A minimal uncompressed PDF: one Helvetica text stream per page, `Tj` per line.
    pub(crate) fn pdf(pages: &[&[&str]]) -> Vec<u8> {
        let mut objects: Vec<Vec<u8>> = Vec::new();
        let kids: Vec<String> = (0..pages.len())
            .map(|i| format!("{} 0 R", 4 + 2 * i))
            .collect();
        objects.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
        objects.push(
            format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                kids.join(" "),
                pages.len()
            )
            .into_bytes(),
        );
        objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
        for (i, lines) in pages.iter().enumerate() {
            let content_no = 5 + 2 * i;
            objects.push(
                format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>",
                    content_no
                )
                .into_bytes(),
            );
            let ops: Vec<String> = lines
                .iter()
                .map(|l| {
                    format!(
                        "({}) Tj T*",
                        l.replace('\\', "\\\\")
                            .replace('(', "\\(")
                            .replace(')', "\\)")
                    )
                })
                .collect();
            let stream = format!("BT /F1 12 Tf 72 720 Td 14 TL {} ET", ops.join(" "));
            let mut obj = format!("<< /Length {} >>\nstream\n", stream.len()).into_bytes();
            obj.extend_from_slice(stream.as_bytes());
            obj.extend_from_slice(b"\nendstream");
            objects.push(obj);
        }
        let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
        let mut offsets: Vec<usize> = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(obj);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for off in offsets {
            out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                objects.len() + 1,
                xref
            )
            .as_bytes(),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures;
    use super::*;
    use std::cell::Cell;
    use std::path::PathBuf;

    thread_local! {
        /// When set, the PDF child started from this thread sleeps this long first (see
        /// [`pdf_child_command`]); thread-local so parallel tests do not hang each other.
        static CHILD_HANG_MS: Cell<Option<u64>> = const { Cell::new(None) };
    }

    pub(super) fn child_hang_ms() -> Option<u64> {
        CHILD_HANG_MS.with(|c| c.get())
    }

    fn extract(name: &str, raw: &[u8]) -> Result<Option<ExtractedDocument>, String> {
        extract_from_bytes(Path::new(name), raw, &ExtractLimits::default())
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("documents-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Helper body, not a test of its own: the PDF child under `cargo test` (see
    /// [`pdf_child_command`]). Reads its arguments from the environment, runs the real child
    /// entry point and exits with its code before libtest prints anything after it.
    #[test]
    fn pdf_child_helper() {
        let Ok(path) = std::env::var("RETRIVIO_PDF_CHILD_PATH") else {
            return;
        };
        let text_bytes = std::env::var("RETRIVIO_PDF_CHILD_TEXT_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let code = pdf_child_main(Path::new(&path), text_bytes);
        std::process::exit(code);
    }

    #[test]
    fn scanner_handles_tags_text_comments_cdata_and_entities() {
        let events: Vec<XmlEvent> = XmlScanner::new(
            "<?xml version=\"1.0\"?><!-- c --><a:p x=\"1>2\"><a:t>A &amp; B &#233;&#x41;</a:t><br/><![CDATA[<raw>]]></a:p>",
        )
        .collect();
        assert_eq!(
            events,
            vec![
                XmlEvent::Start {
                    name: "a:p".into(),
                    attrs: "x=\"1>2\"".into(),
                    self_closing: false
                },
                XmlEvent::Start {
                    name: "a:t".into(),
                    attrs: String::new(),
                    self_closing: false
                },
                XmlEvent::Text("A & B éA".into()),
                XmlEvent::End("a:t".into()),
                XmlEvent::Start {
                    name: "br".into(),
                    attrs: String::new(),
                    self_closing: true
                },
                XmlEvent::Text("<raw>".into()),
                XmlEvent::End("a:p".into()),
            ]
        );
        assert_eq!(attr_value("x=\"1>2\" t='s' r=A1", "t"), Some("s".into()));
        assert_eq!(attr_value("x=\"1>2\" t='s' r=A1", "r"), Some("A1".into()));
        assert_eq!(
            attr_value("Type=\"a/notesSlide\" Target=\"../n.xml\"", "target"),
            Some("../n.xml".into())
        );
        assert_eq!(
            attr_value("rt=\"1\"", "t"),
            None,
            "no partial attribute names"
        );
        // Malformed input: an unterminated tag never panics.
        let _: Vec<XmlEvent> = XmlScanner::new("text <unterminated attr=\"x").collect();
        let _: Vec<XmlEvent> = XmlScanner::new("<").collect();
        let _: Vec<XmlEvent> = XmlScanner::new("<!--").collect();
    }

    #[test]
    fn docx_paragraphs_come_out_in_order_with_title() {
        let raw = fixtures::docx(
            &[
                "Heading One",
                "The quick brown fox &amp; friends.",
                "Second paragraph",
            ],
            Some("Fox Report"),
        );
        let doc = extract("report.docx", &raw).unwrap().unwrap();
        assert_eq!(
            doc.text,
            "Heading One\nThe quick brown fox & friends.\nSecond paragraph"
        );
        assert_eq!(doc.title.as_deref(), Some("Fox Report"));
    }

    #[test]
    fn docx_tabs_breaks_fields_and_deletions() {
        let document = "<w:document xmlns:w=\"w\"><w:body>\
<w:p><w:r><w:t>Name</w:t></w:r><w:r><w:tab/><w:t>Value</w:t></w:r><w:r><w:br/><w:t>next line</w:t></w:r></w:p>\
<w:p><w:r><w:instrText>PAGE</w:instrText></w:r><w:r><w:t>visible</w:t></w:r><w:del><w:r><w:delText>gone</w:delText></w:r></w:del></w:p>\
<w:p/></w:body></w:document>";
        let raw = fixtures::zip_of(&[("word/document.xml", document)]);
        let doc = extract("x.docx", &raw).unwrap().unwrap();
        assert_eq!(doc.text, "Name\tValue\nnext line\nvisible");
        assert_eq!(doc.title, None);
    }

    #[test]
    fn pptx_slides_in_numeric_order_with_notes_after_their_slide() {
        // Ten slides so that lexical order (slide1, slide10, slide2) would be wrong.
        let slide_texts: Vec<Vec<&str>> = (1..=10)
            .map(|i| {
                vec![match i {
                    1 => "Intro slide",
                    2 => "Agenda",
                    10 => "Closing remarks",
                    _ => "Body",
                }]
            })
            .collect();
        let slides: Vec<&[&str]> = slide_texts.iter().map(|v| v.as_slice()).collect();
        let mut notes: Vec<Option<&str>> = vec![None; 10];
        notes[0] = Some("Speaker: welcome the zebra committee");
        notes[9] = Some("Thank the sponsors");
        let raw = fixtures::pptx(&slides, &notes);
        let doc = extract("deck.pptx", &raw).unwrap().unwrap();
        let text = doc.text;
        let pos = |needle: &str| {
            text.find(needle)
                .unwrap_or_else(|| panic!("{} in {}", needle, text))
        };
        assert!(text.starts_with("Slide 1\nIntro slide\nNotes:\nSpeaker: welcome the zebra committee\n\nSlide 2\nAgenda"), "{}", text);
        assert!(pos("Slide 2\n") < pos("Slide 10\n"));
        assert!(pos("Slide 9\n") < pos("Slide 10\n"));
        assert!(
            text.ends_with("Slide 10\nClosing remarks\nNotes:\nThank the sponsors"),
            "{}",
            text
        );
        assert_eq!(text.matches("Notes:").count(), 2);
        assert!(
            !text.contains("Slide 1\n1\n"),
            "slide-number field skipped: {}",
            text
        );
    }

    #[test]
    fn pptx_without_slides_is_an_error_and_a_corrupt_zip_is_an_error() {
        let raw = fixtures::zip_of(&[("[Content_Types].xml", "<Types/>")]);
        let err = extract("empty.pptx", &raw).unwrap_err();
        assert!(err.contains("no slides"), "{}", err);
        let err = extract("bad.pptx", b"PK\x03\x04 this is not really a zip").unwrap_err();
        assert!(err.contains("not a zip archive"), "{}", err);
        let err = extract("bad.docx", &fixtures::zip_of(&[("other.xml", "<a/>")])).unwrap_err();
        assert!(
            err.contains("missing zip entry word/document.xml"),
            "{}",
            err
        );
    }

    #[test]
    fn odt_headings_paragraphs_and_spans() {
        let raw = fixtures::odt(&["Title line", "Body with <text:s text:c=\"3\"/>gap", "Last"]);
        let doc = extract("notes.odt", &raw).unwrap().unwrap();
        assert_eq!(doc.text, "Title line\nBody with gap\nLast");
    }

    #[test]
    fn odp_pages_and_notes() {
        let content = "<office:document-content xmlns:office=\"o\" xmlns:text=\"t\" xmlns:draw=\"d\" xmlns:presentation=\"p\"><office:body><office:presentation>\
<draw:page draw:name=\"page1\"><draw:frame><draw:text-box><text:p>First slide</text:p></draw:text-box></draw:frame>\
<presentation:notes><draw:frame><draw:text-box><text:p>Remember the demo</text:p></draw:text-box></draw:frame></presentation:notes></draw:page>\
<draw:page draw:name=\"page2\"><draw:frame><draw:text-box><text:p>Second slide</text:p></draw:text-box></draw:frame></draw:page>\
</office:presentation></office:body></office:document-content>";
        let raw = fixtures::zip_of(&[("content.xml", content)]);
        let doc = extract("deck.odp", &raw).unwrap().unwrap();
        assert_eq!(
            doc.text,
            "Slide 1\nFirst slide\nNotes:\nRemember the demo\n\nSlide 2\nSecond slide"
        );
    }

    #[test]
    fn xlsx_rows_cells_and_sheet_headings() {
        let raw = fixtures::xlsx(&[
            (
                "Budget",
                &[
                    &["Item", "Cost", "Approved"][..],
                    &["Widgets", "12.5", "TRUE"][..],
                    &["Total", "=12.5"][..],
                ],
            ),
            ("Notes", &[&["giraffe enclosure quote"][..]]),
        ]);
        let doc = extract("book.xlsx", &raw).unwrap().unwrap();
        assert_eq!(
            doc.text,
            "Sheet: Budget\nItem\tCost\tApproved\nWidgets\t12.5\tTRUE\nTotal\t12.5\n\nSheet: Notes\ngiraffe enclosure quote"
        );
    }

    #[test]
    fn html_drops_script_style_nav_head_and_comments_and_decodes_entities() {
        let html = r#"<!DOCTYPE html><html><head><title>Zoo &amp; Aquarium Plan</title><style>p{color:red}</style><script>var x = "<p>not text</p>";</script></head>
<body><nav><a href="/">Home</a> | <a href="/about">About</a></nav>
<!-- an html comment with <b>tags</b> -->
<h1>Otter&nbsp;Habitat</h1>
<p>Budget is &euro;12&#44;000 &mdash; approved &#x2713;.</p>
<ul><li>First   item</li><li>Second<br>line</li></ul>
<table><tr><th>Animal</th><th>Count</th></tr><tr><td>Otter</td><td>4</td></tr></table>
<pre>keep
  this
    shape</pre>
<script type="text/javascript">document.write("<p>still not text</p>")</script>
<p>Tail&lt;end&gt;</p></body></html>"#;
        let doc = extract("plan.html", html.as_bytes()).unwrap().unwrap();
        assert_eq!(doc.title.as_deref(), Some("Zoo & Aquarium Plan"));
        assert_eq!(
            doc.text,
            "Otter Habitat\n\nBudget is €12,000 — approved ✓.\nFirst item\nSecond\nline\nAnimal\tCount\nOtter\t4\nkeep\nthis\nshape\nTail<end>"
        );
        assert!(!doc.text.contains("not text"));
        assert!(!doc.text.contains("Home"));
        assert!(!doc.text.contains("color"));
        assert_eq!(html_to_text("a<br>b"), "a\nb");
        assert_eq!(html_to_text("<p>x</p><p></p><p>y</p>"), "x\ny");
    }

    #[test]
    fn pdf_text_pages_in_order_and_garbage_is_an_error_not_a_panic() {
        let raw = fixtures::pdf(&[
            &[
                "Quarterly Orion demo readiness review",
                "Bedrock region us-west-2 confirmed",
            ],
            &["Page two: zebra migration checklist"],
        ]);
        let doc = extract_pdf_bytes(&raw, usize::MAX).unwrap();
        assert!(
            doc.text.contains("Quarterly Orion demo readiness review"),
            "{}",
            doc.text
        );
        assert!(
            doc.text.find("Bedrock region us-west-2 confirmed").unwrap()
                < doc
                    .text
                    .find("Page two: zebra migration checklist")
                    .unwrap(),
            "{}",
            doc.text
        );
        assert!(!doc.truncated);
        let err = extract_pdf_bytes(b"%PDF-1.4 garbage without objects", usize::MAX).unwrap_err();
        assert!(err.starts_with("pdf:"), "{}", err);
        // A panic inside a parser is reported, not propagated.
        let panicked = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            panic!("boom")
        }));
        assert!(panicked.is_err());
        assert_eq!(panic_message(panicked.unwrap_err().as_ref()), "boom");
    }

    #[test]
    fn pdf_runs_in_a_child_process_whose_output_is_framed_and_bounded() {
        let dir = scratch_dir("pdf-child");
        let path = dir.join("review.pdf");
        std::fs::write(
            &path,
            fixtures::pdf(&[
                &["Quarterly Orion demo readiness review"],
                &["Page two: zebra migration checklist"],
            ]),
        )
        .unwrap();
        // Through the real parent path: spawn, framed stdout, exit code.
        let doc = extract_from_bytes(
            &path,
            b"ignored: the child reads the file",
            &ExtractLimits::default(),
        )
        .unwrap()
        .unwrap();
        assert!(
            doc.text.contains("Quarterly Orion demo readiness review"),
            "{}",
            doc.text
        );
        assert!(doc.text.contains("zebra migration checklist"));
        assert!(!doc.truncated);

        // The child honours the text limit and says so in its header.
        let capped = extract_from_bytes(&path, b"", &ExtractLimits::with_text_bytes(20))
            .unwrap()
            .unwrap();
        assert!(capped.text.len() <= 20, "{:?}", capped.text);
        assert!(capped.truncated);

        // A broken file is a `pdf:` error carried back from the child's stderr, exit code 1.
        let bad = dir.join("bad.pdf");
        std::fs::write(&bad, b"%PDF-1.4 garbage without objects").unwrap();
        let err = extract_from_bytes(&bad, b"", &ExtractLimits::default()).unwrap_err();
        assert!(err.starts_with("pdf:"), "{}", err);
        // A missing file is reported the same way (the child's read fails).
        let err =
            extract_from_bytes(&dir.join("gone.pdf"), b"", &ExtractLimits::default()).unwrap_err();
        assert!(err.contains("cannot read"), "{}", err);

        // The frame parser itself: junk before the header is skipped, a short body is an error.
        let framed = format!("running 1 test\n{}5\nhello", PDF_CHILD_HEADER);
        assert_eq!(
            parse_pdf_child_output(framed.as_bytes()).unwrap().text,
            "hello"
        );
        let short = format!("{}9 truncated\nhello", PDF_CHILD_HEADER);
        assert!(parse_pdf_child_output(short.as_bytes())
            .unwrap_err()
            .contains("announced 9"));
        assert!(parse_pdf_child_output(b"nothing here\n")
            .unwrap_err()
            .contains("no result"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_hanging_pdf_child_is_killed_at_the_deadline() {
        let dir = scratch_dir("pdf-hang");
        let path = dir.join("slow.pdf");
        std::fs::write(&path, fixtures::pdf(&[&["never read in time"]])).unwrap();
        CHILD_HANG_MS.with(|c| c.set(Some(30_000)));
        let limits = ExtractLimits {
            pdf_timeout: Duration::from_millis(700),
            ..ExtractLimits::default()
        };
        let started = Instant::now();
        let err = extract_from_bytes(&path, b"", &limits).unwrap_err();
        let elapsed = started.elapsed();
        CHILD_HANG_MS.with(|c| c.set(None));
        assert!(
            err.contains("document_extract_timeout_ms=700") && err.contains("child killed"),
            "{}",
            err
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "the parent must not wait for the hang: {:?}",
            elapsed
        );
        // Without the hang the same file extracts fine (the deadline is generous enough).
        let doc = extract_from_bytes(&path, b"", &limits).unwrap().unwrap();
        assert!(doc.text.contains("never read in time"), "{}", doc.text);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn size_cap_refuses_before_reading_and_non_documents_are_none() {
        let dir = scratch_dir("cap");
        let big: PathBuf = dir.join("big.docx");
        std::fs::write(&big, fixtures::docx(&["small really"], None)).unwrap();
        let len = std::fs::metadata(&big).unwrap().len();
        let limits = ExtractLimits::default();
        let err = extract_document_text(&big, len - 1, &limits).unwrap_err();
        assert!(err.contains("exceeds max_document_bytes"), "{}", err);
        let ok = extract_document_text(&big, len, &limits).unwrap().unwrap();
        assert_eq!(ok.text, "small really");
        assert_eq!(
            extract_document_text(&dir.join("missing.md"), 10, &limits).unwrap(),
            None,
            "not a document suffix: no stat, no error"
        );
        assert!(extract_document_text(&dir.join("missing.docx"), 10, &limits).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_limit_stops_collection_and_never_exceeds_the_cap() {
        let paragraphs: Vec<String> = (0..2000)
            .map(|i| format!("paragraph number {}", i))
            .collect();
        let refs: Vec<&str> = paragraphs.iter().map(|s| s.as_str()).collect();
        let raw = fixtures::docx(&refs, None);
        let doc = extract_from_bytes(
            Path::new("long.docx"),
            &raw,
            &ExtractLimits::with_text_bytes(500),
        )
        .unwrap()
        .unwrap();
        assert!(
            doc.text.len() <= 500 && doc.text.len() >= 480,
            "{}",
            doc.text.len()
        );
        assert!(doc.truncated);
        assert!(doc
            .text
            .starts_with("paragraph number 0\nparagraph number 1\n"));
        // Under the cap: complete, not truncated.
        let small = extract_from_bytes(
            Path::new("short.docx"),
            &fixtures::docx(&["one", "two"], None),
            &ExtractLimits::with_text_bytes(500),
        )
        .unwrap()
        .unwrap();
        assert_eq!(small.text, "one\ntwo");
        assert!(!small.truncated);
    }

    #[test]
    fn a_single_huge_text_node_is_cut_at_the_cap() {
        // One 5 MB text node in one paragraph: the cap applies while appending, so the sink
        // never holds the node. Both the XML path and the HTML path.
        let node = "x".repeat(5 * 1024 * 1024);
        let document = format!(
            "<w:document xmlns:w=\"w\"><w:body><w:p><w:r><w:t>{}</w:t></w:r></w:p><w:p><w:r><w:t>after</w:t></w:r></w:p></w:body></w:document>",
            node
        );
        let raw = fixtures::zip_of(&[("word/document.xml", document.as_str())]);
        let doc = extract_from_bytes(
            Path::new("huge.docx"),
            &raw,
            &ExtractLimits::with_text_bytes(1000),
        )
        .unwrap()
        .unwrap();
        assert_eq!(doc.text.len(), 1000);
        assert!(doc.text.chars().all(|c| c == 'x'));
        assert!(doc.truncated);
        assert!(!doc.text.contains("after"), "scanning stopped at the cap");

        let html = format!("<html><body><p>{}</p><p>after</p></body></html>", node);
        let doc = extract_from_bytes(
            Path::new("huge.html"),
            html.as_bytes(),
            &ExtractLimits::with_text_bytes(1000),
        )
        .unwrap()
        .unwrap();
        assert_eq!(doc.text.len(), 1000);
        assert!(doc.truncated);

        // Multi-byte text is cut on a char boundary, never inside a code point.
        let wide = "é".repeat(4000);
        let html = format!("<p>{}</p>", wide);
        let doc = extract_from_bytes(
            Path::new("wide.html"),
            html.as_bytes(),
            &ExtractLimits::with_text_bytes(1001),
        )
        .unwrap()
        .unwrap();
        assert!(
            doc.text.len() <= 1001 && doc.text.len() >= 1000,
            "{}",
            doc.text.len()
        );
        assert!(doc.text.chars().all(|c| c == 'é'));
    }

    #[test]
    fn archives_over_the_declared_size_or_entry_count_are_rejected_unread() {
        let raw = fixtures::docx(&["a normal little document"], Some("Title"));
        let declared: u128 = {
            let mut a = zip::ZipArchive::new(Cursor::new(raw.as_slice())).unwrap();
            (0..a.len())
                .map(|i| u128::from(a.by_index_raw(i).unwrap().size()))
                .sum()
        };
        assert!(declared > 100, "fixture declares {} bytes", declared);
        let tight = ExtractLimits {
            archive_uncompressed_bytes: (declared - 1) as u64,
            ..ExtractLimits::default()
        };
        let err = extract_from_bytes(Path::new("bomb.docx"), &raw, &tight).unwrap_err();
        assert!(
            err.contains("declares") && err.contains("max_document_uncompressed_bytes"),
            "{}",
            err
        );
        let exact = ExtractLimits {
            archive_uncompressed_bytes: declared as u64,
            ..ExtractLimits::default()
        };
        assert_eq!(
            extract_from_bytes(Path::new("ok.docx"), &raw, &exact)
                .unwrap()
                .unwrap()
                .text,
            "a normal little document"
        );

        // Too many entries: refused before any of them is opened.
        let names: Vec<String> = (0..MAX_ZIP_ENTRIES + 1)
            .map(|i| format!("e/{}.xml", i))
            .collect();
        let entries: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), "")).collect();
        let many = fixtures::zip_of(&entries);
        let err = extract_from_bytes(Path::new("many.docx"), &many, &ExtractLimits::default())
            .unwrap_err();
        assert!(
            err.contains(&format!("{} entries", MAX_ZIP_ENTRIES + 1)),
            "{}",
            err
        );

        // A zip entry over the per-entry bound rejects the document; nothing is truncated.
        let big_part = "<w:document xmlns:w=\"w\"><w:body><w:p><w:r><w:t>".to_string()
            + &"y".repeat(20_000)
            + "</w:t></w:r></w:p></w:body></w:document>";
        let raw = fixtures::zip_of(&[("word/document.xml", big_part.as_str())]);
        let small_entries = ExtractLimits {
            entry_bytes: 1000,
            ..ExtractLimits::default()
        };
        let err = extract_from_bytes(Path::new("bigpart.docx"), &raw, &small_entries).unwrap_err();
        assert!(
            err.contains("word/document.xml") && err.contains("at most 1000"),
            "{}",
            err
        );
        // The actual size is checked too: a lying header (declared small) is caught on read.
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        w.start_file("word/document.xml", opts).unwrap();
        w.write_all(big_part.as_bytes()).unwrap();
        let mut lying = w.finish().unwrap().into_inner();
        // Rewrite the central directory's uncompressed size to 10: the entry is 20 KB+.
        let real_size = (big_part.len() as u32).to_le_bytes();
        let mut patched = 0;
        let mut i = 0;
        while i + 4 <= lying.len() {
            if lying[i..i + 4] == real_size {
                lying[i..i + 4].copy_from_slice(&10u32.to_le_bytes());
                patched += 1;
            }
            i += 1;
        }
        assert!(patched >= 1, "size field not found to patch");
        let err = extract_from_bytes(Path::new("liar.docx"), &lying, &small_entries).unwrap_err();
        assert!(err.contains("word/document.xml"), "{}", err);
    }

    #[test]
    fn an_ampersand_followed_by_multibyte_text_is_not_a_panic() {
        // Regression: the entity window `rest[..12]` cut inside `≤` on real docx files
        // (drug monographs with `&` near `≤`, `≥`, `μ`, `³`).
        assert_eq!(decode_entities("5 &≤ 10 ≥ 3 μ"), "5 &≤ 10 ≥ 3 μ");
        assert_eq!(
            decode_entities("dose & 100 μg/m³ ≥ 2"),
            "dose & 100 μg/m³ ≥ 2"
        );
        assert_eq!(decode_entities("a &amp;≤ b"), "a &≤ b");
        assert_eq!(decode_entities("&≤≤≤≤≤≤"), "&≤≤≤≤≤≤");
        assert_eq!(decode_entities("&"), "&");
        let document = "<w:document xmlns:w=\"w\"><w:body><w:p><w:r><w:t>Cmax &≤ 12 μg &amp; ≥ 3 mg/m³</w:t></w:r></w:p></w:body></w:document>";
        let raw = fixtures::zip_of(&[("word/document.xml", document)]);
        let doc = extract("mono.docx", &raw).unwrap().unwrap();
        assert_eq!(doc.text, "Cmax &≤ 12 μg & ≥ 3 mg/m³");
    }

    #[test]
    fn invalid_utf8_inside_xml_is_replaced_not_fatal() {
        let mut document = b"<w:document xmlns:w=\"w\"><w:body><w:p><w:r><w:t>caf".to_vec();
        document.extend_from_slice(&[0xff, 0xfe, 0xc3]);
        document.extend_from_slice(b" ok</w:t></w:r></w:p></w:body></w:document>");
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        w.start_file("word/document.xml", opts).unwrap();
        w.write_all(&document).unwrap();
        let raw = w.finish().unwrap().into_inner();
        let doc = extract("bytes.docx", &raw).unwrap().unwrap();
        assert!(doc.text.starts_with("caf"), "{:?}", doc.text);
        assert!(doc.text.ends_with(" ok"), "{:?}", doc.text);
        assert!(doc.text.contains('\u{fffd}'), "{:?}", doc.text);
        // HTML bytes likewise.
        let mut html = b"<p>na".to_vec();
        html.push(0xff);
        html.extend_from_slice(b"ve</p>");
        let doc = extract("bytes.html", &html).unwrap().unwrap();
        assert_eq!(doc.text, "na\u{fffd}ve");
    }

    #[test]
    fn suffix_lists_are_consistent() {
        for s in DOCUMENT_ONLY_SUFFIXES {
            assert!(is_document_suffix(s));
        }
        assert!(is_document_suffix(".html") && is_document_suffix(".htm"));
        assert!(!is_document_suffix(".md") && !is_document_suffix("docx"));
        assert_eq!(suffix_of(Path::new("A/B.DOCX")), ".docx");
    }
}
