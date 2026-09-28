//! The deterministic description layer (picker spec, Phase 1a): a title, its source and a
//! kind for every indexed file, read from the head of the file itself, never from stored
//! chunk text (the indexer collapses whitespace, which destroys the line boundaries titles
//! need). Display-only: nothing here feeds an embedding.

use std::path::Path;

use crate::code_intel;
use crate::util::collapse_whitespace;

/// Bump when the extraction rules change; `describe_pending` re-describes rows with a lower
/// version.
pub(crate) const EXTRACT_VERSION: i64 = 3;
/// Characters of a file's head the extractors look at.
pub(crate) const HEAD_CHARS: usize = 4096;

/// What the picker shows for a file: the title, where it came from, and the kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileMeta {
    pub(crate) title: String,
    /// `frontmatter | docprops | h1 | crawl | subject | first_sentence | filename`.
    pub(crate) title_source: &'static str,
    /// See [`classify_doc_kind`].
    pub(crate) doc_kind: &'static str,
}

/// Titles that say nothing about the file; a candidate equal to one of these (case-insensitive)
/// is skipped in favour of the next source.
const GENERIC_WORDS: &[&str] = &[
    "notes",
    "note",
    "readme",
    "untitled",
    "document",
    "document1",
    "index",
    "home",
    "title",
    "todo",
    "changelog",
    "summary",
    "overview",
    "new document",
    "presentation",
    "presentation1",
    "slide 1",
    "sheet1",
    "book1",
    "draft",
    "misc",
    "scratch",
];

/// Scripts and build fragments whose `#` comments would otherwise read as headings. They keep
/// their file name as the title and the kind classifier calls them `code`, as it does the
/// languages the AST chunker knows (`code_intel`).
const SCRIPT_EXTS: &[&str] = &[
    "sh", "bash", "zsh", "fish", "ps1", "sql", "rb", "kt", "swift", "scala", "php", "cs", "lua",
    "pl", "r", "mk",
];

/// Configuration formats: file-name title, kind `config`.
const CONFIG_EXTS: &[&str] = &[
    "yml", "yaml", "toml", "ini", "cfg", "conf", "env", "lock", "plist", "json",
];

/// Data formats where the extension is the information: file-name title, kind `data`.
const DATA_EXTS: &[&str] = &["xml", "csv", "tsv", "jsonl", "ndjson", "parquet"];

/// Well-known file names (case-sensitive) that keep their name as the title. Dotfiles
/// (`.bashrc`, `.env.local`, `.dockerignore` ...) do too, by the leading dot.
const KEEP_NAME_BASENAMES: &[&str] = &[
    "Makefile",
    "CMakeLists.txt",
    "Dockerfile",
    "Justfile",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
    "CODEOWNERS",
];

/// The first [`HEAD_CHARS`] characters of a file's bytes, newlines intact, BOM removed.
pub(crate) fn head_text(raw: &[u8]) -> String {
    let end = raw.len().min(HEAD_CHARS * 4);
    let text = String::from_utf8_lossy(&raw[..end]);
    text.trim_start_matches('\u{feff}')
        .chars()
        .take(HEAD_CHARS)
        .collect()
}

fn basename(rel_path: &str) -> &str {
    rel_path.rsplit('/').next().unwrap_or(rel_path)
}

fn ext_of(rel_path: &str) -> String {
    Path::new(basename(rel_path))
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn is_code_path(rel_path: &str) -> bool {
    code_intel::language_for_extension(&ext_of(rel_path)).is_some()
}

/// Files whose title is always the file name: code, scripts, configs, data, dotfiles and the
/// well-known build and licence files.
fn keeps_file_name(rel_path: &str) -> bool {
    let base = basename(rel_path);
    let ext = ext_of(rel_path);
    base.starts_with('.')
        || is_code_path(rel_path)
        || SCRIPT_EXTS.contains(&ext.as_str())
        || CONFIG_EXTS.contains(&ext.as_str())
        || DATA_EXTS.contains(&ext.as_str())
        || KEEP_NAME_BASENAMES.contains(&base)
}

/// Prose formats whose first sentence may serve as a title; extensionless files do not qualify.
fn is_prose_ext(rel_path: &str) -> bool {
    matches!(
        ext_of(rel_path).as_str(),
        "md" | "markdown" | "txt" | "text" | "rst" | "adoc" | "org"
    )
}

/// One line, surrounding quotes and emphasis marks removed, whitespace collapsed.
fn clean_title(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*' | '_'))
        .trim();
    collapse_whitespace(trimmed)
}

/// A heading's text without its closing run of `#`, which counts only when a space precedes
/// it: `Learning C#` keeps its hash, `Title ##` loses two, `Title##` keeps both.
fn strip_closing_hashes(text: &str) -> &str {
    let t = text.trim();
    let stripped = t.trim_end_matches('#');
    if stripped.len() == t.len() {
        t
    } else if stripped.is_empty() || stripped.ends_with(' ') {
        stripped.trim_end()
    } else {
        t
    }
}

/// The text of an ATX heading line: the leading `#` run and the closing hashes removed.
fn heading_text(line: &str) -> &str {
    strip_closing_hashes(line.trim_start().trim_start_matches('#'))
}

/// The file name without directories and extension, `-`/`_` as spaces.
pub(crate) fn humanize_filename(rel_path: &str) -> String {
    let base = basename(rel_path);
    let stem = match base.rfind('.') {
        Some(i) if i > 0 => &base[..i],
        _ => base,
    };
    let spaced: String = stem
        .chars()
        .map(|c| if c == '-' || c == '_' { ' ' } else { c })
        .collect();
    let words: Vec<&str> = spaced.split_whitespace().collect();
    if words.is_empty() {
        base.to_string()
    } else {
        words.join(" ")
    }
}

/// True for a title that adds nothing over the file name.
pub(crate) fn is_generic_title(title: &str, rel_path: &str) -> bool {
    let t = collapse_whitespace(title).to_ascii_lowercase();
    if t.chars().count() < 3 {
        return true;
    }
    if GENERIC_WORDS.contains(&t.as_str()) {
        return true;
    }
    t == humanize_filename(rel_path).to_ascii_lowercase()
        || t == basename(rel_path).to_ascii_lowercase()
}

/// Lines of the leading front-matter block, both `---` fences included: `Some(n)` when the
/// first line is `---` and a closing `---` follows within 60 lines, else `None` (a `---` that
/// nothing closes is a rule, not front matter).
fn front_matter_len(head: &str) -> Option<usize> {
    let mut lines = head.lines();
    if lines.next()?.trim_end() != "---" {
        return None;
    }
    lines
        .take(60)
        .position(|l| l.trim_end() == "---")
        .map(|close| close + 2)
}

/// `title:` inside a closed front-matter block ([`front_matter_len`]).
fn frontmatter_title(head: &str) -> Option<String> {
    let len = front_matter_len(head)?;
    for line in head.lines().take(len - 1).skip(1) {
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("title") {
            let t = clean_title(value);
            return if t.is_empty() { None } else { Some(t) };
        }
    }
    None
}

/// The head's body lines: a leading front-matter block ([`front_matter_len`]; left in place
/// when nothing closes it) and the contents of backtick or tilde code fences are skipped, so
/// YAML keys and code comments never read as headings or sentences.
fn body_lines(head: &str) -> impl Iterator<Item = &str> {
    let front_matter = front_matter_len(head).unwrap_or(0);
    let mut in_fence = false;
    head.lines().skip(front_matter).filter(move |line| {
        let l = line.trim_start();
        if l.starts_with("```") || l.starts_with("~~~") {
            in_fence = !in_fence;
            return false;
        }
        !in_fence
    })
}

/// The first `# ` heading in the body; lines indented four or more columns are code, not
/// headings. Outside Markdown (`.txt`, extensionless files, extracted documents) only the
/// first non-empty body line may be a heading: a `# ` deeper in plain text is a comment or a
/// numbered item, not a title.
fn h1_title(rel_path: &str, head: &str) -> Option<String> {
    let markdown = matches!(ext_of(rel_path).as_str(), "md" | "markdown");
    let lines = body_lines(head).filter(|l| !l.trim().is_empty());
    for line in lines.take(if markdown { 60 } else { 1 }) {
        let indent: usize = line
            .chars()
            .take_while(|c| c.is_whitespace())
            .map(|c| if c == '\t' { 4 } else { 1 })
            .sum();
        if indent >= 4 {
            continue;
        }
        if let Some(rest) = line.trim_start().strip_prefix("# ") {
            let t = clean_title(strip_closing_hashes(rest));
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

/// `Title: …` as the internal-docs crawler writes it (`URL: … Title: … Crawled: …`): the
/// word must open the line or follow the URL on a line that opens with `URL:`; a `Title:`
/// in running prose is not one.
fn crawl_title(head: &str) -> Option<String> {
    for line in body_lines(head).take(10) {
        let mut value = if let Some(v) = line.strip_prefix("Title:") {
            v
        } else if line.starts_with("URL:") {
            match line.find(" Title:") {
                Some(idx) => &line[idx + " Title:".len()..],
                None => continue,
            }
        } else {
            continue;
        };
        if let Some(cut) = value.find(" Crawled:") {
            value = &value[..cut];
        }
        let t = clean_title(value);
        if !t.is_empty() {
            return Some(t);
        }
    }
    None
}

fn subject_title(head: &str) -> Option<String> {
    for line in body_lines(head).take(30) {
        if let Some(v) = line.trim_start().strip_prefix("Subject:") {
            let t = clean_title(v);
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

/// `To: `, `Date: `, `Source: `, `Subject: `: a short word or phrase (two to twenty-one
/// letters, spaces and hyphens, opening with a letter), then a colon and a space.
fn is_header_line(l: &str) -> bool {
    let Some(colon) = l.find(": ") else {
        return false;
    };
    let key = &l[..colon];
    (2..=21).contains(&key.len())
        && key.starts_with(|c: char| c.is_ascii_alphabetic())
        && key
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-')
}

/// The line without a leading list marker (`- `, `* `, `1. `).
fn strip_list_marker(l: &str) -> &str {
    if let Some(rest) = l.strip_prefix("- ").or_else(|| l.strip_prefix("* ")) {
        return rest.trim_start();
    }
    let digits = l.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 {
        if let Some(rest) = l[digits..].strip_prefix(". ") {
            return rest.trim_start();
        }
    }
    l
}

/// The first sentence of a prose file, at most 90 characters, cut at a word. Header lines
/// (`Date: …`, `Source: …`; the ones other sources read are skipped here too) do not count,
/// and a list marker in front of the sentence is dropped.
fn first_sentence(rel_path: &str, head: &str) -> Option<String> {
    if !is_prose_ext(rel_path) {
        return None;
    }
    for line in body_lines(head).take(20) {
        let l = line.trim();
        if l.is_empty()
            || l.starts_with('#')
            || l.starts_with("```")
            || l.starts_with("http")
            || l.starts_with('|')
            || l.starts_with("---")
            || l.starts_with("<!--")
        {
            continue;
        }
        let l = strip_list_marker(l);
        if is_header_line(l) {
            continue;
        }
        if l.split_whitespace().count() < 3 {
            continue;
        }
        let code_marks = l
            .chars()
            .filter(|c| matches!(c, '{' | '}' | ';' | '=' | '<' | '>'))
            .count();
        if code_marks > 2 {
            return None;
        }
        let mut s = clean_title(l);
        for sep in [". ", "? ", "! "] {
            if let Some(i) = s.find(sep) {
                s.truncate(i + 1);
                break;
            }
        }
        if s.chars().count() > 90 {
            let cut: String = s.chars().take(90).collect();
            let cut = match cut.rfind(' ') {
                Some(i) if i > 40 => cut[..i].to_string(),
                _ => cut,
            };
            s = format!("{}…", cut.trim_end());
        }
        return Some(s);
    }
    None
}

/// The title of a file and the source it came from, in order of trust: front-matter `title:`,
/// the document's own metadata title, the first `# ` heading, a crawler `Title:` line, an
/// email `Subject:`, the first sentence of prose, the humanised file name. Generic candidates
/// ([`is_generic_title`]) are skipped. Code, script, config and data files
/// ([`keeps_file_name`]) keep their file name.
pub(crate) fn extract_title(
    rel_path: &str,
    head: &str,
    doc_title: Option<&str>,
) -> (String, &'static str) {
    if keeps_file_name(rel_path) {
        return (basename(rel_path).to_string(), "filename");
    }
    let candidates: [(Option<String>, &'static str); 6] = [
        (frontmatter_title(head), "frontmatter"),
        (
            doc_title.map(clean_title).filter(|t| !t.is_empty()),
            "docprops",
        ),
        (h1_title(rel_path, head), "h1"),
        (crawl_title(head), "crawl"),
        (subject_title(head), "subject"),
        (first_sentence(rel_path, head), "first_sentence"),
    ];
    for (candidate, source) in candidates {
        if let Some(t) = candidate {
            if !is_generic_title(&t, rel_path) {
                return (t, source);
            }
        }
    }
    // Prose keeps a readable stem; anything else (config, data, binaries) keeps its full file
    // name, extension included, because the extension is the information.
    let fallback = if is_prose_ext(rel_path) {
        humanize_filename(rel_path)
    } else {
        basename(rel_path).to_string()
    };
    (fallback, "filename")
}

/// An email: it opens with `To:` or `From:`, or has two header lines, or has `Subject:` and
/// at least one of `To:`/`From:`/`Cc:` in its first 12 lines. `Subject:` alone is prose.
fn looks_like_email(head: &str) -> bool {
    let first: Vec<&str> = head.lines().take(12).collect();
    let starts = first
        .first()
        .map(|l| l.starts_with("To:") || l.starts_with("From:"))
        .unwrap_or(false);
    let subject = first.iter().any(|l| l.starts_with("Subject:"));
    let header_lines = first
        .iter()
        .filter(|l| l.starts_with("To:") || l.starts_with("From:") || l.starts_with("Cc:"))
        .count();
    starts || (subject && header_lines >= 1) || header_lines >= 2
}

/// `00:12`, `[01:02:03]`, `00:00:01,000` at the start of a line.
fn is_timecode_line(line: &str) -> bool {
    let l = line.trim_start().trim_start_matches('[');
    let stamp: String = l
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ':')
        .collect();
    stamp.len() >= 4
        && stamp.contains(':')
        && stamp.split(':').all(|p| !p.is_empty() && p.len() <= 2)
}

fn looks_like_transcript(head: &str) -> bool {
    head.lines()
        .take(40)
        .filter(|l| is_timecode_line(l))
        .count()
        >= 3
}

/// The lowercased words of a path, split on `/`, `.`, `_`, `-` and space.
fn path_tokens(rel_path: &str) -> Vec<String> {
    rel_path
        .to_ascii_lowercase()
        .split(['/', '.', '_', '-', ' '])
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

fn has_token(tokens: &[String], words: &[&str]) -> bool {
    tokens.iter().any(|t| words.contains(&t.as_str()))
}

/// The kind of thing a file is, for the picker's badge: `transcript` from an `srt`/`vtt`
/// extension, then the file's format (code, config, data, sheet, deck, pdf, html, doc). What
/// remains is prose: header lines make an `email`, timecodes on three lines a `transcript`;
/// then whole words of the path (`transcript`, `handoff`, `1on1`, `meeting`, `call notes`,
/// `slides`, `frames`, `spec`, `design`, `rfc`, singular or plural) decide, so `specialist`
/// and `wireframes` match nothing; then `md`/`txt`/`file`. A `handoff.sh` is code, a
/// `transcript.pdf` is a pdf, and a `.docx` with `Subject:` or times in it is a doc.
pub(crate) fn classify_doc_kind(rel_path: &str, head: &str) -> &'static str {
    let ext = ext_of(rel_path);
    if matches!(ext.as_str(), "srt" | "vtt") {
        return "transcript";
    }
    if is_code_path(rel_path) || SCRIPT_EXTS.contains(&ext.as_str()) {
        return "code";
    }
    if CONFIG_EXTS.contains(&ext.as_str()) {
        return "config";
    }
    if DATA_EXTS.contains(&ext.as_str()) {
        return "data";
    }
    match ext.as_str() {
        "xlsx" | "xls" | "ods" | "numbers" => return "sheet",
        "pptx" | "ppt" | "odp" => return "deck",
        "pdf" => return "pdf",
        "html" | "htm" => return "html",
        "docx" | "doc" | "odt" | "rtf" | "pages" => return "doc",
        _ => {}
    }
    if looks_like_email(head) {
        return "email";
    }
    if looks_like_transcript(head) {
        return "transcript";
    }
    let tokens = path_tokens(rel_path);
    if has_token(&tokens, &["transcript", "transcripts"]) {
        return "transcript";
    }
    if has_token(&tokens, &["handoff", "handoffs"]) {
        return "handoff";
    }
    if has_token(&tokens, &["1on1", "1on1s", "meeting", "meetings"])
        || tokens.windows(2).any(|w| w[0] == "call" && w[1] == "notes")
        || tokens
            .windows(3)
            .any(|w| w[0] == "1" && w[1] == "on" && w[2] == "1")
    {
        return "notes";
    }
    if has_token(&tokens, &["slides", "frames"])
        || head
            .lines()
            .take(5)
            .any(|l| l.trim_start().starts_with("# Slides"))
    {
        return "slides";
    }
    if has_token(
        &tokens,
        &["spec", "specs", "design", "designs", "rfc", "rfcs"],
    ) {
        return "spec";
    }
    match ext.as_str() {
        "md" | "markdown" => "md",
        "txt" | "text" => "txt",
        _ => "file",
    }
}

/// Width of the badge column.
pub(crate) const KIND_BADGE_WIDTH: usize = 7;

/// The badge text for a kind, at most [`KIND_BADGE_WIDTH`] cells.
pub(crate) fn kind_badge(kind: &str) -> &'static str {
    match kind {
        "email" => "email",
        "transcript" => "transcr",
        "handoff" => "handoff",
        "code" => "code",
        "config" => "config",
        "data" => "data",
        "sheet" => "sheet",
        "deck" => "deck",
        "pdf" => "pdf",
        "html" => "html",
        "doc" => "doc",
        "notes" => "notes",
        "slides" => "slides",
        "spec" => "spec",
        "md" => "md",
        "txt" => "txt",
        _ => "file",
    }
}

/// Title and kind for one file. `head` is [`head_text`] of the raw bytes (or of the extracted
/// text for a document); `doc_title` is the document's own metadata title when it has one.
pub(crate) fn describe_file(rel_path: &str, head: &str, doc_title: Option<&str>) -> FileMeta {
    let (title, title_source) = extract_title(rel_path, head, doc_title);
    FileMeta {
        title,
        title_source,
        doc_kind: classify_doc_kind(rel_path, head),
    }
}

/// Cut at `max` characters on a word boundary, with `…` when cut.
fn cut_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    let head = match head.rfind(' ') {
        Some(i) if i > max / 2 => head[..i].to_string(),
        _ => head,
    };
    format!("{}…", head.trim_end())
}

/// One line about a project from its README: the first heading (unless generic) and the first
/// paragraph, joined with ` — `, at most 160 characters. The paragraph ends at a blank line or
/// the next heading. Front matter, fenced code, HTML, badges, tables, rules, setext underlines
/// and comments are skipped. `None` when the README has no prose.
pub(crate) fn readme_synopsis(text: &str) -> Option<String> {
    let mut heading: Option<String> = None;
    let mut paragraph = String::new();
    for line in body_lines(text).take(80) {
        let l = line.trim();
        if l.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        if l.starts_with('#') {
            if !paragraph.is_empty() {
                break;
            }
            if heading.is_none() {
                let t = clean_title(heading_text(l));
                if !t.is_empty() && !GENERIC_WORDS.contains(&t.to_ascii_lowercase().as_str()) {
                    heading = Some(t);
                }
            }
            continue;
        }
        if l.starts_with("![")
            || l.starts_with("[![")
            || l.starts_with('|')
            || l.starts_with('<')
            || l.starts_with("---")
            || l.starts_with("===")
        {
            continue;
        }
        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(l);
    }
    let paragraph = collapse_whitespace(&paragraph);
    let joined = match (heading, paragraph.is_empty()) {
        (Some(h), true) => h,
        (Some(h), false) => format!("{} — {}", h, paragraph),
        (None, false) => paragraph,
        (None, true) => return None,
    };
    Some(cut_chars(&joined, 160))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h1_wins_when_there_is_no_front_matter() {
        let head =
            "# Slides 08:45 to 19:00 (frames 0035 to 0076) — Acme design review\nSource: frames\n";
        let (title, source) = extract_title("customer-signals/Acme/20260918-review.md", head, None);
        assert_eq!(
            title,
            "Slides 08:45 to 19:00 (frames 0035 to 0076) — Acme design review"
        );
        assert_eq!(source, "h1");
    }

    #[test]
    fn front_matter_title_beats_h1() {
        let head = "---\ndate: 2026-09-01\ntitle: \"Semantic layer strategy\"\n---\n# Draft\n";
        let (title, source) = extract_title("docs/strategy.md", head, None);
        assert_eq!(title, "Semantic layer strategy");
        assert_eq!(source, "frontmatter");
    }

    #[test]
    fn document_metadata_title_is_used_unless_generic() {
        let (title, source) = extract_title("deck/coa.pptx", "", Some("COA Solution Deck"));
        assert_eq!((title.as_str(), source), ("COA Solution Deck", "docprops"));
        let (title, source) = extract_title(
            "deck/coa.pptx",
            "# Agenda for the COA review\n",
            Some("Presentation1"),
        );
        assert_eq!(
            (title.as_str(), source),
            ("Agenda for the COA review", "h1")
        );
    }

    #[test]
    fn crawler_title_line_is_recognised() {
        let head = "URL: https://docs.hub.amazon.dev/cd-signer/ Title: Using entitlements to sign applications — CDSigner user guide Crawled: 2025-11-07T17:42:36Z\n====\n";
        let (title, source) = extract_title("crawl/cdsigner.md", head, None);
        assert_eq!(
            title,
            "Using entitlements to sign applications — CDSigner user guide"
        );
        assert_eq!(source, "crawl");
    }

    #[test]
    fn email_subject_is_a_title() {
        let head = "To: Daniel John, Mike Salonga\nFrom: Eric Stouffer\nSubject: AWS Context POC use case thread\n\nHi all,\n";
        let (title, source) = extract_title("emails/04-thread.md", head, None);
        assert_eq!(
            (title.as_str(), source),
            ("AWS Context POC use case thread", "subject")
        );
    }

    #[test]
    fn first_sentence_for_prose_without_heading() {
        let head = "FieldIQ is an activity logging agent for AWS field roles. It turns what you tell it into a draft.\n";
        let (title, source) = extract_title("notes/fieldiq.md", head, None);
        assert_eq!(
            title,
            "FieldIQ is an activity logging agent for AWS field roles."
        );
        assert_eq!(source, "first_sentence");
    }

    #[test]
    fn long_first_sentence_is_cut_at_a_word_boundary() {
        let head = "This is a very long opening line that keeps going and going without any punctuation to stop it before the ninety character limit is reached at all\n";
        let (title, source) = extract_title("notes/long.txt", head, None);
        assert_eq!(source, "first_sentence");
        assert!(title.chars().count() <= 91, "{}", title);
        assert!(title.ends_with('…'));
        assert!(!title.contains("  "));
    }

    #[test]
    fn filename_is_the_fallback_and_is_humanised() {
        let (title, source) = extract_title(
            "customer-signals/Acme/20260918-planning-suite_review.md",
            "",
            None,
        );
        assert_eq!(
            (title.as_str(), source),
            ("20260918 planning suite review", "filename")
        );
    }

    #[test]
    fn code_files_keep_their_file_name() {
        let (title, source) = extract_title(
            "crates/retrivio/src/pick.rs",
            "//! The jump and pick flows\n",
            None,
        );
        assert_eq!((title.as_str(), source), ("pick.rs", "filename"));
    }

    #[test]
    fn generic_headings_fall_through() {
        let head = "# Notes\n\nCustomer asked about S3 Tables replication across regions.\n";
        let (title, source) = extract_title("acme/2026-08-notes.md", head, None);
        assert_eq!(source, "first_sentence");
        assert!(title.starts_with("Customer asked about S3 Tables"));
        assert!(is_generic_title("README", "README.md"));
        assert!(is_generic_title("2026 08 notes", "acme/2026-08-notes.md"));
        assert!(!is_generic_title(
            "Semantic layer strategy",
            "docs/strategy.md"
        ));
    }

    #[test]
    fn json_and_config_files_do_not_get_a_sentence_title() {
        let (title, source) =
            extract_title("deck/config.json", "{\"a\": 1, \"b\": 2, \"c\": 3}\n", None);
        assert_eq!((title.as_str(), source), ("config.json", "filename"));
    }

    #[test]
    fn head_text_strips_the_bom_and_caps_length() {
        let mut raw = vec![0xEF, 0xBB, 0xBF];
        raw.extend(std::iter::repeat_n(b'x', 10_000));
        let head = head_text(&raw);
        assert!(head.starts_with('x'));
        assert_eq!(head.chars().count(), HEAD_CHARS);
    }

    #[test]
    fn front_matter_without_a_title_is_not_read_as_body() {
        let head = "---\ndate: 2026-09-01\ntags: aws s3 replication\n---\nCustomer asked about S3 Tables.\n";
        let (title, source) = extract_title("notes/x.md", head, None);
        assert_eq!(
            (title.as_str(), source),
            ("Customer asked about S3 Tables.", "first_sentence")
        );
    }

    #[test]
    fn a_hash_comment_inside_front_matter_is_not_an_h1() {
        let head = "---\n# managed by obsidian\ndate: 2026-09-01\n---\nBody text here now.\n";
        let (title, source) = extract_title("notes/x.md", head, None);
        assert_eq!(
            (title.as_str(), source),
            ("Body text here now.", "first_sentence")
        );
    }

    #[test]
    fn a_hash_comment_inside_a_fence_is_not_an_h1() {
        let head =
            "Setup steps for the demo box\n\n```bash\n# install the toolchain first\ncurl …\n```\n";
        let (title, source) = extract_title("notes/setup.md", head, None);
        assert_eq!(
            (title.as_str(), source),
            ("Setup steps for the demo box", "first_sentence")
        );
    }

    #[test]
    fn shell_scripts_keep_their_file_name() {
        let head = "#!/usr/bin/env bash\n# shellcheck disable=SC2086\nset -euo pipefail\n";
        let (title, source) = extract_title("scripts/build.sh", head, None);
        assert_eq!((title.as_str(), source), ("build.sh", "filename"));
    }

    #[test]
    fn yaml_configs_keep_their_file_name() {
        let head = "# Copyright 2026 Example Corp\nservices:\n";
        let (title, source) = extract_title("docker-compose.yml", head, None);
        assert_eq!((title.as_str(), source), ("docker-compose.yml", "filename"));
    }

    #[test]
    fn makefiles_keep_their_file_name() {
        let (title, source) = extract_title("Makefile", "all: build test lint\n", None);
        assert_eq!((title.as_str(), source), ("Makefile", "filename"));
    }

    #[test]
    fn extensionless_readme_takes_an_h1_but_never_a_sentence() {
        let (title, source) = extract_title(
            "README",
            "# Retrivio quick start\n\nInstall with cargo.\n",
            None,
        );
        assert_eq!((title.as_str(), source), ("Retrivio quick start", "h1"));
        let (title, source) =
            extract_title("README", "Install with cargo and run it once.\n", None);
        assert_eq!((title.as_str(), source), ("README", "filename"));
    }

    #[test]
    fn kinds_from_content_shape() {
        assert_eq!(
            classify_doc_kind("emails/04-thread.md", "To: A, B\nFrom: C\nSubject: x\n"),
            "email"
        );
        assert_eq!(
            classify_doc_kind(
                "calls/2026-09-04-acme.txt",
                "00:00 Eric: hi\n00:12 Tej: hello\n00:40 Eric: agenda\n"
            ),
            "transcript"
        );
        assert_eq!(
            classify_doc_kind(
                "decks/frames.md",
                "# Slides 08:45 to 19:00 (frames 0035 to 0076)\n"
            ),
            "slides"
        );
    }

    #[test]
    fn kinds_from_path() {
        assert_eq!(
            classify_doc_kind("docs/sessions/HANDOFF-2026-09-21.md", "# State\n"),
            "handoff"
        );
        assert_eq!(
            classify_doc_kind("Private/1on1s/1on1 - Christie.md", "- topic\n"),
            "notes"
        );
        assert_eq!(
            classify_doc_kind("customer-signals/Acme/transcript-2026.txt", "hello\n"),
            "transcript"
        );
        assert_eq!(
            classify_doc_kind("docs/superpowers/specs/2026-design.md", "# Design\n"),
            "spec"
        );
        assert_eq!(
            classify_doc_kind("call.srt", "1\n00:00:01,000 --> 00:00:02,000\nhi\n"),
            "transcript"
        );
    }

    #[test]
    fn kinds_from_extension() {
        assert_eq!(classify_doc_kind("src/main.rs", "fn main() {}\n"), "code");
        assert_eq!(classify_doc_kind("scripts/run.sh", "#!/bin/bash\n"), "code");
        assert_eq!(classify_doc_kind("config.toml", "a = 1\n"), "config");
        assert_eq!(classify_doc_kind("data/rows.csv", "a,b\n1,2\n"), "data");
        assert_eq!(classify_doc_kind("tracker.xlsx", ""), "sheet");
        assert_eq!(classify_doc_kind("deck/final.pptx", ""), "deck");
        assert_eq!(classify_doc_kind("deck/preview/final.pdf", ""), "pdf");
        assert_eq!(classify_doc_kind("site/index.html", "<html>"), "html");
        assert_eq!(classify_doc_kind("memo.docx", ""), "doc");
        assert_eq!(classify_doc_kind("notes.md", "plain prose here\n"), "md");
        assert_eq!(classify_doc_kind("notes.txt", "plain prose here\n"), "txt");
        assert_eq!(classify_doc_kind("blob.bin", ""), "file");
    }

    #[test]
    fn badges_fit_the_column() {
        for kind in [
            "email",
            "transcript",
            "handoff",
            "code",
            "config",
            "data",
            "sheet",
            "deck",
            "pdf",
            "html",
            "doc",
            "notes",
            "slides",
            "spec",
            "md",
            "txt",
            "file",
            "unknown",
        ] {
            assert!(
                kind_badge(kind).chars().count() <= KIND_BADGE_WIDTH,
                "{}",
                kind
            );
        }
        assert_eq!(kind_badge("transcript"), "transcr");
        assert_eq!(kind_badge("unknown"), "file");
    }

    #[test]
    fn format_beats_path_words() {
        assert_eq!(
            classify_doc_kind("scripts/handoff.sh", "#!/bin/bash\n"),
            "code"
        );
        assert_eq!(
            classify_doc_kind("transcripts/index.csv", "a,b\n1,2\n"),
            "data"
        );
        assert_eq!(
            classify_doc_kind("customer-signals/Acme/transcript-2026.pdf", ""),
            "pdf"
        );
    }

    #[test]
    fn describe_file_bundles_title_and_kind() {
        let meta = describe_file("emails/04-thread.md", "To: A\nSubject: POC thread\n", None);
        assert_eq!(
            meta,
            FileMeta {
                title: "POC thread".to_string(),
                title_source: "subject",
                doc_kind: "email"
            }
        );
    }

    #[test]
    fn readme_synopsis_joins_heading_and_first_paragraph() {
        let text = "# Retrivio\n\n![badge](x.png)\n\nSemantic project memory for local files.\nIndexes folders and answers by meaning.\n\n## Install\nmore\n";
        assert_eq!(
            readme_synopsis(text).as_deref(),
            Some("Retrivio — Semantic project memory for local files. Indexes folders and answers by meaning.")
        );
    }

    #[test]
    fn readme_synopsis_skips_generic_heading_and_fences() {
        let text = "# README\n```\ncode\n```\nA tracker for the semantic-layer customer list.\n";
        assert_eq!(
            readme_synopsis(text).as_deref(),
            Some("A tracker for the semantic-layer customer list.")
        );
        assert_eq!(readme_synopsis("\n\n"), None);
    }

    #[test]
    fn readme_synopsis_is_cut_at_160() {
        let long = format!("# T\n\n{}\n", "word ".repeat(80));
        let s = readme_synopsis(&long).unwrap();
        assert!(s.chars().count() <= 161, "{}", s);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn readme_synopsis_skips_html_and_setext_underlines() {
        let text = "<p align=\"center\">\n  <img src=\"logo.png\">\n</p>\n\n# Project\nReal description.\n";
        assert_eq!(
            readme_synopsis(text).as_deref(),
            Some("Project — Real description.")
        );
        assert_eq!(
            readme_synopsis("Retrivio\n========\nDescription.\n").as_deref(),
            Some("Retrivio Description.")
        );
    }

    #[test]
    fn readme_synopsis_stops_at_the_next_heading() {
        let text = "Some intro.\n## Install\nRun cargo install.\n";
        assert_eq!(readme_synopsis(text).as_deref(), Some("Some intro."));
    }

    // Fix wave, item 7: keep-name coverage.
    #[test]
    fn dotfiles_make_fragments_and_cmake_lists_keep_their_file_name() {
        for (path, head) in [
            (
                ".bashrc",
                "# ~/.bashrc: executed by bash for non-login shells\nexport PATH\n",
            ),
            (
                "home/.env.local",
                "# local overrides for the dev box\nAPI_KEY=x\n",
            ),
            (
                "build/build.mk",
                "# build rules for the whole tree\nall: lint\n",
            ),
            (
                "CMakeLists.txt",
                "# CMake project file for the demo\ncmake_minimum_required(VERSION 3.20)\n",
            ),
            (".zshrc", "# zsh startup for this machine\n"),
            (".npmrc", "registry=https://example.invalid\n"),
            (".dockerignore", "target\n"),
        ] {
            let (title, source) = extract_title(path, head, None);
            assert_eq!(
                (title.as_str(), source),
                (basename(path), "filename"),
                "{}",
                path
            );
        }
    }

    // Fix wave, item 8a: an unclosed front-matter block is not front matter.
    #[test]
    fn unclosed_front_matter_has_no_title() {
        assert_eq!(
            frontmatter_title("---\ntitle: Foo bar\nbody without closing fence\n"),
            None
        );
        assert_eq!(
            frontmatter_title("---\ntitle: Foo bar\n---\n"),
            Some("Foo bar".to_string())
        );
        let (title, source) = extract_title(
            "notes/x.md",
            "---\ntitle: Foo bar\nbody without closing fence\n",
            None,
        );
        assert_eq!(
            (title.as_str(), source),
            ("body without closing fence", "first_sentence")
        );
    }

    // Fix wave, item 8b: `Title:` counts at the line start or on a crawler `URL:` line.
    #[test]
    fn crawl_title_is_anchored_to_the_line_start_or_a_url_line() {
        assert_eq!(
            crawl_title("Notes from the meeting about the book Title: Something\n"),
            None
        );
        assert_eq!(
            crawl_title("Title: Direct title line\n"),
            Some("Direct title line".to_string())
        );
        assert_eq!(
            crawl_title("URL: https://x Title: From the crawler Crawled: 2025\n"),
            Some("From the crawler".to_string())
        );
        let (title, source) = extract_title(
            "notes/x.md",
            "Notes from the meeting about the book Title: Something\n",
            None,
        );
        assert_eq!(
            (title.as_str(), source),
            (
                "Notes from the meeting about the book Title: Something",
                "first_sentence"
            )
        );
    }

    // Fix wave, item 8c: outside Markdown a `# ` heading counts only on the first line.
    #[test]
    fn txt_headings_count_only_on_the_first_non_empty_line() {
        let (title, source) = extract_title(
            "notes/a.txt",
            "Some intro line here first.\n\n# Not a heading in plain text\n",
            None,
        );
        assert_eq!(
            (title.as_str(), source),
            ("Some intro line here first.", "first_sentence")
        );
        let (title, source) = extract_title("notes/a.txt", "\n\n# Real heading\nbody\n", None);
        assert_eq!((title.as_str(), source), ("Real heading", "h1"));
        // Markdown keeps accepting a later heading.
        let (title, source) = extract_title(
            "notes/a.md",
            "Some intro line here first.\n\n# Later heading\n",
            None,
        );
        assert_eq!((title.as_str(), source), ("Later heading", "h1"));
    }

    // Fix wave, item 8d: a closing `#` run is stripped only after a space.
    #[test]
    fn closing_hashes_are_stripped_only_after_a_space() {
        let (title, _) = extract_title("notes/a.md", "# Learning C#\n", None);
        assert_eq!(title, "Learning C#");
        let (title, _) = extract_title("notes/a.md", "# Migration plan ##\n", None);
        assert_eq!(title, "Migration plan");
        let (title, _) = extract_title("notes/a.md", "# Migration plan##\n", None);
        assert_eq!(title, "Migration plan##");
        assert_eq!(
            readme_synopsis("## Learning C# ##\nA course.\n").as_deref(),
            Some("Learning C# — A course.")
        );
    }

    // Fix wave, item 8e: header lines are skipped and list markers dropped.
    #[test]
    fn first_sentence_skips_header_lines_and_list_markers() {
        let head = "To: A\nDate: 2026-09-01\nSource: internal docs crawler\n- Customer asked about S3 Tables replication.\n";
        assert_eq!(
            first_sentence("notes/x.md", head),
            Some("Customer asked about S3 Tables replication.".to_string())
        );
        assert_eq!(
            first_sentence("notes/x.md", "1. First numbered point about the plan.\n"),
            Some("First numbered point about the plan.".to_string())
        );
        assert_eq!(
            first_sentence("notes/x.md", "* Starred point about the plan.\n"),
            Some("Starred point about the plan.".to_string())
        );
        assert_eq!(
            first_sentence("notes/x.md", "Subject: not a sentence at all\n"),
            None
        );
    }

    // Fix wave, item 9a: path words are whole tokens.
    #[test]
    fn path_words_match_whole_tokens() {
        let prose = "plain prose here\n";
        assert_eq!(classify_doc_kind("aws-specialist/notes.md", prose), "md");
        assert_eq!(classify_doc_kind("inspection-checklist.md", prose), "md");
        assert_eq!(classify_doc_kind("wireframes.md", prose), "md");
        assert_eq!(classify_doc_kind("redesign-notes.md", prose), "md");
        assert_eq!(classify_doc_kind("docs/specs/x.md", prose), "spec");
        assert_eq!(classify_doc_kind("frames/0035.md", prose), "slides");
        assert_eq!(
            classify_doc_kind("customers/call-notes-2026.md", prose),
            "notes"
        );
        assert_eq!(
            classify_doc_kind("customers/call notes 2026.md", prose),
            "notes"
        );
        assert_eq!(classify_doc_kind("team/1-on-1 alice.md", prose), "notes");
        assert_eq!(
            classify_doc_kind("transcripts/2026-09-04.md", prose),
            "transcript"
        );
    }

    // Fix wave, item 9b: `Subject:` alone is not an email.
    #[test]
    fn email_needs_a_subject_and_a_header() {
        assert_eq!(
            classify_doc_kind("notes/x.md", "Subject: just a word used in prose\n\nbody\n"),
            "md"
        );
        assert_eq!(
            classify_doc_kind("notes/x.md", "Subject: x\nFrom: a@b\n\nbody\n"),
            "email"
        );
        assert_eq!(classify_doc_kind("notes/x.md", "From: a\nCc: b\n"), "email");
        assert_eq!(classify_doc_kind("notes/x.md", "To: a\n"), "email");
    }

    // Fix wave addendum E12: the email sniff is for prose too; a document with header lines
    // keeps its format kind.
    #[test]
    fn email_sniff_applies_to_prose_only() {
        let head = "To: A, B\nFrom: C\nSubject: Q3 plan\n\nHi all,\n";
        assert_eq!(classify_doc_kind("memo.docx", head), "doc");
        assert_eq!(classify_doc_kind("memo.pdf", head), "pdf");
        assert_eq!(classify_doc_kind("tracker.xlsx", head), "sheet");
        assert_eq!(classify_doc_kind("mail.json", head), "config");
        assert_eq!(classify_doc_kind("thread.md", head), "email");
        assert_eq!(classify_doc_kind("thread.txt", head), "email");
        assert_eq!(classify_doc_kind("thread.eml", head), "email");
        assert_eq!(classify_doc_kind("thread", head), "email");
    }

    // Fix wave, items 9c and 9d: the transcript sniff is for prose; `.key` is not a deck.
    #[test]
    fn transcript_sniff_applies_to_prose_only_and_key_is_not_a_deck() {
        let times = "09:00 Welcome\n09:15 Safety briefing\n09:30 Tour of the floor\n";
        assert_eq!(classify_doc_kind("sop/onboarding.docx", times), "doc");
        assert_eq!(classify_doc_kind("sop/agenda.json", times), "config");
        assert_eq!(classify_doc_kind("sop/onboarding.md", times), "transcript");
        assert_eq!(classify_doc_kind("sop/onboarding.txt", times), "transcript");
        assert_eq!(classify_doc_kind("call.vtt", "WEBVTT\n"), "transcript");
        assert_eq!(classify_doc_kind("deck/talk.key", ""), "file");
        assert_eq!(classify_doc_kind("deck/talk.pptx", ""), "deck");
    }
}
