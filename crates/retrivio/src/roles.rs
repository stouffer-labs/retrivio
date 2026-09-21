//! Temporal roles, text shape, series identity and noise directories (Slice 3).
//!
//! Everything here is pure string work on *relative* paths, file names and chunk text: no
//! file I/O, no database. The indexer stores nothing new; search and recall call these at
//! query time over the candidate set.
//!
//! Roles say what kind of time a document lives in:
//! - `state`: handoffs, status briefs and plans. Meant to be superseded by the next one.
//! - `knowledge`: specs, learnings, product notes, READMEs, code, extracted documents.
//!   Ages, but is never "an event".
//! - `record`: transcripts, call notes, customer signals, meeting notes. Point-in-time events
//!   dated by when they happened, never stale.
//!
//! Classification reads path *components* and file-name tokens, never substrings of the
//! absolute path, so a project folder called `202609-ai-handoff` does not make every file in
//! it a handoff.

/// Temporal role of a document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    State,
    Knowledge,
    Record,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::State => "state",
            Role::Knowledge => "knowledge",
            Role::Record => "record",
        }
    }

    pub fn is_record(self) -> bool {
        matches!(self, Role::Record)
    }
}

/// What the text of a chunk looks like, as far as roles and noise are concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextShape {
    /// Ordinary prose, code or data.
    Prose,
    /// A transcript: a "transcript" heading or subtitle timecodes (speaker-labelled lines count
    /// only in text that kept its newlines, which indexed prose has not).
    Transcript,
    /// A machine dump of a chat: dense `Human:`/`Assistant:` turns or JSON-lines message
    /// records (see [`text_shape`] for the density rule).
    ChatDump,
}

/// Directory names that hold copies rather than originals. Results under them are penalised
/// mildly and lose duplicate collapses to a copy that is not under one.
pub fn is_noise_dir(component: &str) -> bool {
    matches!(
        component.to_ascii_lowercase().as_str(),
        "snapshot"
            | "snapshots"
            | "memory-snapshot"
            | "backup"
            | "backups"
            | "archive"
            | "archived"
            | "copy"
    )
}

/// True when any directory component of the relative path is a noise directory.
pub fn under_noise_dir(doc_rel_path: &str) -> bool {
    dir_components(doc_rel_path).any(is_noise_dir)
}

fn dir_components(doc_rel_path: &str) -> impl Iterator<Item = &str> {
    let mut parts: Vec<&str> = doc_rel_path
        .split(['/', '\\'])
        .filter(|c| !c.is_empty())
        .collect();
    parts.pop(); // the file name
    parts.into_iter()
}

/// Last path component (the file name).
pub fn file_name(doc_rel_path: &str) -> &str {
    doc_rel_path
        .rsplit(['/', '\\'])
        .find(|c| !c.is_empty())
        .unwrap_or("")
}

/// Lower-case extension with its dot (`".md"`), empty when there is none.
pub fn extension(file_name: &str) -> String {
    match file_name.rsplit_once('.') {
        Some((stem, ext))
            if !stem.is_empty()
                && !ext.is_empty()
                && ext.len() <= 10
                && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            format!(".{}", ext.to_ascii_lowercase())
        }
        _ => String::new(),
    }
}

/// Human-written or extracted document formats. Everything else (code, config, data) is
/// knowledge regardless of where it sits.
fn is_document_extension(ext: &str) -> bool {
    matches!(
        ext,
        ".md"
            | ".markdown"
            | ".txt"
            | ".rst"
            | ".adoc"
            | ".srt"
            | ".vtt"
            | ".docx"
            | ".pptx"
            | ".odt"
            | ".odp"
            | ".pdf"
            | ".html"
            | ".htm"
    )
}

/// Lower-case tokens of a file-name stem: split on `-`, `_`, space, `.`, brackets.
fn stem_tokens(file_name: &str) -> Vec<String> {
    let lower = file_name.to_lowercase();
    let stem = match lower.rsplit_once('.') {
        Some((s, ext)) if !s.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric()) => s,
        _ => lower.as_str(),
    };
    stem.split(['-', '_', ' ', '.', '(', ')', '[', ']'])
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Does a user pattern from `recency_record_patterns` match this relative path?
///
/// Patterns are matched against components, never substrings of the whole path:
/// `.ext` matches the file extension; `a/b` matches consecutive directory components;
/// a plain word matches a whole directory component, a whole token of a directory name or a
/// whole token of the file name, exactly or as its plural with `s` (`transcript` matches
/// `transcripts/` and `20260413-transcript-call.txt`, not `transcription-notes.md`; `meeting`
/// matches `meetings/` and `meeting-notes/`, not `meetinghouse/`).
fn user_pattern_matches(
    pattern: &str,
    dirs: &[String],
    dir_tokens: &[String],
    tokens: &[String],
    ext: &str,
) -> bool {
    let p = pattern.trim().to_ascii_lowercase();
    if p.is_empty() {
        return false;
    }
    if p.starts_with('.') && !p.contains('/') {
        return ext == p;
    }
    if p.contains('/') {
        let want: Vec<&str> = p.split('/').filter(|c| !c.is_empty()).collect();
        if want.is_empty() {
            return false;
        }
        return dirs
            .windows(want.len())
            .any(|w| w.iter().zip(want.iter()).all(|(a, b)| a == b));
    }
    let hit = |t: &str| {
        t == p || (t.len() == p.len() + 1 && t.starts_with(p.as_str()) && t.ends_with('s'))
    };
    dirs.iter().any(|d| hit(d))
        || dir_tokens.iter().any(|t| hit(t))
        || tokens.iter().any(|t| hit(t))
}

/// Set of true names for a legacy `recency_record_patterns` entry whose meaning changed in
/// 0.2.0: patterns are matched against path components relative to the project, no longer as
/// substrings of the absolute path. An entry with a `/`, or one that looks absolute (`/...`,
/// `~/...`), was written for the old rule and deserves one note at config load.
pub fn legacy_record_pattern(pattern: &str) -> bool {
    let p = pattern.trim();
    !p.is_empty() && (p.contains('/') || p.starts_with('~'))
}

/// Classify a document from its path relative to the project, the shape of the retrieved
/// chunk text and the user's extra record patterns. State rules win over record rules so a
/// `HANDOFF` pattern left in an old config cannot turn handoffs back into records.
pub fn classify(doc_rel_path: &str, shape: TextShape, user_record_patterns: &[String]) -> Role {
    let name = file_name(doc_rel_path);
    let ext = extension(name);
    if !ext.is_empty() && !is_document_extension(&ext) {
        return Role::Knowledge;
    }
    let dirs: Vec<String> = dir_components(doc_rel_path)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let tokens = stem_tokens(name);

    // State: handoffs, dated status briefs, plans under specs, everything in docs/sessions.
    // File-name tokens must match whole: `handoff` in `HANDOFF-2026-06-10.md` or
    // `successor-handoff.md`, never inside another word. `status` alone is too common (HTTP
    // status notes, `status.md` of a service): it counts with a date in the file name.
    let in_docs_sessions = dirs
        .windows(2)
        .any(|w| w[0] == "docs" && w[1] == "sessions");
    let handoff_dir = dirs.iter().any(|d| d == "handoff" || d == "handoffs");
    let handoff_name = tokens.iter().any(|t| t == "handoff" || t == "handoffs");
    let dated_name = tokens.iter().any(|t| is_date_like_digits(t));
    let status_name = tokens.iter().any(|t| t == "status") && dated_name;
    let plan_under_specs = tokens.last().map(|t| t == "plan").unwrap_or(false)
        && dirs
            .iter()
            .any(|d| matches!(d.as_str(), "specs" | "spec" | "plans" | "planning"));
    if in_docs_sessions || handoff_dir || handoff_name || status_name || plan_under_specs {
        return Role::State;
    }

    // Records: transcripts, customer signals, call and meeting notes, subtitles, transcripts
    // by content shape for bare .txt files. Directory names match whole (`meetings/`,
    // `meeting-notes/`; not `meetinghouse/` or `meeting-tools/`), file-name tokens match whole,
    // and `call-notes` as a token sequence of the file name (`call-notes.md`,
    // `2026-03-01_call_notes.md`) counts like the `call-notes/` directory; `recall-notes.md`
    // and `call-with-acme.md` do not.
    let record_dir = dirs.iter().any(|d| {
        matches!(
            d.as_str(),
            "transcripts"
                | "transcript"
                | "customer-signals"
                | "call-notes"
                | "meeting"
                | "meetings"
                | "meeting-notes"
                | "meeting_notes"
                | "discussionlog"
                | "discussionlogs"
                | "discussion-log"
                | "discussion-logs"
        )
    });
    let record_name = tokens.iter().any(|t| {
        matches!(
            t.as_str(),
            "transcript" | "transcripts" | "meeting" | "meetings"
        )
    }) || tokens.windows(2).any(|w| w[0] == "call" && w[1] == "notes");
    let subtitle = matches!(ext.as_str(), ".srt" | ".vtt");
    let text_transcript = ext == ".txt" && shape == TextShape::Transcript;
    if record_dir || record_name || subtitle || text_transcript {
        return Role::Record;
    }
    if user_record_patterns.is_empty() {
        return Role::Knowledge;
    }
    // Tokens of the directory names, computed once for all patterns: this runs for every
    // candidate and every evidence hit of a query.
    let dir_tokens: Vec<String> = dirs.iter().flat_map(|d| stem_tokens(d)).collect();
    if user_record_patterns
        .iter()
        .any(|p| user_pattern_matches(p, &dirs, &dir_tokens, &tokens, &ext))
    {
        return Role::Record;
    }
    Role::Knowledge
}

/// JSON message-record markers (Claude Code / OpenAI transcripts in JSON lines).
const JSON_MESSAGE_MARKERS: &[&str] = &[
    "\"type\":\"user\"",
    "\"type\": \"user\"",
    "\"type\":\"assistant\"",
    "\"type\": \"assistant\"",
    "\"role\":\"user\"",
    "\"role\": \"user\"",
    "\"role\":\"assistant\"",
    "\"role\": \"assistant\"",
    "\"parentUuid\"",
    "\"sessionId\"",
];

fn json_message_markers(text: &str) -> usize {
    JSON_MESSAGE_MARKERS
        .iter()
        .map(|m| text.matches(m).count())
        .sum()
}

/// Shape of a chunk's text. Only the first 4 KB are examined.
///
/// A chat dump needs *density*, not a mention: with line structure, three or more lines that
/// start with a turn marker (`Human:`, `Assistant:`, `User:`, `AI:`, `System:`, markdown
/// decoration tolerated) making up at least a fifth of the non-empty lines, or JSON message
/// records on at least half of the lines; a README that documents `Human:` and `Assistant:`
/// once each is prose. Indexed prose has its newlines collapsed to spaces, so the same test
/// runs on whitespace-separated tokens: three or more turn markers averaging at least one per
/// 150 tokens, or JSON records opening the text. A dump whose turns run longer than about 150
/// words each can pass as prose; that is the accepted limit of judging collapsed text.
pub fn text_shape(text: &str) -> TextShape {
    let mut end = text.len().min(4096);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let head = &text[..end];
    let tokens: Vec<&str> = head.split_whitespace().collect();
    let lines: Vec<&str> = head
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.len() >= 3 {
        let marker_lines = lines
            .iter()
            .filter(|l| {
                l.split_whitespace()
                    .next()
                    .map(is_chat_turn_token)
                    .unwrap_or(false)
            })
            .count();
        let json_lines = lines
            .iter()
            .filter(|l| l.starts_with('{') && json_message_markers(l) > 0)
            .count();
        if (marker_lines >= 3 && marker_lines * 5 >= lines.len())
            || (json_lines >= 2 && json_lines * 2 >= lines.len())
        {
            return TextShape::ChatDump;
        }
    }
    let chat_turns = tokens.iter().filter(|t| is_chat_turn_token(t)).count();
    if (chat_turns >= 3 && chat_turns * 150 >= tokens.len())
        || (json_message_markers(head) >= 2 && head.trim_start().starts_with('{'))
    {
        return TextShape::ChatDump;
    }

    // Transcripts: a heading that says so, subtitle timecodes, or speaker-labelled lines from
    // at least two speakers (line shape survives only in text that kept its newlines).
    let heading_says_transcript = first_heading(head)
        .map(|h| h.contains("transcript"))
        .unwrap_or(false);
    let timecodes = tokens
        .windows(2)
        .filter(|w| w[1] == "-->" && is_timecode(w[0]))
        .count();
    let mut speaker_lines = 0usize;
    let mut speakers: Vec<String> = Vec::new();
    for raw in head.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(label) = speaker_label(line) {
            speaker_lines += 1;
            if !speakers.contains(&label) {
                speakers.push(label);
            }
        }
    }
    if heading_says_transcript || timecodes >= 2 || (speaker_lines >= 4 && speakers.len() >= 2) {
        return TextShape::Transcript;
    }
    TextShape::Prose
}

/// Text of the first markdown heading (lower-case), whether or not newlines survived: from
/// the first `#` run to the next `#` run or line end.
fn first_heading(head: &str) -> Option<String> {
    let start = head.find('#')?;
    let after = head[start..].trim_start_matches('#');
    if !after.starts_with(' ') {
        return None;
    }
    let body = after.trim_start();
    let stop = body.find(['\n', '#']).unwrap_or(body.len());
    let title = body[..stop].trim();
    if title.is_empty() {
        None
    } else {
        Some(title.to_ascii_lowercase())
    }
}

fn is_chat_turn_token(token: &str) -> bool {
    let t = token.trim_start_matches(['>', '*', '#', '_', '`']);
    let t = t.trim_end_matches(['*', '_', '`']);
    matches!(t, "Human:" | "Assistant:" | "User:" | "AI:" | "System:")
}

/// `00:12:34,567` (SRT) or `00:12.340` (VTT) style timecode token.
fn is_timecode(token: &str) -> bool {
    token.len() >= 5
        && token.chars().filter(|c| *c == ':').count() >= 1
        && token
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, ':' | ',' | '.'))
}

/// `Name:` or `[00:12] Name:` speaker label at the start of a line: a capitalised word or
/// two to four capitalised words, then a colon and text. YAML keys (`title: x`) are lower-case
/// and do not count; markdown bold labels (`**Date:**`) do not count.
fn speaker_label(line: &str) -> Option<String> {
    let mut rest = line;
    if rest.starts_with('[') {
        let close = rest.find(']')?;
        let inside = &rest[1..close];
        if !inside
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, ':' | '.' | ','))
        {
            return None;
        }
        rest = rest[close + 1..].trim_start();
    }
    let colon = rest.find(':')?;
    let label = rest[..colon].trim();
    let after = rest[colon + 1..].trim_start();
    if label.is_empty() || after.is_empty() || label.len() > 40 {
        return None;
    }
    if !label
        .chars()
        .next()
        .map(char::is_uppercase)
        .unwrap_or(false)
    {
        return None;
    }
    if label.contains(['*', '`', '"', '/', '\\', '(', ')', '=']) {
        return None;
    }
    let words: Vec<&str> = label.split_whitespace().collect();
    if words.is_empty() || words.len() > 4 {
        return None;
    }
    if !words.iter().all(|w| {
        w.chars()
            .all(|c| c.is_alphabetic() || matches!(c, '.' | '\'' | '-'))
    }) {
        return None;
    }
    // Chat-turn markers are dumps, not speakers; `Date:` / `Summary:` style headings are
    // document metadata, not people.
    if is_chat_turn_token(&format!("{}:", label)) || is_heading_label(label) {
        return None;
    }
    Some(label.to_string())
}

fn is_heading_label(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "date"
            | "updated"
            | "last updated"
            | "summary"
            | "notes"
            | "note"
            | "title"
            | "author"
            | "status"
            | "subject"
            | "from"
            | "to"
            | "cc"
            | "attendees"
            | "participants"
            | "agenda"
            | "action items"
            | "next steps"
            | "duration"
            | "location"
            | "time"
            | "source"
            | "tags"
            | "see also"
            | "example"
            | "warning"
            | "important"
            | "tip"
    )
}

// ---------------------------------------------------------------------------------------------
// Series identity (shared by search supersession and recall series collapse)
// ---------------------------------------------------------------------------------------------

/// Filename tokens that mark a revision rather than a distinct document.
const VERSION_TOKENS: &[&str] = &["draft", "final", "copy"];

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// `YYYY-MM-DD` / `YYYY_MM_DD` runs and `(<n>)` copy markers become spaces (token separators),
/// so they vanish before the token pass; everything else is kept.
fn blank_dates_and_copy_markers(stem: &str) -> String {
    let chars: Vec<char> = stem.chars().collect();
    let mut out = String::with_capacity(stem.len());
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '(' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && j < chars.len() && chars[j] == ')' {
                out.push(' ');
                i = j + 1;
                continue;
            }
        }
        if c.is_ascii_digit()
            && (i == 0 || !chars[i - 1].is_alphanumeric())
            && i + 10 <= chars.len()
        {
            let d = &chars[i..i + 10];
            let dashed = d[..4].iter().all(|c| c.is_ascii_digit())
                && matches!(d[4], '-' | '_')
                && d[5..7].iter().all(|c| c.is_ascii_digit())
                && matches!(d[7], '-' | '_')
                && d[8..10].iter().all(|c| c.is_ascii_digit())
                && chars
                    .get(i + 10)
                    .map(|n| !n.is_alphanumeric())
                    .unwrap_or(true);
            if dashed {
                out.push(' ');
                i += 10;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Digit-only tokens that read as dates: a year (1900-2099), `YYYYMM` with a valid month or
/// `YYYYMMDD` with a valid month and day. Any other number (`12345`, a ticket id, a serial)
/// tells documents apart and is kept.
fn is_date_like_digits(tok: &str) -> bool {
    if !all_digits(tok) {
        return false;
    }
    let year_ok = |y: &str| y.starts_with("19") || y.starts_with("20");
    let field = |s: &str, lo: u32, hi: u32| {
        s.parse::<u32>()
            .map(|v| (lo..=hi).contains(&v))
            .unwrap_or(false)
    };
    match tok.len() {
        4 => year_ok(tok),
        6 => year_ok(&tok[..4]) && field(&tok[4..6], 1, 12),
        8 => year_ok(&tok[..4]) && field(&tok[4..6], 1, 12) && field(&tok[6..8], 1, 31),
        _ => false,
    }
}

/// A day-of-month token next to a month name (`june-19`, `19-june`).
fn is_day_token(tok: &str) -> bool {
    all_digits(tok)
        && tok.len() <= 2
        && tok
            .parse::<u32>()
            .map(|d| (1..=31).contains(&d))
            .unwrap_or(false)
}

/// Tokens that mark a revision of the same document rather than a different one: date-like
/// digit runs (a year, `YYYYMM`, `YYYYMMDD`; `YYYY-MM-DD` is blanked earlier), `v<n>`,
/// `rev<n>`, `r<n>`, `draft`, `final`, `copy`. Other numbers (`12345`) and digits inside
/// words (`s3`, `ec2`, `core3`) are meaningful and kept.
fn is_series_token(tok: &str) -> bool {
    if all_digits(tok) {
        return is_date_like_digits(tok);
    }
    if VERSION_TOKENS.contains(&tok) {
        return true;
    }
    ["rev", "v", "r"]
        .iter()
        .any(|p| tok.strip_prefix(p).map(all_digits).unwrap_or(false))
}

/// Filename stem with date runs (including a month name with a day or a year next to it),
/// revision markers and copy counters removed, separators squeezed.
pub fn normalize_stem(file_name: &str) -> String {
    let lower = file_name.to_lowercase();
    let stem = match lower.rsplit_once('.') {
        Some((s, ext))
            if !s.is_empty()
                && !ext.is_empty()
                && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            s
        }
        _ => lower.as_str(),
    };
    let blanked = blank_dates_and_copy_markers(stem);
    let toks: Vec<&str> = blanked
        .split(['-', '_', ' ', '.', '(', ')', '[', ']'])
        .filter(|tok| !tok.is_empty())
        .collect();
    let mut parts: Vec<&str> = Vec::with_capacity(toks.len());
    let mut i = 0usize;
    while i < toks.len() {
        let tok = toks[i];
        // Finder-style `name copy 2`: the counter belongs to the copy marker.
        if tok == "copy" && toks.get(i + 1).map(|n| all_digits(n)).unwrap_or(false) {
            i += 2;
            continue;
        }
        // A month name with a day or a year next to it is a date: `june-19`, `19-june`,
        // `june-2026`. A month name on its own is a word.
        if MONTHS.contains(&tok) {
            let next_is_date = toks
                .get(i + 1)
                .map(|n| is_day_token(n) || is_date_like_digits(n))
                .unwrap_or(false);
            let prev_is_day =
                i > 0 && is_day_token(toks[i - 1]) && parts.last() == Some(&toks[i - 1]);
            if next_is_date {
                i += 2;
                continue;
            }
            if prev_is_day {
                parts.pop();
                i += 1;
                continue;
            }
        }
        if !is_series_token(tok) {
            parts.push(tok);
        }
        i += 1;
    }
    parts.join("-")
}

/// Series identity: project, parent directory (relative) and normalised stem. Two files with
/// the same key are revisions of one document (`worklog/2026-06-19-SESSION-HANDOFF.md` and
/// `worklog/2026-06-24-SESSION-HANDOFF.md`); the same stem in another directory is not.
pub fn series_key(project_path: &str, doc_rel_path: &str) -> String {
    let normalized = doc_rel_path.replace('\\', "/");
    let (parent, name) = match normalized.rsplit_once('/') {
        Some((p, n)) => (p, n),
        None => ("", normalized.as_str()),
    };
    format!(
        "{}\u{1}{}\u{1}{}",
        project_path,
        parent.to_lowercase(),
        normalize_stem(name)
    )
}

// ---------------------------------------------------------------------------------------------
// Query hints
// ---------------------------------------------------------------------------------------------

const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
];

fn query_words(query: &str) -> Vec<String> {
    query
        .split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '.'))
        .filter(|w| !w.is_empty())
        .map(|w| w.trim_matches(['-', '.']).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// A year (`2026`), `YYYYMM`, `YYYYMMDD`, `YYYY-MM` or `YYYY-MM-DD` token.
fn looks_like_date_token(w: &str) -> bool {
    let digits: String = w.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() != w.replace('-', "").len() {
        return false;
    }
    match digits.len() {
        4 | 6 | 8 => digits.starts_with("19") || digits.starts_with("20"),
        _ => false,
    }
}

/// `phrase` occurs as consecutive words.
fn has_phrase(words: &[String], phrase: &[&str]) -> bool {
    words
        .windows(phrase.len())
        .any(|w| w.iter().zip(phrase.iter()).all(|(a, b)| a == b))
}

/// True when the prompt asks for history explicitly, in which case superseded state files
/// are shown at full strength: `history`, `historical`, `originally`, `changelog`,
/// `timeline`, `previous`/`earlier`/`older version`, `what did ... say`, `back in`, a month
/// name, a year or a date. Words that merely can refer to the past (`before`, `old`, `ago`,
/// `version` alone) do not count: "Before deploying, read the current status" wants the
/// current status.
pub fn history_query(query: &str) -> bool {
    let words = query_words(query);
    for w in &words {
        if matches!(
            w.as_str(),
            "history" | "historical" | "originally" | "changelog" | "timeline"
        ) {
            return true;
        }
        if MONTHS.contains(&w.as_str()) || looks_like_date_token(w) {
            return true;
        }
    }
    if has_phrase(&words, &["previous", "version"])
        || has_phrase(&words, &["previous", "versions"])
        || has_phrase(&words, &["earlier", "version"])
        || has_phrase(&words, &["earlier", "versions"])
        || has_phrase(&words, &["older", "version"])
        || has_phrase(&words, &["older", "versions"])
        || has_phrase(&words, &["back", "in"])
    {
        return true;
    }
    // "what did <someone> say / said"
    if let Some(pos) = words
        .windows(2)
        .position(|w| w[0] == "what" && w[1] == "did")
    {
        if words[pos + 2..].iter().any(|w| w == "say" || w == "said") {
            return true;
        }
    }
    false
}

/// Role the prompt is asking for, if it says so: transcripts, calls, meetings and what someone
/// said point at records; current status, latest state and "where we left off" point at state.
pub fn role_hint(query: &str) -> Option<Role> {
    let words = query_words(query);
    let lower = query.to_lowercase();
    let record = words.iter().any(|w| {
        matches!(
            w.as_str(),
            "transcript"
                | "transcripts"
                | "call"
                | "calls"
                | "meeting"
                | "meetings"
                | "said"
                | "told"
                | "discussed"
                | "conversation"
                | "recording"
                | "standup"
        )
    });
    let state = words.iter().any(|w| {
        matches!(
            w.as_str(),
            "status" | "handoff" | "handoffs" | "latest" | "resume" | "progress"
        )
    }) || lower.contains("current state")
        || lower.contains("state of")
        || lower.contains("where we left off")
        || lower.contains("where did we leave")
        || lower.contains("up to speed")
        || lower.contains("next steps")
        || lower.contains("pick up where");
    match (record, state) {
        (true, false) => Some(Role::Record),
        (false, true) => Some(Role::State),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pats(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn roles_come_from_relative_components_not_the_project_name() {
        let none: Vec<String> = Vec::new();
        // A project folder named `202609-ai-handoff` says nothing about its files.
        assert_eq!(
            classify("AGENTS.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("projects/AI-Activity-Context.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        // State: handoffs, status, docs/sessions, plans under specs.
        assert_eq!(
            classify(
                "docs/sessions/REPORT-2026-09-19.md",
                TextShape::Prose,
                &none
            ),
            Role::State
        );
        assert_eq!(
            classify(
                "worklog/2026-06-19-SESSION-HANDOFF.md",
                TextShape::Prose,
                &none
            ),
            Role::State
        );
        assert_eq!(
            classify("workshop/HANDOFF.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("HANDOFF-FOR-NEXT-AI.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("PREP-STATUS-2026-09-03.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("20260903-status.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("PREP-STATUS.md", TextShape::Prose, &none),
            Role::Knowledge,
            "status without a date is a note"
        );
        assert_eq!(
            classify("api/status.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("docs/http-status-codes.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify(
                "worklog/2026-09-07-successor-handoff.md",
                TextShape::Prose,
                &none
            ),
            Role::State
        );
        assert_eq!(
            classify("handoffs.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("guides/handoff-process.md", TextShape::Prose, &none),
            Role::State,
            "handoff as a whole token"
        );
        assert_eq!(
            classify("guides/handoffs-in-aviation.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify("guides/prehandoff.md", TextShape::Prose, &none),
            Role::Knowledge,
            "handoff inside a word is not a handoff"
        );
        assert_eq!(
            classify("handoff/notes.md", TextShape::Prose, &none),
            Role::State
        );
        assert_eq!(
            classify(
                "docs/superpowers/specs/2026-09-20-retrivio-0.2-plan.md",
                TextShape::Prose,
                &none
            ),
            Role::State
        );
        assert_eq!(
            classify("plan.md", TextShape::Prose, &none),
            Role::Knowledge,
            "a plan outside specs is a note"
        );
        assert_eq!(
            classify("api/sourcestatus.md", TextShape::Prose, &none),
            Role::Knowledge,
            "status inside a word is not a status brief"
        );
        // Records.
        assert_eq!(
            classify("transcripts/2026-call.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify(
                "20260413-transcript-call-with-customer.txt",
                TextShape::Prose,
                &none
            ),
            Role::Record
        );
        assert_eq!(
            classify(
                "customer-signals/acme/x/analysis.md",
                TextShape::Prose,
                &none
            ),
            Role::Record
        );
        assert_eq!(
            classify("notes/meeting-notes-2026-02-12.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify("meetings/2026-02-12.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify("meeting-notes/2026-02-12.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify("meetinghouse/README.md", TextShape::Prose, &none),
            Role::Knowledge,
            "a directory that starts with meeting is not a meeting"
        );
        assert_eq!(
            classify("meeting-tools/README.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("transcription-service/README.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("docs/transcription-notes.md", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("docs/meetings-with-acme.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify(
                "discussionlog/20260618-sync/transcript.txt",
                TextShape::Prose,
                &none
            ),
            Role::Record
        );
        assert_eq!(
            classify("media/call.SRT", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify("call1-export.txt", TextShape::Transcript, &none),
            Role::Record
        );
        assert_eq!(
            classify("call1-export.txt", TextShape::Prose, &none),
            Role::Knowledge
        );
        // `call-notes` as a file-name token sequence, like the `call-notes/` directory.
        assert_eq!(
            classify("notes/call-notes.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify("2026-03-01_call_notes.md", TextShape::Prose, &none),
            Role::Record
        );
        assert_eq!(
            classify(
                "customers/acme-call-notes-2026-03-01.txt",
                TextShape::Prose,
                &none
            ),
            Role::Record
        );
        assert_eq!(
            classify("recall-notes.md", TextShape::Prose, &none),
            Role::Knowledge,
            "call inside another word is not a call"
        );
        assert_eq!(
            classify("call-with-acme.md", TextShape::Prose, &none),
            Role::Knowledge,
            "call alone is not call notes"
        );
        assert_eq!(
            classify("notes-call.md", TextShape::Prose, &none),
            Role::Knowledge,
            "the two tokens in the other order are not call notes"
        );
        // Code is knowledge wherever it sits.
        assert_eq!(
            classify(
                "archived/meeting-transcribe/transcribe_meeting.py",
                TextShape::Prose,
                &none
            ),
            Role::Knowledge
        );
        assert_eq!(
            classify("docs/sessions/tool.py", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("src/main.rs", TextShape::Prose, &none),
            Role::Knowledge
        );
        assert_eq!(
            classify("specs/design.md", TextShape::ChatDump, &none),
            Role::Knowledge
        );
    }

    #[test]
    fn user_patterns_match_components_not_substrings() {
        let p = pats(&["call-notes", "1on1s", ".vtt", "notes/customers", "HANDOFF"]);
        assert_eq!(
            classify("call-notes/2026-03-01.md", TextShape::Prose, &p),
            Role::Record
        );
        assert_eq!(
            classify("1on1s & team calls/2026.md", TextShape::Prose, &p),
            Role::Record
        );
        assert_eq!(classify("media/x.vtt", TextShape::Prose, &p), Role::Record);
        assert_eq!(
            classify("notes/customers/acme.md", TextShape::Prose, &p),
            Role::Record
        );
        assert_eq!(
            classify("customers/notes/acme.md", TextShape::Prose, &p),
            Role::Knowledge,
            "consecutive components in order"
        );
        // A `HANDOFF` pattern from an old config does not turn handoffs into records.
        assert_eq!(
            classify("docs/sessions/HANDOFF-2026.md", TextShape::Prose, &p),
            Role::State
        );
        // Whole tokens (or their plural), never a prefix or a substring of the whole path.
        assert_eq!(
            classify("recall-notes.md", TextShape::Prose, &pats(&["call"])),
            Role::Knowledge
        );
        assert_eq!(
            classify("call-with-acme.md", TextShape::Prose, &pats(&["call"])),
            Role::Record
        );
        assert_eq!(
            classify("calls/acme.md", TextShape::Prose, &pats(&["call"])),
            Role::Record,
            "plural of the pattern"
        );
        assert_eq!(
            classify("callbacks/acme.md", TextShape::Prose, &pats(&["call"])),
            Role::Knowledge,
            "a prefix is not a match"
        );
        assert_eq!(
            classify("meetinghouse/x.md", TextShape::Prose, &pats(&["meeting"])),
            Role::Knowledge
        );
        assert_eq!(
            classify("team-meeting/x.md", TextShape::Prose, &pats(&["meeting"])),
            Role::Record,
            "a token of the directory name"
        );
        assert_eq!(
            classify("x.md", TextShape::Prose, &pats(&["", "  "])),
            Role::Knowledge
        );
        assert!(legacy_record_pattern("docs/sessions"));
        assert!(legacy_record_pattern("/Users/me/customer-signals/"));
        assert!(legacy_record_pattern("~/notes"));
        assert!(!legacy_record_pattern("transcript"));
        assert!(!legacy_record_pattern(".srt"));
        assert!(!legacy_record_pattern(""));
    }

    #[test]
    fn text_shapes() {
        assert_eq!(
            text_shape("Plain notes about storage.\nMore prose."),
            TextShape::Prose
        );
        assert_eq!(
            text_shape("# Meeting Transcript **Date:** Friday, March 13, 2026\n## Summary\nThe team discussed"),
            TextShape::Transcript
        );
        assert_eq!(
            text_shape("Alice: hello there\nBob: hi\nAlice: how is the migration\nBob: on track\n"),
            TextShape::Transcript
        );
        assert_eq!(
            text_shape("1\n00:00:01,000 --> 00:00:03,500\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld"),
            TextShape::Transcript
        );
        assert_eq!(
            text_shape(
                "Human: fix the tests\n\nAssistant: Sure, running them now.\n\nHuman: thanks\n"
            ),
            TextShape::ChatDump
        );
        assert_eq!(
            text_shape("Human: fix the tests\n\nAssistant: Sure, running them now.\n"),
            TextShape::Prose,
            "two markers are a mention, not a dump"
        );
        // A README that documents the markers once each is prose, with or without newlines.
        let readme = "# Export format\n\nEach turn starts with a marker.\nThe `Human:` marker opens the user's turn.\nThe `Assistant:` marker opens the reply.\nTurns are separated by a blank line.\nThe exporter writes UTF-8.\nMetadata goes first.\nNothing else is special.\n";
        assert_eq!(text_shape(readme), TextShape::Prose);
        assert_eq!(
            text_shape(&readme.split_whitespace().collect::<Vec<_>>().join(" ")),
            TextShape::Prose
        );
        // Three markers in a long prose document: a quoted example, not a dump (density).
        let mut long_doc = String::from("# Guide\n\n");
        for _ in 0..60 {
            long_doc.push_str(
                "A line of ordinary documentation prose about the exporter and its options.\n",
            );
        }
        long_doc.push_str("Human: hello\nAssistant: hi\nHuman: bye\n");
        assert_eq!(text_shape(&long_doc), TextShape::Prose);
        assert_eq!(
            text_shape(&long_doc.split_whitespace().collect::<Vec<_>>().join(" ")),
            TextShape::Prose,
            "collapsed: three markers in 700 tokens is not dense"
        );
        // Turns that wrap over several lines still read as a dump through token density.
        let wrapped = "Human: plan the migration in\nthree waves please\n\nAssistant: The migration moves\nthe fleet in three waves,\none per region.\n\nHuman: and pricing?\n\nAssistant: three tiers,\none per wave.\n";
        assert_eq!(text_shape(wrapped), TextShape::ChatDump);
        assert_eq!(
            text_shape("{\"type\":\"user\",\"message\":{\"role\":\"user\"},\"parentUuid\":null}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\"}}"),
            TextShape::ChatDump
        );
        // Indexed chunk text has its newlines collapsed: the same shapes must still show.
        assert_eq!(
            text_shape("Human: plan the migration Assistant: The migration moves the fleet in three waves. Human: and pricing? Assistant: three tiers."),
            TextShape::ChatDump
        );
        assert_eq!(
            text_shape("# Meeting Transcript **Date:** Friday, March 13, 2026 at 2:05 PM ## Summary The team discussed latency"),
            TextShape::Transcript
        );
        assert_eq!(
            text_shape(
                "1 00:00:01,000 --> 00:00:03,500 Hello 2 00:00:04,000 --> 00:00:06,000 World"
            ),
            TextShape::Transcript
        );
        assert_eq!(
            text_shape(
                "# Design notes Assistant: is a role in the org chart, Human: resources too."
            ),
            TextShape::Prose,
            "two turn markers are a mention"
        );
        assert_eq!(
            text_shape("# Design notes The Human: label appears once here."),
            TextShape::Prose
        );
        assert_eq!(
            text_shape("User: one AI: two User: three AI: four"),
            TextShape::ChatDump,
            "dense markers in collapsed text"
        );
        assert_eq!(
            text_shape(
                "{\"a\": 1, \"role\": \"user\"} and {\"role\": \"assistant\"} in prose about roles"
            ),
            TextShape::ChatDump
        );
        assert_eq!(
            text_shape(
                "prose mentioning \"role\": \"user\" and \"role\": \"assistant\" fields of the API"
            ),
            TextShape::Prose,
            "JSON records must open the text"
        );
        // YAML keys and one `Date:` heading are not speakers; a lone Human: line is not a dump.
        assert_eq!(
            text_shape("---\ntitle: x\ndate: 2026-09-01\n---\nDate: 2026\nSummary: none\nNotes: a\nHuman: b"),
            TextShape::Prose
        );
        assert_eq!(
            text_shape("Alice: one\nAlice: two\nAlice: three\nAlice: four\nAlice: five"),
            TextShape::Prose,
            "one speaker is a list, not a transcript"
        );
    }

    #[test]
    fn noise_dirs() {
        assert!(under_noise_dir("memory-snapshot/-Users-x/memory/MEMORY.md"));
        assert!(under_noise_dir("docs/Backups/old.md"));
        assert!(under_noise_dir("archive/2025/x.md"));
        assert!(
            !under_noise_dir("docs/snapshot.md"),
            "a file name is not a directory"
        );
        assert!(!under_noise_dir("src/backup_tool.rs"));
        assert!(!under_noise_dir("README.md"));
    }

    #[test]
    fn series_stem_normalization() {
        assert_eq!(
            normalize_stem("HANDOFF-2026-08-28-orion.md"),
            "handoff-orion"
        );
        assert_eq!(
            normalize_stem("HANDOFF-2026-08-28-orion.md"),
            normalize_stem("HANDOFF-2026-09-10-orion.md")
        );
        assert_eq!(normalize_stem("design-v2-final.md"), "design");
        assert_eq!(normalize_stem("design.md"), "design");
        assert_eq!(normalize_stem("notes copy (2).md"), "notes");
        assert_eq!(normalize_stem("notes copy 2.md"), "notes");
        assert_eq!(normalize_stem("notes copy.md"), "notes");
        assert_eq!(normalize_stem("notes 2.md"), "notes-2");
        assert_eq!(normalize_stem("Brief (draft).md"), "brief");
        assert_ne!(
            normalize_stem("runbook.md"),
            normalize_stem("HANDOFF-2026-09-10-orion.md")
        );
        assert_eq!(normalize_stem("20260812-vendor-call.txt"), "vendor-call");
        assert_eq!(normalize_stem("202609_notes.md"), "notes");
        assert_eq!(normalize_stem("2026_09_10_notes.md"), "notes");
        assert_eq!(
            normalize_stem("2026-09-19-proactive-recall-design.md"),
            "proactive-recall-design"
        );
        assert_eq!(normalize_stem(".hidden"), "hidden");
        assert_eq!(normalize_stem("README"), "readme");
        assert_ne!(
            normalize_stem("s3-tables.md"),
            normalize_stem("s4-tables.md")
        );
        assert_eq!(normalize_stem("s3-tables.md"), "s3-tables");
        assert_eq!(normalize_stem("ec2-core3-access.md"), "ec2-core3-access");
        assert_ne!(
            normalize_stem("ec2-notes.md"),
            normalize_stem("ec3-notes.md")
        );
        assert_eq!(
            normalize_stem("deck-v11.html"),
            normalize_stem("deck-v12.html")
        );
        assert_eq!(normalize_stem("deck-v11.html"), "deck");
        assert_eq!(normalize_stem("spec-rev3.md"), "spec");
        assert_eq!(normalize_stem("spec_r12.md"), "spec");
        assert_eq!(normalize_stem("spec-final-copy.md"), "spec");
        // Only date-like numbers are revision markers; other numbers tell documents apart.
        assert_eq!(normalize_stem("invoice-12345.pdf"), "invoice-12345");
        assert_ne!(
            normalize_stem("HANDOFF-12345.md"),
            normalize_stem("HANDOFF-67890.md"),
            "ticket ids are not dates"
        );
        assert_eq!(
            normalize_stem("HANDOFF-20260610.md"),
            normalize_stem("HANDOFF-20260624.md")
        );
        assert_eq!(normalize_stem("HANDOFF-20260610.md"), "handoff");
        assert_eq!(normalize_stem("report-2026.md"), "report");
        assert_eq!(normalize_stem("report-1999.md"), "report");
        assert_eq!(normalize_stem("report-3000.md"), "report-3000");
        assert_eq!(
            normalize_stem("notes-20261340.md"),
            "notes-20261340",
            "not a date"
        );
        assert_eq!(
            normalize_stem("notes-202613.md"),
            "notes-202613",
            "month 13"
        );
        assert_eq!(
            normalize_stem("notes-june-19.md"),
            normalize_stem("notes-july-3.md")
        );
        assert_eq!(normalize_stem("notes-june-19.md"), "notes");
        assert_eq!(normalize_stem("19-june-notes.md"), "notes");
        assert_eq!(normalize_stem("notes-june-2026.md"), "notes");
        assert_eq!(
            normalize_stem("june-plan.md"),
            "june-plan",
            "a month alone is a word"
        );
        assert_eq!(normalize_stem("q3-report-123.md"), "q3-report-123");
        assert_eq!(normalize_stem("version-notes.md"), "version-notes");
        assert_eq!(normalize_stem("revenue-drafting.md"), "revenue-drafting");
    }

    #[test]
    fn series_key_includes_project_and_parent_directory() {
        let a = series_key("/p/orion", "worklog/2026-06-19-SESSION-HANDOFF.md");
        let b = series_key("/p/orion", "worklog/2026-06-24-SESSION-HANDOFF.md");
        let c = series_key("/p/orion", "workshop/SESSION-HANDOFF.md");
        let d = series_key("/p/other", "worklog/2026-06-24-SESSION-HANDOFF.md");
        let e = series_key("/p/orion", "worklog/2026-06-10-HANDOFF.md");
        assert_eq!(a, b);
        assert_ne!(a, c, "another directory is another document");
        assert_ne!(a, d, "another project is another document");
        assert_ne!(a, e, "another stem is another document");
        assert_eq!(
            series_key("/p", "HANDOFF-2026-09-10.md"),
            series_key("/p", "handoff-2026-09-17.md")
        );
    }

    #[test]
    fn history_and_role_hints() {
        assert!(history_query("what did the handoff say in June"));
        assert!(history_query("show the previous version of the plan"));
        assert!(history_query("earlier versions of the design"));
        assert!(history_query("notes from 2026-06-19"));
        assert!(history_query("what did Alice say about pricing"));
        assert!(history_query("what was the plan back in 2025"));
        assert!(history_query("back in the spring we chose lancedb"));
        assert!(history_query("the changelog for the workshop"));
        assert!(history_query("history of the demo"));
        assert!(history_query("what was originally proposed"));
        assert!(history_query("timeline of the migration"));
        assert!(history_query("handoff from 20260619"));
        assert!(!history_query("what is the current status of the workshop"));
        assert!(!history_query("S3 Tables replication cost allocation"));
        assert!(!history_query("verify the deployment"));
        assert!(
            !history_query("Before deploying, read the current status"),
            "before is not a history word"
        );
        assert!(!history_query("the old plan is fine, go ahead"));
        assert!(!history_query("two days ago we fixed the tests"));
        assert!(!history_query("compare v2 and v3 of the design"));
        assert!(!history_query("what changed since last month"));
        assert!(!history_query("the version field is missing"));
        assert!(
            !history_query("put the feedback in the review"),
            "back in needs the words"
        );
        assert!(!history_query("what did you do"), "what did without say");

        assert_eq!(
            role_hint("what did Kun say on the standup call"),
            Some(Role::Record)
        );
        assert_eq!(role_hint("summarize the transcript"), Some(Role::Record));
        assert_eq!(
            role_hint("what is the current status of the prep"),
            Some(Role::State)
        );
        assert_eq!(
            role_hint("get up to speed where we left off"),
            Some(Role::State)
        );
        assert_eq!(role_hint("read the handoff and the meeting notes"), None);
        assert_eq!(role_hint("S3 Tables replication cost allocation"), None);
    }
}
