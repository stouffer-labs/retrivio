//! Pure rendering for the picker (spec Phase 1b/1c): sanitising untrusted text, cell-width
//! layout, title-first rows, the header and the preview cards. No I/O, no SQLite.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::describe::{kind_badge, KIND_BADGE_WIDTH};
use crate::util::{collapse_whitespace, strip_terminal_control_sequences, word_tokens};

/// Text safe to hand to fzf under `--ansi`: escape sequences (CSI, SS3, and the string
/// families OSC, DCS, APC, PM, SOS), C0/C1 control characters, bidi controls and invisible
/// formatting characters (zero-width space and non-joiner, the directional marks, word joiner
/// and the invisible operators, the deprecated format characters U+206A–U+206F, the
/// interlinear annotation characters, soft hyphen, Mongolian vowel separator, byte-order mark,
/// tag characters) removed; tabs, newlines and the line and paragraph separators become
/// spaces. Deliberate exception to "zero-width removed": the zero-width joiner (U+200D) and
/// the emoji variation selectors (U+FE0E, U+FE0F) stay, so emoji sequences keep their glyph
/// and their width. Indexed files are untrusted input.
pub(crate) fn sanitize_display(raw: &str) -> String {
    let stripped = strip_terminal_control_sequences(&strip_esc_strings(raw));
    let mut out = String::with_capacity(stripped.len());
    for ch in stripped.chars() {
        match ch {
            '\t' | '\n' | '\r' | '\u{2028}' | '\u{2029}' => out.push(' '),
            c if c.is_control() => {}
            '\u{00AD}'
            | '\u{061C}'
            | '\u{180E}'
            | '\u{200B}'
            | '\u{200C}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{E0000}'..='\u{E007F}' => {}
            c => out.push(c),
        }
    }
    out
}

/// Remove the ESC string sequences, payload included: OSC (`ESC ]`), DCS (`ESC P`), APC
/// (`ESC _`), PM (`ESC ^`) and SOS (`ESC X`), each ended by BEL or `ESC \`. A bare ESC inside
/// the payload ends the string and is handed on as the start of the next sequence; an
/// unterminated string swallows the rest of the text. Any other ESC is left in place for
/// `strip_terminal_control_sequences`.
fn strip_esc_strings(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            out.push(ch);
            continue;
        }
        let mut pending_esc = true;
        while pending_esc {
            pending_esc = false;
            if !matches!(chars.peek(), Some(']' | 'P' | '_' | '^' | 'X')) {
                out.push('\u{1b}');
                break;
            }
            chars.next();
            while let Some(c) = chars.next() {
                if c == '\u{07}' {
                    break;
                }
                if c == '\u{1b}' {
                    if chars.peek() == Some(&'\\') {
                        chars.next();
                    } else {
                        pending_esc = true;
                    }
                    break;
                }
            }
        }
    }
    out
}

pub(crate) fn cells(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// At most `max` cells, ending with `…` when cut.
pub(crate) fn fit_cells(text: &str, max: usize) -> String {
    if cells(text) <= max {
        return text.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w > max - 1 {
            break;
        }
        out.push(ch);
        used += w;
    }
    // Per-character widths under-count sequences the string measure counts as one glyph
    // (emoji presentation, `U+26A0 U+FE0F` is 1 + 0 by char but 2 as a string), so enforce
    // the budget on the assembled string before adding the ellipsis.
    while cells(&out) + 1 > max {
        out.pop();
    }
    out.push('…');
    out
}

/// Exactly `width` cells: cut with `…` or padded with spaces on the right.
pub(crate) fn pad_cells(text: &str, width: usize) -> String {
    let t = fit_cells(text, width);
    let w = cells(&t);
    format!("{}{}", t, " ".repeat(width.saturating_sub(w)))
}

fn take_cells(text: &str, n: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w > n {
            break;
        }
        out.push(ch);
        used += w;
    }
    while cells(&out) > n {
        out.pop();
    }
    out
}

fn take_last_cells(text: &str, n: usize) -> String {
    let mut kept: Vec<char> = Vec::new();
    let mut used = 0usize;
    for ch in text.chars().rev() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w > n {
            break;
        }
        kept.push(ch);
        used += w;
    }
    let mut out: String = kept.iter().rev().collect();
    while cells(&out) > n {
        out.remove(0);
    }
    out
}

/// A path cut to `max` cells with the basename kept whole when at all possible: directories
/// in the middle are replaced by `…` (`first/…/last/base`, then `first/…/base`, then `…/base`);
/// only when the basename alone does not fit is its middle elided, keeping the extension.
pub(crate) fn elide_middle(path: &str, max: usize) -> String {
    if cells(path) <= max {
        return path.to_string();
    }
    let (dir, base) = match path.rfind('/') {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => ("", path),
    };
    if !dir.is_empty() {
        let parts: Vec<&str> = dir.split('/').collect();
        let first = parts[0];
        let mut attempts: Vec<String> = Vec::new();
        if parts.len() >= 2 {
            attempts.push(format!("{}/…/{}/{}", first, parts[parts.len() - 1], base));
        }
        attempts.push(format!("{}/…/{}", first, base));
        // The project name cut, the basename whole: `202606-navigati…/<base>`.
        let first_budget = max.saturating_sub(cells(base) + 1);
        if first_budget >= 6 {
            attempts.push(format!("{}/{}", fit_cells(first, first_budget), base));
        }
        attempts.push(format!("…/{}", base));
        for a in attempts {
            if cells(&a) <= max {
                return a;
            }
        }
    }
    let (stem, ext) = match base.rfind('.') {
        Some(i) if i > 0 => (&base[..i], &base[i..]),
        _ => (base, ""),
    };
    let keep = max.saturating_sub(cells(ext) + 1);
    let head_n = keep * 6 / 10;
    let tail_n = keep.saturating_sub(head_n);
    let mut head = take_cells(stem, head_n);
    let mut tail = take_last_cells(stem, tail_n);
    // Each piece fits its own budget; measure the assembled string as well and trim the wider
    // piece until it fits. Only `…<ext>` itself may exceed a `max` smaller than its width.
    loop {
        let out = format!("{}…{}{}", head, tail, ext);
        if cells(&out) <= max || (head.is_empty() && tail.is_empty()) {
            return out;
        }
        if !head.is_empty() && (tail.is_empty() || cells(&head) >= cells(&tail)) {
            head.pop();
        } else {
            tail.remove(0);
        }
    }
}

/// `today`, `10d`, `3mo`, `2y`; empty for a missing age.
pub(crate) fn age_label(age_days: f64) -> String {
    if !age_days.is_finite() || age_days < 0.0 {
        return String::new();
    }
    let d = age_days.floor() as i64;
    if d < 1 {
        "today".to_string()
    } else if d < 45 {
        format!("{}d", d)
    } else if d < 730 {
        format!("{}mo", ((d as f64) / 30.44).round() as i64)
    } else {
        format!("{}y", ((d as f64) / 365.25).floor() as i64)
    }
}

fn relation_short(relation: &str) -> &'static str {
    match relation {
        "seed" => "seed",
        "same_project" => "project",
        "related_project" => "linked",
        _ => "",
    }
}

/// The row's match column, at most 18 cells: what matched (`words`, `meaning`, `both`, `path`,
/// `weak`) and the graph relation (`seed`, `project` for the same project as a seed, `linked`
/// for a related project); the widest is `meaning · project` at 17. Never a number: the fused
/// score is relative to the other results of the same query.
pub(crate) fn match_label(semantic: f64, lexical: f64, relation: &str, why: &str) -> String {
    let path_hit = why.split('+').any(|p| p == "path") || relation == "path_keyword";
    let what = if lexical >= 0.7 && semantic >= 0.7 {
        "both"
    } else if lexical >= 0.7 {
        "words"
    } else if semantic >= 0.7 {
        "meaning"
    } else if path_hit {
        "path"
    } else {
        "weak"
    };
    let how = relation_short(relation);
    if how.is_empty() {
        what.to_string()
    } else {
        format!("{} · {}", what, how)
    }
}

pub(crate) const GAP: &str = "  ";
pub(crate) const MATCH_WIDTH: usize = 18;
const WHEN_WIDTH: usize = 5;

/// Column widths for one terminal width. Below 100 columns only kind, title and match are
/// shown; from 100 the path joins; from 120 the age too. The title gets 45 % of the flexible
/// space, the path the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowLayout {
    pub(crate) cols: usize,
    pub(crate) kind_w: usize,
    pub(crate) title_w: usize,
    pub(crate) where_w: usize,
    pub(crate) when_w: usize,
    pub(crate) match_w: usize,
}

pub(crate) fn layout_for_width(cols: usize) -> RowLayout {
    let cols = cols.clamp(60, 400);
    let kind_w = KIND_BADGE_WIDTH;
    let match_w = MATCH_WIDTH;
    let show_where = cols >= 100;
    let when_w = if cols >= 120 { WHEN_WIDTH } else { 0 };
    // kind, title and match are always shown; where and when join with width.
    let columns = 3 + usize::from(show_where) + usize::from(when_w > 0);
    let fixed = kind_w + match_w + when_w + (columns - 1) * GAP.len();
    let flexible = cols.saturating_sub(fixed).max(24);
    let (title_w, where_w) = if show_where {
        let t = (flexible * 45 / 100).max(20);
        (t, flexible - t)
    } else {
        (flexible, 0)
    };
    RowLayout {
        cols,
        kind_w,
        title_w,
        where_w,
        when_w,
        match_w,
    }
}

/// Wrap `text` in an SGR sequence when `color` is on. `sgr` is the parameter list (`1`, `2`,
/// `36`, `1;4`).
pub(crate) fn paint(text: &str, sgr: &str, color: bool) -> String {
    if color {
        format!("\x1b[{}m{}\x1b[0m", sgr, text)
    } else {
        text.to_string()
    }
}

pub(crate) struct FileRow<'a> {
    pub(crate) kind: &'a str,
    pub(crate) title: &'a str,
    pub(crate) project: &'a str,
    pub(crate) rel_path: &'a str,
    pub(crate) age_days: f64,
    pub(crate) match_label: &'a str,
}

fn render_cells(
    l: &RowLayout,
    badge: &str,
    title: &str,
    where_: &str,
    age_days: f64,
    match_label: &str,
    color: bool,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(5);
    parts.push(paint(&pad_cells(badge, l.kind_w), "36", color));
    parts.push(paint(
        &pad_cells(&sanitize_display(title), l.title_w),
        "1",
        color,
    ));
    if l.where_w > 0 {
        let w = elide_middle(&sanitize_display(where_), l.where_w);
        parts.push(paint(&pad_cells(&w, l.where_w), "2", color));
    }
    if l.when_w > 0 {
        let a = fit_cells(&age_label(age_days), l.when_w);
        parts.push(paint(
            &format!("{}{}", " ".repeat(l.when_w.saturating_sub(cells(&a))), a),
            "2",
            color,
        ));
    }
    parts.push(paint(
        &pad_cells(&sanitize_display(match_label), l.match_w),
        "2",
        color,
    ));
    parts.join(GAP)
}

/// One title-first row: `type  title  where  when  match`, exactly `l.cols` cells wide
/// (ANSI codes excluded).
pub(crate) fn render_file_row(r: &FileRow<'_>, l: &RowLayout, color: bool) -> String {
    let where_ = format!("{}/{}", r.project, r.rel_path);
    render_cells(
        l,
        kind_badge(r.kind),
        r.title,
        &where_,
        r.age_days,
        r.match_label,
        color,
    )
}

pub(crate) struct DirRow<'a> {
    pub(crate) name: &'a str,
    /// Synopsis, or the fallback the caller built (`N files · top: <title>`).
    pub(crate) about: &'a str,
    pub(crate) age_days: f64,
    pub(crate) match_label: &'a str,
}

/// One project row: `dir  name  about  when  match`.
pub(crate) fn render_dir_row(r: &DirRow<'_>, l: &RowLayout, color: bool) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(5);
    parts.push(paint(&pad_cells("dir", l.kind_w), "36", color));
    parts.push(paint(
        &pad_cells(&sanitize_display(r.name), l.title_w),
        "1",
        color,
    ));
    if l.where_w > 0 {
        parts.push(paint(
            &pad_cells(&sanitize_display(r.about), l.where_w),
            "2",
            color,
        ));
    }
    if l.when_w > 0 {
        let a = fit_cells(&age_label(r.age_days), l.when_w);
        parts.push(paint(
            &format!("{}{}", " ".repeat(l.when_w.saturating_sub(cells(&a))), a),
            "2",
            color,
        ));
    }
    parts.push(paint(
        &pad_cells(&sanitize_display(r.match_label), l.match_w),
        "2",
        color,
    ));
    parts.join(GAP)
}

/// Greedy wrap of `text` on its two-space separators: tokens are packed into lines joined by
/// [`GAP`] while `cells(line) + 2 + cells(token) <= cols`; a token wider than `cols` stands on
/// its own line, cut with [`fit_cells`]. Content and order are unchanged.
fn wrap_gap_tokens(text: &str, cols: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for token in text.split(GAP).filter(|t| !t.is_empty()) {
        if line.is_empty() {
            line = fit_cells(token, cols);
        } else if cells(&line) + GAP.len() + cells(token) <= cols {
            line.push_str(GAP);
            line.push_str(token);
        } else {
            lines.push(std::mem::take(&mut line));
            line = fit_cells(token, cols);
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The header: the column legend aligned to `l`, then the keys and the match words wrapped to
/// the layout width — two lines on wide terminals, more on narrow ones.
pub(crate) fn header_lines(l: &RowLayout, view: &str) -> String {
    let files = view == "files";
    let mut parts: Vec<String> = vec![
        pad_cells("type", l.kind_w),
        pad_cells(if files { "title" } else { "project" }, l.title_w),
    ];
    if l.where_w > 0 {
        parts.push(pad_cells(if files { "where" } else { "about" }, l.where_w));
    }
    if l.when_w > 0 {
        parts.push(format!("{:>w$}", "when", w = l.when_w));
    }
    parts.push("match".to_string());
    let legend = parts.join(GAP);
    let keys = "Enter=select  Tab=toggle dir/file  Ctrl-D=dirs  Ctrl-F=files  Ctrl-U=clear  |  match: words=exact terms  meaning=semantic  both  path  weak  ·  seed / project / linked = graph relation (project = same project as a seed)";
    let mut lines = vec![legend];
    lines.extend(wrap_gap_tokens(keys, l.cols));
    lines.join("\n")
}

/// The preview's plain-English account of the score, from the same signals `why` encodes.
pub(crate) fn why_phrases(
    semantic: f64,
    lexical: f64,
    relation: &str,
    freshness_tier: &str,
    quality: f64,
    frecency: f64,
    why: &str,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |s: &str| {
        if !out.iter().any(|x| x == s) {
            out.push(s.to_string());
        }
    };
    if lexical >= 0.7 {
        push("exact words");
    } else if lexical >= 0.3 {
        push("some words");
    }
    if semantic >= 0.7 {
        push("strong meaning match");
    } else if semantic >= 0.4 {
        push("related meaning");
    }
    match relation {
        "seed" => push("seed result"),
        "same_project" => push("same project as a seed"),
        "related_project" => push("linked project"),
        "lexical" => push("words only"),
        "path_keyword" => push("path matches"),
        _ => {}
    }
    for part in why.split('+') {
        match part {
            "path" => push("path matches"),
            "role" => push("role fits the question"),
            "path-penalty" => push("scratch or copy directory"),
            "summary-page" => push("summary page"),
            "noise" => push("machine artefact"),
            "superseded" => push("superseded"),
            _ => {}
        }
    }
    match freshness_tier {
        "fresh" | "aging" | "stale" => push(freshness_tier),
        _ => {}
    }
    if frecency > 0.0 {
        push("you picked this recently");
    }
    if quality > 0.0 && quality < 0.6 {
        push("low-quality text");
    }
    if out.is_empty() {
        out.push("weak match".to_string());
    }
    out
}

/// Whole-word, ASCII-case-insensitive occurrence of `needle` (already lowercase ASCII) in
/// `hay` (already lowercase ASCII), as byte offsets.
fn find_word(hay: &str, needle: &str) -> Option<(usize, usize)> {
    let bytes = hay.as_bytes();
    let mut from = 0usize;
    while from <= hay.len() {
        let i = hay[from..].find(needle)?;
        let s = from + i;
        let e = s + needle.len();
        let before_ok = s == 0 || !bytes[s - 1].is_ascii_alphanumeric();
        let after_ok = e >= bytes.len() || !bytes[e].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some((s, e));
        }
        from = s + 1;
        while from < hay.len() && !hay.is_char_boundary(from) {
            from += 1;
        }
    }
    None
}

fn mark_terms(window: &str, terms: &[String]) -> Vec<(String, bool)> {
    let lower = window.to_ascii_lowercase();
    let mut marks: Vec<(usize, usize)> = Vec::new();
    for t in terms {
        let mut from = 0usize;
        while from < lower.len() {
            let Some((s, e)) = find_word(&lower[from..], t) else {
                break;
            };
            marks.push((from + s, from + e));
            from += e;
        }
    }
    marks.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in marks {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    for (s, e) in merged {
        if s > pos {
            out.push((window[pos..s].to_string(), false));
        }
        out.push((window[s..e].to_string(), true));
        pos = e;
    }
    if pos < window.len() {
        out.push((window[pos..].to_string(), false));
    }
    out
}

/// A query-centred excerpt as `(text, is_match)` segments: `radius` characters either side of
/// the first whole-word hit of a query term, snapped to word boundaries, with `…` where text
/// was cut; the start of the text when no term occurs.
pub(crate) fn snippet_around(text: &str, query: &str, radius: usize) -> Vec<(String, bool)> {
    let flat = collapse_whitespace(text);
    if flat.is_empty() {
        return Vec::new();
    }
    let terms: Vec<String> = word_tokens(query)
        .into_iter()
        .filter(|t| t.len() >= 2)
        .collect();
    let lower = flat.to_ascii_lowercase();
    let hit = terms
        .iter()
        .filter_map(|t| find_word(&lower, t))
        .min_by_key(|(s, _)| *s);
    let (mut ws, mut we) = match hit {
        Some((s, e)) => (s.saturating_sub(radius), (e + radius).min(flat.len())),
        None => (0, (2 * radius).min(flat.len())),
    };
    while ws > 0 && !flat.is_char_boundary(ws) {
        ws -= 1;
    }
    while we < flat.len() && !flat.is_char_boundary(we) {
        we += 1;
    }
    if ws > 0 {
        if let Some(sp) = flat[ws..we].find(' ') {
            if sp < 24 {
                ws += sp + 1;
            }
        }
    }
    if we < flat.len() {
        if let Some(sp) = flat[ws..we].rfind(' ') {
            if we - (ws + sp) < 24 {
                we = ws + sp;
            }
        }
    }
    let mut segs = mark_terms(&flat[ws..we], &terms);
    if ws > 0 {
        segs.insert(0, ("…".to_string(), false));
    }
    if we < flat.len() {
        segs.push(("…".to_string(), false));
    }
    segs
}

/// A document named in a preview: its title, path relative to the project, and score.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RelatedDoc {
    pub(crate) title: String,
    pub(crate) rel_path: String,
    pub(crate) score: f64,
}

pub(crate) struct FilePreview<'a> {
    pub(crate) title: &'a str,
    pub(crate) kind: &'a str,
    pub(crate) role: &'a str,
    pub(crate) date_ymd: &'a str,
    pub(crate) age_days: f64,
    pub(crate) project: &'a str,
    pub(crate) rel_path: &'a str,
    pub(crate) why: &'a [String],
    pub(crate) raw_signals: &'a str,
    pub(crate) snippet: &'a [(String, bool)],
    pub(crate) chunk_index: i64,
    pub(crate) chunk_total: i64,
    pub(crate) related: Option<&'a RelatedDoc>,
}

/// One fzf field from multi-line text: backslashes doubled, carriage returns dropped, tabs
/// (the field delimiter) widened to spaces and newlines written as the two characters `\` `n`,
/// which `printf '%b'` turns back into line breaks.
pub(crate) fn pick_preview_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('\r', "")
        .replace('\t', "    ")
        .replace('\n', "\\n")
}

/// Content for the preview field: sanitised, then escaped exactly once.
fn pv(text: &str) -> String {
    pick_preview_escape(&sanitize_display(text))
}

const BOLD: &str = "\\033[1m";
const DIM: &str = "\\033[2m";
const HIT: &str = "\\033[1;4m";
const RESET: &str = "\\033[0m";

fn snippet_line(snippet: &[(String, bool)]) -> String {
    let mut out = String::new();
    for (text, hit) in snippet {
        if *hit {
            out.push_str(&format!("{}{}{}", HIT, pv(text), RESET));
        } else {
            out.push_str(&pv(text));
        }
    }
    out
}

fn why_line(why: &[String], raw_signals: &str) -> String {
    let words = why.iter().map(|w| pv(w)).collect::<Vec<_>>().join(" · ");
    if raw_signals.is_empty() {
        format!("{}why:{} {}", DIM, RESET, words)
    } else {
        format!(
            "{}why:{} {}  {}({}){}",
            DIM,
            RESET,
            words,
            DIM,
            pv(raw_signals),
            RESET
        )
    }
}

/// The file card: title; kind · role · date (age) · project · path; why; the query-centred
/// snippet with `(chunk i of n)`; the related document by title. Lines are joined with the
/// two characters `\` `n` and colours written as `\033[..m`, for `printf '%b'`.
pub(crate) fn render_file_preview(p: &FilePreview<'_>) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(5);
    lines.push(format!("{}{}{}", BOLD, pv(p.title), RESET));
    lines.push(format!(
        "{}{} · {} · {} ({}) · {} · {}{}",
        DIM,
        pv(p.kind),
        pv(p.role),
        pv(p.date_ymd),
        age_label(p.age_days),
        pv(p.project),
        pv(p.rel_path),
        RESET
    ));
    lines.push(why_line(p.why, p.raw_signals));
    lines.push(format!(
        "{}  {}(chunk {} of {}){}",
        snippet_line(p.snippet),
        DIM,
        p.chunk_index + 1,
        p.chunk_total.max(1),
        RESET
    ));
    if let Some(r) = p.related {
        lines.push(format!(
            "{}related:{} {} {}— {} ({:.2}){}",
            DIM,
            RESET,
            pv(&r.title),
            DIM,
            pv(&r.rel_path),
            r.score,
            RESET
        ));
    }
    lines.join("\\n")
}

pub(crate) struct DirPreview<'a> {
    pub(crate) name: &'a str,
    pub(crate) synopsis: &'a str,
    pub(crate) file_count: i64,
    pub(crate) age_days: f64,
    pub(crate) why: &'a [String],
    pub(crate) raw_signals: &'a str,
    pub(crate) evidence: &'a [RelatedDoc],
    pub(crate) snippet: &'a [(String, bool)],
}

/// The project card: name; synopsis (when any); `N files · age`; why; up to three evidence
/// documents by title; the best chunk's snippet.
pub(crate) fn render_dir_preview(p: &DirPreview<'_>) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(7);
    lines.push(format!("{}{}{}", BOLD, pv(p.name), RESET));
    if !p.synopsis.is_empty() {
        lines.push(pv(p.synopsis));
    }
    lines.push(format!(
        "{}{} files · newest activity {}{}",
        DIM,
        p.file_count,
        age_label(p.age_days),
        RESET
    ));
    lines.push(why_line(p.why, p.raw_signals));
    for (i, ev) in p.evidence.iter().take(3).enumerate() {
        lines.push(format!(
            "{}{}{} {} {}— {} ({:.2}){}",
            DIM,
            if i == 0 { "top:" } else { "    " },
            RESET,
            pv(&ev.title),
            DIM,
            pv(&ev.rel_path),
            ev.score,
            RESET
        ));
    }
    if !p.snippet.is_empty() {
        lines.push(snippet_line(p.snippet));
    }
    lines.join("\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_removes_terminal_controls_and_invisibles() {
        let dirty = "ok\x1b[31mred\x1b[0m\x1b]8;;http://x\x07link\x1b]8;;\x07\u{202E}rtl\u{200B}\ttab\nnl\u{0085}";
        let clean = sanitize_display(dirty);
        assert!(!clean.contains('\x1b'));
        assert!(!clean.contains('\u{202E}'));
        assert!(!clean.contains('\u{200B}'));
        assert!(!clean.contains('\t') && !clean.contains('\n'));
        assert!(clean.contains("ok") && clean.contains("red") && clean.contains("rtl"));
        // The OSC payloads, the SGR sequences, U+202E, U+200B and U+0085 are all gone; tab
        // and newline became spaces.
        assert_eq!(clean, "okredlinkrtl tab nl");
        // Invisible formatting characters beyond the zero-width space block: word joiner
        // (U+2060), soft hyphen (U+00AD), a tag character (U+E0041), Mongolian vowel
        // separator (U+180E) and invisible plus (U+2064).
        let invisible = "a\u{2060}b\u{00AD}c\u{E0041}d\u{180E}e\u{2064}f";
        assert_eq!(sanitize_display(invisible), "abcdef");
    }

    #[test]
    fn esc_string_families_are_swallowed() {
        assert_eq!(strip_esc_strings("a\x1bPpayload\x1b\\b"), "ab");
        assert_eq!(strip_esc_strings("a\x1b_x\x07b"), "ab");
        assert_eq!(strip_esc_strings("a\x1b^pm\x1b\\b"), "ab");
        assert_eq!(strip_esc_strings("a\x1bXsos\x07b"), "ab");
        assert_eq!(strip_esc_strings("a\x1b]8;;http://x\x07b"), "ab");
        // Not a string opener: left in place for the CSI stripper.
        assert_eq!(strip_esc_strings("a\x1b[1mb"), "a\x1b[1mb");
        // A bare ESC ends the string and opens the next sequence, which the CSI stripper
        // then removes; an unterminated string swallows the rest.
        assert_eq!(sanitize_display("a\x1b]0;title\x1b[31mred\x1b[0m"), "ared");
        assert_eq!(sanitize_display("a\x1bPnever closed"), "a");
    }

    #[test]
    fn sanitize_keeps_emoji_sequences_but_drops_invisibles() {
        // ZWJ and the emoji variation selectors are the deliberate exception: removing them
        // would change a sequence's width.
        assert_eq!(sanitize_display("👨\u{200D}💻"), "👨\u{200D}💻");
        assert_eq!(
            sanitize_display("⚠\u{FE0F} ok \u{FE0E}"),
            "⚠\u{FE0F} ok \u{FE0E}"
        );
        assert_eq!(
            sanitize_display("a\u{200B}b\u{200C}c\u{200E}d\u{200F}e"),
            "abcde"
        );
        assert_eq!(
            sanitize_display("a\u{206A}b\u{206F}c\u{FFF9}d\u{FFFA}e\u{FFFB}f"),
            "abcdef"
        );
        // Line and paragraph separators are line breaks: a space, like `\n`.
        assert_eq!(sanitize_display("a\u{2028}b\u{2029}c"), "a b c");
    }

    #[test]
    fn cell_helpers_use_display_width() {
        assert_eq!(cells("日本"), 4);
        assert_eq!(fit_cells("abcdef", 4), "abc…");
        assert_eq!(fit_cells("abc", 4), "abc");
        assert_eq!(fit_cells("日本語", 4), "日…");
        assert_eq!(pad_cells("ab", 4), "ab  ");
        assert_eq!(cells(&pad_cells("日本語テキスト", 6)), 6);
    }

    #[test]
    fn cell_helpers_hold_width_on_emoji_presentation_sequences() {
        // "⚠️" is U+26A0 U+FE0F: the string measures 2 cells while its chars measure 1 + 0,
        // so a per-character accumulation under-counts the cut.
        let title = "⚠️ Warning: planning suite design review";
        assert_eq!(cells("⚠️"), 2);
        for width in [6, 10, 12] {
            let padded = pad_cells(title, width);
            assert_eq!(cells(&padded), width, "{:?}", padded);
            let fitted = fit_cells(title, width);
            assert!(
                cells(&fitted) <= width,
                "{:?} = {} cells",
                fitted,
                cells(&fitted)
            );
        }
        for symbol in ["☁️", "❤️", "✔️", "ℹ️", "⚙️"] {
            let text = format!("{} planning suite design review", symbol);
            for width in [3, 6, 10] {
                assert_eq!(cells(&pad_cells(&text, width)), width, "{:?}", text);
            }
        }
        // The stem-elision branch of elide_middle holds the width too, with the sequence at
        // the head and in the tail of the stem.
        let path = "⚠️-warning-planning-suite-design-review-⚠️-final.md";
        for max in 8..30 {
            let e = elide_middle(path, max);
            assert!(
                cells(&e) <= max,
                "max={} -> {:?} ({} cells)",
                max,
                e,
                cells(&e)
            );
        }
    }

    #[test]
    fn elide_middle_keeps_the_basename() {
        let p = "202606-navigating-the-semantic-jungle/customer-signals/Acme/20260918-planning-suite-design-review.md";
        let e = elide_middle(p, 60);
        assert!(cells(&e) <= 60, "{}", e);
        assert!(
            e.ends_with("20260918-planning-suite-design-review.md"),
            "{}",
            e
        );
        assert!(
            e.starts_with("202606-nav"),
            "the project survives, cut: {}",
            e
        );
        assert!(e.contains('…'));
        assert_eq!(elide_middle("short/a.md", 60), "short/a.md");
        let tight = elide_middle(p, 30);
        assert!(cells(&tight) <= 30, "{}", tight);
        assert!(tight.ends_with(".md"), "{}", tight);
    }

    #[test]
    fn age_labels() {
        assert_eq!(age_label(0.4), "today");
        assert_eq!(age_label(10.2), "10d");
        assert_eq!(age_label(44.9), "44d");
        assert_eq!(age_label(61.0), "2mo");
        assert_eq!(age_label(800.0), "2y");
        assert_eq!(age_label(f64::NAN), "");
    }

    #[test]
    fn match_labels_are_words_not_numbers() {
        assert_eq!(match_label(0.9, 0.9, "seed", ""), "both · seed");
        assert_eq!(match_label(0.2, 1.0, "same_project", ""), "words · project");
        assert_eq!(
            match_label(0.8, 0.1, "related_project", ""),
            "meaning · linked"
        );
        assert_eq!(match_label(0.1, 0.0, "direct", "path"), "path");
        assert_eq!(match_label(0.3, 0.2, "direct", ""), "weak");
        // The widest combination: the longest `what` with the longest relation word.
        assert_eq!(
            match_label(0.9, 0.2, "same_project", ""),
            "meaning · project"
        );
        // Every `what` (both, words, meaning, path, weak) with every relation stays within
        // the 18-cell column.
        let whats: [(f64, f64, &str); 5] = [
            (0.9, 0.9, ""),
            (0.2, 1.0, ""),
            (0.8, 0.1, ""),
            (0.1, 0.0, "path"),
            (0.3, 0.2, ""),
        ];
        let relations = [
            "seed",
            "same_project",
            "related_project",
            "direct",
            "lexical",
            "path_keyword",
        ];
        for (semantic, lexical, why) in whats {
            for relation in relations {
                let s = match_label(semantic, lexical, relation, why);
                assert!(cells(&s) <= 18, "{} ({} cells)", s, cells(&s));
            }
        }
    }

    #[test]
    fn why_phrases_read_as_english() {
        let w = why_phrases(
            0.9,
            1.0,
            "seed",
            "fresh",
            1.0,
            0.0,
            "semantic:0.61+lexical:1.00+graph:seed+recency:fresh",
        );
        assert_eq!(
            w,
            vec![
                "exact words",
                "strong meaning match",
                "seed result",
                "fresh"
            ]
        );
        let w = why_phrases(
            0.5,
            0.0,
            "same_project",
            "stale",
            0.4,
            0.3,
            "cosine:0.41+path-penalty",
        );
        assert_eq!(
            w,
            vec![
                "related meaning",
                "same project as a seed",
                "scratch or copy directory",
                "stale",
                "you picked this recently",
                "low-quality text"
            ]
        );
        assert_eq!(
            why_phrases(0.0, 0.0, "direct", "", 1.0, 0.0, ""),
            vec!["weak match"]
        );
    }

    #[test]
    fn snippet_centres_on_the_first_hit_and_marks_terms() {
        let text = format!("{} the Semantic Layer sits between Planning Suite and the warehouse; the Acme semantic layer maps it. {}", "filler ".repeat(60), "tail ".repeat(60));
        let segs = snippet_around(&text, "acme semantic", 40);
        let joined: String = segs.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            joined.starts_with('…') && joined.ends_with('…'),
            "{}",
            joined
        );
        assert!(joined.contains("Semantic Layer"), "{}", joined);
        assert!(segs
            .iter()
            .any(|(t, hit)| *hit && t.eq_ignore_ascii_case("semantic")));
        assert!(cells(&joined) <= 2 * 40 + 40, "{}", joined);
        let none = snippet_around("plain text without the terms", "zzz", 10);
        let none_joined: String = none.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(none_joined, "plain text without…");
        assert!(none.iter().all(|(_, hit)| !hit));
    }

    #[test]
    fn snippet_matches_whole_words_only() {
        let segs = snippet_around("cat concatenate cat", "cat", 50);
        let hits: Vec<&str> = segs
            .iter()
            .filter(|(_, h)| *h)
            .map(|(t, _)| t.as_str())
            .collect();
        assert_eq!(hits, vec!["cat", "cat"]);
    }

    #[test]
    fn layouts_by_width() {
        let l = layout_for_width(80);
        assert_eq!((l.where_w, l.when_w), (0, 0));
        assert_eq!(l.kind_w + GAP.len() + l.title_w + GAP.len() + l.match_w, 80);
        let l = layout_for_width(100);
        assert!(l.where_w > 0 && l.when_w == 0);
        assert_eq!(
            l.kind_w + l.title_w + l.where_w + l.match_w + 3 * GAP.len(),
            100
        );
        let l = layout_for_width(160);
        assert!(l.where_w > 0 && l.when_w == 5);
        assert_eq!(
            l.kind_w + l.title_w + l.where_w + l.when_w + l.match_w + 4 * GAP.len(),
            160
        );
        assert!(l.title_w >= l.where_w * 3 / 4, "title gets ~45%: {:?}", l);
        assert_eq!(layout_for_width(10).cols, 60, "clamped");
    }

    #[test]
    fn file_row_is_exactly_the_layout_width_without_colour() {
        for cols in [80usize, 100, 120, 160, 220] {
            let l = layout_for_width(cols);
            let row = render_file_row(
                &FileRow {
                    kind: "md",
                    title: "Acme Planning Suite \"Semantic Layer\" design review — a long title that must be cut",
                    project: "202606-navigating-the-semantic-jungle",
                    rel_path: "customer-signals/Acme/20260918-planning-suite-design-review.md",
                    age_days: 10.0,
                    match_label: "both · seed",
                },
                &l,
                false,
            );
            assert_eq!(cells(&row), cols, "cols={} row={:?}", cols, row);
            assert!(row.starts_with("md     "), "{}", row);
            if l.where_w > 0 {
                assert!(row.contains("design-review.md"), "{}", row);
            }
            if l.when_w > 0 {
                assert!(row.contains(" 10d"), "{}", row);
            }
        }
        // An absurd age cannot widen the `when` column past its 5 cells.
        let l = layout_for_width(160);
        let row = render_file_row(
            &FileRow {
                kind: "md",
                title: "T",
                project: "p",
                rel_path: "a.md",
                age_days: f64::MAX,
                match_label: "weak",
            },
            &l,
            false,
        );
        assert_eq!(cells(&row), 160, "{:?}", row);
    }

    #[test]
    fn coloured_rows_carry_sgr_but_same_visible_width() {
        let l = layout_for_width(120);
        let plain = render_file_row(
            &FileRow {
                kind: "email",
                title: "T",
                project: "p",
                rel_path: "a.md",
                age_days: 1.0,
                match_label: "words",
            },
            &l,
            false,
        );
        let coloured = render_file_row(
            &FileRow {
                kind: "email",
                title: "T",
                project: "p",
                rel_path: "a.md",
                age_days: 1.0,
                match_label: "words",
            },
            &l,
            true,
        );
        assert!(coloured.contains("\x1b[") && !plain.contains("\x1b["));
        assert_eq!(cells(&sanitize_display(&coloured)), cells(&plain));
    }

    #[test]
    fn row_text_is_sanitised() {
        let l = layout_for_width(120);
        let row = render_file_row(
            &FileRow {
                kind: "md",
                title: "bad\x1b]8;;http://x\x07title\x1b]8;;\x07",
                project: "p",
                rel_path: "a\tb.md",
                age_days: 1.0,
                match_label: "",
            },
            &l,
            true,
        );
        assert!(!row.contains("]8;;"), "{}", row);
        assert!(!row.contains('\t'), "{}", row);
    }

    #[test]
    fn dir_row_and_header_align() {
        let l = layout_for_width(140);
        let row = render_dir_row(
            &DirRow {
                name: "202606-navigating-the-semantic-jungle",
                about: "Acme semantic-layer work: BT/AWS context POC, tracker overhaul",
                age_days: 2.0,
                match_label: "both · seed",
            },
            &l,
            false,
        );
        assert_eq!(cells(&row), 140, "{}", row);
        assert!(row.starts_with("dir    "), "{}", row);
        let header = header_lines(&l, "files");
        let first = header.lines().next().unwrap();
        assert!(first.starts_with("type   "), "{}", first);
        assert!(
            first.contains("title")
                && first.contains("where")
                && first.contains("when")
                && first.ends_with("match")
        );
        for line in header.lines() {
            assert!(cells(line) <= l.cols, "{} cells: {}", cells(line), line);
        }
        assert!(header.lines().nth(1).unwrap().contains("Tab=toggle"));
        assert!(
            header
                .lines()
                .any(|line| line.contains("project = same project as a seed")),
            "{}",
            header
        );
        assert!(header_lines(&l, "projects")
            .lines()
            .next()
            .unwrap()
            .contains("project"));
        let narrow = layout_for_width(80);
        let header = header_lines(&narrow, "files");
        assert!(header.lines().count() >= 3, "{}", header);
        for line in header.lines() {
            assert!(cells(line) <= 80, "{} cells: {}", cells(line), line);
        }
        assert!(
            header
                .lines()
                .any(|line| line.contains("project = same project as a seed")),
            "{}",
            header
        );
    }
    #[test]
    fn file_preview_is_one_escaped_line_with_five_rows() {
        let why = vec!["exact words".to_string(), "seed result".to_string()];
        let snippet = vec![
            ("…the ".to_string(), false),
            ("acme".to_string(), true),
            (" layer…".to_string(), false),
        ];
        let related = RelatedDoc {
            title: "AWS Context POC use case thread".to_string(),
            rel_path: "customer-signals/Acme/04-thread.md".to_string(),
            score: 0.852,
        };
        let p = FilePreview {
            title: "Acme \"Semantic Layer\" review 100% done\\ok",
            kind: "md",
            role: "record",
            date_ymd: "2026-09-18",
            age_days: 10.0,
            project: "202606-navigating-the-semantic-jungle",
            rel_path: "customer-signals/Acme/20260918-review.md",
            why: &why,
            raw_signals: "s .53  l 1.0  g 1.0  q 1.0",
            snippet: &snippet,
            chunk_index: 2,
            chunk_total: 7,
            related: Some(&related),
        };
        let out = render_file_preview(&p);
        assert!(!out.contains('\n') && !out.contains('\t'), "{}", out);
        assert_eq!(out.matches("\\n").count(), 4, "{}", out);
        assert!(out.contains("\\033[1mAcme"), "{}", out);
        assert!(
            out.contains("100% done\\\\ok"),
            "backslash in content is doubled once: {}",
            out
        );
        assert!(out.contains("\\033[1;4macme\\033[0m"), "{}", out);
        assert!(out.contains("(chunk 3 of 7)"), "{}", out);
        assert!(
            out.contains("related:") && out.contains("(0.85)"),
            "{}",
            out
        );
        assert!(out.contains("2026-09-18 (10d)"), "{}", out);
    }

    #[test]
    fn preview_content_cannot_inject_escapes() {
        let why: Vec<String> = Vec::new();
        let snippet: Vec<(String, bool)> = vec![("\x1b[31mred\x1b[0m text".to_string(), false)];
        let p = FilePreview {
            title: "t\x1b]8;;http://x\x07",
            kind: "md",
            role: "state",
            date_ymd: "",
            age_days: 0.0,
            project: "p",
            rel_path: "r",
            why: &why,
            raw_signals: "",
            snippet: &snippet,
            chunk_index: 0,
            chunk_total: 1,
            related: None,
        };
        let out = render_file_preview(&p);
        assert!(!out.contains('\x1b'), "{}", out);
        assert!(!out.contains("]8;;"), "{}", out);
        assert_eq!(out.matches("\\n").count(), 3, "no related line: {}", out);
    }

    #[test]
    fn preview_field_renders_through_printf_b() {
        // The real consumer: fzf runs `printf '%b' {3}` in the user's shell.
        let why = vec!["exact words".to_string()];
        let snippet = vec![("a ".to_string(), false), ("hit".to_string(), true)];
        let p = FilePreview {
            title: "T 100% \\ok",
            kind: "md",
            role: "record",
            date_ymd: "2026-09-18",
            age_days: 1.0,
            project: "p",
            rel_path: "r.md",
            why: &why,
            raw_signals: "s 1.00",
            snippet: &snippet,
            chunk_index: 0,
            chunk_total: 2,
            related: None,
        };
        let field = render_file_preview(&p);
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg("printf '%b' \"$1\"")
            .arg("_")
            .arg(&field)
            .output()
            .expect("sh available");
        let rendered = String::from_utf8_lossy(&out.stdout);
        assert_eq!(rendered.lines().count(), 4, "{:?}", rendered);
        assert!(
            rendered.contains("\x1b[1mT 100% \\ok\x1b[0m"),
            "{:?}",
            rendered
        );
        assert!(rendered.contains("\x1b[1;4mhit\x1b[0m"), "{:?}", rendered);
    }

    #[test]
    fn dir_preview_lists_evidence_titles() {
        let why = vec!["strong meaning match".to_string()];
        let ev = vec![
            RelatedDoc {
                title: "AWS Context POC thread".to_string(),
                rel_path: "a.md".to_string(),
                score: 0.8,
            },
            RelatedDoc {
                title: "Tracker overhaul plan".to_string(),
                rel_path: "b.md".to_string(),
                score: 0.7,
            },
        ];
        let snippet: Vec<(String, bool)> = vec![("hello".to_string(), false)];
        let p = DirPreview {
            name: "202606-navigating-the-semantic-jungle",
            synopsis: "",
            file_count: 346,
            age_days: 2.0,
            why: &why,
            raw_signals: "s 1.0  l 1.0",
            evidence: &ev,
            snippet: &snippet,
        };
        let out = render_dir_preview(&p);
        assert!(
            out.starts_with("\\033[1m202606-navigating-the-semantic-jungle\\033[0m"),
            "{}",
            out
        );
        assert!(out.contains("346 files"), "{}", out);
        assert!(
            out.contains("AWS Context POC thread") && out.contains("Tracker overhaul plan"),
            "{}",
            out
        );
        assert!(out.contains("why:"), "{}", out);
    }
}
