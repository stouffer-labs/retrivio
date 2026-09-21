//! Freshness model: content-date resolution, recency score and tier labels.
//! See docs/superpowers/specs/2026-09-19-proactive-recall-design.md §4; document roles
//! (state, knowledge, record) live in `roles.rs` and decide how a date is chosen here.
//!
//! Everything here is pure arithmetic on paths, timestamps and small strings: no file I/O,
//! no database access, no `chrono`. Callers supply `now` so results are reproducible in tests.

use crate::roles::Role;

/// Seconds in one UTC day.
pub const DAY_SECS: f64 = 86_400.0;

/// Candidate dates more than this far in the future are treated as implausible and ignored.
pub const FUTURE_SLACK_DAYS: f64 = 2.0;

/// Age boundary (days, exclusive) below which a living document is `fresh`.
pub const FRESH_MAX_DAYS: f64 = 14.0;

/// Age boundary (days, inclusive) up to which a living document is `aging`; older is `stale`.
pub const AGING_MAX_DAYS: f64 = 35.0;

/// Earliest year accepted in a path date; anything earlier is treated as a serial number.
const MIN_YEAR: i64 = 1980;
/// Latest year accepted in a path date.
const MAX_YEAR: i64 = 2100;

/// Days since 1970-01-01 for a proleptic Gregorian civil date.
///
/// Howard Hinnant's `days_from_civil` algorithm; valid for the whole `i64` range we care about.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let m = month as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`]: `(year, month, day)` for a day count since 1970-01-01.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Midnight UTC of a validated civil date as a unix timestamp, or `None` if out of range.
fn civil_ts(year: i64, month: u32, day: u32) -> Option<f64> {
    if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
        return None;
    }
    if !(1..=12).contains(&month) {
        return None;
    }
    if day < 1 || day > days_in_month(year, month) {
        return None;
    }
    Some(days_from_civil(year, month, day) as f64 * DAY_SECS)
}

fn digits_value(s: &[u8]) -> Option<i64> {
    if s.is_empty() || !s.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut v: i64 = 0;
    for b in s {
        v = v * 10 + (b - b'0') as i64;
    }
    Some(v)
}

fn is_delim_or_end(bytes: &[u8], idx: usize) -> bool {
    match bytes.get(idx) {
        None => true,
        Some(b) => matches!(*b, b'-' | b'_' | b'.'),
    }
}

/// Parse a date prefix from one path component.
///
/// Returns `(timestamp, precision)` where precision is `8` for `YYYYMMDD` / `YYYY-MM-DD`
/// and `6` for `YYYYMM` (resolved to the first of the month). The digits must start the
/// component and be followed by `-`, `_`, `.` or the end of the component.
fn component_date(comp: &str) -> Option<(f64, u8)> {
    let b = comp.as_bytes();
    // YYYYMMDD
    if b.len() >= 8 && is_delim_or_end(b, 8) {
        if let Some(v) = digits_value(&b[..8]) {
            let (y, m, d) = (v / 10_000, ((v / 100) % 100) as u32, (v % 100) as u32);
            if let Some(ts) = civil_ts(y, m, d) {
                return Some((ts, 8));
            }
        }
    }
    // YYYY-MM-DD (common for dated notes and specs)
    if b.len() >= 10 && b[4] == b'-' && b[7] == b'-' && is_delim_or_end(b, 10) {
        if let (Some(y), Some(m), Some(d)) = (
            digits_value(&b[..4]),
            digits_value(&b[5..7]),
            digits_value(&b[8..10]),
        ) {
            if let Some(ts) = civil_ts(y, m as u32, d as u32) {
                return Some((ts, 8));
            }
        }
    }
    // YYYYMM
    if b.len() >= 6 && is_delim_or_end(b, 6) {
        if let Some(v) = digits_value(&b[..6]) {
            let (y, m) = (v / 100, (v % 100) as u32);
            if let Some(ts) = civil_ts(y, m, 1) {
                return Some((ts, 6));
            }
        }
    }
    None
}

/// Date carried by the path itself, as a unix timestamp (midnight UTC), if any component
/// starts with `YYYYMMDD`, `YYYY-MM-DD` or `YYYYMM` followed by `-`, `_`, `.` or the end.
///
/// The most specific component wins (a day-precision component beats a month-precision
/// parent); among equally specific components the deepest wins.
pub fn path_date(path: &str) -> Option<f64> {
    let mut best: Option<(f64, u8)> = None;
    for comp in path.split(['/', '\\']) {
        if comp.is_empty() {
            continue;
        }
        if let Some((ts, precision)) = component_date(comp) {
            let better = match best {
                None => true,
                Some((_, best_precision)) => precision >= best_precision,
            };
            if better {
                best = Some((ts, precision));
            }
        }
    }
    best.map(|(ts, _)| ts)
}

/// Content date used for ranking: the newer of the path date and `doc_mtime`.
///
/// Returns `(timestamp, source)` with source `"path-date"` or `"mtime"`. A candidate more than
/// [`FUTURE_SLACK_DAYS`] ahead of `now` is implausible and is skipped; if both candidates are
/// implausible the result falls back to `(doc_mtime, "mtime")`. This is the rule for `state`
/// and `knowledge` documents; see [`content_date_for_role`] for records.
pub fn content_date(path: &str, doc_mtime: f64, now: f64) -> (f64, &'static str) {
    let horizon = now + FUTURE_SLACK_DAYS * DAY_SECS;
    let mtime_ok = doc_mtime.is_finite() && doc_mtime <= horizon;
    let path_ok = path_date(path).filter(|ts| *ts <= horizon);
    match (path_ok, mtime_ok) {
        (Some(pd), true) if pd > doc_mtime => (pd, "path-date"),
        (Some(_), true) => (doc_mtime, "mtime"),
        (Some(pd), false) => (pd, "path-date"),
        (None, true) => (doc_mtime, "mtime"),
        // Every candidate is implausibly far in the future (clock skew, bad archive
        // timestamps): treat the document as current rather than trusting a future date.
        (None, false) => (now, "mtime"),
    }
}

/// Like [`path_date`], but the date may sit anywhere in a component at a token boundary
/// (`HANDOFF-2026-08-28-orion.md`, `notes_20260828.md`), not only at its start. The most
/// specific date wins, then the deepest component, then the first in the component.
pub fn embedded_path_date(path: &str) -> Option<f64> {
    let mut best: Option<(f64, u8)> = None;
    for comp in path.split(['/', '\\']) {
        let bytes = comp.as_bytes();
        let mut comp_best: Option<(f64, u8)> = None;
        for i in 0..bytes.len() {
            if !bytes[i].is_ascii_digit() {
                continue;
            }
            if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                continue;
            }
            if let Some((ts, precision)) = component_date(&comp[i..]) {
                let better = match comp_best {
                    None => true,
                    Some((_, p)) => precision > p,
                };
                if better {
                    comp_best = Some((ts, precision));
                }
            }
        }
        if let Some((ts, precision)) = comp_best {
            let better = match best {
                None => true,
                Some((_, best_precision)) => precision >= best_precision,
            };
            if better {
                best = Some((ts, precision));
            }
        }
    }
    best.map(|(ts, _)| ts)
}

/// Date that orders the revisions of one series (supersession): a plausible date anywhere in
/// the path relative to the project ([`embedded_path_date`]: file name or directory, most
/// specific wins), else the last edit, else `now`. Unlike [`content_date`] a fresh mtime never
/// beats the date in the file name, so editing an old handoff does not make it the newest of
/// its series. Pass the path relative to the project: a dated project folder must not date
/// every file in it.
pub fn revision_date(doc_rel_path: &str, doc_mtime: f64, now: f64) -> f64 {
    let horizon = now + FUTURE_SLACK_DAYS * DAY_SECS;
    if let Some(pd) = embedded_path_date(doc_rel_path).filter(|ts| *ts <= horizon) {
        return pd;
    }
    if doc_mtime.is_finite() && doc_mtime <= horizon {
        doc_mtime
    } else {
        now
    }
}

/// Content date by role. Records are events: a plausible path date is *the* event date even
/// when the file was edited later (a transcript touched up in September still happened in
/// July); without a path date the mtime stands in until recall reads the front matter. State
/// and knowledge use [`content_date`], the newer of path date and last edit.
pub fn content_date_for_role(
    path: &str,
    doc_mtime: f64,
    now: f64,
    role: Role,
) -> (f64, &'static str) {
    if role != Role::Record {
        return content_date(path, doc_mtime, now);
    }
    let horizon = now + FUTURE_SLACK_DAYS * DAY_SECS;
    if let Some(pd) = path_date(path).filter(|ts| *ts <= horizon) {
        return (pd, "path-date");
    }
    if doc_mtime.is_finite() && doc_mtime <= horizon {
        (doc_mtime, "mtime")
    } else {
        (now, "mtime")
    }
}

/// Maximum lines of a leading YAML front matter block (opening and closing `---` included).
const FRONTMATTER_MAX_LINES: usize = 60;
/// Plain `Date:` / `Updated:` lines are only honoured this early in the document.
const PLAIN_DATE_MAX_LINES: usize = 10;

/// Date declared at the top of a document, as midnight UTC. Two shapes are accepted:
///
/// (a) a leading YAML block: first line `---`, closing `---` within 60 lines; inside it
///     `updated`, `last_updated`, `last-updated` and `modified` win over `date`;
/// (b) otherwise a `Date:`, `Updated:` or `Last updated:` line (case-insensitive, markdown
///     decoration such as `**` allowed) among the first 10 non-empty lines outside code fences.
///
/// Values are `YYYY-MM-DD`, optionally followed by a time part that is ignored. Anything else,
/// in particular dates deeper in the body, is rejected. Only the first 2 KB are examined.
///
/// Display-only (spec §4 step 1): callers such as `retrivio recall` read the head of the top-N
/// result files at query time; ranking never does file I/O.
pub fn parse_frontmatter_date(text: &str) -> Option<f64> {
    let mut end = text.len().min(2048);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let head = &text[..end];
    let lines: Vec<&str> = head.lines().collect();
    let (block_date, body_start) = match yaml_block_end(&lines) {
        Some(close) => (yaml_block_date(&lines[1..close]), close + 1),
        None => (None, 0),
    };
    if block_date.is_some() {
        return block_date;
    }
    plain_date_line(&lines[body_start.min(lines.len())..])
}

/// Index of the closing `---` of a leading YAML block, if the text starts with one.
fn yaml_block_end(lines: &[&str]) -> Option<usize> {
    let first = lines.first()?.trim_start_matches('\u{feff}').trim_end();
    if first != "---" {
        return None;
    }
    lines
        .iter()
        .enumerate()
        .skip(1)
        .take(FRONTMATTER_MAX_LINES - 1)
        .find(|(_, l)| l.trim_end() == "---")
        .map(|(i, _)| i)
}

/// Preferred date inside a YAML block: an `updated`-style key beats `date`.
fn yaml_block_date(body: &[&str]) -> Option<f64> {
    let mut updated: Option<f64> = None;
    let mut date: Option<f64> = None;
    for line in body {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.starts_with(char::is_whitespace) {
            continue; // nested key
        }
        let key = key.trim().to_ascii_lowercase();
        let value = clean_date_value(value);
        match key.as_str() {
            "updated" | "last_updated" | "last-updated" | "modified" => {
                if updated.is_none() {
                    updated = parse_date_value(value);
                }
            }
            "date" => {
                if date.is_none() {
                    date = parse_date_value(value);
                }
            }
            _ => {}
        }
    }
    updated.or(date)
}

/// `Date:` / `Updated:` / `Last updated:` among the first non-empty lines outside code fences.
/// Indented lines are skipped (nested YAML or code, never a document header).
fn plain_date_line(lines: &[&str]) -> Option<f64> {
    let mut seen = 0usize;
    let mut fence: Option<char> = None;
    for raw in lines {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let marker = if trimmed.starts_with("```") {
            Some('`')
        } else if trimmed.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        match (fence, marker) {
            (None, Some(c)) => {
                fence = Some(c);
                seen += 1;
                if seen >= PLAIN_DATE_MAX_LINES {
                    break;
                }
                continue;
            }
            (Some(open), Some(c)) if open == c => {
                fence = None;
                seen += 1;
                if seen >= PLAIN_DATE_MAX_LINES {
                    break;
                }
                continue;
            }
            (Some(_), _) => continue,
            (None, None) => {}
        }
        seen += 1;
        if !raw.starts_with(char::is_whitespace) {
            let line = trimmed.trim_start_matches(['*', '_', '#', '-', '>', ' ']);
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().trim_end_matches(['*', '_']).to_ascii_lowercase();
                if matches!(key.as_str(), "date" | "updated" | "last updated") {
                    if let Some(ts) = parse_date_value(clean_date_value(value)) {
                        return Some(ts);
                    }
                }
            }
        }
        if seen >= PLAIN_DATE_MAX_LINES {
            break;
        }
    }
    None
}

/// `YYYY-MM-DD` (time part ignored) or an English date (`July 10, 2026`, `17 September 2026`,
/// `19 Sep 2026`, `Thursday, March 5, 2026 9:00`).
fn parse_date_value(value: &str) -> Option<f64> {
    parse_ymd(value).or_else(|| parse_english_date(value))
}

const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const WEEKDAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "mon",
    "tue",
    "tues",
    "wed",
    "thu",
    "thur",
    "thurs",
    "fri",
    "sat",
    "sun",
];

/// 1-based month for a full name or an abbreviation of at least three letters (`Sep`, `Sept`).
fn month_number(word: &str) -> Option<u32> {
    let w = word.to_ascii_lowercase();
    if w.len() < 3 {
        return None;
    }
    MONTHS
        .iter()
        .position(|m| m.starts_with(w.as_str()))
        .map(|i| i as u32 + 1)
}

fn english_day(word: &str) -> Option<u32> {
    let digits = word.trim_end_matches(|c: char| c.is_ascii_alphabetic()); // 5th, 22nd
    if digits.is_empty() || digits.len() > 2 {
        return None;
    }
    digits_value(digits.as_bytes()).map(|v| v as u32)
}

fn english_year(word: &str) -> Option<i64> {
    (word.len() == 4)
        .then(|| digits_value(word.as_bytes()))
        .flatten()
}

/// `Month D YYYY` or `D Month YYYY`, commas and periods ignored, an optional leading weekday,
/// ordinal suffixes allowed, anything after the year ignored.
fn parse_english_date(value: &str) -> Option<f64> {
    let mut words: Vec<&str> = value
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | '.'))
        .filter(|w| !w.is_empty())
        .take(4)
        .collect();
    if words
        .first()
        .map(|w| WEEKDAYS.contains(&w.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
    {
        words.remove(0);
    }
    if words.len() < 3 {
        return None;
    }
    let (m, d) = if let Some(m) = month_number(words[0]) {
        (m, english_day(words[1])?)
    } else if let Some(m) = month_number(words[1]) {
        (m, english_day(words[0])?)
    } else {
        return None;
    };
    civil_ts(english_year(words[2])?, m, d)
}

fn clean_date_value(value: &str) -> &str {
    value
        .trim()
        .trim_start_matches(['*', '_'])
        .trim()
        .trim_matches(['"', '\'', '[', ']'])
        .trim()
}

/// Parse a leading `YYYY-MM-DD`. The date must end at a non-alphanumeric boundary
/// (`2026-09-19abc` is rejected); a time part introduced by `T` plus a digit is ignored
/// (`2026-09-19T10:00:00Z` is accepted).
fn parse_ymd(value: &str) -> Option<f64> {
    let b = value.as_bytes();
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    if let Some(next) = b.get(10) {
        let time_part = (*next == b'T' || *next == b't')
            && b.get(11).map(|c| c.is_ascii_digit()).unwrap_or(false);
        if next.is_ascii_alphanumeric() && !time_part {
            return None;
        }
    }
    let y = digits_value(&b[..4])?;
    let m = digits_value(&b[5..7])? as u32;
    let d = digits_value(&b[8..10])? as u32;
    civil_ts(y, m, d)
}

/// Exponential decay `0.5 ^ (max(age, 0) / half_life)`; 1.0 for brand-new content.
pub fn recency_score(age_days: f64, half_life_days: f64) -> f64 {
    let half_life = if half_life_days.is_finite() && half_life_days > 0.0 {
        half_life_days
    } else {
        1.0
    };
    let age = if age_days.is_finite() {
        age_days.max(0.0)
    } else {
        0.0
    };
    0.5f64.powf(age / half_life)
}

/// Linear blend `(1 - w) * score + w * recency`.
pub fn blend(score: f64, recency: f64, weight: f64) -> f64 {
    let w = weight.clamp(0.0, 1.0);
    (1.0 - w) * score + w * recency
}

/// Display tier: `fresh` (< 14 d), `aging` (14..=35 d), `stale` (> 35 d); records are `record`.
pub fn tier(age_days: f64, is_record: bool) -> &'static str {
    tier_for_role(
        age_days,
        if is_record {
            Role::Record
        } else {
            Role::Knowledge
        },
    )
}

/// Display tier by role. Records are events and never stale: always `record`. State
/// (handoffs, status, plans) older than 35 days is `verify`: it was the truth once and must be
/// checked before it is repeated. Knowledge keeps `fresh` / `aging` / `stale`.
pub fn tier_for_role(age_days: f64, role: Role) -> &'static str {
    match role {
        Role::Record => "record",
        Role::State | Role::Knowledge => {
            if age_days < FRESH_MAX_DAYS {
                "fresh"
            } else if age_days <= AGING_MAX_DAYS {
                "aging"
            } else if role == Role::State {
                "verify"
            } else {
                "stale"
            }
        }
    }
}

/// True when a `state` document is old enough that its facts need re-checking (> 35 days).
pub fn needs_verify(age_days: f64, role: Role) -> bool {
    role == Role::State && age_days > AGING_MAX_DAYS
}

/// Age in days of `ts` relative to `now`, never negative.
pub fn age_days(now: f64, ts: f64) -> f64 {
    if !now.is_finite() || !ts.is_finite() {
        return 0.0;
    }
    ((now - ts) / DAY_SECS).max(0.0)
}

/// `YYYY-MM-DD` in UTC for a unix timestamp.
pub fn format_ymd(ts: f64) -> String {
    let days = if ts.is_finite() {
        (ts / DAY_SECS).floor() as i64
    } else {
        0
    };
    let (y, m, d) = civil_from_days(days);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ymd(y: i64, m: u32, d: u32) -> f64 {
        days_from_civil(y, m, d) as f64 * DAY_SECS
    }

    #[test]
    fn civil_round_trip_and_known_epochs() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for days in [-1000, 0, 1, 10_957, 19_723, 20_000, 40_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "round trip {}", days);
        }
        assert_eq!(format_ymd(ymd(2026, 9, 19)), "2026-09-19");
        assert_eq!(format_ymd(ymd(2024, 2, 29) + 3_600.0), "2024-02-29");
    }

    #[test]
    fn path_date_month_prefix_resolves_to_first_of_month() {
        assert_eq!(path_date("202609-foo/bar.md"), Some(ymd(2026, 9, 1)));
        assert_eq!(
            path_date("/Users/x/c-projects/202609-foo/notes/bar.md"),
            Some(ymd(2026, 9, 1))
        );
    }

    #[test]
    fn path_date_day_prefix() {
        assert_eq!(path_date("20260715-x.md"), Some(ymd(2026, 7, 15)));
        assert_eq!(path_date("a/20260715_x.md"), Some(ymd(2026, 7, 15)));
        assert_eq!(path_date("a/20260715.md"), Some(ymd(2026, 7, 15)));
        assert_eq!(path_date("a/20260715"), Some(ymd(2026, 7, 15)));
        assert_eq!(
            path_date("specs/2026-09-19-proactive-recall-design.md"),
            Some(ymd(2026, 9, 19))
        );
    }

    #[test]
    fn path_date_day_component_beats_month_parent() {
        assert_eq!(
            path_date("202609-foo/20260715-x.md"),
            Some(ymd(2026, 7, 15))
        );
        // Precision wins even when the month component is deeper.
        assert_eq!(
            path_date("20260715-x/202609-foo/bar.md"),
            Some(ymd(2026, 7, 15))
        );
        // Equal precision: deepest wins.
        assert_eq!(
            path_date("20260101-a/20260715-b/c.md"),
            Some(ymd(2026, 7, 15))
        );
    }

    #[test]
    fn path_date_rejects_non_dated_and_invalid() {
        assert_eq!(path_date("src/main.rs"), None);
        assert_eq!(path_date("/Users/x/projects/retrivio/README.md"), None);
        assert_eq!(path_date("202613-foo/bar.md"), None, "month 13");
        assert_eq!(path_date("20260231-foo.md"), None, "Feb 31");
        assert_eq!(path_date("20260700-foo.md"), None, "day 0");
        assert_eq!(path_date("12345678-foo.md"), None, "year out of range");
        assert_eq!(
            path_date("202609foo/bar.md"),
            None,
            "no delimiter after digits"
        );
        assert_eq!(path_date("2026091-foo.md"), None, "seven digits");
        assert_eq!(
            path_date("x20260715-foo.md"),
            None,
            "not at component start"
        );
        assert_eq!(path_date("1234567890-id.md"), None, "long serial number");
        assert_eq!(path_date("2026-13-01-foo.md"), None, "dashed month 13");
    }

    #[test]
    fn content_date_prefers_newer_source_and_rejects_future() {
        let now = ymd(2026, 9, 19);
        let path = "202609-foo/bar.md"; // 2026-09-01
        let (ts, src) = content_date(path, ymd(2026, 8, 20), now);
        assert_eq!((ts, src), (ymd(2026, 9, 1), "path-date"));
        let (ts, src) = content_date(path, ymd(2026, 9, 10), now);
        assert_eq!((ts, src), (ymd(2026, 9, 10), "mtime"));
        let (ts, src) = content_date("plain/bar.md", ymd(2026, 9, 10), now);
        assert_eq!((ts, src), (ymd(2026, 9, 10), "mtime"));
        // Future path date is implausible: fall back to mtime.
        let (ts, src) = content_date("20271231-plan.md", ymd(2026, 9, 10), now);
        assert_eq!((ts, src), (ymd(2026, 9, 10), "mtime"));
        // Future mtime is implausible: use the path date.
        let (ts, src) = content_date(path, ymd(2030, 1, 1), now);
        assert_eq!((ts, src), (ymd(2026, 9, 1), "path-date"));
        // Both implausible: the document counts as current.
        let (ts, src) = content_date("20271231-plan.md", ymd(2030, 1, 1), now);
        assert_eq!((ts, src), (now, "mtime"));
        // No path date and a far-future mtime: also current.
        let (ts, src) = content_date("plain/bar.md", ymd(2030, 1, 1), now);
        assert_eq!((ts, src), (now, "mtime"));
        let (ts, src) = content_date("plain/bar.md", f64::NAN, now);
        assert_eq!((ts, src), (now, "mtime"));
        // Within the two-day slack is fine.
        let (ts, src) = content_date("20260920-tomorrow.md", ymd(2026, 9, 1), now);
        assert_eq!((ts, src), (ymd(2026, 9, 20), "path-date"));
    }

    #[test]
    fn frontmatter_date_keys() {
        assert_eq!(
            parse_frontmatter_date("---\ntitle: x\ndate: 2026-09-19\n---\nbody"),
            Some(ymd(2026, 9, 19))
        );
        assert_eq!(
            parse_frontmatter_date("Updated: 2026-08-01T10:00:00Z\n"),
            Some(ymd(2026, 8, 1))
        );
        assert_eq!(
            parse_frontmatter_date("---\nlast_updated: \"2026-07-04\"\n---\n"),
            Some(ymd(2026, 7, 4))
        );
        assert_eq!(
            parse_frontmatter_date("Last Updated: 2026-07-05"),
            Some(ymd(2026, 7, 5))
        );
        assert_eq!(
            parse_frontmatter_date("---\nmodified: 2026-02-30\n---\n"),
            None
        );
        assert_eq!(parse_frontmatter_date("date: yesterday"), None);
        assert_eq!(parse_frontmatter_date("no dates here"), None);
        // Beyond the first 2 KB is ignored.
        let far = format!("{}\ndate: 2026-09-19\n", "x".repeat(2100));
        assert_eq!(parse_frontmatter_date(&far), None);
    }

    #[test]
    fn yaml_block_prefers_updated_over_date_and_must_close() {
        let doc = "---\ntitle: \"Bedrock setup\"\ndate: 2026-05-01\nupdated: 2026-09-15\n---\n# Heading\n";
        assert_eq!(parse_frontmatter_date(doc), Some(ymd(2026, 9, 15)));
        let reversed = "---\nupdated: 2026-09-15\ndate: 2026-05-01\n---\n";
        assert_eq!(parse_frontmatter_date(reversed), Some(ymd(2026, 9, 15)));
        assert_eq!(
            parse_frontmatter_date("---\nlast-updated: 2026-09-14\ndate: 2026-05-01\n---\n"),
            Some(ymd(2026, 9, 14))
        );
        assert_eq!(
            parse_frontmatter_date("---\nmodified: 2026-09-13\ndate: 2026-05-01\n---\n"),
            Some(ymd(2026, 9, 13))
        );
        // Nested keys inside the block are not top-level dates.
        assert_eq!(
            parse_frontmatter_date("---\nmeta:\n  date: 2026-05-01\ntitle: x\n---\nbody"),
            None
        );
        // A BOM before the opening fence is tolerated.
        assert_eq!(
            parse_frontmatter_date("\u{feff}---\ndate: 2026-09-01\n---\n"),
            Some(ymd(2026, 9, 1))
        );
        // Block without a date key: fall through to a plain line right after it.
        assert_eq!(
            parse_frontmatter_date("---\ntitle: x\n---\n\nDate: 2026-09-02\n"),
            Some(ymd(2026, 9, 2))
        );
        // An unclosed block is not front matter: its `date:` is a plain line, `Date:` style
        // keys only. (`date:` lowercase plain line still matches: keys are case-insensitive.)
        let unclosed = format!("---\n{}date: 2026-09-03\n", "line\n".repeat(70));
        assert_eq!(
            parse_frontmatter_date(&unclosed),
            None,
            "beyond 10 non-empty lines"
        );
        assert_eq!(
            parse_frontmatter_date("---\ndate: 2026-09-03\nno close"),
            Some(ymd(2026, 9, 3))
        );
        // Closing fence must arrive within 60 lines.
        let long_block = format!("---\n{}date: 2026-09-04\n---\n", "k: v\n".repeat(70));
        assert_eq!(parse_frontmatter_date(&long_block), None);
        let ok_block = format!("---\n{}date: 2026-09-04\n---\n", "k: v\n".repeat(50));
        assert_eq!(parse_frontmatter_date(&ok_block), Some(ymd(2026, 9, 4)));
    }

    #[test]
    fn plain_date_line_rules() {
        // Eric's specs: `Date: 2026-09-19` on line 3.
        let spec = "# Proactive recall design\n\nDate: 2026-09-19\nAuthor: Eric\n\n## 1. Goal\n";
        assert_eq!(parse_frontmatter_date(spec), Some(ymd(2026, 9, 19)));
        assert_eq!(
            parse_frontmatter_date("**Updated:** 2026-09-18\n"),
            Some(ymd(2026, 9, 18))
        );
        assert_eq!(
            parse_frontmatter_date("- Last updated: 2026-09-17"),
            Some(ymd(2026, 9, 17))
        );
        assert_eq!(
            parse_frontmatter_date("> date: 2026-09-16"),
            Some(ymd(2026, 9, 16))
        );
        // Only the accepted keys count outside a YAML block.
        assert_eq!(parse_frontmatter_date("modified: 2026-09-15\n"), None);
        assert_eq!(parse_frontmatter_date("last_updated: 2026-09-15\n"), None);
        assert_eq!(parse_frontmatter_date("Due date: 2026-09-15\n"), None);
        assert_eq!(parse_frontmatter_date("Start: 2026-09-15\n"), None);
        // Beyond the first 10 non-empty lines: rejected.
        let deep = format!("{}Date: 2026-09-19\n", "text line\n\n".repeat(10));
        assert_eq!(parse_frontmatter_date(&deep), None);
        let tenth = format!("{}Date: 2026-09-19\n", "text line\n\n".repeat(9));
        assert_eq!(parse_frontmatter_date(&tenth), Some(ymd(2026, 9, 19)));
        // Code fences are skipped (and count as lines).
        let fenced = "# T\n```yaml\ndate: 2026-01-01\n```\nDate: 2026-09-19\n";
        assert_eq!(parse_frontmatter_date(fenced), Some(ymd(2026, 9, 19)));
        let only_fenced = "# T\n```\nDate: 2026-01-01\n```\n";
        assert_eq!(parse_frontmatter_date(only_fenced), None);
        // A plain `date:` deep in a body never counts.
        let body = "# T\nintro\nmore\nmore\nmore\nmore\nmore\nmore\nmore\nmore\ndate: 2026-09-19\n";
        assert_eq!(parse_frontmatter_date(body), None);
        // Indented lines are never header lines (an unclosed block's nested key, code).
        assert_eq!(
            parse_frontmatter_date("---\nmeta:\n  date: 2026-05-01\nno close"),
            None
        );
        assert_eq!(parse_frontmatter_date("# T\n    Date: 2026-05-01\n"), None);
        // Fences must be closed by the same marker.
        let mixed = "~~~\n```\nDate: 2026-01-01\n~~~\nDate: 2026-09-19\n";
        assert_eq!(parse_frontmatter_date(mixed), Some(ymd(2026, 9, 19)));
    }

    #[test]
    fn english_dates_from_the_corpus() {
        assert_eq!(
            parse_frontmatter_date(
                "# AWS Context\n\nLast updated: 17 September 2026, GA roadmap draft pass\n"
            ),
            Some(ymd(2026, 9, 17))
        );
        assert_eq!(
            parse_frontmatter_date(
                "---\ntitle: \"Scout\"\nauthor: \"x\"\ndate: \"July 10, 2026\"\n---\n"
            ),
            Some(ymd(2026, 7, 10))
        );
        assert_eq!(
            parse_frontmatter_date(
                "# Firewall review\n\nDate: 19 Sep 2026. Appliance `fw-edge-01`\n"
            ),
            Some(ymd(2026, 9, 19))
        );
        assert_eq!(
            parse_frontmatter_date("**Date:** November 5, 2025\n"),
            Some(ymd(2025, 11, 5))
        );
        assert_eq!(
            parse_frontmatter_date("**Date**: May 22nd, 2026\n"),
            Some(ymd(2026, 5, 22))
        );
        assert_eq!(
            parse_frontmatter_date("**Date:** Thursday, March 5, 2026, 9:00 PT\n"),
            Some(ymd(2026, 3, 5))
        );
        assert_eq!(
            parse_frontmatter_date("Date: 1 Sept 2026\n"),
            Some(ymd(2026, 9, 1))
        );
        // Placeholders and partial dates are rejected.
        assert_eq!(
            parse_frontmatter_date("**Date:** Add to the table above.\n"),
            None
        );
        assert_eq!(parse_frontmatter_date("**Date**: [Date]\n"), None);
        assert_eq!(parse_frontmatter_date("Date: May 2026\n"), None);
        assert_eq!(parse_frontmatter_date("Date: 5 mayonnaise 2026\n"), None);
        assert_eq!(parse_frontmatter_date("Date: 32 March 2026\n"), None);
        assert_eq!(parse_frontmatter_date("Date: March 5, 26\n"), None);
        assert_eq!(parse_english_date("Ma 5 2026"), None);
    }

    #[test]
    fn ymd_boundary_rules() {
        assert_eq!(parse_ymd("2026-09-19"), Some(ymd(2026, 9, 19)));
        assert_eq!(parse_ymd("2026-09-19T10:00:00Z"), Some(ymd(2026, 9, 19)));
        assert_eq!(parse_ymd("2026-09-19t10:00"), Some(ymd(2026, 9, 19)));
        assert_eq!(parse_ymd("2026-09-19 10:00"), Some(ymd(2026, 9, 19)));
        assert_eq!(parse_ymd("2026-09-19."), Some(ymd(2026, 9, 19)));
        assert_eq!(parse_ymd("2026-09-19abc"), None);
        assert_eq!(parse_ymd("2026-09-19Tabc"), None);
        assert_eq!(parse_ymd("2026-09-191"), None);
        assert_eq!(parse_ymd("2026-09-1"), None);
        assert_eq!(parse_ymd("2026/09/19"), None);
        assert_eq!(parse_ymd("2026-13-01"), None);
    }

    #[test]
    fn records_are_dated_by_the_event_not_the_last_edit() {
        let now = ymd(2026, 9, 19);
        // A transcript from July, touched up in September: still a July event.
        let (ts, src) = content_date_for_role(
            "customer-signals/acme/20260715-workshop/transcript.txt",
            ymd(2026, 9, 10),
            now,
            Role::Record,
        );
        assert_eq!((ts, src), (ymd(2026, 7, 15), "path-date"));
        // The same file as knowledge would take the newer edit.
        let (ts, src) = content_date_for_role(
            "customer-signals/acme/20260715-workshop/transcript.txt",
            ymd(2026, 9, 10),
            now,
            Role::Knowledge,
        );
        assert_eq!((ts, src), (ymd(2026, 9, 10), "mtime"));
        // No path date: the mtime stands in.
        let (ts, src) =
            content_date_for_role("transcripts/call.txt", ymd(2026, 8, 1), now, Role::Record);
        assert_eq!((ts, src), (ymd(2026, 8, 1), "mtime"));
        // A future path date is implausible; a future mtime too.
        let (ts, src) =
            content_date_for_role("20271231-call.txt", ymd(2026, 8, 1), now, Role::Record);
        assert_eq!((ts, src), (ymd(2026, 8, 1), "mtime"));
        let (ts, src) =
            content_date_for_role("transcripts/call.txt", ymd(2030, 1, 1), now, Role::Record);
        assert_eq!((ts, src), (now, "mtime"));
    }

    #[test]
    fn recency_score_half_life() {
        assert!((recency_score(0.0, 21.0) - 1.0).abs() < 1e-12);
        assert!((recency_score(21.0, 21.0) - 0.5).abs() < 1e-12);
        assert!((recency_score(42.0, 21.0) - 0.25).abs() < 1e-12);
        assert!(
            (recency_score(-5.0, 21.0) - 1.0).abs() < 1e-12,
            "negative age clamps to 0"
        );
        assert!((recency_score(90.0, 90.0) - 0.5).abs() < 1e-12);
        assert!(
            recency_score(10.0, 0.0) > 0.0,
            "degenerate half-life does not panic"
        );
    }

    #[test]
    fn blend_math() {
        assert!((blend(0.8, 0.5, 0.0) - 0.8).abs() < 1e-12);
        assert!((blend(0.8, 0.5, 1.0) - 0.5).abs() < 1e-12);
        assert!((blend(0.8, 0.5, 0.12) - (0.88 * 0.8 + 0.12 * 0.5)).abs() < 1e-12);
        assert!((blend(0.8, 1.0, 0.04) - (0.96 * 0.8 + 0.04)).abs() < 1e-12);
    }

    #[test]
    fn tiers() {
        assert_eq!(tier(0.0, false), "fresh");
        assert_eq!(tier(13.9, false), "fresh");
        assert_eq!(tier(14.0, false), "aging");
        assert_eq!(tier(35.0, false), "aging");
        assert_eq!(tier(35.1, false), "stale");
        assert_eq!(tier(400.0, false), "stale");
        assert_eq!(tier(400.0, true), "record");
        assert_eq!(tier(1.0, true), "record");
        // By role: state turns to `verify` instead of `stale`; records never age out.
        assert_eq!(tier_for_role(3.0, Role::State), "fresh");
        assert_eq!(tier_for_role(20.0, Role::State), "aging");
        assert_eq!(tier_for_role(36.0, Role::State), "verify");
        assert_eq!(tier_for_role(36.0, Role::Knowledge), "stale");
        assert_eq!(tier_for_role(36.0, Role::Record), "record");
        assert_eq!(tier_for_role(1.0, Role::Record), "record");
        assert!(needs_verify(36.0, Role::State));
        assert!(!needs_verify(35.0, Role::State));
        assert!(!needs_verify(400.0, Role::Knowledge));
        assert!(!needs_verify(400.0, Role::Record));
    }

    #[test]
    fn age_days_never_negative() {
        let now = ymd(2026, 9, 19);
        assert!((age_days(now, ymd(2026, 9, 9)) - 10.0).abs() < 1e-9);
        assert_eq!(age_days(now, ymd(2026, 9, 29)), 0.0);
    }
}
