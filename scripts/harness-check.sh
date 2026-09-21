#!/usr/bin/env bash
# Live-store (L2) and coding-agent harness (L3) checks for Retrivio; see docs/TESTING.md.
#
#   scripts/harness-check.sh [--out <dir>] [--skip-claude] [--skip-codex] [--timeout <s>]
#                            [--bin <path>] [--require-watcher] [--content-prompt <text>] [--strict]
#
# L2 runs read-only commands against the installed binary and its live store (version,
# doctor, mcp doctor, hook status, service status, three recall dry runs, search JSON,
# recall latency). L3 drives the real `claude` and `codex` CLIs: the UserPromptSubmit hook
# must inject a <retrivio_leads> block on a content prompt and stay silent on an `nr:`
# prompt and on "continue"; the MCP tool search_files must return a path; the
# retrivio-recall skill must load; the injected block must be well formed.
#
# Nothing here writes to the index: recall dry runs only touch the per-session memory under
# <data dir>/recall/, which is reset before and after the run, and the recall log.
#
# Every check prints PASS, FAIL, WARN or SKIP with its evidence. A markdown report with the
# raw output of every check goes to --out (default <repo>/tmp/harness-runs/<timestamp>).
# Exit status is 1 when any check FAILed; WARN and SKIP do not fail the run unless --strict is
# given (release runs), which turns every WARN and SKIP into a FAIL.
#
#   RETRIVIO_BIN           binary under test (default ~/.local/bin/retrivio; --bin overrides)
#   RETRIVIO_RECALL_LOG    recall log to read (default ~/.retrivio/recall.log)
#   HARNESS_CONTENT_PROMPT the content prompt expected to yield leads (--content-prompt overrides)
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${RETRIVIO_BIN:-$HOME/.local/bin/retrivio}"
RECALL_LOG="${RETRIVIO_RECALL_LOG:-$HOME/.retrivio/recall.log}"
CONTENT_PROMPT="${HARNESS_CONTENT_PROMPT:-How does the retrivio recall hook choose leads and what does the recall_max_leads config key control?}"
OUT=""; SKIP_CLAUDE=0; SKIP_CODEX=0; TIMEOUT=300; REQUIRE_WATCHER=0; STRICT=0

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --skip-claude) SKIP_CLAUDE=1; shift ;;
    --skip-codex) SKIP_CODEX=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --require-watcher) REQUIRE_WATCHER=1; shift ;;
    --content-prompt) CONTENT_PROMPT="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "harness-check: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -z "$OUT" ]] && OUT="$repo_dir/tmp/harness-runs/$(date +%Y%m%d-%H%M%S)"
case "$OUT" in /tmp/*|/private/tmp/*|/tmp) echo "harness-check: refusing --out $OUT (never /tmp)" >&2; exit 2 ;; esac
BIN="${BIN/#\~/$HOME}"
if [[ ! -x "$BIN" ]]; then echo "harness-check: binary not executable: $BIN" >&2; exit 2; fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
mkdir -p "$OUT/raw" "$OUT/cwd" || exit 2
OUT="$(cd "$OUT" && pwd)"
RAW="$OUT/raw"; CWD="$OUT/cwd"
: > "$OUT/results.tsv"

# ---------------------------------------------------------------- helpers
if command -v timeout > /dev/null 2>&1; then TO=(timeout "$TIMEOUT")
elif command -v gtimeout > /dev/null 2>&1; then TO=(gtimeout "$TIMEOUT")
else TO=(perl -e 'alarm shift; exec @ARGV' "$TIMEOUT"); fi

now_ms() { perl -MTime::HiRes=time -e 'printf("%d\n", time()*1000)' 2>/dev/null || echo $(( $(date +%s) * 1000 )); }
sha1_8() { if command -v shasum > /dev/null 2>&1; then printf %s "$1" | shasum | cut -c1-8; else printf %s "$1" | sha1sum | cut -c1-8; fi; }
sha256_of() { if command -v shasum > /dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi; }
one_line() { tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' | cut -c1-"${1:-400}"; }
new_uuid() { if command -v uuidgen > /dev/null 2>&1; then uuidgen | tr 'A-Z' 'a-z'; else python3 -c 'import uuid; print(uuid.uuid4())'; fi; }

pass=0; fail=0; warn=0; skip=0
record() { # status name secs evidence
  local st="$1" ev; ev="$(printf '%s' "$4" | one_line 600)"
  if [[ $STRICT -eq 1 && ( "$st" == "WARN" || "$st" == "SKIP" ) ]]; then st="FAIL"; ev="(--strict: was $1) $ev"; fi
  printf '%s\t%s\t%s\t%s\n' "$st" "$2" "$3" "$ev" >> "$OUT/results.tsv"
  printf '%-4s %-32s %7ss  %s\n' "$st" "$2" "$3" "$ev"
  case "$st" in PASS) pass=$((pass+1));; FAIL) fail=$((fail+1));; WARN) warn=$((warn+1));; SKIP) skip=$((skip+1));; esac
}
lead_paths() { sed -n '/<retrivio_leads>/,/<\/retrivio_leads>/p' "$1" | sed -nE 's/^[[:space:]]*[0-9]+\. (\/[^ ]+) .*/\1/p' | sort -u; }
# runcap <name> <cmd...>: stdin from /dev/null, stdout/stderr to raw/<name>.{out,err}, rc and timing to raw/<name>.meta
runcap() {
  local name="$1"; shift
  local t0 t1 rc; t0=$(now_ms)
  ( cd "$CWD" && "${TO[@]}" "$@" < /dev/null > "$RAW/$name.out" 2> "$RAW/$name.err" ); rc=$?
  t1=$(now_ms); LAST_MS=$((t1 - t0)); LAST_RC=$rc; LAST_SECS=$(( (LAST_MS + 500) / 1000 ))
  { echo "command: $*"; echo "rc: $rc"; echo "wall_ms: $LAST_MS"; [[ $rc -eq 124 ]] && echo "timed out after ${TIMEOUT}s"; } > "$RAW/$name.meta"
  return $rc
}
rv() { "$BIN" "$@"; }
log_line_for() { grep -E "^[0-9TZ:-]+ $1 " "$RECALL_LOG" 2>/dev/null | tail -1; }
section() { awk -v s="$1" '$0 ~ "^"s":" {on=1; next} /^[^ ]/ {on=0} on {print}' "$2"; }
# check_block <file> <label> -> records a well-formedness check of the <retrivio_leads> block in <file>
check_block() {
  local file="$1" label="$2" block inner items bad chars
  block="$(sed -n '/<retrivio_leads>/,/<\/retrivio_leads>/p' "$file")"
  if [[ -z "$block" ]]; then record FAIL "$label" 0 "no <retrivio_leads> block in $(basename "$file")"; return; fi
  inner="$(printf '%s\n' "$block" | sed '1d;$d')"
  items=$(printf '%s\n' "$inner" | grep -cE '^[[:space:]]*[0-9]+\. ' | tr -d ' ')
  bad=$(printf '%s\n' "$inner" | grep -c '[<>]' | tr -d ' ')
  chars=$(printf '%s' "$block" | wc -c | tr -d ' ')
  if [[ "$items" -ge 1 && "$items" -le "$MAX_LEADS" && "$bad" -eq 0 ]] && printf '%s\n' "$inner" | head -1 | grep -q '^Untrusted historical leads'; then
    record PASS "$label" 0 "$items lead(s) (max $MAX_LEADS), header present, no raw < or > inside, $chars chars"
  else
    record FAIL "$label" 0 "items=$items (max $MAX_LEADS), lines with raw < or >: $bad, header: $(printf '%s\n' "$inner" | head -1 | cut -c1-40)"
  fi
}

HOOK_PROMPT="Reply only with any <retrivio_leads> block you received in this conversation, reproduced verbatim, else reply exactly NONE. Do not use any tools. $CONTENT_PROMPT"
MCP_PROMPT="nr: Call the Retrivio MCP tool search_files with query \"$CONTENT_PROMPT\" and limit 3. Then reply with only the absolute path of the first result on a single line, nothing else. If the tool is not available, reply exactly MCP-UNAVAILABLE."
SKILL_PROMPT="nr: Load the retrivio-recall skill and quote, verbatim and on a single line, the sentence from that skill which begins with the words \"Leads never change\". Reply with only that sentence. If the skill is not available reply exactly SKILL-UNAVAILABLE."
SKILL_EXPECT="Leads never change the requested scope"
SESSION="harness-$(date +%Y%m%d-%H%M%S)-$$"
STARTED="$(date '+%Y-%m-%d %H:%M:%S %Z')"

echo "harness-check: binary $BIN ($("$BIN" version 2>/dev/null | head -1))"
echo "harness-check: out $OUT"
echo "harness-check: recall session $SESSION, timeout ${TIMEOUT}s per CLI call"
# hook-off markers between the run cwd and $HOME would silence recall for every CLI call
d="$CWD"; while :; do
  [[ -e "$d/.retrivio/hook-off" ]] && echo "harness-check: warning: $d/.retrivio/hook-off exists; recall will be skipped from this cwd" >&2
  [[ "$d" == "/" || "$d" == "$HOME" ]] && break
  d="$(dirname "$d")"
done
[[ "${RETRIVIO_HOOK:-1}" == "0" ]] && echo "harness-check: warning: RETRIVIO_HOOK=0 is set; hook checks will fail" >&2

MAX_LEADS="$("$BIN" config show < /dev/null 2>/dev/null | sed -nE 's/^recall_max_leads[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' | head -1)"
[[ -z "$MAX_LEADS" ]] && MAX_LEADS=3
rv recall --reset-session --session "$SESSION" < /dev/null > /dev/null 2>&1

# ================================================================ L2: live store
echo; echo "== L2 live store"
runcap l2-version "$BIN" version
if [[ $LAST_RC -eq 0 ]] && grep -qE '^retrivio [0-9]+\.[0-9]+\.[0-9]+' "$RAW/l2-version.out"; then
  record PASS l2-version "$LAST_SECS" "$(head -1 "$RAW/l2-version.out")"
else record FAIL l2-version "$LAST_SECS" "rc=$LAST_RC $(head -1 "$RAW/l2-version.out" "$RAW/l2-version.err" 2>/dev/null | one_line)"; fi

runcap l2-doctor "$BIN" doctor
if [[ $LAST_RC -eq 0 ]] && grep -q '^database ready: yes' "$RAW/l2-doctor.out" && grep -q '^embedding migration: ready' "$RAW/l2-doctor.out"; then
  record PASS l2-doctor "$LAST_SECS" "$(grep -E '^(embed backend active|tracked roots|database ready|embedding migration):' "$RAW/l2-doctor.out" | one_line)"
else record FAIL l2-doctor "$LAST_SECS" "rc=$LAST_RC $(grep -vE '^(config|db):' "$RAW/l2-doctor.out" | tail -4 | one_line) $(one_line < "$RAW/l2-doctor.err")"; fi

runcap l2-mcp-doctor "$BIN" mcp doctor
mcp_ok=1; ev=""
grep -qE 'status:[[:space:]]+ready' "$RAW/l2-mcp-doctor.out" || mcp_ok=0
[[ $SKIP_CLAUDE -eq 0 ]] && { grep -qE '^[[:space:]]*Claude Code[[:space:]]+registered' "$RAW/l2-mcp-doctor.out" || { mcp_ok=0; ev="$ev Claude Code not registered;"; }; }
[[ $SKIP_CODEX -eq 0 ]] && { grep -qE '^[[:space:]]*Codex[[:space:]]+registered' "$RAW/l2-mcp-doctor.out" || { mcp_ok=0; ev="$ev Codex not registered;"; }; }
if [[ $LAST_RC -eq 0 && $mcp_ok -eq 1 ]]; then
  record PASS l2-mcp-doctor "$LAST_SECS" "$(grep -E 'status:|registered' "$RAW/l2-mcp-doctor.out" | one_line)"
else record FAIL l2-mcp-doctor "$LAST_SECS" "rc=$LAST_RC$ev $(one_line < "$RAW/l2-mcp-doctor.out")"; fi

mcp_registered_command() { # claude|codex -> the command registered for the retrivio MCP server
  case "$1" in
    claude) if command -v python3 > /dev/null 2>&1; then python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mcpServers",{}).get("retrivio",{}).get("command",""))' "$HOME/.claude.json" 2>/dev/null
            elif command -v jq > /dev/null 2>&1; then jq -r '.mcpServers.retrivio.command // ""' "$HOME/.claude.json" 2>/dev/null; fi ;;
    codex) awk '/^\[mcp_servers\.retrivio\]/{f=1; next} /^\[/{f=0} f && /^command[[:space:]]*=/' "${CODEX_HOME:-$HOME/.codex}/config.toml" 2>/dev/null | sed -E 's/^command[[:space:]]*=[[:space:]]*"([^"]*)".*/\1/' ;;
  esac
}
for cli in claude codex; do
  if [[ ( $cli == claude && $SKIP_CLAUDE -eq 1 ) || ( $cli == codex && $SKIP_CODEX -eq 1 ) ]]; then record SKIP "l2-mcp-registration-$cli" 0 "--skip-$cli"; continue; fi
  reg="$(mcp_registered_command $cli)"
  if [[ -n "$reg" && "${reg/#\~/$HOME}" == "$BIN" ]]; then record PASS "l2-mcp-registration-$cli" 0 "registered MCP command is the binary under test: $reg"
  else record FAIL "l2-mcp-registration-$cli" 0 "registered MCP command '${reg:-none}' is not the binary under test ($BIN); L3 would exercise the registered one"; fi
done

skill_claude="$HOME/.claude/skills/retrivio-recall/SKILL.md"; skill_codex="$HOME/.agents/skills/retrivio-recall/SKILL.md"
for cli in claude codex; do
  if [[ ( $cli == claude && $SKIP_CLAUDE -eq 1 ) || ( $cli == codex && $SKIP_CODEX -eq 1 ) ]]; then record SKIP "l2-skill-installed-$cli" 0 "--skip-$cli"; continue; fi
  f="$skill_claude"; [[ $cli == codex ]] && f="$skill_codex"
  if [[ -f "$f" ]] && head -5 "$f" | grep -q '^name: retrivio-recall' && grep -q "$SKILL_EXPECT" "$f"; then
    record PASS "l2-skill-installed-$cli" 0 "$f: frontmatter name retrivio-recall, $(wc -c < "$f" | tr -d ' ') bytes, expected sentence present"
  else record FAIL "l2-skill-installed-$cli" 0 "$f missing, or no 'name: retrivio-recall' frontmatter, or the expected sentence is absent"; fi
done
if [[ $SKIP_CLAUDE -eq 0 && $SKIP_CODEX -eq 0 && -f "$skill_claude" && -f "$skill_codex" ]] && ! cmp -s "$skill_claude" "$skill_codex"; then
  record WARN l2-skill-same-content 0 "the Claude and Codex copies of SKILL.md differ"
fi

runcap l2-hook-status "$BIN" hook status
if [[ $SKIP_CLAUDE -eq 0 ]]; then
  sec="$(section 'Claude Code' "$RAW/l2-hook-status.out")"
  if [[ $LAST_RC -eq 0 ]] && printf '%s\n' "$sec" | grep -q 'installed: yes (UserPromptSubmit + SessionStart)' && printf '%s\n' "$sec" | grep -q 'matches current binary'; then
    record PASS l2-hook-status-claude "$LAST_SECS" "$(printf '%s\n' "$sec" | grep -E 'installed:|bin:' | one_line)"
  else record FAIL l2-hook-status-claude "$LAST_SECS" "rc=$LAST_RC $(printf '%s\n' "$sec" | one_line)"; fi
else record SKIP l2-hook-status-claude 0 "--skip-claude"; fi
if [[ $SKIP_CODEX -eq 0 ]]; then
  sec="$(section 'Codex' "$RAW/l2-hook-status.out")"
  if [[ $LAST_RC -eq 0 ]] && printf '%s\n' "$sec" | grep -q 'installed: yes (UserPromptSubmit + SessionStart)' && printf '%s\n' "$sec" | grep -q 'matches current binary' && printf '%s\n' "$sec" | grep -q 'trust: userPromptSubmit trusted'; then
    record PASS l2-hook-status-codex "$LAST_SECS" "$(printf '%s\n' "$sec" | grep -E 'installed:|trust:' | one_line)"
  else record FAIL l2-hook-status-codex "$LAST_SECS" "rc=$LAST_RC $(printf '%s\n' "$sec" | one_line)"; fi
else record SKIP l2-hook-status-codex 0 "--skip-codex"; fi

runcap l2-service-status "$BIN" service status
loaded="$(sed -nE 's/^[[:space:]]*loaded: (yes|no).*/\1/p' "$RAW/l2-service-status.out" | head -1)"
ev="$(grep -E '^[[:space:]]*(loaded|pid|state|plist):' "$RAW/l2-service-status.out" | one_line)"
if [[ $LAST_RC -eq 0 && "$loaded" == "yes" ]]; then record PASS l2-service-status "$LAST_SECS" "$ev"
elif [[ $LAST_RC -eq 0 && "$loaded" == "no" && $REQUIRE_WATCHER -eq 0 ]]; then record WARN l2-service-status "$LAST_SECS" "watcher not loaded (pass --require-watcher to fail on this): $ev"
else record FAIL l2-service-status "$LAST_SECS" "rc=$LAST_RC loaded=${loaded:-?} $ev"; fi

# three fixed recall dry runs against the live store
runcap l2-recall-content "$BIN" recall --format text --query "$CONTENT_PROMPT" --session "$SESSION" --cwd "$CWD" --verbose
cold_ms=$LAST_MS; cold_rc=$LAST_RC
logl="$(grep -E '^retrivio recall: [0-9TZ:-]+ [0-9a-f-]+ ' "$RAW/l2-recall-content.err" | tail -1 | sed 's/^retrivio recall: //')"
mode="$(printf '%s' "$logl" | awk '{print $3}')"; nleads="$(printf '%s' "$logl" | sed -nE 's/.*leads=([0-9]+).*/\1/p')"
if [[ $LAST_RC -eq 0 ]] && grep -q '<retrivio_leads>' "$RAW/l2-recall-content.out" && [[ "${nleads:-0}" -ge 1 ]]; then
  record PASS l2-recall-content "$LAST_SECS" "leads=$nleads mode=$mode wall=${cold_ms}ms; lead 1: $(grep -E '^1\. ' "$RAW/l2-recall-content.out" | cut -d' ' -f2 | one_line 200)"
else record FAIL l2-recall-content "$LAST_SECS" "rc=$LAST_RC log: ${logl:-none}; stdout: $(one_line 200 < "$RAW/l2-recall-content.out")"; fi
case "$mode" in
  semantic) record PASS l2-recall-semantic 0 "embedding backend answered ($logl)" ;;
  lexical) record WARN l2-recall-semantic 0 "lexical fallback: embedding backend unavailable or breaker open (files embed-breaker/embed-slow under the data dir's recall/): $logl" ;;
  *) record FAIL l2-recall-semantic 0 "no log line with a mode: ${logl:-none}" ;;
esac
check_block "$RAW/l2-recall-content.out" l2-recall-block-wellformed

runcap l2-recall-nr "$BIN" recall --format text --query "nr: $CONTENT_PROMPT" --session "$SESSION" --cwd "$CWD" --verbose
logl="$(grep -E '^retrivio recall: [0-9TZ:-]+ ' "$RAW/l2-recall-nr.err" | tail -1 | sed 's/^retrivio recall: //')"
if [[ $LAST_RC -eq 0 && ! -s "$RAW/l2-recall-nr.out" ]] && printf '%s' "$logl" | grep -q ' skipped:nr-prefix '; then
  record PASS l2-recall-nr "$LAST_SECS" "empty stdout; $logl"
else record FAIL l2-recall-nr "$LAST_SECS" "rc=$LAST_RC stdout bytes=$(wc -c < "$RAW/l2-recall-nr.out" | tr -d ' ') log: ${logl:-none}"; fi

runcap l2-recall-ack "$BIN" recall --format text --query "continue" --session "$SESSION" --cwd "$CWD" --verbose
logl="$(grep -E '^retrivio recall: [0-9TZ:-]+ ' "$RAW/l2-recall-ack.err" | tail -1 | sed 's/^retrivio recall: //')"
if [[ $LAST_RC -eq 0 && ! -s "$RAW/l2-recall-ack.out" ]] && printf '%s' "$logl" | grep -q ' skipped:ack '; then
  record PASS l2-recall-ack "$LAST_SECS" "empty stdout; $logl"
else record FAIL l2-recall-ack "$LAST_SECS" "rc=$LAST_RC stdout bytes=$(wc -c < "$RAW/l2-recall-ack.out" | tr -d ' ') log: ${logl:-none}"; fi

# latency: the hook has a 4 s hard deadline; the cold run above and a warm repeat must both fit
rv recall --reset-session --session "$SESSION" < /dev/null > /dev/null 2>&1
warm_ms=0; warm_rc=0; warm_list=""
for i in 1 2 3; do
  "$BIN" recall --reset-session --session "$SESSION" < /dev/null > /dev/null 2>&1
  runcap "l2-recall-warm$i" "$BIN" recall --format text --query "$CONTENT_PROMPT" --session "$SESSION" --cwd "$CWD" --verbose
  warm_list="$warm_list ${LAST_MS}ms"; [[ $LAST_MS -gt $warm_ms ]] && warm_ms=$LAST_MS; [[ $LAST_RC -ne 0 ]] && warm_rc=$LAST_RC
done
if [[ $cold_rc -eq 0 && $warm_rc -eq 0 && $cold_ms -lt 4000 && $warm_ms -lt 4000 ]]; then record PASS l2-recall-latency 0 "cold ${cold_ms}ms, warm max ${warm_ms}ms of$warm_list (budget 4000ms wall each)"
else record FAIL l2-recall-latency 0 "cold ${cold_ms}ms (rc=$cold_rc), warm max ${warm_ms}ms of$warm_list (rc=$warm_rc) (budget 4000ms wall)"; fi

runcap l2-search-json "$BIN" search --view files --limit 3 --json "$CONTENT_PROMPT"
if command -v python3 > /dev/null 2>&1; then
  shape="$(python3 - "$RAW/l2-search-json.out" <<'PY' 2>&1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("BAD stdout is not JSON (%s): %r" % (e.__class__.__name__, open(sys.argv[1]).read(120))); sys.exit(0)
need_top = ["query", "results", "timing_ms", "view"]
need_res = ["path", "score", "project_path", "content_date", "date_source", "age_days", "freshness_tier", "is_record"]
missing = [k for k in need_top if k not in d]
res = d.get("results", [])
if res:
    missing += ["results[0]." + k for k in need_res if k not in res[0]]
ok = not missing and len(res) >= 1
print(("OK" if ok else "BAD") + " results=%d timing_ms=%s view=%s missing=%s first=%s" % (len(res), d.get("timing_ms"), d.get("view"), missing, res[0].get("path") if res else None))
PY
)"
  if [[ $LAST_RC -eq 0 && "$shape" == OK* ]]; then record PASS l2-search-json "$LAST_SECS" "$shape"
  else record FAIL l2-search-json "$LAST_SECS" "rc=$LAST_RC $shape $(one_line 200 < "$RAW/l2-search-json.err")"; fi
else
  if [[ $LAST_RC -eq 0 ]] && grep -q '"results"' "$RAW/l2-search-json.out" && grep -q '"freshness_tier"' "$RAW/l2-search-json.out"; then
    record PASS l2-search-json "$LAST_SECS" "results and freshness_tier keys present (python3 missing, grep check only)"
  else record FAIL l2-search-json "$LAST_SECS" "rc=$LAST_RC $(one_line 200 < "$RAW/l2-search-json.out")"; fi
fi

# paths the CLI search returned for the content prompt; the MCP tool must return one of them
L2_SEARCH_PATHS="$( { command -v python3 > /dev/null 2>&1 && python3 -c 'import json,sys; [print(r["path"]) for r in json.load(open(sys.argv[1])).get("results",[])]' "$RAW/l2-search-json.out"; } 2>/dev/null || grep -oE '"path":"[^"]+"' "$RAW/l2-search-json.out" | sed 's/"path":"//; s/"$//')"
# dry run of the exact prompt the CLIs will send (instruction prefix included), fresh session, same cwd: the reference block
"$BIN" recall --reset-session --session "$SESSION" < /dev/null > /dev/null 2>&1
runcap l2-recall-hook-prompt "$BIN" recall --format text --query "$HOOK_PROMPT" --session "$SESSION" --cwd "$CWD" --verbose
L2_LEADS="$(lead_paths "$RAW/l2-recall-hook-prompt.out")"
leads_match() { # <cli> <reply file>: the leads the model reproduced must be the leads recall emitted for the same prompt and cwd
  local got; got="$(lead_paths "$2")"
  if [[ -n "$got" && "$got" == "$L2_LEADS" ]]; then record PASS "l3-$1-leads-match-dryrun" 0 "same lead path(s) as the L2 dry run: $(printf '%s' "$got" | one_line 200)"
  else record WARN "l3-$1-leads-match-dryrun" 0 "reply leads [$(printf '%s' "$got" | one_line 150)] differ from the dry run of the same prompt [$(printf '%s' "$L2_LEADS" | one_line 150)] (ranking drift between runs, or a block the model composed; compare raw/l2-recall-hook-prompt.out with the reply)"; fi
}
mcp_path_ok() { # <path>: a regular file that the CLI search also returned in its top 3 for the same query
  [[ -n "$1" && -f "$1" ]] && printf '%s\n' "$L2_SEARCH_PATHS" | grep -qxF "$1"
}

# ================================================================ L3: Claude Code
echo; echo "== L3 Claude Code"
if [[ $SKIP_CLAUDE -eq 1 ]]; then
  for c in hook-content block-wellformed hook-nr hook-ack mcp-search-files skill; do record SKIP "l3-claude-$c" 0 "--skip-claude"; done
elif ! command -v claude > /dev/null 2>&1; then
  for c in hook-content block-wellformed hook-nr hook-ack mcp-search-files skill; do record SKIP "l3-claude-$c" 0 "claude CLI not on PATH"; done
else
  CLAUDE_VER="$(claude --version 2>/dev/null | head -1)"
  claude_call() { # name sid prompt [extra args...]
    local name="$1" sid="$2" prompt="$3"; shift 3
    runcap "$name" claude -p "$prompt" --output-format text --session-id "$sid" --no-session-persistence "$@"
  }
  sid=$(new_uuid); h=$(sha1_8 "$sid")
  claude_call l3-claude-hook-content "$sid" "$HOOK_PROMPT" --max-turns 3
  logl="$(log_line_for "$h")"; nleads="$(printf '%s' "$logl" | sed -nE 's/.*leads=([0-9]+).*/\1/p')"
  if [[ $LAST_RC -eq 0 ]] && grep -q '<retrivio_leads>' "$RAW/l3-claude-hook-content.out" && [[ "${nleads:-0}" -ge 1 ]]; then
    record PASS l3-claude-hook-content "$LAST_SECS" "block in reply; log[$h]: $logl"
  else record FAIL l3-claude-hook-content "$LAST_SECS" "rc=$LAST_RC log[$h]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-claude-hook-content.out") $(one_line 200 < "$RAW/l3-claude-hook-content.err")"; fi
  check_block "$RAW/l3-claude-hook-content.out" l3-claude-block-wellformed
  leads_match claude "$RAW/l3-claude-hook-content.out"

  sid=$(new_uuid); h=$(sha1_8 "$sid")
  claude_call l3-claude-hook-nr "$sid" "nr: $HOOK_PROMPT" --max-turns 3
  logl="$(log_line_for "$h")"
  if [[ $LAST_RC -eq 0 ]] && ! grep -q '<retrivio_leads>' "$RAW/l3-claude-hook-nr.out" && printf '%s' "$logl" | grep -q ' skipped:nr-prefix '; then
    record PASS l3-claude-hook-nr "$LAST_SECS" "no block; reply: $(one_line 60 < "$RAW/l3-claude-hook-nr.out"); log[$h]: $logl"
  else record FAIL l3-claude-hook-nr "$LAST_SECS" "rc=$LAST_RC log[$h]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-claude-hook-nr.out")"; fi

  sid=$(new_uuid); h=$(sha1_8 "$sid")
  claude_call l3-claude-hook-ack "$sid" "continue" --max-turns 3 --tools ""
  logl="$(log_line_for "$h")"
  # judged on the hook evidence: the CLI may end with a max-turns error when the model wanders on a bare "continue"
  if ! grep -q '<retrivio_leads>' "$RAW/l3-claude-hook-ack.out" && printf '%s' "$logl" | grep -q ' skipped:ack '; then
    record PASS l3-claude-hook-ack "$LAST_SECS" "no block; log[$h]: $logl"
    [[ $LAST_RC -ne 0 ]] && record WARN l3-claude-hook-ack-cli 0 "claude exited $LAST_RC on the bare 'continue' prompt (hook evidence was fine): $(one_line 150 < "$RAW/l3-claude-hook-ack.out")"
  else record FAIL l3-claude-hook-ack "$LAST_SECS" "rc=$LAST_RC log[$h]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-claude-hook-ack.out")"; fi

  sid=$(new_uuid)
  claude_call l3-claude-mcp-search-files "$sid" "$MCP_PROMPT" --allowedTools "mcp__retrivio__search_files" --max-turns 4
  p="$(grep -m1 -E '^/' "$RAW/l3-claude-mcp-search-files.out" | sed 's/[[:space:]]*$//')"
  if [[ $LAST_RC -eq 0 ]] && mcp_path_ok "$p"; then record PASS l3-claude-mcp-search-files "$LAST_SECS" "first result is a file and is in the CLI search top 3 for the same query: $p"
  else record FAIL l3-claude-mcp-search-files "$LAST_SECS" "rc=$LAST_RC path='${p:-none}' (file: $([[ -f "$p" ]] && echo yes || echo no); in search top 3: $(printf '%s\n' "$L2_SEARCH_PATHS" | grep -qxF "$p" && echo yes || echo no)); reply: $(one_line 200 < "$RAW/l3-claude-mcp-search-files.out") $(grep -v awsCredentialExport "$RAW/l3-claude-mcp-search-files.err" | one_line 200)"; fi

  sid=$(new_uuid)
  claude_call l3-claude-skill "$sid" "$SKILL_PROMPT" --max-turns 4
  quoted="$(grep -m1 "$SKILL_EXPECT" "$RAW/l3-claude-skill.out" | sed -E 's/^[[:space:]"*]+//; s/[[:space:]"*]+$//')"
  if [[ $LAST_RC -eq 0 && -n "$quoted" ]] && grep -qF "$quoted" "$skill_claude"; then record PASS l3-claude-skill "$LAST_SECS" "quoted text is verbatim in $skill_claude: $(printf '%s' "$quoted" | one_line 120)"
  else record FAIL l3-claude-skill "$LAST_SECS" "rc=$LAST_RC reply: $(one_line 200 < "$RAW/l3-claude-skill.out")"; fi
fi

# ================================================================ L3: Codex
echo; echo "== L3 Codex"
if [[ $SKIP_CODEX -eq 1 ]]; then
  for c in hook-content block-wellformed hook-nr hook-ack mcp-search-files skill; do record SKIP "l3-codex-$c" 0 "--skip-codex"; done
elif ! command -v codex > /dev/null 2>&1; then
  for c in hook-content block-wellformed hook-nr hook-ack mcp-search-files skill; do record SKIP "l3-codex-$c" 0 "codex CLI not on PATH"; done
else
  CODEX_VER="$(codex --version 2>/dev/null | head -1)"
  codex_call() { # name prompt  (stdin must be /dev/null or codex exec hangs; runcap does that)
    local name="$1" prompt="$2"
    runcap "$name" codex exec --skip-git-repo-check --ephemeral -C "$CWD" -o "$RAW/$name.last" "$prompt"
    CODEX_SID="$(sed -nE 's/^session id: ([0-9a-f-]+).*/\1/p' "$RAW/$name.err" | head -1)"
    CODEX_H="$([[ -n "$CODEX_SID" ]] && sha1_8 "$CODEX_SID" || echo '-')"
    CODEX_HOOK_DONE=$(grep -c 'hook: UserPromptSubmit Completed' "$RAW/$name.err" | tr -d ' ')
  }
  codex_call l3-codex-hook-content "$HOOK_PROMPT"
  logl="$(log_line_for "$CODEX_H")"; nleads="$(printf '%s' "$logl" | sed -nE 's/.*leads=([0-9]+).*/\1/p')"
  if [[ $LAST_RC -eq 0 && $CODEX_HOOK_DONE -ge 1 ]] && grep -q '<retrivio_leads>' "$RAW/l3-codex-hook-content.last" && [[ "${nleads:-0}" -ge 1 ]]; then
    record PASS l3-codex-hook-content "$LAST_SECS" "hook: UserPromptSubmit Completed; block in reply; log[$CODEX_H]: $logl"
  else record FAIL l3-codex-hook-content "$LAST_SECS" "rc=$LAST_RC hook-completed=$CODEX_HOOK_DONE log[$CODEX_H]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-codex-hook-content.last" 2>/dev/null) $(grep -iE 'error' "$RAW/l3-codex-hook-content.err" | one_line 200)"; fi
  check_block "$RAW/l3-codex-hook-content.last" l3-codex-block-wellformed
  leads_match codex "$RAW/l3-codex-hook-content.last"

  codex_call l3-codex-hook-nr "nr: $HOOK_PROMPT"
  logl="$(log_line_for "$CODEX_H")"
  if [[ $LAST_RC -eq 0 ]] && ! grep -q '<retrivio_leads>' "$RAW/l3-codex-hook-nr.last" && printf '%s' "$logl" | grep -q ' skipped:nr-prefix '; then
    record PASS l3-codex-hook-nr "$LAST_SECS" "no block; reply: $(one_line 60 < "$RAW/l3-codex-hook-nr.last"); log[$CODEX_H]: $logl"
  else record FAIL l3-codex-hook-nr "$LAST_SECS" "rc=$LAST_RC log[$CODEX_H]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-codex-hook-nr.last" 2>/dev/null)"; fi

  codex_call l3-codex-hook-ack "continue"
  logl="$(log_line_for "$CODEX_H")"
  if [[ $LAST_RC -eq 0 && $CODEX_HOOK_DONE -ge 1 ]] && ! grep -q '<retrivio_leads>' "$RAW/l3-codex-hook-ack.last" && printf '%s' "$logl" | grep -q ' skipped:ack '; then
    record PASS l3-codex-hook-ack "$LAST_SECS" "hook ran, no block; log[$CODEX_H]: $logl"
  else record FAIL l3-codex-hook-ack "$LAST_SECS" "rc=$LAST_RC hook-completed=$CODEX_HOOK_DONE log[$CODEX_H]: ${logl:-none}; reply: $(one_line 200 < "$RAW/l3-codex-hook-ack.last" 2>/dev/null)"; fi

  codex_call l3-codex-mcp-search-files "$MCP_PROMPT"
  p="$(grep -m1 -E '^/' "$RAW/l3-codex-mcp-search-files.last" 2>/dev/null | sed 's/[[:space:]]*$//')"
  mcpl="$(grep -m1 -E 'mcp: retrivio/search_files \(completed\)' "$RAW/l3-codex-mcp-search-files.err")"
  if [[ $LAST_RC -eq 0 && -n "$mcpl" ]] && mcp_path_ok "$p"; then record PASS l3-codex-mcp-search-files "$LAST_SECS" "$mcpl; first result is a file and is in the CLI search top 3 for the same query: $p"
  else record FAIL l3-codex-mcp-search-files "$LAST_SECS" "rc=$LAST_RC mcp line: ${mcpl:-none}; reply: $(one_line 200 < "$RAW/l3-codex-mcp-search-files.last" 2>/dev/null) $(grep -iE 'error' "$RAW/l3-codex-mcp-search-files.err" | one_line 200)"; fi

  codex_call l3-codex-skill "$SKILL_PROMPT"
  quoted="$(grep -m1 "$SKILL_EXPECT" "$RAW/l3-codex-skill.last" 2>/dev/null | sed -E 's/^[[:space:]"*]+//; s/[[:space:]"*]+$//')"
  if [[ $LAST_RC -eq 0 && -n "$quoted" ]] && grep -qF "$quoted" "$skill_codex"; then record PASS l3-codex-skill "$LAST_SECS" "quoted text is verbatim in $skill_codex: $(printf '%s' "$quoted" | one_line 120)"
  else record FAIL l3-codex-skill "$LAST_SECS" "rc=$LAST_RC reply: $(one_line 200 < "$RAW/l3-codex-skill.last" 2>/dev/null)"; fi
fi

rv recall --reset-session --session "$SESSION" < /dev/null > /dev/null 2>&1

# ================================================================ report
{
  echo "# Retrivio harness check"
  echo
  echo "- started: $STARTED, finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- binary: \`$BIN\` ($("$BIN" version 2>/dev/null | head -1)), sha256 \`$(sha256_of "$BIN")\`"
  echo "- host: $(uname -sr); claude: ${CLAUDE_VER:-not run}; codex: ${CODEX_VER:-not run}"
  echo "- recall log: \`$RECALL_LOG\`; recall_max_leads: $MAX_LEADS; dry-run session: \`$SESSION\`; CLI timeout: ${TIMEOUT}s"
  echo "- content prompt: \"$CONTENT_PROMPT\""
  echo "- result: $pass PASS, $fail FAIL, $warn WARN, $skip SKIP$([[ $STRICT -eq 1 ]] && echo ' (--strict: WARN and SKIP counted as FAIL)')"
  echo "- this report contains absolute paths, lead excerpts and raw CLI output from this machine; keep it private"
  echo
  echo "| check | status | secs | evidence |"
  echo "|---|---|---|---|"
  while IFS=$'\t' read -r st name secs ev; do printf '| %s | %s | %s | %s |\n' "$name" "$st" "$secs" "$(printf '%s' "$ev" | sed 's/|/\\|/g')"; done < "$OUT/results.tsv"
  echo
  echo "## Raw outputs"
  echo
  for meta in "$RAW"/*.meta; do
    [[ -e "$meta" ]] || continue
    n="$(basename "$meta" .meta)"
    echo "### $n"; echo; echo '```'; cat "$meta"; echo '```'
    for ext in out last err; do
      f="$RAW/$n.$ext"; [[ -s "$f" ]] || continue
      echo; echo "$ext (first 60 lines):"; echo '```'; head -60 "$f" | cut -c1-400; echo '```'
    done
    echo
  done
} > "$OUT/report.md"

echo
echo "harness-check: $pass PASS, $fail FAIL, $warn WARN, $skip SKIP; report $OUT/report.md"
[[ $fail -eq 0 ]]
