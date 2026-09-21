#!/usr/bin/env bash
# Offline smoke test of the Retrivio indexer (layer L1 in docs/TESTING.md).
#
# Needs no credentials and no network: it uses the `hash` embedding backend on a
# synthetic corpus it generates itself, in an isolated data dir. It builds nothing;
# point RETRIVIO_BIN at the binary under test.
#
#   RETRIVIO_BIN=target/release/retrivio scripts/smoke.sh [work_dir]
#
#   RETRIVIO_BIN   binary under test (default: <repo>/target/release/retrivio)
#   work_dir       scratch directory, recreated on every run (default: <repo>/tmp/smoke);
#                  never under /tmp
#
# Every step prints PASS/FAIL/SKIP with the evidence line it was judged on, the raw
# output of each command is kept under <work_dir>/logs/, and the exit status is 1 when
# any step failed. See docs/TESTING.md for what each assertion protects.
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${RETRIVIO_BIN:-$repo_dir/target/release/retrivio}"
WORK="${1:-$repo_dir/tmp/smoke}"

case "$WORK" in
  /tmp/*|/private/tmp/*|/tmp) echo "smoke: refusing to use $WORK (never /tmp); pass a directory inside the project" >&2; exit 2 ;;
esac
if [[ ! -x "$BIN" ]]; then
  echo "smoke: binary not executable: $BIN (build with: cargo build --release -p retrivio, or set RETRIVIO_BIN)" >&2
  exit 2
fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

marker=".retrivio-smoke-workdir"
if [[ -e "$WORK" && ! -e "$WORK/$marker" ]]; then
  echo "smoke: $WORK exists and was not created by this script; refusing to delete it" >&2
  exit 2
fi
rm -rf "$WORK"
mkdir -p "$WORK/corpus" "$WORK/data" "$WORK/logs"
: > "$WORK/$marker"
WORK="$(cd "$WORK" && pwd)"
CORPUS="$WORK/corpus"
LOGS="$WORK/logs"

pass=0; fail=0; skip=0
report() { # status name evidence
  case "$1" in PASS) pass=$((pass+1));; FAIL) fail=$((fail+1));; SKIP) skip=$((skip+1));; esac
  printf '%-4s %-28s %s\n' "$1" "$2" "$3"
}
run() { # name -- command...; captures stdout+stderr to logs/<name>.log, returns rc
  local name="$1"; shift
  "$@" < /dev/null > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "$rc" > "$LOGS/$name.rc"
  return $rc
}
rv() { "$BIN" --data-dir "$WORK/data" --config "$WORK/config.toml" "$@"; }

# Counter extractors for the `index` / `refresh` summary block and the `prune` summary.
n_projects_found()   { sed -nE 's/^projects found: ([0-9]+).*/\1/p' "$1" | tail -1; }
n_projects_updated() { sed -nE 's/^projects updated: ([0-9]+).*/\1/p' "$1" | tail -1; }
n_projects_skipped() { sed -nE 's/^projects skipped \(unchanged\): ([0-9]+).*/\1/p' "$1" | tail -1; }
files_line()         { sed -nE 's/^files selected: ([0-9]+) \(unchanged ([0-9]+), rechunked ([0-9]+)\).*/\1 \2 \3/p' "$1" | tail -1; }
chunks_line()        { sed -nE 's/^chunks embedded: ([0-9]+), reused: ([0-9]+), deleted: ([0-9]+).*/\1 \2 \3/p' "$1" | tail -1; }
n_chunks_indexed()   { sed -nE 's/^chunks indexed: ([0-9]+).*/\1/p' "$1" | tail -1; }
pruned_line()        { sed -nE 's/^stale chunks pruned: ([0-9]+) \(from ([0-9]+) files\).*/\1 \2/p' "$1" | tail -1; }
docs_line()          { sed -nE 's/^documents extracted: ([0-9]+) \(failed: ([0-9]+)\).*/\1 \2/p' "$1" | tail -1; }
indexed_projects()   { sed -nE 's/^\[[0-9]+\/[0-9]+\] index (.+) files=.*/\1/p' "$1" | tr '\n' ',' | sed 's/,$//'; }
prune_chunks()       { sed -nE 's/^ *chunks (would prune|pruned): ([0-9]+) \(from ([0-9]+) files\).*/\2 \3/p' "$1" | tail -1; }
summary_line()       { grep -E '^(files selected|chunks embedded|stale chunks pruned):' "$1" | tr '\n' ';' | sed 's/;$//'; }

json_paths() { # file -> one path per line, via python3 or jq, else a grep fallback
  if command -v python3 > /dev/null 2>&1; then
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(r["path"]) for r in d["results"]]' "$1" 2>/dev/null
  elif command -v jq > /dev/null 2>&1; then jq -r '.results[].path' "$1" 2>/dev/null
  else grep -oE '"path":"[^"]+"' "$1" | sed 's/"path":"//; s/"$//'; fi
}
search_check() { # name query expected-suffix
  local name="$1" query="$2" want="$3" paths
  run "$name" rv search --view files --limit 5 --json "$query"; rc=$?
  paths="$(json_paths "$LOGS/$name.log")"
  if [[ $rc -eq 0 && -n "$paths" ]] && printf '%s\n' "$paths" | grep -qE "$want\$"; then
    report PASS "$name" "\"$query\" -> $(printf '%s\n' "$paths" | grep -E "$want\$" | head -1) (rank $(printf '%s\n' "$paths" | grep -nE "$want\$" | head -1 | cut -d: -f1) of $(printf '%s\n' "$paths" | wc -l | tr -d ' '))"
  else
    report FAIL "$name" "rc=$rc \"$query\" -> expected *$want in: $(printf '%s' "$paths" | tr '\n' ' ' | cut -c1-300)"
  fi
}
search_absent() { # name query forbidden-suffix: the query must run and the file must not be among the results
  local name="$1" query="$2" bad="$3" paths
  run "$name" rv search --view files --limit 10 --json "$query"; rc=$?
  paths="$(json_paths "$LOGS/$name.log")"
  if [[ $rc -eq 0 ]] && ! printf '%s\n' "$paths" | grep -qE "$bad\$"; then
    report PASS "$name" "\"$query\" -> $(printf '%s\n' "$paths" | grep -c . | tr -d ' ') results, none is *$bad"
  else
    report FAIL "$name" "rc=$rc \"$query\" -> *$bad still returned: $(printf '%s' "$paths" | tr '\n' ' ' | cut -c1-300)"
  fi
}

echo "smoke: binary $BIN"
echo "smoke: work dir $WORK"

# ---------------------------------------------------------------- corpus
mkdir -p "$CORPUS/alpha-notes" "$CORPUS/beta-tool/nested/deep"
cat > "$CORPUS/alpha-notes/DECISIONS.md" <<'EOF'
# Alpha notes

Decision 2026-09-01: the zebra migration uses the quokka bridge.
Follow-up: measure the pangolin cache before the cutover.
EOF
cat > "$CORPUS/beta-tool/bridge.py" <<'EOF'
def quokka_bridge(x):
    """Bridge helper for the zebra migration."""
    return x * 2
EOF
cat > "$CORPUS/beta-tool/config.yaml" <<'EOF'
service: beta-tool
region: us-west-2
keys:
  - wombat
EOF
echo "Deep nested note about the pangolin cache warmup." > "$CORPUS/beta-tool/nested/deep/README.md"
echo "Root level file about the axolotl ledger." > "$CORPUS/ROOT-NOTE.md"

have_docs=0
if command -v python3 > /dev/null 2>&1; then
  mkdir -p "$CORPUS/gamma-docs"
  if python3 - "$CORPUS/gamma-docs" <<'PY' > "$LOGS/make-docs.log" 2>&1
import sys, zipfile, os
out = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PK = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"

def docx(path, title, paragraphs):
    body = "".join('<w:p><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % p for p in paragraphs)
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="%s"><w:body>%s<w:sectPr/></w:body></w:document>' % (W, body))
    core = ('<cp:coreProperties xmlns:cp="x" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:title>%s</dc:title></cp:coreProperties>' % title)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<Types xmlns="%s"/>' % CT)
        z.writestr("_rels/.rels", '<Relationships xmlns="%s"/>' % PK)
        z.writestr("word/document.xml", doc)
        z.writestr("docProps/core.xml", core)

def pptx(path, slides, notes):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<Types xmlns="%s"/>' % CT)
        z.writestr("_rels/.rels", '<Relationships xmlns="%s"/>' % PK)
        for i, paragraphs in enumerate(slides, 1):
            body = "".join('<a:p><a:r><a:rPr lang="en-US"/><a:t>%s</a:t></a:r></a:p>' % p for p in paragraphs)
            z.writestr("ppt/slides/slide%d.xml" % i,
                       '<p:sld xmlns:a="%s" xmlns:p="%s"><p:cSld><p:spTree><p:sp><p:txBody>%s'
                       '</p:txBody></p:sp></p:spTree></p:cSld></p:sld>' % (A, P, body))
            note = notes.get(i)
            if note:
                z.writestr("ppt/slides/_rels/slide%d.xml.rels" % i,
                           '<Relationships xmlns="%s"><Relationship Id="rId2" Type="%s/notesSlide" '
                           'Target="../notesSlides/notesSlide%d.xml"/></Relationships>' % (PK, R, i))
                z.writestr("ppt/notesSlides/notesSlide%d.xml" % i,
                           '<p:notes xmlns:a="%s" xmlns:p="%s"><p:cSld><p:spTree><p:sp><p:txBody>'
                           '<a:p><a:r><a:t>%s</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>' % (A, P, note))

docx(os.path.join(out, "budget.docx"), "Budget report",
     ["Budget report for the enclosure programme.", "The giraffe enclosure roof replacement costs 42 units."])
pptx(os.path.join(out, "deck.pptx"),
     [["Quarterly review", "Status of the enclosure programme."], ["The capybara pond needs a new filter."]],
     {1: "Speaker note: mention the okapi fence repair."})
print("wrote budget.docx and deck.pptx")
PY
  then have_docs=1; report PASS "corpus-documents" "$(cat "$LOGS/make-docs.log")"
  else rm -rf "$CORPUS/gamma-docs"; report SKIP "corpus-documents" "python3 zipfile generation failed; see $LOGS/make-docs.log"; fi
else
  report SKIP "corpus-documents" "python3 not found; docx/pptx fixtures not generated"
fi

# Expected project count: alpha-notes, beta-tool, root files (+ gamma-docs); expected file count.
exp_projects=3; exp_files=5
if [[ $have_docs -eq 1 ]]; then exp_projects=4; exp_files=7; fi

cat > "$WORK/config.toml" <<EOF
root = "$CORPUS"
embed_backend = "hash"
local_embed_dim = 128
retrieval_backend = "lancedb"
reranker_enabled = false
hyde_enabled = false
EOF

# ---------------------------------------------------------------- steps
if run version rv version && grep -qE '^retrivio [0-9]+\.[0-9]+\.[0-9]+' "$LOGS/version.log"; then
  report PASS "version" "$(head -1 "$LOGS/version.log")"
else report FAIL "version" "rc=$(cat "$LOGS/version.rc") $(head -1 "$LOGS/version.log")"; fi

if run add rv add "$CORPUS" --no-refresh && grep -q "$CORPUS" "$LOGS/add.log"; then
  report PASS "add-root" "$(grep -E '^- ' "$LOGS/add.log" | head -1)"
else report FAIL "add-root" "rc=$(cat "$LOGS/add.rc") $(head -2 "$LOGS/add.log" | tr '\n' ' ')"; fi

# 1. first index: every file read, every chunk embedded, nothing reused
run index1 rv index; rc=$?
pf=$(n_projects_found "$LOGS/index1.log"); read -r fs fu fr <<< "$(files_line "$LOGS/index1.log")"
read -r ce cr cd <<< "$(chunks_line "$LOGS/index1.log")"; ci=$(n_chunks_indexed "$LOGS/index1.log")
read -r de df <<< "$(docs_line "$LOGS/index1.log")"
ev="projects found: ${pf:-?}; files selected: ${fs:-?} (unchanged ${fu:-?}, rechunked ${fr:-?}); chunks indexed: ${ci:-?}; chunks embedded: ${ce:-?}, reused: ${cr:-?}, deleted: ${cd:-?}; documents extracted: ${de:-?} (failed: ${df:-?})"
if [[ $rc -eq 0 && "${pf:-x}" == "$exp_projects" && "${fs:-x}" == "$exp_files" && "${fu:-x}" == 0 && "${fr:-x}" == "$exp_files" \
      && "${ce:-0}" -gt 0 && "${ce:-x}" == "${ci:-y}" && "${cr:-x}" == 0 && "${cd:-x}" == 0 ]]; then
  report PASS "index-first" "$ev (expected $exp_projects projects, $exp_files files)"
else report FAIL "index-first" "rc=$rc $ev (expected $exp_projects projects, $exp_files files)"; fi
first_chunks="${ci:-0}"

if [[ $have_docs -eq 1 ]]; then
  if [[ "${de:-x}" == 2 && "${df:-x}" == 0 ]]; then report PASS "index-documents" "documents extracted: $de (failed: $df)"
  else report FAIL "index-documents" "documents extracted: ${de:-?} (failed: ${df:-?}); expected 2 (failed: 0)"; fi
fi

# 2. second index with nothing changed: every project skipped, nothing embedded
run index2 rv index; rc=$?
pf2=$(n_projects_found "$LOGS/index2.log"); ps2=$(n_projects_skipped "$LOGS/index2.log")
read -r fs fu fr <<< "$(files_line "$LOGS/index2.log")"; read -r ce cr cd <<< "$(chunks_line "$LOGS/index2.log")"
ev="projects skipped (unchanged): ${ps2:-?} of ${pf2:-?} (expected $exp_projects); files selected: ${fs:-?}; chunks embedded: ${ce:-?}, reused: ${cr:-?}, deleted: ${cd:-?}"
if [[ $rc -eq 0 && "${pf2:-x}" == "$exp_projects" && "${ps2:-x}" == "${pf2:-y}" && "${fs:-x}" == 0 && "${ce:-x}" == 0 && "${cd:-x}" == 0 ]]; then
  report PASS "index-steady" "$ev"
else report FAIL "index-steady" "rc=$rc $ev"; fi

# 3. forced refresh: every file re-read, every stored vector reused, nothing embedded
run refresh rv refresh; rc=$?
read -r fs fu fr <<< "$(files_line "$LOGS/refresh.log")"; read -r ce cr cd <<< "$(chunks_line "$LOGS/refresh.log")"
ev="files selected: ${fs:-?} (unchanged ${fu:-?}, rechunked ${fr:-?}); chunks embedded: ${ce:-?}, reused: ${cr:-?}, deleted: ${cd:-?}"
if [[ $rc -eq 0 && "${fr:-x}" == "$exp_files" && "${ce:-x}" == 0 && "${cr:-x}" == "$first_chunks" && "${cd:-x}" == 0 ]]; then
  report PASS "refresh-reuses-vectors" "$ev (all $first_chunks chunks reused)"
else report FAIL "refresh-reuses-vectors" "rc=$rc $ev (expected reused $first_chunks, embedded 0)"; fi

search_check "search-python" "quokka_bridge helper" "beta-tool/bridge.py"
search_check "search-yaml" "wombat" "beta-tool/config.yaml"

# 4. edit one file: only that file is rechunked and only its chunk is embedded
cat > "$CORPUS/beta-tool/bridge.py" <<'EOF'
def quokka_bridge(x):
    """Bridge helper for the zebra migration, revised."""
    return x * 3
EOF
run index-edit rv index; rc=$?
read -r fs fu fr <<< "$(files_line "$LOGS/index-edit.log")"; read -r ce cr cd <<< "$(chunks_line "$LOGS/index-edit.log")"
pu=$(n_projects_updated "$LOGS/index-edit.log"); ip=$(indexed_projects "$LOGS/index-edit.log")
ev="projects updated: ${pu:-?} [$ip]; files selected: ${fs:-?} (unchanged ${fu:-?}, rechunked ${fr:-?}); chunks embedded: ${ce:-?}, reused: ${cr:-?}, deleted: ${cd:-?}"
if [[ $rc -eq 0 && "${pu:-x}" == 1 && "$ip" == "beta-tool" && "${fs:-x}" == 3 && "${fu:-x}" == 2 && "${fr:-x}" == 1 && "${ce:-x}" == 1 && "${cd:-x}" == 0 ]]; then
  report PASS "index-edit-one-file" "$ev"
else report FAIL "index-edit-one-file" "rc=$rc $ev (expected beta-tool only, files 3 (unchanged 2, rechunked 1), embedded 1)"; fi
search_check "search-edited-text" "revised bridge helper" "beta-tool/bridge.py"

# 5. delete one file: its chunk is deleted and pruned on the next incremental run
rm -f "$CORPUS/beta-tool/config.yaml"
run index-delete rv index; rc=$?
read -r ce cr cd <<< "$(chunks_line "$LOGS/index-delete.log")"; read -r pc pfl <<< "$(pruned_line "$LOGS/index-delete.log")"
ev="chunks embedded: ${ce:-?}, reused: ${cr:-?}, deleted: ${cd:-?}; stale chunks pruned: ${pc:-?} (from ${pfl:-?} files)"
if [[ $rc -eq 0 && "${ce:-x}" == 0 && "${cd:-x}" == 1 && "${pc:-x}" == 1 && "${pfl:-x}" == 1 ]]; then
  report PASS "index-delete-one-file" "$ev"
else report FAIL "index-delete-one-file" "rc=$rc $ev (expected deleted 1, pruned 1 from 1 file)"; fi
search_absent "search-deleted-absent" "wombat" "beta-tool/config.yaml"

# 6. search: JSON payload shape and the expected file
search_check "search-markdown" "quokka bridge zebra migration" "alpha-notes/DECISIONS.md"
search_check "search-root-file" "axolotl ledger" "ROOT-NOTE.md"
search_check "search-nested-dir" "pangolin cache warmup" "nested/deep/README.md"
if [[ $have_docs -eq 1 ]]; then
  search_check "search-docx-text" "giraffe enclosure roof replacement" "gamma-docs/budget.docx"
  search_check "search-pptx-notes" "okapi fence repair" "gamma-docs/deck.pptx"
fi
if command -v python3 > /dev/null 2>&1; then
  keys="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d["results"][0]; print(",".join(k for k in ("path","score","content_date","age_days","freshness_tier","date_source","project_path") if k in r)); print("query" in d and "timing_ms" in d and "results" in d and "view" in d)' "$LOGS/search-markdown.log" 2>/dev/null | tr '\n' ' ')"
  if [[ "$keys" == *"path,score,content_date,age_days,freshness_tier,date_source,project_path True"* ]]; then
    report PASS "search-json-shape" "top keys query/results/timing_ms/view; result keys $keys"
  else report FAIL "search-json-shape" "missing keys; got: $keys"; fi
fi

# 7. recall dry run against the smoke store (hook mode, no CLI involved)
run recall-reset rv recall --reset-session --session smoke
run recall rv recall --format text --query "zebra migration quokka bridge decision" --session smoke --cwd "$CORPUS" --verbose; rc=$?
logline="$(grep -E 'retrivio recall: [0-9T:Z-]+ [0-9a-f-]+ (semantic|lexical|skipped:[a-z-]+) [0-9]+ms cand=[0-9]+ leads=[0-9]+' "$LOGS/recall.log" | tail -1 | sed 's/^retrivio recall: //')"
if [[ $rc -eq 0 ]] && grep -q '^<retrivio_leads>$' "$LOGS/recall.log" && grep -q '^</retrivio_leads>$' "$LOGS/recall.log" \
   && grep -qE '^1\. .*alpha-notes/DECISIONS\.md ' "$LOGS/recall.log" && [[ -s "$WORK/data/recall.log" ]]; then
  report PASS "recall-dry-run" "block with lead 1 = alpha-notes/DECISIONS.md; log: $logline"
else report FAIL "recall-dry-run" "rc=$rc log: ${logline:-none}; stdout: $(grep -v '^retrivio recall:' "$LOGS/recall.log" | head -3 | tr '\n' ' ' | cut -c1-300)"; fi
run recall-reset2 rv recall --reset-session --session smoke

# 8. prune: dry run reports the stale chunk, the real run removes it and compacts LanceDB
rm -f "$CORPUS/beta-tool/nested/deep/README.md"
run prune-dry rv prune --dry-run; rc=$?
read -r wc wf <<< "$(prune_chunks "$LOGS/prune-dry.log")"
if [[ $rc -eq 0 && "${wc:-x}" == 1 && "${wf:-x}" == 1 ]] && grep -q 'dry run, nothing written' "$LOGS/prune-dry.log"; then
  report PASS "prune-dry-run" "chunks would prune: $wc (from $wf files); nothing written"
else report FAIL "prune-dry-run" "rc=$rc $(grep -E 'chunks would prune|error' "$LOGS/prune-dry.log" | head -2 | tr '\n' ' ')"; fi
run prune rv prune; rc=$?
read -r pc pfl <<< "$(prune_chunks "$LOGS/prune.log")"; compact="$(grep -E '^ *lancedb compacted:' "$LOGS/prune.log" | head -1 | sed 's/^ *//')"
if [[ $rc -eq 0 && "${pc:-x}" == 1 && "${pfl:-x}" == 1 && -n "$compact" ]]; then
  report PASS "prune" "chunks pruned: $pc (from $pfl files); $compact"
else report FAIL "prune" "rc=$rc pruned=${pc:-?} from ${pfl:-?}; compaction line: ${compact:-missing}"; fi

# 9. after prune the next incremental run may re-collect the pruned project once (prune does not
#    advance the change-gate signature) but must read, embed and delete nothing; the run after
#    that must skip every project.
run index-after-prune rv index; rc=$?
read -r fs fu fr <<< "$(files_line "$LOGS/index-after-prune.log")"; read -r ce cr cd <<< "$(chunks_line "$LOGS/index-after-prune.log")"; pfa=$(n_projects_found "$LOGS/index-after-prune.log")
if [[ $rc -eq 0 && "${pfa:-x}" == "$exp_projects" && "${fr:-x}" == 0 && "${ce:-x}" == 0 && "${cd:-x}" == 0 ]]; then
  report PASS "index-after-prune" "projects found: $pfa; files selected: ${fs:-?} (unchanged ${fu:-?}, rechunked $fr); chunks embedded: $ce, reused: $cr, deleted: $cd"
else report FAIL "index-after-prune" "rc=$rc $(summary_line "$LOGS/index-after-prune.log") (expected rechunked 0, embedded 0, deleted 0)"; fi
run index-final rv index; rc=$?
pf3=$(n_projects_found "$LOGS/index-final.log"); ps3=$(n_projects_skipped "$LOGS/index-final.log"); read -r ce cr cd <<< "$(chunks_line "$LOGS/index-final.log")"
if [[ $rc -eq 0 && "${pf3:-x}" == "$exp_projects" && "${ps3:-x}" == "${pf3:-y}" && "${ce:-x}" == 0 && "${cd:-x}" == 0 ]]; then
  report PASS "index-final-steady" "projects skipped (unchanged): $ps3 of $pf3 (expected $exp_projects); chunks embedded: $ce, deleted: $cd"
else report FAIL "index-final-steady" "rc=$rc projects skipped ${ps3:-?} of ${pf3:-?}; $(summary_line "$LOGS/index-final.log")"; fi

echo
echo "smoke: $pass passed, $fail failed, $skip skipped; logs in $LOGS"
[[ $fail -eq 0 ]]
