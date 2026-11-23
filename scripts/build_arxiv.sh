#!/bin/bash
# Build _upload/arxiv_source.zip: the paper's LaTeX source in de-anonymized preprint mode, for arXiv.
# Uses the author block in paper/submission/arxiv_authors.tex, ships the compiled bibliography (main.bbl) instead of
# the .bib, strips TeX comments (arXiv publishes the source) and test-compiles the result.
#     bash scripts/build_arxiv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD; T=$(mktemp -d); A=$T/arxiv; mkdir -p "$A"
mkdir -p "$T/bbl"                                                              # fresh main.bbl
(cd paper && tectonic -X compile main.tex --outdir "$T/bbl" --keep-intermediates >/dev/null 2>"$T/bbl.log") || { cat "$T/bbl.log" >&2; exit 1; }
cd paper
cp main.tex math_commands.tex tmlr.sty tmlr.bst fancyhdr.sty "$A/"
cp "$T/bbl/main.bbl" "$A/main.bbl"
mkdir -p "$A/sections" "$A/tables" "$A/figures/web"
cp sections/*.tex "$A/sections/"; cp tables/*.tex "$A/tables/"
grep -ho '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' main.tex sections/*.tex tables/*.tex | sed 's/.*{//; s/}$//' | sort -u \
    | while read -r f; do cp "$f" "$A/$f"; done
python3 - "$A" "$ROOT/paper/submission/arxiv_authors.tex" <<'PY'
import glob, re, sys
A, authors = sys.argv[1], sys.argv[2]
block = "".join(l for l in open(authors) if not l.lstrip().startswith("%")).strip()
m = open(f"{A}/main.tex").read()
for a, b in [("\\usepackage{tmlr}", "\\usepackage[preprint]{tmlr}"),
             ("\\author{\\name Anonymous Author(s) \\email anonymous@example.com}", block)]:
    if a in m:                                                 # an anonymous source is converted; a preprint one is kept
        m = m.replace(a, b)
assert "\\usepackage[preprint]{tmlr}" in m and "Anonymous Author" not in m, "main.tex is not in preprint mode"
open(f"{A}/main.tex", "w").write(m)
for f in [f"{A}/main.tex", f"{A}/math_commands.tex"] + glob.glob(f"{A}/sections/*.tex") + glob.glob(f"{A}/tables/*.tex"):
    out = []
    for line in open(f).read().split("\n"):
        if line.lstrip().startswith("%"):
            continue                                               # whole-line comment: drop it
        out.append(re.sub(r"(?<!\\)%.*$", "%", line))              # trailing comment: keep the % (it eats the newline)
    open(f, "w").write("\n".join(out))
PY
if grep -q "Author Name" "$A/main.tex"; then echo "warning: paper/submission/arxiv_authors.tex still has the placeholder author" >&2; fi
# test compile on a copy (tectonic runs bibtex, so the copy gets the .bib; arXiv itself uses main.bbl)
cp -R "$A" "$T/test"; cp references.bib "$T/test/"
(cd "$T/test" && tectonic -X compile main.tex --keep-intermediates >/dev/null 2>"$T/test.log") || { cat "$T/test.log" >&2; exit 1; }
cmp -s <(grep -v '^%' "$T/test/main.bbl") <(grep -v '^%' "$A/main.bbl") || echo "warning: main.bbl differs from a fresh bibtex run" >&2
pdfinfo "$T/test/main.pdf" | grep Pages
cp "$T/test/main.pdf" "$ROOT/_upload/arxiv_preview.pdf"
mkdir -p "$ROOT/_upload"; rm -f "$ROOT/_upload/arxiv_source.zip"
(cd "$A" && zip -r -X -q "$ROOT/_upload/arxiv_source.zip" .)
echo "wrote _upload/arxiv_source.zip: $(du -h "$ROOT/_upload/arxiv_source.zip" | cut -f1), $(unzip -l "$ROOT/_upload/arxiv_source.zip" | tail -1 | awk '{print $2}') files; preview _upload/arxiv_preview.pdf"
rm -rf "$T"
