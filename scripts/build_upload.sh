#!/bin/bash
# Package every file git would track (respecting .gitignore) as _upload/motion_physics_paper.zip,
# for creating the GitHub repository from another machine, and in batches for GitHub's browser upload.
#     bash scripts/build_upload.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD; T=$(mktemp -d); mkdir -p "$T/motion_physics_paper" _upload
git ls-files --cached --others --exclude-standard | while read -r f; do
    [ -f "$f" ] && ! git check-ignore -q --no-index "$f" && echo "$f"
done > "$T/files.txt"
rsync -a --files-from="$T/files.txt" ./ "$T/motion_physics_paper/"
rm -f _upload/motion_physics_paper.zip
(cd "$T" && zip -r -X -q "$ROOT/_upload/motion_physics_paper.zip" motion_physics_paper)
echo "wrote _upload/motion_physics_paper.zip: $(du -h _upload/motion_physics_paper.zip | cut -f1), $(wc -l < "$T/files.txt" | tr -d ' ') files"
# the same files in folders of at most 95, for GitHub's browser upload (100 files per upload, 25 MiB per file)
rm -rf _upload/batches
split -l 95 -a 1 "$T/files.txt" "$T/batch_"
n=0; for b in "$T"/batch_*; do n=$((n + 1)); rsync -a --files-from="$b" ./ "_upload/batches/$(printf %02d $n)/"; done
echo "wrote _upload/batches/01..$(printf %02d $n) for browser upload"
rm -rf "$T"
