#!/bin/bash
# Build paper/submission/supplementary_material.zip, the anonymized supplement: code, result tables, figure pages, both videos.
# Leaves out internal notes, run logs, cached arrays, per-sample predictions and licensed body-model or motion data.
#     bash scripts/build_supplement.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD; T=$(mktemp -d); D=$T/supplementary_material
mkdir -p "$D/videos" "$D/paper"
cp paper/submission/supplement_README.md "$D/README.md"
cp requirements.txt "$D/"
rsync -a --exclude __pycache__ --exclude .DS_Store --exclude '*.pyc' --exclude build_supplement.sh --exclude build_arxiv.sh --exclude build_upload.sh --exclude export_public.py --exclude demo_storyboard.py --exclude make_page_videos.py --exclude make_page_steps.py scripts tools notebooks prompts "$D/"
# the revised presentation is built by video_v2/ (its narration text, voice tests and drafts stay out)
mkdir -p "$D/video_v2" && cp video_v2/make_video.py video_v2/charts.py "$D/video_v2/"
rsync -a --exclude '*.npy' --exclude __pycache__ --exclude .DS_Store tests "$D/"
# figure pages: only what the paper's figures use (no palette studies, layout candidates or demo storyboard)
rsync -a --exclude .DS_Store --exclude 'palettes*.html' --exclude treatments.html --exclude 'fig_setup_candidates.html' \
      --exclude 'fig_setup_noise.html' --exclude 'demo_storyboard.*' --exclude study/ --exclude 'palette_study*.js' \
      --exclude treat_box.js --exclude seaborn_luts.json --exclude 'demo_*.png' --exclude 'treat_*.jpg' paper/figures_web "$D/paper/"
rsync -a --prune-empty-dirs --exclude '*_smoke/' --exclude 'archive_*/' --exclude 'preds/' --exclude run_log.txt \
      --include '*/' --include '*.csv' --include '*.txt' --include '*.json' --exclude '*' results "$D/"
cp paper/supplement/demo_video.mp4 paper/supplement/presentation.mp4 paper/supplement/presentation.srt "$D/videos/"
# anonymity guard: home paths, the account name, e-mail addresses, and pointers to the internal notes
if grep -r -I -l -i -E "/Users/|/home/|$USER|[a-z0-9._-]+@[a-z0-9-]+\.[a-z]{2,}" "$D" --exclude d3.min.js \
   || grep -r -I -l -E "HANDOFF|FINDINGS|RESEARCH_PLAN" "$D"; then
    echo "identifying or internal text in the files above; fix before zipping" >&2; rm -rf "$T"; exit 1
fi
rm -f paper/submission/supplementary_material.zip
(cd "$T" && zip -r -X -q "$ROOT/paper/submission/supplementary_material.zip" supplementary_material)
rm -rf "$T"
echo "wrote paper/submission/supplementary_material.zip: $(du -h paper/submission/supplementary_material.zip | cut -f1), $(unzip -l paper/submission/supplementary_material.zip | tail -1 | awk '{print $2}') files"
