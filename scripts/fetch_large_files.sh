#!/bin/bash
# Make a fresh checkout whole: fetch the public large files git does not track, check the licensed ones, and name the
# command that regenerates each generated result. Inventory, sources and licences: docs/LARGE_FILES.md.
#     bash scripts/fetch_large_files.sh                      fetch every missing public item, report the rest
#     bash scripts/fetch_large_files.sh --check              report only: present or missing, and how to get it
#     bash scripts/fetch_large_files.sh --only clip,intphys  only these items
# Safe to rerun: present items are skipped (checked by size or git commit), downloads resume and are moved into place
# only when their SHA-256 matches. Licensed items (SMPL, SMPL+H, AMASS) are never downloaded: put the archive where
# the steps say and rerun, and the script unpacks it. Generated results are never recomputed.
# Needs uv, git, curl and unzip; the Google Drive items need the venv item first (gdown).
set -uo pipefail
cd "$(dirname "$0")/.."
CHECK=0 ONLY="" FAILED=0 NAMES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1; shift;;
        --only) ONLY=${2:?"--only needs an item name"}; shift 2;;
        -h|--help) sed -n '2,10p' "$0"; exit 0;;
        *) echo "unknown option: $1 (options: --check, --only <name>[,<name>])" >&2; exit 2;;
    esac
done

say() { printf '%-8s %-19s %s\n' "$1" "$2" "$3"; }
bytes() { wc -c < "$1" | tr -d ' '; }
size_is() { [ -f "$1" ] && [ "$(bytes "$1")" = "$2" ]; }
count() { ls -d "$@" 2>/dev/null | wc -l | tr -d ' '; }
verify() { [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$2" ] || { echo "  SHA-256 mismatch: $1" >&2; return 1; }; }
has() {   # python module...: the environment exists and has these modules
    [ -x "$1" ] && "$1" -I -c 'import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in sys.argv[1:]))' "${@:2}" 2>/dev/null
}
at_commit() { [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" = "$2" ]; }
git_at() {   # dir url commit: exactly that commit, shallow (GitHub serves any commit by its full hash)
    mkdir -p "$1" && git -C "$1" init -q &&
        { git -C "$1" remote get-url origin >/dev/null 2>&1 || git -C "$1" remote add origin "$2"; } &&
        git -C "$1" fetch -q --depth 1 origin "$3" && git -C "$1" -c advice.detachedHead=false checkout -q FETCH_HEAD
}
fetch_url() {   # url dest [sha256]: resumable; the file appears at dest only once verified
    [ -f "$2" ] && return 0
    mkdir -p "$(dirname "$2")" && curl -fL --retry 3 --progress-bar -C - -o "$2.part" "$1" || return 1
    if [ -n "${3:-}" ] && ! verify "$2.part" "$3"; then rm -f "$2.part"; return 1; fi
    mv "$2.part" "$2"
}
drive_zip() {   # drive_id dest_dir: an official checkpoint zip from Google Drive, unpacked into dest_dir
    local z="$2/download_$1.zip"
    [ -x .venv/bin/gdown ] || { echo "  needs the venv item first (gdown)" >&2; return 1; }
    mkdir -p "$2"
    [ -f "$z" ] || .venv/bin/gdown "$1" -O "$z" || { echo "  Drive refused (often a quota limit): open" \
        "https://drive.google.com/file/d/$1 in a browser, save the zip as $z and rerun" >&2; return 1; }
    unzip -q -o "$z" -x '__MACOSX/*' -d "$2" && rm -f "$z"
}

# item NAME KIND WHERE TEST ACTION
#   auto:   ACTION fetches the item; TEST is run again afterwards
#   manual: ACTION prints the steps, or unpacks the archive once you have put it in place
#   regen:  ACTION is the command that regenerates the item (printed, never run)
item() {
    NAMES="$NAMES,$1"
    [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]] || return 0
    if eval "$4"; then say ok "$1" "$3"; return 0; fi
    case $2 in
        auto) if [ $CHECK = 1 ]; then say MISSING "$1" "$3 (this script fetches it)"
              elif say fetch "$1" "$3" && eval "$5" && eval "$4"; then say ok "$1" "$3"
              else say FAILED "$1" "$3"; FAILED=1; fi;;
        manual) say MANUAL "$1" "$3"; if eval "$5" && eval "$4"; then say ok "$1" "$3"; fi;;
        regen) say REGEN "$1" "$3"; echo "         command: $5";;
    esac
}
licensed() {   # archive sha256 unpack-command step...: steps until the archive is in place, then unpack it
    local a=$1 sha=$2 unpack=$3; shift 3
    if [ ! -f "$a" ]; then printf '         %s\n' "$@" "Save it as $a and rerun this script: it checks and unpacks it."; return 1; fi
    if [ $CHECK = 1 ]; then echo "         $a is in place: run without --check to unpack it"; return 1; fi
    verify "$a" "$sha" && eval "$unpack"
}

echo "ok: present | MISSING: fetched by this script | MANUAL: licensed, steps below | REGEN: generated, command below"

# Virtual environment (README's Setup); the Drive and Hugging Face items use it.
venv_get() {
    local req rc; req=$(mktemp) && grep -v '^clip ' requirements.txt > "$req" || return 1
    { [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv; } &&
        uv pip install -q --python .venv/bin/python -r "$req" &&
        uv pip install -q --python .venv/bin/python --no-build-isolation "$(grep '^clip ' requirements.txt)"
    rc=$?; rm -f "$req"; return $rc
}
item venv auto ".venv/" 'has .venv/bin/python torch clip gdown huggingface_hub' venv_get

# Third-party code at the commits the paper used.
MDM=ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778 HML=9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23 MD=aba848edd22133919ca96b67f4908399c67685b1
item mdm auto "vendor/mdm/" 'at_commit vendor/mdm $MDM' \
    'git_at vendor/mdm https://github.com/GuyTevet/motion-diffusion-model.git $MDM'
item humanml3d auto "vendor/HumanML3D/" 'at_commit vendor/HumanML3D $HML && [ -d vendor/HumanML3D/HumanML3D/texts ]' \
    'git_at vendor/HumanML3D https://github.com/EricGuo5513/HumanML3D.git $HML &&
     unzip -q -o vendor/HumanML3D/HumanML3D/texts.zip -d vendor/HumanML3D/HumanML3D'
item motiondiffuse auto "vendor/motiondiffuse/" 'at_commit vendor/motiondiffuse $MD' \
    'git_at vendor/motiondiffuse https://github.com/MotrixLab/MotionDiffuse.git $MD'

# Public model files and data.
mdm50_get() {
    local d=vendor/mdm/save/humanml_enc_512_50steps
    size_is $d/model000750000.pt 81818987 || drive_zip 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr vendor/mdm/save || return 1
    verify $d/model000750000.pt 0fbdc8547c8f262b8838645586790b55f983d90db3bb7ed58e4b5d49429587ca &&
        mkdir -p vendor/checkpoints && ln -sfn ../mdm/save/humanml_enc_512_50steps vendor/checkpoints/ckpt_50steps
}
item mdm-50steps auto "vendor/checkpoints/ckpt_50steps/" \
    'size_is vendor/checkpoints/ckpt_50steps/model000750000.pt 81818987' mdm50_get
C1000=vendor/checkpoints/ckpt_1000steps/humanml_trans_enc_512/model000475000.pt
item mdm-1000steps auto "vendor/checkpoints/ckpt_1000steps/" 'size_is $C1000 81806834' \
    'drive_zip 1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821 vendor/checkpoints/ckpt_1000steps &&
     verify $C1000 ce43c1d4232fc2b897999d7996c2d2b889791d99585d87e1289af773ab56ab93'
MDC=vendor/motiondiffuse/text2motion/checkpoints/t2m/t2m_motiondiffuse
MDU=https://huggingface.co/spaces/mingyuan/MotionDiffuse/resolve/09ca4087f432f026cdcd5866eeb03327e6fd1da7/checkpoints/t2m/t2m_motiondiffuse
item motiondiffuse-ckpt auto "$MDC/" 'size_is $MDC/model/latest.tar 953997124 && [ -f $MDC/opt.txt ] && [ -f $MDC/meta/std.npy ]' \
    'fetch_url $MDU/opt.txt $MDC/opt.txt &&
     fetch_url $MDU/meta/mean.npy $MDC/meta/mean.npy ebc7c543b4e27e886dba7e1bcde8fc0149f12a981586a548df98977b4b7c1a6a &&
     fetch_url $MDU/meta/std.npy $MDC/meta/std.npy 9a91111b5b44b2b6785cedf426ede6f8bc412ecb995914563e2810738de179e4 &&
     fetch_url $MDU/model/latest.tar $MDC/model/latest.tar 60a8b95aa5d190a95a6b3c20515301f72d70646b7a0d7f927c7167b968fa1222'
# the clip package downloads into ~/.cache/clip and checks the SHA-256 itself
item clip auto "~/.cache/clip/ViT-B-32.pt" 'size_is "$HOME/.cache/clip/ViT-B-32.pt" 353976522' \
    '.venv/bin/python -I -c "import clip; clip.load(\"ViT-B/32\", device=\"cpu\")" > /dev/null'
item hml-clip auto "tests/012314.npy" '[ -f tests/012314.npy ]' \
    'fetch_url https://raw.githubusercontent.com/EricGuo5513/HumanML3D/$HML/HumanML3D/new_joints/012314.npy tests/012314.npy \
     739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b'
WANU=https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/resolve/0fad780a534b6463e45facd96134c9f345acfa5b/vae
item wan-vae auto "vendor/wan_vae/vae/" \
    'size_is vendor/wan_vae/vae/diffusion_pytorch_model.safetensors 507591892 && [ -f vendor/wan_vae/vae/config.json ]' \
    'fetch_url $WANU/config.json vendor/wan_vae/vae/config.json &&
     fetch_url $WANU/diffusion_pytorch_model.safetensors vendor/wan_vae/vae/diffusion_pytorch_model.safetensors \
       d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793'
# 3.2 GB; the SHA-256 is of the archive whose MD5 matches the published 6a72715007f2f3b0e9546bfa8d8fc39b
item intphys auto "vendor/intphys/dev/" '[ "$(find vendor/intphys/dev -name status.json 2>/dev/null | wc -l)" -eq 360 ]' \
    'fetch_url https://download-intphys.cognitive-ml.fr/dev.tar.gz vendor/intphys/dev.tar.gz \
       441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a && tar -xzf vendor/intphys/dev.tar.gz -C vendor/intphys'
kokoro_get() {   # narration only: its own environment (Kokoro's pins clash with the project's), model from Hugging Face
    local k=vendor/kokoro-venv/bin/python p
    { [ -x $k ] || uv venv -q --python 3.12 vendor/kokoro-venv; } &&
        uv pip install -q --python $k "kokoro==0.9.4" "misaki[en]==0.9.4" "torch==2.15.0" "soundfile==0.14.0" \
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" &&
        p=$($k -I -c "from huggingface_hub import hf_hub_download as d
for f in ('config.json', 'voices/am_michael.pt'): d('hexgrad/Kokoro-82M', f)
print(d('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth'))") &&
        verify "$p" 496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4
}
item kokoro auto "vendor/kokoro-venv/ and the Kokoro-82M model" \
    'has vendor/kokoro-venv/bin/python kokoro soundfile en_core_web_sm &&
     compgen -G "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--hexgrad--Kokoro-82M/snapshots/*/kokoro-v1_0.pth" > /dev/null' kokoro_get

# Licensed: register, download yourself, put the archive in place.
SMPL_PKL=vendor/smpl_raw/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
item smpl manual "vendor/mdm/body_models/smpl/SMPL_NEUTRAL.pkl" '[ -s vendor/mdm/body_models/smpl/SMPL_NEUTRAL.pkl ]' \
    'licensed vendor/SMPL_python_v.1.1.0.zip 87c9e6cb1dddad79cf3b8b01760e240a9cd0c29d7ff14eda30fb07e3bd430c27 \
       "unzip -q -j -o vendor/SMPL_python_v.1.1.0.zip SMPL_python_v.1.1.0/smpl/models/$(basename $SMPL_PKL) -d vendor/smpl_raw &&
        .venv/bin/python scripts/convert_smpl.py $SMPL_PKL" \
       "1. Register at https://smpl.is.tue.mpg.de and accept the SMPL licence (non-commercial research)." \
       "2. Downloads: SMPL for Python, version 1.1.0 (female, male, neutral): SMPL_python_v.1.1.0.zip (331 MB)."'
item smplh manual "vendor/smplh/" '[ -f vendor/smplh/male/model.npz ] && [ -f vendor/smplh/female/model.npz ] && [ -f vendor/smplh/neutral/model.npz ]' \
    'licensed vendor/smplh.tar.xz 46d5b8687be48c91181fa88271feff3a5e83aa62a481fad8a0bcb9254b2a74f1 \
       "mkdir -p vendor/smplh && tar -xJf vendor/smplh.tar.xz -C vendor/smplh" \
       "1. Register at https://mano.is.tue.mpg.de and accept the MANO / SMPL+H licence (non-commercial research)." \
       "2. Downloads: \"Extended SMPL+H model (used in AMASS project)\": smplh.tar.xz (113 MB)."'
item amass manual "vendor/amass/BMLmovi/" '[ "$(find vendor/amass/BMLmovi -name "*.npz" 2>/dev/null | wc -l)" -ge 1887 ]' \
    'licensed vendor/amass/BMLmovi.tar.bz2 15a771effd6694383aa8f26b03d37a5b9a8a9ee0bbab702172db8cc5b3b98cf4 \
       "mkdir -p vendor/amass/BMLmovi && tar -xjf vendor/amass/BMLmovi.tar.bz2 -C vendor/amass/BMLmovi" \
       "1. Register at https://amass.is.tue.mpg.de and accept the AMASS licence (non-commercial research)." \
       "2. Downloads: BMLmovi, body data \"SMPL+H G\": BMLmovi.tar.bz2 (749 MB)."'

# Generated: the command is printed, never run.
runs_ok() { local d=$1; shift; for r; do [ -f "$d/$r/labels.npz" ] || return 1; done; }
item data-phase2 regen "data/phase2/ (16 GB)" 'runs_ok data/phase2 {50steps,1000steps,motiondiffuse}_seed{0,1,2}' \
    '.venv/bin/python scripts/run_phase2_local.py --seeds 0,1,2 (about 12 min), the same with --checkpoint 1000steps (about 47 min), .venv/bin/python scripts/run_motiondiffuse.py --tag phase2 --seeds 0,1,2'
item data-jump-heavy regen "data/jump_heavy/ (15 GB)" 'runs_ok data/jump_heavy {50steps,1000steps,motiondiffuse}_seed{0,1,2}' \
    '.venv/bin/python scripts/run_phase2_local.py --prompts-file prompts/jump_heavy.txt --tag jump_heavy --seeds 0,1,2 --generate-only, the same with --checkpoint 1000steps, .venv/bin/python scripts/run_motiondiffuse.py --tag jump_heavy --seeds 0,1,2'
item data-cond-token regen "data/cond_token/ (5.3 GB)" 'runs_ok data/cond_token 50steps_seed{0,1,2}' \
    '.venv/bin/python scripts/run_phase2_local.py --tag cond_token --seeds 0,1,2 --generate-only --record-cond-token'
item intphys-latents regen "data/intphys_vae/latents.npz (281 MB)" '[ -f data/intphys_vae/latents.npz ]' \
    '.venv/bin/python scripts/intphys_vae_encode.py (about 1 h; needs wan-vae and intphys)'
item data-sensitivity regen "data/sensitivity/ (132 MB)" '[ $(count data/sensitivity/*/) -ge 6 ]' \
    '.venv/bin/python scripts/label_sensitivity.py controls'
item preds regen "results/**/preds/ (470 MB)" \
    '[ -d results/phase2/fair/preds ] && [ -d results/jump_heavy/fair/preds ] && [ -d results/cond_token/fair/preds ] && [ -d results/phase2/controls/preds ]' \
    'the scripts/fair_readouts.py runs in README.md (each writes its preds/) and .venv/bin/python scripts/phase2_controls.py --save-preds'
item figure-caches regen "data/{demo_video,video_v2,smpl_web,presentation,smpl_teaser}/ (410 MB)" \
    '[ -d data/demo_video ] && [ -d data/video_v2 ] && [ -d data/smpl_web ] && [ -d data/presentation ] && [ -f data/smpl_teaser/fits.npz ]' \
    'scripts/make_demo_video.py prepare, render, compose; scripts/make_web_figures.py; scripts/fit_smpl_teaser.py; scripts/tts_kokoro.py then scripts/make_presentation.py; video_v2/make_video.py (need smpl; narration needs kokoro)'

if [ -n "$ONLY" ]; then
    for n in ${ONLY//,/ }; do
        [[ "$NAMES," == *",$n,"* ]] || { echo "unknown item: $n (items:${NAMES//,/ })" >&2; FAILED=1; }
    done
fi
exit $FAILED
