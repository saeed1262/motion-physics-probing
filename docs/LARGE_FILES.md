# Large files

About 55 GB of this project lives outside git: the environment, downloaded code and models, licensed body models
and motion capture, and cached generator runs. This page lists each item, where it comes from and what needs it.
Everything git tracks (code, the paper, result tables and figures, the videos, the project page) is about 240 MB,
with no file over 30 MB.

To make a fresh checkout whole:

```
bash scripts/fetch_large_files.sh --check       # what is present, what is missing, how to get it
bash scripts/fetch_large_files.sh               # fetch every missing public item (sizes below)
bash scripts/fetch_large_files.sh --only mdm-50steps,clip     # some items only
```

The script is safe to rerun: it skips what is present (checked by size or git commit), resumes interrupted
downloads and moves a file into place only after its SHA-256 matches. It never downloads licensed data and never
recomputes results; for those it prints the steps or the command. The `--only` names are in the first column below.
Sizes are as measured on November 22, 2026. "Pipeline" means needed to rerun the analyses; "figures" means needed
only to rebuild figures or videos. Paper 2 (`../humanoid_executability`) reads several of these items; they are
marked "paper 2".

## 1. Licensed data you must download yourself

Each needs a free account and the provider's licence (non-commercial research). Put the archive at the path shown
and rerun the script: it checks the SHA-256 and unpacks it. Nothing from these is redistributed.

| `--only` | Path and size | What it is | Where to get it | Needed for |
|---|---|---|---|---|
| `smpl` | `vendor/SMPL_python_v.1.1.0.zip` (331 MB), unpacked to `vendor/smpl_raw/` (236 MB), converted to `vendor/mdm/body_models/smpl/SMPL_NEUTRAL.pkl` (87 MB) | SMPL body model v1.1.0; the neutral model is converted by `scripts/convert_smpl.py` (the script runs it) | smpl.is.tue.mpg.de, Downloads, SMPL for Python, version 1.1.0 (female, male, neutral): `SMPL_python_v.1.1.0.zip` | Figures (Figure 1, body images, demo video). Paper 2: pipeline (its SMPL fits) |
| `smplh` | `vendor/smplh.tar.xz` (113 MB), unpacked to `vendor/smplh/` (250 MB) | Extended SMPL+H model, as used by AMASS | mano.is.tue.mpg.de, Downloads, "Extended SMPL+H model (used in AMASS project)": `smplh.tar.xz` | Pipeline (real-motion label check). Paper 2: real-motion control |
| `amass` | `vendor/amass/BMLmovi.tar.bz2` (749 MB), unpacked to `vendor/amass/BMLmovi/` (1.5 GB) | AMASS BMLmovi motion capture, SMPL+H G body data, 1,887 files | amass.is.tue.mpg.de, Downloads, BMLmovi, "SMPL+H G": `BMLmovi.tar.bz2` | Pipeline (`results/amass/`). Paper 2: real-motion control |
| none | not present | HumanML3D's processed motion set (derived from AMASS) | HumanML3D's README | Not needed: only one processed clip is used (`hml-clip` below), and it is public |

## 2. Public downloads the script fetches

| `--only` | Path and size | What it is | Source | Licence | Needed for |
|---|---|---|---|---|---|
| `mdm-50steps` | `vendor/mdm/save/humanml_enc_512_50steps/` (215 MB), linked as `vendor/checkpoints/ckpt_50steps` | MDM HumanML3D checkpoint, 50 diffusion steps | Google Drive `1cfadR1eZ116TIdXK7qDX1RugAerEiJXr` (`humanml_enc_512_50steps.zip`, 197 MB; id from MDM's README), with `gdown` | MIT | Pipeline; paper 2 and 3 |
| `mdm-1000steps` | `vendor/checkpoints/ckpt_1000steps/` (429 MB) | MDM checkpoint, 1000 steps (the original MDM) | Google Drive `1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821` (`humanml_trans_enc_512.zip`, 393 MB) | MIT | Pipeline; paper 2 |
| `motiondiffuse-ckpt` | `vendor/motiondiffuse/text2motion/checkpoints/t2m/t2m_motiondiffuse/` (910 MB) | MotionDiffuse HumanML3D checkpoint, mean and std | Hugging Face Space `mingyuan/MotionDiffuse`, revision `09ca408`, `checkpoints/t2m/t2m_motiondiffuse/` | S-Lab License 1.0 (non-commercial) | Pipeline |
| `clip` | `~/.cache/clip/ViT-B-32.pt` (354 MB, outside the repository) | CLIP ViT-B/32, the text encoder of MDM and MotionDiffuse | OpenAI's URL inside the `clip` package, which checks its own SHA-256 | MIT | Pipeline; paper 2 and 3 |
| `hml-clip` | `tests/012314.npy` (45 KB) | One HumanML3D processed clip: the skeleton reference of `amass_label_check.py` | `raw.githubusercontent.com`, HumanML3D at commit `9176e8f` | MIT | Pipeline (real-motion check) |
| `wan-vae` | `vendor/wan_vae/vae/` (484 MB) | Wan 2.1 video VAE (no diffusion model) | Hugging Face `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision `0fad780`, `vae/` | Apache-2.0 | Pipeline, video test only |
| `intphys` | `vendor/intphys/dev.tar.gz` (3.2 GB), unpacked to `vendor/intphys/dev/` (3.0 GB, 360 videos) | IntPhys 2019 development set | https://download-intphys.cognitive-ml.fr/dev.tar.gz (published MD5 `6a72715007f2f3b0e9546bfa8d8fc39b`) | CC BY 4.0 | Pipeline, video test only. The archive can be deleted once unpacked |

## 3. Third-party code (git, pinned commits)

| `--only` | Path and size | What it is | Source | Licence | Needed for |
|---|---|---|---|---|---|
| `mdm` | `vendor/mdm/` (92 MB without `save/`) | MDM code; also its skeleton and SMPL-fitting code | github.com/GuyTevet/motion-diffusion-model at `ef8edce` | MIT | Pipeline; paper 2 and 3 |
| `humanml3d` | `vendor/HumanML3D/` (235 MB) | HumanML3D repository: normalization statistics, captions (`texts.zip`, unpacked by the script), test split, clip index | github.com/EricGuo5513/HumanML3D at `9176e8f` | MIT | Pipeline; paper 2 and 3 |
| `motiondiffuse` | `vendor/motiondiffuse/` (61 MB without checkpoints) | MotionDiffuse code | github.com/MotrixLab/MotionDiffuse at `aba848e` | S-Lab License 1.0 | Pipeline |

## 4. Virtual environments

| `--only` | Path and size | What it is | How it is built | Needed for |
|---|---|---|---|---|
| `venv` | `.venv/` (1.0 GB) | The project environment (Python 3.12, PyTorch, CLIP, gdown) | README's Setup: `uv venv --python 3.12 .venv`, then `requirements.txt` (CLIP with `--no-build-isolation`); the script does both | Pipeline; paper 2 and 3 run many scripts with it |
| `kokoro` | `vendor/kokoro-venv/` (878 MB), plus the model in `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M` (327 MB) | Kokoro text-to-speech (`kokoro` 0.9.4, spaCy `en_core_web_sm` 3.8.0) and its model, voice `am_michael` | The script: uv environment, pinned packages, model from Hugging Face `hexgrad/Kokoro-82M` (Apache-2.0) | Figures: presentation narration (paper 1 and paper 2's video) |
| none | `vendor/chatterbox-venv/` (1.0 GB), `vendor/hf-chatterbox/` (3.0 GB) | Chatterbox voice cloning, tried for the narration and rejected | Not rebuilt | Not needed (`docs/HANDOFF.md`: can be deleted) |

## 5. Generated results (the script prints the command; it never runs it)

Times are on an Apple M4 Pro. The runs are seeded (noise seed `seed * 100003 + batch`).

| `--only` | Path and size | What it is | Command | Needed for |
|---|---|---|---|---|
| `data-phase2` | `data/phase2/` (16 GB; 9 runs of about 1.8 GB) | Sampled motions, recorded activations and labels: MDM-50, MDM-1000 and MotionDiffuse, seeds 0 to 2, 1,000 motions each | `.venv/bin/python scripts/run_phase2_local.py --seeds 0,1,2` (about 12 min), the same with `--checkpoint 1000steps` (about 47 min), `.venv/bin/python scripts/run_motiondiffuse.py --tag phase2 --seeds 0,1,2` (not timed) | Pipeline: every probe and control. Paper 2 reads the seed-0 runs; paper 3 the MDM-50 seeds 1 and 2 |
| `data-jump-heavy` | `data/jump_heavy/` (15 GB; 9 runs) | The same for the jump-heavy prompts | `scripts/run_phase2_local.py --prompts-file prompts/jump_heavy.txt --tag jump_heavy --seeds 0,1,2 --generate-only`, again with `--checkpoint 1000steps`; `scripts/run_motiondiffuse.py --tag jump_heavy --seeds 0,1,2` | Pipeline; paper 2 and 3 as above |
| `data-cond-token` | `data/cond_token/` (5.3 GB; 3 runs) | MDM-50 runs that also record the condition token | `scripts/run_phase2_local.py --tag cond_token --seeds 0,1,2 --generate-only --record-cond-token` | Pipeline (condition-token check) |
| `intphys-latents` | `data/intphys_vae/latents.npz` (281 MB) | Clean Wan VAE latents of the 360 IntPhys videos | `scripts/intphys_vae_encode.py` (about 1 h; needs `wan-vae`, `intphys`) | Pipeline, video test only |
| `data-sensitivity` | `data/sensitivity/` (132 MB) | Label-threshold variants of the seed-0 caches | `scripts/label_sensitivity.py controls` | Pipeline (threshold sensitivity) |
| `preds` | `results/**/preds/` (470 MB, gitignored) | Out-of-fold predictions of the readouts | Every `scripts/fair_readouts.py` run writes them (commands in README.md); `scripts/phase2_controls.py --save-preds` | Figures and confidence intervals (`bootstrap_ci.py`, `fair_best_of_family.py`, web figures, demo video) |
| `figure-caches` | `data/demo_video/` (131 MB), `data/video_v2/` (118 MB), `data/smpl_web/` (104 MB), `data/presentation/` (50 MB), `data/smpl_teaser/` (4 MB) | Rendered frames, SMPL fits of shown motions, narration audio | `scripts/make_demo_video.py prepare`, `render`, `compose`; `scripts/make_web_figures.py` (body images through `body_images.py`); `scripts/fit_smpl_teaser.py`; `scripts/tts_kokoro.py` then `scripts/make_presentation.py`; `video_v2/make_video.py` (need `smpl`; narration needs `kokoro`) | Figures only |
| none | `paper/submission/supplementary_material.zip` (33 MB), `_upload/` (169 MB) | Build outputs: the supplement, the GitHub upload package, the arXiv source, and the as-submitted TMLR supplement and video | `scripts/build_supplement.sh`, `scripts/build_upload.sh`, `scripts/build_arxiv.sh` | Not needed. The `_upload/tmlr_*_as_submitted*` files record what was submitted and cannot be rebuilt byte for byte: keep a backup |

## 6. In git, no action needed

`results/` without `preds/` (about 25 MB), `paper/` (126 MB with its ignored build files), `video_v2/` (57 MB, three
MP4s of 28 MB), `site_page/`, `site_page 2/` and `site_page.zip` (17, 17 and 16 MB), `project_page/` (6 MB).
