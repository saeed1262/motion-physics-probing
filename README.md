# Do motion generators know more than they show?

Code, results and paper for a controlled probing study of physics inside text-to-motion
diffusion models (MDM and MotionDiffuse). Every number in the paper comes from the scripts below, and `docs/FINDINGS.md`
logs each result with its source. Paper: [arXiv:2611.24423](https://arxiv.org/abs/2611.24423).

## Setup (tested on an Apple M4 Pro laptop, macOS, Python 3.12)

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt          # CLIP installs with --no-build-isolation
```

External code, data and models are fetched on first use into `vendor/` (not tracked):
MDM (commit `ef8edce`) with its 50-step and 1000-step HumanML3D checkpoints, HumanML3D's
normalization statistics and test captions, CLIP ViT-B/32. MotionDiffuse needs its official code in
`vendor/motiondiffuse` and its HumanML3D checkpoint (see `scripts/run_motiondiffuse.py`). Optional: the Wan 2.1 VAE and the
IntPhys development set (video test), and the SMPL body model (Figure 1 only; requires
registration and licence acceptance at smpl.is.tue.mpg.de, never redistributed here); for the real-motion check, AMASS BMLmovi (amass.is.tue.mpg.de) and the Extended SMPL+H
model (mano.is.tue.mpg.de), likewise registration-only.

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Pilot (phase 1) | `.venv/bin/python scripts/run_phase1_local.py` | `results/phase1/` |
| Sample, record, label, probe (phase 2) | `.venv/bin/python scripts/run_phase2_local.py --seeds 0,1,2` and `--checkpoint 1000steps --seeds 0,1,2` | `data/phase2/` (cache, about 2 GB per run), `results/phase2/<checkpoint>/` |
| Sample and record (MotionDiffuse) | `scripts/run_motiondiffuse.py --tag phase2 --seeds 0,1,2` (and `--tag jump_heavy`) | `data/<tag>/motiondiffuse_seed*/` |
| Curated vs test captions | `scripts/phase2_by_source.py` | `probe_scores_by_source.csv` |
| Fair-readout controls, learning curve | `scripts/phase2_controls.py` (see its docstring) | `results/phase2/controls/` |
| Temporal readout | `scripts/temporal_readout.py` | `results/phase2/controls/*_temporal_readout.csv` |
| Tuned readouts, seeds pooled, conditional gain, learning curves | `scripts/fair_readouts.py --checkpoint 50steps --device mps --curve` (and `1000steps`); `--layer-spec sweep` for every layer and all layers; `--refit-seed 1`, `2` for training variance; then `scripts/fair_best_of_family.py` | `results/phase2/fair/` |
| Positive control | `scripts/fair_readouts.py --layer-spec inject` | `results/{phase2,jump_heavy}/fair/50steps_inject_*` |
| Text control | `scripts/fair_readouts.py --layer-spec text` | `results/*/fair/` |
| Condition-token check | `scripts/run_phase2_local.py --tag cond_token --seeds 0,1,2 --generate-only --record-cond-token`, then `scripts/fair_readouts.py --tag cond_token --layer-spec cond` | `results/cond_token/fair/` |
| Jump-heavy run (gravity) | `scripts/make_jump_prompts.py`, then `scripts/run_phase2_local.py --prompts-file prompts/jump_heavy.txt --tag jump_heavy --seeds 0,1,2 --generate-only` (also with `--checkpoint 1000steps`), then `scripts/fair_readouts.py --tag jump_heavy --targets flight_gravity,in_flight --curve-targets flight_gravity --curve` | `data/jump_heavy/`, `results/jump_heavy/fair/` |
| Noise levels | `scripts/noise_levels.py` | `paper/tables/tab_snr.tex` |
| Confidence intervals | `scripts/phase2_controls.py --save-preds`, then `scripts/bootstrap_ci.py` | `bootstrap_ci.csv` |
| Label threshold sensitivity | `scripts/label_sensitivity.py physics` and `controls` | `results/sensitivity/` |
| Labels on real motion capture | `scripts/amass_label_check.py` (needs AMASS BMLmovi and the Extended SMPL+H model, downloaded yourself under their licences) | `results/amass/` |
| Early warning | `scripts/early_warning_sim.py` | `results/phase2/early_warning/` |
| Video test (IntPhys) | `scripts/intphys_vae_encode.py`, `scripts/intphys_vae_probe.py` (and `--extra-only`, `--null-only 1000`), `scripts/intphys_change_stats.py` | `results/intphys_vae/` |
| Paper tables and result figures | `scripts/make_paper_assets.py` (runs `make_result_figures.py` and `make_tuned_assets.py`) | `paper/tables/`, `paper/figures/` |
| Motion figures | `scripts/make_label_figure.py`; Figure 1: `convert_smpl.py`, `fit_smpl_teaser.py`, `make_teaser_smpl.py` | `paper/figures/` |
| Web figures | `scripts/make_web_figures.py [names]` (headless Google Chrome) | `paper/figures/web/` |
| Demo video | `scripts/make_demo_video.py prepare`, `render`, `compose` (needs `imageio-ffmpeg` and the SMPL model) | `paper/supplement/demo_video.mp4` |
| Presentation | `scripts/tts_kokoro.py` (Kokoro, in its own environment), then `scripts/make_presentation.py` | `paper/supplement/presentation.mp4`, `.srt` |
| Anonymized supplement | `bash scripts/build_supplement.sh` | `paper/submission/supplementary_material.zip` |
| Paper | `cd paper && tectonic -X compile main.tex --outdir build` | `paper/build/main.pdf` |

The two `run_phase*_local.py` scripts are generated from the notebook sources by
`tools/build_local.py`; edit `tools/build_nb*.py` and regenerate rather than editing them.
Synthetic-data tests: `tests/test_phase1_analysis.py`, `tests/test_phase2_pipeline.py`,
`tests/real_clip_check.py`.

## Licence

This repository's own code is under the MIT licence (`LICENSE`).

## Licences of third-party material

MDM code and checkpoints (MIT), MotionDiffuse (see its repository), HumanML3D (see its repository), CLIP (MIT), Wan 2.1 (Apache-2.0),
IntPhys (CC BY 4.0), D3 (ISC), Literata and IBM Plex fonts (SIL OFL), Kokoro (Apache-2.0), SMPL (non-commercial research licence; register and download it yourself).
