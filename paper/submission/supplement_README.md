# Supplementary material

Anonymized supplement for "Probing Physics in Motion Diffusion Models Against Their Own Estimate".

## Contents

| Path | What it is |
| --- | --- |
| `videos/presentation.mp4` | Narrated overview of the paper (6:25, 1080p) with burned-in captions; `presentation.srt` holds the captions as text |
| `videos/demo_video.mp4` | Silent demo (47 s, 1080p): one jump at 0, 20 and 40% of denoising and the final motion, a quarter-speed replay, a floating walk, and three jumps where a layer adds to the estimate (selected for that reason) |
| `scripts/`, `tools/`, `notebooks/` | Code for every experiment, table and figure, from sampling to the final plots |
| `tests/` | Synthetic-data tests of the analysis pipeline |
| `prompts/jump_heavy.txt` | The jump-heavy prompt set |
| `results/` | Every result table the paper reads (CSV and text); cached arrays and per-sample predictions are left out for size |
| `paper/figures_web/` | The figure pages (D3) that `scripts/make_web_figures.py` prints to vector PDF, with fonts (SIL OFL) and their exported data |

Every number in the paper, the figures and the videos comes from these scripts and result files. The bodies in the
figures and videos are SMPL meshes fitted to the generated joints for display only; no number is computed from them.

## Setup (tested on an Apple M4 Pro laptop, macOS, Python 3.12)

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt          # CLIP installs with --no-build-isolation
```

External code, data and models are fetched on first use into `vendor/`: MDM (commit `ef8edce`) with its 50-step
and 1000-step HumanML3D checkpoints, HumanML3D's normalization statistics and test captions, and CLIP ViT-B/32.
MotionDiffuse needs its official code in `vendor/motiondiffuse` and its HumanML3D checkpoint (see the docstring of
`scripts/run_motiondiffuse.py`). Optional: the Wan 2.1 VAE and the IntPhys development set (video test); the SMPL
body model (figures and videos only); AMASS BMLmovi and the Extended SMPL+H model (real-motion label check). SMPL,
SMPL+H and AMASS require registration and licence acceptance on their own sites and are not included here.

## Pipeline

| Step | Command | Output |
| --- | --- | --- |
| Pilot (phase 1) | `scripts/run_phase1_local.py` | `results/phase1/` |
| Sample, record, label, probe (MDM) | `scripts/run_phase2_local.py --seeds 0,1,2` and `--checkpoint 1000steps --seeds 0,1,2` | `data/phase2/` (about 2 GB per run), `results/phase2/<checkpoint>/` |
| Sample and record (MotionDiffuse) | `scripts/run_motiondiffuse.py --tag phase2 --seeds 0,1,2` (and `--tag jump_heavy`) | `data/<tag>/motiondiffuse_seed*/` |
| Curated vs test captions | `scripts/phase2_by_source.py` | `probe_scores_by_source.csv` |
| Fair-readout controls, learning curve | `scripts/phase2_controls.py` (see its docstring) | `results/phase2/controls/` |
| Temporal readout | `scripts/temporal_readout.py` | `results/phase2/controls/*_temporal_readout.csv` |
| Tuned readouts, conditional gain, learning curves | `scripts/fair_readouts.py --checkpoint 50steps --device mps --curve` (also `1000steps`, `motiondiffuse`); `--layer-spec sweep` for every layer; `--refit-seed 1`, `2` for training variance; `--step K` for later steps; then `scripts/fair_best_of_family.py` | `results/phase2/fair/` |
| Positive control | `scripts/fair_readouts.py --layer-spec inject` | `results/{phase2,jump_heavy}/fair/50steps_inject_*` |
| Text and condition-token controls | `scripts/fair_readouts.py --layer-spec text`; `scripts/run_phase2_local.py --tag cond_token --seeds 0,1,2 --generate-only --record-cond-token`, then `scripts/fair_readouts.py --tag cond_token --layer-spec cond` | `results/*/fair/`, `results/cond_token/fair/` |
| Jump-heavy run (fall rate) | `scripts/make_jump_prompts.py`, then `scripts/run_phase2_local.py --prompts-file prompts/jump_heavy.txt --tag jump_heavy --seeds 0,1,2 --generate-only` (also `--checkpoint 1000steps`), then `scripts/fair_readouts.py --tag jump_heavy --targets flight_gravity,in_flight --curve-targets flight_gravity --curve` | `results/jump_heavy/fair/` |
| Noise levels | `scripts/noise_levels.py` | `paper/tables/tab_snr.tex` |
| Confidence intervals | `scripts/phase2_controls.py --save-preds`, then `scripts/bootstrap_ci.py` | `bootstrap_ci.csv` |
| Label threshold sensitivity | `scripts/label_sensitivity.py physics` and `controls` | `results/sensitivity/` |
| Labels on real motion capture | `scripts/amass_label_check.py` (needs AMASS BMLmovi and the Extended SMPL+H model) | `results/amass/` |
| Early warning | `scripts/early_warning_sim.py` | `results/phase2/early_warning/` |
| Video test (IntPhys) | `scripts/intphys_vae_encode.py`, `scripts/intphys_vae_probe.py` (and `--extra-only`, `--null-only 1000`), `scripts/intphys_change_stats.py` | `results/intphys_vae/` |
| Paper tables and result figures | `scripts/make_paper_assets.py` (runs `make_result_figures.py` and `make_tuned_assets.py`) | `paper/tables/`, `paper/figures/` |
| Web figures | `scripts/make_web_figures.py [names]` (prints the pages with headless Google Chrome; set its path in the script) | `paper/figures/web/` |
| Demo video | `scripts/make_demo_video.py prepare`, `render`, `compose` (needs `imageio-ffmpeg` and the SMPL model) | `paper/supplement/demo_video.mp4` (here in `videos/`) |
| Presentation | narration with Kokoro, offline (`scripts/tts_kokoro.py`, in its own environment with `kokoro` and `soundfile`), then `video_v2/make_video.py`, whose animated chart slides are drawn by `video_v2/charts.py` from the paper's figure data (its sentence-split narration input is not included; the narration text is in `videos/presentation.srt`) | `paper/supplement/presentation.mp4`, `.srt` (here in `videos/`) |

All scripts run from the repository root with `.venv/bin/python`. The two `run_phase*_local.py` scripts are
generated from the notebook sources by `tools/build_local.py`. Synthetic-data tests: `tests/test_phase1_analysis.py`,
`tests/test_phase2_pipeline.py` (set `HML_DIR` to the HumanML3D folder that holds `test.txt`); `tests/real_clip_check.py`
downloads one HumanML3D clip.

## Licences of third-party material

MDM code and checkpoints (MIT), MotionDiffuse (see its repository), HumanML3D (see its repository), CLIP (MIT),
Wan 2.1 (Apache-2.0), IntPhys (CC BY 4.0), D3 (ISC, `paper/figures_web/lib/d3.min.js`), Literata and IBM Plex fonts
(SIL OFL), Kokoro (Apache-2.0). SMPL, SMPL+H and AMASS: register and download them yourself under their licences.
