# Paper (TMLR format)

First full draft, November 16, 2026: "Do Motion Generators Know More Than They Show? Probing Physics
in a Diffusion Model with Fair Baselines". Every number comes from `../docs/FINDINGS.md` and the CSVs in
`../results/`.

## Build
- Locally: `tectonic -X compile main.tex --outdir build` (Tectonic installs the LaTeX packages it needs).
- Overleaf: New Project -> Upload Project -> zip of this folder (include `figures/` and `tables/`).

## Generated files (do not edit by hand)
- `tables/*.tex` and the result figures (`fig_steps`, `fig_fair`, `fig_learning`, `fig_early`) are written by
  `../scripts/make_paper_assets.py` (which calls `../scripts/make_result_figures.py`) from the result CSVs.
  Rerun it after any change to the results: `.venv/bin/python scripts/make_paper_assets.py`.
- All figures share one palette with fixed meanings (`../scripts/paper_style.py`): blue = the model's layers,
  amber to vermillion = readouts of the model's own estimate, green = both, gray = noisy input.
- `figures/fig_teaser.pdf` (Figure 1, SMPL bodies) is written by `../scripts/make_teaser_smpl.py` from meshes fitted
  by `../scripts/fit_smpl_teaser.py`; it needs the SMPL neutral model converted by `../scripts/convert_smpl.py`
  (licence-restricted, kept in `vendor/`, never committed). `../scripts/make_teaser.py` gives the stick-figure version.
- `figures/fig_labels.pdf` (the label explainer) is written by `../scripts/make_label_figure.py`. Example selection
  rules for both are in `../scripts/example_picks.py`.
- `tables/prompts.tex` (the 64 curated prompts) comes from `../tools/build_nb.py`.

## Markers
- Red [TODO: ...] = something still to write or decide (none left).

## arXiv version
`main.tex` is the arXiv preprint (`\usepackage[preprint]{tmlr}`, with the author block of
`submission/arxiv_authors.tex`). `bash ../scripts/build_arxiv.sh` packs it for arXiv as `_upload/arxiv_source.zip`.
