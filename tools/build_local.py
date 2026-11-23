"""Generate the local (Mac: Apple GPU via MPS, else CPU) scripts from the notebook sources.

    scripts/run_phase1_local.py   from build_nb.py
    scripts/run_phase2_local.py   from build_nb2.py

Generation, recording, physics labels, probes, baselines and early prediction are copied
verbatim from the notebook cells. Only Colab-specific pieces are swapped: the setup/download
cells, the device line, Google Drive paths, IPython display and plt.show(). Every swap must
match exactly once, and the analysis cells are checked to appear unchanged in the output, so a
later edit to build_nb*.py can't make the notebooks and scripts drift silently.

    python tools/build_local.py
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")


def load_cells(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(tag, src) for kind, tag, src in mod.cells if kind == "code"]


def sub(src, old, new):
    assert src.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return src.replace(old, new)


# ---------------------------------------------------------------- shared local pieces

COMMON = r'''
VENDOR = os.path.join(ROOT, "vendor")
os.environ.setdefault("MDM_DIR", os.path.join(VENDOR, "mdm"))
os.environ.setdefault("HML_DIR", os.path.join(VENDOR, "HumanML3D", "HumanML3D"))
RUN_T0 = time.time()


class _Tee:
    # Mirror everything printed into OUT/run_log.txt (and into a table file while capturing).
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_log = open(os.path.join(OUT, "run_log.txt"), "w")
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)
_cap = None


def capture_start():
    global _cap
    _cap = io.StringIO()
    sys.stdout.streams.append(_cap)


def capture_save(name):
    sys.stdout.streams.remove(_cap)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(_cap.getvalue())


import matplotlib
matplotlib.use("Agg")   # no windows: every figure is saved to OUT


def display(obj):
    # Stand-in for IPython's display(): print whole tables.
    import pandas as pd
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        print(obj)


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"Command failed: {cmd}")
    return r.stdout


MDM_COMMIT = "ef8edce"   # the commit the loading code was checked against
CKPT_IDS = {"50steps": "1cfadR1eZ116TIdXK7qDX1RugAerEiJXr",     # Google Drive ids from MDM's README
            "1000steps": "1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821"}
LOCAL_ZIP = os.environ.get("LOCAL_ZIP", "")   # a checkpoint zip downloaded by hand, if Drive refuses gdown


def fetch_repos():
    # Clone MDM at the checked commit and HumanML3D (Mean/Std, test split, captions) once.
    os.makedirs(VENDOR, exist_ok=True)
    if not os.path.exists(os.environ["MDM_DIR"]):
        sh(f"git clone -q https://github.com/GuyTevet/motion-diffusion-model.git '{os.environ['MDM_DIR']}'")
        sh(f"git checkout -q {MDM_COMMIT}", cwd=os.environ["MDM_DIR"])
    if not os.path.exists(os.environ["HML_DIR"]):
        sh(f"git clone -q --depth 1 https://github.com/EricGuo5513/HumanML3D.git '{os.path.join(VENDOR, 'HumanML3D')}'")
    print("MDM commit:", sh("git rev-parse --short HEAD", cwd=os.environ["MDM_DIR"]).strip())


def fetch_checkpoint(name, dest):
    # Download and unzip one official checkpoint into dest, unless it is already there.
    if glob.glob(os.path.join(dest, "**", "args.json"), recursive=True):
        return
    os.makedirs(dest, exist_ok=True)
    zpath = LOCAL_ZIP or os.path.join(dest, "ckpt.zip")
    if not LOCAL_ZIP:
        try:   # pass the bare file id: gdown 6 dropped --fuzzy, and every version accepts an id
            sh(f"'{os.path.join(os.path.dirname(sys.executable), 'gdown')}' '{CKPT_IDS[name]}' -O '{zpath}'")
        except RuntimeError:
            raise RuntimeError("Checkpoint download failed (see the error above; a Drive quota limit is the usual "
                               f"cause). Download https://drive.google.com/file/d/{CKPT_IDS[name]} in a browser, "
                               "then rerun with LOCAL_ZIP=/path/to/the.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    if not LOCAL_ZIP:
        os.remove(zpath)
'''

DEVICE_OLD = 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"'
DEVICE_NEW = ('# Local: Apple GPU (MPS) if available, else CPU (or forced with --device).\n'
              'DEVICE = CLI.device if CLI.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")')
RIC_IMPORT = "from data_loaders.humanml.scripts.motion_process import recover_from_ric\n"
MPS_SHIM = r'''
# Local: MPS has no float64, and MDM moves its float64 noise-schedule tables to the device
# before casting to float32. Index on the CPU and cast first: same float32 values, any device.
import diffusion.gaussian_diffusion as _gd


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr)[timesteps.cpu()].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


_gd._extract_into_tensor = _extract_into_tensor
'''


def local_model_load(s):
    return sub(sub(s, DEVICE_OLD, DEVICE_NEW), RIC_IMPORT, RIC_IMPORT + MPS_SHIM)


def build(cells, prelude, setup, transforms, verbatim, out_name):
    # prelude first; the local setup replaces the first Colab setup cell; other setup cells are dropped.
    parts, used, setup_done = [prelude.lstrip("\n")], [], False
    for n, (tag, src) in enumerate(cells):
        if tag == "setup":
            if not setup_done:
                parts.append("# %% local: setup (replaces the Colab setup cells)\n" + setup.lstrip("\n"))
                setup_done = True
            continue
        hits = [i for i, (marker, _) in enumerate(transforms) if marker in src]
        assert len(hits) <= 1, f"cell {n} matches several transforms"
        before, body, after = transforms[hits[0]][1](src) if hits else ("", src, "")
        used += hits
        if any(m in src for m in verbatim):
            assert body == src, f"cell {n} is meant to be copied verbatim"
        parts.append(f"# %% notebook cell {n}\n{before}{body}\n{after}")
    assert setup_done, "no setup cell found"
    assert sorted(used) == list(range(len(transforms))), "a transform matched no cell (or several)"
    text = "\n\n".join(p.rstrip("\n") for p in parts) + "\n"
    for m in verbatim:   # final check: the verbatim cells are in the script unchanged
        assert next(src for _, src in cells if m in src) in text
    path = os.path.join(SCRIPTS, out_name)
    os.makedirs(SCRIPTS, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    compile(text, path, "exec")
    print("built", os.path.normpath(path))


# ---------------------------------------------------------------- phase 1

P1_PRELUDE = r'''
"""Phase 1 feasibility run on this Mac (Apple GPU via MPS if available, else CPU).

GENERATED by tools/build_local.py from the notebook cells in tools/build_nb.py. Don't edit
this file by hand: change build_nb.py (analysis) or build_local.py (local plumbing), rerun
`python tools/build_local.py`.

    .venv/bin/python scripts/run_phase1_local.py --smoke   # 2 prompts x 1 sample: loading + recording only
    .venv/bin/python scripts/run_phase1_local.py           # full phase 1: 64 prompts x 4 samples

Outputs: results/phase1/ (results/phase1_smoke/ for --smoke), including run_log.txt.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before torch is imported
import argparse, glob, io, json, subprocess, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser(description="Phase 1 feasibility run (local).")
_ap.add_argument("--smoke", action="store_true", help="2 prompts x 1 sample; stops after the physics labels")
_ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
_ap.add_argument("--out", default=None, help="output folder (default: results/phase1 or results/phase1_smoke)")
CLI = _ap.parse_args()
SMOKE = CLI.smoke
OUT = os.path.abspath(CLI.out or os.path.join(ROOT, "results", "phase1_smoke" if SMOKE else "phase1"))
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    os.environ["N_REPS"] = "1"
''' + COMMON

P1_SETUP = r'''
# No nvidia-smi, no Drive mount. The 50-step checkpoint goes where the notebook expects it.
fetch_repos()
fetch_checkpoint("50steps", os.path.join(os.environ["MDM_DIR"], "save"))
'''

P1_SMOKE_PROMPTS = r'''
if SMOKE:   # local smoke test: one flight prompt and one balance prompt (the two the stick-figure cell draws)
    PROMPTS = ["a person jumps forward.", "a person stands on one leg."]
    print(f"SMOKE TEST: {len(PROMPTS)} prompts x {N_REPS} rep = {len(PROMPTS) * N_REPS} motions")
'''

P1_AFTER_GENERATION = r'''
_bad = {k: int((~np.isfinite(v.astype(np.float32))).sum()) for k, v in ACTS.items()}
print(f"generation: {time.time() - t0:.0f}s on {DEVICE} | non-finite activation values: {sum(_bad.values())} | "
      f"non-finite final joints: {int((~np.isfinite(J_final)).sum())}")
print("recorded:", len(ACTS), "(step, layer) keys, each", next(iter(ACTS.values())).shape,
      "| x_t", XT[RECORD_STEPS[0]].shape, "| x0 joints", X0J[RECORD_STEPS[0]].shape,
      "| x0 features", X0F[RECORD_STEPS[0]].shape, "| text", TXT.shape)
'''

P1_SMOKE_STOP = r'''
if SMOKE:
    np.savez_compressed(os.path.join(OUT, "smoke_motions.npz"), joints=J_final.astype(np.float32),
                        prompts=np.array(all_prompts), **{f"label_{k}": v for k, v in LAB.items()})
    print("Smoke test finished. Probes and early prediction need the full prompt set "
          "(prompt-grouped splits), so they are skipped here.")
    print(f"total time {time.time() - RUN_T0:.0f}s; outputs in {OUT}")
    sys.exit(0)
'''

P1_AFTER_EARLY = r'''
EARLY.pivot_table(index=["failure", "source"], columns="step",
                  values=["spearman", "within_prompt"]).round(3).to_csv(os.path.join(OUT, "early_prediction_summary.csv"))
'''

P1_EPILOGUE = r'''
# Local: counts the go/no-go review needs (how many motions actually have >= 0.2 s of flight),
# plus the run configuration.
has_g = np.isfinite(LAB["flight_gravity"]).any(1)
info = dict(device=DEVICE, torch=torch.__version__, n_motions=int(len(all_prompts)), n_prompts=len(PROMPTS),
            n_reps=N_REPS, record_steps=[int(k) for k in RECORD_STEPS], test_prompts=sorted(int(i) for i in test_prompts),
            motions_with_gravity_fit=int(has_g.sum()),
            motions_with_gravity_fit_by_group={c: int(has_g[cats == c].sum()) for c in ["flight", "balance", "other"]},
            frames_with_gravity_label=int(np.isfinite(LAB["flight_gravity"][:, FR]).sum()),
            test_frames_with_gravity_label=int(np.isfinite(LAB["flight_gravity"][is_test][:, FR]).sum()),
            runtime_s=round(time.time() - RUN_T0))
with open(os.path.join(OUT, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
'''

P1_TRANSFORMS = [
    ('DEVICE = "cuda"', lambda s: ("", local_model_load(s), "")),
    ("PROMPTS = [", lambda s: ("", s, P1_SMOKE_PROMPTS)),
    ("J_final, F_final, TXT, ACTS", lambda s: ("", s, P1_AFTER_GENERATION)),
    ("def category(p)", lambda s: ("capture_start()\n", s, '\ncapture_save("physics_table.txt")\n')),
    ("def animate(i, title)", lambda s: ("", sub(sub(
        s, "from IPython.display import HTML, display", "# local: animations are saved as GIFs, not shown inline"),
        "    return HTML(anim.to_jshtml())",
        '    path = os.path.join(OUT, "anim_" + title.replace("a person ", "").strip(". ").replace(" ", "_") + ".gif")\n'
        '    anim.save(path, writer="pillow")\n'
        '    return path'), P1_SMOKE_STOP)),
    ("layer_tab = d[d.source", lambda s: ("", sub(sub(sub(
        s, "from IPython.display import display", "# local: display() is defined at the top of this script"),
        "    plt.tight_layout(); plt.show()",
        '    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"probes_{name}.png")); plt.close(fig)'),
        "display(summary)", 'display(summary)\nsummary.to_csv(os.path.join(OUT, "probe_summary.csv"))'), "")),
    ("def failure_rates", lambda s: ("", sub(
        s, "plt.tight_layout(); plt.show()",
        'plt.tight_layout(); plt.savefig(os.path.join(OUT, "early_prediction.png")); plt.close(fig)'), P1_AFTER_EARLY)),
    ("np.savez_compressed(\"/content/", lambda s: ("", sub(sub(sub(sub(
        s, '"/content/balance_probe_results.npz"', 'os.path.join(OUT, "balance_probe_results.npz")'),
        '"/content/probe_scores.csv"', 'os.path.join(OUT, "probe_scores.csv")'),
        '"/content/early_prediction.csv"', 'os.path.join(OUT, "early_prediction.csv")'),
        'print("Saved balance_probe_results.npz, probe_scores.csv and early_prediction.csv in /content (Files panel).")',
        'print(f"Saved results to {OUT}")'), P1_EPILOGUE)),
]
P1_VERBATIM = ["PROMPTS = [", "class Recorder", "J_final, F_final, TXT, ACTS", "from shapely.geometry import",
               "def category(p)", "from sklearn.linear_model import RidgeCV"]

# ---------------------------------------------------------------- phase 2

P2_PRELUDE = r'''
"""Phase 2 (scale and robustness) on this Mac (Apple GPU via MPS if available, else CPU).

GENERATED by tools/build_local.py from the notebook cells in tools/build_nb2.py. Don't edit
this file by hand: change build_nb2.py (analysis) or build_local.py (local plumbing), rerun
`python tools/build_local.py`.

    .venv/bin/python scripts/run_phase2_local.py --smoke                 # 70 prompts x 2 samples, 1 run
    .venv/bin/python scripts/run_phase2_local.py                         # notebook defaults: seeds 0,1
    .venv/bin/python scripts/run_phase2_local.py --seeds 0,1,2           # three runs
    .venv/bin/python scripts/run_phase2_local.py --checkpoint 1000steps  # original 1000-step MDM
    .venv/bin/python scripts/run_phase2_local.py --prompts-file prompts/jump_heavy.txt \
        --tag jump_heavy --seeds 0,1,2 --generate-only                   # jump-heavy data, no analysis

Per-batch arrays, labels and per-run CSVs are cached in data/phase2/ (the notebook's Drive
folder; data/<tag>/ with --tag) so reruns resume. Tables and figures go to
results/phase2/<checkpoint>/ (results/<tag>/<checkpoint>/).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before torch is imported
import argparse, glob, io, json, shutil, subprocess, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser(description="Phase 2 scale and robustness run (local).")
_ap.add_argument("--smoke", action="store_true", help="64 curated + 6 test prompts x 2 samples, seed 0 only")
_ap.add_argument("--checkpoint", choices=["50steps", "1000steps"], default="50steps")
_ap.add_argument("--seeds", default="0,1", help="comma-separated run seeds (notebook default 0,1)")
_ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
_ap.add_argument("--prompts-file", default=None,
                 help="one prompt per line; replaces the 64 curated prompts + sampled test captions")
_ap.add_argument("--samples-per-prompt", type=int, default=None, help="default: the notebook's 4 (2 with --smoke)")
_ap.add_argument("--frame-stride", type=int, default=None, help="default: the notebook's 4")
_ap.add_argument("--tag", default=None, help="data/<tag>/ and results/<tag>/ (default phase2, or phase2_smoke)")
_ap.add_argument("--generate-only", action="store_true", help="sample, record and label, then stop before the analysis")
_ap.add_argument("--record-cond-token", action="store_true", help="also store each layer's condition-token output (act_<step>_<layer>c)")
CLI = _ap.parse_args()
if CLI.prompts_file:   # the setup changes directory into the MDM repository, so fix the path now
    CLI.prompts_file = os.path.abspath(CLI.prompts_file)
SMOKE = CLI.smoke
_tag = CLI.tag or ("phase2_smoke" if SMOKE else "phase2")
DATA = os.path.join(ROOT, "data", _tag)                    # replaces MyDrive/motion_probe/phase2
OUT = os.path.join(ROOT, "results", _tag, CLI.checkpoint)
os.makedirs(OUT, exist_ok=True)
''' + COMMON

P2_SETTINGS = r'''
# Local overrides of the settings above.
CHECKPOINT = CLI.checkpoint
RUN_SEEDS = [int(s) for s in CLI.seeds.split(",")]
DRIVE_ROOT = DATA
os.makedirs(DRIVE_ROOT, exist_ok=True)
if SMOKE:
    N_TEST_CAPTIONS, SAMPLES_PER_PROMPT, RUN_SEEDS = 6, 2, [0]
if CLI.samples_per_prompt:
    SAMPLES_PER_PROMPT = CLI.samples_per_prompt
if CLI.frame_stride:
    FRAME_STRIDE = CLI.frame_stride
print(f"checkpoint {CHECKPOINT} | seeds {RUN_SEEDS} | {N_TEST_CAPTIONS} test captions | "
      f"{SAMPLES_PER_PROMPT} samples per prompt | cache {DRIVE_ROOT}")
'''

P2_SETUP = r'''
# No nvidia-smi, no Drive mount. Checkpoints live in vendor/checkpoints/ckpt_<name>; the 50-step
# one reuses phase 1's copy (same file) instead of downloading it again.
fetch_repos()
if not os.path.exists(os.path.join(os.environ["HML_DIR"], "texts")):
    with zipfile.ZipFile(os.path.join(os.environ["HML_DIR"], "texts.zip")) as z:
        z.extractall(os.environ["HML_DIR"])
CKPT_DIR = os.path.join(VENDOR, "checkpoints", f"ckpt_{CHECKPOINT}")
_p1 = glob.glob(os.path.join(os.environ["MDM_DIR"], "save", "**", "args.json"), recursive=True)
if CHECKPOINT == "50steps" and _p1 and not os.path.exists(CKPT_DIR):
    os.makedirs(os.path.dirname(CKPT_DIR), exist_ok=True)
    os.symlink(os.path.dirname(_p1[0]), CKPT_DIR)
fetch_checkpoint(CHECKPOINT, CKPT_DIR)
print("Setup done:", glob.glob(f"{CKPT_DIR}/**/args.json", recursive=True))
'''

P2_PROMPTS_FILE = r'''
if CLI.prompts_file:   # local: a fixed prompt list; prompts that are also curated keep that source
    PROMPTS = [l.strip() for l in open(CLI.prompts_file) if l.strip()]
    SOURCE = np.array(["curated" if p in CURATED else "test" for p in PROMPTS])
    all_prompts = [p for _ in range(SAMPLES_PER_PROMPT) for p in PROMPTS]
    prompt_id = np.array([i for _ in range(SAMPLES_PER_PROMPT) for i in range(len(PROMPTS))])
    print(f"prompts from {CLI.prompts_file}: {len(PROMPTS)} prompts ({int((SOURCE == 'curated').sum())} curated) x "
          f"{SAMPLES_PER_PROMPT} = {len(all_prompts)} motions per run")
'''

P2_COND_TOKEN = r'''
if CLI.record_cond_token:   # local: also keep each layer's condition-token output (token 0), [bs, 1, d]
    def _cond_hook(li):
        def hook(module, inputs, out):
            if recorder.on:
                recorder.buf[f"{li}c"] = out[:1].permute(1, 0, 2).to(torch.float16).cpu()
        return hook
    for _li, _layer in enumerate(LAYERS):
        _layer.register_forward_hook(_cond_hook(_li))
'''

P2_GENERATE_ONLY = r'''
if CLI.generate_only:
    print(f"--generate-only: samples and labels are in {DRIVE_ROOT}; stopping before the analysis "
          f"({time.time() - RUN_T0:.0f}s)")
    sys.exit(0)
'''

P2_AFTER_EARLY = r'''
EARLY.to_csv(os.path.join(OUT, "early_prediction_all_runs.csv"), index=False)
(other[other.source.isin([FIXED_LAYER, "x0 probe", "explicit (x0 guess)", "text only"])]
 .pivot_table(index=["failure", "source"], columns="step", values=["spearman", "within_prompt"]).round(3)
 .to_csv(os.path.join(OUT, "early_prediction_fixed_layer_other_runs.csv")))
'''

P2_AFTER_COMPARE = r'''
if len({f.checkpoint.iloc[0] for f in frames}) >= 2:
    best.groupby(["target", "checkpoint", "progress"]).score.mean().unstack("progress").round(3).to_csv(
        os.path.join(os.path.dirname(OUT), "checkpoint_comparison.csv"))
'''

P2_EPILOGUE = r'''
# Local: replaces the Drive file listing. Copies the combined tables next to the figures and
# records the run configuration and the flight counts per run.
RES.to_csv(os.path.join(OUT, "probe_scores_all_runs.csv"), index=False)
info = dict(device=DEVICE, torch=torch.__version__, checkpoint=CHECKPOINT, seeds=RUN_SEEDS,
            n_prompts=len(PROMPTS), n_curated=len(CURATED), samples_per_prompt=SAMPLES_PER_PROMPT,
            n_motions_per_run=len(all_prompts), record_steps=[int(k) for k in RECORD_STEPS],
            frame_stride=FRAME_STRIDE, fixed_layer=FIXED_LAYER, runtime_s=round(time.time() - RUN_T0),
            motions_with_gravity_fit={int(s): {g: int(np.isfinite(LABELS[s][0]["flight_gravity"]).any(1)[GROUPS == g].sum())
                                               for g in ["curated: flight", "curated: balance", "curated: other", "test captions"]}
                                      for s in RUN_SEEDS})
with open(os.path.join(OUT, "run_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
print("Tables and figures in", OUT, "| cached arrays and per-run CSVs in", DRIVE_ROOT)
'''

P2_TRANSFORMS = [
    ('CHECKPOINT = "50steps"', lambda s: ("", s, P2_SETTINGS)),
    ("CURATED = [", lambda s: ("", s, P2_PROMPTS_FILE)),
    ("LABELS = {}", lambda s: ("", s, P2_GENERATE_ONLY)),
    ("class Recorder", lambda s: ("", s, P2_COND_TOKEN)),
    ('DEVICE = "cuda"', lambda s: ("", sub(local_model_load(s),
        'MDM_DIR, HML_DIR = "/content/mdm", "/content/hml/HumanML3D"',
        'MDM_DIR, HML_DIR = os.environ["MDM_DIR"], os.environ["HML_DIR"]   # local paths'), "")),
    ("def group(p, src)", lambda s: ("capture_start()\n", s, '\ncapture_save("physics_table.txt")\n')),
    ("TARGETS = {", lambda s: ("", sub(
        s, "from IPython.display import display", "# local: display() is defined at the top of this script"), "")),
    ("Mean over runs (lines)", lambda s: ("", sub(sub(
        s, "    plt.tight_layout(); plt.show()",
        '    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"probes_{name}.png")); plt.close(fig)'),
        "display(summary)", 'display(summary)\nsummary.to_csv(os.path.join(OUT, "probe_summary.csv"))'), "")),
    ("def failure_rates", lambda s: ("", sub(
        s, "plt.tight_layout(); plt.show()",
        'plt.tight_layout(); plt.savefig(os.path.join(OUT, "early_prediction.png")); plt.close(fig)'), P2_AFTER_EARLY)),
    ("frames = []", lambda s: ("", s, P2_AFTER_COMPARE)),
    ('print("All results are on Drive in"', lambda s: ("", P2_EPILOGUE.strip("\n"), "")),
]
P2_VERBATIM = ["CURATED = [", "class Recorder", "def run_dir(seed)", "def batch_files(seed)", "def group(p, src)"]


if __name__ == "__main__":
    build(load_cells("build_nb"), P1_PRELUDE, P1_SETUP, P1_TRANSFORMS, P1_VERBATIM, "run_phase1_local.py")
    build(load_cells("build_nb2"), P2_PRELUDE, P2_SETUP, P2_TRANSFORMS, P2_VERBATIM, "run_phase2_local.py")
