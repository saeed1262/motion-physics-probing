import nbformat as nbf

import os
HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "..", "notebooks")
LABEL_FUNCS = open(os.path.join(HERE, "p2_labels_funcs.py")).read()
RECORDER = open(os.path.join(HERE, "p2_recorder.py")).read()

cells = []  # (kind, tag, source)


def md(s):
    cells.append(("md", None, s.strip("\n")))


def code(s, tag="main"):
    cells.append(("code", tag, s.strip("\n")))


md(r"""
# Phase 2: scale and robustness

Phase 1 asked whether physics is readable inside MDM at all. Phase 2 checks that the answer **holds up**:

1. **More and more varied prompts:** the 64 curated prompts plus captions from HumanML3D's public test split (about 1,000 motions per run).
2. **Several independent runs (seeds),** to report mean ± spread instead of one number.
3. **Optionally the original 1,000-step MDM** (set `CHECKPOINT` below), to see if results depend on the sampler.
4. **Cross-validated probes,** with all folds split by prompt.

**Everything is saved to your Google Drive** in `MyDrive/motion_probe/phase2/`. If Colab disconnects, just run all cells again: finished batches and analyses are skipped.

**How to run:** `Runtime` → `Change runtime type` → **T4 GPU**, then `Runtime` → **Run all**. Approve the Google Drive prompt when it appears. You need about **2 GB of free Drive space per run** (two runs by default). Expect about 30–60 minutes for two runs with the 50-step model (longer with the 1,000-step model).
""")

md("## 1. Settings")

code(r"""
CHECKPOINT = "50steps"      # "50steps" (default) or "1000steps" (the original MDM; about 20x slower to sample)
RUN_SEEDS = [0, 1]          # one independent run per seed; add 2 for a third run
N_TEST_CAPTIONS = 186       # HumanML3D test captions added to the 64 curated prompts (250 prompts total)
SAMPLES_PER_PROMPT = 4      # 250 prompts x 4 = 1,000 motions per run
N_FRAMES = 120              # 6 s at 20 fps
FRAME_STRIDE = 4            # keep every 4th frame of activations (30 per motion) to fit Drive and RAM
BATCH = 64
GUIDANCE = 2.5              # MDM's default classifier-free guidance scale
N_RECORD = 6                # denoising steps recorded, evenly spaced from noise to final
DRIVE_ROOT = "/content/drive/MyDrive/motion_probe/phase2"
""")

md("## 2. Setup (installs, Drive, model download)")

code(r"""
import os, subprocess, glob, json, time

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"Command failed: {cmd}")
    return r.stdout

print(sh("nvidia-smi -L || echo 'No GPU found: use Runtime > Change runtime type > T4 GPU'"))
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)

os.chdir("/content")
if not os.path.exists("/content/mdm"):
    sh("git clone -q https://github.com/GuyTevet/motion-diffusion-model.git mdm")
if not os.path.exists("/content/hml"):
    sh("git clone -q --depth 1 https://github.com/EricGuo5513/HumanML3D.git hml")   # Mean/Std, captions, test split
if not os.path.exists("/content/hml/HumanML3D/texts"):
    sh("cd /content/hml/HumanML3D && unzip -q -o texts.zip")
sh("pip -q install git+https://github.com/openai/CLIP.git gdown shapely")

# Official checkpoints from the MDM README.
CKPT_URLS = {
    "50steps": "https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing",
    "1000steps": "https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing",
}
# If Google Drive refuses the download (quota), save the zip to your own Drive and put its path here.
LOCAL_ZIP = ""   # e.g. "/content/drive/MyDrive/humanml_enc_512_50steps.zip"

CKPT_DIR = f"/content/mdm/save/ckpt_{CHECKPOINT}"
if not glob.glob(f"{CKPT_DIR}/**/args.json", recursive=True):
    os.makedirs(CKPT_DIR, exist_ok=True)
    if LOCAL_ZIP:
        sh(f"cp '{LOCAL_ZIP}' {CKPT_DIR}/ckpt.zip")
    else:
        try:
            sh(f"gdown --fuzzy '{CKPT_URLS[CHECKPOINT]}' -O {CKPT_DIR}/ckpt.zip")
        except RuntimeError:
            raise RuntimeError("Google Drive blocked the download (quota). Open the link in CKPT_URLS, save the "
                               "zip to your Drive, set LOCAL_ZIP to its path, and rerun this cell.")
    sh(f"cd {CKPT_DIR} && unzip -q -o ckpt.zip && rm ckpt.zip")
print("Setup done:", glob.glob(f"{CKPT_DIR}/**/args.json", recursive=True))
""", tag="setup")

md("## 3. Load MDM")

code(r"""
import sys, types
import numpy as np
import torch, torch.nn as nn

MDM_DIR, HML_DIR = "/content/mdm", "/content/hml/HumanML3D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.chdir(MDM_DIR); sys.path.insert(0, MDM_DIR)

# Skip the SMPL body layer (unused when MDM outputs HumanML3D joint positions).
_stub = types.ModuleType("model.rotation2xyz")
class _Rot2xyzStub:
    def __init__(self, device="cpu", dataset=None):
        self.smpl_model = nn.Identity()
    def __call__(self, x, *args, **kwargs):
        return x
_stub.Rotation2xyz = _Rot2xyzStub
sys.modules["model.rotation2xyz"] = _stub
try:
    import scipy.ndimage.filters  # noqa: F401  (MDM imports this; newer SciPy may drop it)
except ImportError:
    import scipy.ndimage
    sys.modules["scipy.ndimage.filters"] = scipy.ndimage

from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import recover_from_ric

args_path = glob.glob(f"{CKPT_DIR}/**/args.json", recursive=True)[0]
ckpt_path = sorted(glob.glob(os.path.join(os.path.dirname(args_path), "model*.pt")))[-1]
defaults = dict(dataset="humanml", unconstrained=False, arch="trans_enc", text_encoder_type="clip",
                latent_dim=512, layers=8, cond_mask_prob=0.1, emb_trans_dec=False, pos_embed_max_len=5000,
                mask_frames=False, diffusion_steps=1000, noise_schedule="cosine", sigma_small=True,
                lambda_vel=0.0, lambda_rcxyz=0.0, lambda_fc=0.0, lambda_target_loc=0.0, use_ema=False)
with open(args_path) as f:
    args = types.SimpleNamespace(**{**defaults, **json.load(f)})

class _NoData:
    dataset = types.SimpleNamespace()

model, diffusion = create_model_and_diffusion(args, _NoData())
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model_avg"] if (args.use_ema and "model_avg" in sd) else sd["model"]
load_model_wo_clip(model, sd)
model.to(DEVICE)   # note: MDM.to() returns None, so no chaining
model.eval()

MEAN = torch.tensor(np.load(os.path.join(HML_DIR, "Mean.npy"))).float()
STD = torch.tensor(np.load(os.path.join(HML_DIR, "Std.npy"))).float()
LAYERS = list(model.seqTransEncoder.layers)
N_STEPS = diffusion.num_timesteps
RECORD_STEPS = sorted(set(np.linspace(0, N_STEPS - 1, N_RECORD).round().astype(int).tolist()))
FR = np.arange(0, N_FRAMES, FRAME_STRIDE)
print(f"Loaded {CHECKPOINT}: {os.path.basename(ckpt_path)} | layers={len(LAYERS)} | steps={N_STEPS} | "
      f"recording steps {RECORD_STEPS}")
""")

md("""
## 4. Prompts

The 64 curated prompts from phase 1 (jumps and balance-heavy actions) plus captions sampled from HumanML3D's **test** split, so prompts never overlap the model's training captions. The prompt list is fixed by `seed 0`, so every run uses the same prompts with different sampling noise.
""")

code(r"""
CURATED = [
    "a person walks forward.", "a person walks forward slowly and carefully.", "a person jogs forward.",
    "a person runs forward quickly.", "a person walks in a circle.", "a person walks backward.",
    "a person sidesteps to the left.", "a person walks forward and turns around.",
    "a person jumps up in place.", "a person jumps forward.", "a person does a long jump forward.",
    "a person hops on their left leg.", "a person hops on their right leg.", "a person jumps over an obstacle.",
    "a person skips forward.", "a person does jumping jacks.", "a person jumps up and spins in the air.",
    "a person leaps to the side.", "a person runs and jumps.", "a person jumps backward.",
    "a person jumps with both feet three times.", "a person jumps and lands on one foot.",
    "a person stands on one leg.", "a person balances on their right foot with arms out.",
    "a person leans forward.", "a person leans to the left.", "a person stands on their toes.",
    "a person walks on a balance beam.", "a person loses balance and catches themselves.",
    "a person stumbles forward.", "a person bends down to touch their toes.", "a person squats down and stands up.",
    "a person does a lunge forward.", "a person kicks with their right leg.", "a person kicks high with their left leg.",
    "a person does a yoga tree pose.", "a person tiptoes forward.", "a person runs and then stops suddenly.",
    "a person waves with their right hand.", "a person throws a ball.", "a person picks something up from the ground.",
    "a person reaches up high.", "a person claps their hands.", "a person stretches their arms.",
    "a person punches forward.", "a person boxes.", "a person swings a golf club.", "a person kicks a soccer ball.",
    "a person dances.", "a person does a spin.", "a person marches in place.", "a person crouches and walks.",
    "a person walks while carrying something heavy.", "a person dribbles a basketball and shoots.",
    "a person plays tennis.", "a person shuffles side to side.", "a person steps forward and back.",
    "a person turns in place.", "a person stands still.", "a person stretches their legs.",
    "a person shakes their body.", "a person runs in a zigzag.", "a person jogs in place.", "a person sprints.",
]

def test_captions(n, seed=0):
    ids = [l.strip() for l in open(os.path.join(HML_DIR, "test.txt")) if l.strip()]
    rng = np.random.default_rng(seed)
    out, seen = [], set(CURATED)
    for i in rng.permutation(len(ids)):
        path = os.path.join(HML_DIR, "texts", ids[i] + ".txt")
        if not os.path.exists(path):
            continue
        lines = [l.split("#")[0].strip() for l in open(path) if l.strip()]
        lines = [c for c in lines if 3 <= len(c.split()) <= 20 and c not in seen]
        if lines:
            cap = lines[rng.integers(len(lines))]
            out.append(cap if cap.endswith(".") else cap + "."); seen.add(cap)
        if len(out) == n:
            break
    return out

PROMPTS = CURATED + test_captions(N_TEST_CAPTIONS)
SOURCE = np.array(["curated"] * len(CURATED) + ["test"] * (len(PROMPTS) - len(CURATED)))
all_prompts = [p for _ in range(SAMPLES_PER_PROMPT) for p in PROMPTS]
prompt_id = np.array([i for _ in range(SAMPLES_PER_PROMPT) for i in range(len(PROMPTS))])
print(f"{len(PROMPTS)} prompts ({len(CURATED)} curated + {len(PROMPTS) - len(CURATED)} test) x "
      f"{SAMPLES_PER_PROMPT} = {len(all_prompts)} motions per run. Examples:", PROMPTS[len(CURATED):len(CURATED) + 3])
""")

md("## 5. Generate and save (resumes automatically)")

code(RECORDER)

code(r"""
def run_dir(seed):
    d = os.path.join(DRIVE_ROOT, f"{CHECKPOINT}_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def generate_run(seed):
    d = run_dir(seed)
    meta = dict(checkpoint=CHECKPOINT, seed=seed, prompts=PROMPTS, all_prompts=all_prompts, prompt_id=prompt_id.tolist(),
                record_steps=RECORD_STEPS, frame_stride=FRAME_STRIDE, n_frames=N_FRAMES, batch=BATCH,
                n_layers=len(LAYERS), guidance=GUIDANCE)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    n_batches = -(-len(all_prompts) // BATCH)
    t0 = time.time()
    for b in range(n_batches):
        path = os.path.join(d, f"batch_{b:03d}.npz")
        if os.path.exists(path):
            continue
        j, f, acts, xt, x0, txt = sample_batch(all_prompts[b * BATCH:(b + 1) * BATCH], seed=seed * 100003 + b)
        arrs = dict(joints=j.astype(np.float32), feats=f[:, FR].astype(np.float16), txt=txt.astype(np.float16))
        for (k, li), v in acts.items():
            arrs[f"act_{k}_{li}"] = v
        for k in RECORD_STEPS:
            arrs[f"xt_{k}"] = xt[k]
            arrs[f"x0j_{k}"] = x0[k][0].astype(np.float16)
            arrs[f"x0f_{k}"] = x0[k][1][:, FR].astype(np.float16)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrs)
        os.replace(tmp, path)   # only complete batches ever appear under the final name
        print(f"seed {seed}: batch {b + 1}/{n_batches} saved ({time.time() - t0:.0f}s)")
    print(f"seed {seed}: all {n_batches} batches present in {d}")


for s in RUN_SEEDS:
    generate_run(s)
""")

md("""
## 6. Physics labels

Same definitions as phase 1: gravity during flight (the cleanest physics test), dynamic and static balance margins, lowest-foot height and flight. Labels for each run are cached on Drive. Balance margins are probed as continuous quantities; "XCoM out %" also flags normal stepping, so it is descriptive, not a failure rate.
""")

code(LABEL_FUNCS + r"""

def batch_files(seed):
    return sorted(glob.glob(os.path.join(run_dir(seed), "batch_*.npz")))


def load_key(seed, key):
    return np.concatenate([np.load(p)[key] for p in batch_files(seed)])


def run_labels(seed):
    path = os.path.join(run_dir(seed), "labels.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["final"].item(), z["x0"].item()
    final = labels_for(load_key(seed, "joints"))
    x0 = {k: labels_for(load_key(seed, f"x0j_{k}").astype(np.float32)) for k in RECORD_STEPS}
    np.savez(path, final=np.array(final, dtype=object), x0=np.array(x0, dtype=object))
    return final, x0


LABELS = {}
for s in RUN_SEEDS:
    t0 = time.time()
    LABELS[s] = run_labels(s)
    print(f"seed {s}: labels ready ({time.time() - t0:.0f}s)")
""")

code(r"""
# Failure rates by prompt group, per run (sanity check that the runs agree)
def group(p, src):
    if src == "test":
        return "test captions"
    if any(w in p for w in ["jump", "hop", "leap", "skip"]):
        return "curated: flight"
    if any(w in p for w in ["one leg", "balanc", "lean", "toes", "tiptoe", "stumble", "lunge", "kick",
                            "tree pose", "squat", "bends down", "stops suddenly", "right foot"]):
        return "curated: balance"
    return "curated: other"

GROUPS = np.array([group(all_prompts[i], SOURCE[prompt_id[i]]) for i in range(len(all_prompts))])
for s in RUN_SEEDS:
    LAB = LABELS[s][0]
    print(f"\nseed {s}")
    print(f"{'group':18s} {'motions':>7s} {'flight %':>9s} {'float>5cm %':>12s} {'XCoM out %':>11s} "
          f"{'airborne %':>11s} {'floating %':>11s} {'jump g m/s2':>12s}")
    for g in ["curated: flight", "curated: balance", "curated: other", "test captions"]:
        m = GROUPS == g
        ground = LAB["in_flight"][m] == 0
        xm = LAB["xcom_margin"][m]
        print(f"{g:18s} {m.sum():7d} {100 * LAB['in_flight'][m].mean():9.1f} "
              f"{100 * (LAB['lowest_foot'][m][ground] > 0.05).mean():12.1f} {100 * np.mean(xm[np.isfinite(xm)] < 0):11.1f} "
              f"{100 * LAB['airborne'][m].mean():11.1f} {100 * (LAB['flight_gravity'][m] < FLOAT_G_MAX).mean():11.1f} "
              f"{np.nanmedian(np.where(LAB['flight_gravity'][m] >= FLOAT_G_MAX, LAB['flight_gravity'][m], np.nan)):12.2f}")
""")

md("""
## 7. Per-frame probes, cross-validated by prompt

For every run, step and layer: a ridge probe predicts the **final** motion's per-frame label, with 5 folds that never share a prompt between train and test. The same baselines as phase 1 (noisy input, the model's clean-motion guess, and labels computed from that guess). Results are cached per run.
""")

code(r"""
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

TARGETS = {"flight_gravity": "r2", "xcom_margin": "r2", "balance_margin": "r2", "lowest_foot": "r2", "in_flight": "auc"}
gkf = GroupKFold(n_splits=5)
FR_LAB = np.arange(0, N_FRAMES, FRAME_STRIDE)


def flat(a):
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])

GROUPS_FRAME = flat(np.repeat(prompt_id[:, None], len(FR_LAB), 1))


def enough(ok):
    # Prompt-grouped 5-fold CV needs at least 5 prompts and a few dozen frames with a defined label.
    return ok.sum() >= 50 and len(np.unique(GROUPS_FRAME[ok])) >= 5


def cv_score(X, y, ok, kind):
    if not enough(ok):
        return np.nan
    X = X.astype(np.float32)
    pred = np.full(len(y), np.nan)
    idx = np.where(ok)[0]
    for tr, te in gkf.split(idx, groups=GROUPS_FRAME[idx]):
        tr, te = idx[tr], idx[te]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(0, 4, 5)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    if kind == "auc":
        return roc_auc_score(y[ok], pred[ok]) if len(np.unique(y[ok])) > 1 else np.nan
    return r2_score(y[ok], pred[ok])


def explicit_score(xe, y, ok, kind):
    if not enough(ok):
        return np.nan
    fill = np.nanmedian(xe[ok]) if np.isfinite(xe[ok]).any() else 0.0
    xe = np.where(np.isfinite(xe), xe, fill)
    if kind == "auc":
        return roc_auc_score(y[ok], xe[ok]) if len(np.unique(y[ok])) > 1 else np.nan
    pred = np.full(len(y), np.nan)
    idx = np.where(ok)[0]
    for tr, te in gkf.split(idx, groups=GROUPS_FRAME[idx]):
        tr, te = idx[tr], idx[te]
        pred[te] = LinearRegression().fit(xe[tr, None], y[tr]).predict(xe[te, None])
    return r2_score(y[ok], pred[ok])


def probe_run(seed):
    path = os.path.join(run_dir(seed), "probe_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    final, x0lab = LABELS[seed]
    rows, t0 = [], time.time()
    for k in RECORD_STEPS:
        files = batch_files(seed)
        acts = {li: np.concatenate([np.load(p)[f"act_{k}_{li}"] for p in files]) for li in range(len(LAYERS))}
        xt = np.concatenate([np.load(p)[f"xt_{k}"] for p in files])
        x0f = np.concatenate([np.load(p)[f"x0f_{k}"] for p in files])
        for name, kind in TARGETS.items():
            y = flat(final[name][:, FR_LAB]).astype(float)
            ok = np.isfinite(y)
            rows.append(dict(seed=seed, target=name, step=k, source="x_t probe", score=cv_score(flat(xt), y, ok, kind)))
            rows.append(dict(seed=seed, target=name, step=k, source="x0 probe", score=cv_score(flat(x0f), y, ok, kind)))
            rows.append(dict(seed=seed, target=name, step=k, source="explicit",
                             score=explicit_score(flat(x0lab[k][name][:, FR_LAB]).astype(float), y, ok, kind)))
            for li in range(len(LAYERS)):
                rows.append(dict(seed=seed, target=name, step=k, source=f"layer {li}",
                                 score=cv_score(flat(acts[li]), y, ok, kind)))
        del acts
        print(f"seed {seed}: step {k} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


RES = pd.concat([probe_run(s) for s in RUN_SEEDS], ignore_index=True)
""")

code(r"""
# Mean over runs (lines) with min-max across runs (bands)
for name, kind in TARGETS.items():
    d = RES[RES.target == name]
    layer_best = (d[d.source.str.startswith("layer")].groupby(["seed", "step"]).score.max()
                  .reset_index().assign(source="best layer"))
    dd = pd.concat([layer_best, d[d.source.isin(["x0 probe", "explicit", "x_t probe"])]])
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=90)
    for src in ["best layer", "x0 probe", "explicit", "x_t probe"]:
        g = dd[dd.source == src].groupby("step").score
        ax.plot(RECORD_STEPS, g.mean().reindex(RECORD_STEPS), "o-" if src == "best layer" else "o--", label=src)
        ax.fill_between(RECORD_STEPS, g.min().reindex(RECORD_STEPS), g.max().reindex(RECORD_STEPS), alpha=0.15)
    lo = 0.5 if kind == "auc" else 0.0
    ax.set_ylim(lo - 0.1, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xlabel(f"denoising step (0 = noise, {N_STEPS - 1} = final)"); ax.set_ylabel(kind.upper())
    ax.set_title(f"{name}: mean over {len(RUN_SEEDS)} runs, band = range")
    plt.tight_layout(); plt.show()

summary = (RES.groupby(["target", "source", "step"]).score.agg(["mean", "std"]).round(3)
           .unstack("step"))
display(summary)
""")

md("""
## 8. Early failure prediction, across runs

As in phase 1: from frame-averaged activations at each step, predict each motion's final **gravity error (jumps)** (motions with a falling airborne stretch only), **floating rate** (off the ground but barely falling), **float rate** and exploratory **off-balance rate**, against the text-only baseline. **Within-prompt** correlation (samples of the same prompt ranked against each other) is the cleanest test of sample-specific knowledge; with 4 samples per prompt it is now much better powered than in phase 1.
""")

code(r"""
def failure_rates(lab):
    ground = lab["in_flight"] == 0
    flt = np.where(ground, lab["lowest_foot"] > 0.05, np.nan).astype(float)
    xm = lab["xcom_margin"]
    bal = np.where(np.isfinite(xm), xm < 0, np.nan).astype(float)
    g = lab["flight_gravity"]
    # Unsupported stretches split in two: jumps (judged by how far their fall rate is from 9.81)
    # and floating (the body is off the ground but barely falls).
    jump_err = np.where(g >= FLOAT_G_MAX, np.abs(g - G_EARTH), np.nan)
    floating = (np.isfinite(g) & (g < FLOAT_G_MAX)).astype(float)
    with np.errstate(all="ignore"):
        return {"gravity error (jumps)": np.nanmean(jump_err, 1), "floating rate": floating.mean(1),
                "float rate": np.nanmean(flt, 1), "off-balance rate (exploratory)": np.nanmean(bal, 1)}


def within_prompt(pred, y, ok):
    rp, ry, keep = pred.copy(), y.copy(), np.zeros(len(y), bool)
    for pid in np.unique(prompt_id[ok]):
        m = ok & (prompt_id == pid)
        if m.sum() >= 2:
            rp[m] -= rp[m].mean(); ry[m] -= ry[m].mean(); keep |= m
    return spearmanr(rp[keep], ry[keep]).correlation if keep.sum() >= 10 else np.nan


def oof_ridge(X, y, ok, alpha=10.0):
    # Fits only on samples where the outcome exists (e.g. gravity error (jumps) needs a jump).
    X = X.astype(np.float32)
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=prompt_id):
        tr = tr[ok[tr]]
        if len(tr) < 10:
            pred[te] = np.nan
            continue
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred


def early_run(seed):
    path = os.path.join(run_dir(seed), "early_prediction.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    final, x0lab = LABELS[seed]
    fail = failure_rates(final)
    fail_x0 = {k: failure_rates(x0lab[k]) for k in RECORD_STEPS}
    txt = load_key(seed, "txt").astype(np.float32)
    rows = []
    pooled = {}
    for k in RECORD_STEPS:
        files = batch_files(seed)
        pooled[k] = {li: np.concatenate([np.load(p)[f"act_{k}_{li}"].astype(np.float32).mean(1) for p in files])
                     for li in range(len(LAYERS))}
        pooled[(k, "x0")] = np.concatenate([np.load(p)[f"x0f_{k}"].astype(np.float32).mean(1) for p in files])
    for fname, yv in fail.items():
        ok = np.isfinite(yv)
        y = np.where(ok, yv, np.nanmean(yv))
        def add(step, source, pred):
            good = ok & np.isfinite(pred)
            rows.append(dict(seed=seed, failure=fname, step=step, source=source, n=int(ok.sum()),
                             spearman=spearmanr(pred[good], y[good]).correlation if good.sum() >= 10 else np.nan,
                             within_prompt=within_prompt(pred, y, good)))
        add(-1, "text only", oof_ridge(txt, y, ok, alpha=100.0))
        for k in RECORD_STEPS:
            e = fail_x0[k][fname]
            add(k, "explicit (x0 guess)", np.where(np.isfinite(e), e, np.nanmean(e)))
            add(k, "x0 probe", oof_ridge(pooled[(k, "x0")], y, ok))
            for li in range(len(LAYERS)):
                add(k, f"layer {li}", oof_ridge(pooled[k][li], y, ok))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


EARLY = pd.concat([early_run(s) for s in RUN_SEEDS], ignore_index=True)

# Pick ONE layer in advance from seed RUN_SEEDS[0] (best mean within-prompt score), then report it on the other runs.
first = EARLY[(EARLY.seed == RUN_SEEDS[0]) & EARLY.source.str.startswith("layer")]
FIXED_LAYER = first.groupby("source").within_prompt.mean().fillna(-1).idxmax()
print("Layer fixed on the first run:", FIXED_LAYER, "(the other runs are an honest test of it)")

FAILS = ["gravity error (jumps)", "floating rate", "float rate", "off-balance rate (exploratory)"]
fig, axes = plt.subplots(1, len(FAILS), figsize=(5.5 * len(FAILS), 3.6), dpi=90)
for ax, fname in zip(axes, FAILS):
    d = EARLY[EARLY.failure == fname]
    txt = d[d.source == "text only"].spearman.mean()
    for src, style in [(FIXED_LAYER, "o-"), ("x0 probe", "o--"), ("explicit (x0 guess)", "o--")]:
        g = d[d.source == src].groupby("step")
        ax.plot(RECORD_STEPS, g.spearman.mean().reindex(RECORD_STEPS), style, label=src)
        if src == FIXED_LAYER:
            ax.plot(RECORD_STEPS, g.within_prompt.mean().reindex(RECORD_STEPS), "o:", label=f"{src}, within prompt")
    ax.axhline(txt, color="gray", ls="--", label="text only")
    ax.set_title(f"predicting final {fname} (mean over runs)"); ax.set_xlabel("denoising step (0 = noise)")
    ax.set_ylabel("Spearman ρ"); ax.set_ylim(-0.3, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

other = EARLY[EARLY.seed != RUN_SEEDS[0]] if len(RUN_SEEDS) > 1 else EARLY
display(other[other.source.isin([FIXED_LAYER, "x0 probe", "explicit (x0 guess)", "text only"])]
        .pivot_table(index=["failure", "source"], columns="step", values=["spearman", "within_prompt"]).round(3))
""")

md("""
## 9. Compare checkpoints (optional)

After running this notebook with `CHECKPOINT = "50steps"`, set `CHECKPOINT = "1000steps"` in section 1 and run all again. Both runs stay on Drive, and this cell compares them once both exist.
""")

code(r"""
frames = []
for ck in ["50steps", "1000steps"]:
    for p in glob.glob(os.path.join(DRIVE_ROOT, f"{ck}_seed*", "probe_scores.csv")):
        frames.append(pd.read_csv(p).assign(checkpoint=ck))
if len({f.checkpoint.iloc[0] for f in frames}) < 2:
    print("Only one checkpoint has results so far; rerun with the other CHECKPOINT to compare.")
else:
    allres = pd.concat(frames)
    best = (allres[allres.source.str.startswith("layer")].groupby(["checkpoint", "seed", "target", "step"]).score.max()
            .reset_index())
    best["progress"] = best.groupby(["checkpoint"]).step.transform(lambda s: s / s.max())  # 0 = noise, 1 = final
    display(best.groupby(["target", "checkpoint", "progress"]).score.mean().unstack("progress").round(3))
""")

code(r"""
print("All results are on Drive in", DRIVE_ROOT)
for p in sorted(glob.glob(os.path.join(DRIVE_ROOT, "*", "*.csv"))):
    print(" ", p)
""", tag="save")

md("""
## 10. What to send back

Screenshots of the group table (section 6), the four probe plots and the summary table (section 7), and the early-prediction plot and table (section 8). The CSV files on Drive hold every number if we need more detail.

**Go/no-go for phase 3:** results should look similar across runs (narrow bands) and on test captions as on curated prompts. If they do, the phase 1 finding is robust and we move to causal tests.
""")


def build(path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    for kind, tag, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path)


if __name__ == "__main__":
    build(os.path.join(NB_DIR, "phase2_scale_robustness.ipynb"))
    print("built")
