"""Post-hoc phase 2 check: do probe scores hold on HumanML3D test captions as on curated prompts?

The phase 2 notebook scores each probe on all prompts pooled. This reruns the same prompt-grouped
5-fold cross-validation (same RidgeCV alphas, same folds, same explicit baseline) on the cached
phase 2 arrays, keeps the out-of-fold predictions, and scores curated-prompt frames and
test-caption frames separately. It reads data/phase2/<checkpoint>_seed*/ and needs no model.

    .venv/bin/python scripts/phase2_by_source.py [--checkpoint 50steps] [--steps 0,10,20]

Writes results/phase2/<checkpoint>/probe_scores_by_source.csv.
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="50steps")
ap.add_argument("--tag", default="phase2", help="phase2 or phase2_smoke")
ap.add_argument("--steps", default=None, help="comma-separated recorded steps (default: the first three)")
CLI = ap.parse_args()
OUT = os.path.join(ROOT, "results", CLI.tag, CLI.checkpoint)
info = json.load(open(os.path.join(OUT, "run_info.json")))
N_CURATED = info["n_curated"]

TARGETS = {"flight_gravity": "r2", "xcom_margin": "r2", "balance_margin": "r2", "lowest_foot": "r2", "in_flight": "auc"}
gkf = GroupKFold(n_splits=5)


def flat(a):
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])


def score(kind, y, p):
    if kind == "auc":
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return r2_score(y, p)


rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "data", CLI.tag, f"{CLI.checkpoint}_seed*"))):
    meta = json.load(open(os.path.join(d, "meta.json")))
    seed, steps = meta["seed"], meta["record_steps"]
    steps = [int(s) for s in CLI.steps.split(",")] if CLI.steps else steps[:3]
    prompt_id = np.array(meta["prompt_id"])
    FR_LAB = np.arange(0, meta["n_frames"], meta["frame_stride"])
    groups = flat(np.repeat(prompt_id[:, None], len(FR_LAB), 1))
    is_test = groups >= N_CURATED          # curated prompts come first in PROMPTS
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    final, x0lab = lab["final"].item(), lab["x0"].item()
    t0 = time.time()
    for k in steps:
        feats = {f"layer {li}": np.concatenate([np.load(p)[f"act_{k}_{li}"] for p in files]) for li in range(meta["n_layers"])}
        feats["x0 probe"] = np.concatenate([np.load(p)[f"x0f_{k}"] for p in files])
        for name, kind in TARGETS.items():
            y = flat(final[name][:, FR_LAB]).astype(float)
            ok = np.isfinite(y)
            if ok.sum() < 50 or len(np.unique(groups[ok])) < 5:
                continue
            idx = np.where(ok)[0]
            folds = [(idx[tr], idx[te]) for tr, te in gkf.split(idx, groups=groups[idx])]
            preds = {}
            for src, X in feats.items():
                X = flat(X).astype(np.float32)
                pred = np.full(len(y), np.nan)
                for tr, te in folds:
                    sc = StandardScaler().fit(X[tr])
                    pred[te] = RidgeCV(alphas=np.logspace(0, 4, 5)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
                preds[src] = pred
            xe = flat(x0lab[k][name][:, FR_LAB]).astype(float)
            xe = np.where(np.isfinite(xe), xe, np.nanmedian(xe[ok]) if np.isfinite(xe[ok]).any() else 0.0)
            if kind == "auc":
                preds["explicit"] = xe
            else:
                pred = np.full(len(y), np.nan)
                for tr, te in folds:
                    pred[te] = LinearRegression().fit(xe[tr, None], y[tr]).predict(xe[te, None])
                preds["explicit"] = pred
            for src, pred in preds.items():
                for subset, m in [("all", ok), ("curated", ok & ~is_test), ("test captions", ok & is_test)]:
                    n = int(m.sum())
                    rows.append(dict(seed=seed, step=k, target=name, source=src, subset=subset, n_frames=n,
                                     score=score(kind, y[m], pred[m]) if n >= 20 else np.nan))
        print(f"seed {seed}: step {k} done ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "probe_scores_by_source.csv"), index=False)
print("wrote", os.path.join(OUT, "probe_scores_by_source.csv"))
