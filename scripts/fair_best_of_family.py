"""Best-of-family gains from the tuned readouts' saved predictions (fair_readouts.py).

The "tuned" readout picks ridge or MLP by validation error, which can miss the family that
scores best on the test folds (selection is by MSE; flight is scored by AUC). As a check that
favors neither side, this takes, for each feature set, whichever of its ridge, MLP and tuned
readouts scores best on the pooled test frames, and compares:

    gain: best(estimate + layer) - best(estimate)
    fair: best(estimate) - best(layer)

with 95% intervals from the same prompt bootstrap as fair_readouts.py (the family choice is made
once on the full data and kept fixed in every replicate, so the intervals ignore the selection).

    .venv/bin/python scripts/fair_best_of_family.py [--tag phase2]

Writes results/<tag>/fair/best_of_family.csv.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="phase2")
ap.add_argument("--reps", type=int, default=1000)
CLI = ap.parse_args()
FAIR = os.path.join(ROOT, "results", CLI.tag, "fair")


def wcorr(a, b, w):
    w = w / w.sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())


def centre(v, cell, w):
    return v - (np.bincount(cell, w * v) / np.maximum(np.bincount(cell, w), 1e-12))[cell]


def scores(y, p, g, cell, auc, w):
    m = w > 0
    y, p, g, cell, w = y[m], p[m], g[m], cell[m], w[m]
    main = roc_auc_score(y, p, sample_weight=w) if auc else r2_score(y, p, sample_weight=w)
    return np.array([main, wcorr(centre(y, g, w), centre(p, g, w), w), wcorr(centre(y, cell, w), centre(p, cell, w), w)])


rows = []
rng = np.random.default_rng(0)
for f in sorted(glob.glob(os.path.join(FAIR, "preds", "*.npz"))):
    ck, target = os.path.basename(f)[:-4].split("_", 1)
    z = np.load(f)
    y, g, fr = z["y"], z["groups"], z["frame"]
    cell = np.unique(g * 1000 + fr, return_inverse=True)[1]
    auc = target == "in_flight"
    one = np.ones(len(y))
    best = {}
    for fs in ["estimate", "layer", "estimate + layer"]:
        cand = {fam: scores(y, z[f"{fs} | {fam}"], g, cell, auc, one)[0] for fam in ["ridge", "mlp", "tuned"]}
        best[fs] = max(cand, key=cand.get)
    keys = {fs: f"{fs} | {fam}" for fs, fam in best.items()}
    prompts = np.unique(g)
    pos = np.searchsorted(prompts, g)
    obs = {fs: scores(y, z[k], g, cell, auc, one) for fs, k in keys.items()}
    boot = []
    for _ in range(CLI.reps):
        w = np.bincount(rng.integers(len(prompts), size=len(prompts)), minlength=len(prompts))[pos].astype(float)
        boot.append({fs: scores(y, z[k], g, cell, auc, w) for fs, k in keys.items()})
    for name, a, b in [("gain: best(estimate + layer) - best(estimate)", "estimate + layer", "estimate"),
                       ("fair: best(estimate) - best(layer)", "estimate", "layer")]:
        for mi, meas in enumerate(["score", "within_prompt_r", "within_prompt_frame_r"]):
            d = np.array([bt[a][mi] - bt[b][mi] for bt in boot])
            rows.append(dict(tag=CLI.tag, checkpoint=ck, target=target, comparison=name, measure=meas,
                             families=f"{a}: {best[a]}, {b}: {best[b]}", observed=obs[a][mi] - obs[b][mi],
                             ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    print(ck, target, best, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(FAIR, "best_of_family.csv"), index=False)
pd.set_option("display.width", 250)
print(df[df.measure != "within_prompt_r"].round(3).to_string())
