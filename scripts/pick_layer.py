"""Pick the probed layer per target for a checkpoint without phase 2 linear probes (MotionDiffuse), by the
same rule used for MDM: the layer whose linear probe scores best at 20% of denoising.

Ridge on one layer's per-frame activations, inputs standardized, penalty from {1, 10, ..., 1e4} chosen
on 5 folds grouped by prompt (seeds pooled); score R2, or AUC for in_flight. The choice never looks at
the estimate, so it cannot favour the conditional gain.

    .venv/bin/python scripts/pick_layer.py --checkpoint motiondiffuse [--tag phase2]

Writes results/<tag>/<checkpoint>_layers.json ({target: layer}) and _layer_scores.csv.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="motiondiffuse")
ap.add_argument("--tag", default="phase2")
ap.add_argument("--step", type=int, default=200, help="recorded step at 20% of denoising")
CLI = ap.parse_args()
TARGETS = {"xcom_margin": "r2", "balance_margin": "r2", "in_flight": "auc", "flight_gravity": "r2"}
ALPHAS = 10.0 ** np.arange(5)

runs = sorted(glob.glob(os.path.join(ROOT, "data", CLI.tag, f"{CLI.checkpoint}_seed*")))
assert runs, "no runs"
rows, best = [], {}
for target, kind in TARGETS.items():
    Y, G, A = [], [], {l: [] for l in range(8)}
    for d in runs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        FR = np.arange(0, meta["n_frames"], meta["frame_stride"])
        y = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)["final"].item()[target][:, FR].astype(float)
        m = np.isfinite(y)
        files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
        Y.append(y[m]); G.append(np.repeat(np.array(meta["prompt_id"])[:, None], len(FR), 1)[m])
        for l in range(8):
            A[l].append(np.concatenate([np.load(p)[f"act_{CLI.step}_{l}"] for p in files])[m].astype(np.float32))
    y, g = np.concatenate(Y), np.concatenate(G)
    folds = list(GroupKFold(n_splits=5).split(y, groups=g))
    for l in range(8):
        X = np.concatenate(A[l]); A[l] = None
        pred = {a: np.zeros(len(y)) for a in ALPHAS}
        for tr, te in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ym = y[tr].mean()
            lam, V = np.linalg.eigh(Xtr.T.astype(np.float64) @ Xtr)
            Vty = V.T @ (Xtr.T.astype(np.float64) @ (y[tr] - ym))
            for a in ALPHAS:
                pred[a][te] = Xte @ (V @ (Vty / (lam + a))) + ym
        score = lambda p: roc_auc_score(y, p) if kind == "auc" else r2_score(y, p)
        s = max(score(p) for p in pred.values())
        rows.append(dict(checkpoint=CLI.checkpoint, target=target, layer=l, score=s, frames=len(y)))
        print(f"{target} layer {l}: {s:.3f}", flush=True)
    d_ = [r for r in rows if r["target"] == target]
    best[target] = int(max(d_, key=lambda r: r["score"])["layer"])
best.update(float_vs_fall=best["flight_gravity"], fall_gravity=best["flight_gravity"])   # derived from gravity, as for MDM
out = os.path.join(ROOT, "results", CLI.tag)
os.makedirs(out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out, f"{CLI.checkpoint}_layer_scores.csv"), index=False)
json.dump(best, open(os.path.join(out, f"{CLI.checkpoint}_layers.json"), "w"), indent=1)
print("chosen:", best)
