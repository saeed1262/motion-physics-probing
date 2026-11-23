"""95% confidence intervals for the key readout comparisons, by bootstrapping prompts.

Uses the out-of-fold predictions saved by `phase2_controls.py --save-preds`. Each bootstrap
replicate resamples the 250 prompts with replacement (the same prompts in every seed, since all
runs share the prompt list), recomputes every readout's score on the resampled frames in each
seed, averages over seeds, and takes the differences below. Reported: the observed difference,
the 2.5 and 97.5 percentiles over 2,000 replicates, and the share of replicates above zero.

Comparisons (at 20% of denoising):
- layer, linear minus explicit                        (the standard comparison)
- estimate, MLP minus layer, linear                   (the fair comparison)
- estimate + layer, MLP minus estimate, MLP           (does the layer add anything)
each for the pooled score and for the within-prompt correlation.

    .venv/bin/python scripts/bootstrap_ci.py

Writes results/phase2/controls/bootstrap_ci.csv.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="phase2"); ap.add_argument("--reps", type=int, default=2000)
CLI = ap.parse_args()
PRED = os.path.join(ROOT, "results", CLI.tag, "controls", "preds")
B = CLI.reps
EARLY = {"50steps": 10, "1000steps": 200}
COMPARISONS = [("layer, linear minus explicit", "layer | linear", "explicit | linear"),
               ("estimate, MLP minus layer, linear", "everything the guess shows | mlp", "layer | linear"),
               ("estimate + layer minus estimate, MLP", "everything the guess shows + layer | mlp", "everything the guess shows | mlp")]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def within_r(y, p, g, uniq_inv):
    # prompt-mean-centred Pearson r; uniq_inv maps each frame to its (resampled) prompt slot
    cnt = np.bincount(uniq_inv)
    yc = y - (np.bincount(uniq_inv, y) / cnt)[uniq_inv]
    pc = p - (np.bincount(uniq_inv, p) / cnt)[uniq_inv]
    return (yc * pc).sum() / np.sqrt((yc ** 2).sum() * (pc ** 2).sum())


def load(ck, target):
    runs = {}
    for f in sorted(glob.glob(os.path.join(PRED, f"{ck}_{target}_seed*_step{EARLY[ck]}.npz"))):
        z = np.load(f)
        keys = {re.sub(r"layer \d", "layer", k): z[k] for k in z.files if k not in ("y", "groups")}
        runs[int(re.search(r"seed(\d+)", f).group(1))] = (z["y"], z["groups"], keys)
    return runs


rows = []
rng = np.random.default_rng(0)
for ck in EARLY:
    for target in ["xcom_margin", "balance_margin", "in_flight", "flight_gravity"]:
        runs = load(ck, target)
        if not runs:
            continue
        auc = target == "in_flight"
        prompts = np.unique(np.concatenate([g for _, g, _ in runs.values()]))
        by_prompt = {s: {p: np.where(g == p)[0] for p in np.unique(g)} for s, (_, g, _) in runs.items()}

        def scores(sample):
            out = {}
            for s, (y, g, preds) in runs.items():
                parts = [(by_prompt[s][p], j) for j, p in enumerate(sample) if p in by_prompt[s]]
                idx = np.concatenate([ix for ix, _ in parts])
                slot = np.concatenate([np.full(len(ix), j) for ix, j in parts])
                _, inv = np.unique(slot, return_inverse=True)
                for key, p in preds.items():
                    sc = roc_auc_score(y[idx], p[idx]) if auc else r2(y[idx], p[idx])
                    wr = within_r(y[idx], p[idx], None, inv)
                    out.setdefault(key, [[], []]); out[key][0].append(sc); out[key][1].append(wr)
            return {k: (np.mean(v[0]), np.mean(v[1])) for k, v in out.items()}

        obs = scores(prompts)
        boot = [scores(rng.choice(prompts, size=len(prompts), replace=True)) for _ in range(B)]
        for name, a, b in COMPARISONS:
            for m, label in [(0, "pooled"), (1, "within prompt")]:
                d = np.array([bt[a][m] - bt[b][m] for bt in boot])
                rows.append(dict(checkpoint=ck, target=target, comparison=name, measure=label, seeds=len(runs),
                                 observed=obs[a][m] - obs[b][m], ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                                 share_above_zero=(d > 0).mean()))
        print(f"{ck} {target}: done ({len(runs)} seeds)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", CLI.tag, "controls", "bootstrap_ci.csv"), index=False)
pd.set_option("display.width", 220)
print(df.round(3).to_string())
