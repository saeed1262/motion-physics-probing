"""Post-hoc controls for the phase 2 dynamic-balance (XCoM margin) result. Reads cached arrays only.

The phase 2 result: early in denoising, a linear probe on layer 7 predicts the final motion's
XCoM margin better than the label computed from the model's own clean-motion guess ("explicit").
Two ways that could be misleading, and the check for each:

1. Nonlinear readout. The XCoM margin is a nonlinear function of the motion, so a *linear* probe
   on the guess's 263 features can't compute it, and "explicit" can't correct the guess's errors.
   Check: the same small MLP on (a) the guess's features, (b) the guess's features plus every
   physics label computed from the guess ("everything the guess shows"), (c) layer 7, and (d) (b)
   plus layer 7. If (c) or (d) beat (b), the layer holds information the guess does not show.

2. Phase 1 disagreed (explicit won at step 10). Check: (a) phase 1's exact split (train on 52
   curated prompts, test on its 12) applied to phase 2's arrays; (b) a learning curve, training
   the linear probe and explicit baseline on 13 to all ~200 training prompts per fold.

Layer 7 is fixed in advance: it was the best layer for XCoM margin at 20% of denoising in every
phase 2 run. Folds are the phase 2 notebook's (GroupKFold(5) by prompt).

    .venv/bin/python scripts/phase2_controls.py --checkpoint 50steps --steps 10,20
    .venv/bin/python scripts/phase2_controls.py --checkpoint 1000steps --steps 200,400 --skip-curve
    .venv/bin/python scripts/phase2_controls.py --target in_flight --layer 6 --steps 10,20 --skip-curve

Writes results/<tag>/controls/<checkpoint>_<target>_nonlinear_controls.csv (and, for
xcom_margin without --skip-curve, <checkpoint>_training_size.csv). The learning curve is
written for xcom_margin only.
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="50steps")
ap.add_argument("--steps", default="10,20")
ap.add_argument("--tag", default="phase2", help="phase2 or phase2_smoke")
ap.add_argument("--layer", type=int, default=7)
ap.add_argument("--target", default="xcom_margin",
                choices=["xcom_margin", "balance_margin", "lowest_foot", "in_flight", "flight_gravity"])
ap.add_argument("--with-text", action="store_true",
                help="also give the guess-side readout the prompt's CLIP embedding (the model's text input)")
ap.add_argument("--skip-curve", action="store_true", help="skip the phase-1 split and learning curve")
ap.add_argument("--save-preds", action="store_true", help="also save out-of-fold predictions (for confidence intervals)")
CLI = ap.parse_args()
OUT = os.path.join(ROOT, "results", CLI.tag, "controls")
os.makedirs(OUT, exist_ok=True)
TARGET = CLI.target
KIND = "auc" if TARGET == "in_flight" else "r2"   # same metrics as the notebooks
X0_LABELS = ["xcom_margin", "balance_margin", "lowest_foot", "in_flight", "airborne", "skate"]
X0_LABELS += [TARGET] if TARGET not in X0_LABELS else []   # the guess's own value of the target
PHASE1_TEST = json.load(open(os.path.join(ROOT, "results", "phase1", "run_info.json")))["test_prompts"]
gkf = GroupKFold(n_splits=5)
torch.set_num_threads(8)


def score(y, p):
    return roc_auc_score(y, p) if KIND == "auc" else r2_score(y, p)


def within_prompt_r(y, p, g):
    # Pearson r after subtracting each prompt's mean from both target and prediction: only
    # sample-specific information (not what the prompt alone predicts) can score here.
    yc, pc = y.copy(), p.copy()
    for pid in np.unique(g):
        m = g == pid
        yc[m] -= y[m].mean(); pc[m] -= p[m].mean()
    return np.corrcoef(yc, pc)[0, 1]


def flat(a):
    return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])


def ridge_fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return RidgeCV(alphas=np.logspace(0, 4, 5)).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def explicit_fit_predict(xtr, ytr, xte):
    return LinearRegression().fit(xtr[:, None], ytr).predict(xte[:, None])


def mlp_fit_predict(Xtr, ytr, gtr, Xte, seed=0):
    # Same small MLP for every feature set: 2 x 256 ReLU, dropout 0.1, AdamW, early stopping on
    # 15% of the *training prompts* (grouped, so validation never shares a prompt with training).
    torch.manual_seed(seed)
    vi_tr, vi_va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(Xtr, groups=gtr))
    sx = StandardScaler().fit(Xtr[vi_tr])
    my, sy = ytr[vi_tr].mean(), ytr[vi_tr].std() + 1e-8
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xa, ya = t(sx.transform(Xtr[vi_tr])), t((ytr[vi_tr] - my) / sy)
    Xv, yv = t(sx.transform(Xtr[vi_va])), t((ytr[vi_va] - my) / sy)
    net = nn.Sequential(nn.Linear(Xa.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(100):
        net.train()
        for idx in torch.randperm(len(Xa)).split(512):
            opt.zero_grad()
            nn.functional.mse_loss(net(Xa[idx]).squeeze(1), ya[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(Xv).squeeze(1), yv).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(t(sx.transform(Xte))).squeeze(1).numpy() * sy + my


def x0_shown(x0lab_k, FR):
    # Every physics label computed from the guess at each frame, NaNs filled with 0 plus a
    # missing-value indicator, and the 4 foot-contact flags.
    cols = []
    for name in X0_LABELS:
        v = flat(x0lab_k[name][:, FR]).astype(float)
        cols += [np.nan_to_num(v), (~np.isfinite(v)).astype(float)]
    cols.append(flat(x0lab_k["contact"][:, FR].astype(float)))
    return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])


rows, curve = [], []
for d in sorted(glob.glob(os.path.join(ROOT, "data", CLI.tag, f"{CLI.checkpoint}_seed*"))):
    meta = json.load(open(os.path.join(d, "meta.json")))
    seed = meta["seed"]
    prompt_id = np.array(meta["prompt_id"])
    FR = np.arange(0, meta["n_frames"], meta["frame_stride"])
    groups = flat(np.repeat(prompt_id[:, None], len(FR), 1))
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    final, x0lab = lab["final"].item(), lab["x0"].item()
    y = flat(final[TARGET][:, FR]).astype(float)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    folds = [(idx[tr], idx[te]) for tr, te in gkf.split(idx, groups=groups[idx])]
    for k in [int(s) for s in CLI.steps.split(",")]:
        t0 = time.time()
        L = flat(np.concatenate([np.load(p)[f"act_{k}_{CLI.layer}"] for p in files])).astype(np.float32)
        X0F = flat(np.concatenate([np.load(p)[f"x0f_{k}"] for p in files])).astype(np.float32)
        SHOWN = np.hstack([X0F, x0_shown(x0lab[k], FR)]).astype(np.float32)
        xe = flat(x0lab[k][TARGET][:, FR]).astype(float)
        xe = np.where(np.isfinite(xe), xe, np.nanmedian(xe[ok]))
        feats = {"guess features": X0F, "everything the guess shows": SHOWN, f"layer {CLI.layer}": L,
                 f"everything the guess shows + layer {CLI.layer}": np.hstack([SHOWN, L])}
        if CLI.with_text:   # the prompt is an input the readout may use too: separates prompt info from physics
            TXT = np.concatenate([np.load(p)["txt"] for p in files]).astype(np.float32)
            TXTF = np.repeat(TXT, len(FR), axis=0)
            feats = {"everything the guess shows + prompt": np.hstack([SHOWN, TXTF]),
                     f"everything the guess shows + prompt + layer {CLI.layer}": np.hstack([SHOWN, TXTF, L])}
        preds = {}
        p = np.full(len(y), np.nan)
        for tr, te in folds:   # for AUC the notebooks score the raw explicit value
            p[te] = xe[te] if KIND == "auc" else explicit_fit_predict(xe[tr], y[tr], xe[te])
        preds[("explicit", "linear")] = p
        for name, X in feats.items():
            for model in ["linear", "mlp"]:
                p = np.full(len(y), np.nan)
                for fi, (tr, te) in enumerate(folds):
                    p[te] = (ridge_fit_predict(X[tr], y[tr], X[te]) if model == "linear"
                             else mlp_fit_predict(X[tr], y[tr], groups[tr], X[te], seed=fi))
                preds[(name, model)] = p
        for (name, model), p in preds.items():
            rows.append(dict(checkpoint=CLI.checkpoint, target=TARGET, metric=KIND, seed=seed, step=k,
                             features=name, model=model, score=score(y[ok], p[ok]),
                             within_prompt_r=within_prompt_r(y[ok], p[ok], groups[ok])))
        print(f"seed {seed} step {k}: controls done ({time.time() - t0:.0f}s)", flush=True)
        if CLI.save_preds:   # out-of-fold predictions, for prompt-bootstrap confidence intervals
            pd_dir = os.path.join(OUT, "preds")
            os.makedirs(pd_dir, exist_ok=True)
            np.savez_compressed(os.path.join(pd_dir, f"{CLI.checkpoint}_{TARGET}_seed{seed}_step{k}.npz"),
                                y=y[ok], groups=groups[ok], **{f"{n} | {m}": p[ok] for (n, m), p in preds.items()})

        if CLI.skip_curve:
            continue
        # (a) phase 1's split: curated prompts only, train on 52, test on phase 1's 12.
        cur = ok & (groups < 64)
        te_m, tr_m = cur & np.isin(groups, PHASE1_TEST), cur & ~np.isin(groups, PHASE1_TEST)
        for name, fn in [("explicit", lambda tr, te: explicit_fit_predict(xe[tr], y[tr], xe[te])),
                         (f"layer {CLI.layer} (linear)", lambda tr, te: ridge_fit_predict(L[tr], y[tr], L[te]))]:
            curve.append(dict(checkpoint=CLI.checkpoint, seed=seed, step=k, design="phase 1 split (52 curated train, 12 test)",
                              n_train_prompts=52, rep=0, source=name, r2=r2_score(y[te_m], fn(tr_m, te_m))))
        # (b) learning curve over the number of training prompts, same test folds.
        rng = np.random.default_rng(seed)
        min_train = min(len(np.unique(groups[tr])) for tr, _ in folds)
        for n_prompts in [n for n in [13, 26, 52, 100] if n < min_train] + [None]:
            for rep in range(1 if n_prompts is None else 3):
                pe, pl = np.full(len(y), np.nan), np.full(len(y), np.nan)
                for tr, te in folds:
                    pids = np.unique(groups[tr])
                    keep = pids if n_prompts is None else rng.choice(pids, size=n_prompts, replace=False)
                    tr = tr[np.isin(groups[tr], keep)]
                    pe[te] = explicit_fit_predict(xe[tr], y[tr], xe[te])
                    pl[te] = ridge_fit_predict(L[tr], y[tr], L[te])
                for name, p in [("explicit", pe), (f"layer {CLI.layer} (linear)", pl)]:
                    curve.append(dict(checkpoint=CLI.checkpoint, seed=seed, step=k, design="5-fold, subsampled training prompts",
                                      n_train_prompts=n_prompts or len(np.unique(groups[folds[0][0]])), rep=rep,
                                      source=name, r2=r2_score(y[ok], p[ok])))
        print(f"seed {seed} step {k}: split check and learning curve done ({time.time() - t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{CLI.checkpoint}_{TARGET}{'_with_text' if CLI.with_text else ''}_nonlinear_controls.csv"), index=False)
if curve:
    pd.DataFrame(curve).to_csv(os.path.join(OUT, f"{CLI.checkpoint}_training_size.csv"), index=False)
print("wrote", OUT)
