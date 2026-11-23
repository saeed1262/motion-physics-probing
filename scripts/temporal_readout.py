"""Temporal readout control: the fair comparison with a readout that sees the whole motion.

The per-frame readouts in phase2_controls.py see one frame at a time. Here the same comparison is
made with a small temporal network that reads all 30 recorded frames of a motion and predicts the
final motion's label at every frame: two dilated 1D convolutions over time (128 channels, kernel 5,
GELU, dropout 0.1) and a per-frame linear output, trained with AdamW on a masked loss (frames where
the label is undefined are ignored) and early-stopped on 15% of the training prompts.

Inputs at 20% of denoising: everything the estimate shows (its 263 features plus the labels computed
from it, as in phase2_controls.py), one layer's activations (the same fixed layer per target), or both.
Folds, metrics and the within-prompt score are the same as in phase2_controls.py.

    .venv/bin/python scripts/temporal_readout.py [--checkpoint 50steps] [--targets xcom_margin,in_flight,flight_gravity]

Writes results/<tag>/controls/<checkpoint>_temporal_readout.csv (tag phase2 by default).
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="50steps")
ap.add_argument("--tag", default="phase2", help="data/<tag>/ and results/<tag>/controls/")
ap.add_argument("--suffix", default="", help="appended to the output file name")
ap.add_argument("--targets", default="xcom_margin,in_flight,flight_gravity")
ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: all complete runs)")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--quick", action="store_true", help="tiny run to test the code (first 300 motions, 3 epochs)")
CLI = ap.parse_args()
torch.set_num_threads(CLI.threads)
EARLY = {"50steps": 10, "1000steps": 200}[CLI.checkpoint]
LAYER = {("xcom_margin", "50steps"): 7, ("xcom_margin", "1000steps"): 7, ("in_flight", "50steps"): 6,
         ("in_flight", "1000steps"): 6, ("flight_gravity", "50steps"): 5, ("flight_gravity", "1000steps"): 5,
         ("balance_margin", "50steps"): 7, ("balance_margin", "1000steps"): 7}
SHOWN_LABELS = ["xcom_margin", "balance_margin", "lowest_foot", "in_flight", "airborne", "skate"]
# Targets derived from flight gravity at the existing floating cut-off (as in fair_readouts.py).
_G = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), _G)
FLOAT_G = float(_G["FLOAT_G_MAX"])
BASE = {"fall_gravity": "flight_gravity", "float_vs_fall": "flight_gravity"}
LAYER.update({("fall_gravity", "50steps"): 5, ("fall_gravity", "1000steps"): 5,
              ("float_vs_fall", "50steps"): 5, ("float_vs_fall", "1000steps"): 5})


def target_matrix(final, target, FR):
    y = final[BASE.get(target, target)][:, FR].astype(float)
    if target == "fall_gravity":
        y = np.where(y >= FLOAT_G, y, np.nan)
    elif target == "float_vs_fall":
        y = np.where(np.isfinite(y), (y < FLOAT_G).astype(float), np.nan)
    return y


class Temporal(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d_in, 128, 5, padding=2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Conv1d(128, 128, 5, padding=4, dilation=2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Conv1d(128, 1, 1))

    def forward(self, x):                      # x [B, T, D] -> [B, T]
        return self.net(x.transpose(1, 2)).squeeze(1)


def fit_predict(Xtr, ytr, mtr, gtr, Xte, seed):
    rng_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, v = next(rng_split.split(Xtr, groups=gtr))
    flat = Xtr[a].reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    ym, ys = ytr[a][mtr[a]].mean(), ytr[a][mtr[a]].std() + 1e-8
    t = lambda z: torch.tensor(z, dtype=torch.float32)
    Xa, Xv, Xt = t((Xtr[a] - mu) / sd), t((Xtr[v] - mu) / sd), t((Xte - mu) / sd)
    ya, yv = t(np.nan_to_num((ytr[a] - ym) / ys)), t(np.nan_to_num((ytr[v] - ym) / ys))
    ma, mv = t(mtr[a].astype(np.float32)), t(mtr[v].astype(np.float32))
    torch.manual_seed(seed)
    net = Temporal(Xa.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = lambda p, y, m: ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)
    best, state, bad = np.inf, None, 0
    for epoch in range(3 if CLI.quick else 200):
        net.train()
        for idx in torch.randperm(len(Xa)).split(32):
            opt.zero_grad(); loss_fn(net(Xa[idx]), ya[idx], ma[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xv), yv, mv).item()
        if val < best - 1e-4:
            best, state, bad = val, {k: w.clone() for k, w in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return net(Xt).numpy() * ys + ym


def within_prompt_r(y, p, g):
    yc, pc = y.copy(), p.copy()
    for pid in np.unique(g):
        m = g == pid
        yc[m] -= y[m].mean(); pc[m] -= p[m].mean()
    return np.corrcoef(yc, pc)[0, 1]


rows = []
dirs = sorted(glob.glob(os.path.join(ROOT, "data", CLI.tag, f"{CLI.checkpoint}_seed*")))
if CLI.seeds:
    dirs = [d for d in dirs if d.split("_seed")[-1] in CLI.seeds.split(",")]
for d in dirs:
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    if len(files) * meta["batch"] < len(meta["all_prompts"]):
        continue                                   # still sampling
    pid = np.array(meta["prompt_id"])
    FR = np.arange(0, meta["n_frames"], meta["frame_stride"])
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    final, x0lab = lab["final"].item(), lab["x0"].item()
    X0F = np.concatenate([np.load(p)[f"x0f_{EARLY}"] for p in files]).astype(np.float32)
    for target in CLI.targets.split(","):
        t0 = time.time()
        kind = "auc" if target in ("in_flight", "float_vs_fall") else "r2"
        y = target_matrix(final, target, FR)                          # [N, T]
        mask = np.isfinite(y)
        shown = [X0F]
        base = BASE.get(target, target)
        for name in SHOWN_LABELS + ([base] if base not in SHOWN_LABELS else []):
            v = x0lab[EARLY][name][:, FR].astype(float)
            shown += [np.nan_to_num(v)[..., None], (~np.isfinite(v)).astype(np.float32)[..., None]]
        shown.append(x0lab[EARLY]["contact"][:, FR].astype(np.float32))
        S = np.concatenate(shown, -1).astype(np.float32)
        L = np.concatenate([np.load(p)[f"act_{EARLY}_{LAYER[(target, CLI.checkpoint)]}"] for p in files]).astype(np.float32)
        feats = {"estimate": S, "layer": L, "estimate + layer": np.concatenate([S, L], -1)}
        keep = mask.any(1)                                            # motions with at least one labelled frame
        if CLI.quick:
            keep &= np.arange(len(keep)) < 300
        idx = np.where(keep)[0]
        folds = list(GroupKFold(n_splits=5).split(idx, groups=pid[idx]))
        for name, X in feats.items():
            pred = np.full(y.shape, np.nan)
            for fi, (tr, te) in enumerate(folds):
                tr, te = idx[tr], idx[te]
                pred[te] = fit_predict(X[tr], y[tr], mask[tr], pid[tr], X[te], seed=fi)
            ev = mask & keep[:, None]                                 # labelled frames of predicted motions
            yy, pp = y[ev], pred[ev]
            gg = np.repeat(pid[:, None], y.shape[1], 1)[ev]
            score = roc_auc_score(yy, pp) if kind == "auc" else r2_score(yy, pp)
            rows.append(dict(checkpoint=CLI.checkpoint, seed=meta["seed"], step=EARLY, target=target, metric=kind,
                             features=name, model="temporal", score=score, within_prompt_r=within_prompt_r(yy, pp, gg)))
        print(f"seed {meta['seed']} {target}: " + ", ".join(f"{r['features']} {r['score']:.3f}" for r in rows[-3:])
              + f" ({time.time() - t0:.0f}s)", flush=True)
os.makedirs(os.path.join(ROOT, "results", CLI.tag, "controls"), exist_ok=True)
out = os.path.join(ROOT, "results", CLI.tag, "controls", f"{CLI.checkpoint}_temporal_readout{CLI.suffix}{'_quick' if CLI.quick else ''}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
