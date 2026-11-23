"""Is IntPhys plausibility readable from the clean Wan VAE latent? (tests "absent from the VAE input")

Esmati et al. report chance-level plausibility readout from the clean VAE latent and conclude the
signal is built by the diffusion model. Their public code does not include that baseline; their
model features are mean-pooled over all tokens before a linear probe, so the baseline is
reproduced here the same way (16 numbers per video). The question is whether readouts that keep
the time axis recover what pooling removes.

Feature sets, all from the clean latent [16 channels, 25 frames, 32, 32]:
- pooled (their baseline as read): mean over frames and space, 16 numbers;
- time kept: mean over space per frame, 16 x 25;
- time + 4x4 space: average-pooled to a 4 x 4 grid per frame, 16 x 25 x 16;
- the same plus frame-to-frame changes (absolute differences), linear;
- a small nonlinear temporal probe (1D conv over frames on the 4x4-grid features and their
  changes, max- and mean-pooled over time).
Extra sets (--extra-only runs just these, linear probes, and writes *_extra files), to locate
where the 4x4 + changes gain comes from:
- time kept + frame changes: the 16 x 25 frame means plus their absolute frame-to-frame changes;
- frame changes only: the absolute 4x4-grid changes alone, 16 x 24 x 16;
- time + 4x4 space + signed frame changes: signed differences are a linear function of the
  per-frame features, so a linear probe should gain nothing from them; any gain from the
  absolute changes is then due to that fixed nonlinearity.

Splits keep every scenario's four videos together (as Esmati et al.'s code does): grouped 5-fold
cross-validation, repeated with different shuffles. Metrics: per-video accuracy and pairwise
accuracy (share of possible/impossible pairs within a scenario where the possible video scores
higher; ties 0.5), both 50% at chance. Linear probes (ridge classifiers) also get a label-permutation null
(labels shuffled within each scenario, which keeps two possible and two impossible per scenario).

    .venv/bin/python scripts/intphys_vae_probe.py

Writes results/intphys_vae/{probe_results.csv, summary.txt}.
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(ROOT, "data", "intphys_vae", "latents.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "intphys_vae"))
ap.add_argument("--quick", action="store_true", help="tiny run to test the code")
ap.add_argument("--extra-only", action="store_true", help="only the extra feature sets (see docstring)")
ap.add_argument("--null-only", type=int, default=0, metavar="N",
                help="only the label-shuffle null, with N shuffles, for the readouts in --null-features")
ap.add_argument("--null-features", default="time + 4x4 space + frame changes,frame changes only,time + 4x4 space + signed frame changes")
CLI = ap.parse_args()
OUT = CLI.out
os.makedirs(OUT, exist_ok=True)
REPEATS, PERM_REPEATS, CNN_SEEDS = (2, 2, 1) if CLI.quick else (10, 20, 3)
CNN_REPEATS = 2 if CLI.quick else 3
ALPHA_SCALE = np.logspace(-3, 3, 7)   # ridge penalty per feature dimension
torch.set_num_threads(8)

d = np.load(CLI.data)
z = d["z"].astype(np.float32)                        # [N, 16, 25, 32, 32]
label, block, scen = d["label"], d["block"], d["scenario"]
_, groups = np.unique(scen, return_inverse=True)
N, C, T = z.shape[:3]
grid = torch.nn.functional.adaptive_avg_pool3d(torch.from_numpy(z), (T, 4, 4)).numpy()   # [N, 16, 25, 4, 4]
per_frame = grid.transpose(0, 2, 1, 3, 4).reshape(N, T, -1)                                # [N, 25, 256]
change = np.abs(np.diff(per_frame, axis=1))                                                # [N, 24, 256]
FEATS = {
    "pooled (their baseline as read)": z.mean(axis=(2, 3, 4)),
    "time kept": z.mean(axis=(3, 4)).reshape(N, -1),
    "time + 4x4 space": per_frame.reshape(N, -1),
    "time + 4x4 space + frame changes": np.hstack([per_frame.reshape(N, -1), change.reshape(N, -1)]),
}
frame_mean = z.mean(axis=(3, 4)).transpose(0, 2, 1)                                        # [N, 25, 16]
EXTRA = {
    "time kept + frame changes": np.hstack([frame_mean.reshape(N, -1), np.abs(np.diff(frame_mean, axis=1)).reshape(N, -1)]),
    "frame changes only": change.reshape(N, -1),
    "time + 4x4 space + signed frame changes": np.hstack([per_frame.reshape(N, -1), np.diff(per_frame, axis=1).reshape(N, -1)]),
}
if CLI.extra_only:
    FEATS = EXTRA
if CLI.null_only:
    ALL = {**FEATS, **EXTRA}
    FEATS = {k: ALL[k] for k in CLI.null_features.split(",")}
SEQ = np.concatenate([per_frame, np.concatenate([np.zeros_like(change[:, :1]), change], 1)], axis=2)   # [N, 25, 512]


def pairwise(score, y, g):
    hits, n = 0.0, 0
    for s in np.unique(g):
        m = g == s
        pos, neg = score[m][y[m] == 1], score[m][y[m] == 0]
        for a in pos:
            for b in neg:
                hits += 1.0 if a > b else (0.5 if a == b else 0.0)
                n += 1
    return hits / n if n else np.nan


def fit_linear(Xtr, ytr, gtr, Xte):
    # Linear probe: ridge classifier on standardized features (targets -1/+1), solved in the
    # dual (n << d here). The penalty is chosen by grouped inner CV on the training scenarios.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    t = np.where(ytr == 1, 1.0, -1.0)

    def dual_scores(Xa, ta, Xb, alphas):
        K, Kb = Xa @ Xa.T, Xb @ Xa.T
        w, V = np.linalg.eigh(K)
        Vt = V.T @ (ta - ta.mean())
        return [Kb @ (V @ (Vt / (w + a))) + ta.mean() for a in alphas]

    alphas = ALPHA_SCALE * Xtr.shape[1]
    accs = np.zeros(len(alphas))
    for itr, iva in GroupKFold(n_splits=4).split(Xtr, t, gtr):
        for k, sc in enumerate(dual_scores(Xtr[itr], t[itr], Xtr[iva], alphas)):
            accs[k] += ((sc > 0) == (t[iva] > 0)).mean()
    return dual_scores(Xtr, t, Xte, [alphas[int(np.argmax(accs))]])[0]


class TemporalCNN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d_in, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(128, 1)

    def forward(self, x):                      # x [B, T, D]
        h = self.net(x.transpose(1, 2))        # [B, 64, T]
        return self.head(torch.cat([h.max(2).values, h.mean(2)], 1)).squeeze(1)


def fit_cnn(Xtr, ytr, gtr, Xte):
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.zeros(len(Xte))
    for seed in range(CNN_SEEDS):
        rng = np.random.default_rng(seed)
        ug = np.unique(gtr)
        va_g = rng.choice(ug, size=max(1, len(ug) // 5), replace=False)   # early-stopping scenarios
        va = np.isin(gtr, va_g)
        torch.manual_seed(seed)
        net = TemporalCNN(Xtr.shape[2])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        xa, ya = torch.tensor(Xtr[~va], dtype=torch.float32), torch.tensor(ytr[~va], dtype=torch.float32)
        xv, yv = torch.tensor(Xtr[va], dtype=torch.float32), torch.tensor(ytr[va], dtype=torch.float32)
        best, state, bad = np.inf, None, 0
        for epoch in range(300):
            net.train()
            for idx in torch.randperm(len(xa)).split(16):
                opt.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(net(xa[idx]), ya[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = nn.functional.binary_cross_entropy_with_logits(net(xv), yv).item()
            if v < best - 1e-4:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 25:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            out += net(torch.tensor(Xte, dtype=torch.float32)).numpy() / CNN_SEEDS
    return out


def evaluate(name, X, fit, mask, y, repeats, tag):
    rows = []
    Xm, ym, gm = X[mask], y[mask], groups[mask]
    for r in range(repeats):
        perm_g = np.random.default_rng(100 + r).permutation(gm.max() + 1)[gm]   # reshuffles which scenarios share a fold
        score = np.zeros(len(ym))
        for tr, te in GroupKFold(n_splits=5).split(Xm, ym, perm_g):
            score[te] = fit(Xm[tr], ym[tr], gm[tr], Xm[te])
        rows.append(dict(features=name, subset=tag, repeat=r, per_video=((score > 0) == ym).mean(),
                         pairwise=pairwise(score, ym, gm)))
    return rows


def shuffled_within_scenario(y, rng):
    y2 = y.copy()
    for s in np.unique(groups):
        m = np.where(groups == s)[0]
        y2[m] = rng.permutation(y[m])
    return y2


rows, t0 = [], time.time()
if CLI.null_only:   # observed accuracy from the saved runs, null from N fresh shuffles
    PERM_REPEATS = CLI.null_only
    rng = np.random.default_rng(11)
    null = []
    for p in range(PERM_REPEATS):
        yp = shuffled_within_scenario(label, rng)
        for name, X in FEATS.items():
            r = evaluate(name, X, fit_linear, np.ones(N, bool), yp, 1, "all blocks, labels shuffled")[0]
            r["perm"] = p
            null.append(r)
        if (p + 1) % 100 == 0:
            print(f"{p + 1} shuffles ({time.time() - t0:.0f}s)", flush=True)
    nf = pd.DataFrame(null)
    obs_df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ["probe_results.csv", "probe_results_extra.csv"]])
    obs = obs_df[obs_df.subset == "all blocks"].groupby("features")[["per_video", "pairwise"]].mean()
    lines = [f"Label-shuffle null, {PERM_REPEATS} shuffles within scenario (all blocks):"]
    for f in FEATS:
        x = nf[nf.features == f]
        for m in ["per_video", "pairwise"]:
            pv = (1 + (x[m] >= obs.loc[f, m]).sum()) / (1 + PERM_REPEATS)
            lines.append(f"  {f} | {m}: observed {obs.loc[f, m]:.3f}, null mean {x[m].mean():.3f}, "
                         f"95th pct {x[m].quantile(0.95):.3f}, max {x[m].max():.3f}, p = {pv:.4f}")
    nf.to_csv(os.path.join(OUT, f"permutation_null_{PERM_REPEATS}.csv"), index=False)
    open(os.path.join(OUT, f"null_{PERM_REPEATS}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    raise SystemExit
for tag, mask in [("all blocks", np.ones(N, bool))] + [(b, block == b) for b in ["O1", "O2", "O3"]]:
    for name, X in FEATS.items():
        rows += evaluate(name, X, fit_linear, mask, label, REPEATS, tag)
    if not CLI.extra_only:
        rows += evaluate("nonlinear temporal probe", SEQ, fit_cnn, mask, label, CNN_REPEATS, tag)
    print(f"{tag}: done ({time.time() - t0:.0f}s)", flush=True)

# Permutation null for the linear probes on all blocks (labels shuffled within scenario).
null = []
rng = np.random.default_rng(7)
for p in range(PERM_REPEATS):
    yp = shuffled_within_scenario(label, rng)
    for name, X in FEATS.items():
        r = evaluate(name, X, fit_linear, np.ones(N, bool), yp, 1, "all blocks, labels shuffled")[0]
        r["perm"] = p
        null.append(r)
print(f"permutation null done ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
nf = pd.DataFrame(null)
SUF = "_extra" if CLI.extra_only else ""
df.to_csv(os.path.join(OUT, f"probe_results{SUF}.csv"), index=False)
nf.to_csv(os.path.join(OUT, f"permutation_null{SUF}.csv"), index=False)
summ = df.groupby(["subset", "features"])[["per_video", "pairwise"]].agg(["mean", "std"]).round(3)
nulls = nf.groupby("features")[["per_video", "pairwise"]].agg(["mean", lambda s: s.quantile(0.95)]).round(3)
obs = df[df.subset == "all blocks"].groupby("features")[["per_video", "pairwise"]].mean()
pvals = {f: {m: (1 + (nf[nf.features == f][m] >= obs.loc[f, m]).sum()) / (1 + PERM_REPEATS) for m in ["per_video", "pairwise"]}
         for f in FEATS}
text = (f"IntPhys dev: {N} videos, {groups.max() + 1} scenarios, {label.mean():.2f} possible\n\n"
        f"Accuracy (mean, sd over {REPEATS} grouped 5-fold repeats" + ("" if CLI.extra_only else f"; the nonlinear probe over {CNN_REPEATS}") + f"):\n{summ.to_string()}\n\n"
        f"Permutation null, linear probes, all blocks ({PERM_REPEATS} shuffles; mean and 95th percentile):\n{nulls.to_string()}\n\n"
        f"Permutation p-values (all blocks): {pvals}\n")
open(os.path.join(OUT, f"summary{SUF}.txt"), "w").write(text)
print(text)
