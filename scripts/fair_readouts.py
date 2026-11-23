"""Tuned readouts on pooled seeds: does the fair comparison survive readout choice and more data?

At 20% of denoising, pools every complete run of a checkpoint (same prompts, different sampling
noise) and compares, under 5-fold cross-validation grouped by prompt (a prompt is never in train
and test, in any seed):

    explicit           the label computed from the estimate, linearly calibrated (AUC: raw value)
    estimate raw       the estimate's 263 features only
    estimate           everything the estimate shows: its 263 features, every physics label computed
                       from it (NaN filled, plus a missing flag) and its 4 foot-contact flags
    layer              one layer's activations from the conditional pass (per frame)
    estimate + layer   both

Every feature set is read by the same two tuned families, selected on the same grouped validation
split (15% of each outer fold's training prompts):

    ridge   alpha from logspace(-1, 6, 8) by validation MSE, then refit on all training rows
    mlp     a linear map plus an MLP (so the family contains the linear readout), one of the
            CONFIGS below, AdamW, early stopping on the validation split; the configuration
            with the lowest validation MSE is kept
    tuned   whichever of the two has the lower validation MSE
    tuned+fallback (combined sets only)   the lowest validation MSE among the combined set's own
            candidates and the estimate's candidates, so the combined readout can fall back to
            the estimate alone when the larger input does not help on held-out prompts

Targets: the phase 2 labels, plus two that split flight gravity at the existing floating cut-off
(FLOAT_G_MAX in tools/p2_labels_funcs.py, 2 m/s^2): float_vs_fall (on frames with a fitted
gravity: 1 if the stretch floats, AUC) and fall_gravity (the fitted gravity on falling stretches
only, R2). No label definition is changed.

Layer specs (--layer-spec):
    default   the layer with the best mean linear score at 20% under the current labels
    sweep     every layer 0 to 7 on its own, and all eight concatenated (estimate readouts once)
    cond      needs runs recorded with --record-cond-token: the default layer and all layers,
              each with and without the condition token's output (broadcast to every frame)
    text      the default layer with and without the prompt's CLIP text embedding (`txt`, broadcast to
              every frame), and "layer adds beyond text"; its fallback also chooses among the smaller
              "estimate + ..." sets, so each gain compares a readout with every readout it contains
--step K reads recorded denoising step K instead of 20% (outputs get a _stepK suffix).

Scores: pooled R2 (AUC for classification), the within-prompt correlation (prompt mean removed from
target and prediction), and the within-prompt-and-frame correlation. 95% intervals: prompts
resampled with replacement (B replicates), the same draw for every readout. --refit-seed changes the
validation splits and initializations (test folds stay fixed), to measure training variance.

    .venv/bin/python scripts/fair_readouts.py --checkpoint 50steps [--tag phase2] [--device mps] [--curve]
    .venv/bin/python scripts/fair_readouts.py --checkpoint 50steps --layer-spec sweep --device mps

Writes results/<tag>/fair/<checkpoint>[_<spec>][_refit<r>]_{scores,configs,chosen,bootstrap,curve}.csv
and out-of-fold predictions in results/<tag>/fair/preds/.
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
ap.add_argument("--tag", default="phase2")
ap.add_argument("--targets", default="xcom_margin,balance_margin,in_flight,flight_gravity,float_vs_fall,fall_gravity")
ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: every complete run)")
ap.add_argument("--layer-spec", default="default", choices=["default", "sweep", "cond", "text", "inject"])
ap.add_argument("--inject-levels", default="0.005,0.01,0.02", help="positive control: known gains (R2, squared-error scale)")
ap.add_argument("--step", type=int, default=None, help="recorded denoising step to read (default: 20%% of denoising)")
ap.add_argument("--refit-seed", type=int, default=0)
ap.add_argument("--no-raw", action="store_true", help="skip the raw-feature estimate readout (for refits)")
ap.add_argument("--out-suffix", default="", help="appended to output names, to add targets to an earlier run without overwriting it")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--reps", type=int, default=1000, help="bootstrap replicates")
ap.add_argument("--curve", action="store_true", help="also a learning curve over training prompts (slow)")
ap.add_argument("--curve-targets", default="xcom_margin,flight_gravity")
ap.add_argument("--quick", action="store_true", help="tiny run to test the code")
CLI = ap.parse_args()
torch.set_num_threads(CLI.threads)
DEV = torch.device(CLI.device)
EARLY = {"50steps": 10, "1000steps": 200, "motiondiffuse": 200}[CLI.checkpoint] if CLI.step is None else CLI.step
_G = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), _G)
FLOAT_G = float(_G["FLOAT_G_MAX"])
# Best mean linear layer at 20% under the current labels (results/phase2/<ckpt>/probe_scores_all_runs.csv),
# the same for both checkpoints; the gravity-derived targets use gravity's layer.
LAYER = {"xcom_margin": 7, "balance_margin": 7, "in_flight": 6, "flight_gravity": 5, "float_vs_fall": 5, "fall_gravity": 5}
if CLI.checkpoint == "motiondiffuse":   # chosen by the same rule on its main runs (scripts/pick_layer.py)
    LAYER = json.load(open(os.path.join(ROOT, "results", "phase2", "motiondiffuse_layers.json")))
BASE = {"float_vs_fall": "flight_gravity", "fall_gravity": "flight_gravity"}   # label the target is derived from
AUC_TARGETS = {"in_flight", "float_vs_fall"}
SHOWN_LABELS = ["xcom_margin", "balance_margin", "lowest_foot", "in_flight", "airborne", "skate"]
ALPHAS = np.logspace(-1, 6, 8)
CONFIGS = [dict(width=256, depth=2, drop=0.1, wd=1e-4),      # the fixed MLP's settings
           dict(width=256, depth=2, drop=0.3, wd=1e-2),
           dict(width=64, depth=2, drop=0.1, wd=1e-3),
           dict(width=1024, depth=1, drop=0.3, wd=1e-2),
           dict(width=1024, depth=2, drop=0.3, wd=1e-3)]
if CLI.quick:
    CONFIGS = CONFIGS[:2]
OUT = os.path.join(ROOT, "results", CLI.tag, "fair" + ("_quick" if CLI.quick else ""))
os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
SUFFIX = (("" if CLI.layer_spec == "default" else f"_{CLI.layer_spec}") + (f"_step{CLI.step}" if CLI.step is not None else "")
          + (f"_refit{CLI.refit_seed}" if CLI.refit_seed else "") + CLI.out_suffix)


# ------------------------------------------------------------------ data

def complete_runs():
    dirs = sorted(glob.glob(os.path.join(ROOT, "data", CLI.tag, f"{CLI.checkpoint}_seed*")))
    if CLI.seeds:
        dirs = [d for d in dirs if d.split("_seed")[-1] in CLI.seeds.split(",")]
    out = []
    for d in dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
        if len(files) * meta["batch"] >= len(meta["all_prompts"]) and os.path.exists(os.path.join(d, "labels.npz")):
            out.append((d, meta, files))
    assert out, "no complete runs"
    assert all(m["prompts"] == out[0][1]["prompts"] for _, m, _ in out), "runs use different prompt lists"
    return out


def target_values(final, x0lab, target, FR):
    """Final-motion target and the estimate's own value of it ("explicit"), per recorded frame."""
    base = BASE.get(target, target)
    y, xe = final[base][:, FR].astype(float), x0lab[base][:, FR].astype(float)
    if target == "float_vs_fall":        # 1 = the stretch floats; explicit score: lower fitted g = more floating
        y = np.where(np.isfinite(y), (y < FLOAT_G).astype(float), np.nan)
        xe = -xe
    elif target == "fall_gravity":       # fitted gravity of stretches that do fall
        y = np.where(y >= FLOAT_G, y, np.nan)
    return y, xe


def load(target, runs, layer_ids, cond, text=False):
    """Rows = (motion, recorded frame) with a defined target, pooled over runs."""
    Y, XE, RAW, S, G, F, R, TX = [], [], [], [], [], [], [], []
    L = {l: [] for l in layer_ids}
    C = {l: [] for l in layer_ids} if cond else {}
    base = BASE.get(target, target)
    for d, meta, files in runs:
        FR = np.arange(0, meta["n_frames"], meta["frame_stride"])
        lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
        final, x0lab = lab["final"].item(), lab["x0"].item()[EARLY]
        y, xe = target_values(final, x0lab, target, FR)                       # [N, T]
        x0f = np.concatenate([np.load(p)[f"x0f_{EARLY}"] for p in files]).astype(np.float32)
        cols = [x0f]
        for name in SHOWN_LABELS + ([base] if base not in SHOWN_LABELS else []):
            v = x0lab[name][:, FR].astype(float)
            cols += [np.nan_to_num(v)[..., None], (~np.isfinite(v))[..., None]]
        cols.append(x0lab["contact"][:, FR])
        shown = np.concatenate([c.astype(np.float32) for c in cols], -1)
        pid = np.repeat(np.array(meta["prompt_id"])[:, None], len(FR), 1)
        frame = np.repeat(np.arange(len(FR))[None], len(pid), 0)
        m = np.isfinite(y)
        if CLI.quick:
            m &= pid < 60
        Y.append(y[m]); XE.append(xe[m]); RAW.append(x0f[m]); S.append(shown[m])
        G.append(pid[m]); F.append(frame[m]); R.append(np.full(m.sum(), meta["seed"]))
        if text:   # the prompt's CLIP text embedding [N, 512], the same for every frame of a motion
            tx = np.concatenate([np.load(p)["txt"] for p in files]).astype(np.float32)
            TX.append(np.broadcast_to(tx[:, None], (tx.shape[0], len(FR), tx.shape[1]))[m])
        for l in layer_ids:
            L[l].append(np.concatenate([np.load(p)[f"act_{EARLY}_{l}"] for p in files])[m].astype(np.float32))
            if cond:   # condition-token output [N, 1, d], the same for every frame of a motion
                c = np.concatenate([np.load(p)[f"act_{EARLY}_{l}c"] for p in files])
                C[l].append(np.broadcast_to(c, (c.shape[0], len(FR), c.shape[2]))[m].astype(np.float32))
    cat = np.concatenate
    return (cat(Y), cat(XE), cat(RAW), cat(S), {l: cat(v) for l, v in L.items()}, {l: cat(v) for l, v in C.items()},
            cat(G), cat(F), cat(R), cat(TX) if text else None)


# ------------------------------------------------------------------ readouts

def standardize(Xtr, *rest):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(X - mu) / sd for X in (Xtr, *rest)]


def ridge_path(Xtr, ytr, alphas):
    """Ridge weights for every alpha from one eigendecomposition (X standardized, y centred)."""
    lam, V = np.linalg.eigh(Xtr.T.astype(np.float64) @ Xtr)
    Vty = V.T @ (Xtr.T.astype(np.float64) @ ytr)
    return [V @ (Vty / (lam + a)) for a in alphas]


def fit_ridge(Xtr, ytr, Xte, va_split):
    a, v = va_split
    Xa, Xv = standardize(Xtr[a], Xtr[v])
    ym = ytr[a].mean()
    val = [np.mean((Xv @ w + ym - ytr[v]) ** 2) for w in ridge_path(Xa, ytr[a] - ym, ALPHAS)]
    best = int(np.argmin(val))
    Xf, Xt = standardize(Xtr, Xte)
    w = ridge_path(Xf, ytr - ytr.mean(), [ALPHAS[best]])[0]
    return Xt @ w + ytr.mean(), val[best], ALPHAS[best]


class SkipMLP(nn.Module):
    def __init__(self, d, width, depth, drop):
        super().__init__()
        layers, h = [], d
        for _ in range(depth):
            layers += [nn.Linear(h, width), nn.ReLU(), nn.Dropout(drop)]
            h = width
        self.mlp = nn.Sequential(*layers, nn.Linear(h, 1))
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return (self.lin(x) + self.mlp(x)).squeeze(1)


def fit_mlp(Xtr, ytr, Xte, va_split, cfg, seed):
    a, v = va_split
    Xa, Xv, Xt = standardize(Xtr[a], Xtr[v], Xte)
    ym, ys = ytr[a].mean(), ytr[a].std() + 1e-8
    t = lambda z: torch.tensor(np.ascontiguousarray(z), dtype=torch.float32, device=DEV)
    Xa, Xv, Xt, ya, yv = t(Xa), t(Xv), t(Xt), t((ytr[a] - ym) / ys), t((ytr[v] - ym) / ys)
    torch.manual_seed(seed)
    net = SkipMLP(Xa.shape[1], cfg["width"], cfg["depth"], cfg["drop"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    best, state, bad = np.inf, None, 0
    for epoch in range(3 if CLI.quick else 80):
        net.train()
        for idx in torch.randperm(len(Xa), device=DEV).split(1024):
            opt.zero_grad()
            nn.functional.mse_loss(net(Xa[idx]), ya[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(net(Xv), yv).item()
        if val < best - 1e-4:
            best, state, bad = val, {k: w.detach().clone() for k, w in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return net(Xt).cpu().numpy() * ys + ym, best * ys ** 2


def readouts(feats, xe, y, g, folds, kind, tag, log=True, nested=False):
    """Out-of-fold predictions for explicit and, per feature set, ridge / mlp / tuned (and tuned+fallback
    for sets named "estimate + ..."), plus per-config predictions and the chosen models.

    feats: ordered dict name -> array or zero-argument function returning the array (built once per set).
    "estimate" must come before any "estimate + ..." set. nested: the fallback also chooses among the candidates of
    every earlier "estimate + ..." set whose parts are a subset of this one's (so "A adds beyond B" compares like with like)."""
    n = len(y)
    preds = {"explicit": np.full(n, np.nan)}
    chosen = []
    va_splits = [next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fi + 100 * CLI.refit_seed)
                      .split(tr, groups=g[tr])) for fi, (tr, te) in enumerate(folds)]
    for fi, (tr, te) in enumerate(folds):
        if kind == "auc":
            preds["explicit"][te] = xe[te]
        else:
            A = np.column_stack([xe[tr], np.ones(len(tr))])
            coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            preds["explicit"][te] = xe[te] * coef[0] + coef[1]
    est_cands = {}                                   # fold -> [(val, test pred)] of the estimate's candidates
    set_cands = {}                                   # (set name, fold) -> candidates, for the nested fallback
    for name, X in feats.items():
        X = X() if callable(X) else X
        for fam in ["ridge", "mlp", "tuned"] + (["tuned+fallback"] if name.startswith("estimate +") else []):
            preds[f"{name} | {fam}"] = np.full(n, np.nan)
        for ci in range(len(CONFIGS)):
            preds[f"{name} | mlp config {ci}"] = np.full(n, np.nan)
        t0 = time.time()
        for fi, (tr, te) in enumerate(folds):
            va_split = va_splits[fi]
            pr, vr, alpha = fit_ridge(X[tr], y[tr], X[te], va_split)
            preds[f"{name} | ridge"][te] = pr
            runs_ = []
            for ci, cfg in enumerate(CONFIGS):
                pm, vm = fit_mlp(X[tr], y[tr], X[te], va_split, cfg, seed=fi + 100 * CLI.refit_seed)
                preds[f"{name} | mlp config {ci}"][te] = pm
                runs_.append((vm, ci, pm))
            vm, ci, pm = min(runs_, key=lambda r: r[0])
            preds[f"{name} | mlp"][te] = pm
            preds[f"{name} | tuned"][te] = pm if vm < vr else pr
            cands = [(vr, pr)] + [(v_, p_) for v_, _, p_ in runs_]
            if name == "estimate":
                est_cands[fi] = cands
            fb = ""
            if name.startswith("estimate +"):
                pool = [(v_, p_, "combined") for v_, p_ in cands] + [(v_, p_, "estimate") for v_, p_ in est_cands[fi]]
                if nested:
                    parts = set(name.split(" + "))
                    pool += [(v_, p_, s) for (s, f), cs in set_cands.items() if f == fi and set(s.split(" + ")) < parts
                             for v_, p_ in cs]
                vb, pb, fb = min(pool, key=lambda c: c[0])
                set_cands[(name, fi)] = cands
                preds[f"{name} | tuned+fallback"][te] = pb
            chosen.append(dict(target=tag, features=name, fold=fi, alpha=alpha, ridge_val_mse=vr, mlp_config=ci,
                               mlp_val_mse=vm, tuned="mlp" if vm < vr else "ridge", fallback_choice=fb))
        if log:
            print(f"   {name}: done ({time.time() - t0:.0f}s)", flush=True)
        del X
    return preds, chosen


# ------------------------------------------------------------------ scores

def wcorr(a, b, w):
    w = w / w.sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())


def centre(v, cell, w):
    tot = np.bincount(cell, w * v)
    cnt = np.bincount(cell, w)
    return v - (tot / np.maximum(cnt, 1e-12))[cell]


def scores(y, p, g, cell, kind, w=None):
    w = np.ones(len(y)) if w is None else w
    m = w > 0
    y, p, g, cell, w = y[m], p[m], g[m], cell[m], w[m]
    main = roc_auc_score(y, p, sample_weight=w) if kind == "auc" else r2_score(y, p, sample_weight=w)
    return (main, wcorr(centre(y, g, w), centre(p, g, w), w), wcorr(centre(y, cell, w), centre(p, cell, w), w))


MEASURES = ["score", "within_prompt_r", "within_prompt_frame_r"]


def comparisons(feat_names):
    out = []
    if CLI.layer_spec == "default":
        out += [("standard: layer ridge - explicit", "layer | ridge", "explicit"),
                ("fair: estimate tuned - layer tuned", "estimate | tuned", "layer | tuned"),
                ("raw: estimate raw tuned - layer tuned", "estimate raw | tuned", "layer | tuned"),
                ("labels: estimate tuned - estimate raw tuned", "estimate | tuned", "estimate raw | tuned")]
        if CLI.no_raw:
            out = [c for c in out if "raw" not in c[1] + c[2]]
    for nm in feat_names:
        if nm.startswith("estimate +"):
            lay = nm[len("estimate + "):]
            out += [(f"conditional gain ({lay}): tuned", f"{nm} | tuned", "estimate | tuned"),
                    (f"conditional gain ({lay}): fallback", f"{nm} | tuned+fallback", "estimate | tuned"),
                    (f"conditional gain ({lay}): ridge only", f"{nm} | ridge", "estimate | ridge")]
    if CLI.layer_spec == "text":
        out.append(("layer adds beyond text: fallback", "estimate + text + layer | tuned+fallback", "estimate + text | tuned+fallback"))
    if CLI.layer_spec == "cond":
        for lay in ["layer", "layers all"]:
            out.append((f"condition token adds ({lay}): fallback", f"estimate + {lay} + cond | tuned+fallback",
                        f"estimate + {lay} | tuned+fallback"))
    return out


def bootstrap(y, preds, g, cell, kind, keys, reps, rng):
    prompts = np.unique(g)
    pos = np.searchsorted(prompts, g)
    out = []
    for _ in range(reps):
        cnt = np.bincount(rng.integers(len(prompts), size=len(prompts)), minlength=len(prompts))
        w = cnt[pos].astype(float)
        out.append({k: scores(y, preds[k], g, cell, kind, w) for k in keys})
    return out


# ------------------------------------------------------------------ main

runs = complete_runs()
print(f"{CLI.checkpoint} ({CLI.tag}): pooling seeds {[m['seed'] for _, m, _ in runs]}, step {EARLY}, device {DEV}, "
      f"layer spec {CLI.layer_spec}, refit seed {CLI.refit_seed}")
rows, cfg_rows, boot_rows, curve_rows, chosen_all = [], [], [], [], []
rng = np.random.default_rng(0)
for target in CLI.targets.split(","):
    T0 = time.time()
    kind = "auc" if target in AUC_TARGETS else "r2"
    layer_ids = list(range(8)) if CLI.layer_spec == "sweep" else (list(range(8)) if CLI.layer_spec == "cond" else [LAYER[target]])
    y, xe, RAW, S, L, C, g, fr, rs, TX = load(target, runs, layer_ids, CLI.layer_spec == "cond", CLI.layer_spec == "text")
    xe = np.where(np.isfinite(xe), xe, np.nanmedian(xe))
    cell = np.unique(g * 1000 + fr, return_inverse=True)[1]
    folds = list(GroupKFold(n_splits=5).split(y, groups=g))
    if CLI.layer_spec == "default":
        Ld = L[LAYER[target]]
        feats = {"estimate raw": RAW, "estimate": S, "layer": Ld, "estimate + layer": lambda: np.hstack([S, Ld])}
        if CLI.no_raw:
            del feats["estimate raw"]
    elif CLI.layer_spec == "sweep":
        feats = {"estimate": S}          # standalone per-layer scores come from the phase 2 linear probes
        for l in range(8):
            feats[f"estimate + layer {l}"] = (lambda l=l: np.hstack([S, L[l]]))
        feats["layers all"] = lambda: np.hstack([L[l] for l in range(8)])
        feats["estimate + layers all"] = lambda: np.hstack([S] + [L[l] for l in range(8)])
    elif CLI.layer_spec == "text":
        Ld = L[LAYER[target]]
        feats = {"estimate": S, "estimate + text": lambda: np.hstack([S, TX]), "estimate + layer": lambda: np.hstack([S, Ld]),
                 "estimate + text + layer": lambda: np.hstack([S, TX, Ld])}
    elif CLI.layer_spec == "inject":
        # Positive control. The estimate readout's out-of-fold residual r (from the default run) plus noise sized so
        # that a perfect readout of the signal s would gain q in R2 on the squared-error scale,
        # q = Var(r)^2 / ((Var(r) + sigma^2) Var(y)), added to the layer along one random direction with, after
        # standardization, the variance of a typical direction of the activations. The pipeline is then run unchanged.
        Ld = L[LAYER[target]]
        ref = np.load(os.path.join(OUT, "preds", f"{CLI.checkpoint}_{target}.npz"))
        assert len(ref["y"]) == len(y) and np.allclose(ref["y"], y), "rows differ from the default run"
        r = y - ref["estimate | tuned"]
        var_r, var_y = r.var(), y.var()
        w = np.random.default_rng(7).standard_normal(Ld.shape[1]); w /= np.linalg.norm(w)
        wsd = (w * Ld.std(0)).astype(np.float32)

        def injected(q):
            sig2 = max(var_r ** 2 / (q * var_y) - var_r, 0.0)
            s_ = r + np.sqrt(sig2) * np.random.default_rng(int(q * 1e6)).standard_normal(len(r))
            s_ = ((s_ - s_.mean()) / s_.std()).astype(np.float32)
            return np.hstack([S, Ld + s_[:, None] * wsd[None]])
        feats = {"estimate": S, "estimate + layer": lambda: np.hstack([S, Ld])}
        for q in [float(v) for v in CLI.inject_levels.split(",")]:
            feats[f"estimate + injected {q:g}"] = (lambda q=q: injected(q))
    else:
        Ld, Cd = L[LAYER[target]], C[LAYER[target]]
        feats = {"estimate": S, "estimate + layer": lambda: np.hstack([S, Ld]),
                 "estimate + layer + cond": lambda: np.hstack([S, Ld, Cd]),
                 "estimate + layers all": lambda: np.hstack([S] + [L[l] for l in range(8)]),
                 "estimate + layers all + cond": lambda: np.hstack([S] + [L[l] for l in range(8)] + [C[l] for l in range(8)])}
    print(f"{target}: {len(y)} frames, {len(np.unique(g))} prompts, layers {layer_ids}", flush=True)
    preds, chosen = readouts(feats, xe, y, g, folds, kind, target, nested=CLI.layer_spec == "text")
    chosen_all += chosen
    for key, p in preds.items():
        sc = dict(zip(MEASURES, scores(y, p, g, cell, kind)))
        row = dict(checkpoint=CLI.checkpoint, tag=CLI.tag, target=target, metric=kind, readout=key, layer_spec=CLI.layer_spec,
                   refit_seed=CLI.refit_seed, seeds=len(runs), frames=len(y), prompts=len(np.unique(g)), **sc)
        (cfg_rows if "config" in key else rows).append(row)
    np.savez_compressed(os.path.join(OUT, "preds", f"{CLI.checkpoint}{SUFFIX}_{target}.npz"), y=y, groups=g, frame=fr, seed=rs,
                        **{k: v for k, v in preds.items() if "config" not in k})
    comps = comparisons(list(feats))
    keys = sorted({k for _, a, b in comps for k in (a, b)})
    obs = {k: scores(y, preds[k], g, cell, kind) for k in keys}
    boot = bootstrap(y, preds, g, cell, kind, keys, 50 if CLI.quick else CLI.reps, rng)
    for name, a, b in comps:
        for mi, meas in enumerate(MEASURES):
            d = np.array([bt[a][mi] - bt[b][mi] for bt in boot])
            boot_rows.append(dict(checkpoint=CLI.checkpoint, tag=CLI.tag, target=target, layer_spec=CLI.layer_spec,
                                  refit_seed=CLI.refit_seed, comparison=name, measure=meas, observed=obs[a][mi] - obs[b][mi],
                                  ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), share_above_zero=(d > 0).mean()))
    print(f"{target}: " + ", ".join(f"{k} {obs[k][0]:.3f}" for k in keys if "|" in k and "config" not in k) + f" ({time.time() - T0:.0f}s)", flush=True)

    if CLI.curve and CLI.layer_spec == "default" and target in CLI.curve_targets.split(","):
        CK = ["explicit", "layer | ridge", "estimate | tuned", "layer | tuned", "estimate + layer | tuned", "estimate + layer | tuned+fallback"]
        for n_prompts in ([12, 25] if CLI.quick else [25, 50, 100]):
            for rep in range(2):
                sub_folds = []
                for tr, te in folds:
                    pids = np.unique(g[tr])
                    keep = rng.choice(pids, size=min(n_prompts, len(pids)), replace=False)
                    sub_folds.append((tr[np.isin(g[tr], keep)], te))
                p_sub, _ = readouts({k: v for k, v in feats.items() if k != "estimate raw"}, xe, y, g, sub_folds, kind, target, log=False)
                for key in CK:
                    curve_rows.append(dict(checkpoint=CLI.checkpoint, tag=CLI.tag, target=target, n_train_prompts=n_prompts,
                                           rep=rep, readout=key, **dict(zip(MEASURES, scores(y, p_sub[key], g, cell, kind)))))
        for key in CK:
            curve_rows.append(dict(checkpoint=CLI.checkpoint, tag=CLI.tag, target=target,
                                   n_train_prompts=int(np.mean([len(np.unique(g[tr])) for tr, _ in folds])), rep=0,
                                   readout=key, **dict(zip(MEASURES, scores(y, preds[key], g, cell, kind)))))
        print(f"{target}: learning curve done ({time.time() - T0:.0f}s)", flush=True)

    # write after every target, so a long run leaves usable partial results
    pre = os.path.join(OUT, CLI.checkpoint + SUFFIX)
    pd.DataFrame(rows).to_csv(pre + "_scores.csv", index=False)
    pd.DataFrame(cfg_rows).to_csv(pre + "_configs.csv", index=False)
    pd.DataFrame(chosen_all).to_csv(pre + "_chosen.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(pre + "_bootstrap.csv", index=False)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(pre + "_curve.csv", index=False)
    del L, C, S, RAW, TX

pd.set_option("display.width", 220)
print(pd.DataFrame(rows).pivot_table(index="readout", columns="target", values="score").round(3).to_string())
print("wrote", OUT)
