"""Early warning: abort a sample partway through denoising when it looks doomed, and resample.

Simulated offline on the cached phase 2 samples. Every saved sample already records what an
online run would see: its decoded clean-motion guess at each recorded step and its final
motion. Samples of the same prompt are pooled across seeds and redrawn with replacement, so a
"retry" is a fresh draw from the model for that prompt.

Failure (the outcome to avoid): the final motion has any floating, i.e. an unsupported airborne
stretch of at least 0.2 s that falls slower than FLOAT_G_MAX (the label definitions as fixed on
November 16). No new thresholds.

Strategies, cost counted in denoising steps (each step is the same two forward passes):
- generate once: no checks (cost N, failure = base rate);
- finish, then filter: generate fully, check the final motion, retry if it floats;
- early abort: at step k, score the sample; abort and retry if the score is above a threshold;
- early abort + final filter: both checks.
Retries stop after MAX_ATTEMPTS; the last attempt always runs to the end and is delivered.

Early scores:
- guess: the floating rate of the model's own decoded clean-motion guess at step k (training-free);
- guess readout: the same ridge as the layer probe, on everything the guess shows
  (frame-averaged guess features plus the four failure measures computed from the guess);
- layer probe: ridge on frame-averaged activations of one layer (the layer fixed on seed 0 by
  phase 2), predicting the final floating rate, out of fold with prompt-grouped folds;
- random: abort at random at the same rate (an information-free control);
- oracle: the final outcome itself (a perfect detector), to show the ceiling.

Matched comparison: "finish, then filter" and "early abort + final filter" both end with a
final check, so their delivered failure comes only from running out of retries. Both are run
for retry budgets 1 to 20, and compute is compared at equal delivered failure.

    .venv/bin/python scripts/early_warning_sim.py

Writes results/phase2/early_warning/{summary.csv, operating_points.csv, tradeoff.png}.
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase2", "early_warning")
os.makedirs(OUT, exist_ok=True)
MAX_ATTEMPTS, REPS = 10, 200
BUDGETS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
ABORT_RATES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]

G = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), G)   # label constants
_src = open(os.path.join(ROOT, "tools", "build_nb2.py")).read()
exec(_src[_src.index("def failure_rates(lab):"):_src.index("def within_prompt(")], G)   # the notebook's failure_rates
failure_rates = G["failure_rates"]


def oof_ridge(X, y, groups, alpha=10.0):
    # Same as the notebooks' oof_ridge: prompt-grouped 5-fold, out-of-fold predictions.
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        pred[te] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return pred


def simulate(fail, score, thr, k, n_steps, by_prompt, mode, rng, abort_p=None, max_attempts=MAX_ATTEMPTS):
    # Returns (mean steps per delivered motion, failure rate of delivered motions).
    costs, fails = [], []
    for idx in by_prompt:
        for _ in range(REPS):
            cost = 0
            for attempt in range(max_attempts):
                s = idx[rng.integers(len(idx))]
                last = attempt == max_attempts - 1
                if mode in ("early", "early+final") and not last:
                    flagged = rng.random() < abort_p if abort_p is not None else score[s] > thr
                    if flagged:
                        cost += k + 1
                        continue
                cost += n_steps
                if mode in ("final", "early+final") and fail[s] and not last:
                    continue
                break
            costs.append(cost)
            fails.append(fail[s])
    return np.mean(costs), np.mean(fails)


rows, ops, rank_rows = [], [], []
for ck in ["50steps", "1000steps"]:
    runs = sorted(glob.glob(os.path.join(ROOT, "data", "phase2", f"{ck}_seed*")))
    info = json.load(open(os.path.join(ROOT, "results", "phase2", ck, "run_info.json")))
    layer = int(info["fixed_layer"].split()[-1])
    fail, prompt, rate, scores = [], [], [], {}
    for d in runs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        assert meta["prompts"] == json.load(open(os.path.join(runs[0], "meta.json")))["prompts"]
        n_steps, steps = len(range(meta["record_steps"][-1] + 1)), meta["record_steps"]
        z = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
        final, x0 = z["final"].item(), z["x0"].item()
        y = failure_rates(final)["floating rate"]
        fail.append(y > 0)
        pid = np.array(meta["prompt_id"])
        prompt.append(pid)
        files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
        for k in steps[1:3]:   # 20% and 40% of denoising
            fx = failure_rates(x0[k])
            scores.setdefault(("guess", k), []).append(fx["floating rate"])
            shown = [np.concatenate([np.load(p)[f"x0f_{k}"].astype(np.float32).mean(1) for p in files])]
            for v in fx.values():
                shown += [np.nan_to_num(v)[:, None], (~np.isfinite(v)).astype(np.float32)[:, None]]
            shown = np.hstack(shown)
            scores.setdefault(("guess readout", k), []).append(oof_ridge(shown, y, pid))
            A = np.concatenate([np.load(p)[f"act_{k}_{layer}"].astype(np.float32).mean(1) for p in files])
            scores.setdefault((f"layer {layer} probe", k), []).append(oof_ridge(A, y, pid))
            # conditional test at the sample level: does the layer add to the guess readout?
            scores.setdefault(("guess readout + layer", k), []).append(oof_ridge(np.hstack([shown, A]), y, pid))
        rate.append(y)
    fail, prompt, rate = np.concatenate(fail), np.concatenate(prompt), np.concatenate(rate)
    scores = {key: np.concatenate(v) for key, v in scores.items()}
    for k in steps[1:3]:
        scores[("oracle", k)] = fail.astype(float)
    by_prompt = [np.where(prompt == p)[0] for p in np.unique(prompt)]
    crn = lambda: np.random.default_rng(0)   # common random numbers: every strategy sees the same draws
    base_cost, base_fail = simulate(fail, None, None, 0, n_steps, by_prompt, "none", crn())
    fin_cost, fin_fail = simulate(fail, None, None, 0, n_steps, by_prompt, "final", crn())
    for m in BUDGETS:
        cost, fr = simulate(fail, None, None, 0, n_steps, by_prompt, "final", crn(), max_attempts=m)
        rows.append(dict(checkpoint=ck, score="none", check_step=np.nan, abort_rate=0.0, strategy=f"finish+filter, budget {m}",
                         budget=m, rel_compute=cost / n_steps, failure_rate=fr))
    for name, cost, fr in [("generate once", base_cost, base_fail), ("finish, then filter", fin_cost, fin_fail)]:
        rows.append(dict(checkpoint=ck, score="none", check_step=np.nan, abort_rate=0.0, strategy=name,
                         rel_compute=cost / n_steps, failure_rate=fr))
    for (sname, k), sc in scores.items():
        for q in ([] if sname == "oracle" else ABORT_RATES):   # oracle scores are 0/1, so only its matched curve is meaningful
            thr = np.quantile(sc, 1 - q)
            for mode in ["early", "early+final"]:
                cost, fr = simulate(fail, sc, thr, k, n_steps, by_prompt, mode, crn())
                rows.append(dict(checkpoint=ck, score=sname, check_step=k, abort_rate=q, strategy=mode,
                                 rel_compute=cost / n_steps, failure_rate=fr))
                if sname == "guess":
                    cost, fr = simulate(fail, sc, thr, k, n_steps, by_prompt, mode, crn(), abort_p=q)
                    rows.append(dict(checkpoint=ck, score="random", check_step=k, abort_rate=q, strategy=mode,
                                     rel_compute=cost / n_steps, failure_rate=fr))
        if sname in ("guess", f"layer {layer} probe", "oracle", "guess readout", "guess readout + layer"):
            # Matched comparison over retry budgets. The guess uses its training-free rule (any
            # floating); the layer probe aborts the same share of samples.
            thr = {"guess": 0.0, "oracle": 0.5}.get(sname, np.quantile(sc, 1 - (scores[("guess", k)] > 0).mean()))
            for m in BUDGETS:
                cost, fr = simulate(fail, sc, thr, k, n_steps, by_prompt, "early+final", crn(), max_attempts=m)
                rows.append(dict(checkpoint=ck, score=sname, check_step=k, abort_rate=np.nan, strategy=f"early+final, budget {m}",
                                 budget=m, rel_compute=cost / n_steps, failure_rate=fr))
        if sname == "guess":   # the natural training-free rule: abort if the guess floats at all
            flagged = sc > 0
            for mode in ["early", "early+final"]:
                cost, fr = simulate(fail, sc, 0.0, k, n_steps, by_prompt, mode, crn())
                ops.append(dict(checkpoint=ck, rule=f"abort if the guess at step {k} shows any floating", strategy=mode,
                                flagged_share=flagged.mean(), precision=fail[flagged].mean(),
                                recall=flagged[fail].mean(), rel_compute=cost / n_steps, failure_rate=fr,
                                base_failure=base_fail, finish_filter_compute=fin_cost / n_steps,
                                finish_filter_failure=fin_fail))
    for (sname, k), sc in scores.items():   # how well each score ranks samples by their final floating rate
        if sname == "oracle":
            continue
        cen = lambda v: v - (np.bincount(np.unique(prompt, return_inverse=True)[1], v) / np.bincount(np.unique(prompt, return_inverse=True)[1]))[np.unique(prompt, return_inverse=True)[1]]
        rank_rows.append(dict(checkpoint=ck, score=sname, check_step=k, spearman=spearmanr(sc, rate)[0],
                              within_prompt_spearman=spearmanr(cen(sc), cen(rate))[0]))
    print(ck, "done: base failure", round(base_fail, 3), "| finish-then-filter compute", round(fin_cost / n_steps, 3),
          "failure", round(fin_fail, 3), flush=True)

df, op = pd.DataFrame(rows), pd.DataFrame(ops)
pd.DataFrame(rank_rows).to_csv(os.path.join(OUT, "ranking.csv"), index=False)
df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
op.to_csv(os.path.join(OUT, "operating_points.csv"), index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=90)
match = []
for col, ck in enumerate(["50steps", "1000steps"]):
    d = df[df.checkpoint == ck]
    ax = axes[0, col]
    for (sname, k), g in d[d.strategy == "early"].groupby(["score", "check_step"]):
        style = {"guess": "o-", "guess readout": "d-", "random": "x:", "oracle": "k^:"}.get(sname, "s--")
        ax.plot(g.rel_compute, 100 * g.failure_rate, style, label=f"{sname}, step {int(k)}", ms=4)
    for name, mk in [("generate once", "k*"), ("finish, then filter", "r*")]:
        r = d[d.strategy == name].iloc[0]
        ax.plot(r.rel_compute, 100 * r.failure_rate, mk, ms=12, label=name)
    ax.set_xlabel("compute per delivered motion (1 = one full sample)"); ax.set_ylabel("delivered motions that float (%)")
    ax.set_title(f"{ck}: early abort only (no final check)"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    ax = axes[1, col]
    ff = d[d.strategy.str.startswith("finish+filter")].sort_values("budget")
    ax.plot(ff.rel_compute, 100 * ff.failure_rate, "r*-", label="finish, then filter")
    for (sname, k), g in d[d.strategy.str.startswith("early+final, budget")].groupby(["score", "check_step"]):
        g = g.sort_values("budget")
        ax.plot(g.rel_compute, 100 * g.failure_rate, {"guess": "o-", "oracle": "k^:"}.get(sname, "s--"), ms=4,
                label=f"early abort ({sname}, step {int(k)}) + final filter")
        # compute needed to reach each finish-then-filter failure level (interpolated along the budget curve)
        # Match at finish-then-filter budgets 2, 3 and 5; larger budgets sit on the floor set by prompts
        # whose samples (almost) always float, where both curves go flat and matching is meaningless.
        for _, r in ff[ff.budget.isin([2, 3, 5])].iterrows():
            fr_curve, c_curve = g.failure_rate.values[::-1], g.rel_compute.values[::-1]   # failure decreases with budget
            if fr_curve.min() <= r.failure_rate <= fr_curve.max():
                c = np.interp(r.failure_rate, fr_curve, c_curve)
                match.append(dict(checkpoint=ck, score=sname, check_step=int(k), failure_rate=r.failure_rate,
                                  finish_filter_compute=r.rel_compute, early_compute=c,
                                  saving=1 - c / r.rel_compute))
    ax.set_xlabel("compute per delivered motion (1 = one full sample)"); ax.set_ylabel("delivered motions that float (%)")
    ax.set_title(f"{ck}: both with a final check, retry budget 1 to 20"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "tradeoff.png"))
mt = pd.DataFrame(match)
mt.to_csv(os.path.join(OUT, "matched_failure_compute.csv"), index=False)
pd.set_option("display.width", 250)
print(op.round(3).to_string())
print(mt.round(3).to_string())
print("wrote", OUT)
