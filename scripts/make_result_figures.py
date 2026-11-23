"""Result figures 3 to 6, drawn from the result CSVs with the shared paper style (paper_style.py).

    fig_steps.pdf     probe scores across denoising (50-step model), seed spread as bands
    fig_fair.pdf      the fair comparison as a dot plot (no truncated bars), both checkpoints
    fig_learning.pdf  how the layer-versus-explicit comparison depends on training prompts
    fig_early.pdf     delivered floating against compute for the early-abort strategies

    .venv/bin/python scripts/make_result_figures.py
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from paper_style import COL, INK, MUTED, setup, title

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
setup()
CK = {"50steps": "50-step", "1000steps": "1000-step"}
EARLY = {"50steps": 10, "1000steps": 200}
ctrl = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "phase2", "controls", "*_nonlinear_controls.csv"))])
ctrl["key"] = ctrl.features.str.replace(r"layer \d", "layer", regex=True) + " | " + ctrl.model


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    fig.savefig(os.path.join(FIG, name.replace(".pdf", "_preview.png")), dpi=170)
    plt.close(fig)
    print("figure", name)


# ------------------------------------------------------------------ Figure 3: scores across denoising
r = pd.read_csv(os.path.join(RES, "phase2", "50steps", "probe_scores_all_runs.csv"))
steps = sorted(r.step.unique())
x = 100 * np.array(steps) / steps[-1]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
for ax, (t, name, metric, letter) in zip(axes, [("xcom_margin", "Dynamic balance", "$R^2$", "a"), ("in_flight", "In flight", "AUC", "b")]):
    d = r[r.target == t]
    best = d[d.source.str.startswith("layer")].groupby(["seed", "step"]).score.max().reset_index()
    series = [("best layer, linear probe", best, COL["layer lin"], "o", "-", 1.6),
              ("explicit label from the estimate", d[d.source == "explicit"], COL["explicit"], "s", "-", 1.3),
              ("estimate, linear probe", d[d.source == "x0 probe"], COL["est lin"], "^", "--", 1.1),
              ("noisy input, linear probe", d[d.source == "x_t probe"], COL["noisy"], "v", ":", 1.1)]
    for lab, dd, col, mk, ls, lw in series:
        g = dd.groupby("step").score
        m, lo, hi = g.mean().reindex(steps), g.min().reindex(steps), g.max().reindex(steps)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, m, ls=ls, color=col, lw=lw, marker=mk, ms=3.2, mec="white", mew=0.5, label=lab)
    cc = ctrl[(ctrl.checkpoint == "50steps") & (ctrl.target == t) & (ctrl.key == "everything the guess shows | mlp")]
    g = cc.groupby("step").score
    xs = 100 * np.array(g.mean().index) / steps[-1]
    ax.errorbar(xs, g.mean().values, yerr=g.std().values, fmt="D", color=COL["est mlp"], ms=4.5, mec="white", mew=0.6,
                capsize=1.5, lw=0.8, label="estimate, MLP (all it shows)", zorder=5)
    ax.axvline(20, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.text(21, (0.03 if metric != "AUC" else 0.475), "20% denoised", color=MUTED, fontsize=6)
    ax.set_xlim(-3, 103); ax.set_ylim((0.0, 1.03) if metric != "AUC" else (0.45, 1.015))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("denoising done"); ax.set_ylabel(metric)
    title(ax, f"{name}, 50-step model", letter)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_steps.pdf")

# ------------------------------------------------------------------ Figure 4: fair comparison, dot plot
ROWS = [("explicit | linear", "explicit label", COL["explicit"]), ("guess features | linear", "estimate, linear", COL["est lin"]),
        ("everything the guess shows | mlp", "estimate, MLP", COL["est mlp"]), ("layer | linear", "layer, linear", COL["layer lin"]),
        ("layer | mlp", "layer, MLP", COL["layer mlp"]), ("everything the guess shows + layer | mlp", "estimate + layer, MLP", COL["both"])]
FT = [("xcom_margin", "Dynamic balance", "$R^2$"), ("balance_margin", "Static balance", "$R^2$"),
      ("in_flight", "In flight", "AUC"), ("flight_gravity", "Flight gravity", "$R^2$")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.45), sharey=True)
ypos = np.arange(len(ROWS))[::-1]
for ax, (t, name, metric), letter in zip(axes, FT, "abcd"):
    ax.axhspan(ypos[2] - 0.42, ypos[2] + 0.42, color=COL["est mlp"], alpha=0.07, lw=0, zorder=0)
    for (key, lab, col), y in zip(ROWS, ypos):
        for ck, off, filled in [("50steps", 0.14, True), ("1000steps", -0.14, False)]:
            d = ctrl[(ctrl.checkpoint == ck) & (ctrl.target == t) & (ctrl.step == EARLY[ck]) & (ctrl.key == key)].score
            ax.errorbar(d.mean(), y + off, xerr=d.std(), fmt="o", ms=4.2, color=col, mfc=col if filled else "white",
                        mec=col, mew=1.1, capsize=1.6, lw=0.9, zorder=3)
    ref = ctrl[(ctrl.checkpoint == "50steps") & (ctrl.target == t) & (ctrl.step == EARLY["50steps"]) &
               (ctrl.key == "everything the guess shows | mlp")].score.mean()
    ax.axvline(ref, color=COL["est mlp"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(metric)
    ax.grid(axis="y", visible=False)
    title(ax, name, letter)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([lab for _, lab, _ in ROWS])
for tick, (_, _, col) in zip(axes[0].get_yticklabels(), ROWS):
    tick.set_color(col if col != COL["est lin"] else "#B8860B")
axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
handles = [Line2D([], [], marker="o", ls="", color=INK, mfc=INK, ms=4.2, label="50-step model"),
           Line2D([], [], marker="o", ls="", color=INK, mfc="white", mew=1.1, ms=4.2, label="1000-step model"),
           Line2D([], [], color=COL["est mlp"], lw=0.7, ls=(0, (2, 2)), label="estimate, MLP (50-step)")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.08))
fig.tight_layout(w_pad=0.8)
save(fig, "fig_fair.pdf")

# ------------------------------------------------------------------ Figure 5: learning curve
lc = pd.read_csv(os.path.join(RES, "phase2", "controls", "50steps_training_size.csv"))
lc = lc[(lc.step == 10) & lc.design.str.startswith("5-fold")].copy()
lc["n"] = lc.n_train_prompts.where(lc.n_train_prompts < 150, 200)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.axvspan(48, 56, color=MUTED, alpha=0.10, lw=0)
ax.text(52, 0.60, "pilot\n(52)", ha="center", va="center", color=MUTED, fontsize=6,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
curves = {}
for src, col, lab, mk in [("explicit", COL["explicit"], "explicit label from the estimate (1 feature)", "s"),
                          ("layer 7 (linear)", COL["layer lin"], "layer 7, linear probe (512 features)", "o")]:
    g = lc[lc.source == src].groupby("n").r2
    m, sd = g.mean(), g.std().fillna(0)
    curves[src] = m
    ax.fill_between(m.index, m - sd, m + sd, color=col, alpha=0.18, lw=0)
    ax.plot(m.index, m.values, "-", color=col, lw=1.6, marker=mk, ms=3.5, mec="white", mew=0.5, label=lab)
# Mark where the curves cross: between the two measured sizes that bracket it (no interpolated value).
diff = (curves["layer 7 (linear)"] - curves["explicit"]).sort_index()
k = np.where(np.diff(np.sign(diff.values)) > 0)[0]
if len(k):
    a, b = diff.index[k[0]], diff.index[k[0] + 1]
    ax.axvspan(a, b, ymin=0.83, ymax=0.86, color=INK, alpha=0.5, lw=0)
    ax.text(np.sqrt(a * b), 0.85, f"layer probe overtakes\nbetween {a} and {b} prompts", ha="center", va="bottom", fontsize=6, color=INK)
ax.set_xscale("log")
ax.set_xticks([13, 26, 52, 100, 200]); ax.set_xticklabels(["13", "26", "52", "100", "200"])
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_ylim(0.49, 0.90)
ax.set_xlabel("training prompts (per fold)"); ax.set_ylabel("$R^2$, dynamic balance")
ax.legend(loc="lower right", handlelength=2.0)
title(ax, "Dynamic balance, 20% denoised, 50-step model")
fig.tight_layout()
save(fig, "fig_learning.pdf")

# ------------------------------------------------------------------ Figure 6: early warning
ew = pd.read_csv(os.path.join(RES, "phase2", "early_warning", "summary.csv"))
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
for ax, ck, letter in zip(axes, CK, "ab"):
    d = ew[ew.checkpoint == ck]
    ff = d[d.strategy.str.startswith("finish+filter")].sort_values("budget")
    probe = next(s for s in d.score.dropna().unique() if "probe" in str(s))
    oracle = d[(d.score == "oracle") & (d.check_step == EARLY[ck]) & d.strategy.str.startswith("early+final, budget")].sort_values("budget")
    ax.fill_between(np.r_[ff.rel_compute.values, oracle.rel_compute.values[::-1]],
                    np.r_[100 * ff.failure_rate.values, 100 * oracle.failure_rate.values[::-1]], color=COL["oracle"], alpha=0.05, lw=0)
    ax.plot(ff.rel_compute, 100 * ff.failure_rate, "-", color=COL["filter"], lw=1.6, marker="*", ms=5.5, mec="white", mew=0.4,
            label="finish, then filter", zorder=4)
    for _, row in ff[ff.budget.isin([1, 2, 3, 5, 10, 20])].iterrows():
        txt = "generate once" if row.budget == 1 else f"{int(row.budget)} tries"
        ax.annotate(txt, (row.rel_compute, 100 * row.failure_rate), xytext=(4, 3), textcoords="offset points",
                    fontsize=5.5, color=COL["filter"])
    for sc, col, mk, ls, lab in [("guess", COL["est mlp"], "o", "-", "early abort: estimate shows floating"),
                                 (probe, COL["layer lin"], "s", "--", "early abort: layer probe"),
                                 ("oracle", COL["oracle"], "^", ":", "early abort: perfect detector")]:
        g = d[(d.score == sc) & (d.check_step == EARLY[ck]) & d.strategy.str.startswith("early+final, budget")].sort_values("budget")
        ax.plot(g.rel_compute, 100 * g.failure_rate, ls=ls, color=col, lw=1.3, marker=mk, ms=3, mec="white", mew=0.4, label=lab, zorder=3)
    ax.set_xlim(0.97, ff.rel_compute.max() + 0.12); ax.set_ylim(0, None)
    ax.set_xlabel("compute per delivered motion (1 = one sample)"); ax.set_ylabel("delivered motions that float (%)")
    title(ax, f"{CK[ck]} model, check at 20% of denoising", letter)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(w_pad=2.0)
save(fig, "fig_early.pdf")
