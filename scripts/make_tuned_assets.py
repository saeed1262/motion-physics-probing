"""Tables and figures for the tuned, seed-pooled readouts (scripts/fair_readouts.py outputs).

    tables/tab_tuned.tex       main targets: tuned scores and the conditional gain (fallback estimator)
    tables/tab_raw.tex         raw versus engineered estimate input, by readout family (appendix)
    figures/fig_gain.pdf       estimate minus layer, and the gain from adding the layer, three scores
    figures/fig_layers.pdf     the gain for every layer and for all layers together
    figures/fig_tuned_curve.pdf  tuned readouts against training prompts

    .venv/bin/python scripts/make_tuned_assets.py      (also run by make_paper_assets.py)

Rows whose results do not exist yet are skipped, so the script runs on partial results.
"""
import os
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
TAB = os.path.join(ROOT, "paper", "tables")
setup()
SETS = [("phase2", "50steps", "50-step"), ("phase2", "1000steps", "1000-step"), ("phase2", "motiondiffuse", "MotionDiffuse"),
        ("jump_heavy", "50steps", "50-step, jump prompts"), ("jump_heavy", "1000steps", "1000-step, jump prompts"),
        ("jump_heavy", "motiondiffuse", "MotionDiffuse, jump prompts")]
TARGETS = [("xcom_margin", "Dynamic balance"), ("balance_margin", "Static balance"), ("in_flight", "In flight"),
           ("float_vs_fall", "Floats or falls"), ("fall_gravity", "Fall rate (falling only)")]
GAIN = "conditional gain (layer): fallback"


def read(kind, spec=""):
    out = []
    for tag, ck, _ in SETS:
        p = os.path.join(RES, tag, "fair", f"{ck}{spec}_{kind}.csv")
        if os.path.exists(p):
            out.append(pd.read_csv(p).assign(_tag=tag, _ck=ck))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


sc, bt, cv, ch = read("scores"), read("bootstrap"), read("curve"), read("chosen")
sw = read("bootstrap", "_sweep")
assert len(sc), "no fair_readouts results yet"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    fig.savefig(os.path.join(FIG, name.replace(".pdf", "_preview.png")), dpi=170)
    plt.close(fig)
    print("figure", name)


def ci(b):
    if not np.isfinite([b.observed, b.ci_low, b.ci_high]).all():   # too few prompts for this score
        return "--"
    return f"${b.observed:+.3f}$ [${b.ci_low:+.3f}$, ${b.ci_high:+.3f}$]"


def sel(df, tag, ck, t):
    return df[(df.tag == tag) & (df.checkpoint == ck) & (df.target == t)]


def n_fallback(tag, ck, t):
    """Outer folds (of 5) in which the readout of estimate + layer fell back to the estimate alone."""
    c = ch[(ch.target == t) & (ch.features == "estimate + layer") & (ch._tag == tag) & (ch._ck == ck)]
    return int((c.fallback_choice == "estimate").sum())


# ------------------------------------------------------------------ main table
COLS = [("explicit", "Explicit"), ("layer | ridge", "Layer, ridge"), ("estimate | tuned", "Estimate"),
        ("layer | tuned", "Layer"), ("estimate + layer | tuned+fallback", "Both")]
lines = []
for t, tname in TARGETS:
    for tag, ck, cname in SETS:
        d = sel(sc, tag, ck, t).set_index("readout")
        if d.empty or "estimate + layer | tuned+fallback" not in d.index:
            continue
        vals = [d.loc[k, "score"] for k, _ in COLS]
        best = max(vals[2:])
        num = lambda v: f"${v:.3f}$" if v < 0 else f"{v:.3f}"   # a math minus, not a hyphen
        cells = [(f"\\textbf{{{v:.3f}}}" if i >= 2 and abs(v - best) < 5e-4 else num(v)) for i, v in enumerate(vals)]
        b = sel(bt, tag, ck, t)
        if n_fallback(tag, ck, t) == 5:   # the fallback kept the estimate in every fold: a selection outcome, no interval
            g = ["$0$ (all folds)"] * 2
        else:
            g = [ci(b[(b.comparison == GAIN) & (b.measure == m)].iloc[0]) for m in ["score", "within_prompt_frame_r"]]
        lines.append(f"{tname} & {cname} & " + " & ".join(cells) + " & " + " & ".join(g) + " \\\\")
    lines.append("\\addlinespace[2pt]")
with open(os.path.join(TAB, "tab_tuned.tex"), "w") as f:
    f.write("\\begin{tabular}{llccccccc}\n\\toprule\n"
            "& & & & \\multicolumn{3}{c}{Tuned on held-out prompts} & \\multicolumn{2}{c}{Gain from adding the layer [95\\% CI]} \\\\\n"
            "\\cmidrule(lr){5-7} \\cmidrule(lr){8-9}\n"
            "Target & Model & Explicit & Layer, ridge & Estimate & Layer & Both & Pooled score & Within prompt and frame \\\\\n\\midrule\n"
            + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
print("table tab_tuned.tex")

# ------------------------------------------------------------------ the gain in detail (appendix)
lines = []
for t, tname in TARGETS:
    for tag, ck, cname in SETS:
        d = sel(sc, tag, ck, t).set_index("readout")
        if d.empty or "estimate + layer | tuned+fallback" not in d.index:
            continue
        b = sel(bt, tag, ck, t)
        get = lambda comp: b[(b.comparison == comp) & (b.measure == "score")].iloc[0]
        nf = n_fallback(tag, ck, t)
        e, f = d.loc["estimate | tuned", "score"], d.loc["estimate + layer | tuned+fallback", "score"]
        if t in ("in_flight", "float_vs_fall"):   # AUC: share of misranked pairs that adding the layer removes
            scale = f"{100 * (f - e) / (1 - e):+.0f}\\%"
        else:                                      # R2: conditional usable information in nats (Proposition 1)
            scale = f"{0.5 * np.log((1 - e) / (1 - f)):+.3f}"
        scale = scale.replace("-", "$-$")
        fb = "$0$ (all folds)" if nf == 5 else ci(get(GAIN))
        lines.append(f"{tname} & {cname} & {fb} & {ci(get('conditional gain (layer): tuned'))} & {nf} & {scale} \\\\")
    lines.append("\\addlinespace[2pt]")
with open(os.path.join(TAB, "tab_gain_detail.tex"), "w") as f:
    f.write("\\begin{tabular}{llcccc}\n\\toprule\n"
            "Target & Model & Gain with fallback & Gain without fallback & Estimate kept & Gain, rescaled \\\\\n\\midrule\n"
            + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
print("table tab_gain_detail.tex")

# ------------------------------------------------------------------ positive control (appendix)
CTRL = [("phase2", "xcom_margin", "Dynamic balance ($R^2$)"), ("phase2", "in_flight", "In flight (AUC)"),
        ("jump_heavy", "float_vs_fall", "Floats or falls, jump prompts (AUC)")]
lines = []
for tag, t, tname in CTRL:
    p_ = os.path.join(RES, tag, "fair", "50steps_inject_bootstrap.csv")
    if not os.path.exists(p_):
        continue
    b = pd.read_csv(p_)
    b = b[b.target == t]
    for src, lab in [("layer", "the layer (real)"), ("injected 0.005", "injected, 0.005"), ("injected 0.01", "injected, 0.01"),
                     ("injected 0.02", "injected, 0.02")]:
        cells = [ci(b[(b.comparison == f"conditional gain ({src}): fallback") & (b.measure == m)].iloc[0])
                 for m in ["score", "within_prompt_frame_r"]]
        lines.append(f"{tname if src == 'layer' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    lines.append("\\addlinespace[2pt]")
if lines:
    with open(os.path.join(TAB, "tab_control.tex"), "w") as f:
        f.write("\\begin{tabular}{llcc}\n\\toprule\n& Added to the estimate & Pooled score & Within prompt and frame \\\\\n\\midrule\n"
                + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    print("table tab_control.tex")

# ------------------------------------------------------------------ raw versus engineered input
RAW = [("estimate raw | ridge", "raw, ridge"), ("estimate raw | mlp", "raw, MLP"), ("estimate | ridge", "all it shows, ridge"),
       ("estimate | mlp", "all it shows, MLP"), ("layer | ridge", "ridge"), ("layer | mlp", "MLP")]
lines = []
for t, tname in TARGETS:
    for tag, ck, cname in SETS:
        d = sel(sc, tag, ck, t).set_index("readout")
        if d.empty or "estimate raw | ridge" not in d.index:
            continue
        # a fixed-configuration MLP that diverged in a fold (pooled R2 far below zero) is marked, not hidden
        fmt = lambda v: f"${v:.1f}^\\dagger$" if v < -1 else (f"${v:.3f}$" if v < 0 else f"{v:.3f}")
        lines.append(f"{tname} & {cname} & " + " & ".join(fmt(d.loc[k, 'score']) for k, _ in RAW) + " \\\\")
if lines:
    with open(os.path.join(TAB, "tab_raw.tex"), "w") as f:
        f.write("\\begin{tabular}{llcccccc}\n\\toprule\n& & \\multicolumn{4}{c}{Estimate} & \\multicolumn{2}{c}{Layer} \\\\\n"
                "\\cmidrule(lr){3-6} \\cmidrule(lr){7-8}\nTarget & Model & " + " & ".join(n for _, n in RAW) + " \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("table tab_raw.tex")

# ------------------------------------------------------------------ forest plot
COMP = [("fair: estimate tuned - layer tuned", "Estimate minus layer (tuned)"), (GAIN, "Gain from adding the layer to the estimate")]
MEAS = [("score", "pooled $R^2$ / AUC", "o", INK), ("within_prompt_r", "within prompt ($r$)", "s", "#7a7a7a"),
        ("within_prompt_frame_r", "within prompt and frame ($r$)", "D", "#b0b0b0")]
rows = [(t, tname, tag, ck, cname) for t, tname in TARGETS for tag, ck, cname in SETS
        if len(sel(bt, tag, ck, t)[lambda x: x.comparison == GAIN])]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(rows) + 0.9), sharey=True)
ypos = np.arange(len(rows))[::-1]
for ax, (comp, ctitle), letter in zip(axes, COMP, "ab"):
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for (t, tname, tag, ck, cname), y in zip(rows, ypos):
        for j, (meas, mlab, mk, col) in enumerate(MEAS):
            b = sel(bt, tag, ck, t)
            b = b[(b.comparison == comp) & (b.measure == meas)]
            if b.empty:
                continue
            b = b.iloc[0]
            yy = y + 0.22 - 0.22 * j
            ax.plot([b.ci_low, b.ci_high], [yy, yy], color=col, lw=1.1, zorder=2)
            ax.plot(b.observed, yy, marker=mk, ls="", color=col, ms=3.4, mec="white", mew=0.4, zorder=3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference")
    title(ax, ctitle, letter)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
for y in ypos[1::2]:
    for ax in axes:
        ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", lw=0, zorder=0)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([f"{tname}, {cname}" for _, tname, _, _, cname in rows])
axes[0].set_ylim(-0.6, len(rows) - 0.4)
handles = [Line2D([], [], marker=mk, ls="-", color=col, ms=3.4, lw=1.1, label=mlab) for _, mlab, mk, col in MEAS]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.6, -0.05 if len(rows) > 8 else -0.1))
fig.tight_layout(w_pad=1.0)
save(fig, "fig_gain.pdf")

# ------------------------------------------------------------------ gain for every layer
if len(sw):
    LAY = [str(l) for l in range(8)] + ["all"]
    panels = [(t, tname) for t, tname in [("xcom_margin", "Dynamic balance"), ("in_flight", "In flight"),
                                           ("float_vs_fall", "Floats or falls"), ("fall_gravity", "Fall rate (falling only)")]
              if len(sw[sw.target == t])]
    STY = {("phase2", "50steps"): (INK, "o", "-", "50-step"), ("phase2", "1000steps"): (INK, "o", "--", "1000-step"),
           ("jump_heavy", "50steps"): (COL["est mlp"], "D", "-", "50-step, jump prompts"),
           ("jump_heavy", "1000steps"): (COL["est mlp"], "D", "--", "1000-step, jump prompts")}
    fig, axes = plt.subplots(1, len(panels), figsize=(1.8 * len(panels) + 0.4, 2.3), squeeze=False)
    for ax, (t, tname), letter in zip(axes[0], panels, "abcd"):
        ax.axhline(0, color=MUTED, lw=0.8)
        for (tag, ck), (col, mk, ls, lab) in STY.items():
            d = sw[(sw.tag == tag) & (sw.checkpoint == ck) & (sw.target == t) & (sw.measure == "score")]
            if d.empty or (t == "fall_gravity" and tag == "phase2"):   # too few falling frames in the main data
                continue
            obs, lo, hi = [], [], []
            for l in LAY:
                name = f"conditional gain (layer {l}): fallback" if l != "all" else "conditional gain (layers all): fallback"
                r = d[d.comparison == name].iloc[0]
                obs.append(r.observed); lo.append(r.ci_low); hi.append(r.ci_high)
            x = np.arange(len(LAY)) + {("phase2", "50steps"): -0.21, ("phase2", "1000steps"): -0.07,
                                       ("jump_heavy", "50steps"): 0.07, ("jump_heavy", "1000steps"): 0.21}[(tag, ck)]
            ax.errorbar(x, obs, yerr=[np.array(obs) - lo, np.array(hi) - obs], color=col, marker=mk, ls=ls, ms=3,
                        lw=0.9, capsize=1.2, mec="white", mew=0.4, label=lab)
        ax.set_xticks(range(len(LAY))); ax.set_xticklabels(LAY)
        ax.set_xlabel("layer"); ax.set_ylabel("gain")
        title(ax, tname, letter)
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_layers.pdf")

# ------------------------------------------------------------------ learning curves of the tuned readouts
if len(cv):
    CURVE = [("explicit", "explicit label", COL["explicit"], "s", "-"), ("layer | ridge", "layer, ridge", COL["layer lin"], "o", "--"),
             ("layer | tuned", "layer, tuned", COL["layer lin"], "o", "-"), ("estimate | tuned", "estimate, tuned", COL["est mlp"], "D", "-"),
             ("estimate + layer | tuned+fallback", "both, tuned with fallback", COL["both"], "^", "-")]
    WANT = [("phase2", "50steps", "xcom_margin", "Dynamic balance, 50-step"),
            ("jump_heavy", "50steps", "fall_gravity", "Fall rate, jump prompts, 50-step"),
            ("jump_heavy", "1000steps", "fall_gravity", "Fall rate, jump prompts, 1000-step")]
    panels = [p for p in WANT if len(sel(cv, p[0], p[1], p[2]))]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 2.3), squeeze=False)
    for ax, (tag, ck, t, name), letter in zip(axes[0], panels, "abc"):
        d = sel(cv, tag, ck, t)
        full_n = d.n_train_prompts.iloc[-1]   # the last rows are the full training set
        d = d[(d.n_train_prompts < full_n) | (d.index >= d.index[-len(d[d.n_train_prompts == full_n])])]
        for key, lab, col, mk, ls in CURVE:
            g = d[d.readout == key].groupby("n_train_prompts").score
            if not len(g):
                continue
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(m.index, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(m.index, m.values, ls=ls, color=col, lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.5, label=lab)
        ax.set_xscale("log")
        xt = sorted(d.n_train_prompts.unique())
        ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in xt])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("training prompts (per fold)"); ax.set_ylabel("$R^2$")
        title(ax, name, letter)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_tuned_curve.pdf")
