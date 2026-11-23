"""Build every figure and table in the paper from the result CSVs (no numbers typed by hand).

    .venv/bin/python scripts/make_paper_assets.py

Writes paper/tables/*.tex, then runs make_result_figures.py for paper/figures/fig_{steps,fair,learning,early}.pdf.
"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
CK = {"50steps": "50-step", "1000steps": "1000-step"}
EARLY = {"50steps": 10, "1000steps": 200}          # 20% of denoising
LATE = {"50steps": 20, "1000steps": 400}           # 40% of denoising
TARGETS = [("xcom_margin", "Dynamic balance", "$R^2$"), ("balance_margin", "Static balance", "$R^2$"),
           ("lowest_foot", "Lowest foot", "$R^2$"), ("in_flight", "In flight", "AUC"),
           ("flight_gravity", "Flight gravity", "$R^2$")]
# Okabe-Ito colours
C = {"explicit": "#E69F00", "guess lin": "#F0E442", "guess mlp": "#D55E00", "layer lin": "#0072B2",
     "layer mlp": "#56B4E9", "both mlp": "#009E73", "xt": "#999999", "oracle": "#000000"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def ms(s, nd=3):
    s = pd.Series(s).dropna()
    return f"{s.mean():.{nd}f}" + (f" $\\pm$ {s.std():.{nd}f}" if len(s) > 1 else "")


def write(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)
    print("table", name)


# ------------------------------------------------------------------ physics table
def parse_physics(path):
    rows, seed = [], None
    for line in open(path):
        m = re.match(r"seed (\d+)", line.strip())
        if m:
            seed = int(m.group(1)); continue
        parts = line.split()
        if seed is None or len(parts) < 8 or not parts[-1].replace(".", "").replace("-", "").isdigit():
            continue
        name = " ".join(parts[:-7]); v = [float(x) for x in parts[-7:]]
        rows.append(dict(seed=seed, group=name, motions=v[0], flight=v[1], float5=v[2], xcom_out=v[3],
                         airborne=v[4], floating=v[5], jump_g=v[6]))
    return pd.DataFrame(rows)


phys = {ck: parse_physics(os.path.join(RES, "phase2", ck, "physics_table.txt")) for ck in CK}
def rng(ck, group, col, nd=1):
    v = phys[ck][phys[ck].group == group][col]
    return f"{v.min():.{nd}f}--{v.max():.{nd}f}"
lines = []
for label, col, nd in [("Floating, \\% of frames", "floating", 1), ("Jump gravity, median (m/s$^2$)", "jump_g", 2),
                       ("Raised contact ($>$5\\,cm), \\% of contact frames", "float5", 1)]:
    for group, gl in [("curated: flight", "jump prompts"), ("test captions", "test captions")]:
        lines.append(f"{label}, {gl} & {rng('50steps', group, col, nd)} & {rng('1000steps', group, col, nd)} \\\\")
write("tab_physics.tex", "\\begin{tabular}{lcc}\n\\toprule\nMeasure & 50-step (3 seeds) & 1000-step (3 seeds) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ naive comparison table
lines = []
for ck in CK:
    r = pd.read_csv(os.path.join(RES, "phase2", ck, "probe_scores_all_runs.csv"))
    k = EARLY[ck]
    for i, (t, name, metric) in enumerate(TARGETS):
        d = r[(r.target == t) & (r.step == k)]
        best = d[d.source.str.startswith("layer")].groupby("seed").score.max()
        cells = [ms(best), ms(d[d.source == "explicit"].score), ms(d[d.source == "x0 probe"].score), ms(d[d.source == "x_t probe"].score)]
        first = f"\\multirow{{5}}{{*}}{{{CK[ck]}}}" if i == 0 else ""
        lines.append(f"{first} & {name} ({metric}) & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule" if ck == "50steps" else "")
write("tab_naive.tex", "\\begin{tabular}{llcccc}\n\\toprule\nModel & Target & Best layer & Explicit & Estimate, linear & Noisy input \\\\\n\\midrule\n"
      + "\n".join(l for l in lines if l) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ fair comparison (controls)
ctrl = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "phase2", "controls", "*_nonlinear_controls.csv"))])
ctrl["key"] = ctrl.features.str.replace(r"layer \d", "layer", regex=True) + " | " + ctrl.model
KEYS = [("explicit | linear", "Explicit"), ("guess features | linear", "Estimate, linear"),
        ("everything the guess shows | mlp", "Estimate, MLP"), ("layer | linear", "Layer, linear"),
        ("layer | mlp", "Layer, MLP"), ("everything the guess shows + layer | mlp", "Estimate + layer, MLP")]
FT = [t for t in TARGETS if t[0] != "lowest_foot"]
for metric, fname, cap in [("score", "tab_fair.tex", None), ("within_prompt_r", "tab_fair_within.tex", None)]:
    lines = []
    for ck in CK:
        for i, (t, name, m) in enumerate(FT):
            d = ctrl[(ctrl.checkpoint == ck) & (ctrl.target == t) & (ctrl.step == EARLY[ck])]
            vals = {k: d[d.key == k][metric] for k, _ in KEYS}
            means = {k: v.mean() for k, v in vals.items()}
            top = max(means, key=means.get)
            cells = [("\\textbf{" + ms(vals[k]) + "}") if k == top else ms(vals[k]) for k, _ in KEYS]
            first = f"\\multirow{{4}}{{*}}{{{CK[ck]}}}" if i == 0 else ""
            unit = m if metric == "score" else "$r$"
            lines.append(f"{first} & {name} ({unit}) & " + " & ".join(cells) + " \\\\")
        if ck == "50steps":
            lines.append("\\midrule")
    hdr = " & ".join(h for _, h in KEYS)
    write(fname, "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{ll" + "c" * len(KEYS) + "}\n\\toprule\nModel & Target & " + hdr
          + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")

# "adding the layer" ranges, for the text
add = []
for ck in CK:
    for t, name, _ in FT:
        d = ctrl[(ctrl.checkpoint == ck) & (ctrl.target == t) & (ctrl.step == EARLY[ck])]
        piv = d.pivot_table(index=["seed", "step"], columns="key", values="score")
        g = piv["everything the guess shows + layer | mlp"] - piv["everything the guess shows | mlp"]
        add.append(dict(checkpoint=ck, target=name, min=g.min(), max=g.max()))
pd.DataFrame(add).round(3).to_csv(os.path.join(TAB, "adding_layer_ranges.csv"), index=False)

op = pd.read_csv(os.path.join(RES, "phase2", "early_warning", "operating_points.csv"))
op = op[op.strategy == "early"]
lines = []
for _, o in op.iterrows():
    k = int(re.search(r"step (\d+)", o.rule).group(1))
    frac = "20\\%" if k == EARLY[o.checkpoint] else "40\\%"
    lines.append(f"{CK[o.checkpoint]} & {frac} & {100 * o.flagged_share:.1f}\\% & {100 * o.precision:.0f}\\% & {100 * o.recall:.0f}\\% \\\\")
write("tab_detector.tex", "\\begin{tabular}{llccc}\n\\toprule\nModel & Check at & Flagged & Precision & Recall \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
mt = pd.read_csv(os.path.join(RES, "phase2", "early_warning", "matched_failure_compute.csv"))
mt = mt[mt.check_step.isin([10, 200])]
piv = mt.pivot_table(index=["checkpoint", "failure_rate", "finish_filter_compute"], columns="score", values="saving")
lines = []
for (ck, fr, fc), row in piv.sort_index(ascending=[True, False, True]).iterrows():
    probe = next(c for c in piv.columns if "probe" in c and not np.isnan(row[c]))
    vals = [row["guess"], row["guess readout"], row[probe], row["guess readout + layer"], row["oracle"]]
    lines.append((f"{CK[ck]} & {100 * fr:.1f}\\% & {fc:.2f} & " + " & ".join(f"{100 * v:.1f}\\%" for v in vals) + " \\\\").replace(" -", " $-$"))
write("tab_savings.tex", "\\begin{tabular}{lccccccc}\n\\toprule\nModel & Delivered & Finish, then filter & \\multicolumn{5}{c}{Compute saved by early abort} \\\\\n"
      "\\cmidrule(lr){4-8} & floating & (compute) & estimate rule & estimate readout & layer probe & readout + layer & perfect detector \\\\\n\\midrule\n"
      + "\n".join(sorted(lines, key=lambda s: (not s.startswith("50"), s))) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ IntPhys table
ip = pd.concat([pd.read_csv(os.path.join(RES, "intphys_vae", f)) for f in ["probe_results.csv", "probe_results_extra.csv"]])
NULL_F = os.path.join(RES, "intphys_vae", "permutation_null_1000.csv")   # 1,000 shuffles; the 20-shuffle null otherwise
nl = pd.read_csv(NULL_F if os.path.exists(NULL_F) else os.path.join(RES, "intphys_vae", "permutation_null.csv"))
a = ip[ip.subset == "all blocks"]
GROUPS_IP = [("Linear in the latent", [
                 ("pooled (their baseline as read)", "Pooled over time and space"), ("time kept", "Time kept"),
                 ("time + 4x4 space", "Time + $4{\\times}4$ space"),
                 ("time + 4x4 space + signed frame changes", "Time + $4{\\times}4$ space + signed changes")]),
             ("Linear after a fixed nonlinearity: absolute frame-to-frame change", [
                 ("time kept + frame changes", "Time kept + $|\\Delta|$"),
                 ("frame changes only", "$4{\\times}4$ space, $|\\Delta|$ only"),
                 ("time + 4x4 space + frame changes", "Time + $4{\\times}4$ space + $|\\Delta|$")]),
             ("Trained nonlinear readout", [("nonlinear temporal probe", "Small temporal CNN")])]
lines = []
for head, rows_ip in GROUPS_IP:
    lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{head}}}}} \\\\")
    for k, lab in rows_ip:
        d = a[a.features == k]
        pv, pw = 100 * d.per_video, 100 * d.pairwise
        lines.append(f"\\quad {lab} & {pv.mean():.1f} $\\pm$ {pv.std():.1f} & {pw.mean():.1f} $\\pm$ {pw.std():.1f} \\\\")
n = nl[nl.features == "time + 4x4 space + frame changes"]
null_line = (f"Shuffled labels ({nl.perm.nunique():,} shuffles, last $|\\Delta|$ row; mean, 95th pct.) & {100 * n.per_video.mean():.1f}, {100 * n.per_video.quantile(0.95):.1f} & "
             f"{100 * n.pairwise.mean():.1f}, {100 * n.pairwise.quantile(0.95):.1f} \\\\")
write("tab_intphys.tex", "\\begin{tabular}{lcc}\n\\toprule\nReadout of the clean VAE latent & Per-video (\\%) & Pairwise (\\%) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\midrule\n" + null_line + "\n\\bottomrule\n\\end{tabular}\n")
# Result figures (3 to 6) are drawn by make_result_figures.py with the shared paper style.
import runpy
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_result_figures.py"), run_name="__main__")
print("figures:", sorted(os.listdir(FIG)))
# Tuned, seed-pooled readouts (fair_readouts.py), once any of its results exist.
if os.path.exists(os.path.join(RES, "phase2", "fair", "50steps_scores.csv")):
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_tuned_assets.py"), run_name="__main__")

# ------------------------------------------------------------------ label threshold sensitivity (appendix)
SENS = os.path.join(RES, "sensitivity")
if os.path.exists(os.path.join(SENS, "physics.csv")):
    ph, dt = pd.read_csv(os.path.join(SENS, "physics.csv")), pd.read_csv(os.path.join(SENS, "detector.csv"))
    rng_ = lambda s, f="{:.1f}": f.format(s.min()) if abs(s.max() - s.min()) < 1e-9 else (f.format(s.min()) + "--" + f.format(s.max()))
    order = ["default", "contact x0.75", "contact x1.25", "support 10 cm", "support 20 cm", "floating 1 m/s2", "floating 3 m/s2"]
    label = {"default": "Default", "contact x0.75": "Contact heights $\\times 0.75$", "contact x1.25": "Contact heights $\\times 1.25$",
             "support 10 cm": "Support rule 10\\,cm", "support 20 cm": "Support rule 20\\,cm",
             "floating 1 m/s2": "Floating cut-off 1\\,m/s$^2$", "floating 3 m/s2": "Floating cut-off 3\\,m/s$^2$"}
    lines = []
    for st in order:
        p, d = ph[ph.setting == st], dt[dt.setting == st]
        c = [rng_(p[p.checkpoint == ck].floating_test) for ck in ["50steps", "1000steps"]]
        c += [rng_(p[p.checkpoint == ck].jump_g_jump, "{:.2f}") for ck in ["50steps", "1000steps"]]
        c += [rng_(d.precision, "{:.2f}"), rng_(d.recall, "{:.2f}")]
        lines.append(f"{label[st]} & " + " & ".join(c) + " \\\\")
    nseeds = ph[ph.setting == "default"].groupby("checkpoint").seed.nunique().to_dict()
    write("tab_sensitivity.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n"
          "& \\multicolumn{2}{c}{Floating, test captions (\\% of frames)} & \\multicolumn{2}{c}{Jump gravity, curated jump prompts (m/s$^2$)} "
          "& \\multicolumn{2}{c}{Early rule at 20\\%} \\\\\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
          "Setting & 50-step & 1000-step & 50-step & 1000-step & Precision & Recall \\\\\n\\midrule\n"
          + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("sensitivity seeds per checkpoint:", nseeds)
