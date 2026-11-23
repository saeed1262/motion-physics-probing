"""Build the paper's web figures: export each figure's data from the result files, then print its page
(paper/figures_web/<name>.html) to a vector PDF with headless Chrome, plus a PNG preview.

    .venv/bin/python scripts/make_web_figures.py [names ...]

Writes paper/figures_web/data/<name>.js and paper/figures/web/<name>.pdf (+ _preview.png). Every number
drawn comes from a result CSV; the pages only lay them out.
"""
import glob, json, os, subprocess, sys, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "paper", "figures_web")
OUT = os.path.join(ROOT, "paper", "figures", "web")
RES = os.path.join(ROOT, "results")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"paper_figures_chrome_{os.getpid()}")   # one per process


PALETTE = {k: v for k, v in json.load(open(os.path.join(WEB, "palette.json"))).items() if not k.startswith("_")}


def write_palette_css():
    with open(os.path.join(WEB, "palette.css"), "w") as f:
        f.write("/* written by scripts/make_web_figures.py from palette.json; do not edit */\n:root {\n"
                + "".join(f"  --{k}: {v};\n" for k, v in PALETTE.items()) + "}\n")


def silhouettes(V):
    """Flat silhouettes of a sequence of poses in the body color, fading in over time (faint early, solid late)."""
    import body_images as bi
    n = len(V)
    return [(V[k], bi.hex_rgb(PALETTE["body"]), 0.16 + 0.69 * k / max(n - 1, 1)) for k in range(n)]


def write_data(name, obj):
    os.makedirs(os.path.join(WEB, "data"), exist_ok=True)
    with open(os.path.join(WEB, "data", f"{name}.js"), "w") as f:
        f.write(f"window.FIG = {json.dumps(obj)};\n")


def render(name, width, height):
    os.makedirs(OUT, exist_ok=True)
    url = "file://" + os.path.join(WEB, f"{name}.html")
    # A throwaway profile, so the renderer never touches (or hands off to) the user's running Chrome.
    base = [CHROME, "--headless=new", f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
            "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
            "--virtual-time-budget=8000", "--run-all-compositor-stages-before-draw"]
    pdf, png = os.path.join(OUT, f"{name}.pdf"), os.path.join(OUT, f"{name}_preview.png")
    run_chrome(base + ["--no-pdf-header-footer", f"--print-to-pdf={pdf}", url], pdf)
    run_chrome(base + ["--force-device-scale-factor=2", f"--window-size={width},{height}", f"--screenshot={png}", url], png)
    print("figure", os.path.relpath(pdf, ROOT))


def run_chrome(cmd, out, timeout=60):
    # Headless Chrome on macOS writes the file and then often keeps running: wait for the file to appear and
    # stop changing, then end this Chrome (its own throwaway profile only).
    if os.path.exists(out):
        os.remove(out)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0, last = time.time(), -1
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        size = os.path.getsize(out) if os.path.exists(out) else -1
        if size > 0 and size == last:
            break
        last = size
        if proc.poll() is not None and size > 0:
            break
    proc.terminate()
    subprocess.run(["pkill", "-f", f"user-data-dir={PROFILE}"], capture_output=True)
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        raise RuntimeError(f"Chrome did not write {out}")


# ------------------------------------------------------------------ fall rate across denoising
def fig_steps_fall():
    models = []
    for ck, name, short, steps in [("50steps", "50-step: the estimate catches up", "50-step", [10, 20, 29, 39]),
                                   ("1000steps", "1000-step: the same pattern", "1000-step", [200, 400, 599, 799])]:
        d = dict(name=name, short=short, steps=[20, 40, 60, 80], explicit=[], estimate=[], layer=[], both=[],
                 gain=[], gain_lo=[], gain_hi=[])
        for k in steps:
            suf = "" if k in (10, 200) else f"_step{k}"
            s = pd.read_csv(os.path.join(RES, "jump_heavy", "fair", f"{ck}{suf}_scores.csv"))
            s = s[s.target == "fall_gravity"].set_index("readout").score
            b = pd.read_csv(os.path.join(RES, "jump_heavy", "fair", f"{ck}{suf}_bootstrap.csv"))
            b = b[(b.target == "fall_gravity") & (b.comparison == "conditional gain (layer): fallback") & (b.measure == "score")].iloc[0]
            d["explicit"].append(s["explicit"]); d["estimate"].append(s["estimate | tuned"]); d["layer"].append(s["layer | tuned"])
            d["both"].append(s["estimate + layer | tuned+fallback"])
            d["gain"].append(b.observed); d["gain_lo"].append(b.ci_low); d["gain_hi"].append(b.ci_high)
        models.append(d)
    write_data("fig_steps_fall", dict(models=models, film=filmstrip_jump()))
    render("fig_steps_fall", 624, 330)


def filmstrip_jump():
    """Top row of fig_steps_fall: one jump from the estimate at every recorded step. The jump is chosen by a
    fixed rule (jump-prompt runs, 50-step, seed 0): among falling stretches of at least 8 frames that the
    estimate shows at every step and over which the body travels at least 0.4 m, the one whose 20%
    fall-rate error is closest to the median error (a typical case, not a selected one)."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from example_picks import stretches, LABELS
    import body_images as bi
    d = os.path.join(ROOT, "data", "jump_heavy", "50steps_seed0")
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    fin, x0 = lab["final"].item(), lab["x0"].item()
    steps = meta["record_steps"]
    est_g = lambda k, i, s, e: next((gg for ss, ee, gg in stretches(x0[k]["flight_gravity"][i]) if ss < e and ee > s), np.nan)
    rows = [(i, s, e, g) for i in range(len(meta["all_prompts"])) for s, e, g in stretches(fin["flight_gravity"][i])
            if g >= LABELS["FLOAT_G_MAX"] and e - s >= 8]
    rows = [r for r in rows if all(np.isfinite(est_g(k, *r[:3])) for k in steps[1:5])]
    J_all = np.concatenate([np.load(p)["joints"] for p in files])
    # display condition: the root travels at least 0.4 m over the stretch, so the arc can be seen from the side
    rows = [r for r in rows if np.linalg.norm(J_all[r[0], r[2] - 1, 0, [0, 2]] - J_all[r[0], r[1], 0, [0, 2]]) >= 0.4]
    err = np.array([abs(est_g(steps[1], *r[:3]) - r[3]) for r in rows])
    i, s, e, g_final = rows[int(np.argmin(np.abs(err - np.median(err))))]
    J = J_all[i]
    X = {k: np.concatenate([np.load(p)[f"x0j_{k}"] for p in files])[i].astype(np.float32) for k in steps}
    to_local = bi.local_frame(J[0, 0, [0, 2]], J[-1, 0, [0, 2]], J[s, 0, [0, 2]])
    com_final = LABELS["center_of_mass"](J)
    apex = s + int(np.argmax(com_final[s:e, 1]))
    frames = [max(s - 3, 0), apex, min(e + 2, len(J) - 1)]
    versions = [(k, X[k], f"{20 * n}%") for n, k in enumerate(steps[:-1])] + [(None, J, "final")]   # the six recorded steps are 0..100% evenly
    seg = np.arange(max(s - 8, 0), min(e + 8, len(J)))
    cols, verts_all = [], []
    for k, jj, name in versions:
        V = bi.fit(jj[frames], f"film_jh50s0_m{i}_{'final' if k is None else k}")
        verts_all.append(to_local(V))
    allv = np.concatenate(verts_all)
    x0b, x1b = allv[..., 0].min() - 0.15, allv[..., 0].max() + 0.15
    y1b = bi.project(allv)[1].max() + 0.05
    box = (x0b, x1b, -0.2, y1b)
    os.makedirs(os.path.join(WEB, "img"), exist_ok=True)
    for (k, jj, name), V in zip(versions, verts_all):
        img = f"img/film_{name.rstrip('%')}.jpg"
        if name in ("0%", "20%", "final"):              # bodies only where they differ visibly; the arcs show every step
            bi.render(silhouettes(V), os.path.join(WEB, img), box, flat=True)
        c = to_local(LABELS["center_of_mass"](jj)[None])[0][seg]
        cx, cy, _ = bi.project(c)
        g = (fin if k is None else x0[k])["flight_gravity"][i][seg]
        gk = g_final if k is None else est_g(k, i, s, e)
        cols.append(dict(name=name, img=img if name in ("0%", "20%", "final") else None,
                         com=[[float(a), float(b)] for a, b in zip(cx, cy)],
                         height=[[float(f / 20.0), float(h)] for f, h in zip(seg, LABELS["center_of_mass"](jj)[seg, 1])],
                         air=[bool(v) for v in np.isfinite(g)], g=float(gk) if np.isfinite(gk) else None))
    return dict(prompt=meta["all_prompts"][i].rstrip("."), g_final=float(g_final), box=dict(zip(["x0", "x1", "y0", "y1"], map(float, box))),
                cols=cols, median_err20=float(np.median(err)), n_stretches=len(rows), stretch=[float(s / 20.0), float((e - 1) / 20.0)])


# ------------------------------------------------------------------ what the layer knows
def fig_layer_knows():
    """Held-out predictions of the final fall rate (jump prompts, 50-step, 20% of denoising): per jump, the
    tuned estimate readout against the tuned readout of estimate + layer (with fallback); and, among jumps
    that travel at least 0.4 m (so the arc can be seen), the three where adding the layer reduces the error
    most, labelled as such."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from example_picks import LABELS
    import body_images as bi
    z = np.load(os.path.join(RES, "jump_heavy", "fair", "preds", "50steps_fall_gravity.npz"))
    y, seed, pe, pb = z["y"], z["seed"], z["estimate | tuned"], z["estimate + layer | tuned+fallback"]
    # rows are (motion, recorded frame) in row-major order per seed, seeds in order: rebuild the motion index
    motion = np.empty(len(y), int)
    runs = sorted(glob.glob(os.path.join(ROOT, "data", "jump_heavy", "50steps_seed*")))
    for d in runs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        FR = np.arange(0, meta["n_frames"], meta["frame_stride"])
        g = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)["final"].item()["flight_gravity"][:, FR]
        mi, _ = np.nonzero(np.isfinite(g) & (g >= LABELS["FLOAT_G_MAX"]))
        sel = seed == meta["seed"]
        assert sel.sum() == len(mi)
        motion[sel] = mi
    key = np.stack([seed, motion, np.round(y, 4)], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    agg = lambda v: np.bincount(inv, v) / np.bincount(inv)
    Y, E, B = agg(y), agg(pe), agg(pb)
    r2 = lambda p: 1 - np.mean((y - p) ** 2) / np.var(y)
    gain = np.abs(E - Y) - np.abs(B - Y)
    ex = []
    for k in np.argsort(-gain):
        if len(ex) == 3:
            break
        s_, m_, gv = int(uk[k, 0]), int(uk[k, 1]), float(Y[k])
        d = [r for r in runs if r.endswith(f"seed{s_}")][0]
        meta = json.load(open(os.path.join(d, "meta.json")))
        files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
        J = np.concatenate([np.load(p)["joints"] for p in files])[m_]
        X = np.concatenate([np.load(p)["x0j_10"] for p in files])[m_].astype(np.float32)
        gfull = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)["final"].item()["flight_gravity"][m_]
        fr = np.where(np.isclose(gfull, gv, atol=1e-3))[0]
        s, e = int(fr.min()), int(fr.max()) + 1
        if np.linalg.norm(J[e - 1, 0, [0, 2]] - J[s, 0, [0, 2]]) < 0.4:   # display condition: a jump that travels, as in (a) of fig_steps_fall
            continue
        com = LABELS["center_of_mass"](J)
        apex = s + int(np.argmax(com[s:e, 1]))
        frames = [max(s - 3, 0), apex, min(e + 2, len(J) - 1)]
        to_local = bi.local_frame(J[0, 0, [0, 2]], J[-1, 0, [0, 2]], J[s, 0, [0, 2]])
        V = to_local(bi.fit(J[frames], f"knows_jh50s{s_}_m{m_}_final"))
        seg = np.arange(max(s - 8, 0), min(e + 8, len(J)))
        cf = bi.project(to_local(com[None])[0][seg]); ce = bi.project(to_local(LABELS["center_of_mass"](X)[None])[0][seg])
        # box around the bodies and the final arc only (the estimate's path is clipped on the page), so all three
        # examples can share one scale
        box = (min(V[..., 0].min(), cf[0].min()) - 0.15, max(V[..., 0].max(), cf[0].max()) + 0.15,
               -0.2, max(bi.project(V)[1].max(), cf[1].max()) + 0.05)
        img = f"img/knows_{len(ex)}.jpg"
        bi.render(silhouettes(V), os.path.join(WEB, img), box, flat=True)
        ex.append(dict(prompt=meta["all_prompts"][m_].rstrip("."), img=img, box=dict(zip(["x0", "x1", "y0", "y1"], map(float, box))),
                       final=[[float(a), float(b)] for a, b in zip(cf[0], cf[1])], est=[[float(a), float(b)] for a, b in zip(ce[0], ce[1])],
                       air=[bool(s <= f < e) for f in seg], g=gv, pe=float(E[k]), pb=float(B[k])))
    write_data("fig_layer_knows", dict(points=[[float(a), float(b), float(c)] for a, b, c in zip(Y, E, B)],
                                       r2_est=float(r2(pe)), r2_both=float(r2(pb)), n=int(len(Y)), examples=ex))
    render("fig_layer_knows", 624, 214)


# ------------------------------------------------------------------ how the labels are computed (Figure 2)
def fig_labels():
    """The label explainer on bodies, same examples as make_label_figure.py (example_picks.py rules):
    (a) a walk at its most and least balanced contact frames, support region and extrapolated center of
    mass drawn on the floor; (b) its margins over time; (c) the jump and (d) the floating walk: center-of-
    mass height with the fitted parabola, and one floating frame with the gap under the feet."""
    import numpy as np
    from shapely.geometry import MultiPoint
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from example_picks import LABELS, FLOAT_G_MAX, load, pick, stretches
    import body_images as bi
    FPS, FOOT, RADIUS = LABELS["FPS"], LABELS["FOOT_JOINTS"], LABELS["FOOT_RADIUS"]
    prompts, pid, J, X0, final, x0lab = load()
    jump, floatw, walk = pick(prompts, pid, J, final, x0lab)
    out = {}
    # (a, b) balance
    j = J[walk].astype(float)
    com = LABELS["center_of_mass"](j); contact = LABELS["foot_contacts"](j)
    v = np.gradient(com, 1.0 / FPS, axis=0)
    xcom = com[:, [0, 2]] + v[:, [0, 2]] / np.sqrt(9.81 / max(np.median(com[:, 1]), 0.3))
    dm, sm = final["xcom_margin"][walk], final["balance_margin"][walk]
    defined = np.where(np.isfinite(dm))[0]
    t0 = int(defined[len(defined) // 3]); win = np.arange(t0, min(t0 + 30, len(j)))
    ok = [t for t in win if np.isfinite(dm[t])]
    t_in, t_out = int(max(ok, key=lambda t: dm[t])), int(min(ok, key=lambda t: dm[t]))
    elev = np.deg2rad(32)
    to_local = bi.local_frame(com[0, [0, 2]], com[-1, [0, 2]], com[t_in, [0, 2]])
    V = to_local(bi.fit(j[[t_in, t_out]].astype(np.float32), f"labels_walk_m{walk}"))
    poses = []
    for n, t in enumerate([t_in, t_out]):
        Vn = V[n].copy(); shift = Vn[:, 0].mean(); Vn[:, 0] -= shift
        loc = lambda p, s=shift: (lambda q: q - np.array([s, 0, 0]))(to_local(p))
        pts = j[t, FOOT][contact[t]][:, [0, 2]]
        poly = np.array(MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(RADIUS).exterior.coords)
        P = lambda xz: bi.project(loc(np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], 1)), elev)
        sp = P(poly); cg = P(com[t:t + 1, [0, 2]]); xc = P(xcom[t:t + 1])
        box = (min(bi.project(Vn, elev)[0].min(), sp[0].min()) - 0.12, max(bi.project(Vn, elev)[0].max(), sp[0].max()) + 0.12,
               min(sp[1].min(), -0.12) - 0.05, bi.project(Vn, elev)[1].max() + 0.04)
        img = f"img/labels_walk_{n}.jpg"
        bi.render([(Vn, bi.hex_rgb(PALETTE["body"]), 0.85)], os.path.join(WEB, img), box, elev=elev, floor_rgb=None, flat=True)
        poses.append(dict(img=img, box=dict(zip(["x0", "x1", "y0", "y1"], map(float, box))), t=t / FPS, margin=float(dm[t]),
                          support=[[float(a), float(b)] for a, b in zip(sp[0], sp[1])], com=[float(cg[0][0]), float(cg[1][0])],
                          xcom=[float(xc[0][0]), float(xc[1][0])]))
    out["walk"] = dict(prompt=prompts[walk].rstrip("."), poses=poses, t=[float(x) for x in win / FPS],
                       dyn=[None if not np.isfinite(x) else float(100 * x) for x in dm[win]],
                       stat=[None if not np.isfinite(x) else float(100 * x) for x in sm[win]])
    # (c, d) the gravity test
    def gravity(i, window):
        jj = J[i].astype(float); c = LABELS["center_of_mass"](jj); g = final["flight_gravity"][i]
        s, e = window
        st = []
        for a_, b_, gg in stretches(g):
            if b_ <= s or a_ >= e:
                continue
            coef = np.polyfit(np.arange(b_ - a_) / FPS, c[a_:b_, 1], 2)
            tf = np.linspace(0, (b_ - a_ - 1) / FPS, 40)
            st.append(dict(t0=a_ / FPS, t1=(b_ - 1) / FPS, g=float(gg), floating=bool(gg < FLOAT_G_MAX),
                           fit=[[float(a_ / FPS + x), float(np.polyval(coef, x))] for x in tf]))
        tt = np.arange(s, e)
        return dict(prompt=prompts[i].rstrip("."), com=[[float(x / FPS), float(c[x, 1])] for x in tt],
                    foot=[[float(x / FPS), float(jj[x, FOOT, 1].min())] for x in tt], stretches=st)
    i, s, e, _ = jump
    out["jump"] = gravity(i, (max(s - 15, 0), min(e + 15, 120)))
    fl = np.isfinite(final["flight_gravity"][floatw]) & (final["flight_gravity"][floatw] < FLOAT_G_MAX)
    fi = np.where(fl)[0]
    out["float"] = gravity(floatw, (max(fi.min() - 20, 0), min(fi.max() + 20, 120)))
    # one floating frame of the teaser's fits, with the gap under the feet
    fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
    frs = list(fits["float_frames"]); k = max(range(len(frs)), key=lambda n: fl[frs[n]] * J[floatw][frs[n], FOOT, 1].min())
    tf_ = frs[k]
    to_l = bi.local_frame(J[floatw][0, 0, [0, 2]], J[floatw][-1, 0, [0, 2]], J[floatw][tf_, 0, [0, 2]])
    Vf = to_l(fits["float_final_verts"][k]); Vf[:, 0] -= Vf[:, 0].mean()
    boxf = (bi.project(Vf)[0].min() - 0.1, bi.project(Vf)[0].max() + 0.1, -0.12, bi.project(Vf)[1].max() + 0.04)
    bi.render([(Vf, bi.hex_rgb(PALETTE["body_fail"]), 0.85)], os.path.join(WEB, "img/labels_float.jpg"), boxf, floor_depth=(-0.3, 0.3), flat=True)
    out["float"]["inset"] = dict(img="img/labels_float.jpg", box=dict(zip(["x0", "x1", "y0", "y1"], map(float, boxf))),
                                 gap=float(J[floatw][tf_, FOOT, 1].min()), lowest_vertex=float(Vf[:, 1].min()), t=float(tf_ / FPS))
    write_data("fig_labels", out)
    render("fig_labels", 624, 330)


# ------------------------------------------------------------------ Figure 3: probe scores across denoising
def fig_probe_steps():
    """Same data as make_result_figures.py's fig_steps.pdf: linear probes per seed on the 50-step runs
    (best layer, explicit label, estimate, noisy input), and the fixed MLP on everything the estimate shows."""
    import glob as _g
    r = pd.read_csv(os.path.join(RES, "phase2", "50steps", "probe_scores_all_runs.csv"))
    ctrl = pd.concat([pd.read_csv(p) for p in _g.glob(os.path.join(RES, "phase2", "controls", "*_nonlinear_controls.csv"))])
    ctrl["key"] = ctrl.features.str.replace(r"layer \d", "layer", regex=True) + " | " + ctrl.model
    steps = sorted(r.step.unique())
    pct = lambda s: round(100 * s / steps[-1])
    out = []
    for t, name, sub, metric in [("xcom_margin", "Balance is readable from the layers early", "dynamic balance", "R²"),
                                 ("in_flight", "So is flight", "in flight", "AUC")]:
        d = r[r.target == t]
        best = d[d.source.str.startswith("layer")].groupby(["seed", "step"]).score.max().reset_index()
        ser = []
        for key, lab, dd in [("layer", "best layer", best), ("explicit", "explicit label", d[d.source == "explicit"]),
                             ("est", "estimate", d[d.source == "x0 probe"]), ("noisy", "noisy input", d[d.source == "x_t probe"])]:
            g = dd.groupby("step").score
            ser.append(dict(key=key, label=lab, pts=[[pct(s), float(g.mean()[s]), float(g.min()[s]), float(g.max()[s])] for s in steps]))
        cc = ctrl[(ctrl.checkpoint == "50steps") & (ctrl.target == t) & (ctrl.key == "everything the guess shows | mlp")].groupby("step").score
        mlp = [[pct(s), float(cc.mean()[s]), float(cc.std()[s])] for s in cc.mean().index]
        out.append(dict(name=name, sub=f"{sub} ({metric}), 50-step, linear probes; bands: seed range", metric=metric, series=ser, mlp=mlp))
    write_data("fig_probe_steps", dict(panels=out))
    render("fig_probe_steps", 624, 196)


# ------------------------------------------------------------------ tuned readouts: shared loading
SETS = [("phase2", "50steps", "50-step"), ("phase2", "1000steps", "1000-step"), ("phase2", "motiondiffuse", "MotionDiffuse"),
        ("jump_heavy", "50steps", "50-step, jump prompts"), ("jump_heavy", "1000steps", "1000-step, jump prompts"),
        ("jump_heavy", "motiondiffuse", "MotionDiffuse, jump prompts")]
TARGETS = [("xcom_margin", "Dynamic balance"), ("balance_margin", "Static balance"), ("in_flight", "In flight"),
           ("float_vs_fall", "Floats or falls"), ("fall_gravity", "Fall rate")]
GAIN = "conditional gain (layer): fallback"


def read_fair(kind, spec=""):
    out = [pd.read_csv(p) for tag, ck, _ in SETS for p in [os.path.join(RES, tag, "fair", f"{ck}{spec}_{kind}.csv")] if os.path.exists(p)]
    return pd.concat(out, ignore_index=True)


def fig_gain():
    """The forest plot of make_tuned_assets.py (fig_gain.pdf), without the main-data fall-rate rows (1,147 and
    1,620 falling frames; intervals about -0.6 to +0.5), which stay in Table tab_tuned."""
    import numpy as np
    bt = read_fair("bootstrap")
    rows = []
    for t, tname in TARGETS:
        for tag, ck, cname in SETS:
            if t == "fall_gravity" and tag == "phase2":
                continue
            b = bt[(bt.tag == tag) & (bt.checkpoint == ck) & (bt.target == t)]
            if not len(b[b.comparison == GAIN]):
                continue
            c = os.path.join(RES, tag, "fair", f"{ck}_chosen.csv")
            c = pd.read_csv(c) if os.path.exists(c) else pd.DataFrame(columns=["target", "features", "fallback_choice"])
            kept = int(((c.target == t) & (c.features == "estimate + layer") & (c.fallback_choice == "estimate")).sum())
            row = dict(target=tname, model=cname, allfb=kept == 5)   # the fallback kept the estimate in every fold
            for comp, k in [("fair: estimate tuned - layer tuned", "fair"), (GAIN, "gain")]:
                # a score whose interval could not be computed (too few prompts) is left out (null)
                row[k] = {m: ([float(r.observed), float(r.ci_low), float(r.ci_high)]
                              if np.isfinite([r.observed, r.ci_low, r.ci_high]).all() else None)
                          for m in ["score", "within_prompt_r", "within_prompt_frame_r"]
                          for r in [b[(b.comparison == comp) & (b.measure == m)].iloc[0]]}
            rows.append(row)
    write_data("fig_gain", dict(rows=rows))
    render("fig_gain", 624, 40 + 17 * len(rows) + 46)


def fig_layers():
    """The per-layer gain of make_tuned_assets.py (fig_layers.pdf): adding one layer (or all eight) to the tuned
    estimate readout, with fallback, at 20% of denoising; the main data's fall rate is left out (too few frames)."""
    sw = read_fair("bootstrap", "_sweep")
    LAY = [str(l) for l in range(8)] + ["all"]
    panels = []
    for t, tname in [("xcom_margin", "Dynamic balance"), ("in_flight", "In flight"), ("float_vs_fall", "Floats or falls"),
                     ("fall_gravity", "Fall rate")]:
        ser = []
        for tag, ck, cname in SETS:
            d = sw[(sw.tag == tag) & (sw.checkpoint == ck) & (sw.target == t) & (sw.measure == "score")]
            if d.empty or (t == "fall_gravity" and tag == "phase2"):
                continue
            pts = []
            for l in LAY:
                nm = f"conditional gain (layer {l}): fallback" if l != "all" else "conditional gain (layers all): fallback"
                r = d[d.comparison == nm].iloc[0]
                pts.append([l, float(r.observed), float(r.ci_low), float(r.ci_high)])
            ser.append(dict(tag=tag, ck=ck, name=cname, pts=pts))
        panels.append(dict(name=tname, series=ser))
    write_data("fig_layers", dict(panels=panels))
    render("fig_layers", 624, 344)


def fig_early():
    """make_result_figures.py's fig_early.pdf: delivered floating against compute, final check kept, retry budget
    1 to 20, for finish-then-filter and early abort at 20% with the rule, a layer probe and a perfect detector."""
    ew = pd.read_csv(os.path.join(RES, "phase2", "early_warning", "summary.csv"))
    out = []
    for ck, name, early in [("50steps", "50-step model", 10), ("1000steps", "1000-step model", 200)]:
        d = ew[ew.checkpoint == ck]
        ff = d[d.strategy.str.startswith("finish+filter")].sort_values("budget")
        probe = next(s for s in d.score.dropna().unique() if "probe" in str(s))
        curve = lambda sc: [[float(r.rel_compute), float(100 * r.failure_rate), int(r.budget)] for _, r in
                            d[(d.score == sc) & (d.check_step == early) & d.strategy.str.startswith("early+final, budget")].sort_values("budget").iterrows()]
        out.append(dict(name=name, filter=[[float(r.rel_compute), float(100 * r.failure_rate), int(r.budget)] for _, r in ff.iterrows()],
                        rule=curve("guess"), probe=curve(probe), oracle=curve("oracle")))
    write_data("fig_early", dict(models=out))
    render("fig_early", 624, 206)


def fig_teaser():
    """Figure 1 in the web style: the examples, fits and numbers of make_teaser_smpl.py (fits from
    fit_smpl_teaser.py), the bodies rendered as images and every label laid out on the page."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from example_picks import LABELS, FLOAT_G_MAX, load, stretches
    import body_images as bi
    fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
    prompts, pid, J, X0, final, x0lab = load()
    VERS = [("est20", 10, "estimate at 20%"), ("est40", 20, "estimate at 40%"), ("final", None, "final motion")]
    out = dict(rows=[v[2] for v in VERS], cols=[])
    for kind in ["jump", "float"]:
        idx = int(fits[f"{kind}_motion"]); frames = list(fits[f"{kind}_frames"])
        to_local = bi.local_frame(J[idx][0, 0, [0, 2]], J[idx][-1, 0, [0, 2]], J[idx][0, 0, [0, 2]])
        cells, boxes, Vs = [], [], []
        for ver, k, _ in VERS:
            V = to_local(fits[f"{kind}_{ver}_verts"])
            if kind == "float":
                V = V.copy()
                for n in range(len(V)):
                    V[n][:, 0] += n * 0.62 - V[n][:, 0].mean()
            Vs.append(V)
        allv = np.concatenate(Vs)
        box = (bi.project(allv)[0].min() - 0.2, bi.project(allv)[0].max() + 0.2, -0.22, bi.project(allv)[1].max() + 0.04)
        for (ver, k, _), V in zip(VERS, Vs):
            jj = J[idx] if k is None else X0[k][idx]
            labs = final if k is None else x0lab[k]
            g = labs["flight_gravity"][idx]
            img = f"img/teaser_{kind}_{ver}.jpg"
            cell = dict(img=img)
            if kind == "jump":
                bodies = silhouettes(V)
                c = to_local(LABELS["center_of_mass"](jj)[None])[0]
                seg = np.arange(frames[0], frames[-1] + 1)
                cx, cy, _ = bi.project(c[seg])
                cell.update(com=[[float(a), float(b)] for a, b in zip(cx, cy)], air=[bool(v) for v in np.isfinite(g[seg])])
                fg = [gg for ss, ee, gg in stretches(g) if ss <= frames[-1] and ee >= frames[0] and gg >= FLOAT_G_MAX]
                cell["text"] = f"fitted fall rate {fg[0]:.1f} m/s²" if fg else "no falling airborne stretch"
            else:
                fl = np.isfinite(g) & (g < FLOAT_G_MAX)
                bodies = [(V[n], bi.hex_rgb(PALETTE["body_fail"] if fl[t_] else PALETTE["body"]), 0.85) for n, t_ in enumerate(frames)]
                cell.update(heights=[[float(n * 0.62), float(100 * jj[t_, [7, 10, 8, 11], 1].min()), bool(fl[t_])] for n, t_ in enumerate(frames)],
                            text=f"floating in {int(fl.sum())} of {len(fl)} frames")
            bi.render(bodies, os.path.join(WEB, img), box, flat=True)
            cells.append(cell)
        out["cols"].append(dict(kind=kind, prompt=prompts[idx].rstrip("."), box=dict(zip(["x0", "x1", "y0", "y1"], map(float, box))), cells=cells))
    write_data("fig_teaser", out)
    render("fig_teaser", 624, 372)


def _ctrl():
    import glob as _g
    c = pd.concat([pd.read_csv(p) for p in _g.glob(os.path.join(RES, "phase2", "controls", "*_nonlinear_controls.csv"))])
    c["key"] = c.features.str.replace(r"layer \d", "layer", regex=True) + " | " + c.model
    return c


def fig_fair():
    """Appendix: make_result_figures.py's fig_fair.pdf, the fixed-MLP comparison per seed (mean and SD over seeds)."""
    ctrl = _ctrl()
    ROWS = [("explicit | linear", "explicit label", "dim"), ("guess features | linear", "estimate, linear", "est"),
            ("everything the guess shows | mlp", "estimate, MLP", "est"), ("layer | linear", "layer, linear", "layer"),
            ("layer | mlp", "layer, MLP", "layer"), ("everything the guess shows + layer | mlp", "estimate + layer, MLP", "both")]
    panels = []
    for t, name, metric in [("xcom_margin", "Dynamic balance", "R²"), ("balance_margin", "Static balance", "R²"),
                            ("in_flight", "In flight", "AUC"), ("flight_gravity", "Flight gravity", "R²")]:
        rows = []
        for key, lab, c in ROWS:
            v = {}
            for ck, early in [("50steps", 10), ("1000steps", 200)]:
                d = ctrl[(ctrl.checkpoint == ck) & (ctrl.target == t) & (ctrl.step == early) & (ctrl.key == key)].score
                v[ck] = [float(d.mean()), float(d.std())]
            rows.append(dict(label=lab, color=c, **v))
        panels.append(dict(name=name, metric=metric, rows=rows))
    write_data("fig_fair", dict(panels=panels))
    render("fig_fair", 624, 186)


def fig_learning():
    """Appendix: make_result_figures.py's fig_learning.pdf, the pilot's learning curve (fixed linear probes)."""
    import numpy as np
    lc = pd.read_csv(os.path.join(RES, "phase2", "controls", "50steps_training_size.csv"))
    lc = lc[(lc.step == 10) & lc.design.str.startswith("5-fold")].copy()
    lc["n"] = lc.n_train_prompts.where(lc.n_train_prompts < 150, 200)
    ser = []
    for src, lab, c in [("explicit", "explicit label (1 feature)", "dim"), ("layer 7 (linear)", "layer 7, linear", "layer")]:
        g = lc[lc.source == src].groupby("n").r2
        m, sd = g.mean(), g.std().fillna(0)
        ser.append(dict(label=lab, color=c, pts=[[int(n), float(m[n]), float(m[n] - sd[n]), float(m[n] + sd[n])] for n in m.index]))
    diff = pd.Series({p[0]: p[1] for p in ser[1]["pts"]}) - pd.Series({p[0]: p[1] for p in ser[0]["pts"]})
    k = np.where(np.diff(np.sign(diff.sort_index().values)) > 0)[0]
    cross = [int(diff.sort_index().index[k[0]]), int(diff.sort_index().index[k[0] + 1])] if len(k) else None
    write_data("fig_learning", dict(series=ser, cross=cross))
    render("fig_learning", 340, 196)


def fig_tuned_curve():
    """make_tuned_assets.py's fig_tuned_curve.pdf: tuned readouts trained on fewer prompts (same test folds; two
    draws per size, mean and range), for dynamic balance (50-step) and the fall rate on the jump prompts."""
    cv = read_fair("curve")
    KEYS = [("explicit", "explicit label"), ("layer | ridge", "layer, ridge"), ("layer | tuned", "layer, tuned"),
            ("estimate | tuned", "estimate, tuned"), ("estimate + layer | tuned+fallback", "both, with fallback")]
    panels = []
    for tag, ck, t, name in [("phase2", "50steps", "xcom_margin", "Dynamic balance, 50-step"),
                             ("jump_heavy", "50steps", "fall_gravity", "Fall rate, jump prompts, 50-step"),
                             ("jump_heavy", "1000steps", "fall_gravity", "Fall rate, jump prompts, 1000-step")]:
        d = cv[(cv.tag == tag) & (cv.checkpoint == ck) & (cv.target == t)]
        full = d.n_train_prompts.max()
        d = d[(d.n_train_prompts < full) | (d.index >= d.index[-len(d[d.n_train_prompts == full])])]
        ser = []
        for k, lab in KEYS:
            g = d[d.readout == k].groupby("n_train_prompts").score
            ser.append(dict(key=k, label=lab, pts=[[int(n), float(g.mean()[n]), float(g.min()[n]), float(g.max()[n])] for n in g.mean().index]))
        panels.append(dict(name=name, series=ser))
    write_data("fig_tuned_curve", dict(panels=panels))
    render("fig_tuned_curve", 624, 200)


def fig_setup():
    # The probing diagram; it draws the jump of fig_steps_fall (that figure's data file and images).
    render("fig_setup", 624, 250)


FIGURES = {"fig_steps_fall": fig_steps_fall, "fig_setup": fig_setup, "fig_layer_knows": fig_layer_knows, "fig_labels": fig_labels,
           "fig_probe_steps": fig_probe_steps, "fig_gain": fig_gain, "fig_layers": fig_layers, "fig_early": fig_early,
           "fig_tuned_curve": fig_tuned_curve, "fig_teaser": fig_teaser, "fig_fair": fig_fair, "fig_learning": fig_learning}

if __name__ == "__main__":
    write_palette_css()
    for n in (sys.argv[1:] or FIGURES):
        FIGURES[n]()
