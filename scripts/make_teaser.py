"""Teaser figure: what the model's own clean-motion estimate shows, early in denoising.

Chronophotography-style side views of two recorded 50-step samples (seed 0): a jump and a
floating walk. Each is drawn three times: the model's estimate at 20% and 40% of denoising,
and the final motion. Examples are picked automatically from the cached labels:
- jump: a curated jump prompt whose final motion has a ballistic stretch (8.5 to 11 m/s^2)
  that the 20% estimate already shows falling (at least 5 m/s^2);
- floating walk: a non-jump curated prompt whose final motion floats and whose 20% estimate
  floats in overlapping frames.

    .venv/bin/python scripts/make_teaser.py

Writes paper/figures/fig_teaser_stick.pdf (the stick-figure version; Figure 1 now uses make_teaser_smpl.py).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "phase2", "50steps_seed0")
OUT = os.path.join(ROOT, "paper", "figures")
G = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), G)
center_of_mass, FLOAT_G_MAX = G["center_of_mass"], G["FLOAT_G_MAX"]
STEPS = [(10, "estimate at 20% of denoising"), (20, "estimate at 40%"), (None, "final motion")]
CHAIN_L = [[0, 1, 4, 7, 10], [9, 13, 16, 18, 20]]
CHAIN_R = [[0, 2, 5, 8, 11], [9, 14, 17, 19, 21]]
SPINE = [[0, 3, 6, 9, 12, 15]]
INK, LEFT, RIGHT, FLOAT_C, FLOOR = "#2b2b2b", "#0072B2", "#56B4E9", "#D55E00", "#bdbdbd"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})

meta = json.load(open(os.path.join(RUN, "meta.json")))
prompts, pid = meta["all_prompts"], np.array(meta["prompt_id"])
files = sorted(glob.glob(os.path.join(RUN, "batch_*.npz")))
J = np.concatenate([np.load(p)["joints"] for p in files]).astype(np.float32)
X0 = {k: np.concatenate([np.load(p)[f"x0j_{k}"] for p in files]).astype(np.float32) for k in (10, 20)}
lab = np.load(os.path.join(RUN, "labels.npz"), allow_pickle=True)
final, x0lab = lab["final"].item(), lab["x0"].item()
is_jump = np.array([any(w in p for w in ["jump", "hop", "leap", "skip"]) for p in prompts])
curated = pid < 64


def stretches(g):
    out, t = [], 0
    while t < len(g):
        if np.isfinite(g[t]):
            s = t
            while t < len(g) and np.isfinite(g[t]) and g[t] == g[s]:
                t += 1
            out.append((s, t, g[s]))
        else:
            t += 1
    return out


# --- pick the jump
best = None
for i in np.where(curated & is_jump)[0]:
    for s, e, g in stretches(final["flight_gravity"][i]):
        if 8.5 <= g <= 11:
            early = [gg for ss, ee, gg in stretches(x0lab[10]["flight_gravity"][i]) if ss < e and ee > s and gg >= 5]
            if early and (best is None or e - s > best[2] - best[1]):
                best = (i, s, e, g)
jump = best
# --- pick the floating walk: a typical case (lowest foot 5 to 15 cm while floating, floor contact elsewhere)
best = None
for i in np.where(curated & ~is_jump)[0]:
    if any(w in prompts[i] for w in ["beam", "toes", "tiptoe", "stairs", "climb"]):   # raised feet could be standing on something
        continue
    fl = np.isfinite(final["flight_gravity"][i]) & (final["flight_gravity"][i] < FLOAT_G_MAX)
    fe = np.isfinite(x0lab[10]["flight_gravity"][i]) & (x0lab[10]["flight_gravity"][i] < FLOAT_G_MAX)
    if fl.sum() < 8 or (fl & fe).sum() == 0:
        continue
    lowest = J[i][:, [7, 10, 8, 11], 1].min(1)
    if not (0.05 <= np.median(lowest[fl]) <= 0.15 and lowest[~fl].min() < 0.03):
        continue
    score = (fl & fe).sum() + (100 if "walk" in prompts[i] else 0)
    if best is None or score > best[1]:
        best = (i, score)
floatw = best[0]
print("jump:", prompts[jump[0]], jump[1:], "| floating walk:", prompts[floatw])


def side_coords(j_all):
    # Project onto the main horizontal direction of travel (fixed per motion, from the final motion).
    root = J_ref[:, 0, [0, 2]]
    d = root[-1] - root[0]
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.3 else np.array([1.0, 0.0])
    u = j_all[..., 0] * d[0] + j_all[..., 2] * d[1]
    return u - J_ref[0, 0, 0] * d[0] - J_ref[0, 0, 2] * d[1], j_all[..., 1]


def draw_pose(ax, u, y, alpha, color_override=None, lw=1.6):
    for chains, col in [(SPINE, INK), (CHAIN_L, LEFT), (CHAIN_R, RIGHT)]:
        for ch in chains:
            ax.plot(u[ch], y[ch], color=color_override or col, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(u[15], y[15] + 0.06, "o", ms=3.2, color=color_override or INK, alpha=alpha)


fig, axes = plt.subplots(3, 2, figsize=(7.0, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
for col, (kind, idx) in enumerate([("jump", jump[0]), ("float", floatw)]):
    J_ref = J[idx]
    for row, (k, title) in enumerate(STEPS):
        ax = axes[row, col]
        jj = J[idx] if k is None else X0[k][idx]
        labs = final if k is None else x0lab[k]
        g = labs["flight_gravity"][idx]
        u, y = side_coords(jj)
        c = center_of_mass(jj)
        cu, cy = side_coords(c[:, None, :])
        cu, cy = cu[:, 0], cy[:, 0]
        if kind == "jump":
            s, e = max(jump[1] - 6, 0), min(jump[2] + 6, len(jj) - 1)
            frames = list(range(s, e + 1, 3))
            for n, t in enumerate(frames):
                draw_pose(ax, u[t], y[t], alpha=0.15 + 0.75 * n / max(len(frames) - 1, 1), lw=1.2)
            seg = np.arange(s, e + 1)
            pts = np.stack([cu[seg], cy[seg]], 1)
            lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), colors=[FLOAT_C if np.isfinite(g[t]) else "#777777" for t in seg[:-1]], lw=1.8)
            ax.add_collection(lc)
            fits = [gg for ss, ee, gg in stretches(g) if ss <= jump[2] and ee >= jump[1]]
            txt = f"fitted fall rate {fits[0]:.1f} m/s$^2$" if fits else "no airborne stretch"
            ax.text(0.5, 1.0, txt, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=FLOAT_C if fits else INK)
            ax.set_xlim(u[frames].min() - 0.3, u[frames].max() + 0.3)
        else:
            fl = np.isfinite(g) & (g < FLOAT_G_MAX)
            fin_fl = np.isfinite(final["flight_gravity"][idx]) & (final["flight_gravity"][idx] < FLOAT_G_MAX)
            fi = np.where(fin_fl)[0]
            lo, hi = max(fi.min() - 8, 0), min(fi.max() + 8, len(jj) - 1)
            frames = list(np.linspace(lo, hi, 9).round().astype(int))
            for n, t in enumerate(frames):
                off = n * 0.55 - u[t, 0]
                draw_pose(ax, u[t] + off, y[t], alpha=0.95, color_override=FLOAT_C if fl[t] else None, lw=1.4)
                lowest = jj[t, [7, 10, 8, 11], 1].min()
                ax.text(n * 0.55, -0.16, f"{100 * lowest:.0f} cm", ha="center", fontsize=5.5, color=FLOAT_C if fl[t] else "#666666")
            ax.set_xlim(-0.45, (len(frames) - 1) * 0.55 + 0.45)
            share = fl[frames].mean()
            ax.text(0.5, 1.0, f"floating in {fl.sum()} of {len(fl)} frames (orange)", transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.5, color=FLOAT_C if fl.any() else INK)
        ax.axhline(0, color=FLOOR, lw=1.2, zorder=0)
        ax.fill_between([-10, 10], -0.3, 0, color="#f0f0f0", zorder=-1)
        ax.set_ylim(-0.24, float(y.max()) + 0.45)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if col == 0:
            ax.text(-0.02, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=6.5, rotation=90)
    axes[0, col].set_title(("(a) " if col == 0 else "(b) ") + f"“{prompts[idx].rstrip('.')}”", fontsize=7.5, loc="left")
plt.tight_layout(w_pad=1.0, h_pad=0.2)
os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "fig_teaser_stick.pdf"))
plt.savefig(os.path.join(OUT, "fig_teaser_stick_preview.png"), dpi=160)
print("wrote fig_teaser_stick.pdf")
