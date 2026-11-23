"""Label explainer figure: how dynamic balance and the gravity test are computed, on real samples.

(a) Top view of a recorded walk: support regions (convex hull of contacting foot joints, padded
    4 cm), the ground projection of the center of mass, and the extrapolated center of mass (XCoM).
(b) The same walk over time: dynamic (XCoM) and static (CoM) balance margins.
(c) A jump and (d) a floating walk: center-of-mass height, airborne frames, and the parabola
    fitted over each airborne stretch, with its fall rate.

    .venv/bin/python scripts/make_label_figure.py

Writes paper/figures/fig_labels.pdf (and a PNG preview).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from example_picks import LABELS, FLOAT_G_MAX, load, pick, stretches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
FPS, FOOT, RADIUS = LABELS["FPS"], LABELS["FOOT_JOINTS"], LABELS["FOOT_RADIUS"]
center_of_mass, foot_contacts = LABELS["center_of_mass"], LABELS["foot_contacts"]
INSIDE, OUTSIDE, INK, LEFT, RIGHT, AIR = "#009E73", "#D55E00", "#2b2b2b", "#0072B2", "#56B4E9", "#F0E442"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6, "pdf.fonttype": 42})

prompts, pid, J, X0, final, x0lab = load()
jump, floatw, walk = pick(prompts, pid, J, final, x0lab)
print("walk:", prompts[walk], "| jump:", prompts[jump[0]], "| floating:", prompts[floatw])

fig = plt.figure(figsize=(7.0, 4.4))
rows = fig.add_gridspec(2, 1, hspace=0.5)
top = rows[0].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)
bot = rows[1].subgridspec(1, 2, wspace=0.22)

# ---------------------------------------------------------------- (a) top view of balance
j = J[walk].astype(float)
com = center_of_mass(j)
contact = foot_contacts(j)
v = np.gradient(com, 1.0 / FPS, axis=0)
omega0 = np.sqrt(9.81 / max(np.median(com[:, 1]), 0.3))
xcom = com[:, [0, 2]] + v[:, [0, 2]] / omega0
d = com[-1, [0, 2]] - com[0, [0, 2]]
d = d / np.linalg.norm(d)
R = np.array([[d[0], d[1]], [-d[1], d[0]]])          # rotate so walking goes to +x
o = com[0, [0, 2]]
rot = lambda p: (np.asarray(p) - o) @ R.T
margin = final["xcom_margin"][walk]
defined = np.where(np.isfinite(margin))[0]
t0 = int(defined[len(defined) // 3])
win = np.arange(t0, min(t0 + 30, len(j)))
ax = fig.add_subplot(top[0, 0])
for t in win[::4]:
    pts = j[t, FOOT][contact[t]][:, [0, 2]]
    if len(pts):
        poly = MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(RADIUS)
        xy = rot(np.array(poly.exterior.coords))
        ax.fill(xy[:, 0], xy[:, 1], color="#bdbdbd", alpha=0.35, lw=0)
        ax.plot(xy[:, 0], xy[:, 1], color="#9e9e9e", lw=0.6)
for side, idx, col in [("left foot", [0, 1], LEFT), ("right foot", [2, 3], RIGHT)]:
    pts = [rot(j[t, FOOT[k], [0, 2]]) for t in win for k in idx if contact[t, k]]
    if pts:
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=2.5, color=col, label=f"{side} in contact")
c_r, x_r = rot(com[win][:, [0, 2]]), rot(xcom[win])
ax.plot(c_r[:, 0], c_r[:, 1], "-", color=INK, lw=1.1, label="center of mass")
ax.plot(x_r[:, 0], x_r[:, 1], "--", color="#7f7f7f", lw=0.9, label="extrapolated center of mass")
for t in win[::4]:
    if np.isfinite(margin[t]):
        p = rot(xcom[t])
        ax.plot(p[0], p[1], "o", ms=3.5, color=INSIDE if margin[t] >= 0 else OUTSIDE, mec="white", mew=0.4)
ax.set_aspect("equal"); ax.set_xlabel("forward (m)"); ax.set_ylabel("sideways (m)")
ax.set_title(f"(a) Top view of a walk: “{prompts[walk].rstrip('.')}”", loc="left")
ax.plot([], [], "o", ms=3.5, color=INSIDE, label="XCoM inside support")
ax.plot([], [], "o", ms=3.5, color=OUTSIDE, label="XCoM outside support")
ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=5.2, columnspacing=1.0, handlelength=1.4, borderaxespad=0.2)
lo, hi = c_r[:, 1].min(), c_r[:, 1].max()
ax.set_ylim(lo - 0.62, hi + 0.25)
ax.grid(alpha=0.2)

# ---------------------------------------------------------------- (b) margins over time
ax = fig.add_subplot(top[0, 1])
tt = win / FPS
dm, sm = final["xcom_margin"][walk][win], final["balance_margin"][walk][win]
ax.axhline(0, color="#9e9e9e", lw=0.8)
ax.fill_between(tt, 0, np.where(dm < 0, dm, 0) * 100, color=OUTSIDE, alpha=0.25, lw=0, label="XCoM outside support")
ax.plot(tt, sm * 100, color=INK, lw=1.0, label="static margin (center of mass)")
ax.plot(tt, dm * 100, color=OUTSIDE, lw=1.2, label="dynamic margin (XCoM)")
ax.set_xlabel("time (s)"); ax.set_ylabel("margin (cm), positive inside")
ax.set_title("(b) Balance margins over the same window", loc="left")
ax.legend(frameon=False, loc="lower left"); ax.grid(alpha=0.2)


# ---------------------------------------------------------------- (c), (d) gravity fits
def gravity_panel(ax, i, title, window):
    jj = J[i].astype(float)
    c = center_of_mass(jj)
    g = final["flight_gravity"][i]
    t = np.arange(len(jj)) / FPS
    s, e = window
    ax.plot(t[s:e], c[s:e, 1], color=INK, lw=1.2, label="center-of-mass height")
    ax.plot(t[s:e], jj[s:e][:, FOOT, 1].min(1), color="#7f7f7f", lw=0.8, label="lowest foot")
    first = True
    for a, b, gg in stretches(g):
        if b <= s or a >= e:
            continue
        ax.axvspan(t[a] - 0.5 / FPS, t[b - 1] + 0.5 / FPS, color=AIR if gg >= FLOAT_G_MAX else OUTSIDE, alpha=0.25, lw=0,
                   label=("airborne, falling" if gg >= FLOAT_G_MAX else "airborne, floating") if first else None)
        coef = np.polyfit(np.arange(b - a) / FPS, c[a:b, 1], 2)
        tf = np.linspace(0, (b - a - 1) / FPS, 50)
        ax.plot(t[a] + tf, np.polyval(coef, tf), ":", color=OUTSIDE if gg < FLOAT_G_MAX else "#B8860B", lw=1.6,
                label="fitted parabola" if first else None)
        ax.text(t[a] + tf[-1] / 2, np.polyval(coef, tf).max() + 0.05, f"$\\hat g$ = {gg:.1f} m/s$^2$", ha="center", va="bottom", fontsize=6.3)
        first = False
    ax.set_xlabel("time (s)"); ax.set_ylabel("height (m)")
    ax.set_ylim(-0.05, float(c[s:e, 1].max()) + 0.3)
    ax.set_title(title, loc="left"); ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="center right", fontsize=5.5)


i, s, e, gj = jump
gravity_panel(fig.add_subplot(bot[0, 0]), i, f"(c) Gravity test, jump: “{prompts[i].rstrip('.')}”", (max(s - 15, 0), min(e + 15, 120)))
fl = np.isfinite(final["flight_gravity"][floatw]) & (final["flight_gravity"][floatw] < FLOAT_G_MAX)
fi = np.where(fl)[0]
gravity_panel(fig.add_subplot(bot[0, 1]), floatw, f"(d) Floating walk: “{prompts[floatw].rstrip('.').replace('a person ', '')}”",
              (max(fi.min() - 20, 0), min(fi.max() + 20, 120)))
os.makedirs(OUT, exist_ok=True)
plt.savefig(os.path.join(OUT, "fig_labels.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig_labels_preview.png"), dpi=160, bbox_inches="tight")
print("wrote fig_labels.pdf")
