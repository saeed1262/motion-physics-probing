"""Figure 1 with SMPL bodies: what the model's own clean-motion estimate shows, early in denoising.

Same layout and examples as make_teaser.py (chosen by example_picks.py), drawn with SMPL meshes
fitted to the joints (fit_smpl_teaser.py) instead of stick figures. Display only: every number in
the figure (fall rates, floating frames, lowest-foot heights) is computed from the joints.

Rendering is a small orthographic painter's-algorithm renderer: a camera slightly above the floor,
Lambert shading, back faces culled, and a vertical shadow of each body on the floor, so the gap
between a floating body and its shadow shows the height directly.

    .venv/bin/python scripts/make_teaser_smpl.py

Writes paper/figures/fig_teaser.pdf (and a PNG preview).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from example_picks import LABELS, FLOAT_G_MAX, load, pick, stretches  # noqa: E402
from paper_style import COL, INK, MUTED  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
FACES = fits["faces"]
center_of_mass = LABELS["center_of_mass"]
prompts, pid, J, X0, final, x0lab = load()
VERSIONS = [("est20", 10, "estimate at 20% of denoising"), ("est40", 20, "estimate at 40%"), ("final", None, "final motion")]
ELEV = np.deg2rad(12)
LIGHT = np.array([-0.45, 0.75, -0.5]); LIGHT /= np.linalg.norm(LIGHT)
TO_CAM = np.array([0.0, np.sin(ELEV), -np.cos(ELEV)])        # (forward, up, depth): camera sits at depth -inf
BODY, FLOAT_BODY, SHADOW, FLOOR = "#7FA7CC", "#E8875A", "#b9b9b9", "#f4f4f4"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})


def frame_axes(idx):
    # forward = direction of travel of the final motion's root (x axis if it barely moves); depth = sideways
    root = J[idx][:, 0, [0, 2]]
    d = root[-1] - root[0]
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.3 else np.array([1.0, 0.0])
    o = J[idx][0, 0, [0, 2]]
    return lambda p: np.stack([(p[..., 0] - o[0]) * d[0] + (p[..., 2] - o[1]) * d[1], p[..., 1],
                               -(p[..., 0] - o[0]) * d[1] + (p[..., 2] - o[1]) * d[0]], -1)


def project(p):                                                # (forward, up, depth) -> screen x, y and depth
    return p[..., 0], p[..., 1] * np.cos(ELEV) + p[..., 2] * np.sin(ELEV), p[..., 2] * np.cos(ELEV) - p[..., 1] * np.sin(ELEV)


def draw_body(ax, V, color, z, shadow=True):
    tri = V[FACES]                                             # [F, 3, 3]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    front = n @ TO_CAM > 0
    sx, sy, depth = project(tri)
    if shadow:
        sh = tri.copy(); sh[..., 1] = 0.0
        hx, hy, _ = project(sh)
        ax.add_collection(PolyCollection(np.stack([hx, hy], -1), facecolors=SHADOW, edgecolors="none", zorder=z - 0.5,
                                         rasterized=True))
    order = np.argsort(-depth[front].mean(1))                  # far to near
    shade = 0.42 + 0.58 * np.clip(n[front] @ LIGHT, 0, 1)
    cols = np.clip(np.array(to_rgb(color))[None] * shade[:, None] + 0.08 * (1 - shade[:, None]), 0, 1)[order]
    polys = np.stack([sx[front], sy[front]], -1)[order]
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.15, zorder=z, rasterized=True))


def floor(ax, x0, x1):
    d0, d1 = -0.45, 0.6
    ys = [d0 * np.sin(ELEV), d1 * np.sin(ELEV)]
    ax.fill_between([x0, x1], ys[0], ys[1], color=FLOOR, zorder=0, lw=0)
    ax.plot([x0, x1], [ys[0], ys[0]], color="#c8c8c8", lw=0.6, zorder=0.1)


def tint(c, a):                                                # blend colour c toward white by 1 - a
    c = np.array(to_rgb(c))
    return tuple(1 - a * (1 - c))


i_jump = int(fits["jump_motion"]); i_float = int(fits["float_motion"])
fig, axes = plt.subplots(3, 2, figsize=(7.0, 4.7), gridspec_kw={"width_ratios": [1, 1.2]})
for col, (kind, idx) in enumerate([("jump", i_jump), ("float", i_float)]):
    to_local = frame_axes(idx)
    frames = list(fits[f"{kind}_frames"])
    for row, (ver, k, label) in enumerate(VERSIONS):
        ax = axes[row, col]
        verts = to_local(fits[f"{kind}_{ver}_verts"])
        jj = J[idx] if k is None else X0[k][idx]
        labs = final if k is None else x0lab[k]
        g = labs["flight_gravity"][idx]
        if kind == "jump":
            xs_all = verts[..., 0]
            floor(ax, xs_all.min() - 0.3, xs_all.max() + 0.3)
            for n, (t, V) in enumerate(zip(frames, verts)):
                a = 0.3 + 0.7 * n / max(len(frames) - 1, 1)
                draw_body(ax, V, tint(BODY, a), z=10 + n)
            c = to_local(center_of_mass(jj)[None])[0]
            seg = np.arange(frames[0], frames[-1] + 1)
            cx, cy, _ = project(c[seg])
            air = np.isfinite(g[seg])
            ax.plot(cx, cy, color=MUTED, lw=1.0, zorder=60, alpha=0.9)
            ax.plot(np.where(air, cx, np.nan), np.where(air, cy, np.nan), color=COL["est mlp"], lw=2.0, zorder=61)
            fits_g = [gg for ss, ee, gg in stretches(g) if ss <= frames[-1] and ee >= frames[0] and gg >= FLOAT_G_MAX]
            txt = f"fitted fall rate {fits_g[0]:.1f} m/s$^2$" if fits_g else "no falling airborne stretch"
            ax.text(0.5, 1.0, txt, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=COL["est mlp"])
            ax.set_xlim(xs_all.min() - 0.25, xs_all.max() + 0.25)
            top = verts[..., 1].max()
        else:
            fl = np.isfinite(g) & (g < FLOAT_G_MAX)
            spacing = 0.62
            floor(ax, -0.5, (len(frames) - 1) * spacing + 0.5)
            for n, (t, V) in enumerate(zip(frames, verts)):
                V = V.copy(); V[:, 0] += n * spacing - V[:, 0].mean()
                draw_body(ax, V, FLOAT_BODY if fl[t] else BODY, z=10 + n)
                lowest = jj[t, [7, 10, 8, 11], 1].min()
                ax.text(n * spacing, -0.2, f"{100 * lowest:.0f} cm", ha="center", fontsize=5.5,
                        color=COL["est mlp"] if fl[t] else MUTED)
            ax.text(0.5, 1.0, f"floating in {fl.sum()} of {len(fl)} frames (orange)", transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.5, color=COL["est mlp"] if fl.any() else INK)
            ax.set_xlim(-0.45, (len(frames) - 1) * spacing + 0.45)
            top = verts[..., 1].max()
        ax.set_ylim(-0.27, top + 0.42)
        ax.set_aspect("equal"); ax.axis("off")
        if col == 0:
            ax.text(-0.03, 0.45, label, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=6.5, color=INK)
    head = prompts[idx].rstrip(".")
    axes[0, col].set_title(("(a) " if col == 0 else "(b) ") + f"“{head}”", fontsize=7.5, loc="left",
                           color=INK, fontweight="bold")
fig.tight_layout(w_pad=1.2, h_pad=0.1)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_teaser.pdf"), dpi=300)
fig.savefig(os.path.join(OUT, "fig_teaser_preview.png"), dpi=170)
print("wrote fig_teaser.pdf")
