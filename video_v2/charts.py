"""Animated chart slides for the revised supplementary video (1920x1080, the presentation's dark stage).

Each draw_* function takes a progress t in [0, 1] (the build-up animation) and returns a PIL image. Every number comes
from the paper: the data behind Figures 5, 8 and 10 (paper/figures_web/data/*.js), Table 2 and Table 6. Scores are
shown on a 100-point scale (R² or AUC times 100), as the narration says them.

    .venv/bin/python video_v2/charts.py <out_dir>      # writes candidate stills of every slide
"""
import io, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "paper", "figures_web", "data")
FONTS = os.path.join(ROOT, "paper", "figures_web", "fonts")
for f in ("Literata-Regular-static.ttf", "Literata-SemiBold-static.ttf"):
    font_manager.fontManager.addfont(os.path.join(FONTS, f))
plt.rcParams.update({"font.family": "Literata", "font.size": 22, "axes.linewidth": 0})

W, H, DPI = 1920, 1080, 100
INK, MUTED, FAINT, TRACK = "#eef2f8", "#9aa7bf", "#5d6b88", "#33415e"
TEAL, BLUE, RED, GRAY = "#2ec4c0", "#7896e6", "#ff6b73", "#6f7c98"


def load(name):
    return json.loads(open(os.path.join(DATA, f"{name}.js")).read().split("=", 1)[1].rstrip().rstrip(";"))


# Table 2 of the paper (mean over seeds): best layer and the label computed from the estimate, at 20% of denoising.
NAIVE = [("Dynamic balance", "50-step MDM", 0.838, 0.783), ("Dynamic balance", "1000-step MDM", 0.697, 0.655),
         ("Static balance", "50-step MDM", 0.759, 0.769), ("Static balance", "1000-step MDM", 0.560, 0.546),
         ("In flight", "50-step MDM", 0.973, 0.876), ("In flight", "1000-step MDM", 0.926, 0.788),
         ("Flight gravity", "50-step MDM", 0.820, 0.707), ("Flight gravity", "1000-step MDM", 0.692, 0.691)]
# Table 6 of the paper: pairwise accuracy (%) of readouts of Wan 2.1's clean VAE latent on IntPhys.
VIDEO = [("pooled over time and space", 52.1, False), ("time kept", 50.4, False), ("time and 4×4 space", 54.4, False),
         ("with signed frame-to-frame changes", 52.2, False), ("|frame-to-frame change|, 4×4 space", 77.9, True)]
MODEL = {"50-step": "50-step MDM", "1000-step": "1000-step MDM", "MotionDiffuse": "MotionDiffuse"}


def pts(v):
    """Signed points with one decimal, and a plain 0.0 instead of -0.0 or +0.0."""
    s = f"{v:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def stage(title, sub=None):
    """A figure with the presentation's gradient background and a title block."""
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    g = np.linspace(0, 1, 256)[:, None]
    a0, a1 = np.array([0x1b, 0x27, 0x40]) / 255, np.array([0x0e, 0x15, 0x24]) / 255
    bg.imshow((a0 * (1 - g) + a1 * g)[:, None, :].repeat(2, 1), aspect="auto", extent=[0, 1, 0, 1])
    fig.text(72 / W, 1 - 70 / H, title, color=INK, fontsize=40, fontweight="semibold", va="top")
    if sub:
        fig.text(72 / W, 1 - 140 / H, sub, color=MUTED, fontsize=21, va="top")
    return fig


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def axes(fig, box):
    ax = fig.add_axes([box[0] / W, 1 - (box[1] + box[3]) / H, box[2] / W, box[3] / H])
    ax.set_facecolor("none")
    ax.tick_params(colors=MUTED, length=0, labelsize=19)
    return ax


def draw_readers(t=1.0):
    """Two readers: one sees the estimate, the other the estimate and a layer; both predict the finished motion's physics."""
    fig = stage("The catch, and a fair test", "the estimate is computed from the model's own layers")
    ax = axes(fig, (0, 0, W, H)); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    def box(x, y, w, h, color, head, body, a):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=22", fc="#18223a", ec=color, lw=3, alpha=a))
        ax.text(x + 36, y + 54, head, color=color, fontsize=30, fontweight="semibold", va="center", alpha=a)
        ax.text(x + 36, y + 112, body, color=INK, fontsize=22, va="center", alpha=a)
    a1, a2, a3 = (ease(t * 3 - k) for k in range(3))
    box(120, 330, 800, 170, TEAL, "Reader A", "sees only the clean-motion estimate", a1)
    box(120, 560, 800, 170, BLUE, "Reader B", "sees the estimate and one layer of the model", a2)
    box(1200, 430, 600, 200, RED, "The final motion", "balance · flight · how fast it falls", a3)
    for y0 in (415, 645):
        ax.annotate("", xy=(1185, 530), xytext=(940, y0), arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=2.5, alpha=a3))
    ax.text(W / 2, 810, "the gap between B and A is what the layer adds to the estimate",
            color=MUTED, fontsize=24, ha="center", alpha=a3)
    return to_image(fig)


def comparison_rows(mode):
    if mode == "usual":
        return [(tg, md, (lay - exp) * 100, None, None) for tg, md, lay, exp in NAIVE]
    want = {("Dynamic balance", ""), ("Static balance", ""), ("In flight", ""), ("Fall rate", ", jump prompts")}
    rows = []
    for r in load("fig_gain")["rows"]:
        base, _, suf = r["model"].partition(",")
        if (r["target"], ("," + suf) if suf else "") not in want:
            continue
        v, lo, hi = (x * 100 for x in r["gain"]["score"])
        rows.append((r["target"] + (" of jumps" if suf else ""), MODEL[base], v, lo, hi))
    return rows


def draw_comparison(mode="usual", t=1.0):
    """The usual comparison (best layer minus the estimate's computed label, Table 2) or the fair one (what adding the
    layer to a tuned readout of the estimate gains, Figure 5b), on one shared axis in points."""
    usual = mode == "usual"
    fig = stage("The usual comparison" if usual else "A fair comparison",
                "best layer minus the same label computed from the estimate, at 20% of denoising; on a 100-point scale" if usual
                else "what adding a layer to a tuned readout of the estimate gains, at 20% of denoising; 95% intervals")
    rows = comparison_rows(mode)
    groups = list(dict.fromkeys(r[0] for r in rows))
    n_rows = len(rows) + len(groups) * 0.6
    ax = axes(fig, (700, 200, 1060, 620))
    left = blended_transform_factory(fig.transFigure, ax.transData)   # x in figure fraction, y in rows
    ax.set_xlim(-2, 15.5); ax.set_ylim(n_rows, -0.4)
    for x in (0, 4, 8, 12):
        ax.axvline(x, color=TRACK if x else FAINT, lw=1.5 if x else 2, zorder=0)
    ax.set_xticks([0, 4, 8, 12]); ax.set_xticklabels(["0", "+4", "+8", "+12 points"])
    ax.set_yticks([])
    y, k = 0.0, 0
    for g in groups:
        ax.text(72 / W, y + 0.35, g, color=INK, fontsize=23, fontweight="semibold", ha="left", va="center", transform=left)
        for r in [q for q in rows if q[0] == g]:
            a = ease((t * (len(rows) + 3) - k) / 4)       # rows appear one after another; all shown at t = 1
            ax.text(-2.4, y + 0.35, r[1], color=MUTED, fontsize=19, ha="right", va="center", alpha=a)
            if usual:
                c = BLUE if r[2] >= 0 else TEAL
                ax.add_patch(Rectangle((min(0, r[2]), y + 0.08), abs(r[2]) * a, 0.54, color=c, zorder=2))
                ax.text(15.5, y + 0.35, pts(r[2]), color=INK, fontsize=19, ha="right", va="center", alpha=a)
            else:
                gains = r[3] > 0
                c = BLUE if gains else GRAY
                ax.plot([r[3], r[4]], [y + 0.35] * 2, color=c, lw=3.5, alpha=a, solid_capstyle="round", zorder=2)
                ax.plot([r[2]], [y + 0.35], "o", ms=11, color=c, mec="#16203a", mew=2, alpha=a, zorder=3)
                ax.text(15.5, y + 0.35, pts(r[2]), color=INK if gains else MUTED, fontsize=19, ha="right", va="center",
                        fontweight="semibold" if gains else "regular", alpha=a)
            y += 1; k += 1
        y += 0.6
    return to_image(fig)


def draw_fall_rate(t=1.0, model="50-step"):
    """Figure 8b: R² (times 100) for the finished jump's fall rate, from the estimate alone and with layer 5 added."""
    m = next(x for x in load("fig_steps_fall")["models"] if x["short"] == model)
    fig = stage("The one exception: how fast a jump falls", f"{MODEL[model]}, held-out jump prompts; score on a 100-point scale")
    ax = axes(fig, (200, 220, 1450, 560))
    S = np.array(m["steps"], float); est, both = np.array(m["estimate"]) * 100, np.array(m["both"]) * 100
    ax.set_xlim(15, 92); ax.set_ylim(20, 85)
    for yv in (20, 40, 60, 80):
        ax.axhline(yv, color=TRACK, lw=1.5, zorder=0)
    ax.set_yticks([20, 40, 60, 80]); ax.set_xticks(S); ax.set_xticklabels([f"{int(s)}%" for s in S])
    ax.set_xlabel("how far through denoising", color=MUTED, fontsize=20, labelpad=14)
    # draw along the steps as t grows
    reach = 20 + 60 * ease(t)
    xs = np.linspace(20, reach, 200)
    e_i, b_i = np.interp(xs, S, est), np.interp(xs, S, both)
    ax.fill_between(xs, e_i, b_i, color=BLUE, alpha=0.22, lw=0)
    ax.plot(xs, e_i, color=TEAL, lw=4.5, solid_capstyle="round")
    ax.plot(xs, b_i, color=BLUE, lw=4.5, solid_capstyle="round")
    for i, s in enumerate(S):
        if s <= reach + 1e-6:
            ax.plot([s], [est[i]], "o", ms=12, color=TEAL, mec="#16203a", mew=2)
            ax.plot([s], [both[i]], "o", ms=12, color=BLUE, mec="#16203a", mew=2)
            last = i == len(S) - 1
            ax.text(s - (1.2 if last else 0), both[i] + 3.2, f"+{m['gain'][i] * 100:.1f}", color=INK, fontsize=22,
                    ha="right" if last else "center", fontweight="semibold")
    if reach >= 80:
        ax.text(82, both[-1] + 1.0, "the estimate plus a layer", color=BLUE, fontsize=19, va="bottom")
        ax.text(82, est[-1] - 1.0, "the estimate alone", color=TEAL, fontsize=19, va="top")
    return to_image(fig)


def draw_early(t=1.0, model="50-step model"):
    """Figure 10: floating motions delivered against compute as the number of tries grows (final check kept)."""
    m = next(x for x in load("fig_early")["models"] if x["name"] == model)
    fig = stage("Early warning saves little", "50-step MDM: floating motions delivered against compute, as the number of tries grows")
    ax = axes(fig, (200, 220, 1450, 560))
    ax.set_xlim(0.98, 1.72); ax.set_ylim(0, 10)
    for yv in (0, 2, 4, 6, 8, 10):
        ax.axhline(yv, color=TRACK, lw=1.5, zorder=0)
    ax.set_yticks([0, 2, 4, 6, 8, 10]); ax.set_yticklabels([f"{v}%" for v in (0, 2, 4, 6, 8, 10)])
    ax.set_xticks([1.0, 1.2, 1.4, 1.6]); ax.set_xticklabels(["1.0×", "1.2×", "1.4×", "1.6×"])
    ax.set_xlabel("compute (1 = one full sample)", color=MUTED, fontsize=20, labelpad=14)
    for key, label, c, ls in (("filter", "finish, then filter", GRAY, "-"), ("rule", "abort at 20% if the estimate floats", TEAL, "-"),
                              ("oracle", "a perfect early detector", BLUE, (0, (4, 3)))):
        pts = np.array(m[key])
        n = max(2, int(round(1 + (len(pts) - 1) * ease(t))))
        ax.plot(pts[:n, 0], pts[:n, 1], color=c, lw=4.5, ls=ls, solid_capstyle="round")
        ax.plot(pts[n - 1, 0], pts[n - 1, 1], "o", ms=12, color=c, mec="#16203a", mew=2)
        ax.plot([], [], color=c, lw=4.5, ls=ls, label=label)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=21, handlelength=2.4, labelcolor=INK)
    return to_image(fig)


def draw_video(t=1.0):
    """Table 6: readouts of Wan 2.1's clean VAE latent on IntPhys, pairwise accuracy; chance is 50%."""
    fig = stage("A detour into video", "Wan 2.1's compressed video, IntPhys: which of two videos is physically possible?")
    ax = axes(fig, (720, 250, 1000, 520))
    ax.set_xlim(40, 85); ax.set_ylim(len(VIDEO) - 0.3, -0.7); ax.set_yticks([])
    ax.axvline(50, color=FAINT, lw=2, ls=(0, (4, 3)))
    ax.text(50, -0.75, "chance", color=MUTED, fontsize=18, ha="center", va="bottom")
    ax.set_xticks([40, 50, 60, 70, 80]); ax.set_xticklabels([f"{v}%" for v in (40, 50, 60, 70, 80)])
    for i, (label, v, hl) in enumerate(VIDEO):
        a = ease((t * (len(VIDEO) + 1) - i) / 2)
        c = BLUE if hl else GRAY
        ax.add_patch(Rectangle((40, i - 0.27), (v - 40) * a, 0.54, color=c))
        ax.text(39, i, ("linear after " if hl else "linear, ") + label, color=INK if hl else MUTED, fontsize=21, ha="right", va="center", alpha=a)
        ax.text(40 + (v - 40) * a + 0.6, i, f"{v:.1f}%", color=INK, fontsize=21, va="center",
                fontweight="semibold" if hl else "regular", alpha=a)
    return to_image(fig)


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
    stills = {"readers": draw_readers(), "usual": draw_comparison("usual"), "fair": draw_comparison("fair"),
              "fall_rate": draw_fall_rate(), "early": draw_early(), "video": draw_video()}
    for k, im in stills.items():
        im.save(os.path.join(out, f"{k}.png"))
    sheet = Image.new("RGB", (W, 3 * H // 2), "black")      # 2 x 3 grid at half size
    for i, im in enumerate(stills.values()):
        sheet.paste(im.resize((W // 2, H // 2)), ((i % 2) * W // 2, (i // 2) * H // 2))
    sheet.save(os.path.join(out, "sheet.jpg"), quality=85)
    print("wrote", ", ".join(stills), "and sheet.jpg to", out)
