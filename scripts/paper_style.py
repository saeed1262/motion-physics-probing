"""Shared look for the paper's figures: Okabe-Ito colours with fixed meanings, light axes.

Colour meanings (kept the same in every figure):
- blue: the model's internal layers (dark = linear probe, light = MLP);
- warm (amber to vermillion): readouts of the model's own clean-motion estimate
  (amber = explicit label, light amber = linear probe, vermillion = MLP);
- green: estimate and layer together;
- gray: the noisy input; black: a perfect detector; reddish purple: finish-then-filter.
In the motion figures, vermillion also marks floating or unsupported poses.
"""
import matplotlib.pyplot as plt

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
COL = {
    "layer lin": "#0072B2", "layer mlp": "#56B4E9",
    "explicit": "#E69F00", "est lin": "#F2C46D", "est mlp": "#D55E00",
    "both": "#009E73", "noisy": "#9e9e9e", "oracle": "#2b2b2b", "filter": "#CC79A7",
}


def setup():
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.solid_capstyle": "round", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def title(ax, text, letter=None):
    ax.set_title((f"({letter}) " if letter else "") + text, loc="left", color=INK, fontweight="semibold", pad=4)
