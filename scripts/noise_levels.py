"""Noise level at each recorded step: alpha-bar and SNR = alpha-bar / (1 - alpha-bar), from each model's schedule
(MDM: cosine, Nichol and Dhariwal 2021, as in its args.json; MotionDiffuse: linear betas 1e-4 to 0.02, as in
scripts/run_motiondiffuse.py). Step k done means timestep t = T - 1 - k. Writes paper/tables/tab_snr.tex."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cosine(T, s=0.008):
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    return np.cumprod(1 - np.array([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)]))


def linear(T):
    sc = 1000 / T
    return np.cumprod(1 - np.linspace(sc * 1e-4, sc * 0.02, T))


MODELS = [("MDM, 50-step", cosine(50), [10, 20, 29, 39]), ("MDM, 1000-step", cosine(1000), [200, 400, 599, 799]),
          ("MotionDiffuse", linear(1000), [200, 400, 599, 799])]
assert abs(cosine(1000)[-1 - 200] - 0.094) < 0.001 and abs(linear(1000)[-1 - 200] - 0.0015) < 0.0002   # values quoted in the paper
rows = []
for name, ab, steps in MODELS:
    cells = []
    for k in steps:
        a = ab[len(ab) - 1 - k]
        cells.append(f"{a:.3f} & {a / (1 - a):.3f}")
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
with open(os.path.join(ROOT, "paper", "tables", "tab_snr.tex"), "w") as f:
    f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n& \\multicolumn{2}{c}{20\\%} & \\multicolumn{2}{c}{40\\%} & \\multicolumn{2}{c}{60\\%}"
            " & \\multicolumn{2}{c}{80\\%} \\\\\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n"
            "Model" + " & $\\bar\\alpha$ & SNR" * 4 + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("table tab_snr.tex"); print("\n".join(rows))
