"""Inspect airborne stretches whose fitted gravity is near zero ("hovering").

Written for the original airborne rule (feet only), to explain why that rule was changed; the
label now also requires no knee, wrist or pelvis within 15 cm of the floor.

The flight-gravity label fits a parabola to the centre-of-mass height over every stretch of at
least 0.2 s in which all four foot joints are above their height thresholds. Jump stretches fit
about 9.8 m/s^2; many stretches in non-jump motions fit about 0. This script asks what those
near-zero stretches are, using only the final joints cached by phase 2:

- support by another body part: a knee, wrist or the pelvis within 15 cm of the floor
  (kneeling, crawling, sitting), so "no foot down" is not "unsupported";
- elevated but still feet: both feet nearly stationary horizontally while above the threshold
  (standing on something, or the whole body offset from the floor);
- a motion-wide floor offset: the motion's lowest foot over all frames is itself above 5 cm;
- otherwise unexplained: unsupported, not falling.

    .venv/bin/python scripts/inspect_hovering.py [--run 50steps_seed0] [--gifs 6]

Writes results/phase2/hovering/<run>_stretches.csv, a summary, and a few stick-figure GIFs.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="50steps_seed0")
ap.add_argument("--gifs", type=int, default=6)
CLI = ap.parse_args()
D = os.path.join(ROOT, "data", "phase2", CLI.run)
OUT = os.path.join(ROOT, "results", "phase2", "hovering")
os.makedirs(OUT, exist_ok=True)

FPS = 20.0
FOOT = [7, 10, 8, 11]                      # L ankle, L toe, R ankle, R toe (same as the labels)
FOOT_MAX = np.array([0.12, 0.06, 0.12, 0.06])
OTHER = {"pelvis": 0, "L knee": 4, "R knee": 5, "L wrist": 20, "R wrist": 21}
SEGMENTS = [(12, 15, 0.081), (0, 3, 0.142), (3, 9, 0.355), (16, 18, 0.028), (17, 19, 0.028),
            (18, 20, 0.016), (19, 21, 0.016), (20, 20, 0.006), (21, 21, 0.006), (1, 4, 0.100),
            (2, 5, 0.100), (4, 7, 0.0465), (5, 8, 0.0465), (7, 10, 0.0145), (8, 11, 0.0145)]


def com(j):
    return sum(w * 0.5 * (j[:, a] + j[:, b]) for a, b, w in SEGMENTS)


def is_jump_prompt(p):
    return any(w in p for w in ["jump", "hop", "leap", "skip"])


meta = json.load(open(os.path.join(D, "meta.json")))
prompts = meta["all_prompts"]
J = np.concatenate([np.load(p)["joints"] for p in sorted(glob.glob(os.path.join(D, "batch_*.npz")))])
rows = []
for i, j in enumerate(J):
    feet = j[:, FOOT]
    air = (feet[..., 1] > FOOT_MAX).all(1)
    c = com(j)
    t = 0
    while t < len(j):
        if not air[t]:
            t += 1
            continue
        s = t
        while t < len(j) and air[t]:
            t += 1
        if t - s < 4:
            continue
        g = -2 * np.polyfit(np.arange(t - s) / FPS, c[s:t, 1], 2)[0]
        seg = slice(s, t)
        other_min = {k: float(j[seg, v, 1].min()) for k, v in OTHER.items()}
        hfoot = np.linalg.norm(np.diff(feet[seg][..., [0, 2]], axis=0), axis=-1) * FPS   # m/s per foot joint
        rows.append(dict(
            motion=i, prompt=prompts[i], jump_prompt=is_jump_prompt(prompts[i]),
            test_caption=meta["prompt_id"][i] >= 64, start=s, frames=t - s, gravity=g,
            lowest_foot_mean=float(feet[seg][..., 1].min(1).mean()),
            motion_lowest_foot=float(feet[..., 1].min()),
            com_height_range=float(np.ptp(c[seg, 1])),
            feet_horizontal_speed=float(np.median(hfoot)),
            lowest_other_part=min(other_min, key=other_min.get), lowest_other_height=min(other_min.values())))

df = pd.DataFrame(rows)
df["kind"] = np.where(df.gravity > 5, "ballistic (g > 5)", np.where(df.gravity.abs() < 2, "hovering (|g| < 2)", "other"))


def explain(r):
    if r.lowest_other_height < 0.15:
        return f"supported by {r.lowest_other_part}"
    if r.motion_lowest_foot > 0.05:
        return "whole motion above floor"
    if r.feet_horizontal_speed < 0.1:
        return "feet elevated but still"
    return "unexplained"


df["explanation"] = df.apply(explain, axis=1)
df.to_csv(os.path.join(OUT, f"{CLI.run}_stretches.csv"), index=False)

lines = [f"run {CLI.run}: {len(J)} motions, {len(df)} airborne stretches of >= 0.2 s", ""]
lines.append(df.groupby(["kind", "jump_prompt"]).size().rename("stretches").to_string())
hov = df[df.kind == "hovering (|g| < 2)"]
lines += ["", "hovering stretches by explanation:", hov.explanation.value_counts().to_string()]
lines += ["", "hovering stretches, medians:",
          hov[["frames", "lowest_foot_mean", "motion_lowest_foot", "com_height_range", "feet_horizontal_speed",
               "lowest_other_height"]].median().round(3).to_string()]
lines += ["", "most common prompts with hovering stretches:", hov.prompt.value_counts().head(15).to_string()]
summary = "\n".join(lines)
open(os.path.join(OUT, f"{CLI.run}_summary.txt"), "w").write(summary + "\n")
print(summary)

# A few stick figures: the longest hovering stretches in each explanation group.
if CLI.gifs:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    picks = hov.sort_values("frames", ascending=False).groupby("explanation").head(2).head(CLI.gifs)
    for _, r in picks.iterrows():
        j = J[r.motion]
        air = (j[:, FOOT, 1] > FOOT_MAX).all(1)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        ax.set_xlim(j[..., 0].min() - 0.3, j[..., 0].max() + 0.3); ax.set_ylim(-0.1, 2.0); ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(f"{r.prompt[:45]}\n{r.explanation}, g={r.gravity:.1f}", fontsize=7)
        lines_ = [ax.plot([], [], lw=2)[0] for _ in CHAIN]
        def step(t):
            for ln, ch in zip(lines_, CHAIN):
                ln.set_data(j[t, ch, 0], j[t, ch, 1]); ln.set_color("tab:red" if air[t] else "k")
            return lines_
        anim = animation.FuncAnimation(fig, step, frames=range(0, len(j), 2), interval=100)
        name = f"{CLI.run}_motion{r.motion}_{r.explanation.replace(' ', '_')}.gif"
        anim.save(os.path.join(OUT, name), writer="pillow")
        plt.close(fig)
        print("saved", name)
