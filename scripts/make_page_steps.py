"""Step-through stills for the project page: the jump of Figure 8 at the start, 20% and 40% of denoising and finished,
each drawn from just before take-off to just after landing, with the centre-of-mass path (red while airborne).

Reuses make_page_videos.jump() (the same sample and the same cached SMPL fits as the clips), so every pose and path
comes from the recorded joints. Writes transparent PNGs, one per stage, sharing one screen box.

    .venv/bin/python scripts/make_page_steps.py <out_dir>
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import body_images as bi
import make_page_videos as V

OUT_NAMES = ["jump_0", "jump_20", "jump_40", "jump_final"]
PPM = 220            # pixels per metre, as the storyboard stills
STEP, MARGIN = 2, 0  # every second frame, from take-off to landing


def main(out):
    prompt, panels, apex = V.jump()
    air = panels[-1]["air"]
    # the finished motion's airborne stretch around the apex
    s = apex
    while s > 0 and air[s - 1]:
        s -= 1
    e = apex
    while e + 1 < len(air) and air[e + 1]:
        e += 1
    land = min(e + 1, len(air) - 1)                        # first frame back on the ground
    frames = list(range(max(s - MARGIN, 0), min(land + MARGIN, len(air) - 1) + 1, STEP))
    frames = sorted(set(frames) | {apex, land})            # always show the top and the landing
    seg = np.arange(max(s - 6, 0), min(e + 6, len(air)))   # path a little beyond the drawn bodies
    sx, sy, _ = bi.project(np.concatenate([p["V"][frames] for p in panels]))
    box = (sx.min() - 0.2, sx.max() + 0.2, -0.25, sy.max() + 0.12)
    alpha = lambda n, N: 0.1 + 0.75 * (n / max(N - 1, 1)) ** 1.6
    body = bi.hex_rgb(V.P["body"])
    for p, name in zip(panels, OUT_NAMES):
        bodies = [(p["V"][f], body, alpha(n, len(frames)) if n < len(frames) - 1 else 0.92) for n, f in enumerate(frames)]
        path, path_air = p["com"][seg], p["air"][seg]

        def overlay(ax, path=path, path_air=path_air):
            x, y, _ = bi.project(path)
            ax.plot(x, y, color="#9aa6bd", lw=1.6, solid_capstyle="round", zorder=200)
            for t in range(1, len(x)):
                if path_air[t] and path_air[t - 1]:
                    ax.plot(x[t - 1:t + 1], y[t - 1:t + 1], color=V.P["mark"], lw=2.4, solid_capstyle="round", zorder=201)

        dst = os.path.join(out, f"{name}.png")
        bi.render(bodies, dst, box, px_per_m=PPM, floor_rgb=None, flat=True, overlay=overlay)
        print(f"{name}: frames {frames} (airborne {s} to {e}, apex {apex}, landing {land}), {Image.open(dst).size}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
