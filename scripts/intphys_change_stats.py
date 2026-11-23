"""Training-free frame-change statistics of the IntPhys clean VAE latents (no probes).

Checks that the change-feature readout is not explained by a global property of how impossible
videos are spliced together:
- ratio of the mean absolute frame-to-frame change (over channels and space) in impossible versus
  possible videos, at each of the 24 latent steps;
- "largest change" scores, pairwise within scenario (share of possible/impossible pairs in which the
  impossible video scores higher; ties 0.5; chance 50%):
    global: the largest mean change over steps;
    local:  the largest change of any 4x4-grid cell over steps (the readout's features, untrained).

    .venv/bin/python scripts/intphys_change_stats.py

Writes results/intphys_vae/change_stats.txt.
"""
import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "data", "intphys_vae", "latents.npz"))
z = d["z"].astype(np.float32)                                    # [N, 16, 25, 32, 32]
label, block, scen = d["label"], d["block"], d["scenario"]       # label 1 = possible
N, C, T = z.shape[:3]
glob_change = np.abs(np.diff(z, axis=2)).mean(axis=(1, 3, 4))    # [N, 24]
grid = torch.nn.functional.adaptive_avg_pool3d(torch.from_numpy(z), (T, 4, 4)).numpy()
cell_change = np.abs(np.diff(grid, axis=2)).mean(axis=1)         # [N, 24, 4, 4], mean over channels


def pairwise_impossible_higher(score):
    hits, n = 0.0, 0
    for s in np.unique(scen):
        m = scen == s
        imp, pos = score[m][label[m] == 0], score[m][label[m] == 1]
        for a in imp:
            for b in pos:
                hits += 1.0 if a > b else (0.5 if a == b else 0.0)
                n += 1
    return hits / n


ratio = glob_change[label == 0].mean(0) / glob_change[label == 1].mean(0)
lines = [f"IntPhys dev: {N} videos, {len(np.unique(scen))} scenarios",
         f"Mean |frame change|, impossible / possible, per latent step: min {ratio.min():.3f}, max {ratio.max():.3f}, "
         f"median {np.median(ratio):.3f}",
         f"Largest global change, pairwise (impossible higher): {100 * pairwise_impossible_higher(glob_change.max(1)):.1f}%",
         f"Largest 4x4-cell change, pairwise (impossible higher): {100 * pairwise_impossible_higher(cell_change.reshape(N, -1).max(1)):.1f}%"]
for b in ["O1", "O2", "O3"]:
    m = block == b
    r = glob_change[m & (label == 0)].mean(0) / glob_change[m & (label == 1)].mean(0)
    lines.append(f"  block {b}: change ratio {r.min():.3f} to {r.max():.3f}")
text = "\n".join(lines) + "\n"
open(os.path.join(ROOT, "results", "intphys_vae", "change_stats.txt"), "w").write(text)
print(text)
