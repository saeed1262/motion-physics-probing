"""Shared loading and example selection for the motion figures (make_teaser.py, make_label_figure.py).

All examples come from the cached 50-step run, seed 0, and are chosen by fixed rules:
- jump: a curated jump prompt whose final motion has a ballistic stretch (8.5 to 11 m/s^2) that the
  20% estimate already shows falling (at least 5 m/s^2); the longest such stretch;
- floating walk: a non-jump curated prompt (no beam, toes, stairs or climbing, where raised feet
  could be standing on something) whose final motion floats for at least 8 frames with the lowest
  foot 5 to 15 cm up while floating and at the floor elsewhere, and whose 20% estimate floats in
  overlapping frames; walking prompts preferred;
- balanced walk: a "walks forward" sample with no floating and the most frames with foot support.
"""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "phase2", "50steps_seed0")
LABELS = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), LABELS)
FLOAT_G_MAX = LABELS["FLOAT_G_MAX"]


def load():
    meta = json.load(open(os.path.join(RUN, "meta.json")))
    files = sorted(glob.glob(os.path.join(RUN, "batch_*.npz")))
    J = np.concatenate([np.load(p)["joints"] for p in files]).astype(np.float32)
    X0 = {k: np.concatenate([np.load(p)[f"x0j_{k}"] for p in files]).astype(np.float32) for k in (10, 20)}
    lab = np.load(os.path.join(RUN, "labels.npz"), allow_pickle=True)
    return meta["all_prompts"], np.array(meta["prompt_id"]), J, X0, lab["final"].item(), lab["x0"].item()


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


def pick(prompts, pid, J, final, x0lab):
    is_jump = np.array([any(w in p for w in ["jump", "hop", "leap", "skip"]) for p in prompts])
    curated = pid < 64
    jump = None
    for i in np.where(curated & is_jump)[0]:
        for s, e, g in stretches(final["flight_gravity"][i]):
            if 8.5 <= g <= 11:
                early = [gg for ss, ee, gg in stretches(x0lab[10]["flight_gravity"][i]) if ss < e and ee > s and gg >= 5]
                if early and (jump is None or e - s > jump[2] - jump[1]):
                    jump = (i, s, e, g)
    floatw = None
    for i in np.where(curated & ~is_jump)[0]:
        if any(w in prompts[i] for w in ["beam", "toes", "tiptoe", "stairs", "climb"]):
            continue
        fl = np.isfinite(final["flight_gravity"][i]) & (final["flight_gravity"][i] < FLOAT_G_MAX)
        fe = np.isfinite(x0lab[10]["flight_gravity"][i]) & (x0lab[10]["flight_gravity"][i] < FLOAT_G_MAX)
        if fl.sum() < 8 or (fl & fe).sum() == 0:
            continue
        lowest = J[i][:, [7, 10, 8, 11], 1].min(1)
        if not (0.05 <= np.median(lowest[fl]) <= 0.15 and lowest[~fl].min() < 0.03):
            continue
        score = (fl & fe).sum() + (100 if "walk" in prompts[i] else 0)
        if floatw is None or score > floatw[1]:
            floatw = (i, score)
    walk = None
    for i in np.where(np.array([p == "a person walks forward." for p in prompts]))[0]:
        if np.isfinite(final["flight_gravity"][i]).any():
            continue
        n = int(np.isfinite(final["xcom_margin"][i]).sum())
        if walk is None or n > walk[1]:
            walk = (i, n)
    return jump, floatw[0], walk[0]
