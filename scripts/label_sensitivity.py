"""Sensitivity of the results to the label thresholds (one threshold changed at a time).

Settings: the ankle and toe contact heights (12 / 6 cm) scaled by 0.75 and 1.25; the support rule
(no knee, wrist or pelvis within 15 cm of the floor) at 10 and 20 cm; the floating cut-off
(2 m/s^2) at 1 and 3 m/s^2. Labels are recomputed from the cached joints with the project's own
label code, only the constants changed.

    .venv/bin/python scripts/label_sensitivity.py physics     # (A) physics numbers, (C) early detector
    .venv/bin/python scripts/label_sensitivity.py controls    # (B) the fair comparison at 20%, seed 0

Writes results/sensitivity/{physics.csv, detector.csv} and, for (B), results/sensitivity/<setting>/controls/.
"""
import glob, json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "sensitivity")
os.makedirs(OUT, exist_ok=True)
SETTINGS = {
    "default": {},
    "contact x0.75": {"FOOT_HEIGHT_MAX": np.array([0.09, 0.045, 0.09, 0.045])},
    "contact x1.25": {"FOOT_HEIGHT_MAX": np.array([0.15, 0.075, 0.15, 0.075])},
    "support 10 cm": {"SUPPORT_HEIGHT_MIN": 0.10},
    "support 20 cm": {"SUPPORT_HEIGHT_MIN": 0.20},
    "floating 1 m/s2": {"FLOAT_G_MAX": 1.0},
    "floating 3 m/s2": {"FLOAT_G_MAX": 3.0},
}
EARLY = {"50steps": 10, "1000steps": 200}


def label_namespace(overrides):
    G = {"np": np}
    exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), G)
    G.update(overrides)                       # the label functions read these module-level constants
    return G


def is_jump(p):
    return any(w in p for w in ["jump", "hop", "leap", "skip"])


def runs():
    # Only complete runs (all batches written), so a seed that is still sampling is skipped.
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "phase2", "*_seed*"))):
        meta = json.load(open(os.path.join(d, "meta.json")))
        if len(glob.glob(os.path.join(d, "batch_*.npz"))) * meta["batch"] >= len(meta["all_prompts"]):
            out.append(d)
    return out


def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    ck = os.path.basename(d).split("_seed")[0]
    k = EARLY[ck]
    J = np.concatenate([np.load(p)["joints"] for p in files]).astype(np.float32)
    X = np.concatenate([np.load(p)[f"x0j_{k}"] for p in files]).astype(np.float32)
    return meta, ck, k, J, X


def physics():
    rows, det = [], []
    for d in runs():
        meta, ck, k, J, X = load_run(d)
        prompts, pid = meta["all_prompts"], np.array(meta["prompt_id"])
        test, jump = pid >= 64, np.array([is_jump(p) for p in prompts]) & (pid < 64)
        for name, ov in SETTINGS.items():
            G = label_namespace(ov)
            fin, est = G["labels_for"](J), G["labels_for"](X)
            fg = G["FLOAT_G_MAX"]
            g = fin["flight_gravity"]
            floating = np.isfinite(g) & (g < fg)
            jump_g = np.where(np.isfinite(g) & (g >= fg), g, np.nan)
            ground = fin["in_flight"] == 0
            float5 = np.where(ground, fin["lowest_foot"] > 0.05, np.nan)
            rows.append(dict(setting=name, checkpoint=ck, seed=meta["seed"],
                             floating_test=100 * floating[test].mean(), floating_jump=100 * floating[jump].mean(),
                             jump_g_jump=np.nanmedian(jump_g[jump]), jump_g_test=np.nanmedian(jump_g[test]),
                             float5_test=100 * np.nanmean(float5[test]),
                             motions_floating=int(floating.any(1).sum()), motions_jump_stretch=int(np.isfinite(jump_g).any(1).sum())))
            fail = floating.any(1)
            ge = est["flight_gravity"]
            flag = (np.isfinite(ge) & (ge < fg)).any(1)
            det.append(dict(setting=name, checkpoint=ck, seed=meta["seed"], base_failure=fail.mean(), flagged=flag.mean(),
                            precision=fail[flag].mean() if flag.any() else np.nan, recall=flag[fail].mean() if fail.any() else np.nan))
        print(f"{os.path.basename(d)} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "physics.csv"), index=False)
    pd.DataFrame(det).to_csv(os.path.join(OUT, "detector.csv"), index=False)
    p = pd.DataFrame(rows).groupby(["checkpoint", "setting"]).agg(["min", "max"]).round(2)
    print(p.drop(columns="seed").to_string())
    print(pd.DataFrame(det).groupby(["checkpoint", "setting"])[["base_failure", "flagged", "precision", "recall"]].mean().round(3).to_string())


def controls():
    # Seed 0 of each checkpoint: write a labels.npz per setting next to symlinks of the cached batches,
    # then run the project's control script on it unchanged.
    layers = {("xcom_margin", "50steps"): 7, ("xcom_margin", "1000steps"): 7,
              ("flight_gravity", "50steps"): 5, ("flight_gravity", "1000steps"): 5}
    for d in runs():
        if not d.endswith("_seed0"):
            continue
        meta, ck, k, J, X = load_run(d)
        for name, ov in SETTINGS.items():
            if name == "default":
                continue
            slug = name.replace(" ", "_").replace("/", "")
            rd = os.path.join(ROOT, "data", "sensitivity", slug, os.path.basename(d))
            os.makedirs(rd, exist_ok=True)
            for f in glob.glob(os.path.join(d, "batch_*.npz")) + [os.path.join(d, "meta.json")]:
                dst = os.path.join(rd, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
            G = label_namespace(ov)
            np.savez(os.path.join(rd, "labels.npz"), final=np.array(G["labels_for"](J), dtype=object),
                     x0=np.array({k: G["labels_for"](X)}, dtype=object))
            for target in ["xcom_margin", "flight_gravity"]:
                if target == "xcom_margin" and "FOOT_HEIGHT_MAX" not in ov:
                    continue                  # dynamic balance only depends on the contact thresholds
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "phase2_controls.py"), "--tag", f"sensitivity/{slug}",
                       "--checkpoint", ck, "--steps", str(k), "--target", target, "--layer", str(layers[(target, ck)]), "--skip-curve"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                print(name, ck, target, "exit", r.returncode, (r.stderr.strip().splitlines() or [""])[-1][:200] if r.returncode else "", flush=True)


if __name__ == "__main__":
    {"physics": physics, "controls": controls}[sys.argv[1]]()
