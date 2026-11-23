"""Do the physics labels behave on real motion? A check on AMASS BMLmovi (motion capture).

Real people do not float, and real flight falls at 9.81 m/s^2, so on motion capture the labels
should flag (almost) no floating, and the fitted gravity of airborne stretches shows how noisy the
gravity label is. The motion goes through the same steps as HumanML3D's own preprocessing
(github.com/EricGuo5513/HumanML3D, raw_pose_processing and motion_representation), so it matches
the data MDM was trained on:

1. SMPL+H forward kinematics (Extended SMPL+H model, the clip's gender, first 10 shape
   coefficients, as HumanML3D does); the first 22 joints; plus the global translation.
2. Downsample to 20 fps (every int(fps / 20)-th frame); z-up to y-up with HumanML3D's axis swap and its x
   negation (raw_pose_processing.ipynb), which together make a rotation (the swap alone mirrors the body).
3. Retarget to HumanML3D's uniform skeleton (MDM's own uniform_skeleton routine), with the target
   bone offsets taken from the first frame of a HumanML3D processed clip (tests/012314.npy; all of
   HumanML3D's processed clips share this skeleton). The retargeting scales the root trajectory by
   a leg-length ratio, which scales fitted gravity by the same factor; it is recorded per clip.
4. Floor at the clip's lowest joint.
5. The project's label code (tools/p2_labels_funcs.py), unchanged; and the same one-at-a-time
   threshold settings as scripts/label_sensitivity.py.

    .venv/bin/python -I scripts/amass_label_check.py [--limit 50]

Needs vendor/smplh/{male,female,neutral}/model.npz and vendor/amass/BMLmovi/ (downloaded by the
user under their licences; nothing from them is redistributed). Writes results/amass/.
"""
import argparse, glob, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "mdm"))          # MDM's own skeleton code (vendored repo)
import torch                                                     # noqa: E402
from data_loaders.humanml.common.skeleton import Skeleton        # noqa: E402
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="only the first N clips (to test the code)")
CLI = ap.parse_args()
OUT = os.path.join(ROOT, "results", "amass")
os.makedirs(OUT, exist_ok=True)

LAB = {"np": np}
exec(open(os.path.join(ROOT, "tools", "p2_labels_funcs.py")).read(), LAB)
SETTINGS = {
    "default": {},
    "contact x0.75": {"FOOT_HEIGHT_MAX": np.array([0.09, 0.045, 0.09, 0.045])},
    "contact x1.25": {"FOOT_HEIGHT_MAX": np.array([0.15, 0.075, 0.15, 0.075])},
    "support 10 cm": {"SUPPORT_HEIGHT_MIN": 0.10},
    "support 20 cm": {"SUPPORT_HEIGHT_MIN": 0.20},
    "floating 1 m/s2": {"FLOAT_G_MAX": 1.0},
    "floating 3 m/s2": {"FLOAT_G_MAX": 3.0},
}


# ------------------------------------------------------------------ SMPL+H joints

def load_model(gender):
    z = np.load(os.path.join(ROOT, "vendor", "smplh", gender, "model.npz"), allow_pickle=False)
    parents = z["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    return dict(v=z["v_template"], sd=z["shapedirs"][:, :, :10], Jr=z["J_regressor"], parents=parents)


def rodrigues(r):                      # [..., 3] axis-angle -> [..., 3, 3]
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = r / np.maximum(th, 1e-12)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def smplh_joints(model, poses, betas, trans):
    J = model["Jr"] @ (model["v"] + model["sd"] @ betas[:10])          # [52, 3] rest joints
    R = rodrigues(poses.reshape(len(poses), -1, 3))                    # [T, 52, 3, 3]
    par = model["parents"]
    pos = np.zeros((len(poses), len(par), 3))
    rot = np.zeros((len(poses), len(par), 3, 3))
    for i, p in enumerate(par):
        if p < 0:
            rot[:, i], pos[:, i] = R[:, i], J[i]
        else:
            rot[:, i] = rot[:, p] @ R[:, i]
            pos[:, i] = pos[:, p] + (rot[:, p] @ (J[i] - J[p]))
    return pos[:, :22] + trans[:, None]


# ------------------------------------------------------------------ HumanML3D processing

N_RAW = torch.from_numpy(t2m_raw_offsets)
REF = np.load(os.path.join(ROOT, "tests", "012314.npy"), allow_pickle=False).astype(np.float64)   # HumanML3D new_joints
TGT_OFF = Skeleton(N_RAW, t2m_kinematic_chain, "cpu").get_offsets_joints(torch.from_numpy(REF[0]))
L_IDX1, L_IDX2, FACE = 5, 8, [2, 1, 17, 16]                      # as in MDM's motion_process.py


def uniform_skeleton(positions):
    skel = Skeleton(N_RAW, t2m_kinematic_chain, "cpu")
    src = skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt = TGT_OFF.numpy()
    scale = (np.abs(tgt[L_IDX1]).max() + np.abs(tgt[L_IDX2]).max()) / (np.abs(src[L_IDX1]).max() + np.abs(src[L_IDX2]).max())
    quat = skel.inverse_kinematics_np(positions, FACE)
    skel.set_offset(TGT_OFF)
    return skel.forward_kinematics_np(quat, positions[:, 0] * scale), scale


SWAP = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])               # HumanML3D's z-up to y-up


def process(path, models):
    m = np.load(path, allow_pickle=False)
    gender = str(m["gender"])
    fps = float(m["mocap_framerate"])
    ds = int(fps / 20)
    poses, trans = m["poses"][::ds], m["trans"][::ds]
    j = smplh_joints(models.get(gender, models["neutral"]), poses, m["betas"], trans) @ SWAP
    j[..., 0] *= -1                                                  # HumanML3D's x negation: without it, mirrored
    j, scale = uniform_skeleton(j)
    j[..., 1] -= j.min(axis=(0, 1))[1]
    return j, gender, fps, scale


# ------------------------------------------------------------------ labels

def stretches(airborne, g):
    out, t, n = [], 0, len(airborne)
    while t < n:
        if not airborne[t]:
            t += 1
            continue
        s = t
        while t < n and airborne[t]:
            t += 1
        if np.isfinite(g[s]):
            out.append((s, t - s, float(g[s])))
    return out


def airborne_and_g(j, ov):
    G = dict(LAB); G.update(ov)
    feet = j[:, G["FOOT_JOINTS"]]
    air = (feet[..., 1] > G["FOOT_HEIGHT_MAX"][None]).all(1) & (j[:, G["SUPPORT_JOINTS"], 1] > G["SUPPORT_HEIGHT_MIN"]).all(1)
    com = G["center_of_mass"](j)
    return air, G["flight_gravity"](com[:, 1], air), G["foot_contacts"](j), feet[..., 1].min(1), G["FLOAT_G_MAX"]


files = sorted(glob.glob(os.path.join(ROOT, "vendor", "amass", "BMLmovi", "BMLmovi", "Subject_*", "*_poses.npz")))
if CLI.limit:
    files = files[:CLI.limit]
models = {g: load_model(g) for g in ["male", "female", "neutral"]}
clip_rows, st_rows, t0 = [], [], time.time()
for i, f in enumerate(files):
    name = os.path.basename(f)[:-len("_poses.npz")]
    subj, action = name.split("_")[1], int(name.split("_")[-1])
    j, gender, fps, scale = process(f, models)
    full = LAB["physics_labels"](j)
    xm = full["xcom_margin"]
    for st, ov in SETTINGS.items():
        air, g, contact, lowest, gmax = airborne_and_g(j, ov)
        floating = np.isfinite(g) & (g < gmax)
        ground = contact.any(1)
        clip_rows.append(dict(clip=name, subject=subj, action=action, gender=gender, fps=fps, frames=len(j), setting=st, scale=scale,
                              floating_rate=floating.mean(), raised_contact_rate=np.mean(lowest[ground] > 0.05) if ground.any() else np.nan,
                              in_flight=np.mean(~ground), airborne_stretches=len(stretches(air, g)),
                              xcom_out=np.mean(xm[np.isfinite(xm)] < 0) if st == "default" and np.isfinite(xm).any() else np.nan))
        for s, n, gv in stretches(air, g):
            st_rows.append(dict(clip=name, subject=subj, action=action, setting=st, start=s, frames=n, g=gv, floating=gv < gmax,
                                scale=scale, g_subject_scale=gv / scale))
    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(files)} clips ({time.time() - t0:.0f}s)", flush=True)

import pandas as pd   # noqa: E402
C, S = pd.DataFrame(clip_rows), pd.DataFrame(st_rows)
C.to_csv(os.path.join(OUT, "clips.csv"), index=False)
S.to_csv(os.path.join(OUT, "stretches.csv"), index=False)
d, sd = C[C.setting == "default"], S[S.setting == "default"]
fall = sd[~sd.floating]
w = fall.frames.values
lines = [f"AMASS BMLmovi: {len(d)} clips, {d.subject.nunique()} subjects, {d.frames.sum()} frames at 20 fps "
         f"(source {sorted(d.fps.unique())} fps)",
         f"Floating (default labels): {100 * np.average(d.floating_rate, weights=d.frames):.2f}% of frames; "
         f"{100 * (d.floating_rate > 0).mean():.1f}% of clips have any",
         f"Raised contact (>5 cm): {100 * np.nanmean(d.raised_contact_rate):.2f}% of contact frames (mean over clips)",
         f"Airborne stretches >= 0.2 s: {len(sd)} ({(sd.floating).sum()} floating, {len(fall)} falling), in {sd["clip"].nunique()} clips",
         f"Fitted g, falling stretches: median {fall.g.median():.2f}, mean {fall.g.mean():.2f}, SD {fall.g.std():.2f}, "
         f"IQR {fall.g.quantile(0.25):.2f} to {fall.g.quantile(0.75):.2f} m/s^2; frame-weighted SD {np.sqrt(np.cov(fall.g, aweights=w)):.2f}",
         f"Same stretches at the subject's own scale (g divided by HumanML3D's leg-length retargeting factor): median "
         f"{fall.g_subject_scale.median():.2f}, SD {fall.g_subject_scale.std():.2f}; factor median {d.scale.median():.3f}, range {d.scale.min():.3f} to {d.scale.max():.3f}",
         f"Fitted g, all stretches: share below 2 m/s^2 {100 * (sd.g < 2).mean():.1f}%",
         "Stretches per action index (default): " + ", ".join(f"{a}: {n}" for a, n in sd.groupby("action").size().items()),
         "", "By threshold setting (floating % of frames; clips with any floating %; falling stretches; median g):"]
for st in SETTINGS:
    c, s_ = C[C.setting == st], S[S.setting == st]
    f_ = s_[~s_.floating]
    lines.append(f"  {st}: {100 * np.average(c.floating_rate, weights=c.frames):.2f}%; {100 * (c.floating_rate > 0).mean():.1f}%; "
                 f"{len(f_)}; {f_.g.median() if len(f_) else np.nan:.2f}")
text = "\n".join(lines) + "\n"
open(os.path.join(OUT, "summary.txt"), "w").write(text)
print(text)
