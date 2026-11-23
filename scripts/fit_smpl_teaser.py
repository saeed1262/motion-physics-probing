"""Fit SMPL bodies to the frames shown in Figure 1, for display only (all analysis uses joints).

Uses MDM's own joints-to-SMPL fitter (SMPLify-3D on the 22 HumanML3D joints, 150 iterations,
neutral body) on the CPU. For each of the two Figure 1 motions and each version (the model's
clean-motion estimate at 20% and 40% of denoising, and the final motion), fits the frames the
figure draws, and records how well the fit matches: mean joint error, and the lowest mesh
vertex against the lowest foot joint (so the floor gap in the render can be checked).

Needs vendor/mdm/body_models/smpl/SMPL_NEUTRAL.pkl (scripts/convert_smpl.py).

    .venv/bin/python scripts/fit_smpl_teaser.py

Writes data/smpl_teaser/fits.npz and prints the fit checks.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MDM = os.path.join(ROOT, "vendor", "mdm")
sys.path.insert(0, HERE)
from example_picks import load, pick, FLOAT_G_MAX  # noqa: E402

prompts, pid, J, X0, final, x0lab = load()
jump, floatw, walk = pick(prompts, pid, J, final, x0lab)
i_jump, s, e, _ = jump
fl = np.isfinite(final["flight_gravity"][floatw]) & (final["flight_gravity"][floatw] < FLOAT_G_MAX)
fi = np.where(fl)[0]
FRAMES = {"jump": list(range(max(s - 6, 0), min(e + 6, 119) + 1, 3)),
          "float": list(np.linspace(max(fi.min() - 8, 0), min(fi.max() + 8, 119), 9).round().astype(int))}
MOTION = {"jump": i_jump, "float": floatw}

os.chdir(MDM)                      # MDM's fitter uses paths relative to its repository root
sys.path.insert(0, MDM)
import smplx  # noqa: E402
import h5py  # noqa: E402
from visualize.joints2smpl.src import config  # noqa: E402
from visualize.joints2smpl.src.smplify import SMPLify3D  # noqa: E402

device = torch.device("cpu")
torch.manual_seed(0)
out = {}
for kind, idx in MOTION.items():
    frames = FRAMES[kind]
    n = len(frames)
    model = smplx.create(config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=n).to(device)
    mean = h5py.File(config.SMPL_MEAN_FILE, "r")
    pose0 = torch.from_numpy(mean["pose"][:]).unsqueeze(0).repeat(n, 1).float()
    shape0 = torch.from_numpy(mean["shape"][:]).unsqueeze(0).repeat(n, 1).float()
    fitter = SMPLify3D(smplxmodel=model, batch_size=n, joints_category="AMASS", num_iters=150, device=device)
    for version, src in [("est20", X0[10]), ("est40", X0[20]), ("final", J)]:
        j3d = torch.tensor(src[idx][frames], dtype=torch.float32)
        verts, joints, pose, betas, cam_t, loss = fitter(pose0.clone(), shape0.clone(), torch.zeros(1, 3), j3d,
                                                         conf_3d=torch.ones(22))
        cam_t = cam_t.detach().reshape(-1, 1, 3)       # one translation per frame
        world = (verts + cam_t).numpy()
        model_j = (joints + cam_t).numpy()[:, :22]
        err = np.linalg.norm(model_j - src[idx][frames], axis=-1).mean() * 100
        lowest_vertex = world[..., 1].min(1)
        lowest_foot = src[idx][frames][:, [7, 10, 8, 11], 1].min(1)
        print(f"{kind:5s} {version:5s}: mean joint error {err:.1f} cm | lowest vertex - lowest foot joint "
              f"{100 * np.mean(lowest_vertex - lowest_foot):+.1f} cm (range {100 * np.min(lowest_vertex - lowest_foot):+.1f} to "
              f"{100 * np.max(lowest_vertex - lowest_foot):+.1f})", flush=True)
        out[f"{kind}_{version}_verts"] = world.astype(np.float32)
        out[f"{kind}_{version}_joints_in"] = src[idx][frames].astype(np.float32)
    out[f"{kind}_frames"] = np.array(frames)
    out[f"{kind}_motion"] = np.array(idx)
out["faces"] = model.faces.astype(np.int32)
os.makedirs(os.path.join(ROOT, "data", "smpl_teaser"), exist_ok=True)
np.savez_compressed(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"), **out)
print("saved data/smpl_teaser/fits.npz")
