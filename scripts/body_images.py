"""SMPL bodies for the web figures, for display only (every number in a figure comes from the joints).

fit(joints, key): MDM's joints-to-SMPL fitter (SMPLify-3D on the 22 HumanML3D joints, 150 iterations,
neutral body, CPU), as in fit_smpl_teaser.py; cached in data/smpl_web/<key>.npz.
render(...): the painter's-algorithm renderer of make_teaser_smpl.py (camera slightly above the floor,
Lambert shading, back faces culled, a vertical shadow on the floor) into a transparent PNG, returning the
screen box it covers, so a page can draw arcs and labels on top with project().
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MDM = os.path.join(ROOT, "vendor", "mdm")
CACHE = os.path.join(ROOT, "data", "smpl_web")
ELEV = np.deg2rad(12)
LIGHT = np.array([-0.45, 0.75, -0.5]); LIGHT /= np.linalg.norm(LIGHT)
TO_CAM = np.array([0.0, np.sin(ELEV), -np.cos(ELEV)])
_FACES = None


def faces():
    global _FACES
    if _FACES is None:
        _FACES = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))["faces"]
    return _FACES


def fit(joints, key):
    """joints [n, 22, 3] (metres, y up) -> SMPL vertices [n, 6890, 3] in the same world frame."""
    path = os.path.join(CACHE, f"{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if z["joints_in"].shape == joints.shape and np.allclose(z["joints_in"], joints, atol=1e-4):
            return z["verts"]
    import torch
    cwd = os.getcwd()
    os.chdir(MDM)                    # the fitter uses paths relative to MDM's repository root
    sys.path.insert(0, MDM)
    import smplx, h5py
    from visualize.joints2smpl.src import config
    from visualize.joints2smpl.src.smplify import SMPLify3D
    torch.manual_seed(0)
    n = len(joints)
    model = smplx.create(config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=n)
    mean = h5py.File(config.SMPL_MEAN_FILE, "r")
    pose0 = torch.from_numpy(mean["pose"][:]).unsqueeze(0).repeat(n, 1).float()
    shape0 = torch.from_numpy(mean["shape"][:]).unsqueeze(0).repeat(n, 1).float()
    fitter = SMPLify3D(smplxmodel=model, batch_size=n, joints_category="AMASS", num_iters=150, device=torch.device("cpu"))
    j3d = torch.tensor(joints, dtype=torch.float32)
    verts, jts, pose, betas, cam_t, loss = fitter(pose0, shape0, torch.zeros(1, 3), j3d, conf_3d=torch.ones(22))
    cam_t = cam_t.detach().reshape(-1, 1, 3)
    world = (verts + cam_t).numpy().astype(np.float32)
    err = np.linalg.norm((jts + cam_t).numpy()[:, :22] - joints, axis=-1).mean() * 100
    os.chdir(cwd)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, verts=world, joints_in=joints.astype(np.float32))
    print(f"fit {key}: {n} frames, mean joint error {err:.1f} cm", flush=True)
    return world


def local_frame(root_xz_start, root_xz_end, origin_xz):
    """World -> (forward, up, depth): forward along the motion's travel (x if it barely moves)."""
    d = np.asarray(root_xz_end) - np.asarray(root_xz_start)
    d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.3 else np.array([1.0, 0.0])
    o = np.asarray(origin_xz)
    return lambda p: np.stack([(p[..., 0] - o[0]) * d[0] + (p[..., 2] - o[1]) * d[1], p[..., 1],
                               -(p[..., 0] - o[0]) * d[1] + (p[..., 2] - o[1]) * d[0]], -1)


def project(p, elev=ELEV):
    """(forward, up, depth) -> screen x, y (metres) and depth, for a camera elev radians above the floor."""
    return p[..., 0], p[..., 1] * np.cos(elev) + p[..., 2] * np.sin(elev), p[..., 2] * np.cos(elev) - p[..., 1] * np.sin(elev)


def render(bodies, out_png, box, px_per_m=220, shadow_rgb=(0.765, 0.788, 0.831), floor_rgb=(0.949, 0.957, 0.973),
           elev=ELEV, floor_depth=(-0.45, 0.6), flat=False, overlay=None):
    """bodies: list of (vertices in the local frame [6890, 3], rgb) drawn in order (later on top).
    box: (x0, x1, y0, y1) screen box in metres; the PNG covers exactly this box (transparent background)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    x0, x1, y0, y1 = box
    dpi = 100
    fig = plt.figure(figsize=((x1 - x0) * px_per_m / dpi, (y1 - y0) * px_per_m / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.axis("off")
    F = faces()
    ys = [floor_depth[0] * np.sin(elev), floor_depth[1] * np.sin(elev)]
    if floor_rgb is not None:
        ax.fill_between([x0, x1], ys[0], ys[1], color=floor_rgb, lw=0, zorder=0)
    to_cam = np.array([0.0, np.sin(elev), -np.cos(elev)])
    for zi, body in enumerate(bodies):
        V, rgb = body[0], body[1]
        alpha = body[2] if len(body) > 2 else 1.0
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        front = n @ to_cam > 0
        sx, sy, depth = project(tri, elev)
        sh = tri.copy(); sh[..., 1] = 0.0
        hx, hy, _ = project(sh, elev)
        ax.add_collection(PolyCollection(np.stack([hx, hy], -1), facecolors=shadow_rgb, edgecolors="none", zorder=1 + 2 * zi))
        order = np.argsort(-depth[front].mean(1))
        shade = np.ones(int(front.sum())) if flat else 0.42 + 0.58 * np.clip(n[front] @ LIGHT, 0, 1)
        cols = np.clip(np.array(rgb)[None] * shade[:, None] + 0.08 * (1 - shade[:, None]), 0, 1)[order]
        ax.add_collection(PolyCollection(np.stack([sx[front], sy[front]], -1)[order], facecolors=cols, edgecolors=cols,
                                         linewidths=0.15, zorder=2 + 2 * zi, alpha=alpha))
    if overlay is not None:        # extra drawing in screen metres, e.g. a center-of-mass path
        overlay(ax)
    if out_png.endswith(".jpg"):   # white background, small file (figure pages are white)
        fig.savefig(out_png, dpi=dpi, facecolor="white", pil_kwargs={"quality": 90})
    else:
        fig.savefig(out_png, dpi=dpi, transparent=True)
    plt.close(fig)
    return dict(x0=x0, x1=x1, y0=y0, y1=y1)


def hex_rgb(h, toward_white=0.0):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return tuple(1 - (1 - toward_white) * (1 - c))
