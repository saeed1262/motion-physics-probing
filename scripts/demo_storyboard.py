"""Storyboard stills for the supplementary video, to agree on its look before making it.

Every body, path and number comes from the recorded motions (the jump of Figure fig:stepsfall, the floating
walk of Figure 1, the first example of Figure fig:layerknows); nothing is interpolated between recorded steps.
Writes paper/figures_web/data/demo_storyboard.js and img/demo_*.png, then renders demo_storyboard.html.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_web_figures as M
import body_images as bi
from example_picks import FLOAT_G_MAX, LABELS, load, stretches

ROOT, WEB, P = M.ROOT, M.WEB, M.PALETTE
COM = LABELS["center_of_mass"]
LIGHT_BODY = bi.hex_rgb(P["layer2"], toward_white=0.62)      # bodies on the dark stage
DARK_SHADOW = (0.06, 0.09, 0.15)


def png(name, bodies, box, dark=False):
    out = os.path.join(WEB, "img", f"demo_{name}.png")
    kw = dict(floor_rgb=None, flat=True)
    if dark:
        kw["shadow_rgb"] = DARK_SHADOW
    bi.render(bodies, out, box, **kw)
    return f"img/demo_{name}.png"


def path(to_local, pts):
    x, y, _ = bi.project(to_local(pts[None])[0])
    return [[float(a), float(b)] for a, b in zip(x, y)]


def jump():
    """The jump of fig_steps_fall (50-step, jump prompts, seed 0), picked there by a fixed rule."""
    film = json.loads(open(os.path.join(WEB, "data", "fig_steps_fall.js")).read().split("=", 1)[1].rstrip().rstrip(";"))["film"]
    d = os.path.join(ROOT, "data", "jump_heavy", "50steps_seed0")
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True)
    fin, x0 = lab["final"].item(), lab["x0"].item()
    i = [m for m, p in enumerate(meta["all_prompts"]) if p.rstrip(".") == film["prompt"]
         and any(abs(g - film["g_final"]) < 1e-4 for _, _, g in stretches(fin["flight_gravity"][m]))]
    assert len(i) == 1, i
    i = i[0]
    s, e, g_final = next((s, e, g) for s, e, g in stretches(fin["flight_gravity"][i]) if abs(g - film["g_final"]) < 1e-4)
    J = np.concatenate([np.load(p)["joints"] for p in files])[i]
    vers = [(0, "0%"), (10, "20%"), (20, "40%"), (None, "final")]
    X = {k: np.concatenate([np.load(p)[f"x0j_{k}"] for p in files])[i].astype(np.float32) for k, _ in vers if k is not None}
    apex = s + int(np.argmax(COM(J)[s:e, 1]))
    frames = sorted(set(range(max(s - 8, 0), min(e + 3, len(J)), 2)) | {apex})
    to_local = bi.local_frame(J[0, 0, [0, 2]], J[-1, 0, [0, 2]], J[s, 0, [0, 2]])
    seg = np.arange(max(s - 10, 0), min(e + 10, len(J)))
    out = dict(prompt=film["prompt"], apex_t=apex / 20.0, frames=frames, apex=apex, s=s, e=e, versions=[])
    V = {}
    for k, name in vers:
        jj = J if k is None else X[k]
        V[name] = to_local(bi.fit(jj[frames], f"demo_jh50s0_m{i}_{'final' if k is None else k}"))
        g = (fin if k is None else x0[k])["flight_gravity"][i]
        gk = g_final if k is None else next((gg for ss, ee, gg in stretches(g) if ss < e and ee > s), None)
        out["versions"].append(dict(name=name, step=k, g=None if gk is None else float(gk), com=path(to_local, COM(jj)[seg]),
                                    air=[bool(v) for v in np.isfinite(g[seg])], seg0=int(seg[0])))
    # one box for every panel of the four-stage frame: the bodies up to the apex and the paths up to the apex
    upto = [n for n, f in enumerate(frames) if f <= apex][-5:]
    allv = np.concatenate([V[nm][upto] for _, nm in vers])
    sx, sy, _ = bi.project(allv)
    box4 = (sx.min() - 0.25, sx.max() + 0.35, -0.25, sy.max() + 0.12)
    alpha = lambda n, N: 0.1 + 0.75 * (n / max(N - 1, 1)) ** 1.6
    for _, nm in vers:
        bodies = [(V[nm][f], bi.hex_rgb(P["body"]), alpha(n, len(upto)) if n < len(upto) - 1 else 0.92) for n, f in enumerate(upto)]
        out[f"img4_{nm}"] = png(f"jump4_{nm.rstrip('%')}", bodies, box4)
    out["box4"] = dict(zip(["x0", "x1", "y0", "y1"], map(float, box4)))
    # overlay frame: the whole flight, final motion (body color) and the estimate at 20% (teal), each its own image
    allv = np.concatenate([V["final"], V["20%"]])
    sx, sy, _ = bi.project(allv)
    boxo = (sx.min() - 0.2, sx.max() + 0.2, -0.25, sy.max() + 0.12)
    sel = list(range(0, len(frames), 2))
    for dark in (False, True):
        sfx = "_dark" if dark else ""
        body = LIGHT_BODY if dark else bi.hex_rgb(P["body"])
        out[f"imgo_final{sfx}"] = png(f"overlay_final{sfx}", [(V["final"][n], body, 0.18 + 0.72 * j / (len(sel) - 1)) for j, n in enumerate(sel)], boxo, dark)
        out[f"imgo_est{sfx}"] = png(f"overlay_est{sfx}", [(V["20%"][frames.index(apex)], bi.hex_rgb(P["est"]), 0.9)], boxo, dark)
    out["boxo"] = dict(zip(["x0", "x1", "y0", "y1"], map(float, boxo)))
    return out


def floating():
    """The floating walk of Figure 1 (main data, 50-step), with its fits."""
    fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
    prompts, pid, J, X0, final, x0lab = load()
    idx, frames = int(fits["float_motion"]), [int(f) for f in fits["float_frames"]]
    to_local = bi.local_frame(J[idx][0, 0, [0, 2]], J[idx][-1, 0, [0, 2]], J[idx][0, 0, [0, 2]])
    Vs = {ver: to_local(fits[f"float_{ver}_verts"]) for ver in ("est20", "final")}
    allv = np.concatenate(list(Vs.values()))
    sx, sy, _ = bi.project(allv)
    box = (sx.min() - 0.3, sx.max() + 0.3, -0.25, sy.max() + 0.12)
    out = dict(prompt=prompts[idx].rstrip("."), frames=frames, box=dict(zip(["x0", "x1", "y0", "y1"], map(float, box))), panels=[])
    for ver, k, name in (("est20", 10, "estimate at 20%"), ("final", None, "final motion")):
        jj = J[idx] if k is None else X0[k][idx]
        g = (final if k is None else x0lab[k])["flight_gravity"][idx]
        fl = np.isfinite(g) & (g < FLOAT_G_MAX)
        n = len(frames)
        bodies = [(Vs[ver][m], bi.hex_rgb(P["body_fail"] if fl[t] else P["body"]), 0.12 + 0.6 * m / max(n - 1, 1) if m < n - 1 else 0.92)
                  for m, t in enumerate(frames)]
        t = frames[-1]
        foot = jj[t, [7, 10, 8, 11]]
        lo = foot[np.argmin(foot[:, 1])]
        fx, fy, _ = bi.project(to_local(lo[None, None])[0])
        out["panels"].append(dict(name=name, img=png(f"float_{ver}", bodies, box), floating=bool(fl[t]),
                                  foot=[float(fx[0]), float(fy[0])], foot_cm=float(100 * lo[1]),
                                  n_float=int(fl.sum()), n=int(len(fl))))
    return out


def layer_gain():
    """The first example of fig_layer_knows (c): one of the three jumps where adding layer 5 helps most."""
    k = json.loads(open(os.path.join(WEB, "data", "fig_layer_knows.js")).read().split("=", 1)[1].rstrip().rstrip(";"))
    return dict(k["examples"][0], r2_est=k["r2_est"], r2_both=k["r2_both"])


if __name__ == "__main__":
    data = dict(jump=jump(), floating=floating(), gain=layer_gain())
    M.write_data("demo_storyboard", data)
    M.render("demo_storyboard", 1304, 1356)
