"""Supplementary video (dark stage), following the agreed storyboard (paper/figures_web/demo_storyboard.html).

Every body, path and number comes from the recorded motions and their labels: the jump of Figure fig:stepsfall,
the floating walk of Figure 1, and the three jumps of Figure fig:layerknows (c). Nothing is interpolated: slow
motion holds recorded frames. Bodies are SMPL meshes fitted to the generated joints for display (body_images.fit).

    .venv/bin/python scripts/make_demo_video.py [prepare|render|compose|all]

Writes paper/supplement/demo_video.mp4 (1920x1080, 20 fps). Working files in data/demo_video/.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "data", "demo_video")
OUT = os.path.join(ROOT, "paper", "supplement", "demo_video.mp4")
FONTS = os.path.join(ROOT, "paper", "figures_web", "fonts")
FPS, W, H, SS = 20, 1280, 720, 3                  # layout units; frames are drawn at SS x and downscaled to OW x OH
OW, OH = 1920, 1080
R = SS / 2                                         # body layers were sized for SS = 2
PAL = json.load(open(os.path.join(ROOT, "paper", "figures_web", "palette.json")))
TRAIL = [(9, 0.12), (6, 0.22), (3, 0.38)]          # earlier frames drawn faintly behind the current pose
DARK_SHADOW = (0.06, 0.09, 0.15)


def hexrgb(h, a=255):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def light(h, t):                                    # toward white, for bodies on the dark stage
    r, g, b, _ = hexrgb(h); return tuple(int(c + (255 - c) * t) for c in (r, g, b))


# ------------------------------------------------------------------ prepare: data and SMPL fits

def prepare():
    import body_images as bi
    from example_picks import FLOAT_G_MAX, LABELS, load, stretches
    COM = LABELS["center_of_mass"]
    os.makedirs(WORK, exist_ok=True)
    jobs, meta = [], {}

    # the jump of fig_steps_fall
    film = json.loads(open(os.path.join(ROOT, "paper", "figures_web", "data", "fig_steps_fall.js")).read().split("=", 1)[1].rstrip().rstrip(";"))["film"]
    d = os.path.join(ROOT, "data", "jump_heavy", "50steps_seed0")
    m_ = json.load(open(os.path.join(d, "meta.json"))); files = sorted(glob.glob(os.path.join(d, "batch_*.npz")))
    lab = np.load(os.path.join(d, "labels.npz"), allow_pickle=True); fin, x0 = lab["final"].item(), lab["x0"].item()
    i = [k for k, p in enumerate(m_["all_prompts"]) if p.rstrip(".") == film["prompt"]
         and any(abs(g - film["g_final"]) < 1e-4 for _, _, g in stretches(fin["flight_gravity"][k]))][0]
    s, e, gf = next((a, b, g) for a, b, g in stretches(fin["flight_gravity"][i]) if abs(g - film["g_final"]) < 1e-4)
    J = np.concatenate([np.load(p)["joints"] for p in files])[i]
    seg0, seg1 = max(s - 24, 9), min(e + 16, len(J))
    F = list(range(seg0 - 9, seg1))
    to_local = bi.local_frame(J[0, 0, [0, 2]], J[-1, 0, [0, 2]], J[s, 0, [0, 2]])
    vers = []
    for k, name in [(0, "0%"), (10, "20%"), (20, "40%"), (None, "final")]:
        jj = J if k is None else np.concatenate([np.load(p)[f"x0j_{k}"] for p in files])[i].astype(np.float32)
        g = (fin if k is None else x0[k])["flight_gravity"][i]
        gk = gf if k is None else next((gg for ss, ee, gg in stretches(g) if ss < e and ee > s), None)
        key = f"video_jump_{name.rstrip('%')}"
        jobs.append((key, jj[F].astype(np.float32)))
        com = bi.project(to_local(COM(jj)[None])[0])
        vers.append(dict(name=name, key=key, g=None if gk is None else float(gk), comx=com[0].tolist(), comy=com[1].tolist(),
                         air=[bool(v) for v in np.isfinite(g)]))
    meta["jump"] = dict(prompt=film["prompt"], s=s, e=e, F=F, seg=[seg0, seg1], versions=vers, local=dict(start=J[0, 0, [0, 2]].tolist(),
                        end=J[-1, 0, [0, 2]].tolist(), origin=J[s, 0, [0, 2]].tolist()))

    # the floating walk of Figure 1
    fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
    prompts, pid, JJ, X0, final, x0lab = load()
    idx = int(fits["float_motion"])
    Fw = list(range(len(JJ[idx])))
    to_w = bi.local_frame(JJ[idx][0, 0, [0, 2]], JJ[idx][-1, 0, [0, 2]], JJ[idx][0, 0, [0, 2]])
    wv = []
    for k, name in [(10, "estimate at 20%"), (None, "final motion")]:
        jj = JJ[idx] if k is None else X0[k][idx]
        g = (final if k is None else x0lab[k])["flight_gravity"][idx]
        fl = np.isfinite(g) & (g < FLOAT_G_MAX)
        key = f"video_walk_{'final' if k is None else k}"
        jobs.append((key, jj.astype(np.float32)))
        foot = jj[:, [7, 10, 8, 11]]; lo = foot[np.arange(len(jj)), np.argmin(foot[..., 1], 1)]
        fx = bi.project(to_w(lo[None])[0])
        wv.append(dict(name=name, key=key, floating=[bool(v) for v in fl], foot_x=fx[0].tolist(), foot_y=fx[1].tolist(),
                       foot_cm=(100 * lo[:, 1]).tolist(), n_float=int(fl.sum())))
    meta["walk"] = dict(prompt=prompts[idx].rstrip("."), F=Fw, versions=wv, local=dict(start=JJ[idx][0, 0, [0, 2]].tolist(),
                        end=JJ[idx][-1, 0, [0, 2]].tolist(), origin=JJ[idx][0, 0, [0, 2]].tolist()))

    # the three jumps of fig_layer_knows (c), selected there by the same rule: largest error reduction, travel >= 0.4 m
    RES = os.path.join(ROOT, "results")
    z = np.load(os.path.join(RES, "jump_heavy", "fair", "preds", "50steps_fall_gravity.npz"))
    y, seed, pe, pb = z["y"], z["seed"], z["estimate | tuned"], z["estimate + layer | tuned+fallback"]
    motion = np.empty(len(y), int); runs = sorted(glob.glob(os.path.join(ROOT, "data", "jump_heavy", "50steps_seed*")))
    for dd in runs:
        mm = json.load(open(os.path.join(dd, "meta.json"))); FR = np.arange(0, mm["n_frames"], mm["frame_stride"])
        g = np.load(os.path.join(dd, "labels.npz"), allow_pickle=True)["final"].item()["flight_gravity"][:, FR]
        mi, _ = np.nonzero(np.isfinite(g) & (g >= LABELS["FLOAT_G_MAX"])); motion[seed == mm["seed"]] = mi
    key_ = np.stack([seed, motion, np.round(y, 4)], 1); uk, inv = np.unique(key_, axis=0, return_inverse=True)
    agg = lambda v: np.bincount(inv, v) / np.bincount(inv)
    Y, E, B = agg(y), agg(pe), agg(pb)
    r2 = lambda p: 1 - np.mean((y - p) ** 2) / np.var(y)
    ex = []
    for k in np.argsort(-(np.abs(E - Y) - np.abs(B - Y))):
        if len(ex) == 3:
            break
        s_, mo, gv = int(uk[k, 0]), int(uk[k, 1]), float(Y[k])
        dd = [r for r in runs if r.endswith(f"seed{s_}")][0]
        mm = json.load(open(os.path.join(dd, "meta.json"))); ff = sorted(glob.glob(os.path.join(dd, "batch_*.npz")))
        Jg = np.concatenate([np.load(p)["joints"] for p in ff])[mo]
        Xg = np.concatenate([np.load(p)["x0j_10"] for p in ff])[mo].astype(np.float32)
        gfull = np.load(os.path.join(dd, "labels.npz"), allow_pickle=True)["final"].item()["flight_gravity"][mo]
        fr = np.where(np.isclose(gfull, gv, atol=1e-3))[0]; a, b = int(fr.min()), int(fr.max()) + 1
        if np.linalg.norm(Jg[b - 1, 0, [0, 2]] - Jg[a, 0, [0, 2]]) < 0.4:
            continue
        Fg = list(range(max(a - 29, 0), min(b + 20, len(Jg))))
        tl = bi.local_frame(Jg[0, 0, [0, 2]], Jg[-1, 0, [0, 2]], Jg[a, 0, [0, 2]])
        cf, ce = bi.project(tl(COM(Jg)[None])[0]), bi.project(tl(COM(Xg)[None])[0])
        key = f"video_gain_{len(ex)}"
        jobs.append((key, Jg[Fg].astype(np.float32)))
        ex.append(dict(key=key, prompt=mm["all_prompts"][mo].rstrip("."), s=a, e=b, F=Fg, g=gv, pe=float(E[k]), pb=float(B[k]),
                       comx=cf[0].tolist(), comy=cf[1].tolist(), estx=ce[0].tolist(), esty=ce[1].tolist(),
                       air=[bool(v) for v in np.isfinite(gfull)], local=dict(start=Jg[0, 0, [0, 2]].tolist(), end=Jg[-1, 0, [0, 2]].tolist(),
                       origin=Jg[a, 0, [0, 2]].tolist())))
    meta["gain"] = dict(examples=ex, r2_est=float(r2(pe)), r2_both=float(r2(pb)))

    from multiprocessing import Pool
    with Pool(5) as pool:
        res = pool.map(_fit, jobs)
    for key, V in res:
        np.save(os.path.join(WORK, f"{key}.npy"), V)
    json.dump(meta, open(os.path.join(WORK, "meta.json"), "w"))
    print("prepared:", ", ".join(k for k, _ in jobs))


def _fit(job):
    import torch
    torch.set_num_threads(2)
    import body_images as bi
    key, joints = job
    return key, bi.fit(joints, key).astype(np.float32)


# ------------------------------------------------------------------ render: body layers

def _local(m):
    import body_images as bi
    return bi.local_frame(np.array(m["start"]), np.array(m["end"]), np.array(m["origin"]))


def box_of(V, pad=(0.25, 0.25, 0.25, 0.12)):
    import body_images as bi
    x, y, _ = bi.project(V)
    return (float(x.min() - pad[0]), float(x.max() + pad[1]), -pad[2], float(y.max() + pad[3]))


def render():
    import body_images as bi
    meta = json.load(open(os.path.join(WORK, "meta.json")))
    os.makedirs(os.path.join(WORK, "img"), exist_ok=True)
    jobs, layout = [], {}
    body = tuple(c / 255 for c in light(PAL["layer2"], 0.62)); fail = tuple(c / 255 for c in light(PAL["body_fail"], 0.1))
    est = tuple(c / 255 for c in light(PAL["est"], 0.0))

    def trail_frames(F, f):
        return [(f - o, a) for o, a in TRAIL if f - o >= F[0]] + [(f, 0.95)]

    # clip 1: four versions, one shared box
    J = meta["jump"]; F = J["F"]; tl = _local(J["local"])
    Vs = {v["name"]: tl(np.load(os.path.join(WORK, v["key"] + ".npy"))) for v in J["versions"]}
    box = box_of(np.concatenate([V[F.index(J["seg"][0]) - 9:] for V in Vs.values()]))
    ppm = min(560 * R / (box[1] - box[0]), 620 * R / (box[3] - box[2]))
    layout["jump"] = dict(box=box, ppm=ppm)
    for v in J["versions"]:
        for f in range(*J["seg"]):
            out = os.path.join(WORK, "img", f"jump_{v['name'].rstrip('%')}_{f:03d}.png")
            jobs.append((out, [(v["key"], tl_key("jump"), ff - F[0], body, a) for ff, a in trail_frames(F, f)], box, ppm))
    # clip 1 replay: final motion with a trail, and the estimate at 20% alone, larger
    boxr = box_of(np.concatenate([Vs["final"][F.index(J["s"] - 6):F.index(J["e"] + 4)], Vs["20%"][F.index(J["s"] - 6):F.index(J["e"] + 4)]]))
    ppmr = min(1500 * R / (boxr[1] - boxr[0]), 900 * R / (boxr[3] - boxr[2]))
    layout["replay"] = dict(box=boxr, ppm=ppmr)
    for f in range(J["s"] - 4, J["e"] + 4):
        vf = next(v for v in J["versions"] if v["name"] == "final"); ve = next(v for v in J["versions"] if v["name"] == "20%")
        jobs.append((os.path.join(WORK, "img", f"replay_final_{f:03d}.png"), [(vf["key"], tl_key("jump"), ff - F[0], body, a) for ff, a in trail_frames(F, f)], boxr, ppmr))
        jobs.append((os.path.join(WORK, "img", f"replay_est_{f:03d}.png"), [(ve["key"], tl_key("jump"), f - F[0], est, 0.85)], boxr, ppmr))
    # clip 2: the floating walk, two panels
    Wk = meta["walk"]; tw = _local(Wk["local"])
    Vw = {v["name"]: tw(np.load(os.path.join(WORK, v["key"] + ".npy"))) for v in Wk["versions"]}
    boxw = box_of(np.concatenate(list(Vw.values())), pad=(0.3, 0.3, 0.25, 0.12))
    ppmw = min(1140 * R / (boxw[1] - boxw[0]), 760 * R / (boxw[3] - boxw[2]))
    layout["walk"] = dict(box=boxw, ppm=ppmw)
    for v in Wk["versions"]:
        for f in Wk["F"][9:]:
            bodies = [(v["key"], tl_key("walk"), ff, fail if v["floating"][ff] else body, a) for ff, a in trail_frames(Wk["F"], f)]
            jobs.append((os.path.join(WORK, "img", f"walk_{v['key']}_{f:03d}.png"), bodies, boxw, ppmw))
    # clip 3: three jumps, one panel each, one shared scale
    G = meta["gain"]["examples"]; boxes = []
    for x in G:
        Vg = _local(x["local"])(np.load(os.path.join(WORK, x["key"] + ".npy"))); boxes.append(box_of(Vg))
    ppmg = min(min(1240 * R / (b[1] - b[0]), 880 * R / (b[3] - b[2])) for b in boxes)
    layout["gain"] = dict(boxes=boxes, ppm=ppmg)
    for x, bx in zip(G, boxes):
        for f in x["F"][9:]:
            jobs.append((os.path.join(WORK, "img", f"{x['key']}_{f:03d}.png"), [(x["key"], tl_key(x["key"]), ff - x["F"][0], body, a)
                         for ff, a in trail_frames(x["F"], f)], bx, ppmg))
    json.dump(layout, open(os.path.join(WORK, "layout.json"), "w"))
    from multiprocessing import Pool
    with Pool(12) as pool:
        for n, _ in enumerate(pool.imap_unordered(_render, jobs, chunksize=4)):
            if n % 100 == 0:
                print(f"rendered {n}/{len(jobs)}", flush=True)
    print("rendered", len(jobs), "body layers")


def tl_key(which):
    return which


_CACHE = {}


def _render(job):
    import body_images as bi
    out, bodies, box, ppm = job
    if os.path.exists(out):
        return out
    meta = _CACHE.setdefault("meta", json.load(open(os.path.join(WORK, "meta.json"))))
    def local_for(which):
        if which in ("jump", "walk"):
            return _local(meta[which]["local"])
        return _local(next(x for x in meta["gain"]["examples"] if x["key"] == which)["local"])
    spec = []
    for key, which, n, rgb, a in bodies:
        V = _CACHE.get(key)
        if V is None:
            V = _CACHE[key] = local_for(which)(np.load(os.path.join(WORK, key + ".npy")))
        spec.append((V[n], rgb, a))
    bi.render(spec, out, tuple(box), px_per_m=ppm, floor_rgb=None, shadow_rgb=DARK_SHADOW, flat=True)
    return out


# ------------------------------------------------------------------ compose and encode

def compose():
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    meta = json.load(open(os.path.join(WORK, "meta.json"))); layout = json.load(open(os.path.join(WORK, "layout.json")))
    WW, HH = W * SS, H * SS
    font = lambda name, size: ImageFont.truetype(os.path.join(FONTS, f"Literata-{name}-static.ttf"), size * SS)
    F_T, F_S, F_M, F_B, F_N = font("SemiBold", 30), font("Regular", 16), font("Regular", 14), font("SemiBold", 17), font("SemiBold", 22)
    INK, MUTED, TRACK = (238, 242, 248), (154, 167, 191), (51, 65, 94)
    RED, TEAL, BLUE = hexrgb(PAL["mark"])[:3], hexrgb(PAL["est"])[:3], light(PAL["layer"], 0.35)
    bg = Image.new("RGB", (WW, HH)); g = np.linspace(0, 1, HH)[:, None]
    a0, a1 = np.array(hexrgb("#1b2740")[:3]), np.array(hexrgb("#0e1524")[:3])
    bg = Image.fromarray((a0 * (1 - g) + a1 * g)[:, None, :].repeat(WW, 1).astype(np.uint8).reshape(HH, WW, 3))
    imgcache = {}

    def img(path):
        if path not in imgcache:
            if len(imgcache) > 400:
                imgcache.clear()
            imgcache[path] = Image.open(path).convert("RGBA")
        return imgcache[path]

    def text(d, xy, s, f, fill, anchor="la", alpha=1.0):
        d.text((xy[0] * SS, xy[1] * SS), s, font=f, fill=fill + (int(255 * alpha),), anchor=anchor)

    def header(d, n, title, sub, alpha=1.0):
        if n:
            d.ellipse([24 * SS, 22 * SS, 46 * SS, 44 * SS], fill=INK + (int(255 * alpha),))
            text(d, (35, 33), str(n), F_B, (14, 21, 36), "mm", alpha)
            for k in (1, 2, 3):
                x, y = (W - 62 + 13 * k) * SS, 33 * SS
                d.ellipse([x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS], outline=MUTED + (int(255 * alpha),), width=SS,
                          fill=(INK + (int(255 * alpha),)) if k == n else None)
        text(d, (56 if n else 24, 22), title, F_T, INK, alpha=alpha)
        text(d, (24, 62), sub, F_S, MUTED, alpha=alpha)

    def timebar(d, t, total, speed):
        x0, x1, y = 24 * SS, (W - 24) * SS, (H - 22) * SS
        px = x0 + (x1 - x0) * min(t / total, 1)
        d.line([x0, y, x1, y], fill=TRACK + (255,), width=4 * SS); d.line([x0, y, px, y], fill=MUTED + (255,), width=4 * SS)
        d.ellipse([px - 6 * SS, y - 6 * SS, px + 6 * SS, y + 6 * SS], fill=(14, 21, 36, 255), outline=INK + (255,), width=2 * SS)
        text(d, (24, H - 44), f"t = {t:.2f} s", F_M, MUTED); text(d, (W - 24, H - 44), speed, F_M, MUTED, "ra")

    def place(frame, im, ox, oy, w, h, box=None, ppm=None, band=None):
        """Paste a body layer centred in (ox, oy, w, h) (output px), floor-aligned at the bottom; optionally draw a floor
        band under it first, from x band[0] to band[1] (output px). Returns the layer's top-left corner (SS px)."""
        x, y = pos(im, ox, oy, w, h)
        if band is not None:
            floorband(ImageDraw.Draw(frame), band[0], band[1], int(y + box[3] * ppm))
        frame.alpha_composite(im, (x, y))
        return x, y

    def path(d, xs, ys, air, mp, color, width, dash=False, ground=None):
        pts = [mp(a, b) for a, b in zip(xs, ys)]
        if ground:
            d.line(pts, fill=ground + (150,), width=SS, joint="curve")
        run = []
        for p, a in zip(pts, air):
            if a:
                run.append(p)
            elif run:
                draw_run(d, run, color, width, dash); run = []
        if run:
            draw_run(d, run, color, width, dash)

    def draw_run(d, run, color, width, dash):
        if len(run) < 2:
            return
        if not dash:
            d.line(run, fill=color + (255,), width=width * SS, joint="curve"); return
        for k in range(0, len(run) - 1, 2):
            d.line(run[k:k + 2], fill=color + (255,), width=width * SS)

    def floorband(d, x0, x1, y):
        for k in range(26 * SS):
            d.line([x0 * SS, y + k, x1 * SS, y + k], fill=(37, 50, 77, int(200 * (1 - k / (26 * SS)))))

    def mapper(box, ppm, px, py):
        return lambda X, Y: (px + (X - box[0]) * ppm, py + (box[3] - Y) * ppm)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stills = os.path.join(WORK, "stills"); os.makedirs(stills, exist_ok=True)
    wr = imageio_ffmpeg.write_frames(OUT, (OW, OH), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    wr.send(None)
    state = dict(n=0, last=None, marks={}, clip=None)
    clipdir = os.path.join(WORK, "clips"); os.makedirs(clipdir, exist_ok=True)

    def start(name):
        """Each section is also written as its own file (data/demo_video/clips/<name>.mp4), for the presentation."""
        if state["clip"] is not None:
            state["clip"].close()
        state["clip"] = imageio_ffmpeg.write_frames(os.path.join(clipdir, f"{name}.mp4"), (OW, OH), fps=FPS, codec="libx264",
                                                    pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
        state["clip"].send(None)

    def send(small):
        b = np.asarray(small).tobytes()
        wr.send(b); state["n"] += 1; state["last"] = small
        if state["clip"] is not None:
            state["clip"].send(b)

    def emit(fr, n=1, mark=None):
        small = fr.convert("RGB").resize((OW, OH), Image.LANCZOS)
        if mark and mark not in state["marks"]:
            state["marks"][mark] = True; small.save(os.path.join(stills, f"{mark}.png"))
        for _ in range(n):
            send(small)

    def hold(n):
        for _ in range(n):
            send(state["last"])

    def fade(n=6):
        if state["last"] is None:
            return
        last, base = state["last"], bg.resize((OW, OH))
        for k in range(1, n + 1):
            send(Image.blend(last, base, k / n))

    def pos(im, ox, oy, w, h):
        return int((ox + (w - im.width / SS) / 2) * SS), int((oy + h - im.height / SS) * SS)

    J = meta["jump"]; LJ = layout["jump"]; vers = J["versions"]; F = J["F"]
    start("title")
    # title card: 4 s, the final jump replayed faintly behind the title
    seg = list(range(*J["seg"]))
    for k in range(4 * FPS):
        fr = bg.copy().convert("RGBA"); f = seg[k % len(seg)]
        im = img(os.path.join(WORK, "img", f"jump_final_{f:03d}.png")).copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * 0.22)))
        place(fr, im, 640, 120, 600, 420); d = ImageDraw.Draw(fr); al = min(1, k / 20)
        text(d, (60, 250), "Probing Physics in Motion Diffusion", font("SemiBold", 38), INK, alpha=al)
        text(d, (60, 300), "Models Against Their Own Estimate", font("SemiBold", 38), INK, alpha=al)
        d.rounded_rectangle([60 * SS, 362 * SS, 320 * SS, 368 * SS], radius=3 * SS, fill=hexrgb(PAL["ramp5"])[:3] + (int(255 * al),))
        text(d, (60, 392), "What the model's own clean-motion estimate already shows,", font("Regular", 20), MUTED, alpha=min(1, max(0, (k - 15) / 20)))
        text(d, (60, 420), "and what its layers add.", font("Regular", 20), MUTED, alpha=min(1, max(0, (k - 15) / 20)))
        text(d, (60, 640), "Supplementary video · anonymous submission", F_M, MUTED, alpha=al)
        emit(fr, mark="title" if k == 3 * FPS else None)
    fade()

    start("clip1")
    # clip 1: the four stages in sync, at half speed
    PW, PG = 290, 14
    for f in seg:
        for rep in range(2):
            fr = bg.copy().convert("RGBA"); d = ImageDraw.Draw(fr)
            header(d, 1, "The model's own estimate, step by step", f"“{J['prompt']}”: the clean-motion estimate at three points of denoising, and the final motion")
            for j, v in enumerate(vers):
                ox = 24 + j * (PW + PG)
                d.rounded_rectangle([ox * SS, 110 * SS, (ox + PW) * SS, 610 * SS], radius=10 * SS, fill=(24, 34, 56, 255), outline=(44, 58, 88, 255), width=SS)
                text(d, (ox + 14, 124), "final motion" if v["name"] == "final" else f"estimate at {v['name']}", F_B, INK)
                im = img(os.path.join(WORK, "img", f"jump_{v['name'].rstrip('%')}_{f:03d}.png"))
                x, y = place(fr, im, ox, 150, PW, 400, LJ["box"], LJ["ppm"], (ox + 1, ox + PW - 1))
                mp = mapper(LJ["box"], LJ["ppm"], x, y)
                n = f - F[0] + 1
                path(d, v["comx"][F[0]:f + 1], v["comy"][F[0]:f + 1], v["air"][F[0]:f + 1], mp, RED, 3, ground=MUTED)
                if v["air"][f]:
                    cx, cy = mp(v["comx"][f], v["comy"][f]); d.ellipse([cx - 5 * SS, cy - 5 * SS, cx + 5 * SS, cy + 5 * SS], fill=RED + (255,))
                done = f >= J["e"]
                text(d, (ox + 14, 580), "fall rate", F_M, MUTED)
                if done:
                    gtxt = "no fall" if v["g"] is None else f"{v['g']:.1f} m/s²"
                    text(d, (ox + PW - 14, 580), gtxt, F_B, RED, "ra")
            timebar(d, f / FPS, 6.0, "0.5×")
            emit(fr, mark="clip1_landing" if f == J["e"] + 2 else None)
    hold(2 * FPS)
    fade()

    start("replay")
    # clip 1 replay: quarter speed, the estimate at 20% over the final motion, numbers at the end
    LR = layout["replay"]; vf = next(v for v in vers if v["name"] == "final"); ve = next(v for v in vers if v["name"] == "20%")
    rng = list(range(J["s"] - 4, J["e"] + 4))
    for i_, f in enumerate(rng):
        for rep in range(4):
            fr = bg.copy().convert("RGBA"); d = ImageDraw.Draw(fr)
            header(d, 1, "At 20% of denoising, the jump is already there", "quarter speed: the final motion, and the estimate at 20% of denoising (teal) at the same moment")
            im = img(os.path.join(WORK, "img", f"replay_final_{f:03d}.png")); x, y = place(fr, im, 24, 100, 830, 560, LR["box"], LR["ppm"], (24, 854))
            ie = img(os.path.join(WORK, "img", f"replay_est_{f:03d}.png")); fr.alpha_composite(ie, (x, y))
            mp = mapper(LR["box"], LR["ppm"], x, y)
            path(d, vf["comx"][rng[0]:f + 1], vf["comy"][rng[0]:f + 1], vf["air"][rng[0]:f + 1], mp, RED, 3)
            path(d, ve["comx"][rng[0]:f + 1], ve["comy"][rng[0]:f + 1], ve["air"][rng[0]:f + 1], mp, TEAL, 3, dash=True)
            al = min(1, max(0, (i_ - len(rng) + 6) / 5))
            if al > 0:
                cx = 880
                d.rounded_rectangle([cx * SS, 170 * SS, (cx + 370) * SS, 420 * SS], radius=12 * SS, fill=(31, 43, 69, int(255 * al)), outline=(44, 58, 88, int(255 * al)), width=SS)
                text(d, (cx + 24, 196), "fall rate over the flight", F_B, INK, alpha=al)
                for k, (lab, val, c, dp) in enumerate([("final motion", vf["g"], INK, 1), ("estimate at 20%", ve["g"], TEAL, 1), ("gravity", 9.81, MUTED, 2)]):
                    text(d, (cx + 24, 250 + 46 * k), lab, F_S, c, alpha=al); text(d, (cx + 346, 250 + 46 * k), f"{val:.{dp}f} m/s²", F_B, c, "ra", al)
                text(d, (cx + 24, 392), "parabolas fitted to each center-of-mass path", F_M, MUTED, alpha=al)
            timebar(d, f / FPS, 6.0, "0.25×")
            emit(fr, mark="replay_end" if i_ == len(rng) - 1 else None)
    hold(int(2.5 * FPS))
    fade()

    start("walk")
    # clip 2: the floating walk, normal speed
    Wk = meta["walk"]; LW = layout["walk"]
    for f in Wk["F"][9:]:
        fr = bg.copy().convert("RGBA"); d = ImageDraw.Draw(fr)
        header(d, 2, "A body that floats", f"“{Wk['prompt']}”: the estimate at 20% already floats where the final motion does")
        for j, v in enumerate(Wk["versions"]):
            ox, pw = 24 + j * 622, 600
            d.rounded_rectangle([ox * SS, 104 * SS, (ox + pw) * SS, 632 * SS], radius=10 * SS, fill=(24, 34, 56, 255), outline=(44, 58, 88, 255), width=SS)
            text(d, (ox + 16, 118), v["name"], F_B, INK)
            im = img(os.path.join(WORK, "img", f"walk_{v['key']}_{f:03d}.png")); x, y = place(fr, im, ox, 150, pw, 420, LW["box"], LW["ppm"], (ox + 1, ox + pw - 1))
            mp = mapper(LW["box"], LW["ppm"], x, y)
            fx, fy = mp(v["foot_x"][f], v["foot_y"][f]); gx, gy = mp(v["foot_x"][f], 0.0)
            if v["floating"][f]:
                bx = fx + 22 * SS
                d.line([bx, fy, bx, gy], fill=RED + (255,), width=2 * SS)
                for yy in (fy, gy):
                    d.line([bx - 6 * SS, yy, bx + 6 * SS, yy], fill=RED + (255,), width=2 * SS)
                text(d, (bx / SS + 10, (fy + gy) / 2 / SS - 10), f"{v['foot_cm'][f]:.0f} cm", F_B, RED)
            chip = v["floating"][f]
            d.rounded_rectangle([(ox + 16) * SS, 590 * SS, (ox + 116) * SS, 616 * SS], radius=13 * SS, fill=(RED if chip else (51, 65, 94)) + (255,))
            text(d, (ox + 66, 603), "floating" if chip else "supported", F_M, (255, 255, 255) if chip else MUTED, "mm")
            text(d, (ox + pw - 16, 603), f"floating in {v['n_float']} of {len(Wk['F'])} frames", F_M, MUTED, "rm")
        timebar(d, f / FPS, 6.0, "1×")
        emit(fr, mark="walk_floating" if (Wk["versions"][1]["floating"][f] and f > 30) else None)
    hold(FPS)
    fade()

    start("gain")
    # clip 3: the three jumps where the layer helps most, half speed; bars fill after landing
    G = meta["gain"]; LG = layout["gain"]
    for i_, (x_, bx) in enumerate(zip(G["examples"], LG["boxes"])):
        top = int(np.ceil(max(x_["g"], x_["pe"], x_["pb"], 9.81))) + 1
        seq = x_["F"][9:]
        for k_, f in enumerate(seq + [seq[-1]] * int(1.5 * FPS)):
            for rep in range(2 if k_ < len(seq) else 1):
                fr = bg.copy().convert("RGBA"); d = ImageDraw.Draw(fr)
                header(d, 3, "What a layer adds", "three of the jumps where adding layer 5 helps most, selected for that reason; predictions on held-out prompts")
                d.rounded_rectangle([24 * SS, 104 * SS, 680 * SS, 640 * SS], radius=10 * SS, fill=(24, 34, 56, 255), outline=(44, 58, 88, 255), width=SS)
                text(d, (40, 118), f"“{x_['prompt']}”", font("Italic", 16), INK)
                im = img(os.path.join(WORK, "img", f"{x_['key']}_{f:03d}.png")); X, Y = place(fr, im, 24, 150, 656, 460, bx, LG["ppm"], (25, 679))
                mp = mapper(bx, LG["ppm"], X, Y)
                path(d, x_["comx"][x_["F"][0]:f + 1], x_["comy"][x_["F"][0]:f + 1], x_["air"][x_["F"][0]:f + 1], mp, RED, 3, ground=MUTED)
                path(d, x_["estx"][x_["F"][0]:f + 1], x_["esty"][x_["F"][0]:f + 1], x_["air"][x_["F"][0]:f + 1], mp, TEAL, 3, dash=True)
                bx0, bw = 720, 520
                text(d, (bx0, 150), "fall rate of this jump (m/s²)", F_B, INK)
                prog = min(1, max(0, (k_ - (x_["e"] - seq[0])) / 10))
                for r, (lab, val, c) in enumerate([("final motion", x_["g"], RED), ("predicted from the estimate", x_["pe"], TEAL),
                                                   ("predicted with layer 5 added", x_["pb"], BLUE)]):
                    yy = 200 + 70 * r
                    text(d, (bx0, yy), lab, F_M, MUTED)
                    d.rounded_rectangle([bx0 * SS, (yy + 24) * SS, (bx0 + bw) * SS, (yy + 44) * SS], radius=10 * SS, fill=(37, 50, 77, 255))
                    wv = bw * (val / top) * prog
                    if wv > 2:
                        d.rounded_rectangle([bx0 * SS, (yy + 24) * SS, (bx0 + wv) * SS, (yy + 44) * SS], radius=10 * SS, fill=c + (255,))
                    if prog >= 1:
                        text(d, (bx0 + wv - 8, yy + 34), f"{val:.1f}", F_M, (255, 255, 255), "rm")
                gxp = bx0 + bw * 9.81 / top
                d.line([gxp * SS, 220 * SS, gxp * SS, 400 * SS], fill=MUTED + (200,), width=SS)
                text(d, (gxp, 410), "9.81", F_M, MUTED, "ma")
                if i_ == 2 and k_ >= len(seq):
                    text(d, (bx0, 520), "all held-out jumps:", F_S, MUTED)
                    text(d, (bx0, 550), f"R² {G['r2_est']:.2f} from the estimate, {G['r2_both']:.2f} with the layer", F_B, INK)
                for k in range(3):
                    cx, cy = (28 + 16 * k) * SS, (H - 22) * SS
                    d.ellipse([cx - 5 * SS, cy - 5 * SS, cx + 5 * SS, cy + 5 * SS], fill=(INK if k == i_ else TRACK) + (255,))
                text(d, (80, H - 30), f"jump {i_ + 1} of 3 · 0.5×", F_M, MUTED)
                emit(fr, mark=f"gain_{i_}" if k_ == len(seq) + 5 else None)
        fade(4)

    start("end")
    # end card
    for k in range(3 * FPS):
        fr = bg.copy().convert("RGBA"); d = ImageDraw.Draw(fr); al = min(1, k / 15)
        text(d, (W / 2, 320), "Every body, path and number is computed from the recorded outputs of the models.", font("Regular", 20), INK, "mm", al)
        text(d, (W / 2, 360), "Bodies are SMPL meshes fitted to the generated joints, for display only.", F_S, MUTED, "mm", al)
        emit(fr, mark="end" if k == 2 * FPS else None)
    wr.close(); state["clip"].close()
    print(f"wrote {OUT}: {state['n']} frames, {state['n'] / FPS:.1f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("prepare", "all"):
        prepare()
    if step in ("render", "all"):
        render()
    if step in ("compose", "all"):
        compose()
