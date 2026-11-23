"""Video clips for the project page (project_page/assets/video/), from the paper's recorded motions.

Every body and number comes from the recorded joints. SMPL bodies are fitted to them for display only
(body_images.fit); nothing is interpolated between recorded denoising steps; frames play at the data's 20 fps.
    jump.mp4         the jump of Figure fig:stepsfall (final motion); center-of-mass path, red while airborne
    jump_steps.mp4   the same sample: its estimate at 0%, 20% and 40% of denoising and the final motion, in sync (2x2)
    floating.mp4     the floating walk of Figure 1: the estimate at 20% (top) and the final motion (bottom),
                     bodies red while floating
Each with a JPEG poster. Prints the numbers the page quotes.

    .venv/bin/python scripts/make_page_videos.py        (imageio-ffmpeg is installed in paper 1's venv)
"""
import glob, json, os, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_web_figures as M
import body_images as bi
from example_picks import FLOAT_G_MAX, LABELS, load, stretches

ROOT, WEB, P = M.ROOT, M.WEB, M.PALETTE
OUT = os.path.join(ROOT, "project_page", "assets", "video")
COM = LABELS["center_of_mass"]
FPS, SS = 20, 2                       # data frame rate; supersampling for smooth edges


def frame(V, rgb, box, ppm, path=None, air=None):
    """One body (local frame) on a white floor, with the center-of-mass path so far (red while airborne)."""
    def overlay(ax):
        if path is None:
            return
        x, y, _ = bi.project(path)
        ax.plot(x, y, color="#9aa6bd", lw=1.6 * SS, solid_capstyle="round", zorder=50)
        for t in range(1, len(x)):
            if air[t] and air[t - 1]:
                ax.plot(x[t - 1:t + 1], y[t - 1:t + 1], color=P["mark"], lw=2.4 * SS, solid_capstyle="round", zorder=51)
        ax.plot(x[-1:], y[-1:], "o", ms=4.5 * SS, color=P["mark"] if air[-1] else "#7d879b", zorder=52)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    bi.render([(V, rgb)], tmp, box, px_per_m=ppm * SS, overlay=overlay)
    im = Image.open(tmp).convert("RGBA"); os.remove(tmp)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    return bg.convert("RGB")


def write(name, frames, poster_t):
    """Frames (PIL, supersampled) -> H.264 MP4, sides padded to multiples of 16, plus a JPEG poster."""
    import imageio_ffmpeg
    w, h = frames[0].size[0] // SS, frames[0].size[1] // SS
    W, H = -(-w // 16) * 16, -(-h // 16) * 16
    def fit(im):
        c = Image.new("RGB", (W, H), (255, 255, 255))
        c.paste(im.resize((w, h), Image.LANCZOS), ((W - w) // 2, (H - h) // 2))
        return c
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.mp4")
    part = os.path.join(OUT, f"{name}.part.mp4")      # swapped in only when complete, so a page never loads half a clip
    gen = imageio_ffmpeg.write_frames(part, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                                      output_params=["-crf", "21", "-preset", "slow", "-movflags", "+faststart"])
    gen.send(None)
    for im in frames:
        gen.send(np.asarray(fit(im)).tobytes())
    gen.close()
    os.replace(part, path)
    fit(frames[poster_t]).save(os.path.join(OUT, f"{name}.jpg"), quality=86)
    print(f"wrote {path}: {W}x{H}, {len(frames)} frames, {os.path.getsize(path) / 1e6:.2f} MB", flush=True)


def longest_run_middle(mask):
    """Middle frame of the longest run of True (the poster shows the body while it floats)."""
    best, cur, mid = 0, 0, 0
    for t, m in enumerate(mask):
        cur = cur + 1 if m else 0
        if cur > best:
            best, mid = cur, t - cur // 2
    return mid


def box_of(Vs, pad=0.25):
    sx, sy, _ = bi.project(np.concatenate(Vs))
    return (sx.min() - pad, sx.max() + pad, -0.25, sy.max() + 0.12)


def jump():
    """The jump of fig_steps_fall (50-step, jump prompts, seed 0), selected as demo_storyboard.jump() does."""
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
    s, e, _ = next((s, e, g) for s, e, g in stretches(fin["flight_gravity"][i]) if abs(g - film["g_final"]) < 1e-4)
    J = np.concatenate([np.load(p)["joints"] for p in files])[i].astype(np.float32)
    vers = [(0, "0% of the way"), (10, "20% of the way"), (20, "40% of the way"), (None, "Finished motion")]
    to_local = bi.local_frame(J[0, 0, [0, 2]], J[-1, 0, [0, 2]], J[s, 0, [0, 2]])
    panels = []
    for k, name in vers:
        jj = J if k is None else np.concatenate([np.load(p)[f"x0j_{k}"] for p in files])[i].astype(np.float32)
        g = (fin if k is None else x0[k])["flight_gravity"][i]
        gk = next((gg for ss, ee, gg in stretches(g) if ss < e and ee > s), None)
        V = to_local(bi.fit(jj, f"page_jh50s0_m{i}_{'final' if k is None else k}"))
        panels.append(dict(name=name, V=V, com=to_local(COM(jj)[None])[0], air=np.isfinite(g), g=gk))
        print(f"jump {name}: fall rate {gk}", flush=True)
    apex = s + int(np.argmax(COM(J)[s:e, 1]))
    return film["prompt"], panels, apex


def floating():
    """The floating walk of Figure 1 (main data, 50-step)."""
    fits = np.load(os.path.join(ROOT, "data", "smpl_teaser", "fits.npz"))
    prompts, pid, J, X0, final, x0lab = load()
    idx = int(fits["float_motion"])
    to_local = bi.local_frame(J[idx][0, 0, [0, 2]], J[idx][-1, 0, [0, 2]], J[idx][0, 0, [0, 2]])
    panels = []
    for k, name in ((10, "The model's guess at 20% of the way"), (None, "Finished motion")):
        jj = J[idx] if k is None else X0[k][idx]
        g = (final if k is None else x0lab[k])["flight_gravity"][idx]
        fl = np.isfinite(g) & (g < FLOAT_G_MAX)
        panels.append(dict(name=name, V=to_local(bi.fit(jj, f"page_float_m{idx}_{'final' if k is None else k}")), fl=fl))
        print(f"floating {name}: floating in {int(fl.sum())} of {len(fl)} frames", flush=True)
    return prompts[idx].rstrip("."), panels


def main():
    which = sys.argv[1:] or ["jump", "jump_steps", "floating"]
    body = bi.hex_rgb(P["body"])
    if {"jump", "jump_steps"} & set(which):
        prompt, panels, apex = jump()
        T = len(panels[-1]["V"])
        box = box_of([p["V"] for p in panels])
        if "jump" in which:
            fin = panels[-1]
            frames = [frame(fin["V"][t], body, box, 300, fin["com"][:t + 1], fin["air"][:t + 1]) for t in range(T)]
            write("jump", frames, apex)
        if "jump_steps" in which:
            frames = []
            for t in range(T):
                tiles = []
                for p in panels:
                    tiles.append(frame(p["V"][t], body, box, 190, p["com"][:t + 1], p["air"][:t + 1]))
                w, h = tiles[0].size
                grid = Image.new("RGB", (2 * w, 2 * h), (255, 255, 255))
                for n, tile in enumerate(tiles):
                    grid.paste(tile, ((n % 2) * w, (n // 2) * h))
                d = ImageDraw.Draw(grid)
                d.line([(w, 0), (w, 2 * h)], fill=(228, 233, 241), width=2 * SS)
                d.line([(0, h), (2 * w, h)], fill=(228, 233, 241), width=2 * SS)
                frames.append(grid)
                if t % 20 == 0:
                    print(f"jump_steps frame {t}/{T}", flush=True)
            write("jump_steps", frames, apex)
    if "floating" in which:
        prompt, panels = floating()
        T = len(panels[-1]["V"])
        box = box_of([p["V"] for p in panels], pad=0.3)
        fail = bi.hex_rgb(P["body_fail"])
        frames = []
        for t in range(T):
            tiles = [frame(p["V"][t], fail if p["fl"][t] else body, box, 230) for p in panels]
            w, h = tiles[0].size
            col = Image.new("RGB", (w, 2 * h), (255, 255, 255))
            for n, tile in enumerate(tiles):
                col.paste(tile, (0, n * h))
            ImageDraw.Draw(col).line([(0, h), (w, h)], fill=(228, 233, 241), width=2 * SS)
            frames.append(col)
        write("floating", frames, longest_run_middle(panels[-1]["fl"]))


if __name__ == "__main__":
    main()
