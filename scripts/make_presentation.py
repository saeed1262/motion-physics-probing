"""Narrated presentation video (about 5.5 minutes, dark stage), from paper/supplement/presentation_script.md.

Inputs, made first:
  data/presentation/segments.json, sentences.json  the script split into sentences (captions keep the script's words)
  data/presentation/audio/*.wav, durations.json    Kokoro narration, voice am_michael (scripts/tts_kokoro.py)
  data/demo_video/clips/*.mp4                      the demo clips (scripts/make_demo_video.py compose)
  paper/figures/web/*.pdf                          the paper's figures (rasterized with pdftoppm), shown on light cards
Every number on a slide is copied from the paper's tables. Writes paper/supplement/presentation.mp4 and .srt.

    .venv/bin/python scripts/make_presentation.py
"""
import json, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "presentation"); CLIPS = os.path.join(ROOT, "data", "demo_video", "clips")
FIGS = os.path.join(ROOT, "paper", "figures", "web"); IMG = os.path.join(ROOT, "data", "demo_video", "img")
OUT = os.path.join(ROOT, "paper", "supplement", "presentation.mp4")
FONTS = os.path.join(ROOT, "paper", "figures_web", "fonts")
W, H, SS, FPS, SR = 1280, 720, 3, 20, 24000             # layout units; slides are drawn at SS x and downscaled to OW x OH
OW, OH = 1920, 1080; CS = OW / W                        # output size; captions are drawn at output size, CS x layout
PAUSE, GAP = 0.45, 0.9                                  # silence between sentences and between parts (s)
PAL = json.load(open(os.path.join(ROOT, "paper", "figures_web", "palette.json")))
hexrgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
INK, MUTED, TRACK, CARDBG = (238, 242, 248), (154, 167, 191), (51, 65, 94), (24, 34, 56)
RED, TEAL, BLUE, SUN = hexrgb(PAL["mark"]), hexrgb(PAL["est"]), (120, 150, 230), hexrgb(PAL["ramp5"])
font = lambda name, size, ss=SS: ImageFont.truetype(os.path.join(FONTS, f"Literata-{name}-static.ttf"), int(size * ss))

segs = json.load(open(os.path.join(P, "segments.json")))
dur = json.load(open(os.path.join(P, "audio", "durations.json")))
clip_len = {}


def clip_frames(name):
    """All frames of a demo clip as PIL images (OW x OH)."""
    rd = imageio_ffmpeg.read_frames(os.path.join(CLIPS, f"{name}.mp4")); meta = next(rd)
    w, h = meta["size"]
    return [Image.frombytes("RGB", (w, h), f) for f in rd]


CLIP = {n: clip_frames(n) for n in ["title", "clip1", "replay", "walk", "gain", "end"]}
for n, k in {"title": 6, "clip1": 6, "replay": 6, "walk": 6, "gain": 4}.items():   # drop each clip's closing fade, so a hold shows content
    CLIP[n] = CLIP[n][:-k]
for n, fr in CLIP.items():
    clip_len[n] = len(fr) / FPS

# ------------------------------------------------------------------ the plan: what each part shows, keyed to sentences
# visual specs: ("clip", name) plays a clip from that sentence on; ("slide", fn, step) draws a slide at a build step.
# "after" marks a sentence that waits until the clip currently playing has finished.
PLAN = {
    1: [("clip", "title")] + [None] * 2,
    2: [("slide", "question", 0), ("slide", "question", 1), ("slide", "question", 2), ("slide", "question", 3)],
    3: [("slide", "catch", 0), ("slide", "catch", 1), ("slide", "catch", 2), ("slide", "catch", 3), ("slide", "catch", 3)],
    4: [("slide", "recorded", 0), ("slide", "recorded", 1)],
    5: [("clip", "clip1"), None, None, ("clip", "replay", "after"), None],
    6: [("slide", "usual", 0), ("slide", "usual", 0), ("slide", "usual", 1)],
    7: [("slide", "fair", 0), ("slide", "fair", 1), ("slide", "fair", 2)],
    8: [("slide", "exception", 0), ("slide", "exception", 1), ("slide", "exception", 2), ("slide", "exception", 2),
        ("slide", "exception", 2), ("clip", "gain"), None],
    9: [("clip", "walk"), None, ("slide", "early", 0, "after"), None, None],
    10: [("slide", "video", 0), ("slide", "video", 0), ("slide", "video", 1)],
    11: [("slide", "checks", 0), ("slide", "checks", 1), ("slide", "checks", 2), ("slide", "checks", 3), ("slide", "checks", 4),
         ("slide", "checks", 5)],
    12: [("clip", "end"), None],
}

# ------------------------------------------------------------------ timeline
events, t = [], 0.0                                    # (start, end, kind, payload)
sent_times = []
for g in segs:
    plan = PLAN[g["num"]]; assert len(plan) == len(g["sentences"]), g["num"]
    clip_end, visual = t, None
    for k, cap in enumerate(g["sentences"]):
        sid = f"s{g['num']:02d}_{k}"; spec = plan[k]
        if spec is not None and spec[-1] == "after":
            t = max(t, clip_end)
        if spec is not None:
            if visual is not None:
                visual[1] = t
            visual = [t, None, spec]; events.append(visual)
            if spec[0] == "clip":
                clip_end = t + clip_len[spec[1]]
        sent_times.append((t, t + dur[sid], sid, cap))
        t += dur[sid] + PAUSE
    t = max(t, clip_end) + GAP
    visual[1] = t
TOTAL = t
print(f"timeline {TOTAL:.1f} s ({TOTAL / 60:.1f} min), {len(sent_times)} sentences, {len(events)} visual events")

# ------------------------------------------------------------------ slides (drawn at SS x, downscaled)
bg = None


def background():
    global bg
    if bg is None:
        g_ = np.linspace(0, 1, H * SS)[:, None]
        a0, a1 = np.array(hexrgb("#1b2740")), np.array(hexrgb("#0e1524"))
        bg = Image.fromarray((a0 * (1 - g_) + a1 * g_)[:, None, :].repeat(W * SS, 1).astype(np.uint8).reshape(H * SS, W * SS, 3))
    return bg.copy().convert("RGBA")


def tx(d, xy, s, f, fill, anchor="la", alpha=255):
    d.text((xy[0] * SS, xy[1] * SS), s, font=f, fill=tuple(fill) + (alpha,), anchor=anchor)


def slide_base(title, sub=None):
    fr = background(); d = ImageDraw.Draw(fr)
    tx(d, (48, 40), title, font("SemiBold", 34), INK)
    if sub:
        tx(d, (48, 88), sub, font("Regular", 17), MUTED)
    return fr, d


def figure_card(fr, name, box, covers=(), dim=()):
    """The paper's figure on a white card inside box (x, y, w, h, output px); covers: image-fraction rects hidden in
    white (not yet revealed); dim: fraction rects washed out (not the current focus)."""
    hi = os.path.join(P, "figs", f"{name}.png")         # the vector figure, rasterized once at 3600 px wide
    if not os.path.exists(hi):
        os.makedirs(os.path.dirname(hi), exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "3600", "-scale-to-y", "-1",
                        os.path.join(FIGS, f"{name}.pdf"), hi[:-4]], check=True)
    im = Image.open(hi).convert("RGBA")
    x, y, w, h = box; pad = 18
    sc = min((w - 2 * pad) * SS / im.width, (h - 2 * pad) * SS / im.height)
    im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for (a, b, c, e) in covers:
        d.rectangle([a * im.width, b * im.height, c * im.width, e * im.height], fill=(255, 255, 255, 255))
    for (a, b, c, e) in dim:
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(ov).rectangle([a * im.width, b * im.height, c * im.width, e * im.height], fill=(255, 255, 255, 170))
        im = Image.alpha_composite(im, ov)
    cw, ch = im.width + 2 * pad * SS, im.height + 2 * pad * SS
    cx, cy = int(x * SS + (w * SS - cw) / 2), int(y * SS + (h * SS - ch) / 2)
    ImageDraw.Draw(fr).rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=14 * SS, fill=(255, 255, 255, 255))
    fr.alpha_composite(im, (cx + pad * SS, cy + pad * SS))


def table(d, x, y, rows, widths, highlight=None, size=19):
    """rows: list of tuples of cell strings; first row is the header."""
    for r, row in enumerate(rows):
        yy = y + r * (size * 2.0)
        if r == highlight:
            d.rounded_rectangle([(x - 14) * SS, (yy - 8) * SS, (x + sum(widths) + 6) * SS, (yy + size * 1.5) * SS], radius=8 * SS, fill=(44, 58, 88, 255))
        cx = x
        for c, (cell, wdt) in enumerate(zip(row, widths)):
            f = font("SemiBold" if r == 0 or r == highlight else "Regular", size)
            col = INK if (r == 0 or c == 0 or r == highlight) else MUTED
            if c == 0:
                tx(d, (cx, yy), cell, f, col)
            else:
                tx(d, (cx + wdt, yy), cell, f, col, "ra")
            cx += wdt
        if r == 0:
            d.line([x * SS, (yy + size * 1.55) * SS, (x + sum(widths)) * SS, (yy + size * 1.55) * SS], fill=TRACK + (255,), width=SS)


def body_layer(path, height=None, alpha=1.0):
    """A rendered body layer cropped to what it draws, optionally scaled to a height (SS px)."""
    im = Image.open(path).convert("RGBA"); im = im.crop(im.getbbox())
    if height:
        sc = height / im.height; im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    if alpha < 1:
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    return im


def slide(name, step):
    if name == "question":
        fr, d = slide_base("The question")
        dm = json.load(open(os.path.join(ROOT, "data", "demo_video", "meta.json")))
        fw = next(f for f in range(40, 120) if dm["walk"]["versions"][1]["floating"][f])   # a floating frame of the final walk
        fj = (dm["jump"]["s"] + dm["jump"]["e"]) // 2                                       # mid-flight of the jump
        wimg = body_layer(os.path.join(IMG, f"walk_video_walk_final_{fw:03d}.png"), 300 * SS)
        jimg = body_layer(os.path.join(IMG, f"jump_final_{fj:03d}.png"), 300 * SS)
        fr.alpha_composite(wimg, (int(780 * SS), int(556 * SS - wimg.height))); fr.alpha_composite(jimg, (int(1010 * SS), int(520 * SS - jimg.height)))
        d.rectangle([700 * SS, 552 * SS, 1240 * SS, 572 * SS], fill=(37, 50, 77, 255))
        lines = [("No physics simulator: physics is learned, never enforced", 0), ("Bodies hover above the floor", 1),
                 ("Jumps do not fall as they should", 1), ("Could a probe on the layers catch failing samples early?", 2)]
        for i, (s, k) in enumerate(lines):
            if step >= k:
                tx(d, (64, 200 + 70 * i), ("→  " if k == 2 else "•  ") + s, font("SemiBold" if k == 2 else "Regular", 22),
                   INK if k == 2 else MUTED)
        return fr
    if name == "catch":
        fr, d = slide_base("The catch", "the estimate is computed from the layers, so the fair test is what a layer adds to it")
        top = 0.64
        covers = {0: [(0.42, 0.12, 1, top), (0, top, 1, 1)], 1: [(0.66, 0.12, 1, top), (0, top, 1, 1)], 2: [(0, top, 1, 1)], 3: []}[step]
        figure_card(fr, "fig_setup", (40, 120, 1200, 470), covers)
        return fr
    if name == "recorded":
        fr, d = slide_base("What we recorded")
        for i, (m, sub) in enumerate([("MDM, 50-step", "predicts the clean motion"), ("MDM, 1000-step", "the original checkpoint"),
                                       ("MotionDiffuse", "predicts the noise")]):
            x = 64 + 390 * i
            d.rounded_rectangle([x * SS, 170 * SS, (x + 360) * SS, 290 * SS], radius=14 * SS, fill=CARDBG + (255,), outline=(44, 58, 88, 255), width=SS)
            tx(d, (x + 24, 196), m, font("SemiBold", 26), INK); tx(d, (x + 24, 240), sub, font("Regular", 18), MUTED)
        tx(d, (64, 330), "every layer, along each model's own sampling trajectory", font("Regular", 21), INK)
        tx(d, (64, 368), "6,000 MDM motions and 3,000 MotionDiffuse motions from 250 prompts, plus jump-heavy runs", font("Regular", 19), MUTED)
        if step >= 1:
            for i, s in enumerate(["dynamic balance", "flight", "floats or falls", "fall rate of jumps"]):
                x = 64 + 290 * i
                d.rounded_rectangle([x * SS, 440 * SS, (x + 260) * SS, 496 * SS], radius=28 * SS, fill=(44, 58, 88, 255))
                tx(d, (x + 130, 468), s, font("SemiBold", 20), INK, "mm")
            tx(d, (64, 530), "computed exactly from joints, for the final motion and every intermediate estimate", font("Regular", 19), MUTED)
        return fr
    if name == "usual":
        fr, d = slide_base("The usual comparison", "50-step MDM, 20% of denoising")
        if step == 0:
            figure_card(fr, "fig_probe_steps", (40, 130, 1200, 450))
        else:
            table(d, 120, 190, [("", "best layer, linear", "explicit label", "estimate, linear"),
                                ("Dynamic balance (R²)", "0.838", "0.783", "0.723"), ("In flight (AUC)", "0.973", "0.876", "0.935")],
                  [380, 260, 240, 240], highlight=None, size=22)
            tx(d, (120, 380), "against these baselines, the layers look ahead of the estimate", font("SemiBold", 23), INK)
        return fr
    if name == "fair":
        fr, d = slide_base("The fair comparison", "every readout tuned the same way on held-out prompts, seeds pooled; pooled score, 95% intervals")
        if step < 2:
            gain_chart(d, step)
        else:
            tx(d, (120, 160), "Positive control: a known signal hidden in the layer (50-step, dynamic balance)", font("SemiBold", 23), INK)
            table(d, 120, 220, [("added to the estimate", "gain in R²"), ("the real layer", "+0.000"), ("hidden signal worth 0.005", "+0.013"),
                                ("hidden signal worth 0.01", "+0.025"), ("hidden signal worth 0.02", "+0.040")], [520, 220], highlight=1, size=22)
            tx(d, (120, 470), "the protocol finds a gain of that size; the real layer shows none", font("Regular", 21), MUTED)
        return fr
    if name == "exception":
        fr, d = slide_base("The one exception", "the fall rate of jumps, across denoising")
        covers = {0: [(0.385, 0.42, 1, 1)], 1: [(0.69, 0.42, 1, 1)], 2: []}[step]
        figure_card(fr, "fig_steps_fall", (40, 116, 1200, 488), covers)
        return fr
    if name == "early":
        fr, d = slide_base("Floating and early warning", "aborting at 20% of denoising when the estimate shows floating")
        figure_card(fr, "fig_early", (40, 130, 1200, 440))
        return fr
    if name == "video":
        fr, d = slide_base("A video model", "plausibility read from Wan 2.1's clean VAE latent (IntPhys, pairwise accuracy)")
        table(d, 160, 190, [("readout of the clean latent", "pairwise"), ("linear, pooled over time and space", "52.1%"), ("linear, time kept", "50.4%"),
                            ("linear, time and 4×4 space", "54.4%"), ("linear after |frame-to-frame change|", "79.6%")], [620, 200],
              highlight=4 if step >= 1 else None, size=22)
        tx(d, (160, 470), "chance is 50%", font("Regular", 20), MUTED)
        return fr
    if name == "checks":
        fr, d = slide_base("Five checks", "before concluding that a generator knows more than it shows")
        items = ["Read the model's own estimate as carefully as its layers, and run a positive control",
                 "Score within prompts, and vary the amount of training data", "Try simple transforms before declaring something absent",
                 "Look at what the target actually measures", "For early pruning, compare with the decoded preview"]
        for i, s in enumerate(items):
            if step >= i + 1:
                d.ellipse([64 * SS, (176 + 74 * i) * SS, 100 * SS, (212 + 74 * i) * SS], fill=SUN + (255,))
                tx(d, (82, 194 + 74 * i), str(i + 1), font("SemiBold", 20), (255, 255, 255), "mm")
                tx(d, (124, 182 + 74 * i), s, font("Regular", 23), INK)
        return fr
    raise KeyError(name)


GAIN_ROWS = [("Dynamic balance", "50-step"), ("Dynamic balance", "MotionDiffuse"), ("Static balance", "50-step"), ("Static balance", "MotionDiffuse"),
             ("In flight", "50-step"), ("In flight", "MotionDiffuse"), ("Floats or falls", "50-step, jump prompts"),
             ("Floats or falls", "MotionDiffuse, jump prompts"), ("Fall rate", "50-step, jump prompts"), ("Fall rate", "MotionDiffuse, jump prompts")]


def gain_chart(d, step):
    """Figure 4 for a slide: the same data (fig_gain.js), ten rows, pooled score only. (a) estimate minus layer, (b) gain."""
    rows = json.loads(open(os.path.join(ROOT, "paper", "figures_web", "data", "fig_gain.js")).read().split("=", 1)[1].rstrip().rstrip(";"))["rows"]
    pick = [next(r for r in rows if r["target"] == t and r["model"] == m) for t, m in GAIN_ROWS]
    y0, rh, lx = 168, 40, 420
    panels = [("fair", "(a) estimate minus layer", 470, 360, (-0.10, 0.20)), ("gain", "(b) gain from adding the layer", 880, 340, (-0.02, 0.10))]
    for i, (r, (t, m)) in enumerate(zip(pick, GAIN_ROWS)):
        yy = y0 + i * rh
        if i % 2 == 0:
            d.rectangle([40 * SS, (yy - 4) * SS, 1240 * SS, (yy + 2 * rh - 4) * SS], fill=(22, 31, 52, 255))
            tx(d, (56, yy + rh - 12), t, font("SemiBold", 19), INK, "lm")
        tx(d, (lx, yy + 16), m.replace("jump prompts", "jumps"), font("Regular", 16), MUTED, "rm")
    for j, (k, title_, x0, wd, (lo, hi)) in enumerate(panels):
        if j > step:
            continue
        X = lambda v: x0 + (v - lo) / (hi - lo) * wd
        tx(d, (x0, 140), title_, font("SemiBold", 19), INK)
        d.line([X(0) * SS, (y0 - 6) * SS, X(0) * SS, (y0 + rh * len(pick)) * SS], fill=MUTED + (255,), width=SS)
        for tv in ([-0.1, 0, 0.1, 0.2] if k == "fair" else [0, 0.05, 0.10]):
            tx(d, (X(tv), y0 + rh * len(pick) + 8), f"{tv:+.2f}" if tv else "0", font("Regular", 14), MUTED, "ma")
        for i, r in enumerate(pick):
            v = r[k]["score"]
            if v is None:
                continue
            o, a, b = v; yy = y0 + i * rh + 16
            if k == "fair":
                c = TEAL if a > 5e-4 else (BLUE if b < -5e-4 else MUTED)
            else:
                c = BLUE if a > 5e-4 else (TEAL if b < -5e-4 else MUTED)
            d.line([X(max(a, lo)) * SS, yy * SS, X(min(b, hi)) * SS, yy * SS], fill=c + (255,), width=3 * SS)
            d.ellipse([(X(o) - 6) * SS, (yy - 6) * SS, (X(o) + 6) * SS, (yy + 6) * SS], fill=c + (255,))
    if step >= 1:
        tx(d, (470, y0 + rh * len(pick) + 34), "teal: estimate ahead · blue: layer ahead or adds · gray: interval covers zero", font("Regular", 14), MUTED)


# ------------------------------------------------------------------ captions
F_CAP = font("Regular", 19, CS)


def wrap(s, maxw):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if F_CAP.getlength(test) > maxw and cur:
            lines.append(cur); cur = w_
        else:
            cur = test
    return lines + [cur]


def caption(im, s):
    lines = wrap(s, 1080 * CS)
    d = ImageDraw.Draw(im, "RGBA")
    lh = 27 * CS; hgt = lh * len(lines) + 18 * CS; y0 = OH - 24 * CS - hgt
    wmax = max(F_CAP.getlength(l) for l in lines)
    d.rounded_rectangle([OW / 2 - wmax / 2 - 18 * CS, y0, OW / 2 + wmax / 2 + 18 * CS, y0 + hgt], radius=10 * CS, fill=(8, 12, 22, 200))
    for i, l in enumerate(lines):
        d.text((OW / 2, y0 + 9 * CS + lh * i + lh / 2), l, font=F_CAP, fill=(245, 247, 250, 255), anchor="mm")
    return im


# ------------------------------------------------------------------ render
tmp = os.path.join(P, "video_only.mp4")
wr = imageio_ffmpeg.write_frames(tmp, (OW, OH), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
wr.send(None)
slide_cache, n_frames = {}, int(round(TOTAL * FPS))
stills = os.path.join(P, "stills"); os.makedirs(stills, exist_ok=True)
ev_i, sent_i = 0, 0
for fi in range(n_frames):
    tt = fi / FPS
    while ev_i + 1 < len(events) and tt >= events[ev_i + 1][0]:
        ev_i += 1
    start, end, spec = events[ev_i]
    if spec[0] == "clip":
        fr = CLIP[spec[1]]; k = min(int((tt - start) * FPS), len(fr) - 1); base = fr[k]
    else:
        key = (spec[1], spec[2])
        if key not in slide_cache:
            slide_cache = {key: slide(spec[1], spec[2]).convert("RGB").resize((OW, OH), Image.LANCZOS)}
        base = slide_cache[key]
    cap = next((c for (a, b, _, c) in sent_times if a <= tt < b), None)
    im = caption(base.copy(), cap) if cap else base
    if fi % (FPS * 20) == 0:
        im.save(os.path.join(stills, f"t{int(tt):03d}.png"))
    wr.send(np.asarray(im).tobytes())
wr.close()

# ------------------------------------------------------------------ audio, captions file, mux
audio = np.zeros(int(TOTAL * SR) + SR, np.float32)
for a, b, sid, _ in sent_times:
    with wave.open(os.path.join(P, "audio", f"{sid}.wav")) as w_:
        raw = w_.readframes(w_.getnframes()); sw = w_.getsampwidth()
    x = np.frombuffer(raw, np.int16 if sw == 2 else np.int32).astype(np.float32) / (32768 if sw == 2 else 2 ** 31)
    i0 = int(a * SR); audio[i0:i0 + len(x)] += x
with wave.open(os.path.join(P, "narration.wav"), "wb") as w_:
    w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(SR)
    w_.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
fmt = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)):03d}"
with open(OUT.replace(".mp4", ".srt"), "w") as f:
    for i, (a, b, _, c) in enumerate(sent_times, 1):
        f.write(f"{i}\n{fmt(a)} --> {fmt(b)}\n{c}\n\n")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", os.path.join(P, "narration.wav"),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", OUT], check=True)
print(f"wrote {OUT}: {TOTAL:.1f} s, {os.path.getsize(OUT) / 1e6:.1f} MB; captions {OUT.replace('.mp4', '.srt')}")
