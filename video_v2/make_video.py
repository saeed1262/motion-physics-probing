"""Revised supplementary video (anonymous, dark stage), from video_v2/script.md.

Built like scripts/make_presentation.py, which made the submitted version and is left unchanged so that version stays
reproducible. Differences: animated chart slides (video_v2/charts.py) replace the dense paper figures, clips can loop
while the narration continues, and three static slides are new (the question, the labels figure, the pilot).

Inputs, made first:
  data/video_v2/segments.json, sentences.json   the script split into sentences (captions keep the script's words)
  data/video_v2/audio/*.wav, durations.json     Kokoro narration, voice am_michael (scripts/tts_kokoro.py)
  data/demo_video/clips/*.mp4                   the demo clips of the submitted video (scripts/make_demo_video.py)
  paper/figures/web/*.pdf                       the paper's figures, rasterized with pdftoppm, on light cards
Every number on a slide comes from the paper. Writes video_v2/presentation_v2.mp4 and .srt.

    .venv/bin/python video_v2/make_video.py
"""
import json, os, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import charts                                                       # noqa: E402

P = os.path.join(ROOT, "data", "video_v2"); CLIPS = os.path.join(ROOT, "data", "demo_video", "clips")
FIGS = os.path.join(ROOT, "paper", "figures", "web"); IMG = os.path.join(ROOT, "data", "demo_video", "img")
OUT = os.path.join(HERE, "presentation_v2.mp4")
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


def clip_frames(name):
    """All frames of a demo clip as PIL images (OW x OH)."""
    rd = imageio_ffmpeg.read_frames(os.path.join(CLIPS, f"{name}.mp4")); meta = next(rd)
    w, h = meta["size"]
    return [Image.frombytes("RGB", (w, h), f) for f in rd]


CLIP = {n: clip_frames(n) for n in ["title", "clip1", "replay", "walk", "gain", "end"]}
for n, k in {"title": 6, "clip1": 6, "replay": 6, "walk": 6, "gain": 4}.items():   # drop each clip's closing fade
    CLIP[n] = CLIP[n][:-k]
clip_len = {n: len(fr) / FPS for n, fr in CLIP.items()}

# --public: the same video for the author's own site. The title card's anonymous line is replaced by the credit in
# video_v2/public.json, which stays out of the anonymous supplement (scripts/build_supplement.sh copies only this
# script and charts.py), so nothing identifying is written here.
if "--public" in sys.argv:
    pub = json.load(open(os.path.join(HERE, "public.json")))
    OUT = os.path.join(HERE, pub["out"])
    F_CREDIT = ImageFont.truetype(os.path.join(FONTS, "Literata-Regular-static.ttf"), 24)
    BAND = (80, 950, 760, 1005)                          # where the title card prints its anonymous line (output px)

    def credit(fr):
        a = np.asarray(fr).astype(np.float32)
        band = a[BAND[1]:BAND[3], BAND[0]:BAND[2]]
        bgrow = a[BAND[1]:BAND[3], 1000:1001]            # the background is a vertical gradient: one column per row
        shown = float(np.clip((band.max() - bgrow.max()) / 120.0, 0, 1))   # the line's own fade, from its brightness
        out = a.copy(); out[BAND[1]:BAND[3], BAND[0]:BAND[2]] = bgrow
        im = Image.fromarray(out.astype(np.uint8))
        # drawn under the subtitle, clear of the caption box that covers the card's bottom edge
        ImageDraw.Draw(im, "RGBA").text((90, 745), pub["credit"], font=F_CREDIT, fill=(170, 181, 202, int(255 * shown)), anchor="lm")
        return im

    CLIP["title"] = [credit(fr) for fr in CLIP["title"]]

# Animated slides: name -> (draw function of t in [0, 1], seconds the build-up takes).
ANIM = {
    "readers": (charts.draw_readers, 2.0),
    "usual": (lambda t: charts.draw_comparison("usual", t), 2.5),
    "fair": (lambda t: charts.draw_comparison("fair", t), 3.0),
    "fall": (charts.draw_fall_rate, 4.0),
    "early": (charts.draw_early, 4.0),
    "video": (charts.draw_video, 3.0),
}

# ------------------------------------------------------------------ the plan: what each part shows, keyed to sentences
# ("clip", name[, "loop"|"after"]) plays a clip from that sentence on; ("slide", name, step) a static slide;
# ("anim", name) an animated slide. "after" waits until the clip playing has finished; "loop" repeats a short clip.
PLAN = {
    1: [("clip", "walk", "loop"), None, None, ("clip", "title")],
    2: [("slide", "question", 1), ("slide", "question", 2), None, None, None],
    3: [("clip", "clip1", "loop"), None, None, None, ("clip", "replay")],
    4: [("slide", "labels", 1), ("slide", "labels", 2), None],
    5: [("slide", "catch", 0), ("slide", "catch", 2), ("slide", "catch", 3), ("anim", "readers"), None],
    6: [("anim", "usual"), ("anim", "fair"), None, ("slide", "control", 0)],
    7: [("slide", "pilot", 0), None, None],
    8: [("anim", "fall"), None, None, ("clip", "gain")],
    9: [("clip", "walk", "loop"), None, None, None],
    10: [("anim", "early"), None, None, None, None],
    11: [("anim", "video"), None, None],
    12: [("slide", "checks", 0), ("slide", "checks", 1), ("slide", "checks", 2), ("slide", "checks", 3), ("slide", "checks", 4),
         ("slide", "checks", 5)],
    13: [("clip", "end")],
}

# ------------------------------------------------------------------ timeline
events, t = [], 0.0                                    # [start, end, spec]
sent_times = []
for g in segs:
    plan = PLAN[g["num"]]; assert len(plan) == len(g["sentences"]), (g["num"], len(plan), len(g["sentences"]))
    clip_end, visual = t, None
    for k, cap in enumerate(g["sentences"]):
        sid = f"s{g['num']:02d}_{k}"; spec = plan[k]
        if spec is not None and spec[-1] == "after":
            t = max(t, clip_end)
        if spec is not None:
            if visual is not None:
                visual[1] = t
            visual = [t, None, spec]; events.append(visual)
            if spec[0] == "clip" and spec[-1] != "loop":
                clip_end = t + clip_len[spec[1]]
        sent_times.append((t, t + dur[sid], sid, cap))
        t += dur[sid] + PAUSE
    t = max(t, clip_end) + GAP                         # a clip that is not looping always plays to its end
    visual[1] = t
TOTAL = t
print(f"timeline {TOTAL:.1f} s ({TOTAL / 60:.1f} min), {len(sent_times)} sentences, {len(events)} visual events", flush=True)

# ------------------------------------------------------------------ static slides (drawn at SS x, downscaled)
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


def figure_card(fr, name, box, covers=()):
    """The paper's figure on a white card inside box (x, y, w, h); covers: image-fraction rects hidden in white."""
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


def body_layer(path, height=None):
    """A rendered body layer cropped to what it draws, optionally scaled to a height (SS px)."""
    im = Image.open(path).convert("RGBA"); im = im.crop(im.getbbox())
    if height:
        sc = height / im.height; im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    return im


def card(d, x, y, w, h, head, big, sub):
    d.rounded_rectangle([x * SS, y * SS, (x + w) * SS, (y + h) * SS], radius=14 * SS, fill=CARDBG + (255,), outline=(44, 58, 88, 255), width=SS)
    tx(d, (x + 28, y + 26), head, font("SemiBold", 22), INK)
    tx(d, (x + 28, y + 76), big, font("SemiBold", 40), TEAL)
    tx(d, (x + 28, y + 146), sub, font("Regular", 18), MUTED)


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
        lines = [("No physics simulator", 1), ("Physics learned from motion capture, never enforced", 1),
                 ("Do they know more than they show?", 2)]
        for i, (s, k) in enumerate(lines):
            if step >= k:
                tx(d, (64, 200 + 70 * i), ("→  " if k == 2 else "•  ") + s, font("SemiBold" if k == 2 else "Regular", 22),
                   INK if k == 2 else MUTED)
        return fr
    if name == "labels":
        fr, d = slide_base("Physics, computed from the joints", "balance, a jump's fall rate, and floating (Figure 3)")
        covers = {0: [(0, 0.5, 1, 1)], 1: [(0.5, 0.5, 1, 1)], 2: []}[step]
        figure_card(fr, "fig_labels", (40, 120, 1200, 470), covers)
        return fr
    if name == "catch":
        fr, d = slide_base("The catch", "the estimate is computed from the layers, so the fair test is what a layer adds to it")
        top = 0.64
        covers = {0: [(0.42, 0.12, 1, top), (0, top, 1, 1)], 1: [(0.66, 0.12, 1, top), (0, top, 1, 1)], 2: [(0, top, 1, 1)], 3: []}[step]
        figure_card(fr, "fig_setup", (40, 120, 1200, 470), covers)
        return fr
    if name == "control":
        fr, d = slide_base("A positive control", "a known signal hidden in the layer, 50-step MDM, dynamic balance")
        table(d, 120, 180, [("added to the estimate", "gain in R²"), ("the real layer", "+0.000"), ("hidden signal worth 0.005", "+0.013"),
                            ("hidden signal worth 0.01", "+0.025"), ("hidden signal worth 0.02", "+0.040")], [520, 220], highlight=1, size=22)
        tx(d, (120, 430), "the protocol finds a gain of that size; the real layer shows none", font("Regular", 21), MUTED)
        return fr
    if name == "pilot":
        fr, d = slide_base("Small data can flip the answer", "dynamic balance, 50-step MDM, 20% of denoising")
        card(d, 64, 150, 560, 200, "Pilot: 64 prompts, 12 held out", "0.832 vs 0.814", "the explicit label beats the best layer")
        card(d, 656, 150, 560, 200, "Full study, 25 training prompts", "0.783 · 0.757 · 0.617", "explicit label · tuned estimate · tuned layer")
        tx(d, (64, 410), "From 50 training prompts on, the tuned estimate leads the tuned layer at every size.", font("Regular", 22), INK)
        tx(d, (64, 452), "Only a lead that persists as the training set grows is evidence about the model.", font("Regular", 20), MUTED)
        return fr
    if name == "checks":
        fr, d = slide_base("Five checks", "before concluding that a generator knows more than it shows")
        items = ["Read the model's own estimate as carefully as its layers, and run a positive control",
                 "Score within prompts, and vary the amount of training data", "Try simple transforms before declaring something absent",
                 "Look at what a target's variance is made of", "For early pruning, compare with scoring the decoded preview"]
        for i, s in enumerate(items):
            if step >= i + 1:
                d.ellipse([64 * SS, (176 + 74 * i) * SS, 100 * SS, (212 + 74 * i) * SS], fill=SUN + (255,))
                tx(d, (82, 194 + 74 * i), str(i + 1), font("SemiBold", 20), (255, 255, 255), "mm")
                tx(d, (124, 182 + 74 * i), s, font("Regular", 23), INK)
        return fr
    raise KeyError(name)


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
if __name__ == "__main__":
    tmp = os.path.join(P, "video_only.mp4")
    wr = imageio_ffmpeg.write_frames(tmp, (OW, OH), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    wr.send(None)
    cache, n_frames = {}, int(round(TOTAL * FPS))
    stills = os.path.join(P, "stills"); os.makedirs(stills, exist_ok=True)
    ev_i = 0
    for fi in range(n_frames):
        tt = fi / FPS
        while ev_i + 1 < len(events) and tt >= events[ev_i + 1][0]:
            ev_i += 1
        start, end, spec = events[ev_i]
        if spec[0] == "clip":
            fr = CLIP[spec[1]]; k = int((tt - start) * FPS)
            base = fr[k % len(fr)] if spec[-1] == "loop" else fr[min(k, len(fr) - 1)]
        elif spec[0] == "anim":
            draw, secs = ANIM[spec[1]]; a = min(1.0, (tt - start) / secs)
            key = ("anim", spec[1], 1.0 if a >= 1 else round(a, 3))
            if key not in cache:
                cache = {k_: v for k_, v in cache.items() if k_[2] == 1.0}          # keep only finished animations
                cache[key] = draw(a)
            base = cache[key]
        else:
            key = ("slide", spec[1], spec[2])
            if key not in cache:
                cache = {k_: v for k_, v in cache.items() if k_[0] == "anim" and k_[2] == 1.0}
                cache[key] = slide(spec[1], spec[2]).convert("RGB").resize((OW, OH), Image.LANCZOS)
            base = cache[key]
        cap = next((c for (a_, b_, _, c) in sent_times if a_ <= tt < b_), None)
        im = caption(base.copy(), cap) if cap else base
        if fi % (FPS * 10) == 0:
            im.save(os.path.join(stills, f"t{int(tt):03d}.png"))
            print(f"  {tt:.0f} s / {TOTAL:.0f} s", flush=True)
        wr.send(np.asarray(im).tobytes())
    wr.close()

    # audio, captions file, mux
    audio = np.zeros(int(TOTAL * SR) + SR, np.float32)
    for a_, b_, sid, _ in sent_times:
        with wave.open(os.path.join(P, "audio", f"{sid}.wav")) as w_:
            raw = w_.readframes(w_.getnframes()); sw = w_.getsampwidth()
        x = np.frombuffer(raw, np.int16 if sw == 2 else np.int32).astype(np.float32) / (32768 if sw == 2 else 2 ** 31)
        i0 = int(a_ * SR); audio[i0:i0 + len(x)] += x
    with wave.open(os.path.join(P, "narration.wav"), "wb") as w_:
        w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(SR)
        w_.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    fmt = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(round(s % 1 * 1000)):03d}"
    with open(OUT.replace(".mp4", ".srt"), "w") as f:
        for i, (a_, b_, _, c) in enumerate(sent_times, 1):
            f.write(f"{i}\n{fmt(a_)} --> {fmt(b_)}\n{c}\n\n")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", os.path.join(P, "narration.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", OUT], check=True)
    print(f"wrote {OUT}: {TOTAL:.1f} s, {os.path.getsize(OUT) / 1e6:.1f} MB; captions {OUT.replace('.mp4', '.srt')}")
