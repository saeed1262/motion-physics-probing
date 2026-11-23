"""Measured comparison of candidate palettes for the paper's figures.

Every palette is mapped to the same roles: est (readout of the estimate), expl (the explicit label computed
from the estimate), layer (readout of a layer), layer2 (a second layer readout, e.g. ridge), both (estimate
and layer together), second (a second series, e.g. the other checkpoint), neutral (context such as the noisy
input). For each palette we report, over the five main roles (est, expl, layer, both, neutral):
  - the smallest pairwise CIEDE2000 difference, for normal vision and simulated protanopia, deuteranopia and
    tritanopia (Machado, Oliveira and Fernandes 2009, severity 1);
  - the smallest WCAG contrast ratio of the line colors against white (lines need about 3:1);
  - the smallest lightness (L*) gap between roles, i.e. how well they survive grayscale printing.

    .venv/bin/python -I paper/figures_web/study/palette_study.py
Writes paper/figures_web/study/palettes.json and palette_scores.csv.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CM = sys.argv[1] if len(sys.argv) > 1 else None   # folder with Crameri .txt maps (optional)


# ------------------------------------------------------------------ color science
def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def rgb2hex(c):
    c = np.clip(np.round(np.asarray(c) * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def delin(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def lab(c):
    xyz = M_XYZ @ lin(np.asarray(c, float)) / WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def de2000(l1, l2):
    L1, a1, b1 = l1; L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360 if C1p else 0.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360 if C2p else 0.0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = 0.0 if C1p * C2p == 0 else (h2p - h1p if abs(h2p - h1p) <= 180 else h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp)) + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return float(np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)))


# Sharma, Wu and Dalal (2005) test data: pairs 1, 7 and 17 of their table.
for l1, l2, ref in [((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425), ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
                    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492)]:
    assert abs(de2000(l1, l2) - ref) < 1e-3, (l1, l2, de2000(l1, l2), ref)

CVD = {  # Machado et al. (2009), severity 1, applied in linear RGB
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def simulate(c, kind):
    return np.clip(delin(np.clip(CVD[kind] @ lin(np.asarray(c, float)), 0, 1)), 0, 1)


def contrast_white(c):
    y = (M_XYZ @ lin(np.asarray(c, float)))[1]
    return 1.05 / (y + 0.05)


# ------------------------------------------------------------------ candidates
def crameri(name, ts):
    if not CM or not os.path.exists(os.path.join(CM, f"{name}.txt")):
        return None
    a = np.loadtxt(os.path.join(CM, f"{name}.txt"))
    return [rgb2hex(a[int(round(t * (len(a) - 1)))]) for t in ts]


ROLES = ["est", "expl", "layer", "layer2", "both", "second", "neutral"]
P = {
    # existing and user-suggested
    "A blog (current)": dict(est="#c4572b", expl="#c7862a", layer="#2a5c9e", layer2="#7ea6d6", both="#172033", second="#c7862a", neutral="#5b6680"),
    "G flare + crest": dict(est="#d64f5f", expl="#eb9872", layer="#215584", layer2="#44948f", both="#62296a", second="#33858d", neutral="#8a94a6"),
    "I viridis": dict(est="#22a884", expl="#7ad151", layer="#3b528b", layer2="#2a788e", both="#440154", second="#21918c", neutral="#8f8f8f"),
    # published palettes, mapped to the roles
    "Okabe-Ito": dict(est="#D55E00", expl="#E69F00", layer="#0072B2", layer2="#56B4E9", both="#000000", second="#009E73", neutral="#999999"),
    "Tol vibrant": dict(est="#CC3311", expl="#EE7733", layer="#0077BB", layer2="#33BBEE", both="#000000", second="#009988", neutral="#BBBBBB"),
    "Tableau 10": dict(est="#E15759", expl="#F28E2B", layer="#4E79A7", layer2="#76B7B2", both="#3d3d3d", second="#59A14F", neutral="#BAB0AC"),
    "Economist": dict(est="#E3120B", expl="#EBB434", layer="#006BA2", layer2="#3EBCD2", both="#0C0C0C", second="#379A8B", neutral="#B7C6CF"),
    "IBM Carbon": dict(est="#fa4d56", expl="#b28600", layer="#002d9c", layer2="#1192e8", both="#161616", second="#009d9a", neutral="#a8a8a8"),
}
bl = crameri("batlow", [0.12, 0.42, 0.62, 0.88])
if bl:
    P["Crameri batlow"] = dict(est=bl[2], expl=bl[3], layer=bl[0], layer2=bl[1], both="#101010", second=bl[1], neutral="#9a9a9a")
# designed for this paper: "what the model shows" warm, "what its layers hold" cool, combinations and context neutral
P.update({
    "S1 shown/hidden": dict(est="#d1491f", expl="#e8a04a", layer="#1d4e89", layer2="#6e9fd1", both="#1b1f27", second="#2a8c82", neutral="#9aa3b2"),
    "S2 ember and ink": dict(est="#e0592a", expl="#b9a58c", layer="#22324d", layer2="#8794a8", both="#7a2f3b", second="#5f7a99", neutral="#c9ced6"),
    "S3 flare/crest tuned": dict(est="#d64f5f", expl="#e98b61", layer="#1f4f86", layer2="#3f8f8d", both="#24202e", second="#3f8f8d", neutral="#9aa1ad"),
})

MAIN = ["est", "expl", "layer", "both", "neutral"]
rows, out = [], {}
for name, p in P.items():
    r = dict(palette=name)
    for vis in ["normal", "protan", "deutan", "tritan"]:
        cols = {k: (hex2rgb(p[k]) if vis == "normal" else simulate(hex2rgb(p[k]), vis)) for k in MAIN}
        labs = {k: lab(v) for k, v in cols.items()}
        pairs = [(a, b, de2000(labs[a], labs[b])) for i, a in enumerate(MAIN) for b in MAIN[i + 1:]]
        worst = min(pairs, key=lambda x: x[2])
        r[f"minDE_{vis}"] = round(worst[2], 1)
        if vis == "normal":
            r["closest_pair"] = f"{worst[0]}/{worst[1]}"
        r[f"est_layer_{vis}"] = round(de2000(labs["est"], labs["layer"]), 1)
    Ls = sorted(lab(hex2rgb(p[k]))[0] for k in MAIN)
    r["min_L_gap"] = round(min(b - a for a, b in zip(Ls, Ls[1:])), 1)
    r["min_contrast"] = round(min(contrast_white(hex2rgb(p[k])) for k in ["est", "expl", "layer", "both"]), 2)
    r["worst_CVD_DE"] = min(r["minDE_protan"], r["minDE_deutan"], r["minDE_tritan"])
    rows.append(r)
    out[name] = dict(p, cvd={vis: {k: rgb2hex(simulate(hex2rgb(p[k]), vis)) for k in ROLES} for vis in CVD})
import pandas as pd   # noqa: E402
df = pd.DataFrame(rows).set_index("palette")
df.to_csv(os.path.join(HERE, "palette_scores.csv"))
json.dump(out, open(os.path.join(HERE, "palettes.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(df[["minDE_normal", "closest_pair", "minDE_protan", "minDE_deutan", "minDE_tritan", "est_layer_deutan", "min_L_gap", "min_contrast"]].to_string())


# ------------------------------------------------------------------ optimized, lightness-stepped design
def lab2rgb(L, a, b):
    fy = (L + 16) / 116; fx, fz = fy + a / 500, fy - b / 200
    finv = lambda f: np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    xyz = np.array([finv(fx), finv(fy), finv(fz)]) * WHITE
    return delin(np.linalg.solve(M_XYZ, xyz))


def lch(L, C, h):
    c = lab2rgb(L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))
    return c if np.all((c >= -1e-6) & (c <= 1 + 1e-6)) else None


# Lightness levels chosen so every line color has at least 3:1 contrast on white (L* <= ~61) and roles are
# 11 to 14 L* apart (grayscale-safe); hue ranges keep the semantics (warm = shown by the estimate, cool = layers).
SPEC = {"expl": (60, (55, 85)), "est": (48, (25, 45)), "layer": (35, (235, 265)), "both": (19, (280, 330)), "neutral": (70, (240, 260))}
rng = np.random.default_rng(0)
best, best_score = None, -1
for _ in range(12000):
    p, ok = {}, True
    for k, (L, (h0, h1)) in SPEC.items():
        h = rng.uniform(h0, h1)
        Cmax = {"both": 14, "neutral": 6}.get(k, 75)
        C = rng.uniform(0 if k in ("both", "neutral") else 30, Cmax)
        c = lch(L, C, h)
        if c is None:
            ok = False; break
        p[k] = c
    if not ok:
        continue
    score = min(de2000(lab(simulate(p[a], v) if v != "normal" else p[a]), lab(simulate(p[b], v) if v != "normal" else p[b]))
                for v in ["normal", "protan", "deutan", "tritan"] for i, a in enumerate(MAIN) for b in MAIN[i + 1:])
    if score > best_score and all(contrast_white(p[k]) >= 3.0 for k in ["est", "expl", "layer", "both"]):
        best, best_score = p, score
S4 = {k: rgb2hex(v) for k, v in best.items()}
def lch_in(L, C, h):   # the most saturated in-gamut color up to chroma C
    while lch(L, C, h) is None:
        C -= 1
    return lch(L, C, h)


S4.update(layer2=rgb2hex(lch_in(58, 38, 245)), second=rgb2hex(lch_in(52, 34, 190)))
print("S4 lightness-stepped:", S4, "worst DE over all visions", round(best_score, 1),
      "contrast", {k: round(contrast_white(hex2rgb(S4[k])), 2) for k in ["est", "expl", "layer", "both"]})
out["S4 lightness-stepped"] = dict(S4, cvd={vis: {k: rgb2hex(simulate(hex2rgb(S4[k]), vis)) for k in ROLES} for vis in CVD})
json.dump(out, open(os.path.join(HERE, "palettes.json"), "w"), indent=1)
