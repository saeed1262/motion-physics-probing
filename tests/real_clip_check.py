"""Run the physics labels on the one real motion-capture clip in HumanML3D's public repo (012314:
tennis, "walking sideways and almost falling"). Expected (Nov 2026): flight ~3.5%, lowest foot ~1 cm,
skating ~0.06 m/s, XCoM outside support ~45% overall and 100% of single-support frames
(normal stepping, not a bug), 3 airborne frames so no gravity fit."""
import os, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
url = "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/HumanML3D/new_joints/012314.npy"
path = os.path.join(HERE, "012314.npy")
if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)
G = {"np": np}
exec(open(os.path.join(HERE, "..", "tools", "p2_labels_funcs.py")).read(), G)
L = G["physics_labels"](np.load(path))
xm = L["xcom_margin"]; c = L["contact"]
single = c[:, :2].any(1) ^ c[:, 2:].any(1)
print(f"flight {100*L['in_flight'].mean():.1f}% | lowest foot median {100*np.median(L['lowest_foot']):.1f} cm | "
      f"skate {np.nanmedian(L['skate']):.3f} m/s | XCoM out {100*np.mean(xm[np.isfinite(xm)]<0):.1f}% | "
      f"single-support XCoM out {100*np.mean(xm[single & np.isfinite(xm)]<0):.1f}% | airborne frames {int(L['airborne'].sum())}")
