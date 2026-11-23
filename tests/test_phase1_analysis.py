import sys, types, time, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location("b", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "build_nb.py")); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
codes = [src.replace("from IPython.display import display", "display = print") for kind, tag, src in b.cells if kind == "code"]
def find(s): return next(i for i, c in enumerate(codes) if s in c)

rng = np.random.default_rng(1)
G = dict(np=np, os=os, time=time, plt=plt, display=print, __name__="__main__")
# Constants from the prompt cell (exec it, it only needs `diffusion`)
G["diffusion"] = types.SimpleNamespace(num_timesteps=50)
G["N_REPS"] = 2
os.environ["N_REPS"] = "2"
exec(codes[find("PROMPTS = [")], G)
P = G["PROMPTS"]; NR = G["N_REPS"]; T = G["N_FRAMES"]; FR = G["FR"]; RS = G["RECORD_STEPS"]
all_prompts = [p for _ in range(NR) for p in P]
prompt_id = np.array([i for _ in range(NR) for i in range(len(P))])
N = len(all_prompts)

# Synthetic walking skeleton (Y up). Some samples float, some lean.
base = np.zeros((22, 3))
ys = {0:.95,1:.9,2:.9,3:1.05,4:.5,5:.5,6:1.2,7:.08,8:.08,9:1.35,10:.02,11:.02,12:1.5,13:1.45,14:1.45,15:1.65,16:1.4,17:1.4,18:1.15,19:1.15,20:.9,21:.9}
xs = {1:.1,4:.1,7:.1,10:.1,13:.1,16:.2,18:.25,20:.25,2:-.1,5:-.1,8:-.1,11:-.1,14:-.1,17:-.2,19:-.25,21:-.25}
for j in range(22): base[j] = [xs.get(j, 0.0), ys[j], 0.0]
base[10, 2] = base[11, 2] = 0.12
def make(i):
    t = np.arange(T) / 20.0
    J = np.repeat(base[None], T, 0).copy()
    speed = rng.uniform(0, 1.2)
    J[..., 2] += speed * t[:, None]
    ph = 2 * np.pi * 1.8 * t
    for jj, sgn in [(7, 1), (10, 1), (8, -1), (11, -1)]:
        J[:, jj, 1] += 0.08 * np.clip(np.sin(ph * sgn), 0, None) * (speed > 0.2)
    J[..., 1] += rng.uniform(0, 0.12) * (i % 5 == 0)          # floating samples
    J[:, 9:, 2] += rng.uniform(-0.4, 0.4) * (i % 3 == 0)      # leaning upper body
    return J + rng.normal(0, 0.003, J.shape)
J_final = np.stack([make(i) for i in range(N)])
F_final = rng.normal(size=(N, T, 263)).astype(np.float32)
TXT = rng.normal(size=(N, 512)).astype(np.float32)
X0J = {k: J_final + rng.normal(0, 0.3 * (1 - k / 49), J_final.shape) for k in RS}
X0F = {k: rng.normal(size=(N, len(FR), 263)).astype(np.float16) for k in RS}
XT = {k: rng.normal(size=(N, len(FR), 263)).astype(np.float16) for k in RS}
G.update(all_prompts=all_prompts, prompt_id=prompt_id, J_final=J_final, F_final=F_final, TXT=TXT,
         X0J=X0J, X0F=X0F, XT=XT, LAYERS=list(range(8)))

t0 = time.time()
exec(codes[find("from shapely.geometry import")], G)   # labels
LAB = G["LAB"]
for k, v in LAB.items():
    print(f"{k:15s} shape={v.shape} finite={np.isfinite(v.astype(float)).mean():.2f} mean={np.nanmean(v.astype(float)):.3f}")
# Fake activations that linearly contain the labels (+ noise growing with layer)
D = 64
W = rng.normal(size=(3, D))
sig = np.stack([np.nan_to_num(LAB["xcom_margin"]), LAB["lowest_foot"], LAB["in_flight"]], -1)[:, FR]
G["ACTS"] = {(k, li): (sig @ W + rng.normal(0, 0.05 + 0.3 * li, (N, len(FR), D))).astype(np.float16) for k in RS for li in range(8)}
exec(codes[find("def category(p)")], G)                  # table
exec(codes[find("from sklearn.linear_model import RidgeCV")], G)  # probes
exec(codes[find("for name, kind in TARGETS.items():\n    d = RES")], G)  # plots
exec(codes[find("def failure_rates")], G)                # early prediction
print(G["EARLY"].groupby(["failure", "source"]).spearman.mean().round(3))
print(G["RES"].groupby(["target", "source"]).score.mean().round(3).head(12))
print(f"total {time.time() - t0:.0f}s")
