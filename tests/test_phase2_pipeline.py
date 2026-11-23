import sys, types, time, os, glob, json, shutil
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location("b2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "build_nb2.py")); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
codes = [src.replace("from IPython.display import display", "display = print") for kind, tag, src in b.cells if kind == "code"]
def find(s): return next(i for i, c in enumerate(codes) if s in c)

root = "/tmp/p2drive"; shutil.rmtree(root, ignore_errors=True)
G = dict(np=np, os=os, glob=glob, json=json, time=time, plt=plt, display=print, __name__="__main__")
exec(codes[find('CHECKPOINT = "50steps"')], G)
G.update(DRIVE_ROOT=root, HML_DIR=os.environ.get("HML_DIR", "/tmp/hmldir"), RUN_SEEDS=[0, 1], N_TEST_CAPTIONS=36, SAMPLES_PER_PROMPT=3)
exec(codes[find("CURATED = [")], G)
print(G["PROMPTS"][64:67])
N = len(G["all_prompts"]); T = G["N_FRAMES"]; FRS = G["FRAME_STRIDE"]
G["N_STEPS"] = 50; G["RECORD_STEPS"] = [0, 10, 20, 29, 39, 49]; G["LAYERS"] = list(range(8))
G["FR"] = np.arange(0, T, FRS)
rng = np.random.default_rng(0)

# synthetic skeleton (Y up) like phase-1 harness
base = np.zeros((22, 3))
ys = {0:.95,1:.9,2:.9,3:1.05,4:.5,5:.5,6:1.2,7:.08,8:.08,9:1.35,10:.02,11:.02,12:1.5,13:1.45,14:1.45,15:1.65,16:1.4,17:1.4,18:1.15,19:1.15,20:.9,21:.9}
xs = {1:.1,4:.1,7:.1,10:.1,13:.1,16:.2,18:.25,20:.25,2:-.1,5:-.1,8:-.1,11:-.1,14:-.1,17:-.2,19:-.25,21:-.25}
for j in range(22): base[j] = [xs.get(j, 0.0), ys[j], 0.0]
def make(i):
    J = np.repeat(base[None], T, 0).copy(); t = np.arange(T) / 20.0
    J[:, 7:12:1, 1] += 0.06 * np.clip(np.sin(2*np.pi*1.5*t), 0, None)[:, None] * (i % 2)
    J[..., 1] += rng.uniform(0, 0.12) * (i % 5 == 0)
    J[:, 9:, 2] += rng.uniform(-0.4, 0.4) * (i % 3 == 0)
    return J + rng.normal(0, 0.003, J.shape)

# Fake generate_run: same file layout and keys as the notebook's generate_run
exec(codes[find("def run_dir(seed)")].split("for s in RUN_SEEDS:")[0], G)
D = 32
WG = np.random.default_rng(7).normal(size=(3, D))
def fake_sample_batch(prompts, seed):
    n = len(prompts); idx0 = fake_sample_batch.i; fake_sample_batch.i += n
    J = np.stack([make(idx0 + i) for i in range(n)])
    f = rng.normal(size=(n, T, 263)).astype(np.float32)
    sig = np.stack([J[:, :, [7, 10, 8, 11], 1].min(-1) * 10, J[:, :, 9, 2], J[:, :, 0, 1] * 10], -1)[:, ::FRS]
    W = WG
    acts = {(k, li): (sig @ W + rng.normal(0, 0.05 + 0.2 * li, (n, len(G["FR"]), D))).astype(np.float16) for k in G["RECORD_STEPS"] for li in range(8)}
    xt = {k: rng.normal(size=(n, len(G["FR"]), 263)).astype(np.float16) for k in G["RECORD_STEPS"]}
    x0 = {k: (J + rng.normal(0, 0.2 * (1 - k / 49), J.shape), rng.normal(size=(n, T, 263)).astype(np.float32)) for k in G["RECORD_STEPS"]}
    return J, f, acts, xt, x0, rng.normal(size=(n, 512)).astype(np.float32)
fake_sample_batch.i = 0
G["sample_batch"] = fake_sample_batch
G["BATCH"] = 32
for s in G["RUN_SEEDS"]:
    fake_sample_batch.i = 0
    G["generate_run"](s)
# resume check: rerun should skip everything
G["generate_run"](0)
print(sorted(os.listdir(os.path.join(root, "50steps_seed0")))[:4])

exec(codes[find("def batch_files(seed)")], G)       # labels (+ label funcs)
exec(codes[find("def group(p, src)")], G)            # group table
exec(codes[find("TARGETS = {")], G)                  # probes
exec(codes[find("Mean over runs (lines)")], G)       # plots
exec(codes[find("def failure_rates")], G)            # early prediction
exec(codes[find("frames = []")], G)                  # checkpoint compare
# second pass uses caches
t0 = time.time(); exec(codes[find("def batch_files(seed)")], G); exec(codes[find("TARGETS = {")], G); exec(codes[find("def failure_rates")], G)
print("cached rerun", round(time.time() - t0, 1), "s")
print(G["RES"].groupby(["source"]).score.mean().round(3).head(12))
